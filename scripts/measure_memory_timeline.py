#!/usr/bin/env python3
"""GPU-memory-over-time timeline for one RULER sample.

Runs the same pipeline as eval_ruler_factorized_cache.py (real model, 32K
prefill, cholqr SVD, decode loop) for one cache mode and adds a background
thread that samples torch.cuda.memory_allocated() / memory_reserved() every
~20 ms, plus named stage markers (model loaded, prefill start/end, per-group
SVD done, dense cache freed, every decode step). Everything is dumped to JSON
for plotting.

Modes: dense (FlashAttention-2 baseline), bf16factor (BF16FactorXKVCache),
packed (PackedXKVCache). Run each mode in its own process so the memory
readings are clean. The KV-cache footprint at a stage is its allocated memory
minus the "model_loaded" marker (weights only).

Usage:
    python scripts/measure_memory_timeline.py --model Qwen/Qwen2.5-14B-Instruct-1M --mode dense
    python scripts/measure_memory_timeline.py --model Qwen/Qwen2.5-14B-Instruct-1M --mode bf16factor
    python scripts/measure_memory_timeline.py --model Qwen/Qwen2.5-14B-Instruct-1M --mode packed
"""
import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import torch
from flash_attn import flash_attn_with_kvcache

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "efficiency"))
sys.path.insert(0, str(_ROOT / "evaluate"))

from xkv_common import load_model_and_tokenizer_fa2, balanced_gauge_svd  # noqa: E402
from ops.fused_attention.rope.v1 import decode_attention_fwd_with_lse  # noqa: E402
from factorized_xkv_cache import BF16FactorXKVCache, PackedXKVCache  # noqa: E402
from xKV.customized_cache.turboquant import TurboQuantMSE  # noqa: E402

GROUP_SIZE = 4


def gib(n):
    return n / (1024**3)


class MemorySampler:
    def __init__(self, interval_s=0.02):
        self.interval_s = interval_s
        self.samples = []   # (t, allocated_bytes, reserved_bytes)
        self.markers = []   # (t, label)
        self._t0 = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            self.samples.append((t, torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
            time.sleep(self.interval_s)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def mark(self, label):
        t = time.time() - self._t0
        self.markers.append((t, label))
        print(f"  [{t:7.3f}s] {label}  (allocated={gib(torch.cuda.memory_allocated()):.3f} GiB)")


def capture_all_layers_kv_with_logits(model, input_ids, sampler):
    """Run the prefill with k_proj/v_proj forward hooks to capture every
    layer's pre-RoPE K/V (marking each layer in the timeline); also returns
    the prefill's own last-position logits and its dense cache."""
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    captured = {}
    handles = []

    def make_hook(layer_idx, which):
        def hook(module, inp, out):
            bsz, seqlen, _ = out.shape
            reshaped = out.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()
            captured.setdefault(layer_idx, {})[which] = reshaped
            if which == "v":
                sampler.mark(f"layer_{layer_idx}_captured")
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v")))

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)

    for h in handles:
        h.remove()
    return captured, out.logits, out.past_key_values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-1M")
    ap.add_argument("--mode", choices=["dense", "bf16factor", "packed"], required=True)
    ap.add_argument("--task", default="niah_single_1")
    ap.add_argument("--datalen", type=int, default=32768)
    ap.add_argument("--rank-k", type=int, default=384)
    ap.add_argument("--rank-v", type=int, default=576)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--ignore-eos", action="store_true",
                     help="Keep decoding past the real EOS token (forces a long, real run to stress-test "
                          "tail_capacity/timing -- the generated TEXT becomes meaningless past the natural "
                          "stopping point, but every forward pass/memory sample is still real.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ruler_path = f"evaluate/data/ruler/data/qwen/{args.datalen}/{args.task}/validation.jsonl"

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}   mode={args.mode}")

    sampler = MemorySampler(interval_s=0.02)
    sampler.start()

    print("Loading model ...")
    model, tokenizer = load_model_and_tokenizer_fa2(args.model)
    torch.cuda.synchronize()
    sampler.mark("model_loaded")

    num_layers = model.config.num_hidden_layers
    n_groups = num_layers // GROUP_SIZE
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rope_theta = float(model.config.rope_theta)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attn_impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    with open(_ROOT / ruler_path) as f:
        row = json.loads(f.readline())
    gt = row["outputs"]
    if isinstance(gt, list) and len(gt) == 1:
        gt = gt[0]
    enc = tokenizer(row["input"], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    kv_len = input_ids.shape[1]
    prefix_len = kv_len
    print(f"kv_len={kv_len}")

    sampler.mark("prefill_start")
    captured, first_logits, dense_cache = capture_all_layers_kv_with_logits(model, input_ids, sampler)
    sampler.mark("prefill_end")

    is_dense = args.mode == "dense"
    is_packed = args.mode == "packed"

    if is_dense:
        cache = dense_cache
        del captured
        gc.collect()
        torch.cuda.empty_cache()
        sampler.mark("ready_to_decode")
    else:
        if is_packed:
            k_codec = TurboQuantMSE(dimension=args.rank_k, bits=4).to(device)
            v_codec = TurboQuantMSE(dimension=args.rank_v, bits=4).to(device)
            cache = PackedXKVCache(
                k_codec=k_codec, v_codec=v_codec,
                num_layers=num_layers, group_size=GROUP_SIZE, batch_size=1,
                num_kv_heads=num_kv_heads, head_dim=head_dim, tail_capacity=512,
                device=device, dtype=torch.bfloat16,
            )
        else:
            cache = BF16FactorXKVCache(
                num_layers=num_layers, group_size=GROUP_SIZE, batch_size=1,
                num_kv_heads=num_kv_heads, head_dim=head_dim, tail_capacity=512,
                device=device, dtype=torch.bfloat16,
            )
        cache.begin_prefill(prefix_len)

        sampler.mark("svd_start")
        for g in range(n_groups):
            layer_ids = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
            ks = torch.cat([captured[j]["k"] for j in layer_ids], dim=1)
            vs = torch.cat([captured[j]["v"] for j in layer_ids], dim=1)
            A_k, B_k_group = balanced_gauge_svd(ks, args.rank_k, method="cholqr")
            A_v, B_v_group = balanced_gauge_svd(vs, args.rank_v, method="cholqr")
            cache.load_group_factors(g, layer_ids, A_k.contiguous(), B_k_group.contiguous(),
                                      A_v.contiguous(), B_v_group.contiguous())
            del ks, vs
            sampler.mark(f"svd_group_{g}_done")
        assert cache.state.value == "factor_ready"
        cache.check_invariants()

        del captured
        gc.collect()
        torch.cuda.empty_cache()
        sampler.mark("hook_dict_freed")

        del dense_cache
        gc.collect()
        torch.cuda.empty_cache()
        sampler.mark("dense_cache_freed_ready_to_decode")

    num_kv_splits_val = max(1, prefix_len // 2048)
    num_kv_splits = torch.full((1,), num_kv_splits_val, dtype=torch.int32, device=device)

    def xkv_interface(module, query, key, value, attention_mask, dropout=0.0,
                       scaling=None, sliding_window=None, **kwargs):
        layer_idx = module.layer_idx
        A_k, B_k, A_v, B_v = cache.get_layer_factors(layer_idx)
        q = query.squeeze(2)
        o_p, lse_p = decode_attention_fwd_with_lse(
            q, A_k, B_k, A_v, B_v, num_kv_splits, num_kv_splits_val, sm_scale=scaling, theta=rope_theta,
        )
        tail_len = key.shape[2]
        if tail_len == 0:
            o_merged = o_p.float()
        else:
            q_flash = q.unsqueeze(1)
            k_tail_flash = key.transpose(1, 2).contiguous()
            v_tail_flash = value.transpose(1, 2).contiguous()
            o_t_flash, lse_t_flash = flash_attn_with_kvcache(
                q_flash, k_tail_flash, v_tail_flash,
                softmax_scale=scaling, causal=False, return_softmax_lse=True,
            )
            o_t = o_t_flash.squeeze(1).float()
            lse_t = lse_t_flash.squeeze(-1).float()
            m = torch.maximum(lse_p, lse_t)
            w_p_raw = torch.exp(lse_p - m)
            w_t_raw = torch.exp(lse_t - m)
            w_sum = w_p_raw + w_t_raw
            w_p = (w_p_raw / w_sum).unsqueeze(-1)
            w_t = (w_t_raw / w_sum).unsqueeze(-1)
            o_merged = w_p * o_p.float() + w_t * o_t
        return o_merged.to(q.dtype).unsqueeze(1), None

    generated = []
    pos = kv_len
    eos_id = tokenizer.eos_token_id
    first_id = first_logits[0, -1].argmax().item()
    if first_id == eos_id and not args.ignore_eos:
        next_token = None
    else:
        generated.append(first_id)
        next_token = torch.tensor([[first_id]], device=device)
    del first_logits

    sampler.mark("decode_start")
    step = 0
    while next_token is not None and len(generated) < args.gen_len:
        if not is_dense and cache.tail_len >= cache.tail_capacity:
            break
        cache_position = torch.tensor([pos], device=device)
        if not is_dense:
            ALL_ATTENTION_FUNCTIONS[attn_impl] = xkv_interface
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True,
                        cache_position=cache_position)
        if not is_dense:
            ALL_ATTENTION_FUNCTIONS[attn_impl] = original_fn
        step += 1
        sampler.mark(f"decode_step_{step}")
        next_id = out.logits[0, -1].argmax().item()
        pos += 1
        if next_id == eos_id and not args.ignore_eos:
            break
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)

    sampler.mark("done")
    sampler.stop()

    pred_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"pred={pred_text[:80]!r}  gt={gt!r}")
    print(f"total samples collected: {len(sampler.samples)}")

    out_path = args.out or str(_ROOT / "results" / f"memory_timeline_{args.mode}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode, "gpu": gpu_name, "kv_len": kv_len,
            "samples": sampler.samples, "markers": sampler.markers,
        }, f)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
