#!/usr/bin/env python3
"""Decode speed, GPU memory and RULER accuracy of the factorized xKV caches.

Three modes share one harness (same prefill, generation loop and timing):
  (default)  BF16FactorXKVCache -- xKV factors stored as BF16
  --packed   PackedXKVCache -- left factors stored as 4-bit TurboQuant nibbles
             plus FP32 row norms
  --dense    dense FlashAttention-2 baseline, no xKV

For the xKV modes, the K/V captured by hooks during the dense prefill is
factorized group by group (cholqr SVD), loaded into the cache, and the dense
prefill cache is freed. Decoding attends the compressed prefix with the fused
Triton kernel and the dense tail, which lives in the same cache object, with
FlashAttention-2, merged through log-sum-exp.

Per sample it prints the score, decode ms/token, and GPU memory after prefill
vs ready to decode; a summary follows at the end.

Usage:
    python scripts/eval_ruler_factorized_cache.py --model Qwen/Qwen2.5-14B-Instruct-1M \\
        --task niah_multivalue --num-samples 5 [--packed | --dense]
"""
import argparse
import gc
import json
import sys
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
from data.metrics import needle_score, multi_number  # noqa: E402
from factorized_xkv_cache import BF16FactorXKVCache, PackedXKVCache  # noqa: E402
from xKV.customized_cache.turboquant import TurboQuantMSE  # noqa: E402

GROUP_SIZE = 4


def gib(n):
    return n / (1024**3)


def mem_snapshot(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    print(f"        [mem] {label}: allocated={gib(alloc):.4f} GiB")
    return alloc


def capture_all_layers_kv_with_logits(model, input_ids):
    """Run the prefill with k_proj/v_proj forward hooks to capture every
    layer's pre-RoPE K/V; also returns the prefill's own last-position logits
    and its dense cache."""
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    captured = {}
    handles = []

    def make_hook(layer_idx, which):
        def hook(module, inp, out):
            bsz, seqlen, _ = out.shape
            reshaped = out.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2).contiguous()
            captured.setdefault(layer_idx, {})[which] = reshaped
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i, "v")))

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)

    for h in handles:
        h.remove()
    return captured, out.logits, out.past_key_values


METRIC_FOR_TASK = {
    "niah_single_1": needle_score,
    "niah_single_2": needle_score,
    "niah_single_3": needle_score,
    "niah_multikey_1": needle_score,
    "niah_multikey_2": needle_score,
    "niah_multiquery": multi_number,
    "niah_multivalue": multi_number,
}
GEN_LEN_FOR_TASK = {t: 128 for t in METRIC_FOR_TASK}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(METRIC_FOR_TASK.keys()))
    p.add_argument("--datalen", type=int, default=32768)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--rank-k", type=int, default=384)
    p.add_argument("--rank-v", type=int, default=576)
    p.add_argument("--packed", action="store_true",
                    help="Use PackedXKVCache (real 4-bit nibble packing) instead of "
                         "BF16FactorXKVCache.")
    p.add_argument("--dense", action="store_true",
                    help="Skip xKV entirely -- real dense FlashAttention-2 baseline on the same real "
                         "prefill/generation-loop/timing harness, for a same-GPU, same-script comparison.")
    args = p.parse_args()
    if args.dense and args.packed:
        p.error("--dense and --packed are mutually exclusive")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    data_path = _ROOT / f"evaluate/data/ruler/data/qwen/{args.datalen}/{args.task}/validation.jsonl"
    with open(data_path) as f:
        rows = [json.loads(line) for line in f][:args.num_samples]
    gen_len = GEN_LEN_FOR_TASK[args.task]
    metric_fn = METRIC_FOR_TASK[args.task]
    if args.dense:
        cache_kind = "DENSE (real FlashAttention-2 baseline, no xKV)"
    elif args.packed:
        cache_kind = "PackedXKVCache (real 4-bit nibble packing)"
    else:
        cache_kind = "BF16FactorXKVCache"
    print(f"Task: ruler/{args.task}  samples: {len(rows)}  gen_len: {gen_len}  "
          f"metric: {metric_fn.__name__}  [{cache_kind} owns factors+tail]")

    print(f"Loading model (FlashAttention-2) from {args.model} ...")
    model, tokenizer = load_model_and_tokenizer_fa2(args.model)
    torch.cuda.synchronize()

    num_layers = model.config.num_hidden_layers
    n_groups = num_layers // GROUP_SIZE
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rope_theta = float(model.config.rope_theta)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attn_impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    scores = []
    all_timings = []  # (prefill_s, load_s, decode_s, steps_run) per sample
    all_mem = []  # (mem_after_prefill, mem_ready) bytes, per sample
    t_start_all = time.time()
    for i, row in enumerate(rows):
        gt = row["outputs"]
        if isinstance(gt, list) and len(gt) == 1:
            gt = gt[0]

        enc = tokenizer(row["input"], return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        kv_len = input_ids.shape[1]
        prefix_len = kv_len

        torch.cuda.synchronize()
        t_prefill0 = time.time()
        captured, first_logits, dense_cache = capture_all_layers_kv_with_logits(model, input_ids)
        torch.cuda.synchronize()
        prefill_s = time.time() - t_prefill0
        # Identical memory state across ALL modes at this exact point --
        # weights + the real dense cache + the hook-captured K/V dict are all
        # resident regardless of --dense/--packed, since capture_all_layers_
        # kv_with_logits always builds both (dense mode just never frees
        # dense_cache afterward).
        mem_after_prefill = mem_snapshot("after prefill (dense cache + captured hooks, all modes)")

        if args.dense:
            # Real dense baseline: the SAME real prefill's own DynamicCache,
            # completely unmodified attention -- no SVD, no cache construction.
            cache = dense_cache
            load_s = 0.0
            del captured
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # ONE object owns everything from here on: factors AND tail.
            if args.packed:
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

            torch.cuda.synchronize()
            t_load0 = time.time()
            for g in range(n_groups):
                layer_ids = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
                ks = torch.cat([captured[j]["k"] for j in layer_ids], dim=1)
                vs = torch.cat([captured[j]["v"] for j in layer_ids], dim=1)
                A_k, B_k_group = balanced_gauge_svd(ks, args.rank_k, method="cholqr")
                A_v, B_v_group = balanced_gauge_svd(vs, args.rank_v, method="cholqr")
                # Group-level B, NOT pre-sliced per layer -- PackedXKVCache needs
                # the whole group to fold TurboQuant's rotation into B before
                # slicing; BF16FactorXKVCache slices immediately with no transform.
                cache.load_group_factors(g, layer_ids, A_k.contiguous(), B_k_group.contiguous(),
                                          A_v.contiguous(), B_v_group.contiguous())
                del ks, vs
            assert cache.state.value == "factor_ready"
            cache.check_invariants()
            torch.cuda.synchronize()
            # SVD + load_group_factors combined -- for PackedXKVCache this ALSO
            # includes fake_quantize_pair (real quantization) + pack_turboquant_codes
            # (real nibble packing), a real one-time cost BF16FactorXKVCache doesn't pay.
            load_s = time.time() - t_load0

            del captured
            gc.collect()
            torch.cuda.empty_cache()

            # dense_cache was only needed for the prefill's own logits/
            # bookkeeping above -- xKV modes never read from it, free it now
            # (this is the memory-saving step: freeing the dense prefix once
            # the compressed factors exist).
            del dense_cache
            gc.collect()
            torch.cuda.empty_cache()

        # "Ready to decode" snapshot -- for xKV modes this is AFTER the real
        # dense prefix has been freed (the actual memory-saving step); for
        # --dense it's the same real dense cache from mem_after_prefill,
        # just minus the now-freed `captured` hook dict. This is the fair,
        # structurally-equivalent comparison point across all 3 modes.
        mem_ready = mem_snapshot("ready to decode")

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
        out = None
        if first_id == eos_id:
            next_token = None
        else:
            generated.append(first_id)
            next_token = torch.tensor([[first_id]], device=device)
        del first_logits

        # Timed region excludes the free first token (already produced by
        # the shared prefill call above). steps_run (not len(generated)) is
        # the denominator so an EOS-triggered break (which runs a real
        # forward pass but doesn't append its token) still counts correctly.
        torch.cuda.synchronize()
        t_decode0 = time.time()
        steps_run = 0
        while next_token is not None and len(generated) < gen_len:
            if not args.dense and cache.tail_len >= cache.tail_capacity:
                print("  [stop] tail_capacity would be exceeded -- fail-closed")
                break
            cache_position = torch.tensor([pos], device=device)
            if not args.dense:
                ALL_ATTENTION_FUNCTIONS[attn_impl] = xkv_interface
            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=cache, use_cache=True,
                            cache_position=cache_position)
            if not args.dense:
                ALL_ATTENTION_FUNCTIONS[attn_impl] = original_fn
            steps_run += 1
            next_id = out.logits[0, -1].argmax().item()
            pos += 1
            if next_id == eos_id:
                break
            generated.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
        torch.cuda.synchronize()
        decode_s = time.time() - t_decode0

        census = None
        if not args.dense:
            cache.check_invariants()
            census = cache.byte_census()

        pred_text = tokenizer.decode(generated, skip_special_tokens=True)
        score = metric_fn(pred_text, gt)
        scores.append(score)
        avg_so_far = sum(scores) / len(scores)
        ms_per_tok = (decode_s / steps_run * 1000) if steps_run > 0 else float("nan")
        print(f"  [{i+1}/{len(rows)}] kv_len={kv_len} score={score:.4f} avg_so_far={avg_so_far:.4f}  "
              f"pred={pred_text[:80]!r}  gt={gt!r}")
        if census is not None:
            print(f"        byte_census: factor={census['factor_bytes']/1e6:.1f}MB "
                  f"tail={census['tail_bytes']/1e6:.1f}MB state={census['state']}")
        print(f"        timing: prefill={prefill_s*1000:.1f}ms  load(SVD+quantize+pack)={load_s*1000:.1f}ms  "
              f"decode={decode_s:.3f}s/{steps_run}steps={ms_per_tok:.2f}ms/tok")
        print(f"        memory: after_prefill={gib(mem_after_prefill):.4f}GiB  ready_to_decode={gib(mem_ready):.4f}GiB  "
              f"freed={gib(mem_after_prefill - mem_ready):.4f}GiB")

        all_timings.append((prefill_s, load_s, decode_s, steps_run))
        all_mem.append((mem_after_prefill, mem_ready))

        del cache, out, xkv_interface
        if args.dense:
            # In dense mode `cache` is an alias for `dense_cache`, so `del cache`
            # alone leaves the object alive (only the xKV branch deletes
            # `dense_cache`, right after its SVD step). Without this, the
            # previous sample's ~5.9 GiB dense cache would stay alive through
            # the next sample's prefill and can OOM it.
            del dense_cache
        gc.collect()
        torch.cuda.empty_cache()

    total_s = time.time() - t_start_all
    avg = sum(scores) / len(scores)
    print(f"\n=== FINAL: ruler/{args.task}, {len(scores)} samples, {cache_kind} ===")
    print(f"  avg_score = {avg:.4f}")
    print(f"  total wall time: {total_s:.1f}s")
    print(f"  per-sample scores: {[round(s, 4) for s in scores]}")

    avg_prefill_ms = sum(t[0] for t in all_timings) / len(all_timings) * 1000
    avg_load_ms = sum(t[1] for t in all_timings) / len(all_timings) * 1000
    total_decode_s = sum(t[2] for t in all_timings)
    total_steps = sum(t[3] for t in all_timings)
    overall_ms_per_tok = (total_decode_s / total_steps * 1000) if total_steps > 0 else float("nan")
    print(f"  avg prefill: {avg_prefill_ms:.1f}ms   avg load(SVD+quantize+pack): {avg_load_ms:.1f}ms")
    print(f"  decode: {total_decode_s:.3f}s / {total_steps} steps = {overall_ms_per_tok:.2f} ms/token "
          f"({1000/overall_ms_per_tok:.2f} tok/s)")

    avg_after_prefill = sum(m[0] for m in all_mem) / len(all_mem)
    avg_ready = sum(m[1] for m in all_mem) / len(all_mem)
    print(f"  avg memory after_prefill: {gib(avg_after_prefill):.4f} GiB")
    print(f"  avg memory ready_to_decode: {gib(avg_ready):.4f} GiB")
    print(f"  avg memory freed: {gib(avg_after_prefill - avg_ready):.4f} GiB")


if __name__ == "__main__":
    main()
