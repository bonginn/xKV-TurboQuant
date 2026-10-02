#!/usr/bin/env python3
"""RULER accuracy of xKV on the real kernel path, optionally with TurboQuant
K/V quantization.

Per sample: a dense prefill (forward hooks capture the K/V of every layer), a
balanced-gauge SVD of each layer group's K and V, optional TurboQuant-MSE
quantization of the factors (--quantize, K{k-bits}/V{v-bits}), then the dense
prefill cache is freed. Greedy decoding runs on the compressed prefix through
the fused Triton kernel (decode_attention_fwd_with_lse) and a dense tail in
XKVWiredCache (scripts/xkv_wired_cache.py), merged through log-sum-exp.
Outputs are scored with the official RULER metrics.

Quantization here is simulated: the factors are quantized and immediately
dequantized to BF16, which measures the accuracy effect of the 4-bit codes.
The physically packed storage is exercised by eval_ruler_factorized_cache.py
and measure_memory_timeline.py.

Usage:
    python scripts/eval_ruler_wired_cache.py --model Qwen/Qwen2.5-14B-Instruct-1M \\
        --task niah_multivalue --num-samples 20 [--quantize]
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
sys.path.insert(0, str(_ROOT))
from xkv_common import load_model_and_tokenizer_fa2, balanced_gauge_svd  # noqa: E402
from ops.fused_attention.rope.v1 import decode_attention_fwd_with_lse  # noqa: E402
from data.metrics import needle_score, multi_number, multi_words, string_match_part  # noqa: E402
from xKV.customized_cache.turboquant import TurboQuantMSE  # noqa: E402
from xkv_wired_cache import XKVWiredCache  # noqa: E402

GROUP_SIZE = 4
TAIL_CAPACITY = 512


def capture_all_layers_kv_with_logits(model, input_ids):
    """Run the prefill with k_proj/v_proj forward hooks to capture every
    layer's pre-RoPE K/V. Also returns the prefill's own last-position logits,
    so the first generated token comes from the prefill itself rather than
    from re-feeding the last prompt token (which would shift generation by
    one position)."""
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    num_layers = model.config.num_hidden_layers

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

    return captured, num_layers, num_kv_heads, head_dim, out.past_key_values, out.logits


METRIC_FOR_TASK = {
    "niah_single_1": needle_score,
    "niah_single_2": needle_score,
    "niah_single_3": needle_score,
    "niah_multikey_1": needle_score,
    "niah_multikey_2": needle_score,
    "niah_multiquery": multi_number,
    "niah_multivalue": multi_number,
    # Matches evaluate/data/dataset.py's METRICS_FN/get_metric() exactly. The
    # 10-task macro in the README is the six niah_* tasks other than
    # niah_single_3, plus vt, fwe, qa_1 and qa_2.
    "vt": multi_words,
    "fwe": multi_words,
    "qa_1": string_match_part,
    "qa_2": string_match_part,
}
# Matches evaluate/data/dataset.py's get_gen_len() exactly (niah=128, vt=30, fwe=50, qa=32).
GEN_LEN_FOR_TASK = {t: (128 if t.startswith("niah") else {"vt": 30, "fwe": 50, "qa_1": 32, "qa_2": 32}[t])
                    for t in METRIC_FOR_TASK}


def slice_layer_B(B_group, layer_pos_in_group, num_kv_heads, head_dim):
    bs, rank, total = B_group.shape
    per_layer = num_kv_heads * head_dim
    start = layer_pos_in_group * per_layer
    flat_slice = B_group[:, :, start:start + per_layer]
    reshaped = flat_slice.reshape(bs, rank, num_kv_heads, head_dim)
    return reshaped.permute(0, 2, 1, 3).contiguous(), flat_slice


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(METRIC_FOR_TASK.keys()))
    p.add_argument("--datalen", type=int, default=32768)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--rank-k", type=int, default=384)
    p.add_argument("--rank-v", type=int, default=576)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--k-bits", type=int, default=4)
    p.add_argument("--v-bits", type=int, default=4)
    p.add_argument("--k-codebook-path", type=str, default=None)
    p.add_argument("--v-codebook-path", type=str, default=None)
    p.add_argument("--svd-method", choices=["cholqr", "full"], default="cholqr")
    p.add_argument("--gen-len", type=int, default=None,
                    help="Override the task's default generation length.")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    gen_len = args.gen_len if args.gen_len is not None else GEN_LEN_FOR_TASK[args.task]

    data_path = _ROOT / f"evaluate/data/ruler/data/qwen/{args.datalen}/{args.task}/validation.jsonl"
    with open(data_path) as f:
        rows = [json.loads(line) for line in f][:args.num_samples]
    print(f"Task: ruler/{args.task}  samples: {len(rows)}  gen_len: {gen_len}  "
          f"metric: {METRIC_FOR_TASK[args.task].__name__}  [XKVWiredCache]")

    print(f"Loading model (FlashAttention-2) from {args.model} ...")
    model, tokenizer = load_model_and_tokenizer_fa2(args.model)
    torch.cuda.synchronize()

    num_layers = model.config.num_hidden_layers
    n_groups = num_layers // GROUP_SIZE
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    rope_theta = float(model.config.rope_theta)
    metric_fn = METRIC_FOR_TASK[args.task]

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attn_impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    k_codec = v_codec = None
    if args.quantize:
        k_codec = TurboQuantMSE(dimension=args.rank_k, bits=args.k_bits,
                                 codebook_path=args.k_codebook_path).to(device)
        v_codec = TurboQuantMSE(dimension=args.rank_v, bits=args.v_bits,
                                 codebook_path=args.v_codebook_path).to(device)
        print(f"Quantizing factors: TurboQuant-MSE K{args.k_bits}/V{args.v_bits} "
              f"(dimension={args.rank_k}/{args.rank_v})")

    scores = []
    t_start_all = time.time()
    for i, row in enumerate(rows):
        gt = row["outputs"]
        if isinstance(gt, list) and len(gt) == 1:
            gt = gt[0]

        enc = tokenizer(row["input"], return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        kv_len = input_ids.shape[1]
        prefix_len = kv_len

        captured, _, _, _, dense_cache, first_logits = capture_all_layers_kv_with_logits(model, input_ids)

        layer_factors = {}
        for g in range(n_groups):
            layer_ids = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
            ks = torch.cat([captured[j]["k"] for j in layer_ids], dim=1)
            vs = torch.cat([captured[j]["v"] for j in layer_ids], dim=1)
            A_k, B_k = balanced_gauge_svd(ks, args.rank_k, method=args.svd_method)
            A_v, B_v = balanced_gauge_svd(vs, args.rank_v, method=args.svd_method)
            if args.quantize:
                pair_k = k_codec.fake_quantize_pair(A_k, B_k)
                A_k = pair_k.left_factor.to(torch.bfloat16)
                B_k = pair_k.right_factor.to(torch.bfloat16)
                pair_v = v_codec.fake_quantize_pair(A_v, B_v)
                A_v = pair_v.left_factor.to(torch.bfloat16)
                B_v = pair_v.right_factor.to(torch.bfloat16)
            for pos, layer_idx in enumerate(layer_ids):
                k_B, _ = slice_layer_B(B_k, pos, num_kv_heads, head_dim)
                v_B, _ = slice_layer_B(B_v, pos, num_kv_heads, head_dim)
                layer_factors[layer_idx] = (A_k, k_B.contiguous(), A_v, v_B.contiguous())
            del ks, vs
        del captured

        # THE memory-saving step: free the dense prefill cache entirely. It
        # was only needed to compute layer_factors above (via the
        # k_proj/v_proj hooks); `dense_cache` itself is never read.
        del dense_cache
        gc.collect()
        torch.cuda.empty_cache()

        cache_xkv = XKVWiredCache(
            num_layers=num_layers, batch_size=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
            tail_capacity=TAIL_CAPACITY, device=device, dtype=torch.bfloat16,
        )
        cache_xkv.set_prefix_len(prefix_len)

        num_kv_splits_val = max(1, prefix_len // 2048)
        num_kv_splits = torch.full((1,), num_kv_splits_val, dtype=torch.int32, device=device)

        def xkv_multistep_interface(module, query, key, value, attention_mask, dropout=0.0,
                                     scaling=None, sliding_window=None, **kwargs):
            # `key`/`value` here are directly the dense tail (XKVWiredCache.update()
            # returns only the tail, not prefix+tail) -- no prefix_len slicing needed.
            layer_idx = module.layer_idx
            A_k, k_B, A_v, v_B = layer_factors[layer_idx]
            q = query.squeeze(2)
            o_p, lse_p = decode_attention_fwd_with_lse(
                q, A_k, k_B, A_v, v_B, num_kv_splits, num_kv_splits_val, sm_scale=scaling, theta=rope_theta,
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

            attn_output = o_merged.to(q.dtype).unsqueeze(1)
            return attn_output, None

        # Greedy generation: the first token comes from the prefill's own
        # logits (see capture_all_layers_kv_with_logits), the rest go through
        # the XKVWiredCache-backed decode path.
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

        while next_token is not None and len(generated) < gen_len:
            if cache_xkv.tail_len >= cache_xkv.tail_capacity:
                print(f"  [stop] tail_capacity={cache_xkv.tail_capacity} would be exceeded -- "
                      f"fail-closed, stopping generation (no EOS seen)")
                break
            cache_position = torch.tensor([pos], device=device)
            ALL_ATTENTION_FUNCTIONS[attn_impl] = xkv_multistep_interface
            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=cache_xkv, use_cache=True,
                            cache_position=cache_position)
            ALL_ATTENTION_FUNCTIONS[attn_impl] = original_fn
            next_id = out.logits[0, -1].argmax().item()
            pos += 1
            if next_id == eos_id:
                break
            generated.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)

        pred_text = tokenizer.decode(generated, skip_special_tokens=True)
        score = metric_fn(pred_text, gt)
        scores.append(score)
        avg_so_far = sum(scores) / len(scores)
        print(f"  [{i+1}/{len(rows)}] kv_len={kv_len} score={score:.4f} avg_so_far={avg_so_far:.4f}  "
              f"pred={pred_text[:80]!r}  gt={gt!r}")

        del cache_xkv, layer_factors, out, xkv_multistep_interface
        gc.collect()
        torch.cuda.empty_cache()

    total_s = time.time() - t_start_all
    avg = sum(scores) / len(scores)
    print(f"\n=== FINAL: ruler/{args.task}, {len(scores)} samples, XKVWiredCache path ===")
    print(f"  avg_score = {avg:.4f}")
    print(f"  total wall time: {total_s:.1f}s")
    print(f"  per-sample scores: {[round(s, 4) for s in scores]}")


if __name__ == "__main__":
    main()
