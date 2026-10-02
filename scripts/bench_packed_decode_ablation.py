#!/usr/bin/env python3
"""Ablation of PackedXKVCache's decode optimizations: end-to-end decode speed
with and without each one.

Each RULER sample is compressed once into a PackedXKVCache, then the same
greedy decode is run on that cache with three implementations of
PackedXKVCache._get_group_A, in rotating order inside one process so that GPU
clocks and host noise affect all variants alike:

  no-cache     PyTorch unpack + dequantize on every layer call (how the packed
               cache originally worked)
  group-cache  PyTorch dequantize, but once per layer group (a single-entry
               cache shared by the group's layers)
  fused        group cache + the fused Triton kernel `dequant_fused`
               (the shipped implementation)

The variants are numerically identical, so the generated tokens must match
exactly. An untimed warm-up decode runs first so Triton JIT compilation is not
counted.

Usage:
    python scripts/bench_packed_decode_ablation.py --model Qwen/Qwen2.5-14B-Instruct-1M \\
        --task niah_multivalue --num-samples 5
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "efficiency"))
sys.path.insert(0, str(_ROOT))

from xkv_common import load_model_and_tokenizer_fa2, balanced_gauge_svd  # noqa: E402
from eval_ruler_factorized_cache import GROUP_SIZE, capture_all_layers_kv_with_logits  # noqa: E402
import factorized_xkv_cache as fxc  # noqa: E402
from ops.fused_attention.rope.v1 import decode_attention_fwd_with_lse  # noqa: E402
from flash_attn import flash_attn_with_kvcache  # noqa: E402
from xKV.customized_cache.turboquant import TurboQuantMSE, unpack_turboquant_codes  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

TAIL_CAPACITY = 512
SHIPPED_GROUP_A = fxc.PackedXKVCache._get_group_A


def pytorch_group_A(use_cache):
    """PackedXKVCache._get_group_A with PyTorch unpack + dequantize instead of
    the fused kernel, with or without the single-entry group cache."""
    def _get_group_A(self, group_idx):
        if use_cache and group_idx == self._cached_group_idx:
            return self._cached_group_A
        packed_k = self._group_packed_k[group_idx]
        packed_v = self._group_packed_v[group_idx]
        fxc.validate_packed_codes(packed_k, self.k_codec)
        fxc.validate_packed_codes(packed_v, self.v_codec)
        A_k = self.k_codec.dequantize_rotated(unpack_turboquant_codes(packed_k), output_dtype=torch.bfloat16)
        A_v = self.v_codec.dequantize_rotated(unpack_turboquant_codes(packed_v), output_dtype=torch.bfloat16)
        if use_cache:
            self._cached_group_idx = group_idx
            self._cached_group_A = (A_k, A_v)
        return (A_k, A_v)
    return _get_group_A


VARIANTS = {
    "no-cache": pytorch_group_A(use_cache=False),
    "group-cache": pytorch_group_A(use_cache=True),
    "fused": SHIPPED_GROUP_A,
}


def build_cache(model, captured, kv_len, rank_k, rank_v, device):
    """Compress the captured prefill K/V of every layer group into a PackedXKVCache."""
    num_layers = model.config.num_hidden_layers
    cache = fxc.PackedXKVCache(
        k_codec=TurboQuantMSE(dimension=rank_k, bits=4).to(device),
        v_codec=TurboQuantMSE(dimension=rank_v, bits=4).to(device),
        num_layers=num_layers, group_size=GROUP_SIZE, batch_size=1,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.model.layers[0].self_attn.head_dim,
        tail_capacity=TAIL_CAPACITY, device=device, dtype=torch.bfloat16,
    )
    cache.begin_prefill(kv_len)
    for g in range(num_layers // GROUP_SIZE):
        layer_ids = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
        ks = torch.cat([captured[j]["k"] for j in layer_ids], dim=1)
        vs = torch.cat([captured[j]["v"] for j in layer_ids], dim=1)
        A_k, B_k = balanced_gauge_svd(ks, rank_k, method="cholqr")
        A_v, B_v = balanced_gauge_svd(vs, rank_v, method="cholqr")
        cache.load_group_factors(g, layer_ids, A_k.contiguous(), B_k.contiguous(),
                                 A_v.contiguous(), B_v.contiguous())
        del ks, vs
    return cache


def run_decode(model, tokenizer, cache, first_id, kv_len, gen_len, device):
    """Greedy decode on `cache` starting from its factors; returns (tokens, steps, seconds)."""
    rope_theta = float(model.config.rope_theta)
    attn_impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]
    num_kv_splits_val = max(1, kv_len // 2048)
    num_kv_splits = torch.full((1,), num_kv_splits_val, dtype=torch.int32, device=device)

    def xkv_interface(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, sliding_window=None, **kwargs):
        A_k, B_k, A_v, B_v = cache.get_layer_factors(module.layer_idx)
        q = query.squeeze(2)
        o_p, lse_p = decode_attention_fwd_with_lse(
            q, A_k, B_k, A_v, B_v, num_kv_splits, num_kv_splits_val, sm_scale=scaling, theta=rope_theta,
        )
        if key.shape[2] == 0:
            o_merged = o_p.float()
        else:
            o_t_flash, lse_t_flash = flash_attn_with_kvcache(
                q.unsqueeze(1), key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous(),
                softmax_scale=scaling, causal=False, return_softmax_lse=True,
            )
            o_t, lse_t = o_t_flash.squeeze(1).float(), lse_t_flash.squeeze(-1).float()
            m = torch.maximum(lse_p, lse_t)
            w_p_raw, w_t_raw = torch.exp(lse_p - m), torch.exp(lse_t - m)
            w_sum = w_p_raw + w_t_raw
            o_merged = (w_p_raw / w_sum).unsqueeze(-1) * o_p.float() + (w_t_raw / w_sum).unsqueeze(-1) * o_t
        return o_merged.to(q.dtype).unsqueeze(1), None

    # Restart decoding on the same factors: empty tail, cold group cache.
    cache.tail_len = 0
    cache._cached_group_idx = None
    cache._cached_group_A = None

    eos_id = tokenizer.eos_token_id
    generated = [first_id]
    next_token = torch.tensor([[first_id]], device=device)
    pos = kv_len
    steps = 0
    torch.cuda.synchronize()
    t0 = time.time()
    while next_token is not None and len(generated) < gen_len:
        if cache.tail_len >= cache.tail_capacity:
            break
        ALL_ATTENTION_FUNCTIONS[attn_impl] = xkv_interface
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([pos], device=device))
        ALL_ATTENTION_FUNCTIONS[attn_impl] = original_fn
        steps += 1
        next_id = out.logits[0, -1].argmax().item()
        pos += 1
        if next_id == eos_id:
            break
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
    torch.cuda.synchronize()
    return generated, steps, time.time() - t0


def median(values):
    return sorted(values)[len(values) // 2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="niah_multivalue")
    p.add_argument("--datalen", type=int, default=32768)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--rank-k", type=int, default=384)
    p.add_argument("--rank-v", type=int, default=576)
    p.add_argument("--gen-len", type=int, default=128)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    data_path = _ROOT / f"evaluate/data/ruler/data/qwen/{args.datalen}/{args.task}/validation.jsonl"
    with open(data_path) as f:
        rows = [json.loads(line) for line in f][:args.num_samples]
    print(f"Task: ruler/{args.task}  samples: {len(rows)}")
    print(f"Loading model (FlashAttention-2) from {args.model} ...")
    model, tokenizer = load_model_and_tokenizer_fa2(args.model)

    names = list(VARIANTS)
    ms_per_token = {name: [] for name in names}
    steps = 0
    try:
        for i, row in enumerate(rows):
            input_ids = tokenizer(row["input"], return_tensors="pt")["input_ids"].to(device)
            kv_len = input_ids.shape[1]
            captured, first_logits, dense_cache = capture_all_layers_kv_with_logits(model, input_ids)
            cache = build_cache(model, captured, kv_len, args.rank_k, args.rank_v, device)
            del captured, dense_cache
            gc.collect()
            torch.cuda.empty_cache()
            first_id = first_logits[0, -1].argmax().item()
            del first_logits

            if i == 0:
                for name in names:
                    fxc.PackedXKVCache._get_group_A = VARIANTS[name]
                    run_decode(model, tokenizer, cache, first_id, kv_len, args.gen_len, device)

            order = names[i % len(names):] + names[:i % len(names)]
            tokens = {}
            for name in order:
                fxc.PackedXKVCache._get_group_A = VARIANTS[name]
                generated, steps, seconds = run_decode(model, tokenizer, cache, first_id, kv_len, args.gen_len, device)
                ms_per_token[name].append(seconds / steps * 1000)
                tokens[name] = generated
            identical = all(t == tokens[names[0]] for t in tokens.values())
            print(f"  [{i + 1}/{len(rows)}] kv_len={kv_len} steps={steps}  ms/token: "
                  + "  ".join(f"{name}={ms_per_token[name][-1]:.2f}" for name in names)
                  + f"  identical_tokens={identical}")
            if not identical:
                sys.exit("The variants generated different tokens; they must be numerically identical.")
            del cache
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        fxc.PackedXKVCache._get_group_A = SHIPPED_GROUP_A

    print(f"\n=== PackedXKVCache decode, ms/token ({len(rows)} samples, {steps} decode steps each) ===")
    for name in names:
        v = ms_per_token[name]
        print(f"  {name:12s} mean {sum(v) / len(v):7.2f}   median {median(v):7.2f}   min {min(v):7.2f}   max {max(v):7.2f}")
    no_cache, group_cache, fused = (median(ms_per_token[n]) for n in names)
    print("Savings, from the medians:")
    print(f"  group cache  (no-cache -> group-cache): {no_cache - group_cache:6.2f} ms/token ({(no_cache - group_cache) / no_cache * 100:.1f}%)")
    print(f"  fused kernel (group-cache -> fused):    {group_cache - fused:6.2f} ms/token ({(group_cache - fused) / group_cache * 100:.1f}%)")
    print(f"  both         (no-cache -> fused):       {no_cache - fused:6.2f} ms/token ({(no_cache - fused) / no_cache * 100:.1f}%)")


if __name__ == "__main__":
    main()
