#!/usr/bin/env python3
"""Benchmark of the fused 4-bit dequantization kernel
(efficiency/ops/fused_attention/rope/v1_dequant_fused.py).

For the K-side (rank 384) and V-side (rank 576) left factors, compares the
PyTorch reference pipeline (unpack_turboquant_codes +
TurboQuantMSE.dequantize_rotated, about nine CUDA launches) with the
single-launch Triton kernel `dequant_fused`:

  1. Correctness first: the outputs must match the reference, otherwise no
     timing is reported.
  2. Median CUDA-event latency of each, and the speedup.

The inputs are random BF16 factors encoded with the real TurboQuant-MSE 4-bit
codec; the kernel's speed does not depend on the values, so no model or
dataset is needed.

Usage:
    python scripts/bench_dequant_fused.py [--kv-len 32768] [--rank-k 384] [--rank-v 576]
"""
import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "efficiency"))
sys.path.insert(0, str(_ROOT))

from ops.fused_attention.rope.v1_dequant_fused import dequant_fused  # noqa: E402
from xKV.customized_cache.turboquant import (  # noqa: E402
    TurboQuantMSE, pack_turboquant_codes, unpack_turboquant_codes,
)


def cuda_time_ms(fn, warmup, iters):
    """Median GPU-side latency of fn() in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def prepare_side(rank, kv_len, device):
    codec = TurboQuantMSE(dimension=rank, bits=4).to(device)
    factor = torch.randn(1, kv_len, rank, device=device, dtype=torch.bfloat16)
    packed = pack_turboquant_codes(codec.encode(factor))
    norms = packed.norms.squeeze(-1).contiguous()
    centroids = codec.centroids.contiguous()

    def reference():
        return codec.dequantize_rotated(unpack_turboquant_codes(packed), output_dtype=torch.bfloat16)

    def fused():
        return dequant_fused(packed.packed, norms, centroids, rank)

    return reference, fused


def check_matches_reference(name, rank, reference, fused):
    out_ref, out_fused = reference().float(), fused().float()
    max_diff = (out_fused - out_ref).abs().max().item()
    print(f"  {name} (rank {rank}): max abs diff vs PyTorch reference = {max_diff:.6e}")
    if max_diff > 0.0:
        cos = torch.nn.functional.cosine_similarity(out_fused.reshape(-1), out_ref.reshape(-1), dim=0).item()
        print(f"    cos = {cos:.8f}")
        if cos < 0.999999:
            sys.exit(f"  {name}: CORRECTNESS FAILED -- refusing to report timing")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kv-len", type=int, default=32768)
    p.add_argument("--rank-k", type=int, default=384)
    p.add_argument("--rank-v", type=int, default=576)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required (the kernel is written in Triton).")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}   kv_len={args.kv_len}")

    sides = [("K", args.rank_k), ("V", args.rank_v)]
    funcs = {name: prepare_side(rank, args.kv_len, device) for name, rank in sides}

    print("\n=== Correctness: fused kernel vs PyTorch reference ===")
    for name, rank in sides:
        check_matches_reference(name, rank, *funcs[name])

    print(f"\n=== Median latency (warmup={args.warmup}, iters={args.iters}) ===")
    for name, rank in sides:
        reference, fused = funcs[name]
        ref_ms = cuda_time_ms(reference, args.warmup, args.iters)
        fused_ms = cuda_time_ms(fused, args.warmup, args.iters)
        print(f"  {name} (rank {rank}): PyTorch reference {ref_ms:.4f} ms | "
              f"fused kernel {fused_ms:.4f} ms | speedup {ref_ms / fused_ms:.1f}x")


if __name__ == "__main__":
    main()
