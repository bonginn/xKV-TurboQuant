"""Fused 4-bit unpack + codebook dequantization in a single Triton kernel.

Replaces the Python-level `unpack_turboquant_codes` +
`TurboQuantMSE.dequantize_rotated` pipeline (about nine separate PyTorch/CUDA
launches: one strided assignment for the unpack and roughly eight chunked
gather launches inside `dequantize_rotated`) with ONE Triton launch. The
kernel materializes the BF16 left factor `A` for a whole layer group at once
and is bit-identical to the PyTorch reference.

Why the kernel only dequantizes and is not fused into the attention kernel:
xKV shares one `A` across a group of `group_size` consecutive layers, but
layers execute strictly sequentially (layer L's query depends on layer L-1's
full output), so one kernel launch can never serve all layers of a group.
Reusing `A` across the group therefore requires materializing it once and
keeping it between the per-layer attention calls, which is what
`PackedXKVCache` does. Dequantizing inside the attention kernel would repeat
the work in every layer and measured slower per group than this approach.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_only_kernel(
    Packed_Ptr, Norms_Ptr, Centroids_Ptr, Out_Ptr,
    stride_packed_bs, stride_packed_ls,
    stride_norms_bs,
    stride_out_bs, stride_out_ls,
    kv_len:  tl.constexpr,
    rank_k:  tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < kv_len

    offs_r  = tl.arange(0, BLOCK_R)
    offs_rh = tl.arange(0, BLOCK_R // 2)

    norm_tile = tl.load(Norms_Ptr + cur_batch * stride_norms_bs + offs_n, mask=mask_n, other=1.0)

    for i in range(0, tl.cdiv(rank_k, BLOCK_R)):
        r_off = i * BLOCK_R
        r_off_half = i * (BLOCK_R // 2)

        offs_packed = (
            cur_batch * stride_packed_bs
            + offs_n[:, None] * stride_packed_ls
            + (offs_rh + r_off_half)[None, :]
        )
        packed_tile = tl.load(Packed_Ptr + offs_packed, mask=mask_n[:, None], other=0)

        low_idx  = (packed_tile & 0x0F).to(tl.int32)
        high_idx = ((packed_tile >> 4) & 0x0F).to(tl.int32)
        idx_interleaved = tl.interleave(low_idx, high_idx)  # [BLOCK_N, BLOCK_R], natural order

        centroid_val = tl.load(Centroids_Ptr + idx_interleaved)
        val = (centroid_val * norm_tile[:, None]).to(tl.bfloat16)

        offs_out = (
            cur_batch * stride_out_bs
            + offs_n[:, None] * stride_out_ls
            + (offs_r + r_off)[None, :]
        )
        tl.store(Out_Ptr + offs_out, val, mask=mask_n[:, None])


def _dequant_fallback_pytorch(packed: torch.Tensor, norms: torch.Tensor,
                               centroids: torch.Tensor, rank_k: int) -> torch.Tensor:
    """Pure-PyTorch equivalent of the Triton kernel above, used when `packed`
    isn't on a CUDA device -- Triton kernels cannot run on CPU tensors
    ("Pointer argument ... cannot be accessed from Triton"), and
    PackedXKVCache's unit tests deliberately run on CPU (fast, no GPU
    needed). Same math, operating on the same raw tensors (not the
    TurboQuantCodes/PackedTurboQuantCodes dataclasses) as
    unpack_turboquant_codes+TurboQuantMSE.dequantize_rotated combined --
    verified bit-identical against that real reference, not assumed (and
    exercised on every CPU run of tests/test_factorized_xkv_cache.py)."""
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    indices = torch.empty(*packed.shape[:-1], rank_k, dtype=torch.long, device=packed.device)
    indices[..., 0::2] = low
    indices[..., 1::2] = high
    values = centroids[indices]
    return (values * norms.unsqueeze(-1)).to(torch.bfloat16)


def dequant_fused(packed: torch.Tensor, norms: torch.Tensor, centroids: torch.Tensor,
                   rank_k: int, block_n: int = 64) -> torch.Tensor:
    """packed: [batch, kv_len, rank_k//2] uint8. norms: [batch, kv_len] fp32
    (already squeezed). centroids: [16] fp32. Returns BF16 [batch, kv_len, rank_k],
    bit-identical to unpack_turboquant_codes+dequantize_rotated(output_dtype=bfloat16)
    -- verified by scripts/bench_dequant_fused.py. Falls back to
    a pure-PyTorch equivalent on non-CUDA devices (Triton requires CUDA)."""
    batch, kv_len, half_rank = packed.shape
    assert half_rank == rank_k // 2, f"packed's last dim ({half_rank}) must be rank_k//2 ({rank_k // 2})"
    assert norms.shape == (batch, kv_len), f"norms shape {norms.shape} must be [batch, kv_len]"

    if packed.device.type != "cuda":
        return _dequant_fallback_pytorch(packed, norms, centroids, rank_k)

    out = torch.empty(batch, kv_len, rank_k, dtype=torch.bfloat16, device=packed.device)
    grid = (batch, triton.cdiv(kv_len, block_n))
    _dequant_only_kernel[grid](
        packed, norms, centroids, out,
        packed.stride(0), packed.stride(1),
        norms.stride(0),
        out.stride(0), out.stride(1),
        kv_len=kv_len,
        rank_k=rank_k,
        BLOCK_N=block_n,
        BLOCK_R=64,
    )
    return out
