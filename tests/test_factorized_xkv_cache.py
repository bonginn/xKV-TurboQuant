"""Unit tests for the shared FactorizedXKVCache state machine,
BF16FactorXKVCache, and PackedXKVCache. Pure logic, synthetic tensors, no
real model/GPU needed (runs on CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from factorized_xkv_cache import CacheState, BF16FactorXKVCache, PackedXKVCache  # noqa: E402
from xKV.customized_cache.turboquant import TurboQuantMSE  # noqa: E402


NUM_LAYERS = 8
GROUP_SIZE = 4
N_GROUPS = NUM_LAYERS // GROUP_SIZE
NUM_KV_HEADS = 2
HEAD_DIM = 8
RANK_K = 6
RANK_V = 6
GROUP_WIDTH = GROUP_SIZE * NUM_KV_HEADS * HEAD_DIM
DEVICE = torch.device("cpu")

PACKED_RANK = 96  # must be a dimension with a real offline codebook asset


def make_cache(tail_capacity=16):
    return BF16FactorXKVCache(
        num_layers=NUM_LAYERS, group_size=GROUP_SIZE, batch_size=1,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, tail_capacity=tail_capacity,
        device=DEVICE, dtype=torch.bfloat16,
    )


def make_group_factors(group_idx, prefix_len=100, rank_k=RANK_K, rank_v=RANK_V):
    layer_ids = list(range(group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE))
    A_k = torch.randn(1, prefix_len, rank_k, dtype=torch.bfloat16)
    A_v = torch.randn(1, prefix_len, rank_v, dtype=torch.bfloat16)
    B_k_group = torch.randn(1, rank_k, GROUP_WIDTH, dtype=torch.bfloat16)
    B_v_group = torch.randn(1, rank_v, GROUP_WIDTH, dtype=torch.bfloat16)
    return layer_ids, A_k, B_k_group, A_v, B_v_group


def load_all_groups(cache, prefix_len=100, rank_k=RANK_K, rank_v=RANK_V):
    cache.begin_prefill(prefix_len)
    for g in range(N_GROUPS):
        layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(g, prefix_len, rank_k, rank_v)
        cache.load_group_factors(g, layer_ids, A_k, B_k_group, A_v, B_v_group)
    return cache


# ── State machine ─────────────────────────────────────────────────────────

def test_initial_state_is_empty():
    cache = make_cache()
    assert cache.state == CacheState.EMPTY


def test_begin_prefill_transitions_to_collecting_prefill():
    cache = make_cache()
    cache.begin_prefill(100)
    assert cache.state == CacheState.COLLECTING_PREFILL
    assert cache.prefix_len == 100


def test_full_lifecycle_reaches_factor_ready_then_decoding_with_tail():
    cache = make_cache()
    cache.begin_prefill(100)
    for g in range(N_GROUPS):
        layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(g)
        cache.load_group_factors(g, layer_ids, A_k, B_k_group, A_v, B_v_group)
        if g < N_GROUPS - 1:
            assert cache.state == CacheState.FACTORIZING_GROUP
    assert cache.state == CacheState.FACTOR_READY

    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(NUM_LAYERS):
        cache.update(k, v, layer_idx)
    assert cache.state == CacheState.DECODING_WITH_TAIL


def test_load_group_factors_before_begin_prefill_fails():
    cache = make_cache()
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(0)
    with pytest.raises(RuntimeError, match="COLLECTING_PREFILL"):
        cache.load_group_factors(0, layer_ids, A_k, B_k_group, A_v, B_v_group)


def test_out_of_order_group_loading_fails():
    cache = make_cache()
    cache.begin_prefill(100)
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(1)  # skip group 0
    with pytest.raises(RuntimeError, match="strictly increasing order"):
        cache.load_group_factors(1, layer_ids, A_k, B_k_group, A_v, B_v_group)


def test_duplicate_group_loading_fails():
    cache = make_cache()
    cache.begin_prefill(100)
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(0)
    cache.load_group_factors(0, layer_ids, A_k, B_k_group, A_v, B_v_group)
    with pytest.raises(RuntimeError, match="strictly increasing order"):
        cache.load_group_factors(0, layer_ids, A_k, B_k_group, A_v, B_v_group)


def test_non_bf16_right_factor_fails():
    cache = make_cache()
    cache.begin_prefill(100)
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(0)
    with pytest.raises(TypeError, match="bfloat16"):
        cache.load_group_factors(0, layer_ids, A_k, B_k_group.float(), A_v, B_v_group)


def test_non_bf16_left_factor_fails():
    cache = make_cache()
    cache.begin_prefill(100)
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(0)
    with pytest.raises(TypeError, match="bfloat16"):
        cache.load_group_factors(0, layer_ids, A_k.float(), B_k_group, A_v, B_v_group)


def test_update_before_factor_ready_fails():
    cache = make_cache()
    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="FACTOR_READY"):
        cache.update(k, v, 0)


def test_get_layer_factors_before_ready_fails():
    cache = make_cache()
    with pytest.raises(RuntimeError, match="FACTOR_READY"):
        cache.get_layer_factors(0)


def test_get_layer_factors_returns_correct_tensors():
    cache = make_cache()
    cache.begin_prefill(100)
    stored = {}
    for g in range(N_GROUPS):
        layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(g)
        cache.load_group_factors(g, layer_ids, A_k, B_k_group, A_v, B_v_group)
        for pos, lid in enumerate(layer_ids):
            exp_B_k = BF16FactorXKVCache.slice_layer_B(B_k_group, pos, NUM_KV_HEADS, HEAD_DIM)
            exp_B_v = BF16FactorXKVCache.slice_layer_B(B_v_group, pos, NUM_KV_HEADS, HEAD_DIM)
            stored[lid] = (A_k, exp_B_k, A_v, exp_B_v)

    for layer_idx in range(NUM_LAYERS):
        A_k, B_k, A_v, B_v = cache.get_layer_factors(layer_idx)
        exp_A_k, exp_B_k, exp_A_v, exp_B_v = stored[layer_idx]
        assert torch.equal(A_k, exp_A_k)
        assert torch.equal(B_k, exp_B_k)
        assert torch.equal(A_v, exp_A_v)
        assert torch.equal(B_v, exp_B_v)


# ── Tail boundary (same contract as XKVWiredCache's already-tested one) ──

@pytest.mark.parametrize("capacity", [1, 2, 5, 16])
def test_tail_capacity_boundary(capacity):
    cache = make_cache(tail_capacity=capacity)
    load_all_groups(cache)

    def one_step():
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
        ret = None
        for layer_idx in range(NUM_LAYERS):
            ret = cache.update(k, v, layer_idx)
        return ret

    for _ in range(capacity - 1):
        one_step()
    assert cache.tail_len == capacity - 1

    k_ret, v_ret = one_step()
    assert cache.tail_len == capacity
    assert k_ret.shape[2] == capacity and v_ret.shape[2] == capacity

    with pytest.raises(RuntimeError, match="tail_capacity"):
        one_step()
    assert cache.tail_len == capacity  # unchanged after the failed attempt


# ── Fail-closed HF Cache API surface ──────────────────────────────────────

def test_reorder_cache_fails():
    cache = make_cache()
    with pytest.raises(NotImplementedError, match="beam search"):
        cache.reorder_cache(torch.tensor([0]))


def test_crop_fails():
    cache = make_cache()
    with pytest.raises(NotImplementedError, match="crop"):
        cache.crop(10)


def test_batch_repeat_interleave_fails():
    cache = make_cache()
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        cache.batch_repeat_interleave(2)


def test_batch_select_indices_fails():
    cache = make_cache()
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        cache.batch_select_indices(torch.tensor([0]))


def test_batch_split_fails():
    cache = make_cache()
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        cache.batch_split(2, 1)


def test_batch_size_greater_than_one_fails_at_construction():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        BF16FactorXKVCache(
            num_layers=NUM_LAYERS, group_size=GROUP_SIZE, batch_size=2,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, tail_capacity=16,
            device=DEVICE, dtype=torch.bfloat16,
        )


def test_update_rejects_multi_token_input():
    cache = make_cache()
    load_all_groups(cache)
    k = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)  # seq_len=3, not 1
    v = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="seq_len=1"):
        cache.update(k, v, 0)


# ── get_seq_length / get_max_cache_shape ──────────────────────────────────

def test_seq_length_tracks_prefix_and_tail():
    cache = make_cache(tail_capacity=16)
    load_all_groups(cache, prefix_len=100)
    assert cache.get_seq_length() == 100
    assert cache.get_max_cache_shape() == 100 + 16

    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(NUM_LAYERS):
        cache.update(k, v, layer_idx)
    assert cache.get_seq_length() == 101


# ── reset_state ────────────────────────────────────────────────────────────

def test_reset_state_returns_to_empty_and_clears_everything():
    cache = make_cache()
    load_all_groups(cache)
    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(NUM_LAYERS):
        cache.update(k, v, layer_idx)
    assert cache.state == CacheState.DECODING_WITH_TAIL

    cache.reset_state()
    assert cache.state == CacheState.EMPTY
    assert cache.prefix_len is None
    assert cache.tail_len == 0
    assert cache._tail_k.abs().sum().item() == 0
    with pytest.raises(RuntimeError, match="FACTOR_READY"):
        cache.get_layer_factors(0)

    # Must be fully reusable after reset.
    load_all_groups(cache, prefix_len=50)
    assert cache.state == CacheState.FACTOR_READY
    assert cache.get_seq_length() == 50


# ── Byte census ────────────────────────────────────────────────────────────

def test_byte_census_matches_real_tensor_sizes():
    cache = make_cache(tail_capacity=16)
    census_empty = cache.byte_census()
    assert census_empty["factor_bytes"] == 0
    expected_tail_bytes = cache._tail_k.element_size() * cache._tail_k.nelement() * 2
    assert census_empty["tail_bytes"] == expected_tail_bytes
    assert census_empty["logical_bytes"] == expected_tail_bytes

    load_all_groups(cache)
    census_ready = cache.byte_census()
    assert census_ready["state"] == "factor_ready"
    assert census_ready["factor_bytes"] > 0
    expected_factor_bytes = 0
    for A in list(cache._group_A_k.values()) + list(cache._group_A_v.values()):
        expected_factor_bytes += A.element_size() * A.nelement()
    for B in list(cache._B_k.values()) + list(cache._B_v.values()):
        expected_factor_bytes += B.element_size() * B.nelement()
    assert census_ready["factor_bytes"] == expected_factor_bytes
    assert census_ready["logical_bytes"] == census_ready["tail_bytes"] + census_ready["factor_bytes"]


# ── Invariant checker ──────────────────────────────────────────────────────

def test_check_invariants_passes_on_healthy_cache():
    cache = make_cache()
    cache.check_invariants()  # EMPTY state: must not raise
    load_all_groups(cache)
    cache.check_invariants()  # FACTOR_READY: must not raise

    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(NUM_LAYERS):
        cache.update(k, v, layer_idx)
    cache.check_invariants()  # DECODING_WITH_TAIL: must not raise


def test_check_invariants_catches_incomplete_layer_group_assignment():
    cache = make_cache()
    cache.begin_prefill(100)
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_group_factors(0)
    cache.load_group_factors(0, layer_ids, A_k, B_k_group, A_v, B_v_group)
    # Force state to FACTOR_READY without actually loading all groups, to
    # exercise the invariant checker's own detection (not just the normal
    # code path, which would never let this situation arise on its own).
    cache.state = CacheState.FACTOR_READY
    with pytest.raises(AssertionError, match="factor_ready/decoding_with_tail requires all"):
        cache.check_invariants()


# ══════════════════════════════════════════════════════════════════════════
# PackedXKVCache
# ══════════════════════════════════════════════════════════════════════════

def make_packed_cache(tail_capacity=16):
    k_codec = TurboQuantMSE(PACKED_RANK, 4)
    v_codec = TurboQuantMSE(PACKED_RANK, 4)
    cache = PackedXKVCache(
        k_codec=k_codec, v_codec=v_codec,
        num_layers=NUM_LAYERS, group_size=GROUP_SIZE, batch_size=1,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, tail_capacity=tail_capacity,
        device=DEVICE, dtype=torch.bfloat16,
    )
    return cache, k_codec, v_codec


def make_packed_group_factors(group_idx, prefix_len=100):
    return make_group_factors(group_idx, prefix_len, rank_k=PACKED_RANK, rank_v=PACKED_RANK)


def load_all_packed_groups(cache, prefix_len=100):
    cache.begin_prefill(prefix_len)
    for g in range(N_GROUPS):
        layer_ids, A_k, B_k_group, A_v, B_v_group = make_packed_group_factors(g, prefix_len)
        cache.load_group_factors(g, layer_ids, A_k, B_k_group, A_v, B_v_group)
    return cache


def test_packed_cache_rejects_non_4bit_codec():
    k_codec = TurboQuantMSE(PACKED_RANK, 3)  # bits=3, not 4
    v_codec = TurboQuantMSE(PACKED_RANK, 4)
    with pytest.raises(ValueError, match="bits=4"):
        PackedXKVCache(
            k_codec=k_codec, v_codec=v_codec,
            num_layers=NUM_LAYERS, group_size=GROUP_SIZE, batch_size=1,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, tail_capacity=16,
            device=DEVICE, dtype=torch.bfloat16,
        )


def test_packed_cache_full_lifecycle():
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache)
    assert cache.state == CacheState.FACTOR_READY
    cache.check_invariants()

    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(NUM_LAYERS):
        cache.update(k, v, layer_idx)
    assert cache.state == CacheState.DECODING_WITH_TAIL
    cache.check_invariants()


def test_packed_cache_no_rank_shaped_tensor_persists():
    """Invariant: packed_ready => no persistent tensor shaped
    [batch, sequence, rank] -- the packed tensor's last dim must be
    rank//2, never rank."""
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache)
    for packed in list(cache._group_packed_k.values()) + list(cache._group_packed_v.values()):
        assert packed.packed.dtype == torch.uint8
        assert packed.packed.shape[-1] == PACKED_RANK // 2
        assert packed.packed.shape[-1] != PACKED_RANK


def test_get_layer_factors_returns_bf16_correct_shapes():
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache, prefix_len=100)
    for layer_idx in range(NUM_LAYERS):
        A_k, B_k, A_v, B_v = cache.get_layer_factors(layer_idx)
        assert A_k.dtype == torch.bfloat16 and A_k.shape == (1, 100, PACKED_RANK)
        assert A_v.dtype == torch.bfloat16 and A_v.shape == (1, 100, PACKED_RANK)
        assert B_k.dtype == torch.bfloat16 and B_k.shape == (1, NUM_KV_HEADS, PACKED_RANK, HEAD_DIM)
        assert B_v.dtype == torch.bfloat16 and B_v.shape == (1, NUM_KV_HEADS, PACKED_RANK, HEAD_DIM)


def test_get_layer_factors_is_deterministic_across_calls():
    """The temporary-bridge dequantization must be a pure function of the
    stored packed state -- calling get_layer_factors() twice must give
    bit-identical results, not fresh randomness each time."""
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache)
    A_k1, B_k1, A_v1, B_v1 = cache.get_layer_factors(2)
    A_k2, B_k2, A_v2, B_v2 = cache.get_layer_factors(2)
    assert torch.equal(A_k1, A_k2)
    assert torch.equal(B_k1, B_k2)
    assert torch.equal(A_v1, A_v2)
    assert torch.equal(B_v1, B_v2)


def test_packed_cache_reuses_within_same_group():
    """The group_size=4 layers sharing one group_idx must trigger only ONE
    real dequant_fused() call (covering both K and V), not group_size
    separate ones -- avoiding that redundant work is the point of the
    single-entry decode cache (see the PackedXKVCache docstring).
    _get_group_A() calls factorized_xkv_cache.dequant_fused (imported into
    that module's namespace) on a cache MISS -- patch it there, not on the
    codec (dequant_fused does not go through codec.dequantize_rotated at
    all, real or CPU-fallback path)."""
    import factorized_xkv_cache as fxc

    cache, k_codec, v_codec = make_packed_cache()
    load_all_packed_groups(cache)

    calls = {"n": 0}
    real_dequant_fused = fxc.dequant_fused

    def counting_dequant_fused(*args, **kwargs):
        calls["n"] += 1
        return real_dequant_fused(*args, **kwargs)

    fxc.dequant_fused = counting_dequant_fused
    try:
        # Layers 0,1,2,3 are ALL group 0 (GROUP_SIZE=4) -- mirrors the real
        # per-step access pattern where every layer in a group calls
        # get_layer_factors() once per decode step.
        for layer_idx in range(GROUP_SIZE):
            cache.get_layer_factors(layer_idx)
    finally:
        fxc.dequant_fused = real_dequant_fused

    # Exactly 2 real calls total (one for K, one for V) across all 4
    # same-group layers -- not 8 (2 x 4, one pair per layer).
    assert calls["n"] == 2, f"expected exactly 2 real dequant_fused calls (K+V) for 4 same-group layers, got {calls['n']}"


def test_packed_cache_recomputes_correctly_on_group_change():
    """Moving to a different group_idx must evict the single-entry cache and
    return that OTHER group's correct data, not a stale reuse of the
    previous group's cached tensors."""
    cache, k_codec, v_codec = make_packed_cache()
    load_all_packed_groups(cache)

    # layer 0 -> group 0, layer 4 -> group 1 (GROUP_SIZE=4)
    A_k_g0, _, A_v_g0, _ = cache.get_layer_factors(0)
    assert cache._cached_group_idx == 0
    A_k_g1, _, A_v_g1, _ = cache.get_layer_factors(4)
    assert cache._cached_group_idx == 1, "requesting a different group must evict/replace the cache"
    assert not torch.equal(A_k_g0, A_k_g1), "group 0 and group 1 must have genuinely different A_k"

    # Going BACK to group 0 must recompute correctly (not return group 1's
    # stale cached values) and match what a direct, uncached call gives.
    A_k_g0_again, _, A_v_g0_again, _ = cache.get_layer_factors(1)  # layer 1 is also group 0
    assert torch.equal(A_k_g0_again, A_k_g0)
    assert torch.equal(A_v_g0_again, A_v_g0)


def test_packed_cache_cleared_on_reset():
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache)
    cache.get_layer_factors(0)
    assert cache._cached_group_idx is not None

    cache.reset_state()
    assert cache._cached_group_idx is None
    assert cache._cached_group_A is None


def test_packed_reconstruction_preserves_reasonable_fidelity():
    """Not bit-exact (real lossy 4-bit quantization). Deliberately does NOT
    compare A_k_recon directly against the original A_k: per
    TurboQuantFactorPair's own docstring, the returned left_factor is
    "diag(norms) @ y_hat" IN THE ROTATED FRAME, and right_factor is
    "Pi @ B" with the SAME rotation folded in -- only the PRODUCT
    A_recon @ B_recon is guaranteed to approximate the original A @ B; the
    individual factors live in different coordinate frames and are not
    meaningfully comparable on their own (an earlier version of this test
    compared them directly and got cos~0, which is expected given a random
    rotation, not a bug)."""
    cache, k_codec, _ = make_packed_cache()
    layer_ids, A_k, B_k_group, A_v, B_v_group = make_packed_group_factors(0)
    cache.begin_prefill(100)
    cache.load_group_factors(0, layer_ids, A_k, B_k_group, A_v, B_v_group)
    for g in range(1, N_GROUPS):
        li, ak, bk, av, bv = make_packed_group_factors(g)
        cache.load_group_factors(g, li, ak, bk, av, bv)

    A_k_recon, B_k_recon, _, _ = cache.get_layer_factors(0)
    recon_dense = torch.einsum("bsr,bhrd->bshd", A_k_recon.float(), B_k_recon.float())

    original_B_k_layer0 = BF16FactorXKVCache.slice_layer_B(B_k_group, 0, NUM_KV_HEADS, HEAD_DIM)
    original_dense = torch.einsum("bsr,bhrd->bshd", A_k.float(), original_B_k_layer0.float())

    cos = torch.nn.functional.cosine_similarity(
        recon_dense.reshape(-1), original_dense.reshape(-1), dim=0
    ).item()
    assert cos > 0.9, f"expected high reconstruction fidelity (A_recon @ B_recon vs original A @ B), got cos={cos}"


def test_packed_factor_bytes_smaller_than_bf16_equivalent():
    """The point of packing: A-side storage should shrink
    substantially (4-bit vs 16-bit BF16) relative to the unpacked design,
    for the same shapes."""
    bf16_cache = BF16FactorXKVCache(
        num_layers=NUM_LAYERS, group_size=GROUP_SIZE, batch_size=1,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, tail_capacity=16,
        device=DEVICE, dtype=torch.bfloat16,
    )
    load_all_groups(bf16_cache, rank_k=PACKED_RANK, rank_v=PACKED_RANK)

    packed_cache, _, _ = make_packed_cache()
    load_all_packed_groups(packed_cache)

    bf16_bytes = bf16_cache.byte_census()["factor_bytes"]
    packed_bytes = packed_cache.byte_census()["factor_bytes"]
    assert packed_bytes < bf16_bytes, (
        f"expected packed factor storage ({packed_bytes}) to be smaller than "
        f"BF16 factor storage ({bf16_bytes}) for the same shapes"
    )


def test_packed_reset_state_and_reuse():
    cache, _, _ = make_packed_cache()
    load_all_packed_groups(cache)
    cache.reset_state()
    assert cache.state == CacheState.EMPTY
    assert cache._group_packed_k == {}
    load_all_packed_groups(cache, prefix_len=50)
    assert cache.state == CacheState.FACTOR_READY
    assert cache.get_seq_length() == 50
