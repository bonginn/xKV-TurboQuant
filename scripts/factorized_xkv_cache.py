"""Factorized xKV caches: persistent low-rank factors plus a dense decode tail.

xKV factorizes each group of `group_size` consecutive layers' K (and V) as
X ~= A @ B, with one left factor A [batch, prefix_len, rank] per group and a
right factor B per layer. The caches in this module own BOTH the persistent
factors AND the dense decode tail (the most recent tokens, stored
uncompressed) as one object with an explicit state machine:

    empty -> collecting_prefill -> factorizing_group -> factor_ready
          -> decoding_with_tail

Unlike XKVWiredCache (scripts/xkv_wired_cache.py), which only manages the
dense tail and leaves the factors in a dict owned by the calling script,
these classes hold the factors themselves.

COLLECTING_PREFILL / FACTORIZING_GROUP are driven by explicit calls from the
orchestrating script (begin_prefill() / load_group_factors()), not by the
cache intercepting per-token update() calls during prefill: prefill's own
K/V is captured through a real forward pass (k_proj/v_proj hooks), then SVD
and quantization run externally, group by group, and load_group_factors() is
the seam where each finished group is handed to the cache (see
scripts/eval_ruler_factorized_cache.py).

get_layer_factors(layer_idx) is the read side and always returns ready-to-use
BF16 tensors (A_k, B_k, A_v, B_v): BF16FactorXKVCache returns its persistent
tensors, while PackedXKVCache dequantizes its packed A into a temporary BF16
tensor. The attention code consuming the factors is therefore identical for
both cache types.
"""
import enum
import sys
from pathlib import Path

import torch
from transformers.cache_utils import Cache

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "efficiency"))
from xKV.customized_cache.turboquant import (  # noqa: E402
    pack_turboquant_codes,
    validate_packed_codes,
)
from ops.fused_attention.rope.v1_dequant_fused import dequant_fused  # noqa: E402

class CacheState(enum.Enum):
    EMPTY = "empty"
    COLLECTING_PREFILL = "collecting_prefill"
    FACTORIZING_GROUP = "factorizing_group"
    FACTOR_READY = "factor_ready"
    DECODING_WITH_TAIL = "decoding_with_tail"


class FactorizedXKVCache(Cache):
    """Abstract base: state machine, per-layer BF16 right-factor storage,
    dense BF16 decode tail (same design as XKVWiredCache), byte census, and
    a debug-only invariant checker. Subclasses define how the per-group
    LEFT factor (A_k/A_v) is actually stored via _store_group/_get_group_A/
    _reset_group_storage/_factor_bytes (and may extend
    _check_subclass_invariants)."""

    def __init__(self, *, num_layers, group_size, batch_size, num_kv_heads, head_dim,
                 tail_capacity, device, dtype=torch.bfloat16):
        super().__init__()
        if batch_size != 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports batch_size=1 (this design is scoped "
                f"to batch-1 greedy decode), got batch_size={batch_size}."
            )
        if num_layers % group_size != 0:
            raise ValueError(f"num_layers ({num_layers}) must be divisible by group_size ({group_size})")

        self.num_layers = num_layers
        self.group_size = group_size
        self.n_groups = num_layers // group_size
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tail_capacity = tail_capacity
        self.device = device
        self.dtype = dtype

        self.state = CacheState.EMPTY
        self.prefix_len = None
        self._groups_loaded = []

        self._layer_group = {}
        self._B_k = {}
        self._B_v = {}
        self._group_A_k = {}
        self._group_A_v = {}

        self.tail_len = 0
        self._tail_k = torch.zeros(
            num_layers, batch_size, num_kv_heads, tail_capacity, head_dim, device=device, dtype=dtype,
        )
        self._tail_v = torch.zeros(
            num_layers, batch_size, num_kv_heads, tail_capacity, head_dim, device=device, dtype=dtype,
        )

    # ── Lifecycle: prefill collection + factorization ────────────────────

    def begin_prefill(self, prefix_len: int):
        if self.state != CacheState.EMPTY:
            raise RuntimeError(f"begin_prefill() requires state EMPTY, got {self.state.value}")
        if prefix_len <= 0:
            raise ValueError(f"prefix_len must be positive, got {prefix_len}")
        self.prefix_len = prefix_len
        self.state = CacheState.COLLECTING_PREFILL

    def load_group_factors(self, group_idx, layer_ids, A_k, B_k_group, A_v, B_v_group):
        """A_k/A_v: this group's raw left factors, shape [batch, prefix_len, rank].
        B_k_group/B_v_group: this group's raw, NOT-yet-per-layer-sliced right
        factors, shape [batch, rank, group_size*num_kv_heads*head_dim] --
        matching balanced_gauge_svd_timed's raw output exactly. Deliberately
        NOT pre-sliced per layer here: PackedXKVCache needs the GROUP-shaped
        B to fold TurboQuant's rotation into it BEFORE slicing; BF16FactorXKVCache doesn't need any transform and slices
        immediately. Subclasses do both the (optional) transform and the
        per-layer slicing in `_store_group`, and must populate
        `self._B_k[layer_idx]`/`self._B_v[layer_idx]` for every layer_idx in
        `layer_ids` themselves."""
        if self.state not in (CacheState.COLLECTING_PREFILL, CacheState.FACTORIZING_GROUP):
            raise RuntimeError(
                f"load_group_factors() requires state COLLECTING_PREFILL or FACTORIZING_GROUP, "
                f"got {self.state.value}"
            )
        expected_next = len(self._groups_loaded)
        if group_idx != expected_next:
            raise RuntimeError(
                f"groups must be loaded in strictly increasing order starting at 0: "
                f"expected group {expected_next}, got group {group_idx}"
            )
        if len(layer_ids) != self.group_size:
            raise ValueError(f"expected {self.group_size} layer_ids for a group, got {len(layer_ids)}")

        self.state = CacheState.FACTORIZING_GROUP
        self._store_group(group_idx, layer_ids, A_k, B_k_group, A_v, B_v_group)
        if not all(layer_idx in self._B_k and layer_idx in self._B_v for layer_idx in layer_ids):
            raise RuntimeError(
                f"{type(self).__name__}._store_group() must populate self._B_k/self._B_v for every "
                f"layer in {layer_ids}"
            )
        for layer_idx in layer_ids:
            if self._B_k[layer_idx].dtype != torch.bfloat16 or self._B_v[layer_idx].dtype != torch.bfloat16:
                raise TypeError(
                    f"right factors (B_k/B_v) must be bfloat16, "
                    f"layer {layer_idx}: got {self._B_k[layer_idx].dtype}/{self._B_v[layer_idx].dtype}"
                )
            self._layer_group[layer_idx] = group_idx
        self._groups_loaded.append(group_idx)

        if len(self._groups_loaded) == self.n_groups:
            self.state = CacheState.FACTOR_READY

    def _store_group(self, group_idx, layer_ids, A_k, B_k_group, A_v, B_v_group):
        raise NotImplementedError("subclass must implement _store_group")

    @staticmethod
    def slice_layer_B(B_group, layer_pos_in_group, num_kv_heads, head_dim):
        """Shared helper: slice one layer's [batch, num_kv_heads, rank, head_dim]
        right factor out of a group-concatenated [batch, rank, group_size*num_kv_heads*head_dim]
        tensor (same convention as slice_layer_B in scripts/eval_ruler_wired_cache.py)."""
        bs, rank, total = B_group.shape
        per_layer = num_kv_heads * head_dim
        start = layer_pos_in_group * per_layer
        flat_slice = B_group[:, :, start:start + per_layer]
        reshaped = flat_slice.reshape(bs, rank, num_kv_heads, head_dim)
        return reshaped.permute(0, 2, 1, 3).contiguous()

    def get_layer_factors(self, layer_idx: int):
        """Returns (A_k, B_k, A_v, B_v), all ready-to-use BF16 tensors."""
        if self.state not in (CacheState.FACTOR_READY, CacheState.DECODING_WITH_TAIL):
            raise RuntimeError(
                f"get_layer_factors() requires state FACTOR_READY or DECODING_WITH_TAIL, "
                f"got {self.state.value}"
            )
        group_idx = self._layer_group[layer_idx]
        A_k, A_v = self._get_group_A(group_idx)
        return A_k, self._B_k[layer_idx], A_v, self._B_v[layer_idx]

    def _get_group_A(self, group_idx):
        raise NotImplementedError("subclass must implement _get_group_A")

    # ── HF Cache API: decode-time tail (same design as XKVWiredCache) ────

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.state not in (CacheState.FACTOR_READY, CacheState.DECODING_WITH_TAIL):
            raise RuntimeError(
                f"update() (real decode step) requires state FACTOR_READY or DECODING_WITH_TAIL, "
                f"got {self.state.value} -- prefill factors must be fully loaded first"
            )
        self.state = CacheState.DECODING_WITH_TAIL
        seq_len = key_states.shape[2]
        if seq_len != 1:
            raise RuntimeError(
                f"{type(self).__name__} only supports decode (seq_len=1 per step), got seq_len={seq_len}."
            )
        if key_states.shape[0] != self.batch_size:
            raise NotImplementedError(
                f"{type(self).__name__} was constructed with batch_size={self.batch_size} but received "
                f"key_states with batch dim {key_states.shape[0]} -- batch>1 is not supported."
            )
        if self.tail_len >= self.tail_capacity:
            raise RuntimeError(
                f"{type(self).__name__}: tail_capacity={self.tail_capacity} exceeded -- fail-closed, "
                "not silently overwriting the cache."
            )
        pos = self.tail_len
        self._tail_k[layer_idx, :, :, pos:pos + 1, :] = key_states
        self._tail_v[layer_idx, :, :, pos:pos + 1, :] = value_states
        if layer_idx == self.num_layers - 1:
            self.tail_len += 1
        return self._tail_k[layer_idx, :, :, :pos + 1, :], self._tail_v[layer_idx, :, :, :pos + 1, :]

    def get_seq_length(self, layer_idx=0):
        return (self.prefix_len or 0) + self.tail_len

    def get_max_cache_shape(self):
        return (self.prefix_len or 0) + self.tail_capacity

    # ── Fail-closed for unsupported HF Cache API surface ─────────────────

    def reorder_cache(self, beam_idx):
        raise NotImplementedError(
            f"{type(self).__name__} does not support beam search (reorder_cache) -- this design "
            "is scoped to greedy decode only."
        )

    def crop(self, max_length):
        raise NotImplementedError(
            f"{type(self).__name__} does not support crop() -- the prefix is already factorized; "
            "cropping it would require re-factorizing, which is out of scope."
        )

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError(
            f"{type(self).__name__} only supports batch_size=1 -- batch_repeat_interleave is not supported."
        )

    def batch_select_indices(self, indices):
        raise NotImplementedError(
            f"{type(self).__name__} only supports batch_size=1 -- batch_select_indices is not supported."
        )

    def batch_split(self, full_batch_size, split_size):
        raise NotImplementedError(
            f"{type(self).__name__} only supports batch_size=1 -- batch_split is not supported."
        )

    def reset_state(self):
        """Explicit, opt-in reset back to EMPTY for reuse across requests.
        Not an HF Cache API override -- no `reset()` method exists on the
        base Cache/DynamicCache classes in this transformers version."""
        self.state = CacheState.EMPTY
        self.prefix_len = None
        self._groups_loaded = []
        self._layer_group = {}
        self._B_k = {}
        self._B_v = {}
        self._reset_group_storage()
        self.tail_len = 0
        self._tail_k.zero_()
        self._tail_v.zero_()

    def _reset_group_storage(self):
        raise NotImplementedError("subclass must implement _reset_group_storage")

    # ── Byte census ───────────────────────────────────────────────────────

    def byte_census(self):
        """'logical_bytes' here means real allocated bytes for every tensor
        this cache holds (tail + factors) -- NOT a logical-bit-count formula.
        Every tensor here is freshly allocated and never aliased across
        entries, so this equals the deduplicated storage size; revisit it if
        views or aliases are introduced (e.g. an optimization that slices one
        larger buffer)."""
        tail_bytes = self._tail_k.element_size() * self._tail_k.nelement() * 2  # k + v
        factor_bytes = self._factor_bytes()
        return {
            "state": self.state.value,
            "tail_bytes": tail_bytes,
            "factor_bytes": factor_bytes,
            "logical_bytes": tail_bytes + factor_bytes,
        }

    def _factor_bytes(self):
        raise NotImplementedError("subclass must implement _factor_bytes")

    # ── Debug-only invariant checker ──────────────────────────────────────

    def check_invariants(self):
        """Raises AssertionError on the first violated invariant. Debug-only
        -- not on the hot path; call explicitly in tests/manual debugging."""
        if self.state in (CacheState.FACTOR_READY, CacheState.DECODING_WITH_TAIL):
            assert len(self._groups_loaded) == self.n_groups, (
                f"factor_ready/decoding_with_tail requires all {self.n_groups} groups loaded, "
                f"got {len(self._groups_loaded)}"
            )
            assert set(self._layer_group.keys()) == set(range(self.num_layers)), (
                "factor_ready requires every layer to have a group assignment"
            )
            for B in list(self._B_k.values()) + list(self._B_v.values()):
                assert B.dtype == torch.bfloat16, "right factors must be bfloat16"
        assert self.tail_len <= self.tail_capacity, (
            f"tail_length {self.tail_len} exceeds configured tail_capacity {self.tail_capacity}"
        )
        for layer_idx in range(self.num_layers):
            expected = (self.prefix_len or 0) + self.tail_len
            actual = self.get_seq_length(layer_idx)
            assert actual == expected, f"get_seq_length({layer_idx}) = {actual}, expected {expected}"
        self._check_subclass_invariants()

    def _check_subclass_invariants(self):
        pass


class BF16FactorXKVCache(FactorizedXKVCache):
    """Persistent BF16 K_A/V_A per group and BF16 K_B/V_B per layer -- both
    left and right factors stored as real BF16 tensors, no quantization or
    packing."""

    def _store_group(self, group_idx, layer_ids, A_k, B_k_group, A_v, B_v_group):
        if A_k.dtype != torch.bfloat16 or A_v.dtype != torch.bfloat16:
            raise TypeError(
                f"BF16FactorXKVCache requires bfloat16 A_k/A_v, got {A_k.dtype}/{A_v.dtype}"
            )
        if B_k_group.dtype != torch.bfloat16 or B_v_group.dtype != torch.bfloat16:
            raise TypeError(
                f"BF16FactorXKVCache requires bfloat16 B_k/B_v, got {B_k_group.dtype}/{B_v_group.dtype}"
            )
        self._group_A_k[group_idx] = A_k
        self._group_A_v[group_idx] = A_v
        for pos, layer_idx in enumerate(layer_ids):
            self._B_k[layer_idx] = self.slice_layer_B(B_k_group, pos, self.num_kv_heads, self.head_dim)
            self._B_v[layer_idx] = self.slice_layer_B(B_v_group, pos, self.num_kv_heads, self.head_dim)

    def _get_group_A(self, group_idx):
        return self._group_A_k[group_idx], self._group_A_v[group_idx]

    def _reset_group_storage(self):
        self._group_A_k = {}
        self._group_A_v = {}

    def _factor_bytes(self):
        total = 0
        for tensors in (self._group_A_k.values(), self._group_A_v.values(),
                        self._B_k.values(), self._B_v.values()):
            for t in tensors:
                total += t.element_size() * t.nelement()
        return total

    def _check_subclass_invariants(self):
        for A in list(self._group_A_k.values()) + list(self._group_A_v.values()):
            assert A.dtype == torch.bfloat16, "bf16_ready requires persistent A dtype == bfloat16"


class PackedXKVCache(FactorizedXKVCache):
    """Per-group PACKED K_A/V_A (real 4-bit nibbles + FP32 row norms, via
    pack_turboquant_codes) and per-layer BF16 K_B/V_B with TurboQuant's
    rotation folded into the right factor. No tensor shaped
    [batch, sequence, rank] stays resident after a group is loaded:
    get_layer_factors() dequantizes the packed A into a TEMPORARY BF16
    tensor, so the attention code that consumes its output is identical for
    both cache types.

    Single-entry decode cache: within one decode step get_layer_factors() is
    called once per LAYER, and every layer in a group_size-layer group maps
    to the SAME group_idx, so unpacking and dequantizing on every call would
    repeat the same work group_size times per step -- and again on every
    later step, since the data never changes after prefill. _get_group_A()
    therefore caches only the MOST RECENTLY computed group's (A_k, A_v):
    consecutive calls for the same group_idx skip unpack+dequantize
    entirely, and requesting a different group_idx evicts the entry. At most
    ONE group's BF16 A is ever resident. Caching every group across steps
    would hold all n_groups BF16 A tensors at once -- the memory profile of
    BF16FactorXKVCache -- and erase PackedXKVCache's memory advantage, so it
    is deliberately not done."""

    def __init__(self, *, k_codec, v_codec, **kwargs):
        super().__init__(**kwargs)
        if k_codec.bits != 4 or v_codec.bits != 4:
            raise ValueError(
                f"PackedXKVCache only supports bits=4 (nibble packing), "
                f"got k_codec.bits={k_codec.bits}, v_codec.bits={v_codec.bits}"
            )
        self.k_codec = k_codec
        self.v_codec = v_codec
        self._group_packed_k = {}
        self._group_packed_v = {}
        self._cached_group_idx = None
        self._cached_group_A = None

    def _store_group(self, group_idx, layer_ids, A_k, B_k_group, A_v, B_v_group):
        # fake_quantize_pair returns BOTH the real quantization codes (for A)
        # AND the rotation-folded right factor (for B) in ONE call -- the same
        # numerical pipeline used for the K4/V4 accuracy runs (same codebook
        # lookups, same folding math), so quantization is not reimplemented
        # here.
        pair_k = self.k_codec.fake_quantize_pair(A_k, B_k_group)
        pair_v = self.v_codec.fake_quantize_pair(A_v, B_v_group)

        self._group_packed_k[group_idx] = pack_turboquant_codes(pair_k.codes)
        self._group_packed_v[group_idx] = pack_turboquant_codes(pair_v.codes)

        B_k_folded_bf16 = pair_k.right_factor.to(torch.bfloat16)
        B_v_folded_bf16 = pair_v.right_factor.to(torch.bfloat16)
        for pos, layer_idx in enumerate(layer_ids):
            self._B_k[layer_idx] = self.slice_layer_B(B_k_folded_bf16, pos, self.num_kv_heads, self.head_dim)
            self._B_v[layer_idx] = self.slice_layer_B(B_v_folded_bf16, pos, self.num_kv_heads, self.head_dim)

    def _get_group_A(self, group_idx):
        if group_idx == self._cached_group_idx:
            return self._cached_group_A

        packed_k = self._group_packed_k[group_idx]
        packed_v = self._group_packed_v[group_idx]
        # Catch a codec/codes mismatch (e.g. K codes with the V codec)
        # explicitly, right here, rather than letting a wrong-codec
        # dequantize silently produce garbage.
        validate_packed_codes(packed_k, self.k_codec)
        validate_packed_codes(packed_v, self.v_codec)

        # A fresh BF16 tensor -- never stored back into self._group_packed_k/v
        # (the packed storage itself is untouched). Cached ONLY for this one
        # group_idx, evicted the instant a different group is requested -- see
        # the class docstring for why this is deliberately not a persistent,
        # cross-group cache. dequant_fused() replaces
        # unpack_turboquant_codes()+dequantize_rotated() (~9 separate
        # Python/CUDA kernel launches) with ONE Triton kernel launch that
        # produces the same materialized BF16 tensor, bit-identical. This is
        # only reached on a cache MISS of the single-entry cache above.
        A_k_temp = dequant_fused(
            packed_k.packed, packed_k.norms.squeeze(-1).contiguous(),
            self.k_codec.centroids, self.k_codec.dimension,
        )
        A_v_temp = dequant_fused(
            packed_v.packed, packed_v.norms.squeeze(-1).contiguous(),
            self.v_codec.centroids, self.v_codec.dimension,
        )
        self._cached_group_idx = group_idx
        self._cached_group_A = (A_k_temp, A_v_temp)
        return self._cached_group_A

    def _reset_group_storage(self):
        self._group_packed_k = {}
        self._group_packed_v = {}
        self._cached_group_idx = None
        self._cached_group_A = None

    def _factor_bytes(self):
        total = 0
        for packed in list(self._group_packed_k.values()) + list(self._group_packed_v.values()):
            total += packed.storage_bytes
        for B in list(self._B_k.values()) + list(self._B_v.values()):
            total += B.element_size() * B.nelement()
        return total

    def _check_subclass_invariants(self):
        for packed in list(self._group_packed_k.values()) + list(self._group_packed_v.values()):
            assert packed.packed.dtype == torch.uint8, "packed_ready requires k/v codes dtype == uint8"
            assert packed.packed.ndim >= 1 and packed.packed.shape[-1] == packed.dimension // 2, (
                "packed_ready requires no persistent tensor shaped [batch, sequence, rank] -- "
                "the packed tensor's last dim must be rank//2, never rank"
            )
        for B in list(self._B_k.values()) + list(self._B_v.values()):
            assert B.dtype == torch.bfloat16, "folded right factor must be bfloat16"
        if self._cached_group_idx is not None:
            assert self._cached_group_idx in self._group_packed_k, (
                "single-entry decode cache points at a group_idx that was never loaded"
            )
            cached_A_k, cached_A_v = self._cached_group_A
            assert cached_A_k.dtype == torch.bfloat16 and cached_A_v.dtype == torch.bfloat16, (
                "single-entry decode cache must hold bfloat16 tensors"
            )
