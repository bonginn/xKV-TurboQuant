"""Real memory-saving Cache for the "compressed prefix + dense decode tail"
design.

The compressed prefix (A/B factors) lives entirely OUTSIDE this cache: it is
computed once via SVD and held by the caller (see
scripts/eval_ruler_wired_cache.py). This cache only stores the small, growing
dense TAIL of decoded tokens, pre-allocated once to `tail_capacity` (512 in
the evaluation scripts) and failing closed if that capacity is exceeded
instead of silently overwriting the cache. Because the full-size dense prefix
K/V is not kept after prefill, memory during decode no longer scales with the
prompt length.
"""
import torch
from transformers.cache_utils import Cache


class XKVWiredCache(Cache):
    """Real dense-tail-only cache. `update()` is expected to be called
    exactly once per layer per decode step, for ALL `num_layers` layers, in
    strictly increasing layer_idx order (standard Qwen2Model.forward()
    behavior) -- this is how a single shared `tail_len` counter stays
    correct without a per-layer counter. Only supports decode (seq_len=1
    per call); prefill must be handled by a separate, real dense cache
    upstream of this one (prefill runs dense attention to produce its own output).
    """

    def __init__(self, num_layers, batch_size, num_kv_heads, head_dim, tail_capacity, device, dtype):
        super().__init__()
        if batch_size != 1:
            # This design is scoped to batch=1, greedy decode.
            # Fail loudly and immediately at construction time, rather than
            # letting a batch>1 caller hit a confusing shape-mismatch deep
            # inside update() (or silently broadcast/truncate).
            raise NotImplementedError(
                f"XKVWiredCache only supports batch_size=1 (batch>1 is not supported), "
                f"got batch_size={batch_size}."
            )
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.tail_capacity = tail_capacity
        self.tail_len = 0
        self.prefix_len = None  # set via set_prefix_len() before first decode step
        # Single fixed-size allocation for the whole cache's lifetime -- this
        # IS the real memory-saving property: independent of how long the
        # original prefill context was, this is always
        # num_layers * batch * num_kv_heads * tail_capacity * head_dim * 2(bf16) * 2(k&v) bytes
        # (~96 MiB at this repo's Qwen2.5-14B shape with tail_capacity=512).
        self._tail_k = torch.zeros(
            num_layers, batch_size, num_kv_heads, tail_capacity, head_dim, device=device, dtype=dtype,
        )
        self._tail_v = torch.zeros(
            num_layers, batch_size, num_kv_heads, tail_capacity, head_dim, device=device, dtype=dtype,
        )

    def set_prefix_len(self, prefix_len: int):
        self.prefix_len = prefix_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        seq_len = key_states.shape[2]
        if seq_len != 1:
            raise RuntimeError(
                f"XKVWiredCache only supports decode (seq_len=1 per step), got seq_len={seq_len}. "
                "Prefill must use a separate real dense cache upstream of this one."
            )
        if key_states.shape[0] != self.batch_size:
            # Defense in depth: __init__ already rejects batch_size != 1, but
            # this catches the case where some caller expands the batch dim
            # later (e.g. beam search's batch-duplication helpers) without
            # going through reorder_cache().
            raise NotImplementedError(
                f"XKVWiredCache was constructed with batch_size={self.batch_size} but received "
                f"key_states with batch dim {key_states.shape[0]} -- batch>1 is not supported."
            )
        if self.tail_len >= self.tail_capacity:
            raise RuntimeError(
                f"XKVWiredCache: tail_capacity={self.tail_capacity} exceeded -- failing closed "
                "instead of silently overwriting the cache."
            )
        pos = self.tail_len
        self._tail_k[layer_idx, :, :, pos:pos + 1, :] = key_states
        self._tail_v[layer_idx, :, :, pos:pos + 1, :] = value_states
        if layer_idx == self.num_layers - 1:
            # Only the LAST layer processed in this step advances the shared
            # counter, so all 48 per-layer update() calls within one decode
            # step read back the SAME tail window (0..pos+1), not 48
            # different windows.
            self.tail_len += 1
        return self._tail_k[layer_idx, :, :, :pos + 1, :], self._tail_v[layer_idx, :, :, :pos + 1, :]

    def get_seq_length(self, layer_idx=0):
        return (self.prefix_len or 0) + self.tail_len

    def get_max_cache_shape(self):
        return (self.prefix_len or 0) + self.tail_capacity

    def real_bytes(self):
        return self._tail_k.element_size() * self._tail_k.nelement() * 2  # k + v

    def reorder_cache(self, beam_idx):
        # The base Cache.reorder_cache() reads self.key_cache/self.value_cache
        # (DynamicCache's storage layout), which this class doesn't have --
        # calling it as-is would raise a confusing AttributeError deep inside
        # beam search machinery. Fail explicitly and clearly instead: this
        # design is scoped to greedy decode, batch=1.
        raise NotImplementedError(
            "XKVWiredCache does not support beam search (reorder_cache) -- this design is "
            "scoped to greedy decode only."
        )
