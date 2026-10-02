"""Unit tests for true 4-bit packing: pack_turboquant_codes /
unpack_turboquant_codes / PackedTurboQuantCodes / validate_packed_codes.

Covers:
  - exhaustive 0-15 nibble pair round-trip
  - random rank384/rank576 bit-exact round-trip
  - invalid index >15 fails
  - checksum mismatch fails
  - packed tensor size is exactly logical code count / 2 bytes
  - no implicit uint8-per-code fallback (packed last dim is dimension//2, not dimension)
"""
import torch
import pytest

from xKV.customized_cache.turboquant import (
    TurboQuantMSE,
    TurboQuantCodes,
    PackedTurboQuantCodes,
    pack_turboquant_codes,
    unpack_turboquant_codes,
    validate_packed_codes,
)


def _make_codes(dimension, bits, indices, norms=None, codebook_checksum="cb", rotation_checksum="rot",
                 source_dtype=torch.bfloat16):
    """Build a bare TurboQuantCodes for tests that don't need a real codec
    (e.g. testing pack/unpack's own validation logic in isolation)."""
    if norms is None:
        norms = torch.ones(indices.shape[:-1] + (1,), dtype=torch.float32)
    return TurboQuantCodes(
        indices=indices,
        norms=norms,
        dimension=dimension,
        bits=bits,
        source_dtype=source_dtype,
        codebook_checksum=codebook_checksum,
        rotation_checksum=rotation_checksum,
    )


# ── Exhaustive 0-15 nibble pair round-trip ───────────────────────────────

def test_exhaustive_nibble_pair_roundtrip():
    """All 256 (even, odd) 4-bit value combinations pack/unpack bit-exactly."""
    pairs = [(e, o) for e in range(16) for o in range(16)]
    even_vals = torch.tensor([p[0] for p in pairs], dtype=torch.uint8)
    odd_vals = torch.tensor([p[1] for p in pairs], dtype=torch.uint8)
    indices = torch.stack([even_vals, odd_vals], dim=-1)  # [256, 2]

    codes = _make_codes(dimension=2, bits=4, indices=indices)
    packed = pack_turboquant_codes(codes)

    assert packed.packed.shape == (256, 1)
    assert packed.packed.dtype == torch.uint8

    for i, (e, o) in enumerate(pairs):
        expected_byte = e | (o << 4)
        assert int(packed.packed[i, 0].item()) == expected_byte, f"pair ({e},{o})"

    restored = unpack_turboquant_codes(packed)
    assert torch.equal(restored.indices, indices)


# ── Random rank384/rank576 bit-exact round-trip (real codecs) ───────────

@pytest.mark.parametrize("dimension", [384, 576])
def test_random_rank_bitexact_roundtrip(dimension):
    torch.manual_seed(0)
    quantizer = TurboQuantMSE(dimension, 4)
    value = torch.randn(3, 17, dimension, dtype=torch.bfloat16)  # arbitrary leading dims

    codes = quantizer.encode(value)
    packed = pack_turboquant_codes(codes)
    restored = unpack_turboquant_codes(packed)

    assert torch.equal(restored.indices, codes.indices)
    assert torch.equal(restored.norms, codes.norms.to(torch.float32))
    assert restored.dimension == codes.dimension
    assert restored.bits == codes.bits
    assert restored.codebook_checksum == codes.codebook_checksum
    assert restored.rotation_checksum == codes.rotation_checksum


def test_empty_tensor_roundtrip():
    quantizer = TurboQuantMSE(96, 4)
    value = torch.randn(0, 96, dtype=torch.bfloat16)
    codes = quantizer.encode(value)
    packed = pack_turboquant_codes(codes)
    assert packed.packed.numel() == 0
    restored = unpack_turboquant_codes(packed)
    assert restored.indices.numel() == 0
    assert restored.indices.shape[-1] == 96


def test_noncontiguous_input_roundtrip():
    quantizer = TurboQuantMSE(96, 4)
    value = torch.randn(10, 96, dtype=torch.bfloat16)
    codes = quantizer.encode(value)
    # Make indices non-contiguous via a transpose-transpose round trip on a
    # batched view, forcing a real non-contiguous tensor.
    batched = codes.indices.unsqueeze(0).expand(4, -1, -1)  # non-contiguous (stride-0 dim)
    noncontig_codes = _make_codes(
        dimension=96, bits=4, indices=batched, norms=codes.norms.unsqueeze(0).expand(4, -1, -1),
        codebook_checksum=codes.codebook_checksum, rotation_checksum=codes.rotation_checksum,
    )
    assert not noncontig_codes.indices.is_contiguous()
    packed = pack_turboquant_codes(noncontig_codes)
    restored = unpack_turboquant_codes(packed)
    assert torch.equal(restored.indices, batched)


# ── Invalid index > 15 fails ──────────────────────────────────────────────

def test_invalid_index_above_15_fails():
    indices = torch.tensor([[0, 1, 16, 3]], dtype=torch.uint8)  # 16 is out of range
    codes = _make_codes(dimension=4, bits=4, indices=indices)
    with pytest.raises(ValueError, match="out of range"):
        pack_turboquant_codes(codes)


def test_valid_boundary_index_15_succeeds():
    indices = torch.tensor([[0, 15, 15, 0]], dtype=torch.uint8)
    codes = _make_codes(dimension=4, bits=4, indices=indices)
    packed = pack_turboquant_codes(codes)  # must not raise
    restored = unpack_turboquant_codes(packed)
    assert torch.equal(restored.indices, indices)


def test_wrong_bits_fails_closed():
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    codes = _make_codes(dimension=4, bits=3, indices=indices)  # bits=3, not 4
    with pytest.raises(ValueError, match="bits=4"):
        pack_turboquant_codes(codes)


def test_odd_dimension_fails_closed():
    indices = torch.tensor([[0, 1, 2]], dtype=torch.uint8)  # dimension=3, odd
    codes = _make_codes(dimension=3, bits=4, indices=indices)
    with pytest.raises(ValueError, match="even dimension"):
        pack_turboquant_codes(codes)


def test_non_uint8_indices_fails():
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    codes = _make_codes(dimension=4, bits=4, indices=indices)
    with pytest.raises(TypeError, match="uint8"):
        pack_turboquant_codes(codes)


# ── Checksum / codebook mismatch fails (K-vs-V validation separation) ────

def test_checksum_mismatch_fails():
    torch.manual_seed(1)
    q384 = TurboQuantMSE(384, 4)
    q576 = TurboQuantMSE(576, 4)  # different dimension AND different codebook/rotation

    value = torch.randn(2, 384, dtype=torch.bfloat16)
    codes = q384.encode(value)
    packed = pack_turboquant_codes(codes)

    # Using the K-side packed codes with the V-side quantizer must fail --
    # the wrong codec must never be silently accepted.
    with pytest.raises(ValueError):
        validate_packed_codes(packed, q576)

    # The matching quantizer must validate cleanly.
    validate_packed_codes(packed, q384)


def test_rotation_seed_mismatch_fails():
    torch.manual_seed(2)
    q_a = TurboQuantMSE(96, 4, rotation_seed=42)
    q_b = TurboQuantMSE(96, 4, rotation_seed=43)  # same dimension/bits, different rotation

    value = torch.randn(2, 96, dtype=torch.bfloat16)
    codes = q_a.encode(value)
    packed = pack_turboquant_codes(codes)

    with pytest.raises(ValueError, match="rotation"):
        validate_packed_codes(packed, q_b)


# ── Packed tensor size is exactly logical code count / 2 bytes ──────────

@pytest.mark.parametrize("dimension", [96, 384, 576])
def test_packed_size_is_half_logical_count(dimension):
    torch.manual_seed(3)
    quantizer = TurboQuantMSE(dimension, 4)
    value = torch.randn(5, 7, dimension, dtype=torch.bfloat16)
    codes = quantizer.encode(value)
    packed = pack_turboquant_codes(codes)

    assert packed.packed.numel() == codes.indices.numel() // 2
    assert packed.packed.element_size() == 1  # uint8: 1 byte per element, not "4 bits" abstractly

    # storage_bytes must reflect REAL allocated bytes, not a logical-bit formula.
    expected_bytes = packed.packed.numel() * 1 + packed.norms.numel() * packed.norms.element_size()
    assert packed.storage_bytes == expected_bytes


# ── No implicit uint8-per-code fallback ──────────────────────────────────

@pytest.mark.parametrize("dimension", [96, 384, 576])
def test_packed_last_dim_is_dimension_over_2_not_dimension(dimension):
    """packed's last dim must be dimension//2, never silently fall back to
    `dimension` (which would mean "packing" did nothing -- still one byte
    per logical code)."""
    torch.manual_seed(4)
    quantizer = TurboQuantMSE(dimension, 4)
    value = torch.randn(2, dimension, dtype=torch.bfloat16)
    codes = quantizer.encode(value)
    packed = pack_turboquant_codes(codes)

    assert packed.packed.shape[-1] == dimension // 2
    assert packed.packed.shape[-1] != dimension
