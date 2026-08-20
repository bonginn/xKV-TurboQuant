from dataclasses import replace

import pytest
import torch

from xKV.customized_cache.turboquant import (
    TurboQuantMSE,
    build_haar_rotation,
    load_codebook,
)


def _nonsymmetric_rotation():
    return torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_haar_rotation_is_orthogonal_deterministic_and_rng_local():
    torch.manual_seed(1234)
    state_before = torch.get_rng_state().clone()

    first = build_haar_rotation(12, 77)
    state_after = torch.get_rng_state()
    second = build_haar_rotation(12, 77)
    other = build_haar_rotation(12, 78)

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    identity = torch.eye(12)
    assert torch.allclose(first.T @ first, identity, atol=2e-6, rtol=0)
    assert torch.allclose(first @ first.T, identity, atol=2e-6, rtol=0)


def test_row_major_formula_and_index_selection():
    rotation = _nonsymmetric_rotation()
    quantizer = TurboQuantMSE(
        3,
        1,
        rotation=rotation,
        chunk_size=1,
        norm_dtype=torch.float32,
    )
    value = torch.tensor(
        [[2.0, -1.0, 3.0], [-4.0, 2.0, 1.0]],
        dtype=torch.float32,
    )

    codes = quantizer.encode(value)
    norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    unit = value / norms
    rotated = unit @ rotation.T
    manual_indices = torch.bucketize(
        rotated,
        quantizer.decision_boundaries,
        right=False,
    ).to(torch.uint8)
    y_hat = quantizer.centroids[manual_indices.long()]
    manual_decode = (norms * y_hat) @ rotation

    assert torch.equal(codes.indices, manual_indices)
    assert torch.allclose(quantizer.decode(codes), manual_decode, atol=1e-7)


def test_zero_rows_are_exact_zero_without_nan():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=5)
    value = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        dtype=torch.float32,
    )

    codes = quantizer.encode(value)
    decoded = quantizer.decode(codes)

    assert codes.norms[0].item() == 0.0
    assert torch.equal(decoded[0], torch.zeros(3))
    assert torch.isfinite(decoded).all()


def test_arbitrary_leading_dimensions_and_noncontiguous_input():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=9, chunk_size=2)
    source = torch.randn(2, 5, 3, dtype=torch.float32)
    value = source.transpose(0, 1)
    assert not value.is_contiguous()

    codes = quantizer.encode(value)
    decoded = quantizer.decode(codes)

    assert codes.indices.shape == value.shape
    assert codes.norms.shape == value.shape[:-1] + (1,)
    assert decoded.shape == value.shape
    assert decoded.dtype == value.dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_input_uses_fp32_reference_compute(dtype):
    quantizer = TurboQuantMSE(3, 2, rotation_seed=9)
    value = torch.randn(6, 3, dtype=torch.float32).to(dtype)

    codes = quantizer.encode(value)
    decoded = quantizer.decode(codes)

    assert codes.norms.dtype == torch.float32
    assert decoded.dtype == dtype
    assert torch.isfinite(decoded).all()


def test_fp16_norm_storage_is_explicit_and_matches_fp32_rounding():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=9, norm_dtype=torch.float16)
    value = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]])

    codes = quantizer.encode(value)
    expected = torch.linalg.vector_norm(value, dim=-1, keepdim=True).to(torch.float16)

    assert codes.norms.dtype == torch.float16
    assert torch.equal(codes.norms, expected)


def test_empty_leading_dimension_round_trips():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=9)
    value = torch.empty(2, 0, 3)

    codes = quantizer.encode(value)
    decoded = quantizer.decode(codes)

    assert codes.indices.shape == (2, 0, 3)
    assert codes.norms.shape == (2, 0, 1)
    assert decoded.shape == value.shape


def test_positive_scale_equivariance():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=10)
    value = torch.randn(7, 3, dtype=torch.float32)
    scales = torch.tensor(
        [[0.25], [0.5], [1.0], [2.0], [3.0], [4.0], [8.0]],
        dtype=torch.float32,
    )

    base = quantizer(value)
    scaled = quantizer(value * scales)

    assert torch.allclose(scaled, base * scales, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("batched", [False, True])
def test_folded_decoder_matches_explicit_decode(batched):
    quantizer = TurboQuantMSE(
        3,
        2,
        rotation=_nonsymmetric_rotation(),
        chunk_size=2,
    )
    left = torch.randn(2, 4, 3, dtype=torch.float32)
    if batched:
        right = torch.randn(2, 3, 5, dtype=torch.float32)
    else:
        right = torch.randn(3, 5, dtype=torch.float32)

    result = quantizer.fake_quantize_pair(left, right)
    explicit = quantizer.decode(
        result.codes,
        output_dtype=torch.float32,
    ) @ right
    folded = result.left_factor @ result.right_factor

    assert torch.allclose(explicit, folded, atol=2e-6, rtol=2e-6)


def test_chunked_and_unchunked_paths_match():
    rotation = build_haar_rotation(3, 22)
    codebook = load_codebook(3, 2).centroids
    chunked = TurboQuantMSE(
        3,
        2,
        rotation=rotation,
        codebook=codebook,
        chunk_size=3,
    )
    unchunked = TurboQuantMSE(
        3,
        2,
        rotation=rotation,
        codebook=codebook,
        chunk_size=None,
    )
    value = torch.randn(4, 5, 3, dtype=torch.float32)

    chunked_codes = chunked.encode(value)
    unchunked_codes = unchunked.encode(value)
    assert torch.equal(chunked_codes.indices, unchunked_codes.indices)
    assert torch.equal(chunked_codes.norms, unchunked_codes.norms)
    assert torch.allclose(
        chunked.decode(chunked_codes),
        unchunked.decode(unchunked_codes),
        atol=1e-7,
        rtol=0,
    )


def test_invalid_values_and_indices_fail_fast():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=31)

    with pytest.raises(ValueError, match="last dimension"):
        quantizer.encode(torch.randn(2, 4))
    with pytest.raises(TypeError, match="floating point"):
        quantizer.encode(torch.ones(2, 3, dtype=torch.int64))
    bad_value = torch.randn(2, 3)
    bad_value[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        quantizer.encode(bad_value)

    codes = quantizer.encode(torch.randn(2, 3))
    bad_indices = codes.indices.clone()
    bad_indices[0, 0] = quantizer.levels
    bad_codes = replace(codes, indices=bad_indices)
    with pytest.raises(ValueError, match="code indices"):
        quantizer.decode(bad_codes)
    with pytest.raises(ValueError, match="output_dtype"):
        quantizer.decode(codes, output_dtype=torch.int32)
    with pytest.raises(ValueError, match="output_dtype"):
        quantizer.fold_decoder(torch.randn(3, 4), output_dtype=torch.int32)


def test_codes_cannot_be_decoded_with_a_different_rotation():
    source = TurboQuantMSE(3, 2, rotation_seed=31)
    other = TurboQuantMSE(3, 2, rotation_seed=32)
    codes = source.encode(torch.randn(2, 3))

    with pytest.raises(ValueError, match="different rotation"):
        other.decode(codes)


def test_state_dict_load_refreshes_rotation_identity():
    source = TurboQuantMSE(3, 2, rotation_seed=31)
    target = TurboQuantMSE(3, 2, rotation_seed=32)
    codes = source.encode(torch.randn(2, 3))

    target.load_state_dict(source.state_dict())

    assert target.rotation_checksum == source.rotation_checksum
    assert torch.equal(target.decode(codes), source.decode(codes))


def test_state_dict_load_rejects_nonfinite_rotation():
    source = TurboQuantMSE(3, 2, rotation_seed=31)
    target = TurboQuantMSE(3, 2, rotation_seed=32)
    state = source.state_dict()
    state["rotation"] = state["rotation"].clone()
    state["rotation"][0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        target.load_state_dict(state)


def test_codes_cannot_be_decoded_with_a_different_codebook():
    source = TurboQuantMSE(3, 2, rotation_seed=31)
    other = TurboQuantMSE(
        3,
        2,
        rotation=source.rotation,
        codebook=(-0.8, -0.2, 0.2, 0.8),
    )
    codes = source.encode(torch.randn(2, 3))

    with pytest.raises(ValueError, match="different codebook"):
        other.decode(codes)


def test_outer_autocast_does_not_change_reference_math():
    quantizer = TurboQuantMSE(3, 2, rotation_seed=41)
    value = torch.randn(8, 3)
    reference = quantizer(value)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        observed = quantizer(value)

    assert torch.equal(observed, reference)


@pytest.mark.parametrize("dimension,bits", [(384, 3), (576, 2)])
def test_target_rank_codec_shapes(dimension, bits):
    quantizer = TurboQuantMSE(
        dimension,
        bits,
        rotation=torch.eye(dimension),
        chunk_size=1,
    )
    value = torch.zeros(2, dimension)
    value[0, 0] = 1.0

    codes = quantizer.encode(value)
    decoded = quantizer.decode(codes)

    assert codes.indices.shape == value.shape
    assert codes.norms.shape == (2, 1)
    assert decoded.shape == value.shape
    assert torch.equal(decoded[1], torch.zeros(dimension))


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_isotropic_distortion_tracks_offline_codebook(bits):
    dimension = 96
    row_count = 2048
    quantizer = TurboQuantMSE(dimension, bits, rotation_seed=42, chunk_size=256)
    value = torch.randn(row_count, dimension, dtype=torch.float32)
    value = value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)

    decoded = quantizer(value)
    observed = torch.mean(torch.sum((value - decoded) ** 2, dim=-1)).item()
    expected = load_codebook(dimension, bits).normalized_mse

    assert observed == pytest.approx(expected, rel=0.15, abs=0.003)
