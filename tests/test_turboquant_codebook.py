import json
import math

import pytest

from xKV.customized_cache.turboquant import (
    centroid_checksum,
    load_codebook,
)


@pytest.mark.parametrize(
    "dimension,bits",
    [
        (3, 1),
        (96, 4),
        (384, 3),
        (384, 4),
        (576, 2),
        (576, 3),
        (576, 4),
    ],
)
def test_offline_codebook_structure(dimension, bits):
    spec = load_codebook(dimension, bits)

    assert len(spec.centroids) == 2**bits
    assert len(spec.decision_boundaries) == 2**bits - 1
    assert all(math.isfinite(value) for value in spec.centroids)
    assert all(-1.0 <= value <= 1.0 for value in spec.centroids)
    assert all(
        left < right
        for left, right in zip(spec.centroids[:-1], spec.centroids[1:])
    )
    assert spec.centroids == pytest.approx(
        tuple(-value for value in reversed(spec.centroids)),
        abs=1e-14,
    )
    assert spec.decision_boundaries == pytest.approx(
        tuple(
            0.5 * (left + right)
            for left, right in zip(spec.centroids[:-1], spec.centroids[1:])
        ),
        abs=1e-15,
    )
    assert centroid_checksum(spec.centroids) == spec.checksum


def test_uniform_sphere_d3_one_bit_has_analytic_centroids():
    # For d=3 the coordinate density is uniform on [-1, 1].
    spec = load_codebook(3, 1)
    assert spec.centroids == pytest.approx((-0.5, 0.5), abs=1e-13)
    assert spec.normalized_mse == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_uniform_sphere_d3_codebooks_are_analytic(bits):
    levels = 2**bits
    expected_centroids = tuple(
        -1.0 + (2.0 * index + 1.0) / levels
        for index in range(levels)
    )
    spec = load_codebook(3, bits)

    assert spec.centroids == pytest.approx(expected_centroids, abs=1e-12)
    assert spec.normalized_mse == pytest.approx(4.0**(-bits), abs=1e-12)


@pytest.mark.parametrize("dimension", [384, 576])
def test_one_bit_codebook_matches_finite_dimension_closed_form(dimension):
    positive_centroid = math.exp(
        math.lgamma(dimension / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension + 1.0) / 2.0)
    )
    expected_mse = 1.0 - dimension * positive_centroid**2
    spec = load_codebook(dimension, 1)

    assert spec.centroids == pytest.approx(
        (-positive_centroid, positive_centroid),
        abs=1e-13,
    )
    assert spec.normalized_mse == pytest.approx(expected_mse, abs=1e-12)


@pytest.mark.parametrize(
    "dimension,bits,expected_checksum",
    [
        (
            384,
            3,
            "495da22e15e95a6a25a20ef1cb44d06867106bc73b3be3f52391728a85ca03dc",
        ),
        (
            576,
            2,
            "de6f0c2be0b05b1d863a3df0fb8ee8e63245b6811620073246f4829d12cfcddf",
        ),
    ],
)
def test_target_codebooks_have_reviewed_checksums(
    dimension,
    bits,
    expected_checksum,
):
    assert load_codebook(dimension, bits).checksum == expected_checksum


def test_missing_codebook_fails_closed():
    with pytest.raises(ValueError, match="is missing"):
        load_codebook(17, 3)


def test_nonfinite_stored_boundary_fails_closed(tmp_path):
    source = load_codebook(3, 1)
    payload = {
        "format_version": 1,
        "generator_version": "test",
        "distribution": "sphere_coordinate_beta",
        "entries": {
            "3:1": {
                "dimension": 3,
                "bits": 1,
                "centroids": list(source.centroids),
                "decision_boundaries": [float("nan")],
                "normalized_mse": source.normalized_mse,
                "centroid_checksum": source.checksum,
            }
        },
    }
    path = tmp_path / "bad_codebook.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite decision boundaries"):
        load_codebook(3, 1, path)
