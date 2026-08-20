#!/usr/bin/env python3
"""Generate finite-dimensional TurboQuant-MSE Lloyd-Max codebooks offline.

The target density is the exact marginal density of one coordinate of a point
uniformly distributed on the unit sphere S^(d-1):

    f_d(x) proportional to (1 - x^2)^((d - 3) / 2), x in [-1, 1].

The implementation uses deterministic FP64 Gauss-Legendre quadrature within
each Lloyd interval.  It has no dependence on model activations or calibration
data.  The resulting JSON is consumed by ``turboquant.py`` at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


FORMAT_VERSION = 1
GENERATOR_VERSION = "beta_lloyd_gauss_legendre_v3"
DEFAULT_DIMENSIONS = (3, 96, 144, 192, 288, 384, 512, 576, 768)
DEFAULT_BITS = (1, 2, 3, 4)


def _canonical_float_list(values: Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in values],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def centroid_checksum(values: Sequence[float]) -> str:
    return hashlib.sha256(_canonical_float_list(values).encode("ascii")).hexdigest()


def _interval_moments(
    lower: float,
    upper: float,
    *,
    exponent: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    """Return unnormalized zeroth, first, and second moments on one bin."""

    if exponent == -0.5:
        # d=2 has integrable endpoint singularities that plain Gauss-Legendre
        # quadrature does not resolve reliably.  With x=sin(theta), the three
        # required moments have elementary antiderivatives.
        lower = min(1.0, max(-1.0, lower))
        upper = min(1.0, max(-1.0, upper))
        lower_root = math.sqrt(max(0.0, 1.0 - lower * lower))
        upper_root = math.sqrt(max(0.0, 1.0 - upper * upper))
        lower_angle = math.asin(lower)
        upper_angle = math.asin(upper)
        mass = upper_angle - lower_angle
        first = lower_root - upper_root
        lower_second = 0.5 * (lower_angle - lower * lower_root)
        upper_second = 0.5 * (upper_angle - upper * upper_root)
        return mass, first, upper_second - lower_second

    midpoint = 0.5 * (upper + lower)
    half_width = 0.5 * (upper - lower)
    points = half_width * nodes + midpoint
    log_density = exponent * np.log1p(-(points * points))

    # Scaling prevents underflow in tail bins; the common density normalizer
    # cancels from both Lloyd centroid updates and normalized distortion.
    log_scale = float(np.max(log_density))
    scaled_weights = weights * np.exp(log_density - log_scale)
    scale = half_width * math.exp(log_scale)
    mass = scale * float(np.sum(scaled_weights))
    first = scale * float(np.sum(scaled_weights * points))
    second = scale * float(np.sum(scaled_weights * points * points))
    return mass, first, second


def generate_codebook(
    dimension: int,
    bits: int,
    *,
    quadrature_order: int = 64,
    tolerance: float = 1e-13,
    max_iterations: int = 2000,
) -> Dict[str, object]:
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not 1 <= bits <= 4:
        raise ValueError(
            "the verified reference generator supports bits between 1 and 4"
        )
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    levels = 1 << bits
    if dimension == 3:
        # For d=3 the sphere-coordinate density is exactly uniform on [-1, 1].
        # Its Lloyd-Max solution has equal-width bins and exact binary-rational
        # centroids for power-of-two level counts.  Use the closed form instead
        # of retaining a small iterative/quadrature residual in the asset.
        bin_width = 2.0 / levels
        centroids = -1.0 + (np.arange(levels, dtype=np.float64) + 0.5) * bin_width
        boundaries = -1.0 + np.arange(1, levels, dtype=np.float64) * bin_width
        centroid_values = [float(value) for value in centroids.tolist()]
        return {
            "dimension": dimension,
            "bits": bits,
            "centroids": centroid_values,
            "decision_boundaries": [
                float(value) for value in boundaries.tolist()
            ],
            "normalized_mse": float(1.0 / (levels * levels)),
            "iterations": 0,
            "centroid_checksum": centroid_checksum(centroid_values),
        }

    exponent = 0.5 * (dimension - 3)
    nodes, weights = leggauss(quadrature_order)

    # Gaussian quantiles provide a stable high-dimensional initialization.
    # Lloyd iterations use the exact finite-d sphere density thereafter.
    probabilities = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    normal = NormalDist()
    centroids = np.asarray(
        [normal.inv_cdf(float(probability)) / math.sqrt(dimension)
         for probability in probabilities],
        dtype=np.float64,
    )
    maximum = float(np.max(np.abs(centroids)))
    if maximum >= 1.0:
        # Keep every initial centroid strictly inside the sphere support while
        # preserving the Gaussian-quantile ordering.  This matters for small d
        # and larger bit widths; Lloyd updates still use the exact density.
        centroids *= (1.0 - 0.5 / levels) / maximum

    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        boundaries = np.concatenate(
            (
                np.asarray([-1.0]),
                0.5 * (centroids[:-1] + centroids[1:]),
                np.asarray([1.0]),
            )
        )
        updated = np.empty_like(centroids)
        for index, (lower, upper) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            mass, first, _ = _interval_moments(
                float(lower),
                float(upper),
                exponent=exponent,
                nodes=nodes,
                weights=weights,
            )
            if not mass > 0 or not math.isfinite(mass):
                raise RuntimeError(
                    f"empty/non-finite Lloyd bin for d={dimension}, b={bits}, "
                    f"interval=[{lower}, {upper}]"
                )
            updated[index] = first / mass

        # The target density is exactly symmetric.  Enforcing antisymmetry
        # prevents harmless quadrature roundoff from entering the asset.
        updated = 0.5 * (updated - updated[::-1])
        max_delta = float(np.max(np.abs(updated - centroids)))
        centroids = updated
        if max_delta <= tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Lloyd-Max did not converge for d={dimension}, b={bits}; "
            f"last max delta={max_delta}"
        )

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    full_boundaries = np.concatenate(
        (np.asarray([-1.0]), boundaries, np.asarray([1.0]))
    )
    total_mass = 0.0
    total_distortion = 0.0
    for centroid, lower, upper in zip(
        centroids,
        full_boundaries[:-1],
        full_boundaries[1:],
    ):
        mass, first, second = _interval_moments(
            float(lower),
            float(upper),
            exponent=exponent,
            nodes=nodes,
            weights=weights,
        )
        total_mass += mass
        total_distortion += (
            second - 2.0 * float(centroid) * first
            + float(centroid * centroid) * mass
        )

    centroid_values = [float(value) for value in centroids.tolist()]
    boundary_values = [float(value) for value in boundaries.tolist()]
    normalized_mse = dimension * total_distortion / total_mass
    return {
        "dimension": dimension,
        "bits": bits,
        "centroids": centroid_values,
        "decision_boundaries": boundary_values,
        "normalized_mse": float(normalized_mse),
        "iterations": iteration,
        "centroid_checksum": centroid_checksum(centroid_values),
    }


def build_payload(
    dimensions: Iterable[int],
    bit_widths: Iterable[int],
    *,
    quadrature_order: int,
    tolerance: float,
    max_iterations: int,
) -> Dict[str, object]:
    entries: Dict[str, object] = {}
    for dimension in sorted(set(dimensions)):
        for bits in sorted(set(bit_widths)):
            key = f"{dimension}:{bits}"
            entries[key] = generate_codebook(
                dimension,
                bits,
                quadrature_order=quadrature_order,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

    return {
        "format_version": FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "distribution": "sphere_coordinate_beta",
        "quadrature_order": quadrature_order,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "entries": entries,
    }


def _parse_int_list(raw: str) -> Tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        default=",".join(str(value) for value in DEFAULT_DIMENSIONS),
    )
    parser.add_argument(
        "--bits",
        default=",".join(str(value) for value in DEFAULT_BITS),
    )
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-13)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="xKV/customized_cache/assets/turboquant_mse_codebooks.json",
        help="Output JSON path, or '-' for stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        _parse_int_list(args.dimensions),
        _parse_int_list(args.bits),
        quadrature_order=args.quadrature_order,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    rendered = json.dumps(
        payload,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=False,
    ) + "\n"

    if args.output == "-":
        print(rendered, end="")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)


if __name__ == "__main__":
    main()
