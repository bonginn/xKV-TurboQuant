"""Reference implementation of the TurboQuant-MSE vector codec.

This module implements the accuracy-first codec described in Algorithm 1 of
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(arXiv:2504.19874v1).  It does not implement a fused inference kernel.

Physical 4-bit nibble packing (pack_turboquant_codes / unpack_turboquant_codes /
PackedTurboQuantCodes, below) stores two 4-bit codes per byte.
`TurboQuantCodes.indices` (produced by `encode()`) remains logical (one uint8
byte per code) throughout the rest of this module; callers that want the
physically packed form call `pack_turboquant_codes()` explicitly.
`PackedXKVCache` (scripts/factorized_xkv_cache.py) stores this packed form.

The Lloyd-Max codebooks are generated offline for the exact finite-dimensional
sphere-coordinate distribution and loaded from a checked-in JSON asset at
runtime.  No calibration samples are used.

The paper uses column vectors::

    y = Pi @ x
    x_hat = Pi.T @ y_hat

xKV stores token vectors as rows, so the corresponding operations are::

    y = x @ Pi.T
    x_hat = y_hat @ Pi

For an xKV factorization X ~= A @ B, the inverse rotation can be folded into
the small right factor exactly::

    (norms * y_hat @ Pi) @ B == (norms * y_hat) @ (Pi @ B)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn


_CODEBOOK_FORMAT_VERSION = 1
_DEFAULT_CODEBOOK_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "turboquant_mse_codebooks.json"
)
_SUPPORTED_INDEX_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _canonical_float_list(values: Sequence[float]) -> str:
    """Return the canonical JSON representation used for asset checksums."""

    return json.dumps(
        [float(value) for value in values],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def centroid_checksum(values: Sequence[float]) -> str:
    """SHA-256 checksum for a sequence of Lloyd-Max centroids."""

    payload = _canonical_float_list(values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    """SHA-256 checksum of a tensor's contiguous CPU byte representation."""

    value = tensor.detach().cpu().contiguous()
    payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _autocast_disabled(device: torch.device):
    """Keep the reference path in its configured FP32/FP64 compute dtype."""

    return torch.autocast(device_type=device.type, enabled=False)


@dataclass(frozen=True)
class TurboQuantCodebook:
    """Validated metadata for one offline Lloyd-Max codebook."""

    dimension: int
    bits: int
    centroids: Tuple[float, ...]
    decision_boundaries: Tuple[float, ...]
    normalized_mse: float
    checksum: str
    generator_version: str


@dataclass
class TurboQuantCodes:
    """Logical TurboQuant codes before physical bit packing.

    ``indices`` stores logical b-bit values in a uint8 tensor.  Its allocated
    PyTorch storage is therefore not evidence of a true b-bit cache yet.
    """

    indices: torch.Tensor
    norms: torch.Tensor
    dimension: int
    bits: int
    source_dtype: torch.dtype
    codebook_checksum: str
    rotation_checksum: str


@dataclass
class TurboQuantFactorPair:
    """Folded factors whose product reconstructs the quantized matrix.

    ``left_factor`` is ``diag(norms) @ y_hat`` and ``right_factor`` is
    ``Pi @ B``.  Consequently ``left_factor @ right_factor`` is exactly the
    folded equivalent of ``decode(codes) @ B`` up to floating-point rounding.
    """

    left_factor: torch.Tensor
    right_factor: torch.Tensor
    codes: TurboQuantCodes


# ── Physical 4-bit nibble packing ─────────────────────────────────────────
#
# `TurboQuantCodes.indices` stores one LOGICAL code per uint8 byte, which is
# not evidence of a physically packed cache (its own docstring says so). The
# functions below convert that logical representation to and from a REAL
# packed form where two 4-bit codes share one byte.

@dataclass
class PackedTurboQuantCodes:
    """Physically packed TurboQuant codes: two 4-bit codes per byte.

    ``packed`` has last dimension ``dimension // 2`` (half of the logical
    ``TurboQuantCodes.indices``' last dimension) and dtype uint8. For
    coordinate pair (2*i, 2*i+1), coordinate 2*i (even) is packed into the
    LOW nibble and coordinate 2*i+1 (odd) into the HIGH nibble of
    ``packed[..., i]``.

    ``norms`` is stored as FP32, independent of ``left_factor``'s dtype.
    """

    packed: torch.Tensor
    norms: torch.Tensor
    dimension: int
    bits: int
    source_dtype: torch.dtype
    codebook_checksum: str
    rotation_checksum: str

    @property
    def storage_bytes(self) -> int:
        """Actual physical bytes for packed codes + norms -- measured from
        real tensor storage, never derived from a logical-bit-count formula."""
        return (
            self.packed.numel() * self.packed.element_size()
            + self.norms.numel() * self.norms.element_size()
        )


def pack_turboquant_codes(codes: TurboQuantCodes) -> PackedTurboQuantCodes:
    """Pack ``codes.indices`` (one logical 4-bit value per uint8 byte) into
    real nibbles (two values per byte). Only supports bits=4 -- nibble
    packing is specifically a 2-codes-per-byte scheme; other bit widths are
    out of scope for the current K4/V4 product spec and would need a
    different packing layout.

    Fails closed (raises) rather than silently padding or truncating when
    bits != 4, the dimension is odd (no implicit padding), indices fall
    outside [0, 15], or the indices dtype is not uint8.
    """
    if codes.bits != 4:
        raise ValueError(
            f"pack_turboquant_codes only supports bits=4 (nibble packing), got bits={codes.bits}"
        )
    if codes.dimension % 2 != 0:
        raise ValueError(
            f"pack_turboquant_codes requires an even dimension (fail-closed, no implicit "
            f"padding), got dimension={codes.dimension}"
        )
    indices = codes.indices
    if indices.dtype != torch.uint8:
        raise TypeError(f"codes.indices must be uint8, got {indices.dtype}")
    if indices.shape[-1] != codes.dimension:
        raise ValueError(
            f"codes.indices last dimension ({indices.shape[-1]}) does not match "
            f"codes.dimension ({codes.dimension})"
        )
    if indices.numel() > 0:
        min_idx = int(indices.min().item())
        max_idx = int(indices.max().item())
        if min_idx < 0 or max_idx > 15:
            raise ValueError(
                f"index out of range for 4-bit packing: [{min_idx}, {max_idx}], "
                f"must be in [0, 15]"
            )

    indices = indices.contiguous()
    even = indices[..., 0::2]
    odd = indices[..., 1::2]
    packed = (even | (odd << 4)).to(torch.uint8).contiguous()

    return PackedTurboQuantCodes(
        packed=packed,
        norms=codes.norms.to(torch.float32),
        dimension=codes.dimension,
        bits=codes.bits,
        source_dtype=codes.source_dtype,
        codebook_checksum=codes.codebook_checksum,
        rotation_checksum=codes.rotation_checksum,
    )


def unpack_turboquant_codes(packed_codes: PackedTurboQuantCodes) -> TurboQuantCodes:
    """Inverse of ``pack_turboquant_codes`` -- reconstructs the logical
    uint8-per-code ``TurboQuantCodes.indices`` tensor from packed nibbles,
    bit-exactly (no precision loss: this is a pure bit-layout transform)."""
    packed = packed_codes.packed
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed_codes.packed must be uint8, got {packed.dtype}")
    expected_packed_dim = packed_codes.dimension // 2
    if packed.shape[-1] != expected_packed_dim:
        raise ValueError(
            f"packed_codes.packed last dimension ({packed.shape[-1]}) does not match "
            f"dimension//2 ({expected_packed_dim})"
        )

    packed = packed.contiguous()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    out_shape = packed.shape[:-1] + (packed_codes.dimension,)
    indices = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    indices[..., 0::2] = low
    indices[..., 1::2] = high

    return TurboQuantCodes(
        indices=indices,
        norms=packed_codes.norms,
        dimension=packed_codes.dimension,
        bits=packed_codes.bits,
        source_dtype=packed_codes.source_dtype,
        codebook_checksum=packed_codes.codebook_checksum,
        rotation_checksum=packed_codes.rotation_checksum,
    )


def validate_packed_codes(packed_codes: PackedTurboQuantCodes, quantizer: "TurboQuantMSE") -> None:
    """Checks ``packed_codes`` was produced by (or is compatible with)
    ``quantizer`` -- same dimension, bits, codebook, and rotation. Raises
    ValueError on any mismatch. This is the dedicated K-vs-V check: K and V
    codecs have different ``dimension`` (384 vs 576) and are constructed
    from different codebook assets, so calling this with the wrong
    quantizer for a given packed_codes object fails here, not silently
    later at reconstruction time."""
    if packed_codes.dimension != quantizer.dimension or packed_codes.bits != quantizer.bits:
        raise ValueError(
            f"packed_codes metadata (dimension={packed_codes.dimension}, bits={packed_codes.bits}) "
            f"does not match quantizer (dimension={quantizer.dimension}, bits={quantizer.bits})"
        )
    if packed_codes.codebook_checksum != quantizer.codebook_checksum:
        raise ValueError("packed_codes were produced with a different codebook")
    if packed_codes.rotation_checksum != quantizer.rotation_checksum:
        raise ValueError("packed_codes were produced with a different rotation")


def build_haar_rotation(
    dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a deterministic Haar-distributed orthogonal matrix on CPU.

    A local CPU generator is used so construction does not modify PyTorch's
    global RNG state.  QR signs are canonicalized by multiplying each column
    of Q by ``sign(diag(R))``; zero diagonal entries use a positive sign.
    """

    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("rotation dtype must be torch.float32 or torch.float64")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(
        dimension,
        dimension,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    q_matrix, r_matrix = torch.linalg.qr(gaussian, mode="reduced")
    diagonal = torch.diagonal(r_matrix)
    signs = torch.where(
        diagonal < 0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    q_matrix = q_matrix * signs.unsqueeze(0)
    return q_matrix.to(dtype=dtype).contiguous()


def _validate_centroids(
    centroids: Sequence[float],
    *,
    dimension: int,
    bits: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    levels = 1 << bits
    values = tuple(float(value) for value in centroids)
    if len(values) != levels:
        raise ValueError(
            f"codebook ({dimension}, {bits}) has {len(values)} centroids; "
            f"expected {levels}"
        )

    centroid_tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(centroid_tensor).all().item()):
        raise ValueError("codebook centroids must all be finite")
    if bool((centroid_tensor < -1.0).any().item()) or bool(
        (centroid_tensor > 1.0).any().item()
    ):
        raise ValueError("codebook centroids must lie in [-1, 1]")
    if not bool((centroid_tensor[1:] > centroid_tensor[:-1]).all().item()):
        raise ValueError("codebook centroids must be strictly increasing")

    symmetry_error = torch.max(
        torch.abs(centroid_tensor + torch.flip(centroid_tensor, dims=(0,)))
    ).item()
    if symmetry_error > 1e-12:
        raise ValueError(
            f"codebook centroids are not symmetric (max error {symmetry_error})"
        )

    boundaries = 0.5 * (centroid_tensor[:-1] + centroid_tensor[1:])
    return values, tuple(float(value) for value in boundaries.tolist())


@lru_cache(maxsize=None)
def _load_codebook_from_path(
    path_string: str,
    dimension: int,
    bits: int,
) -> TurboQuantCodebook:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"TurboQuant codebook asset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("TurboQuant codebook asset root must be a JSON object")

    if payload.get("format_version") != _CODEBOOK_FORMAT_VERSION:
        raise ValueError(
            "unsupported TurboQuant codebook format version: "
            f"{payload.get('format_version')!r}"
        )
    if payload.get("distribution") != "sphere_coordinate_beta":
        raise ValueError("unexpected TurboQuant codebook distribution")

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("TurboQuant codebook asset must contain an entries object")

    key = f"{dimension}:{bits}"
    entry = entries.get(key)
    if entry is None:
        raise ValueError(
            f"offline TurboQuant codebook ({dimension}, {bits}) is missing "
            f"from {path}; generate it explicitly instead of falling back"
        )
    if not isinstance(entry, dict):
        raise ValueError(f"codebook {key} must be a JSON object")
    if int(entry.get("dimension", -1)) != dimension:
        raise ValueError(f"codebook {key} has inconsistent dimension metadata")
    if int(entry.get("bits", -1)) != bits:
        raise ValueError(f"codebook {key} has inconsistent bit metadata")

    centroids, computed_boundaries = _validate_centroids(
        entry.get("centroids", ()),
        dimension=dimension,
        bits=bits,
    )
    expected_checksum = str(entry.get("centroid_checksum", ""))
    actual_checksum = centroid_checksum(centroids)
    if expected_checksum != actual_checksum:
        raise ValueError(
            f"codebook {key} checksum mismatch: expected {expected_checksum}, "
            f"computed {actual_checksum}"
        )

    raw_boundaries = entry.get("decision_boundaries")
    if not isinstance(raw_boundaries, list):
        raise ValueError(f"codebook {key} must contain decision_boundaries")
    stored_boundaries = tuple(float(value) for value in raw_boundaries)
    if len(stored_boundaries) != len(computed_boundaries):
        raise ValueError(f"codebook {key} has invalid boundary count")
    if not all(math.isfinite(value) for value in stored_boundaries):
        raise ValueError(f"codebook {key} has non-finite decision boundaries")
    boundary_error = max(
        abs(stored - computed)
        for stored, computed in zip(stored_boundaries, computed_boundaries)
    )
    if boundary_error > 1e-15:
        raise ValueError(
            f"codebook {key} boundaries are not centroid midpoints"
        )

    normalized_mse = float(entry.get("normalized_mse", float("nan")))
    if (
        not torch.isfinite(torch.tensor(normalized_mse)).item()
        or normalized_mse < 0.0
        or normalized_mse > 1.0
    ):
        raise ValueError(f"codebook {key} has invalid normalized_mse")

    return TurboQuantCodebook(
        dimension=dimension,
        bits=bits,
        centroids=centroids,
        # Runtime lookup always derives boundaries from checksum-verified
        # centroids.  The duplicated JSON field is validation/audit metadata.
        decision_boundaries=computed_boundaries,
        normalized_mse=normalized_mse,
        checksum=actual_checksum,
        generator_version=str(payload.get("generator_version", "unknown")),
    )


def load_codebook(
    dimension: int,
    bits: int,
    path: Optional[Union[str, Path]] = None,
) -> TurboQuantCodebook:
    """Load and validate one offline finite-d Lloyd-Max codebook."""

    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise TypeError("bits must be an integer")
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")

    resolved_path = Path(path) if path is not None else _DEFAULT_CODEBOOK_PATH
    return _load_codebook_from_path(
        str(resolved_path.resolve()),
        dimension,
        bits,
    )


class TurboQuantMSE(nn.Module):
    """Accuracy-first TurboQuant-MSE codec for vectors in the last dimension."""

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        rotation_seed: int = 42,
        codebook_path: Optional[Union[str, Path]] = None,
        codebook: Optional[Sequence[float]] = None,
        rotation: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = 4096,
        norm_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float32,
        validate_rotation: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an integer")
        if dimension < 2:
            raise ValueError("dimension must be at least 2")
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise TypeError("bits must be an integer")
        if not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8")
        if chunk_size is not None and (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be None or a positive integer")
        if norm_dtype not in (torch.float16, torch.float32):
            raise ValueError("norm_dtype must be torch.float16 or torch.float32")
        if compute_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "compute_dtype must be torch.float32 or torch.float64"
            )
        if codebook is not None and codebook_path is not None:
            raise ValueError("provide codebook or codebook_path, not both")
        if not isinstance(rotation_seed, int) or isinstance(rotation_seed, bool):
            raise TypeError("rotation_seed must be an integer")

        self.dimension = dimension
        self.bits = bits
        self.rotation_seed = rotation_seed
        self.chunk_size = chunk_size
        self.norm_dtype = norm_dtype
        self.compute_dtype = compute_dtype
        self.validate_rotation = validate_rotation

        if codebook is None:
            spec = load_codebook(dimension, bits, codebook_path)
            centroid_values = spec.centroids
            boundary_values = spec.decision_boundaries
            self.codebook_asset_checksum = spec.checksum
            self.codebook_normalized_mse = spec.normalized_mse
            self.codebook_generator_version = spec.generator_version
        else:
            centroid_values, boundary_values = _validate_centroids(
                codebook,
                dimension=dimension,
                bits=bits,
            )
            self.codebook_asset_checksum = centroid_checksum(centroid_values)
            self.codebook_normalized_mse = float("nan")
            self.codebook_generator_version = "user-provided"

        centroid_tensor = torch.tensor(
            centroid_values,
            dtype=compute_dtype,
            device="cpu",
        )
        boundary_tensor = torch.tensor(
            boundary_values,
            dtype=compute_dtype,
            device="cpu",
        )
        self.register_buffer("centroids", centroid_tensor, persistent=True)
        self.register_buffer("decision_boundaries", boundary_tensor, persistent=True)
        self.codebook_checksum = tensor_checksum(centroid_tensor)

        if rotation is None:
            rotation_tensor = build_haar_rotation(
                dimension,
                rotation_seed,
                dtype=compute_dtype,
            )
        else:
            if not isinstance(rotation, torch.Tensor):
                raise TypeError("rotation must be a torch.Tensor")
            if rotation.shape != (dimension, dimension):
                raise ValueError(
                    f"rotation must have shape {(dimension, dimension)}, "
                    f"got {tuple(rotation.shape)}"
                )
            if rotation.dtype not in (torch.float32, torch.float64):
                raise ValueError("rotation must use float32 or float64")
            if not bool(torch.isfinite(rotation).all().item()):
                raise ValueError("rotation must contain only finite values")
            rotation_tensor = rotation.detach().to(
                device="cpu",
                dtype=compute_dtype,
            ).clone().contiguous()

        if validate_rotation:
            identity = torch.eye(
                dimension,
                dtype=compute_dtype,
                device=rotation_tensor.device,
            )
            with _autocast_disabled(rotation_tensor.device):
                gram = rotation_tensor.transpose(0, 1) @ rotation_tensor
            tolerance = 1e-10 if compute_dtype == torch.float64 else 5e-5
            max_error = torch.max(torch.abs(gram - identity)).item()
            if max_error > tolerance:
                raise ValueError(
                    "rotation is not sufficiently orthogonal: "
                    f"max |Pi.T @ Pi - I| = {max_error}"
                )

        self.register_buffer("rotation", rotation_tensor, persistent=True)
        self.rotation_checksum = tensor_checksum(rotation_tensor)
        self.register_load_state_dict_post_hook(self._refresh_after_state_load)

    def _refresh_after_state_load(self, module, incompatible_keys) -> None:
        """Revalidate static buffers and refresh codec IDs after state loading."""

        del module, incompatible_keys
        try:
            _validate_centroids(
                self.centroids.detach().cpu().tolist(),
                dimension=self.dimension,
                bits=self.bits,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"loaded TurboQuant centroids are invalid: {error}"
            ) from error
        computed_boundaries = 0.5 * (
            self.centroids[:-1] + self.centroids[1:]
        )
        tolerance = 1e-12 if self.compute_dtype == torch.float64 else 1e-6
        if not torch.allclose(
            self.decision_boundaries,
            computed_boundaries,
            atol=tolerance,
            rtol=0,
        ):
            raise RuntimeError(
                "loaded TurboQuant decision boundaries do not match centroids"
            )
        if not bool(torch.isfinite(self.rotation).all().item()):
            raise RuntimeError("loaded TurboQuant rotation contains non-finite values")
        if self.validate_rotation:
            identity = torch.eye(
                self.dimension,
                dtype=self.compute_dtype,
                device=self.rotation.device,
            )
            with _autocast_disabled(self.rotation.device):
                gram = self.rotation.transpose(0, 1) @ self.rotation
            rotation_tolerance = (
                1e-10 if self.compute_dtype == torch.float64 else 5e-5
            )
            max_error = torch.max(torch.abs(gram - identity)).item()
            if max_error > rotation_tolerance:
                raise RuntimeError(
                    "loaded TurboQuant rotation is not sufficiently orthogonal: "
                    f"max error {max_error}"
                )
        self.codebook_checksum = tensor_checksum(self.centroids)
        self.rotation_checksum = tensor_checksum(self.rotation)

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def _check_module_buffers(self, device: torch.device) -> None:
        if self.rotation.device != device:
            raise RuntimeError(
                f"TurboQuant buffers are on {self.rotation.device}, but input is "
                f"on {device}; call quantizer.to(input.device) first"
            )
        if self.centroids.device != device:
            raise RuntimeError("TurboQuant codebook and rotation devices differ")
        if self.decision_boundaries.device != device:
            raise RuntimeError("TurboQuant boundaries and rotation devices differ")
        if self.rotation.dtype != self.compute_dtype:
            raise RuntimeError(
                "TurboQuant rotation dtype changed; move only the device and keep "
                f"the configured compute dtype {self.compute_dtype}"
            )
        if self.centroids.dtype != self.compute_dtype:
            raise RuntimeError("TurboQuant codebook dtype changed unexpectedly")
        if self.decision_boundaries.dtype != self.compute_dtype:
            raise RuntimeError("TurboQuant boundary dtype changed unexpectedly")

    def _validate_input(self, value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension")
        if value.shape[-1] != self.dimension:
            raise ValueError(
                f"{name} last dimension must be {self.dimension}, "
                f"got {value.shape[-1]}"
            )
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point")
        if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                f"{name} dtype must be float16, bfloat16, float32, or float64"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
        self._check_module_buffers(value.device)

    def _chunk_slices(self, row_count: int):
        step = row_count if self.chunk_size is None else self.chunk_size
        if row_count == 0:
            return
        for start in range(0, row_count, step):
            yield slice(start, min(start + step, row_count))

    @torch.no_grad()
    def encode(self, value: torch.Tensor) -> TurboQuantCodes:
        """Encode vectors independently along the final dimension."""

        self._validate_input(value, "value")
        flat_value = value.reshape(-1, self.dimension).to(self.compute_dtype)
        row_count = flat_value.shape[0]

        with _autocast_disabled(value.device):
            norms = torch.linalg.vector_norm(flat_value, dim=-1, keepdim=True)
            safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            unit_vectors = flat_value / safe_norms

        flat_indices = torch.empty(
            (row_count, self.dimension),
            dtype=torch.uint8,
            device=value.device,
        )
        for row_slice in self._chunk_slices(row_count):
            with _autocast_disabled(value.device):
                rotated = (
                    unit_vectors[row_slice]
                    @ self.rotation.transpose(0, 1)
                )
            indices = torch.bucketize(
                rotated.contiguous(),
                self.decision_boundaries,
                right=False,
            )
            flat_indices[row_slice] = indices.to(torch.uint8)

        index_shape = tuple(value.shape)
        norm_shape = tuple(value.shape[:-1]) + (1,)
        stored_norms = norms.to(self.norm_dtype)
        if not bool(torch.isfinite(stored_norms).all().item()):
            raise OverflowError(
                "norm overflow in configured norm_dtype; use norm_dtype=torch.float32"
            )

        return TurboQuantCodes(
            indices=flat_indices.reshape(index_shape),
            norms=stored_norms.reshape(norm_shape),
            dimension=self.dimension,
            bits=self.bits,
            source_dtype=value.dtype,
            codebook_checksum=self.codebook_checksum,
            rotation_checksum=self.rotation_checksum,
        )

    def _validate_codes(self, codes: TurboQuantCodes) -> None:
        if not isinstance(codes, TurboQuantCodes):
            raise TypeError("codes must be a TurboQuantCodes instance")
        if codes.dimension != self.dimension or codes.bits != self.bits:
            raise ValueError(
                "codes metadata does not match this TurboQuantMSE instance"
            )
        if codes.codebook_checksum != self.codebook_checksum:
            raise ValueError("codes were produced with a different codebook")
        if codes.rotation_checksum != self.rotation_checksum:
            raise ValueError("codes were produced with a different rotation")
        if codes.source_dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise ValueError("codes.source_dtype must be a supported floating dtype")
        if codes.indices.ndim < 1 or codes.indices.shape[-1] != self.dimension:
            raise ValueError("codes.indices has an invalid shape")
        expected_norm_shape = tuple(codes.indices.shape[:-1]) + (1,)
        if tuple(codes.norms.shape) != expected_norm_shape:
            raise ValueError(
                f"codes.norms must have shape {expected_norm_shape}, "
                f"got {tuple(codes.norms.shape)}"
            )
        if codes.indices.dtype not in _SUPPORTED_INDEX_DTYPES:
            raise TypeError("codes.indices must use an integer dtype")
        if not torch.is_floating_point(codes.norms):
            raise TypeError("codes.norms must be floating point")
        if codes.indices.device != codes.norms.device:
            raise ValueError("codes.indices and codes.norms must share a device")
        self._check_module_buffers(codes.indices.device)
        if not bool(torch.isfinite(codes.norms).all().item()):
            raise ValueError("codes.norms must contain only finite values")
        if bool((codes.norms < 0).any().item()):
            raise ValueError("codes.norms must be non-negative")

        if codes.indices.numel() > 0:
            minimum = int(codes.indices.min().item())
            maximum = int(codes.indices.max().item())
            if minimum < 0 or maximum >= self.levels:
                raise ValueError(
                    f"code indices must lie in [0, {self.levels - 1}], "
                    f"got [{minimum}, {maximum}]"
                )

    @torch.no_grad()
    def dequantize_rotated(
        self,
        codes: TurboQuantCodes,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return ``diag(norms) @ y_hat`` without inverse rotation."""

        self._validate_codes(codes)
        dtype = codes.source_dtype if output_dtype is None else output_dtype
        if dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise ValueError("output_dtype must be a floating-point dtype")

        flat_indices = codes.indices.reshape(-1, self.dimension).to(torch.long)
        flat_norms = codes.norms.reshape(-1, 1).to(self.compute_dtype)
        flat_output = torch.empty(
            flat_indices.shape,
            dtype=self.compute_dtype,
            device=flat_indices.device,
        )
        for row_slice in self._chunk_slices(flat_indices.shape[0]):
            reconstructed = self.centroids[flat_indices[row_slice]]
            flat_output[row_slice] = reconstructed * flat_norms[row_slice]

        return flat_output.reshape(codes.indices.shape).to(dtype)

    @torch.no_grad()
    def decode(
        self,
        codes: TurboQuantCodes,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decode logical indices and norms back to the original coordinates."""

        rotated = self.dequantize_rotated(
            codes,
            output_dtype=self.compute_dtype,
        )
        dtype = codes.source_dtype if output_dtype is None else output_dtype
        if dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise ValueError("output_dtype must be a floating-point dtype")
        flat_rotated = rotated.reshape(-1, self.dimension)
        flat_output = torch.empty_like(flat_rotated)
        for row_slice in self._chunk_slices(flat_rotated.shape[0]):
            with _autocast_disabled(flat_rotated.device):
                flat_output[row_slice] = (
                    flat_rotated[row_slice] @ self.rotation
                )
        return flat_output.reshape(codes.indices.shape).to(dtype)

    @torch.no_grad()
    def fold_decoder(
        self,
        right_factor: torch.Tensor,
        *,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return ``Pi @ right_factor`` for 2-D or batched right factors."""

        if not isinstance(right_factor, torch.Tensor):
            raise TypeError("right_factor must be a torch.Tensor")
        if right_factor.ndim < 2:
            raise ValueError("right_factor must have at least two dimensions")
        if right_factor.shape[-2] != self.dimension:
            raise ValueError(
                f"right_factor second-to-last dimension must be {self.dimension}, "
                f"got {right_factor.shape[-2]}"
            )
        if not torch.is_floating_point(right_factor):
            raise TypeError("right_factor must be floating point")
        if not bool(torch.isfinite(right_factor).all().item()):
            raise ValueError("right_factor must contain only finite values")
        self._check_module_buffers(right_factor.device)

        dtype = right_factor.dtype if output_dtype is None else output_dtype
        if dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise ValueError("output_dtype must be a floating-point dtype")
        with _autocast_disabled(right_factor.device):
            folded = torch.matmul(
                self.rotation,
                right_factor.to(self.compute_dtype),
            )
        return folded.to(dtype)

    @torch.no_grad()
    def fake_quantize_pair(
        self,
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
    ) -> TurboQuantFactorPair:
        """Encode A and return the exact folded factors for ``A_hat @ B``.

        Both returned factors use ``compute_dtype`` so their product can be
        compared directly with the explicit reference decode without an early
        half-precision cast.
        """

        codes = self.encode(left_factor)
        folded_left = self.dequantize_rotated(
            codes,
            output_dtype=self.compute_dtype,
        )
        folded_right = self.fold_decoder(
            right_factor,
            output_dtype=self.compute_dtype,
        )
        return TurboQuantFactorPair(
            left_factor=folded_left,
            right_factor=folded_right,
            codes=codes,
        )

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Fake-quantize a tensor and reconstruct it in the source dtype."""

        return self.decode(self.encode(value))

    def extra_repr(self) -> str:
        return (
            f"dimension={self.dimension}, bits={self.bits}, "
            f"rotation_seed={self.rotation_seed}, chunk_size={self.chunk_size}, "
            f"norm_dtype={self.norm_dtype}, compute_dtype={self.compute_dtype}, "
            f"codebook_checksum={self.codebook_checksum[:12]}, "
            f"rotation_checksum={self.rotation_checksum[:12]}"
        )
