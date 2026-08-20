"""Accuracy-first factor-pair quantization for the xKV-SR reference cache.

This module intentionally implements fake quantization only. TurboQuant's
logical indices are decoded immediately into FP32 reference factors; the cache
caller chooses the persistent model dtype. It is suitable for accuracy
experiments, not memory or speed claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from xKV.customized_cache.turboquant import (
    TurboQuantFactorPair,
    TurboQuantMSE,
    load_codebook,
)


class TurboQuantFactorQuantizer:
    """Shared TurboQuant codecs for one xKV-SR K/V factor configuration."""

    def __init__(
        self,
        *,
        rank_k: int,
        rank_v: int,
        k_bits: int,
        v_bits: int,
        rotation_seed: int = 42,
        chunk_size: Optional[int] = 4096,
        codebook_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.rank_k = self._validate_rank(rank_k, "rank_k")
        self.rank_v = self._validate_rank(rank_v, "rank_v")
        self.k_bits = self._validate_bits(k_bits, "k_bits")
        self.v_bits = self._validate_bits(v_bits, "v_bits")
        if not isinstance(rotation_seed, int) or isinstance(rotation_seed, bool):
            raise TypeError("rotation_seed must be an integer")
        if chunk_size is not None and (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be None or a positive integer")

        self.rotation_seed = rotation_seed
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.codebook_path = (
            None
            if codebook_path is None
            else str(Path(codebook_path).expanduser().resolve())
        )
        self._registry = {}

        if self.k_bits == 16 and self.v_bits == 16:
            raise ValueError("at least one of k_bits or v_bits must quantize")

        # Build only enabled sides. A value of 16 is an explicit model-dtype
        # bypass used for single-side diagnostic ablations.
        self._codecs = {}
        for side, dimension, bits in (
            ("k", self.rank_k, self.k_bits),
            ("v", self.rank_v, self.v_bits),
        ):
            if bits == 16:
                continue
            # Fail before a long SVD prefill if an offline table is unavailable.
            load_codebook(dimension, bits, self.codebook_path)
            self._codecs[side] = self._get_codec(dimension, bits)

    @staticmethod
    def _validate_rank(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 2:
            raise ValueError(f"{name} must be at least 2")
        return value

    @staticmethod
    def _validate_bits(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value not in (1, 2, 3, 4, 16):
            raise ValueError(f"{name} must be one of 1, 2, 3, 4, or 16")
        return value

    def is_enabled(self, side: str) -> bool:
        """Return whether ``side`` uses TurboQuant instead of model dtype."""

        self.bits_for(side)  # validate side even when no codec was built
        return side in self._codecs

    def bits_for(self, side: str) -> int:
        if side == "k":
            return self.k_bits
        if side == "v":
            return self.v_bits
        raise ValueError("side must be 'k' or 'v'")

    def rank_for(self, side: str) -> int:
        if side == "k":
            return self.rank_k
        if side == "v":
            return self.rank_v
        raise ValueError("side must be 'k' or 'v'")

    def _get_codec(self, rank: int, bits: int) -> TurboQuantMSE:
        registry_key = (
            rank,
            bits,
            self.rotation_seed,
            self.chunk_size,
            self.codebook_path,
            str(self.device),
        )
        codec = self._registry.get(registry_key)
        if codec is None:
            codec = TurboQuantMSE(
                dimension=rank,
                bits=bits,
                rotation_seed=self.rotation_seed,
                chunk_size=self.chunk_size,
                codebook_path=self.codebook_path,
                norm_dtype=torch.float32,
                compute_dtype=torch.float32,
            ).to(device=self.device)
            codec.eval()
            self._registry[registry_key] = codec
        return codec

    def codec_for(self, side: str) -> TurboQuantMSE:
        self.bits_for(side)
        if side not in self._codecs:
            raise ValueError(f"{side.upper()} factor quantization is bypassed")
        return self._codecs[side]

    @property
    def registry_size(self) -> int:
        return len(self._registry)

    @torch.no_grad()
    def transform_pair(
        self,
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        *,
        side: str,
    ) -> TurboQuantFactorPair:
        """Quantize A and fold the inverse rotation into B for ``A @ B``."""

        if not self.is_enabled(side):
            raise ValueError(f"{side.upper()} factor quantization is bypassed")

        expected_rank = self.rank_for(side)
        if left_factor.shape[-1] != expected_rank:
            raise ValueError(
                f"{side.upper()} left-factor width must be {expected_rank}, "
                f"got {left_factor.shape[-1]}"
            )
        if right_factor.ndim < 2 or right_factor.shape[-2] != expected_rank:
            raise ValueError(
                f"{side.upper()} right-factor rank must be {expected_rank}, "
                f"got shape {tuple(right_factor.shape)}"
            )
        if left_factor.device != self.device or right_factor.device != self.device:
            raise ValueError(
                "factor-pair device does not match the configured TurboQuant "
                f"device {self.device}"
            )
        return self.codec_for(side).fake_quantize_pair(
            left_factor,
            right_factor,
        )
