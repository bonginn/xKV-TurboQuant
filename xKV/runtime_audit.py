"""Opt-in, fail-closed runtime evidence for the xKV quantization path.

Set ``XKV_RUNTIME_AUDIT=1`` to enable the audit.  The normal inference path
does not allocate counters, request generation scores, or synchronize tensors
for finite checks when the environment variable is disabled.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def runtime_audit_enabled() -> bool:
    """Return whether the opt-in runtime audit is enabled."""
    raw_value = os.environ.get("XKV_RUNTIME_AUDIT", "").strip().lower()
    if raw_value in _TRUE_VALUES:
        return True
    if raw_value in _FALSE_VALUES:
        return False
    raise ValueError(
        "XKV_RUNTIME_AUDIT must be one of 1/true/yes/on or "
        f"0/false/no/off; got {raw_value!r}"
    )


def _expected_pair_signatures(config) -> list[Dict[str, Any]]:
    if config.kv_quantizer != "turboquant_mse":
        return []

    counts: Dict[tuple[str, int, int], int] = {}
    for group in config.layer_groups:
        for side, enabled in (
            ("k", config.merge_key),
            ("v", config.merge_value),
        ):
            if not enabled:
                continue
            rank = group.rank_k if side == "k" else group.rank_v
            key = (side, int(rank), int(config.quant_bits_for(side)))
            counts[key] = counts.get(key, 0) + 1

    return [
        {
            "side": side,
            "rank": rank,
            "bits": bits,
            "rotation_seed": int(config.turboquant_rotation_seed),
            "chunk_size": config.turboquant_chunk_size,
            "calls_per_cache": calls,
        }
        for (side, rank, bits), calls in sorted(counts.items())
    ]


def create_runtime_audit(
    config,
    *,
    model_num_layers: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Create shared audit state, or ``None`` when auditing is disabled."""
    if not runtime_audit_enabled():
        return None

    configured_num_layers = getattr(config, "num_layers", None)
    if (
        configured_num_layers is not None
        and model_num_layers is not None
        and int(configured_num_layers) != int(model_num_layers)
    ):
        raise ValueError(
            "runtime audit requires config.num_layers to match the model: "
            f"config={configured_num_layers}, model={model_num_layers}"
        )
    expected_num_layers = (
        model_num_layers
        if model_num_layers is not None
        else configured_num_layers
    )
    if expected_num_layers is None:
        raise ValueError(
            "runtime audit requires config.num_layers or model_num_layers"
        )
    audit: Dict[str, Any] = {
        "enabled": True,
        "kv_quantizer": str(config.kv_quantizer),
        "expected_groups_per_cache": len(config.layer_groups),
        "expected_num_layers": int(expected_num_layers),
        "expected_pair_signatures": _expected_pair_signatures(config),
    }
    reset_runtime_audit(audit)
    return audit


def reset_runtime_audit(audit: Dict[str, Any]) -> None:
    """Reset per-evaluation counters while retaining expected configuration."""
    audit.update(
        {
            "cache_created": 0,
            "cache_layouts": [],
            "group_merges": 0,
            "k_pair_calls": 0,
            "v_pair_calls": 0,
            "pair_call_signatures": [],
            "decode_updates": 0,
            "generation_calls": 0,
            "generation_token_counts": [],
            "generated_tokens": 0,
            "expected_decode_updates": 0,
            "score_tensors_checked": 0,
            "reconstruction_tensors_checked": 0,
            "nonfinite_cache": 0,
            "invalid_generation_scores": 0,
        }
    )


def record_cache_created(audit: Optional[Dict[str, Any]], layout: str) -> None:
    if audit is None:
        return
    audit["cache_created"] += 1
    if layout not in audit["cache_layouts"]:
        audit["cache_layouts"].append(layout)


def record_group_merge(audit: Optional[Dict[str, Any]]) -> None:
    if audit is not None:
        audit["group_merges"] += 1


def record_decode_update(audit: Optional[Dict[str, Any]]) -> None:
    if audit is not None:
        audit["decode_updates"] += 1


def record_pair_call(
    audit: Optional[Dict[str, Any]],
    side: str,
    codec,
    factor_pair,
) -> None:
    """Record a pair call only after ``fake_quantize_pair`` has succeeded."""
    if audit is None:
        return
    if side not in ("k", "v"):
        raise ValueError("runtime-audit pair side must be 'k' or 'v'")

    audit[f"{side}_pair_calls"] += 1
    signature = {
        "side": side,
        "rank": int(codec.dimension),
        "bits": int(codec.bits),
        "device": str(factor_pair.codes.indices.device),
        "indices_dtype": str(factor_pair.codes.indices.dtype),
        "norms_dtype": str(factor_pair.codes.norms.dtype),
        "rotation_seed": int(codec.rotation_seed),
        "chunk_size": codec.chunk_size,
        "codebook_checksum": str(codec.codebook_checksum),
        "rotation_checksum": str(codec.rotation_checksum),
    }
    for existing in audit["pair_call_signatures"]:
        if all(existing[key] == value for key, value in signature.items()):
            existing["calls"] += 1
            break
    else:
        signature["calls"] = 1
        audit["pair_call_signatures"].append(signature)


def record_reconstruction_check(
    audit: Optional[Dict[str, Any]],
    *,
    is_finite: bool,
) -> None:
    if audit is None:
        return
    audit["reconstruction_tensors_checked"] += 1
    if not is_finite:
        audit["nonfinite_cache"] += 1


def record_hf_generation(
    audit: Optional[Dict[str, Any]],
    *,
    generated_tokens: int,
    score_tensors_checked: int,
    invalid_score_tensors: int,
) -> None:
    if audit is None:
        return
    audit["generation_calls"] += 1
    audit["generation_token_counts"].append(int(generated_tokens))
    audit["generated_tokens"] += int(generated_tokens)
    audit["score_tensors_checked"] += int(score_tensors_checked)
    audit["invalid_generation_scores"] += int(
        invalid_score_tensors
    )
    audit["expected_decode_updates"] += audit["expected_num_layers"] * max(
        int(generated_tokens) - 1,
        0,
    )


def _actual_pair_counts(audit: Dict[str, Any]) -> Dict[tuple[Any, ...], int]:
    counts: Dict[tuple[Any, ...], int] = {}
    for signature in audit["pair_call_signatures"]:
        key = (
            signature["side"],
            int(signature["rank"]),
            int(signature["bits"]),
            int(signature["rotation_seed"]),
            signature["chunk_size"],
        )
        counts[key] = counts.get(key, 0) + int(signature["calls"])
    return counts


def summarize_runtime_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Validate actual execution against config-derived expectations."""
    errors = []
    cache_created = int(audit["cache_created"])
    expected_group_merges = (
        cache_created * int(audit["expected_groups_per_cache"])
    )
    expected_pair_counts = {
        (
            signature["side"],
            int(signature["rank"]),
            int(signature["bits"]),
            int(signature["rotation_seed"]),
            signature["chunk_size"],
        ): cache_created * int(signature["calls_per_cache"])
        for signature in audit["expected_pair_signatures"]
    }
    actual_pair_counts = _actual_pair_counts(audit)
    generation_calls = int(audit["generation_calls"])

    if audit["kv_quantizer"] == "turboquant_mse":
        if cache_created == 0:
            errors.append("no xKV cache was created")
        if audit["group_merges"] != expected_group_merges:
            errors.append(
                "group merge count mismatch: "
                f"expected {expected_group_merges}, got {audit['group_merges']}"
            )
        if actual_pair_counts != expected_pair_counts:
            errors.append(
                "TurboQuant pair-call signatures/counts do not match config: "
                f"expected {expected_pair_counts}, got {actual_pair_counts}"
            )
        if audit["reconstruction_tensors_checked"] != sum(
            actual_pair_counts.values()
        ):
            errors.append(
                "not every TurboQuant reconstruction received a finite check"
            )
        if generation_calls == 0:
            errors.append("no audited HF generation call was observed")
        elif any(
            token_count < 2
            for token_count in audit["generation_token_counts"]
        ):
            errors.append(
                "generation produced fewer than two tokens, so the compressed "
                "cache decode path was not exercised"
            )

    if audit["nonfinite_cache"]:
        errors.append(
            f"found {audit['nonfinite_cache']} non-finite cache reconstruction(s)"
        )
    if audit["invalid_generation_scores"]:
        errors.append(
            f"found {audit['invalid_generation_scores']} invalid "
            "generation score tensor(s)"
        )

    if generation_calls:
        if generation_calls != cache_created:
            errors.append(
                "HF generation/cache count mismatch: "
                f"{generation_calls} generation(s), {cache_created} cache(s)"
            )
        if audit["score_tensors_checked"] != audit["generated_tokens"]:
            errors.append(
                "generation score count mismatch: "
                f"checked {audit['score_tensors_checked']}, generated "
                f"{audit['generated_tokens']} token(s)"
            )
        if audit["decode_updates"] != audit["expected_decode_updates"]:
            errors.append(
                "decode update count mismatch: "
                f"expected {audit['expected_decode_updates']}, got "
                f"{audit['decode_updates']}"
            )

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "turboquant_executed": (
            audit["kv_quantizer"] == "turboquant_mse"
            and bool(actual_pair_counts)
            and actual_pair_counts == expected_pair_counts
        ),
        "kv_quantizer": audit["kv_quantizer"],
        "cache_created": cache_created,
        "cache_layouts": list(audit["cache_layouts"]),
        "group_merges": int(audit["group_merges"]),
        "expected_group_merges": expected_group_merges,
        "k_pair_calls": int(audit["k_pair_calls"]),
        "v_pair_calls": int(audit["v_pair_calls"]),
        "expected_pair_signatures": list(
            audit["expected_pair_signatures"]
        ),
        "pair_call_signatures": list(audit["pair_call_signatures"]),
        "reconstruction_tensors_checked": int(
            audit["reconstruction_tensors_checked"]
        ),
        "generation_calls": generation_calls,
        "generation_token_counts": list(audit["generation_token_counts"]),
        "generated_tokens": int(audit["generated_tokens"]),
        "decode_updates": int(audit["decode_updates"]),
        "expected_decode_updates": int(audit["expected_decode_updates"]),
        "score_tensors_checked": int(audit["score_tensors_checked"]),
        "nonfinite_cache": int(audit["nonfinite_cache"]),
        "invalid_generation_scores": int(
            audit["invalid_generation_scores"]
        ),
        "errors": errors,
    }
    return summary
