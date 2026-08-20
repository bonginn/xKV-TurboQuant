"""Tests for opt-in xKV/TurboQuant execution evidence."""

import pytest
import torch

from xKV.configurations import LayerGroup, xKVConfig
from xKV.customized_cache.fake_layer_merge_dynamic_cache import fake_svd
from xKV.customized_cache.turboquant import TurboQuantMSE
from xKV.runtime_audit import (
    create_runtime_audit,
    record_cache_created,
    record_decode_update,
    record_group_merge,
    record_hf_generation,
    summarize_runtime_audit,
)


def _config():
    return xKVConfig(
        num_layers=2,
        layer_merge_impl="svd",
        rank_k=3,
        rank_v=3,
        merge_key=True,
        merge_value=True,
        kv_quantizer="turboquant_mse",
        k_bits=3,
        v_bits=2,
        hadamard=False,
        turboquant_rotation_seed=17,
        turboquant_chunk_size=4,
        layer_groups=[LayerGroup(layers=[0, 1])],
    )


def test_runtime_audit_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("XKV_RUNTIME_AUDIT", raising=False)
    assert create_runtime_audit(_config(), model_num_layers=2) is None


def test_runtime_audit_rejects_ambiguous_environment_value(monkeypatch):
    monkeypatch.setenv("XKV_RUNTIME_AUDIT", "sometimes")
    with pytest.raises(ValueError, match="XKV_RUNTIME_AUDIT must be"):
        create_runtime_audit(_config(), model_num_layers=2)


def test_fake_svd_records_only_completed_turboquant_pair_calls(monkeypatch):
    monkeypatch.setenv("XKV_RUNTIME_AUDIT", "1")
    audit = create_runtime_audit(_config(), model_num_layers=2)
    codec = TurboQuantMSE(3, 3, rotation_seed=17, chunk_size=4)
    tensor = torch.randn(1, 2, 4, 2)

    observed = fake_svd(
        tensor,
        rank=3,
        quantizer=codec,
        runtime_audit=audit,
        audit_side="k",
    )

    assert torch.isfinite(observed).all()
    assert audit["k_pair_calls"] == 1
    assert audit["v_pair_calls"] == 0
    assert audit["reconstruction_tensors_checked"] == 1
    assert audit["nonfinite_cache"] == 0
    signature = audit["pair_call_signatures"][0]
    assert signature["side"] == "k"
    assert signature["rank"] == 3
    assert signature["bits"] == 3
    assert signature["indices_dtype"] == "torch.uint8"
    assert signature["calls"] == 1


def test_runtime_audit_summary_requires_both_configured_sides(monkeypatch):
    monkeypatch.setenv("XKV_RUNTIME_AUDIT", "1")
    audit = create_runtime_audit(_config(), model_num_layers=2)
    record_cache_created(audit, "layers")
    record_group_merge(audit)
    record_hf_generation(
        audit,
        generated_tokens=2,
        score_tensors_checked=2,
        invalid_score_tensors=0,
    )
    record_decode_update(audit)
    record_decode_update(audit)

    key_codec = TurboQuantMSE(3, 3, rotation_seed=17, chunk_size=4)
    value_codec = TurboQuantMSE(3, 2, rotation_seed=17, chunk_size=4)
    tensor = torch.randn(1, 2, 4, 2)
    fake_svd(
        tensor,
        rank=3,
        quantizer=key_codec,
        runtime_audit=audit,
        audit_side="k",
    )

    failed_summary = summarize_runtime_audit(audit)
    assert failed_summary["status"] == "FAIL"
    assert failed_summary["turboquant_executed"] is False

    fake_svd(
        tensor,
        rank=3,
        quantizer=value_codec,
        runtime_audit=audit,
        audit_side="v",
    )
    passed_summary = summarize_runtime_audit(audit)

    assert passed_summary["status"] == "PASS"
    assert passed_summary["turboquant_executed"] is True
    assert passed_summary["k_pair_calls"] == 1
    assert passed_summary["v_pair_calls"] == 1
