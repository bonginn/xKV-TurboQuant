"""Tests for the accuracy-only xKV-SR + TurboQuant factor path."""

from types import SimpleNamespace

import pytest
import torch

from evaluate.eval_acc import (
    _configure_xkvsr_factor_quantization,
    _xkvsr_setting,
)
from xKV_SR.models.factor_quant import TurboQuantFactorQuantizer
from xKV_SR.models.kv_cache_xkv import ShadowKVCache_xKV
from xKV_SR.models.merge_configs import generate_consecutive_palu_config


def _args(**overrides):
    values = {
        "method": "xkv_sr",
        "kv_quantizer": "turboquant_mse",
        "kv_bits": 16,
        "k_bits": 4,
        "v_bits": 4,
        "fake_svd": False,
        "hadamard": False,
        "group_size": 0,
        "sym": False,
        "clip_ratio": 1.0,
        "turboquant_rotation_seed": 42,
        "turboquant_chunk_size": 4096,
        "turboquant_codebook_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_xkvsr_turboquant_folded_pair_matches_explicit_decode():
    torch.manual_seed(19)
    quantizer = TurboQuantFactorQuantizer(
        rank_k=3,
        rank_v=3,
        k_bits=3,
        v_bits=2,
        rotation_seed=17,
        chunk_size=3,
        device="cpu",
    )
    left = torch.randn(2, 5, 3, dtype=torch.float32)
    right = torch.randn(2, 3, 7, dtype=torch.float32)

    pair = quantizer.transform_pair(left, right, side="k")
    explicit_left = quantizer.codec_for("k").decode(
        pair.codes,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(
        pair.left_factor @ pair.right_factor,
        explicit_left @ right,
        rtol=1e-5,
        atol=1e-6,
    )
    assert pair.left_factor.dtype == torch.float32
    assert pair.right_factor.dtype == torch.float32
    assert pair.codes.indices.dtype == torch.uint8


def test_xkvsr_turboquant_registry_reuses_identical_codec():
    quantizer = TurboQuantFactorQuantizer(
        rank_k=3,
        rank_v=3,
        k_bits=2,
        v_bits=2,
        rotation_seed=17,
        device="cpu",
    )

    assert quantizer.codec_for("k") is quantizer.codec_for("v")
    assert quantizer.registry_size == 1


@pytest.mark.parametrize("name,value", [("k_bits", 0), ("v_bits", 5)])
def test_xkvsr_turboquant_rejects_unsupported_bit_width(name, value):
    kwargs = {
        "rank_k": 3,
        "rank_v": 3,
        "k_bits": 2,
        "v_bits": 2,
        "device": "cpu",
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match="one of 1, 2, 3, 4, or 16"):
        TurboQuantFactorQuantizer(**kwargs)


def test_xkvsr_turboquant_supports_single_side_bypass():
    quantizer = TurboQuantFactorQuantizer(
        rank_k=3,
        rank_v=3,
        k_bits=16,
        v_bits=2,
        device="cpu",
    )

    assert quantizer.is_enabled("k") is False
    assert quantizer.is_enabled("v") is True
    assert quantizer.registry_size == 1
    with pytest.raises(ValueError, match="K factor quantization is bypassed"):
        quantizer.codec_for("k")


def test_xkvsr_turboquant_rejects_bypassing_both_sides():
    with pytest.raises(ValueError, match="at least one"):
        TurboQuantFactorQuantizer(
            rank_k=3,
            rank_v=3,
            k_bits=16,
            v_bits=16,
            device="cpu",
        )


def test_xkvsr_turboquant_result_key_is_non_colliding():
    assert _xkvsr_setting(_args()) == "xkv_sr_tqmse_native_k4v4_fq"
    assert _xkvsr_setting(
        _args(k_bits=16, v_bits=2)
    ) == "xkv_sr_tqmse_native_kbf16v2_fq"
    assert _xkvsr_setting(
        _args(k_bits=4, v_bits=16)
    ) == "xkv_sr_tqmse_native_k4vbf16_fq"
    assert _xkvsr_setting(
        _args(kv_quantizer="rtn", k_bits=None, v_bits=None)
    ) == "xkv_sr"


def test_xkvsr_cli_configuration_rejects_stacked_hadamard():
    llm = SimpleNamespace(kv_cache=SimpleNamespace())
    with pytest.raises(ValueError, match="do not combine"):
        _configure_xkvsr_factor_quantization(llm, _args(hadamard=True))


def test_xkvsr_cli_configuration_forwards_turboquant_options():
    observed = {}

    class Cache:
        def configure_turboquant(self, **kwargs):
            observed.update(kwargs)

    llm = SimpleNamespace(kv_cache=Cache())
    _configure_xkvsr_factor_quantization(
        llm,
        _args(
            k_bits=3,
            v_bits=2,
            turboquant_rotation_seed=9,
            turboquant_chunk_size=128,
            turboquant_codebook_path="tables/codebooks.json",
        ),
    )

    assert observed == {
        "k_bits": 3,
        "v_bits": 2,
        "rotation_seed": 9,
        "chunk_size": 128,
        "codebook_path": "tables/codebooks.json",
    }


def test_xkvsr_cli_rejects_bypassing_both_sides():
    llm = SimpleNamespace(kv_cache=SimpleNamespace())
    with pytest.raises(ValueError, match="BF16 baseline"):
        _configure_xkvsr_factor_quantization(
            llm,
            _args(k_bits=16, v_bits=16),
        )


def test_xkvsr_single_side_audit_expects_only_enabled_side():
    cache = ShadowKVCache_xKV.__new__(ShadowKVCache_xKV)
    cache.quant_audit_enabled = True
    cache.factor_quantization = TurboQuantFactorQuantizer(
        rank_k=3,
        rank_v=3,
        k_bits=16,
        v_bits=2,
        device="cpu",
    )
    cache.num_layers = 2
    cache.group_size = 2
    cache.rank_k = 3
    cache.rank_v = 3
    cache.dtype = torch.bfloat16
    cache.reset_quant_audit()
    cache._quant_audit.update(
        prefill_calls=1,
        group_merges=1,
        v_pair_calls=1,
        factor_pairs_checked=1,
        v_reconstruction_calls=1,
        reconstruction_tensors_checked=1,
    )

    summary = cache.quant_audit_summary()

    assert summary["status"] == "PASS"
    assert summary["quantization_executed"] is True
    assert summary["quantized_sides"] == ["v"]
    assert summary["bypassed_sides"] == ["k"]
    assert summary["k_pair_calls"] == 0
    assert summary["v_pair_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "k_bits,v_bits,quantized_sides",
    [
        (2, 2, ["k", "v"]),
        (16, 2, ["v"]),
        (2, 16, ["k"]),
    ],
)
def test_xkvsr_cache_uses_turboquant_factors_for_kv_reconstruction(
    monkeypatch,
    k_bits,
    v_bits,
    quantized_sides,
):
    monkeypatch.setenv("XKV_RUNTIME_AUDIT", "1")
    model_config = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=4,
        num_hidden_layers=2,
    )
    merge_config = generate_consecutive_palu_config(
        num_layers=2,
        start_layer=0,
        end_layer=1,
        group_size=2,
        rank_k=3,
        rank_v=3,
    )
    cache = ShadowKVCache_xKV(
        model_config,
        merge_config,
        max_length=16,
        device="cuda:0",
        dtype=torch.bfloat16,
        sparse_budget=2,
        chunk_size=1,
    )
    cache.configure_turboquant(
        k_bits=k_bits,
        v_bits=v_bits,
        rotation_seed=17,
        chunk_size=4,
    )
    cache.reset_quant_audit()
    cache.clear()

    torch.manual_seed(23)
    for layer_idx in range(2):
        key = torch.randn(
            1, 2, 4, 2, device="cuda:0", dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        cache.get_svd(key, value, layer_idx)

    assert len(cache.U_k) == 1
    assert len(cache.U_v) == 1
    assert len(cache.SV_k) == 2
    assert len(cache.SV_v) == 2
    assert cache.U_k[0].dtype == torch.bfloat16
    assert cache.U_v[0].dtype == torch.bfloat16
    assert all(factor.dtype == torch.bfloat16 for factor in cache.SV_k)
    assert all(factor.dtype == torch.bfloat16 for factor in cache.SV_v)

    cache.sparse_start = 0
    cache.sparse_end = 2
    cache.incoming_q_len = 1
    positions = torch.tensor([[[0, 1], [1, 2]]], device="cuda:0")
    value_result = cache.get_value_cache(0, positions)
    key_result = cache.get_key_cache(0, positions, lambda value, _: value)

    assert value_result.dtype == torch.bfloat16
    assert key_result.dtype == torch.bfloat16
    assert torch.isfinite(value_result).all()
    assert torch.isfinite(key_result).all()

    audit = cache.quant_audit_summary()
    assert audit["status"] == "PASS"
    assert audit["quantization_executed"] is True
    assert audit["group_merges"] == 1
    assert audit["quantized_sides"] == quantized_sides
    assert audit["k_pair_calls"] == int("k" in quantized_sides)
    assert audit["v_pair_calls"] == int("v" in quantized_sides)
    assert audit["physically_packed"] is False
    assert audit["persistent_storage"] == "dequantized_model_dtype"
    assert audit["cache_dtype"] == "torch.bfloat16"
