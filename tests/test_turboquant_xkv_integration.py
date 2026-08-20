"""Integration tests for the accuracy-first xKV + TurboQuant-MSE path."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from xKV.configurations import (
    LayerGroup,
    generate_consecutive_xKV_config,
    xKVConfig,
)
from xKV.customized_cache.fake_layer_merge_dynamic_cache import (
    FakeLayerMergingCache,
    fake_svd,
)
from xKV.customized_cache.turboquant import TurboQuantMSE
from utils import add_common_args, apply_kv_compress_patch


def _turboquant_config(**overrides):
    values = {
        "num_layers": 2,
        "layer_merge_impl": "svd",
        "rank_k": 3,
        "rank_v": 3,
        "merge_key": True,
        "merge_value": True,
        "kv_quantizer": "turboquant_mse",
        "k_bits": 3,
        "v_bits": 2,
        "hadamard": False,
        "turboquant_rotation_seed": 17,
        "turboquant_chunk_size": 4,
        "layer_groups": [LayerGroup(layers=[0, 1])],
    }
    values.update(overrides)
    return xKVConfig(**values)


def test_fake_svd_turboquant_matches_manual_folded_factor_product():
    torch.manual_seed(5)
    tensor = torch.randn(2, 2, 5, 2, dtype=torch.float32)
    codec = TurboQuantMSE(3, 2, rotation_seed=17, chunk_size=3)

    observed = fake_svd(tensor, rank=3, quantizer=codec)

    batch_size, num_heads, sequence_length, head_dim = tensor.shape
    matrix = tensor.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )
    u_matrix, singular_values, vh_matrix = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    sqrt_singular_values = torch.sqrt(singular_values[:, :3])
    left_factor = u_matrix[:, :, :3] * sqrt_singular_values.unsqueeze(1)
    right_factor = (
        sqrt_singular_values.unsqueeze(-1) * vh_matrix[:, :3, :]
    )
    factor_pair = codec.fake_quantize_pair(left_factor, right_factor)
    expected = torch.matmul(
        factor_pair.left_factor,
        factor_pair.right_factor,
    )
    expected = expected.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_fake_svd_rejects_stacked_hadamard_and_turboquant():
    tensor = torch.randn(1, 2, 4, 2)
    codec = TurboQuantMSE(3, 2, rotation_seed=17)

    with pytest.raises(ValueError, match="cannot be combined"):
        fake_svd(
            tensor,
            rank=3,
            apply_hadamard_transform=True,
            quantizer=codec,
        )


def test_fake_svd_zero_pads_short_prompt_to_configured_turboquant_rank():
    tensor = torch.randn(1, 2, 2, 2)
    codec = TurboQuantMSE(3, 2, rotation_seed=17)

    observed = fake_svd(tensor, rank=3, quantizer=codec)

    matrix = tensor.transpose(1, 2).reshape(1, 2, 4)
    u_matrix, singular_values, vh_matrix = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    sqrt_singular_values = torch.sqrt(singular_values)
    left_factor = u_matrix * sqrt_singular_values.unsqueeze(1)
    right_factor = sqrt_singular_values.unsqueeze(-1) * vh_matrix
    left_factor = torch.cat(
        [left_factor, left_factor.new_zeros(1, 2, 1)],
        dim=-1,
    )
    right_factor = torch.cat(
        [right_factor, right_factor.new_zeros(1, 1, 4)],
        dim=-2,
    )
    factor_pair = codec.fake_quantize_pair(left_factor, right_factor)
    expected = torch.matmul(
        factor_pair.left_factor,
        factor_pair.right_factor,
    ).view(1, 2, 2, 2).transpose(1, 2)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_fake_svd_turboquant_disables_outer_cpu_autocast_for_reconstruction():
    torch.manual_seed(13)
    tensor = torch.randn(1, 2, 5, 2, dtype=torch.float32)
    codec = TurboQuantMSE(3, 2, rotation_seed=17)

    expected = fake_svd(tensor, rank=3, quantizer=codec)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        observed = fake_svd(tensor, rank=3, quantizer=codec)

    assert observed.dtype == torch.float32
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_cache_uses_asymmetric_codecs_and_reuses_shared_registry():
    config = _turboquant_config()
    shared_registry = {}
    first_cache = FakeLayerMergingCache(
        merge_setup=config,
        turboquant_registry=shared_registry,
    )
    second_cache = FakeLayerMergingCache(
        merge_setup=config,
        turboquant_registry=shared_registry,
    )

    key_codec = first_cache._quantizer_for("k", 3, torch.device("cpu"))
    value_codec = first_cache._quantizer_for("v", 3, torch.device("cpu"))

    assert key_codec.dimension == 3
    assert key_codec.bits == 3
    assert value_codec.dimension == 3
    assert value_codec.bits == 2
    assert key_codec is not value_codec
    assert len(shared_registry) == 2
    assert (
        second_cache._quantizer_for("k", 3, torch.device("cpu"))
        is key_codec
    )
    assert len(shared_registry) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grouped_cache_merge_runs_turboquant_k3_v2_end_to_end():
    config = _turboquant_config()
    shared_registry = {}
    cache = FakeLayerMergingCache(
        merge_setup=config,
        turboquant_registry=shared_registry,
    )
    device = torch.device("cuda")
    tensors = [
        torch.randn(1, 2, 4, 2, device=device)
        for _ in range(4)
    ]

    cache.update(
        tensors[0],
        tensors[1],
        layer_idx=0,
        mode="prefill",
        re_apply_rope=False,
    )
    cache.update(
        tensors[2],
        tensors[3],
        layer_idx=1,
        mode="prefill",
        re_apply_rope=False,
    )

    assert len(shared_registry) == 2
    for layer_index in (0, 1):
        cached_key, cached_value = cache.get_layer_cache(layer_index)
        assert cached_key.shape == tensors[0].shape
        assert cached_value.shape == tensors[1].shape
        assert torch.isfinite(cached_key).all()
        assert torch.isfinite(cached_value).all()


def test_dynamic_cache_storage_adapter_round_trip():
    config = xKVConfig(num_layers=1, layer_groups=[])
    cache = FakeLayerMergingCache(config)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)

    returned_key, returned_value = cache.update(
        key,
        value,
        layer_idx=0,
        mode="decode",
        re_apply_rope=False,
    )
    torch.testing.assert_close(returned_key, key)
    torch.testing.assert_close(returned_value, value)

    replacement_key = torch.randn_like(key)
    replacement_value = torch.randn_like(value)
    cache.update_cache(0, replacement_key, replacement_value)
    stored_key, stored_value = cache.get_layer_cache(0)

    torch.testing.assert_close(stored_key, replacement_key)
    torch.testing.assert_close(stored_value, replacement_value)
    assert cache.get_seq_length(0) == 3
    assert cache.get_max_length(layer_idx=0) in (None, -1)


def test_turboquant_config_yaml_round_trip(tmp_path):
    config = _turboquant_config(
        turboquant_rotation_seed=99,
        turboquant_chunk_size=128,
        turboquant_codebook_path="tables/custom.json",
    )
    path = tmp_path / "turboquant.yaml"

    config.to_yaml(str(path))
    loaded = xKVConfig.from_yaml(str(path))

    assert loaded.kv_quantizer == "turboquant_mse"
    assert loaded.quant_bits_for("k") == 3
    assert loaded.quant_bits_for("v") == 2
    assert loaded.turboquant_rotation_seed == 99
    assert loaded.turboquant_chunk_size == 128
    assert loaded.turboquant_codebook_path == str(
        (tmp_path / "tables" / "custom.json").resolve()
    )
    assert loaded.hadamard is False
    assert loaded.layer_groups[0].rank_k == 3
    assert loaded.layer_groups[0].rank_v == 3


def test_legacy_rtn_defaults_remain_backward_compatible():
    config = xKVConfig(
        num_layers=1,
        rank_k=3,
        rank_v=3,
        kv_bits=4,
        layer_groups=[LayerGroup(layers=[0])],
    )
    cache = FakeLayerMergingCache(config)

    assert config.kv_quantizer == "rtn"
    assert cache.quantizer.n_bits == 4
    assert cache.use_hadamard is True


def test_xkv_config_legacy_positional_argument_order_is_preserved():
    group = LayerGroup(layers=[0])
    config = xKVConfig(
        1,
        "svd",
        3,
        3,
        0.5,
        1.0,
        True,
        True,
        4,
        0,
        False,
        1.0,
        True,
        [group],
        {},
    )

    assert config.kv_bits == 4
    assert config.layer_groups == [group]
    assert config.kv_quantizer == "rtn"


def test_config_generator_legacy_positional_argument_order_is_preserved():
    config = generate_consecutive_xKV_config(
        "svd",
        0,
        0,
        1,
        1,
        3,
        3,
        0.5,
        1.0,
        True,
        True,
        4,
        0,
        False,
        1.0,
        True,
        {},
    )

    assert config.kv_bits == 4
    assert config.layer_groups[0].layers == [0]
    assert config.kv_quantizer == "rtn"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hadamard": True}, "already applies a dense Haar rotation"),
        ({"k_bits": 5}, "K bits must be between 1 and 4"),
        ({"turboquant_chunk_size": 0}, "must be None or a positive integer"),
    ],
)
def test_turboquant_config_rejects_ambiguous_or_unverified_options(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _turboquant_config(**overrides)


def test_rtn_rejects_unused_asymmetric_bit_arguments():
    with pytest.raises(ValueError, match="only supported"):
        xKVConfig(
            num_layers=1,
            rank_k=3,
            rank_v=3,
            kv_quantizer="rtn",
            k_bits=3,
            layer_groups=[LayerGroup(layers=[0])],
        )


def test_turboquant_config_rejects_nonconsecutive_custom_group():
    with pytest.raises(ValueError, match="sorted consecutive layers"):
        _turboquant_config(
            layer_groups=[LayerGroup(layers=[0, 2])],
            num_layers=3,
        )


def test_turboquant_cache_preflights_missing_offline_codebook():
    config = _turboquant_config(
        rank_k=4,
        layer_groups=[LayerGroup(layers=[0, 1], rank_k=4, rank_v=3)],
    )

    with pytest.raises(ValueError, match=r"\(4, 3\)"):
        FakeLayerMergingCache(config)


def test_turboquant_config_rejects_both_merge_sides_disabled():
    with pytest.raises(ValueError, match="requires merge_key and/or merge_value"):
        _turboquant_config(merge_key=False, merge_value=False)


def test_turboquant_config_rejects_empty_layer_groups():
    with pytest.raises(ValueError, match="at least one layer group"):
        _turboquant_config(layer_groups=[])


def test_common_cli_parses_asymmetric_turboquant_arguments():
    parser = add_common_args(ArgumentParser())
    args = parser.parse_args(
        [
            "--kv_quantizer",
            "turboquant_mse",
            "--k_bits",
            "3",
            "--v_bits",
            "2",
            "--turboquant_rotation_seed",
            "99",
        ]
    )

    assert args.kv_quantizer == "turboquant_mse"
    assert args.k_bits == 3
    assert args.v_bits == 2
    assert args.turboquant_rotation_seed == 99
    assert args.hadamard is False


def test_custom_yaml_rejects_mixed_turboquant_cli_flags():
    args = SimpleNamespace(
        customized_merge_config="config.yaml",
        kv_quantizer="turboquant_mse",
        k_bits=3,
        v_bits=2,
        turboquant_rotation_seed=42,
        turboquant_chunk_size=4096,
        turboquant_codebook_path=None,
    )

    with pytest.raises(ValueError, match="customized_merge_config is authoritative"):
        apply_kv_compress_patch(object(), args)
