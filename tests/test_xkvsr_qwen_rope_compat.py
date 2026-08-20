from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("flash_attn")

from xKV_SR.models.qwen import Qwen2


class _ModelLevelRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim=8, offset=10.0):
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(head_dim // 2))
        self.head_dim = head_dim
        self.offset = offset
        self.calls = 0

    def forward(self, x, position_ids):
        self.calls += 1
        positions = position_ids.to(dtype=x.dtype).unsqueeze(-1)
        cos = positions.expand(-1, -1, self.head_dim) + self.offset
        sin = positions.expand(-1, -1, self.head_dim) - self.offset
        return cos, sin


class _LegacyAttentionRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim=8, offset=3.0):
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(head_dim // 2))
        self.head_dim = head_dim
        self.offset = offset
        self.calls = 0

    def forward(self, x, seq_len=None):
        self.calls += 1
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(-1)
        cos = positions.expand(-1, self.head_dim) + self.offset
        sin = positions.expand(-1, self.head_dim) - self.offset
        return cos, sin


def _qwen_stub():
    model = Qwen2.__new__(Qwen2)
    model.device = "cpu"
    model.dtype = torch.float32
    model.hidden_size = 32
    model.head_dim = 8
    model.num_key_value_heads = 2
    model.max_length = 4
    return model


def test_qwen_rope_uses_new_model_level_location_first():
    model_level = _ModelLevelRotaryEmbedding(offset=10.0)
    legacy = _LegacyAttentionRotaryEmbedding(offset=20.0)
    hf_model = SimpleNamespace(
        model=SimpleNamespace(
            rotary_emb=model_level,
            layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=legacy))],
        )
    )

    cos, sin = _qwen_stub()._set_cos_sin_cache(hf_model)

    assert cos.shape == (4, 8)
    assert sin.shape == (4, 8)
    assert torch.equal(cos[:, 0], torch.arange(4, dtype=torch.float32) + 10)
    assert torch.equal(sin[:, 0], torch.arange(4, dtype=torch.float32) - 10)
    assert model_level.calls == 1
    assert legacy.calls == 0


def test_qwen_rope_supports_legacy_attention_level_location():
    legacy = _LegacyAttentionRotaryEmbedding(offset=3.0)
    hf_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=legacy))]
        )
    )

    cos, sin = _qwen_stub()._set_cos_sin_cache(hf_model)

    assert cos.shape == (4, 8)
    assert sin.shape == (4, 8)
    assert legacy.calls == 1


def test_qwen_rope_reports_missing_embedding_locations():
    hf_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace())]
        )
    )

    with pytest.raises(RuntimeError, match="Unable to locate"):
        _qwen_stub()._set_cos_sin_cache(hf_model)


def test_qwen_rope_rejects_unexpected_cache_shape():
    bad_rotary = _ModelLevelRotaryEmbedding(head_dim=6)
    hf_model = SimpleNamespace(
        model=SimpleNamespace(
            rotary_emb=bad_rotary,
            layers=[SimpleNamespace(self_attn=SimpleNamespace())],
        )
    )

    with pytest.raises(RuntimeError, match="Unexpected Qwen2 RoPE cache shapes"):
        _qwen_stub()._set_cos_sin_cache(hf_model)
