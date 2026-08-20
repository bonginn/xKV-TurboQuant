import pytest

from xKV.attn_patch.qwen import _resolve_past_key_value


def test_qwen_cache_keyword_accepts_legacy_singular():
    cache = object()
    kwargs = {"sentinel": 1}

    resolved = _resolve_past_key_value(cache, kwargs)

    assert resolved is cache
    assert kwargs == {"sentinel": 1}


def test_qwen_cache_keyword_accepts_current_plural_and_consumes_it():
    cache = object()
    kwargs = {"past_key_values": cache, "sentinel": 1}

    resolved = _resolve_past_key_value(None, kwargs)

    assert resolved is cache
    assert kwargs == {"sentinel": 1}


def test_qwen_cache_keyword_accepts_same_cache_through_both_names():
    cache = object()
    kwargs = {"past_key_values": cache}

    resolved = _resolve_past_key_value(cache, kwargs)

    assert resolved is cache
    assert kwargs == {}


def test_qwen_cache_keyword_rejects_conflicting_caches():
    with pytest.raises(ValueError, match="different caches"):
        _resolve_past_key_value(
            object(),
            {"past_key_values": object()},
        )
