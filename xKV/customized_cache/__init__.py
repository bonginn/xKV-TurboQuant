"""Customized cache implementations with lazy optional dependencies.

Importing a lightweight submodule such as ``customized_cache.turboquant`` must
not eagerly import Transformers and the original xKV cache stack.  The public
names below retain the previous package API and load that stack only when it is
actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .fake_layer_merge_dynamic_cache import FakeLayerMergingCache


__all__ = ["FakeLayerMergingCache", "method_to_cache_obj"]


def __getattr__(name: str) -> Any:
    if name == "FakeLayerMergingCache":
        from .fake_layer_merge_dynamic_cache import FakeLayerMergingCache

        globals()[name] = FakeLayerMergingCache
        return FakeLayerMergingCache

    if name == "method_to_cache_obj":
        cache_class = __getattr__("FakeLayerMergingCache")
        mapping = {"xKV": cache_class}
        globals()[name] = mapping
        return mapping

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
