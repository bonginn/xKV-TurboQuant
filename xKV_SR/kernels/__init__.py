import os
import sys


class _MissingShadowKV:
    """Delay the optional legacy-extension error until a path uses it."""

    def __init__(self, import_error: BaseException):
        self._import_error = import_error

    def __getattr__(self, name):
        raise RuntimeError(
            "This path requires the legacy top-level `shadowkv` CUDA "
            "extension, but it is not installed. The Qwen xKV-SR accuracy "
            "path does not use this extension; optimized/CPU-offload paths do."
        ) from self._import_error


sys.path.insert(0, os.path.dirname(__file__))
try:
    import shadowkv as shadowkv
except (ImportError, OSError) as exc:
    shadowkv = _MissingShadowKV(exc)
