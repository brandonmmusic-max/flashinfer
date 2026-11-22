"""Norm/residual ops for b12x.

- ``hyperconnection``: learned low-rank multi-stream residual mixing.
- ``inkling2_residual``: Inkling 2 attention-residual mixture-normalization and fused expert commit.
- ``mhc``: fused RMSNorm + hyper-connection residual mixing + projection
  (DeepSeek-style mHC), pre/post/post-pre phases.
- ``torch_rmsnorm``: caller-owned BF16 RMSNorm with Torch's vectorized FP32 reduction order.
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("hyperconnection", "inkling2_residual", "mhc", "torch_rmsnorm")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
