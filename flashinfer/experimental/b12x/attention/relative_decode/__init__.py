"""Warp-specialized SM120 TMA relative-position decode/verify attention with fused KV write and gate-sink.

Streams circular FP8 window rings and paged FP8 caches directly through 4D TMA
without a staging copy, quantizes and stores the live step's BF16 K/V rows in
the same launch, accumulates FP32 scores with per-query relative bias, and
fuses the per-token output gate and per-token sink sigmoid into a single BF16
round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="relative_decode",
    group="attention",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "Plan",
        "RelativeDecodeConfig",
        "RelativeDecodeQuery",
        "bind",
        "clear_caches",
        "is_supported",
        "plan",
        "run",
        "supports",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    provenance=Provenance(
        repo="local",
        commit="local",
        paths=("b12x/attention/relative_decode/",),
    ),
    test_path="",
    since="1.3.0",
    notes=(
        "SM120 4D TMA split-KV relative-position decode/verify attention over "
        "circular FP8 window rings and paged FP8 caches with fused KV write and "
        "per-token gate-sink epilogue."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        RelativeDecodeConfig,
        RelativeDecodeQuery,
        bind,
        clear_caches,
        is_supported,
        plan,
        run,
        supports,
    )

install_lazy_api(globals(), META)
