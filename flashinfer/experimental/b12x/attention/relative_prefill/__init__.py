"""Warp-specialized SM120 4D-TMA relative-position prefill attention with fused gate-sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="relative_prefill",
    group="attention",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "Plan",
        "RelativePrefillConfig",
        "RelativePrefillQuery",
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
        paths=("b12x/attention/relative_prefill/",),
    ),
    test_path="",
    since="1.3.0",
    notes=(
        "SM120 4D TMA paged full-attention prefill with per-query relative bias, "
        "optional GQA grouping and split-KV, and fused per-token gate-sink epilogue."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        RelativePrefillConfig,
        RelativePrefillQuery,
        bind,
        clear_caches,
        is_supported,
        plan,
        run,
        supports,
    )

install_lazy_api(globals(), META)
