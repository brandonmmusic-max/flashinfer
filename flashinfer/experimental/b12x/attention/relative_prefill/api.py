"""Public surface for :mod:`b12x.attention.relative_prefill`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Binding, make_plan
from ._tuning import RelativePrefillConfig, RelativePrefillQuery, validate_query

_COMPONENT = "attention.relative_prefill"


@dataclass(frozen=True, kw_only=True)
class Caps:
    """One paged full or staged-window prefill layer family's contract."""

    device: torch.device | str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    rel_extent: int
    max_batch: int
    max_total_q: int
    window_left: int = -1


def plan(
    caps: Caps, *, invocation=None, override: RelativePrefillConfig | None = None
) -> Plan:
    return make_plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    gated: torch.Tensor,
    softmax_scale: float | None = None,
) -> Binding:
    state = require_prepared(plan, _COMPONENT, q.device)
    return state.bind(
        plan=plan,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        lengths=lengths,
        cu_seqlens_q=cu_seqlens_q,
        bias=bias,
        gate=gate,
        sink=sink,
        gated=gated,
        softmax_scale=softmax_scale,
    )


def run(binding: Binding) -> torch.Tensor:
    if not isinstance(binding, Binding):
        raise TypeError("binding must be relative_prefill.Binding")
    return require_prepared(binding.plan, _COMPONENT, binding.q.device).run(binding)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def supports(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    rel_extent: int,
    max_total_q: int,
    window_left: int = -1,
    device=None,
) -> bool:
    if not is_supported(device):
        return False
    try:
        validate_query(
            RelativePrefillQuery(
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                page_size=int(page_size),
                rel_extent=int(rel_extent),
                max_batch=1,
                max_total_q=int(max_total_q),
                sm_count=1,
                window_left=int(window_left),
            )
        )
    except (TypeError, ValueError):
        return False
    return True


def clear_caches() -> None:
    """Compiled programs live on prepared plans; nothing is cached globally."""


__all__ = [
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
]
