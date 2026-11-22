"""Public surface for :mod:`b12x.attention.relative_decode`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Binding, make_plan
from ._tuning import RelativeDecodeConfig, RelativeDecodeQuery, validate_query

_COMPONENT = "attention.relative_decode"


@dataclass(frozen=True, kw_only=True)
class Caps:
    """One relative-attention layer family's decode/verify contract."""

    device: torch.device | str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    is_window: bool
    commit_ring: bool
    page_or_window: int
    window_left: int
    rel_extent: int
    max_batch: int
    max_q_per_req: int = 1
    max_total_q: int | None = None


def plan(
    caps: Caps, *, invocation=None, override: RelativeDecodeConfig | None = None
) -> Plan:
    """Declare a relative decode/verify layer family; ``PreparationSession`` compiles it."""
    return make_plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch,
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    starts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    gated: torch.Tensor,
    staged_k: torch.Tensor | None = None,
    staged_v: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    state_rows: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> Binding:
    """Bind caller tensors to a prepared plan (capture-safe: no allocation)."""
    state = require_prepared(plan, _COMPONENT, q.device)
    return state.bind(
        plan=plan,
        scratch=scratch,
        q=q,
        new_k=new_k,
        new_v=new_v,
        k_cache=k_cache,
        v_cache=v_cache,
        starts=starts,
        cu_seqlens_q=cu_seqlens_q,
        bias=bias,
        gate=gate,
        sink=sink,
        gated=gated,
        staged_k=staged_k,
        staged_v=staged_v,
        page_table=page_table,
        state_rows=state_rows,
        softmax_scale=softmax_scale,
    )


def run(binding: Binding) -> torch.Tensor:
    """Launch the prepared forward (and split merge); returns ``gated``."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be relative_decode.Binding")
    return require_prepared(binding.plan, _COMPONENT, binding.q.device).run(binding)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def supports(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_window: bool,
    commit_ring: bool,
    page_or_window: int,
    window_left: int,
    rel_extent: int,
    max_q_per_req: int = 1,
    device=None,
) -> bool:
    """Whether a layer family's geometry and device fit this op's contract."""
    if not is_supported(device):
        return False
    try:
        validate_query(
            RelativeDecodeQuery(
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                is_window=bool(is_window),
                commit_ring=bool(commit_ring),
                page_or_window=int(page_or_window),
                window_left=int(window_left),
                rel_extent=int(rel_extent),
                max_batch=1,
                max_q_per_req=int(max_q_per_req),
                max_total_q=int(max_q_per_req),
                sm_count=1,
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
    "RelativeDecodeConfig",
    "RelativeDecodeQuery",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "run",
    "supports",
]
