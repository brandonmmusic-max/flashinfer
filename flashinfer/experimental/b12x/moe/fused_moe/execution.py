"""Declarative canonical fused-MoE execution preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import FrozenMapping


@dataclass(frozen=True, kw_only=True)
class ExecutionCapacity:
    """Token capacity, decode specializations, and output storage geometry.

    ``token`` outputs hold weighted [tokens, hidden] values. ``routes`` exposes
    unweighted [tokens, top_k, hidden] GEMM outputs in activation precision for
    token or expert outer scaling. Invalid route entries are unspecified;
    consumers must mask them with the route IDs before applying their weights.
    Expert-outer NVFP4 routes reduce FC2 slices in a fixed FP32 order before
    the final BF16 conversion.
    """

    max_tokens: int
    top_k: int
    warmup_token_counts: tuple[int, ...] = ()
    route_num_experts: int | None = None
    output_layout: str = "token"

    def __post_init__(self):
        if self.output_layout not in ("token", "routes"):
            raise ValueError("MoE output layout must be token or routes")
        max_tokens, top_k = int(self.max_tokens), int(self.top_k)
        if max_tokens <= 0 or top_k <= 0:
            raise ValueError("max_tokens and top_k must be positive")
        counts = tuple(sorted({int(value) for value in self.warmup_token_counts}))
        if any(value <= 0 or value > max_tokens for value in counts):
            raise ValueError("warmup token counts must be positive and within capacity")
        if self.route_num_experts is not None and int(self.route_num_experts) < 0:
            raise ValueError("route_num_experts cannot be negative")
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "warmup_token_counts", counts)
        if self.route_num_experts is not None:
            object.__setattr__(self, "route_num_experts", int(self.route_num_experts))


@dataclass(frozen=True, kw_only=True)
class RoutingSpec:
    """Immutable routing and numerical controls captured by preparation."""

    apply_router_weight_on_input: bool = False
    logits_dtype: torch.dtype | None = None
    deterministic_output: bool | None = None
    collect_activation_amax: bool = False
    score_func: str = "softmax"
    renormalize: bool = True
    has_correction_bias: bool = False
    has_image_correction_bias: bool = False
    has_image_mask: bool = False
    has_valid_tokens: bool = False
    routed_scaling_factor: float = 1.0


def plan_execution(
    *,
    experts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
    scale_expansion=None,
    smallm_max_tokens: int | None = None,
):
    """Return a composite declaration, never an executable plan or warmup handle.

    ``smallm_max_tokens`` declares how much of the small-M arm's 1..16 token
    range a ``source_format="trellismx"`` plan serves (``None``: the arm's
    whole range, 16; ``0``: the grouped arm serves every token count). The
    choice is validated, recorded in the plan's controls and tuning query, and
    compiled for; it is not a process-wide setting.
    """
    if experts.plan.activation.outer_scale == "token":
        if smallm_max_tokens is not None:
            raise ValueError(
                "smallm_max_tokens applies to the expert-outer MoE plan; the "
                "token-outer plan has no TrellisMX arm"
            )
        from ._token_outer import plan

        # The token-outer plan has no small-M arm, so it takes no such argument.
        branch_kwargs = {}
    else:
        branch_kwargs = {"smallm_max_tokens": smallm_max_tokens}
        if (
            capacity.output_layout == "routes"
            and experts.plan._impl.quant_modes != frozenset({"nvfp4"})
        ):
            raise ValueError("expert-scale route outputs require NVFP4 A4 preparation")
        if routing is not None and routing.has_valid_tokens:
            raise ValueError(
                "valid-token integrated routing requires token outer scaling"
            )
        from ._preparation import plan

    if (
        capacity.output_layout == "routes"
        and routing is not None
        and routing.apply_router_weight_on_input
    ):
        raise ValueError(
            "unweighted route outputs cannot apply router weights on input"
        )

    return plan(
        experts,
        capacity=capacity,
        routing=routing,
        invocation=invocation,
        override=override,
        scale_expansion=scale_expansion,
        **branch_kwargs,
    )


__all__ = ["ExecutionCapacity", "RoutingSpec", "plan_execution"]
