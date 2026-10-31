"""What a P8 capacity declares: its arms, its launch specs and its arena.

This module is the declaration side of the runtime: the program spec a compile
job carries, the arms a capacity of some row count reaches, and the scratch its
calls need. It imports nothing from the kernel tree, so planning a P8 capacity
(and planning it again from a deserialized descriptor) does not load the
kernels; the runtime builds its programs from these same declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .launch import P8LaunchConfig
from .scratch import p8_launch_layout, use_small_m
from .tile_policy import select_tile

__all__ = [
    "P8ProgramSpec",
    "arm_key",
    "launch_spec",
    "p8_capacity_launch_rows",
    "p8_capacity_scratch_nbytes",
    "p8_launch_rows",
    "p8_scratch_layout_for",
    "select_arm",
]


@dataclass(frozen=True)
class P8ProgramSpec:
    """Every runtime setting that selects or specializes the launched programs.

    The spec is the metadata a compile job carries: a change to any field
    changes the compile identity of the programs a call launches, so planned
    and resolved keys agree only when both are built from the same spec. The
    launch axes (``P8LaunchConfig``) are part of it, so a plan can declare and
    compile one spec per raced configuration.

    It carries no tensor-parallel rank. No P8 arm or reducer kernel receives
    one: the rank selects which sidecar slice a runtime holds and is verified
    against that slice's metadata, and every kernel the spec compiles is
    identical on all four ranks. A rank field would give the four ranks of one
    layer different compile identities, and b12x's collective preparation
    requires every rank to declare the same race keys.
    """

    trellis_bits: int
    experts: int
    hidden: int
    intermediate: int
    topk: int
    swiglu_limit: float
    full_coupled: bool
    scale_sandwich: bool
    #: Placement and dispatch overrides that select or specialize a program.
    mac_override: int | None
    deterministic_output: bool
    small_m_scheduler: bool
    force_materialized: bool | None
    rp2_direct_max_tokens: int
    rp2_skip_input_clear: bool
    prefill_chunk_tokens: int
    compact_input_storage: bool
    p8_down_remainder: bool
    dx2_phases: str
    p8_dx2_rowpack: bool
    p8_input_rowpack: bool
    p8_epi_par: bool
    tile_major_tasks: bool
    fc1_tile_n: int
    #: Launch axes (``P8LaunchConfig``): the tile of each arm, the direct
    #: carrier stride, the direct arm's FC1 schedule and the resident-grid cap.
    rp2_direct_tile_m: int
    grouped_tile_m: int
    rp2_stride: int
    fc1_warps: int
    fc1_pipeline_stages: int
    fc1_a_swizzle_rotate: bool
    fc1_warp_quant: bool
    fc1_exact_staging: bool
    fc1_broadcast_a: bool


def launch_spec(
    spec: P8ProgramSpec, config: P8LaunchConfig | None = None
) -> P8ProgramSpec:
    """The program spec of one launch configuration.

    The config's compile axes replace the spec's serving ones; its resident-grid
    cap is a launch argument and stays out of the spec (each compiled arm
    carries the cap it was built with). A plan and the runtime both derive the
    spec from the same declaration this way, so a planned compile job and a
    launched program agree.
    """

    if config is None:
        return spec
    if not isinstance(config, P8LaunchConfig):
        raise TypeError(f"config must be a P8LaunchConfig; got {type(config)}")
    return replace(
        spec,
        rp2_direct_tile_m=config.direct_tile_m,
        grouped_tile_m=config.grouped_tile_m,
        rp2_stride=config.rp2_stride,
        fc1_pipeline_stages=config.fc1_pipeline_stages,
        fc1_warps=config.fc1_warps,
    )


def select_arm(spec: P8ProgramSpec, rows: int) -> tuple[bool, bool]:
    """``(materialized, small_m)`` for a launch of ``rows`` rows.

    The small-M owner is the historical M1 arm; RP2's direct kernels extend it
    to every row count up to the speculative-verify limit, and the grouped arm
    owns the rest. A full-coupled runtime runs the split materialized kernels
    either way, the identity-boundary runtime the monolithic ones, and
    ``force_materialized`` overrides the latter.
    """

    rows = int(rows)
    if rows < 1:
        raise ValueError(f"a P8 launch has at least one row; got {rows}")
    small_m = bool(spec.small_m_scheduler) and (
        use_small_m(True, rows)
        or (spec.p8_input_rowpack and rows <= spec.rp2_direct_max_tokens)
    )
    if spec.force_materialized is not None:
        return bool(spec.force_materialized), small_m
    return bool(spec.full_coupled or small_m), small_m


def arm_key(
    spec: P8ProgramSpec,
    materialized: bool,
    small_m: bool = False,
    expected_m: int | None = None,
) -> tuple[bool, bool, int, int, int, int]:
    """The identity of one compiled arm: what it is and its row geometry.

    The direct arm is one fixed-capacity arm: every enabled call uses one
    direct-row geometry and one residual-carrier stride, and its FC1 schedule
    is part of its identity. The grouped arm's program does not read the
    small-M schedule, so those axes are zero there and one program serves every
    launch-axis value.
    """

    selected_m = 1 if expected_m is None else int(expected_m)
    if selected_m < 1:
        raise ValueError(f"a P8 arm covers at least one row; got {selected_m}")
    if not small_m:
        return (
            materialized,
            False,
            spec.grouped_tile_m or select_tile(selected_m)[0],
            0,
            0,
            0,
        )
    tile_m = spec.rp2_direct_tile_m or select_tile(1)[0]
    # Fixed rather than per-token-count: the RP2 carrier pairs each direct
    # row with a residual row a constant distance away, so one compiled arm
    # serves every enabled row count.
    return (
        materialized,
        True,
        tile_m,
        spec.rp2_stride,
        spec.fc1_pipeline_stages,
        spec.fc1_warps,
    )


def p8_launch_rows(spec: P8ProgramSpec, tokens: int) -> tuple[int, ...]:
    """Row counts of the launches one call of ``tokens`` rows makes.

    A call launches once, or full chunks of ``prefill_chunk_tokens`` rows and
    one tail, so no launch of any call exceeds the chunk.
    """

    chunk = int(spec.prefill_chunk_tokens)
    if chunk and tokens > chunk:
        rows = {chunk}
        if tokens % chunk:
            rows.add(tokens % chunk)
        return tuple(sorted(rows))
    return (tokens,)


def _capacity_row_limit(spec: P8ProgramSpec, tokens: int) -> int:
    """The largest launch any call of 1 to ``tokens`` rows makes."""

    chunk = int(spec.prefill_chunk_tokens)
    return min(int(tokens), chunk) if chunk else int(tokens)


def p8_capacity_launch_rows(spec: P8ProgramSpec, tokens: int) -> tuple[int, ...]:
    """One launch row count for each arm that calls of 1 to ``tokens`` rows reach.

    Every launch of such a call has between 1 and ``min(tokens, chunk)`` rows (a call
    launches once, or full chunks and a tail), and a launch's arm depends only on
    its rows, so these counts launch every arm the capacity accepts.
    """

    first_rows = {}
    for rows in range(1, _capacity_row_limit(spec, tokens) + 1):
        materialized, small_m = select_arm(spec, rows)
        first_rows.setdefault(arm_key(spec, materialized, small_m, rows), rows)
    return tuple(sorted(first_rows.values()))


def p8_scratch_layout_for(spec: P8ProgramSpec, rows: int):
    """The arena one launch of ``rows`` rows of ``spec`` needs."""

    materialized, small_m = select_arm(spec, rows)
    key = arm_key(spec, materialized, small_m, rows)
    return p8_launch_layout(
        tokens=rows,
        tile_m=key[2],
        direct=small_m,
        intermediate=spec.intermediate,
        hidden=spec.hidden,
        experts=spec.experts,
        topk=spec.topk,
        planes=2 if spec.p8_down_remainder else 1,
        # The full-coupled grouped arm addresses the input carrier by token and
        # reads one activation-scale word per token; the direct arm does neither.
        compact_input=bool(
            spec.compact_input_storage
            and spec.full_coupled
            and materialized
            and not small_m
        ),
        grouped_scales=bool(spec.full_coupled and materialized and not small_m),
        deterministic_output=spec.deterministic_output,
        coupled=spec.full_coupled,
        # Only the direct arm's RP2 prologue stages a residual carrier, and the
        # arm carries the stride whether or not the launch configuration names
        # it (``arm_key`` substitutes the serving one).
        rp2_stride=key[3] if small_m and spec.p8_input_rowpack else 0,
    )


def p8_capacity_scratch_nbytes(spec: P8ProgramSpec, tokens: int) -> int:
    """Arena bytes every call of 1 to ``tokens`` rows of ``spec`` fits.

    The arena of a launch grows with its row count and its tile, so the largest
    launch of each arm bounds it; a capacity therefore declares the largest
    arena any call it accepts uses, and every such call fits.
    """

    if int(tokens) < 1:
        raise ValueError("a capacity declares at least one row")
    largest = {}
    for rows in range(1, _capacity_row_limit(spec, tokens) + 1):
        materialized, small_m = select_arm(spec, rows)
        key = arm_key(spec, materialized, small_m, rows)
        largest[key] = rows
    return max(p8_scratch_layout_for(spec, rows).nbytes for rows in largest.values())
