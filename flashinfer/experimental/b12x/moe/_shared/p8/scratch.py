"""Caller-owned scratch layout of one P8 launch.

Every buffer a P8 call touches lives in one arena the caller owns. The plan
declares the arena's size from this layout (``p8_capacity_scratch_nbytes`` in
``plan``) and the runtime carves the named regions out of the bound buffer, so
no call allocates scratch: a call without a large enough arena raises instead.
The result is not part of the arena: a call writes a caller-owned output.

``p8_launch_layout`` is the single description of that arena. Both the plan and
the runtime call it with the same arguments, so the declared size and the
carved extents cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

__all__ = [
    "P8ScratchLayout",
    "P8ScratchRegion",
    "P8_ARENA_ALIGNMENT",
    "p8_clear_spans",
    "p8_launch_layout",
    "use_small_m",
]

#: Byte alignment every arena's start must carry. The launch creates each
#: region's pointer with ``assumed_align=16`` (``_gptr``), and every region
#: offset is 16-byte aligned, so a 16-byte aligned arena makes every pointer
#: carry the alignment the kernels assume.
P8_ARENA_ALIGNMENT = 16

#: Bytes per element of each region dtype.
_DTYPE_BYTES = {"uint8": 1, "int32": 4, "float32": 4, "bfloat16": 2}

#: Single int32 cells the kernel synchronizes through.
_SCALAR_CELLS = (
    "barrier_count",
    "barrier_epoch",
    "pair_head",
    "producers_done",
    "all_published",
    "task_head",
    "task_tail",
)

#: One int32 descriptor per (physical tile, 128-column gate slice).
_TASK_ARRAYS = (
    "task_ready",
    "task_expert",
    "task_m_tile",
    "task_slice_begin",
    "task_slice_count",
    "task_valid_rows",
)

#: Per-route FP32/BF16 partial sum the deterministic reducer consumes.
ROUTE_OUTPUT = "route_output"
#: Region the direct RP2 clear starts at; everything before it is overwritten
#: by the prologue when ``rp2_skip_input_clear`` applies.
INTERMEDIATE = "intermediate_u32"


@dataclass(frozen=True)
class P8ScratchRegion:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class P8ScratchLayout:
    regions: tuple[P8ScratchRegion, ...]
    nbytes: int

    def names(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.regions)

    def region(self, name: str) -> P8ScratchRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"the P8 scratch layout has no region {name!r}")

    def view(self, arena, name: str):
        """``arena``'s bytes of one region, as the dtype and shape it declares."""

        import torch

        region = self.region(name)
        return (
            arena.narrow(0, region.offset, region.nbytes)
            .view(getattr(torch, region.dtype))
            .reshape(region.shape)
        )


def use_small_m(enabled: bool, tokens: int) -> bool:
    """Whether the historical M1 owner serves ``tokens`` rows."""

    return bool(enabled and tokens == 1)


def p8_clear_spans(
    layout: P8ScratchLayout,
    *,
    start: int = 0,
    skip: tuple[str, ...] = (),
) -> tuple[tuple[int, int], ...]:
    """The byte spans of ``layout`` a call clears, as ``(offset, nbytes)``.

    The complement of ``skip`` from ``start``: a region named in ``skip`` is
    completely written by the kernels before anything reads it, so it is not
    cleared, and every other byte (regions, their alignment padding, and the
    arena's trailing padding) is. Regions are declared in offset order, so the
    result is one span per gap; a caller zeroes each span with one fill.
    """

    skipped = frozenset(skip)
    spans = []
    cursor = int(start)
    for region in layout.regions:
        if region.name not in skipped:
            continue
        if region.offset > cursor:
            spans.append((cursor, region.offset - cursor))
        cursor = max(cursor, region.offset + region.nbytes)
    if cursor < layout.nbytes:
        spans.append((cursor, layout.nbytes - cursor))
    return tuple(spans)


def p8_launch_layout(
    *,
    tokens: int,
    tile_m: int,
    direct: bool,
    intermediate: int = 512,
    hidden: int = 4096,
    experts: int = 288,
    topk: int = 8,
    planes: int = 1,
    compact_input: bool = False,
    grouped_scales: bool = False,
    deterministic_output: bool = False,
    coupled: bool = True,
    rp2_stride: int = 0,
) -> P8ScratchLayout:
    """Exact arena of one P8 launch of ``tokens`` rows.

    ``direct`` is the RP2 direct arm, which owns one route per physical tile;
    the grouped arm sizes its tiles from ``experts`` plus the routed rows.
    ``compact_input`` addresses the input carrier by token instead of by padded
    expert row and ``grouped_scales`` reads one activation-scale word per token
    instead of the per-tile plane; both are the full-coupled grouped arm's
    layouts. ``rp2_stride`` is the direct arm's residual-carrier stride: the
    prologue writes token ``t``'s residual plane at row ``rp2_stride + t`` of
    the input carrier, so the carrier holds both planes. ``deterministic_output``
    adds the per-route partial sum the reducer consumes, typed as ``coupled``
    runtime's carrier (FP32) or the identity-boundary runtime's (BF16).
    """

    tokens = int(tokens)
    tile_m = int(tile_m)
    if tokens < 1:
        raise ValueError("a P8 launch covers at least one row")
    if tile_m not in (16, 32, 64, 128):
        raise ValueError(f"unsupported P8 scratch tile size {tile_m}")
    routed = tokens * topk
    physical_tiles = routed if direct else experts + (routed + tile_m - 1) // tile_m
    rows = physical_tiles * tile_m
    max_tasks = physical_tiles * (intermediate // 128)
    input_rows = tokens if compact_input else rows
    if direct and rp2_stride:
        # The residual plane sits ``rp2_stride`` rows after the high plane, and
        # the carrier is addressed by token, so a one-row launch needs
        # ``rp2_stride + 1`` rows even though its tiles cover ``rows``.
        input_rows = max(input_rows, int(rp2_stride) + tokens)
    scale_elements = (
        tokens * (hidden // 32)
        if grouped_scales
        else (experts + routed + 1) * tile_m * (hidden // 8)
    )
    specs = [
        ("packed_a", "uint8", (input_rows * hidden,)),
        ("scale_flat", "uint8", (scale_elements,)),
        (
            INTERMEDIATE,
            "int32",
            (planes * rows * (intermediate + intermediate // 32) // 4,),
        ),
    ]
    specs.extend((name, "int32", (1,)) for name in _SCALAR_CELLS)
    specs.extend((name, "int32", (max_tasks,)) for name in _TASK_ARRAYS)
    specs.extend(
        [
            ("tile_write_count", "int32", (physical_tiles,)),
            ("row_counts", "int32", (experts,)),
            ("expert_write_rows", "int32", (experts,)),
            ("expert_tile_base", "int32", (experts + 1,)),
            ("token_map", "int32", (rows,)),
            ("token_weights", "float32", (rows,)),
        ]
    )
    if deterministic_output:
        specs.append(
            (ROUTE_OUTPUT, "float32" if coupled else "bfloat16", (routed, hidden))
        )
    regions = []
    cursor = 0
    for name, dtype, shape in specs:
        cursor = (cursor + 15) // 16 * 16
        nbytes = prod(shape) * _DTYPE_BYTES[dtype]
        regions.append(P8ScratchRegion(name, dtype, shape, cursor, nbytes))
        cursor += nbytes
    return P8ScratchLayout(tuple(regions), (cursor + 15) // 16 * 16)
