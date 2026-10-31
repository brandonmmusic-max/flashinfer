"""TrellisMX small-M arm (m <= 16) on b12x's W4A16 trellis path.

What the release runtime does at m <= 16, established by measurement and by
reading its image as the executable specification (nothing copied):

* the **A operand is not quantized** and the **FC2 input is not quantized** — the
  small-M arm's output matches the unquantized reference at 0.999998 on five
  independent captures (m = 1, 8, 16 mixed routing plus the 8-row single-route
  one), while the fully-quantized model gives 0.9989-0.9990 with a 12x larger
  ``max|d|``;
* the **per-route store is FP32** (the image's own ``route_output`` is
  ``(m*top_k, hidden) float32``);
* the weights are the released procedural-NPK trellis payloads with UE8M0/K32
  scales, decoded through the released **E4M3** alphabet (the same decode the
  grouped arm uses; the reference's ``npk_decode_states_reference`` path pins
  it) and widened to the W4A16 GEMM's fp16 operand, with the block scale
  applied per K32 block (a power of two, exact in fp16).

The one thing that is *not* shared with the grouped arm is the activation
path: A and the FC2 input stay 16-bit, which is what makes this arm the
small-M one.

This module owns two things:

1. ``packed_decode_npk_to_half2x4`` — the procedural-NPK law widened to fp16
   **with the alpha-2 compander**, which the existing ``mcg`` arm does not
   apply (``_lib/intrinsics/trellis.py`` ends at ``add.rn.f16x2 $0, lo, hi``).
   ``gemm.py``'s ``procedural_npk`` branch calls it.
2. ``TrellisMxSmallM`` — the arm itself, built from pieces that already exist:

   * input boundary: the fused boundary kernel in ``mode="fp16"``
     (``trellismx_boundary_fused.TrellisMxFusedBoundary``), bit-exact to
     ``input_boundary(...).to(torch.float16)``;
   * FC1 + the coupled intermediate boundary + FC2: the W4A16 fused trellis
     kernel with ``external_input_boundary=True`` (the caller owns the input
     stage), ``full_rotation=True`` and ``intermediate_hadamard=True``, which is
     the release's H128/svh/sign/silu-cap10/down_suh chain;
   * output: the kernel's own FP32 top-k sum (router order) followed by the
     per-route H128, ``down_svh`` and the outer H512 — the broadcast-``svh``
     branch of ``W4A16TopKSumKernel``.

Every stage is a device kernel writing into caller-owned buffers, so the whole
arm captures into one CUDA graph and replays without a host sync.

The arm owns no mutable buffer. ``TrellisMxSmallM.plan`` declares a
:class:`SmallMScratchLayout` (every byte the arm writes, at a fixed offset in
one arena) and the compiled programs; ``bind_scratch`` carves the buffers out of
a caller-owned arena, and ``run`` writes only those. Two bindings of one plan
therefore never share storage, and no call allocates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, Uint32, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics.trellis import packed_dequant_trellis_to_half2x4
from b12x._lib.program_cache import program_cache

__all__ = [
    "TrellisMxSmallM",
    "TrellisMxSmallMPlan",
    "SmallMScratch",
    "SmallMScratchLayout",
    "SmallMScratchSpec",
    "carve_smallm_scratch",
    "packed_decode_npk_to_half2x4",
    "smallm_scratch_layout",
    "SMALLM_MAX_TOKENS",
]

#: The largest m the arm serves, 16 like the release runtime's direct limit. This
#: is the arm's own bound: its boundary and fused launches are compiled per token
#: count over 1..SMALLM_MAX_TOKENS. How much of that range a prepared plan hands
#: to the arm is the plan's declared ``smallm_max_tokens``
#: (``fused_moe.plan_execution``), never a process-wide default.
SMALLM_MAX_TOKENS = 16

#: The release's activation limit (``silu-cap10``).
SMALLM_SWIGLU_LIMIT = 10.0

#: The prepared weight layout the arm's scratch is sized for. Preparation sizes
#: the arena from this geometry alone, so an arm plan built on another layout
#: must refuse rather than carve views that do not fit.
_ARM_WEIGHT_LAYOUT = "trellis_t256"


@dsl_user_op
def _double_half2x4(a0, a1, a2, a3, *, loc=None, ip=None):
    """Apply the alpha-2 compander to four packed half2 words: h = h + h."""

    asm = """
        {
            add.rn.f16x2 $0, $4, $4;
            add.rn.f16x2 $1, $5, $5;
            add.rn.f16x2 $2, $6, $6;
            add.rn.f16x2 $3, $7, $7;
        }
        """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@cute.jit
def packed_decode_npk_to_half2x4(
    win_a: Uint32,
    win_b: Uint32,
    bits: cutlass.Constexpr,
):
    """Decode eight procedural-NPK states to four packed fp16 pairs, compander applied.

    Same window geometry, multiplier, mask and OR as the ``mcg`` arm
    (``packed_dequant_trellis_to_half2x4``), plus the alpha-2 compander ``h = h + h``
    that the TrellisMX law declares and
    that the ``mcg`` arm does not apply.

    This is the *law widened to fp16*, i.e. the pre-quantization value. The
    released alphabet quantizes it to E4M3, and that is what the arm's weights
    use (``w4a8_npk_decode.packed_decode_npk_to_e4m3x8``); this helper exists to
    pin the law itself (``test_trellismx_smallm.py``).
    """

    o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(win_a, win_b, int(bits))
    return _double_half2x4(o0, o1, o2, o3)


# --------------------------------------------------------------------------- #
# bundle extraction
# --------------------------------------------------------------------------- #
_BOUNDARY_KEYS = (
    "gate_up_suh_fp16",
    "down_svh_fp16",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
)


def _boundary_vectors(
    experts, tensors, *, hidden_size, intermediate_size, num_experts, device
):
    """Return ``(gate_up_suh, down_svh, intermediate_rotations)`` for the layer.

    The grouped arm's prepared bundle already carries the coupled boundary: its
    ``representation.value`` is the TrellisMX prepared-weight record with the
    released ``gate_suh``/``up_suh``/``down_svh`` vectors and the fp16
    ``intermediate_rotations`` set (values followed by the appended per-expert
    sign draw), so the arm reads the *same* tensors the grouped arm uses. A
    sidecar ``tensors`` mapping is accepted as well for direct callers.
    """

    value = getattr(getattr(experts._impl, "representation", None), "value", None)
    if value is not None and getattr(value, "intermediate_rotations", None) is not None:
        gate_up_suh = torch.as_tensor(value.gate_suh).to(device)
        down_svh = torch.as_tensor(value.down_svh).to(device)
        rotations = torch.as_tensor(value.intermediate_rotations).to(device)
    else:
        if tensors is None:
            raise ValueError(
                "the prepared bundle does not expose the TrellisMX coupled "
                "boundary (representation.value.intermediate_rotations); pass "
                "tensors=... from the released sidecar"
            )
        missing = [name for name in _BOUNDARY_KEYS if name not in tensors]
        if missing:
            raise ValueError(
                f"the TrellisMX boundary mapping is missing tensors: {missing}"
            )
        from b12x.testing.reference.trellismx_moe import rank_local_coupled_signs

        gate_up_suh = torch.as_tensor(tensors["gate_up_suh_fp16"]).to(device)
        down_svh = torch.as_tensor(tensors["down_svh_fp16"]).to(device)
        intermediate_scales = torch.as_tensor(tensors["intermediate_scales_fp16"]).to(
            device
        )
        draws = torch.unique(torch.as_tensor(tensors["coupled_sign_draw_u8"]))
        if draws.numel() != 1:
            raise ValueError(
                "the small-M arm implements one coupled-sign draw per layer; "
                f"coupled_sign_draw_u8 holds {sorted(int(v) for v in draws)}"
            )
        signs = rank_local_coupled_signs(
            intermediate=intermediate_size,
            rank=0,
            world_size=4,
            draw=int(draws[0]),
        ).to(device)
        rotations = torch.cat(
            (intermediate_scales.reshape(num_experts, 3 * intermediate_size), signs),
            dim=1,
        )

    if tuple(gate_up_suh.shape) not in ((hidden_size,), (1, hidden_size)):
        raise ValueError(
            f"gate_up_suh must be [{hidden_size}] or [1, {hidden_size}]; got "
            f"{tuple(gate_up_suh.shape)}"
        )
    if tuple(down_svh.shape) not in ((hidden_size,), (1, hidden_size)):
        raise ValueError(
            f"down_svh must be [{hidden_size}] or [1, {hidden_size}]; got "
            f"{tuple(down_svh.shape)}"
        )
    if tuple(rotations.shape) != (num_experts, 6 * intermediate_size):
        raise ValueError(
            "intermediate_rotations must be "
            f"[{num_experts}, {6 * intermediate_size}]; got {tuple(rotations.shape)}"
        )
    return (
        gate_up_suh.reshape(1, hidden_size).contiguous(),
        down_svh.reshape(1, hidden_size).contiguous(),
        rotations.contiguous(),
    )


# --------------------------------------------------------------------------- #
# caller-owned scratch
# --------------------------------------------------------------------------- #
#: The arm's scratch is one arena of 16-byte-aligned regions: every launch
#: argument takes a 16-byte aligned pointer, so each region starts on one.
_SCRATCH_ALIGNMENT = 16


def _align_up(value: int, alignment: int = _SCRATCH_ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class SmallMScratchSpec:
    """One buffer of the arm's arena: its shape, dtype and byte offset."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset_bytes: int

    @property
    def numel(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= int(dim)
        return numel

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize


@dataclass(frozen=True)
class SmallMScratchLayout:
    """Where the arm's buffers live in a caller-owned arena, without allocating.

    The layout is a pure function of the geometry the arm is planned for
    (``smallm_scratch_layout``), so preparation can size the fused-MoE arena
    from it and the arm plan can carve the same regions from that arena later.
    """

    specs: tuple[SmallMScratchSpec, ...]
    #: total bytes the arena must hold, rounded up to the region alignment
    nbytes: int
    block_size_m: int
    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    num_experts: int


@dataclass(frozen=True)
class SmallMScratch:
    """The arm's buffers, carved from a caller-owned arena.

    ``arena`` keeps the storage alive; every buffer below is a view into it.
    ``layout`` names the geometry the views were carved for, so a launch can
    refuse a scratch plan that belongs to another plan.
    """

    layout: SmallMScratchLayout
    arena: torch.Tensor
    boundary_payload: torch.Tensor
    boundary_scales: torch.Tensor
    x_fp16: torch.Tensor
    out_f32: torch.Tensor
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.arena.device


def smallm_scratch_layout(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    max_tokens: int,
    sms: int,
    weight_layout: str = _ARM_WEIGHT_LAYOUT,
    is_gated: bool = True,
) -> SmallMScratchLayout:
    """Every byte the arm writes for one geometry, at its arena offset.

    The shapes are the ones ``make_w4a16_packed_buffers`` allocates for the same
    geometry; this is that call's plan without the allocation, so a caller can
    size an arena before any device work and the arm plan carves exactly it.
    ``sms`` is the device's SM count, which bounds the packed GEMM scratch.
    """

    from types import SimpleNamespace

    from b12x.moe._shared.kernels.w4a16.host import (
        plan_w4a16_buffers,
        select_route_block_size_m,
    )

    hidden = int(hidden_size)
    intermediate = int(intermediate_size)
    experts = int(num_experts)
    top_k = int(top_k)
    max_tokens = int(max_tokens)
    block_size_m = select_route_block_size_m(max_tokens, top_k, experts)
    plan = plan_w4a16_buffers(
        SimpleNamespace(
            num_experts=experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
            is_gated=bool(is_gated),
            weight_layout=str(weight_layout),
        ),
        m=max_tokens,
        topk=top_k,
        route_num_experts=experts,
        sms=int(sms),
        dtype=torch.float16,
        full_rotation=True,
        block_size_m=block_size_m,
    )
    shapes = (
        # The fused boundary's quantized input payload and its UE8M0 K/32 plane.
        ("boundary_payload", (max_tokens, hidden), torch.uint8),
        ("boundary_scales", (max_tokens, hidden // 32), torch.uint8),
        ("x_fp16", (max_tokens, hidden), torch.float16),
        ("out_f32", (max_tokens, hidden), torch.float32),
        (
            "intermediate_cache13",
            (plan.intermediate_cache13_elements,),
            torch.float16,
        ),
        (
            "intermediate_cache2",
            (
                plan.routed_rows,
                plan.intermediate_cache2_elements // plan.routed_rows,
            ),
            torch.float16,
        ),
        ("fc1_c_tmp", (plan.fc1_c_tmp_elements,), torch.float32),
        ("fc2_c_tmp", (plan.fc2_c_tmp_elements,), torch.float32),
        ("packed_route_indices", (plan.route_slots,), torch.int32),
        ("block_expert_ids", (plan.route_blocks,), torch.int32),
        ("packed_route_count", (1,), torch.int32),
        ("expert_offsets", (experts + 1,), torch.int32),
        ("expert_counts", (experts,), torch.int32),
    )
    specs = []
    offset = 0
    for name, shape, dtype in shapes:
        offset = _align_up(offset)
        spec = SmallMScratchSpec(
            name=name,
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
            offset_bytes=offset,
        )
        specs.append(spec)
        offset += spec.nbytes
    return SmallMScratchLayout(
        specs=tuple(specs),
        nbytes=_align_up(offset),
        block_size_m=block_size_m,
        max_tokens=max_tokens,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
    )


def carve_smallm_scratch(
    arena: torch.Tensor,
    layout: SmallMScratchLayout,
    *,
    offset_bytes: int = 0,
) -> SmallMScratch:
    """View ``layout``'s buffers in ``arena``, starting at ``offset_bytes``.

    The arena is the caller's: this allocates no device memory. A missing arena,
    a non-byte arena, an unaligned base or a region shorter than the layout
    raises instead of launching the arm over storage it does not own.
    """

    if arena is None:
        raise RuntimeError(
            "the TrellisMX small-M arm takes caller-owned scratch: bind a "
            f"buffer of at least {int(layout.nbytes)} bytes"
        )
    if not isinstance(arena, torch.Tensor):
        raise TypeError(
            f"the small-M arm's scratch must be a tensor; got {type(arena).__name__}"
        )
    if arena.dtype != torch.uint8:
        raise TypeError(f"the small-M arm's scratch must be uint8; got {arena.dtype}")
    if not arena.is_contiguous():
        raise ValueError("the small-M arm's scratch must be contiguous")
    offset_bytes = int(offset_bytes)
    if offset_bytes < 0:
        raise ValueError(f"scratch offset must be non-negative; got {offset_bytes}")
    if (arena.data_ptr() + offset_bytes) % _SCRATCH_ALIGNMENT:
        raise ValueError("the small-M arm's scratch must start on a 16-byte boundary")
    available = int(arena.numel()) * arena.element_size() - offset_bytes
    if available < int(layout.nbytes):
        raise ValueError(
            f"the small-M arm's scratch has {available} bytes from offset "
            f"{offset_bytes}, requires {int(layout.nbytes)}"
        )
    flat = arena.reshape(-1)
    views = {
        spec.name: flat[
            offset_bytes + spec.offset_bytes : offset_bytes
            + spec.offset_bytes
            + spec.nbytes
        ]
        .view(spec.dtype)
        .view(spec.shape)
        for spec in layout.specs
    }
    return SmallMScratch(layout=layout, arena=arena, **views)


# --------------------------------------------------------------------------- #
# plan / run
# --------------------------------------------------------------------------- #
@dataclass
class TrellisMxSmallMPlan:
    """One prepared small-M layer: its immutable weights and compiled programs.

    The plan owns no mutable buffer. ``layout`` declares every byte a call
    writes, ``workspace_bytes`` reports the arena that must back it, and
    ``bind_scratch`` carves the buffers out of a caller-owned arena. The plan
    views the bundle's tensors but holds no reference to the bundle object, so a
    bundle that caches its plans does not keep itself alive.
    """

    prepared: object
    fused: object
    boundary: object
    layout: SmallMScratchLayout
    intermediate_rotations: torch.Tensor
    gate_up_suh: torch.Tensor
    down_svh: torch.Tensor
    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    block_size_m: int
    device: torch.device
    #: compiled ``(m -> callable)`` cache for the fused input boundary
    boundary_compiled: dict = field(default_factory=dict)
    #: ``boundary_compiled`` and ``route_pack`` are prepared for every call the
    #: arm accepts, so neither a call nor a capture compiles anything.
    route_pack: object = None

    @property
    def workspace_bytes(self) -> int:
        """Bytes of caller-owned arena one call needs."""

        return int(self.layout.nbytes)

    def bind_scratch(
        self, scratch: torch.Tensor, *, offset_bytes: int = 0
    ) -> SmallMScratch:
        """Carve this plan's buffers out of the caller's arena.

        Allocates nothing. ``offset_bytes`` places the arm's region inside a
        larger arena (the fused-MoE scratch buffer does); the region must be
        16-byte aligned and at least ``workspace_bytes`` long.
        """

        carved = carve_smallm_scratch(scratch, self.layout, offset_bytes=offset_bytes)
        if carved.device != self.device:
            raise ValueError(
                f"the small-M arm's scratch lives on {carved.device}; the plan "
                f"on {self.device}"
            )
        return carved


# The arm's boundary is the grouped arm's input stage in fp16 mode, compiled
# per (hidden, m) from shapes alone: no weight is read and nothing is
# allocated, so preparation's discovery can build it in workers that hide CUDA.
def _compile_smallm_boundary(hidden: int, m: int, device: torch.device):
    """The fused boundary in ``mode="fp16"`` for ``m`` tokens, from shapes alone.

    The kernel, its operands and its compile identity are the grouped arm's, so
    the arm's input stage is byte-for-byte the one the fused kernel produces.
    """
    if not 1 <= int(m) <= SMALLM_MAX_TOKENS:
        raise ValueError(
            f"the small-M boundary covers 1..{SMALLM_MAX_TOKENS} tokens; got {m}"
        )
    from b12x.moe._shared.kernels.trellismx_boundary_fused import (
        BOUNDARY_KERNEL_VERSION,
        TrellisMxFusedBoundary,
    )

    from b12x._lib.utils.device import current_cuda_stream
    from cutlass.cute.runtime import make_fake_compact_tensor

    # Fake compact tensors: preparation's discovery compiles this where the
    # CUDA runtime is hidden, so nothing here may touch a device allocation.
    row_major = (1, 0)
    with torch.cuda.device(device):
        return b12x_compile(
            TrellisMxFusedBoundary(hidden, "fp16"),
            make_fake_compact_tensor(
                cutlass.BFloat16, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            make_fake_compact_tensor(
                cutlass.Float16, (1, hidden), stride_order=row_major, assumed_align=16
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (m, hidden // 32),
                stride_order=row_major,
                assumed_align=16,
            ),
            make_fake_compact_tensor(
                cutlass.Float16, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            current_cuda_stream(),
            # The boundary kernel's own compile identity (BOUNDARY_KERNEL_VERSION).
            compile_spec=KernelCompileSpec.from_key(
                _BOUNDARY_KEY,
                BOUNDARY_KERNEL_VERSION,
                (m, hidden),
                labels=("m", "hidden"),
            ),
        )


@dataclass(frozen=True)
class SmallMPrograms:
    """Every program the arm launches, for one rank-slice geometry."""

    block_size_m: int
    fused: object
    topk_sum: dict
    route_pack: object
    boundary: dict


#: Compile-spec identity of the fused boundary program: every (m, hidden) the
#: arm compiles is one specialization of this kernel, versioned by the kernel.
_BOUNDARY_KEY = "moe.trellismx.smallm.input_boundary_fp16"


@program_cache(scope="preparation")
def compile_smallm_programs(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    bits: int,
    device: torch.device,
    max_tokens: int,
):
    """Every program the arm launches for one rank-slice geometry, from
    metadata alone: no weights are read, so b12x's preparation can compile
    them in workers that reject CUDA. The fused launch, the top-k sum, the
    route pack and the fused boundary are all compiled here, and each takes m
    as a launch argument, so their first call in a
    no-compilation phase never meets one of them (an export that freezes
    kernel resolution rejects lazily compiled programs).
    """

    from b12x._lib.compile_plan import attach_programs
    from b12x.moe._shared.kernels.w4a16.host import (
        route_pack_capacity,
        select_route_block_size_m,
        trellis_decode_table,
    )
    from b12x.moe._shared.kernels.w4a16.kernel.config import _DEFAULT_MAX_SHARED_MEM
    from b12x.moe._shared.kernels.w4a16.kernel.fused_launch import (
        compile_w4a16_fused_moe,
    )
    from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import compile_w4a16_topk_sum
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    hidden = int(hidden_size)
    intermediate = int(intermediate_size)
    block_size_m = select_route_block_size_m(max_tokens, top_k, num_experts)
    # The route arena is sized for the whole capacity, so every call packs into
    # the same blocks whatever its row count.
    capacity_m_blocks = route_pack_capacity(
        max_tokens * top_k,
        block_size_m,
        num_experts,
        topk=top_k,
        bucket_tokens=False,
    )[2]
    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        )

        def compile_launch(rows, tile_config=None):
            return compile_w4a16_fused_moe(
                size_m=rows,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=num_experts,
                top_k=top_k,
                activation="silu",
                apply_router_weight_on_input=False,
                zero_fc2_output=False,
                moe_block_size=block_size_m,
                max_m_blocks=capacity_m_blocks,
                element_dtype="fp16",
                fast_math=True,
                sms=sms,
                max_shared_mem=max_shared_mem,
                swiglu_limit=SMALLM_SWIGLU_LIMIT,
                weight_layout="trellis_t256",
                scale_format="e4m3_k32",
                w13_layout="trellis_t256_proj",
                trellis_bits=bits,
                trellis_codebook="procedural_npk",
                trellis_decode_table=trellis_decode_table(),
                # The gated FC1 payload's N axis is gate-up and the released scale
                # plane is up-gate, so the lookup rotates the plane by I.
                trellis_weight_scale_row_rotation=intermediate,
                intermediate_rotation=True,
                full_rotation=True,
                external_input_boundary=True,
                intermediate_hadamard=True,
                rotation_input_dtype="fp16",
                broadcast_suh=True,
                force_tile_config=tile_config,
            )

        # The launch is specialized per token count (its occupancy and grid
        # bounds follow the live M) while the prepared weights pin one tile
        # geometry for every one of them, so the largest count picks it.
        reference = compile_launch(max_tokens)
        tile_config = (
            reference.fc1_tile_k,
            reference.fc1_tile_n,
            reference.fc2_tile_k,
            reference.fc2_tile_n,
        )
        fused = {
            rows: compile_launch(rows, tile_config) for rows in range(1, max_tokens + 1)
        }
        topk_sum = {
            dtype: compile_w4a16_topk_sum(
                m=max_tokens,
                topk=top_k,
                hidden_size=hidden,
                element_dtype="fp16",
                full_rotation=True,
                intermediate_hadamard=True,
                num_experts=num_experts,
                route_num_experts=0,
                route_ids_dtype=dtype,
                use_expert_map=False,
                broadcast_svh=True,
                float32_output=True,
            )
            for dtype in (torch.int32, torch.int64)
        }
        route_pack = compile_w4a16_route_pack_launches(
            tokens=max_tokens,
            topk=top_k,
            block_size=block_size_m,
            num_experts=num_experts,
            ordinal=device.index
            if device.index is not None
            else torch.cuda.current_device(),
        )
        boundary = {
            # The boundary's shapes are traced from m, so every token count the
            # arm accepts is its own program.
            m: _compile_smallm_boundary(hidden, m, device)
            for m in range(1, max_tokens + 1)
        }
    programs = attach_programs(
        SmallMPrograms(
            block_size_m=block_size_m,
            fused=fused,
            topk_sum=topk_sum,
            route_pack=route_pack,
            boundary=boundary,
        ),
        fused,
        topk_sum,
        route_pack.carriers(),
        boundary,
    )
    # The carriers a compile factory must return, with the programs attached.
    return programs


def _run_input_boundary(
    plan: "TrellisMxSmallMPlan", scratch: "SmallMScratch", x_bf16, m: int
) -> None:
    """The fused boundary in ``mode="fp16"``, compiled once per token count."""

    from b12x._lib.utils.device import current_cuda_stream
    from cutlass.cute.runtime import from_dlpack

    device = scratch.x_fp16.device

    def args():
        return (
            from_dlpack(x_bf16, assumed_align=16),
            from_dlpack(plan.gate_up_suh, assumed_align=16),
            from_dlpack(scratch.boundary_payload[:m], assumed_align=16),
            from_dlpack(scratch.boundary_scales[:m], assumed_align=16),
            from_dlpack(scratch.x_fp16[:m], assumed_align=16),
            current_cuda_stream(),
        )

    compiled = plan.boundary_compiled.get(m)
    if compiled is None:
        compiled = _compile_smallm_boundary(plan.hidden_size, m, device)
        plan.boundary_compiled[m] = compiled
    compiled(*args())


class TrellisMxSmallM:
    """The small-M arm for one TrellisMX rank slice, 1 <= m <= 16."""

    @classmethod
    def plan(
        cls,
        experts,
        *,
        max_tokens: int,
        top_k: int,
        tensors=None,
        meta=None,
    ) -> TrellisMxSmallMPlan:
        """Prepare the arm from the grouped arm's prepared TrellisMX bundle.

        ``experts`` is the ``PreparedExperts`` returned by
        ``prepare_trellismx_experts``. The payload and both UE8M0 planes are
        read from it **as views** (no bytes copied); the boundary vectors come
        from its prepared record
        (``experts._impl.representation.value``: ``gate_suh``, ``down_svh`` and
        ``intermediate_rotations``), or from a sidecar ``tensors`` mapping when
        one is supplied.

        Planning allocates the arm's immutable weights and resolves its
        programs; the mutable buffers a call writes are declared by
        ``layout`` and come from the caller's arena (``bind_scratch``).
        """

        from b12x.moe._shared.kernels.trellismx_boundary_fused import (
            TrellisMxFusedBoundary,
        )
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_trellis256_moe_weights,
        )

        if not 1 <= max_tokens <= SMALLM_MAX_TOKENS:
            raise ValueError(
                f"the small-M arm owns 1..{SMALLM_MAX_TOKENS} tokens; got "
                f"{max_tokens} (larger m belongs to the grouped arm)"
            )
        impl = experts._impl
        if str(getattr(impl, "source_format", "")) != "trellismx":
            raise ValueError(
                "the small-M arm requires the prepared TrellisMX bundle; got "
                f"source_format={getattr(impl, 'source_format', None)!r}"
            )
        hidden = int(impl.hidden_size)
        intermediate = int(impl.intermediate_size)
        num_experts = int(impl.num_experts)
        top_k = int(top_k)
        bits = int(impl.plan.trellis_bits)
        device = impl.w1_fp4.device
        if device.type != "cuda":
            raise ValueError("the small-M arm requires a CUDA prepared bundle")

        gate_up_suh, down_svh, rotations = _boundary_vectors(
            experts,
            tensors,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            device=device,
        )

        # The arm's buffers, kernels and captured shapes all follow one
        # geometry: the tile config the fused launch resolves and the prepared
        # weight
        # layout, and preparation must agree with them.
        # Compile from metadata alone: no weight is read here, so this is the
        # same call discovery makes and the keys it produces are the keys the
        # runtime's launches resolve to.
        programs = compile_smallm_programs(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            bits=bits,
            device=device,
            max_tokens=max_tokens,
        )
        fused = programs.fused[max_tokens]
        block_size_m = int(programs.block_size_m)
        # The decode's process-lifetime device value table is built on first
        # use. Build it here, in preparation: the arm's first launch may be the
        # captured one, and a capture must not copy or allocate.
        from b12x.moe._shared.kernels.w4a16.kernel.config import (
            _trellis256_execution_lut,
        )

        _trellis256_execution_lut(
            device, "procedural_npk", direct_lut=bool(fused.trellis_direct_lut)
        )

        prepared = prepare_trellis256_moe_weights(
            w13=impl.w1_fp4,
            w2=impl.w2_fp4,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            activation="silu",
            fc1_tile_n=int(fused.fc1_tile_n),
            fc2_tile_n=int(fused.fc2_tile_n),
            params_dtype=torch.float16,
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            codebook="procedural_npk",
            w13_block_scale=impl.w1_blockscale,
            w2_block_scale=impl.w2_blockscale,
            gate_suh=gate_up_suh,
            up_suh=gate_up_suh,
            # Preparation takes the three value blocks [gate_svh|up_svh|down_suh];
            # the coupled-boundary sign rows are appended below, exactly as the
            # grouped preparation does (`trellis.py:_uniform_prepared`).
            intermediate_rotations=rotations[:, : 3 * intermediate].contiguous(),
            down_svh=down_svh,
            tile_config=(
                int(fused.fc1_tile_k),
                int(fused.fc1_tile_n),
                int(fused.fc2_tile_k),
                int(fused.fc2_tile_n),
            ),
        )
        from dataclasses import replace as _replace

        prepared = _replace(
            prepared,
            trellis=_replace(
                prepared.trellis,
                intermediate_hadamard=True,
                intermediate_rotations=rotations,
            ),
        )

        # The buffers a call writes are the caller's; the plan only declares
        # their geometry. It is the same buffer plan the arm's packed launch
        # allocates for, read from metadata: preparation sizes the fused-MoE
        # arena from it and every binding carves its own views out of that.
        # Preparation sizes that arena from the geometry alone, so a record
        # with another weight layout or activation would carve views it never
        # reserved.
        weight_layout = str(getattr(prepared, "weight_layout", ""))
        if weight_layout != _ARM_WEIGHT_LAYOUT or not bool(prepared.is_gated):
            raise RuntimeError(
                "the small-M arm's scratch is sized for a gated "
                f"{_ARM_WEIGHT_LAYOUT} record; got weight_layout="
                f"{weight_layout!r} is_gated={bool(prepared.is_gated)}"
            )
        layout = smallm_scratch_layout(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            max_tokens=max_tokens,
            sms=int(torch.cuda.get_device_properties(device).multi_processor_count),
            weight_layout=weight_layout,
            is_gated=bool(prepared.is_gated),
        )
        boundary = TrellisMxFusedBoundary(hidden, mode="fp16")
        plan = TrellisMxSmallMPlan(
            prepared=prepared,
            fused=fused,
            boundary=boundary,
            layout=layout,
            intermediate_rotations=prepared.intermediate_rotations,
            gate_up_suh=gate_up_suh.reshape(1, hidden).contiguous(),
            down_svh=down_svh.reshape(1, hidden).contiguous(),
            max_tokens=max_tokens,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            block_size_m=block_size_m,
            device=device,
            # The boundary is compiled for every token count the arm accepts
            # and the route pack for the whole arena, so a call at any m finds
            # its programs already compiled.
            boundary_compiled=dict(programs.boundary),
            route_pack=programs.route_pack,
        )
        return plan

    @staticmethod
    def run(x_bf16, topk_ids, topk_weights, out_bf16, workspace, scratch=None) -> None:
        """Run one rank slice for ``1 <= m <= 16``, writing BF16 into ``out``.

        ``workspace`` is the :class:`TrellisMxSmallMPlan` returned by ``plan``
        and ``scratch`` the :class:`SmallMScratch` that plan carved from the
        caller's arena (``plan.bind_scratch``). Every launch writes caller-owned
        storage, so the whole call is a single CUDA-graph capture region with no
        host synchronization and no allocation.

        The plan carries a compiled input boundary, route pack and top-k sum
        for every accepted token count. A call at any ``1 <= m <= max_tokens``
        launches those programs and compiles nothing.
        """

        from b12x.moe._shared.kernels.w4a16.kernel.execution import run_w4a16_moe

        plan = workspace
        if not isinstance(plan, TrellisMxSmallMPlan):
            raise TypeError(
                "TrellisMxSmallM.run takes the plan returned by "
                f"TrellisMxSmallM.plan; got {type(plan).__name__}"
            )
        if scratch is None:
            raise RuntimeError(
                "the TrellisMX small-M arm takes caller-owned scratch: bind "
                f"{plan.workspace_bytes} bytes with plan.bind_scratch before "
                "running it"
            )
        if not isinstance(scratch, SmallMScratch):
            raise TypeError(
                "TrellisMxSmallM.run takes the scratch carved by "
                f"plan.bind_scratch; got {type(scratch).__name__}"
            )
        if scratch.layout != plan.layout:
            raise ValueError(
                "the small-M arm's scratch was carved for another geometry "
                f"(arena {scratch.layout.nbytes} bytes for m <= "
                f"{scratch.layout.max_tokens}; the plan serves m <= "
                f"{plan.max_tokens} in {plan.workspace_bytes} bytes)"
            )
        # The layout compares equal across devices, so check the device itself:
        # every launch mixes the scratch with the plan's immutable device
        # tensors, and a caller that carved its own arena bypasses bind_scratch.
        if scratch.device != plan.device:
            raise ValueError(
                f"the small-M arm's scratch lives on {scratch.device}; the plan "
                f"on {plan.device}"
            )
        m = int(x_bf16.shape[0])
        hidden = plan.hidden_size
        if x_bf16.dtype != torch.bfloat16:
            raise ValueError(f"TrellisMX input must be bfloat16, got {x_bf16.dtype}")
        if tuple(x_bf16.shape) != (m, hidden):
            raise ValueError(f"input must be [m, {hidden}]; got {tuple(x_bf16.shape)}")
        if not 1 <= m <= plan.max_tokens:
            raise ValueError(
                f"the small-M arm is planned for at most {plan.max_tokens} tokens; "
                f"got m={m}"
            )
        if tuple(topk_ids.shape) != (m, plan.top_k) or tuple(topk_weights.shape) != (
            m,
            plan.top_k,
        ):
            raise ValueError(
                f"routing must be [m, {plan.top_k}]; got ids "
                f"{tuple(topk_ids.shape)} weights {tuple(topk_weights.shape)}"
            )
        if tuple(out_bf16.shape) != (m, hidden) or out_bf16.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be bfloat16 [m, {hidden}]; got "
                f"{out_bf16.dtype} {tuple(out_bf16.shape)}"
            )
        device = plan.device
        # Both tensors reach the kernels as 16-byte aligned pointers
        # (``from_dlpack(..., assumed_align=16)`` in the boundary and the output
        # copy), the same contract ``_validate_trellismx_binding`` enforces.
        if (
            not x_bf16.is_contiguous()
            or x_bf16.device != device
            or not out_bf16.is_contiguous()
            or out_bf16.device != device
            or x_bf16.data_ptr() % 16
            or out_bf16.data_ptr() % 16
        ):
            raise ValueError(
                "input and output must be contiguous, 16-byte aligned tensors on "
                f"the plan's device ({device})"
            )
        if (
            topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.device != device
            or not topk_ids.is_contiguous()
            or topk_weights.dtype != torch.float32
            or topk_weights.device != device
            or not topk_weights.is_contiguous()
        ):
            raise ValueError(
                "routing must be contiguous on the plan's device, with int32/int64 "
                f"ids and float32 weights; got ids {topk_ids.dtype} on "
                f"{topk_ids.device}, weights {topk_weights.dtype} on "
                f"{topk_weights.device}"
            )
        ids = topk_ids
        weights = topk_weights

        _run_input_boundary(plan, scratch, x_bf16, m)
        run_w4a16_moe(
            scratch.x_fp16[:m],
            plan.prepared,
            weights,
            ids,
            activation="silu",
            intermediate_cache13=scratch.intermediate_cache13,
            intermediate_cache2=scratch.intermediate_cache2,
            output=scratch.out_f32[:m],
            fc1_c_tmp=scratch.fc1_c_tmp,
            fc2_c_tmp=scratch.fc2_c_tmp,
            packed_route_indices=scratch.packed_route_indices,
            block_expert_ids=scratch.block_expert_ids,
            packed_route_count=scratch.packed_route_count,
            expert_offsets=scratch.expert_offsets,
            expert_counts=scratch.expert_counts,
            fast_math=True,
            # The arm's route arena is sized for the plan's whole capacity, so
            # the packer always runs; a direct-topk route would ignore it.
            route_mode="packed",
            route_pack_launches=plan.route_pack,
            # The release's activation limit, for both arms (silu-cap10).
            swiglu_limit=SMALLM_SWIGLU_LIMIT,
            route_block_size_m=plan.block_size_m,
            intermediate_rotation_scales=plan.intermediate_rotations,
            full_rotation=True,
            external_input_boundary=True,
            suh_gate_table=plan.gate_up_suh,
            suh_up_table=plan.gate_up_suh,
            svh_table=plan.down_svh,
        )
        out_bf16.copy_(scratch.out_f32[:m])
