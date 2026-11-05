"""P8 specialization of b12x's shared dynamic MoE kernel.

The kernel body is :class:`b12x.moe._shared.kernels.dynamic.kernel.MoEDynamicKernelBackend`;
this module holds only what the P8 arms add on top of it:

* the exact input boundary (BF16 -> FP16 -> H512 -> signed suh -> H128, then
  K32 amax/E4M3/UE8M0) for the M1 and grouped producers, including the D-x2
  and D-x2-RP variants a diagnostic subclass turns on;
* the phase-1/phase-2 owner classes that implement that boundary in the
  materialized arms;
* the two class-level selections that pin the P8 routing contract and the
  two-input-warp producer split.

Everything else - routing, task publication, the mainloop, the epilogue, the
scatter, the work sources - is the shared module's code, so a fix there
reaches the P8 arms without a hand port. The P8 programs' bytes are pinned by
the serving digests in the runtime tests, and this module is named in the
launch's ``__b12x_source_dependencies__`` so editing it invalidates them.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    T,
    Uint8,
    Uint32,
    Uint64,
    dsl_user_op,
)

from b12x._lib.intrinsics.atomics import ld_shared_u32
from b12x._lib.intrinsics.conversion import (
    cvt_e4m3x4_to_f32x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import div_rn_f32, fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    st_global_f32,
    st_global_u64,
)
from b12x._lib.intrinsics.quantization import quantize_block_fp8_mx
from b12x.moe._shared.kernels.dynamic.kernel import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_trellis_permute_k32,
)

from b12x.moe._shared.p8.kernels.w4a8_phase1 import (
    P8W4A8MaterializedPhase1Kernel,
)
from b12x.moe._shared.p8.kernels.w4a8_phase2 import (
    P8W4A8MaterializedPhase2Kernel,
)

# The phase owners are imported inside ``_owner_phase_kernels`` so a recipe
# that is not a P8 arm never loads them; name them for the compile cache's
# dependency tracer.
__b12x_source_dependencies__ = (
    "b12x.moe._shared.p8.kernels.fc2_k5_funnel",
    "b12x.moe._shared.p8.kernels.direct_m32_fc2",
    "b12x.moe._shared.p8.kernels.policy_m32_fc1",
    "b12x.moe._shared.p8.kernels.policy_m32_fc2",
    "b12x.moe._shared.p8.kernels.route_hoist_prefill",
    "b12x.moe._shared.p8.kernels.grouped_fc2_grid376",
    "b12x.moe._shared.p8.kernels.route_hoist_k5",
    "b12x.moe._shared.p8.kernels.route_hoist_direct32",
    "b12x.moe._shared.p8.kernels.p8_narrow_fc1",
)

# Exact float32 encodings used by torch's terminal ``/ sqrt(N)`` in the
# full-coupled CPU reference.  Multiplying by the reciprocal twice (H128 then
# H4) is algebraically equivalent but can move an E4M3 tie by one code, so keep
# these as divisors and use div.rn.f32.
_P8_SQRT128_F32 = 11.313708305358887  # 0x413504f3
_P8_SQRT512_F32 = 22.627416610717773  # 0x41b504f3


@dsl_user_op
def _p8_mul_rn_f32(
    a: cutlass.Float32,
    b: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """One separately rounded FP32 multiply, opaque to FMA contraction."""

    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _p8_had128_quad_unnormalized(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    lane: Int32,
):
    """Natural-order Sylvester H128 without terminal normalization."""

    s0 = v0 + v1
    d0 = v0 - v1
    s1 = v2 + v3
    d1 = v2 - v3
    h0 = s0 + s1
    h1 = d0 + d1
    h2 = s0 - s1
    h3 = d0 - d1
    for i in cutlass.range_constexpr(5):
        stride = 1 << i
        p0 = cute.arch.shuffle_sync_bfly(h0, offset=stride)
        p1 = cute.arch.shuffle_sync_bfly(h1, offset=stride)
        p2 = cute.arch.shuffle_sync_bfly(h2, offset=stride)
        p3 = cute.arch.shuffle_sync_bfly(h3, offset=stride)
        if (lane & Int32(stride)) != Int32(0):
            h0 = p0 - h0
            h1 = p1 - h1
            h2 = p2 - h2
            h3 = p3 - h3
        else:
            h0 = p0 + h0
            h1 = p1 + h1
            h2 = p2 + h2
            h3 = p3 + h3
    return h0, h1, h2, h3


@cute.jit
def _p8_had128_quad_reference_order(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    lane: Int32,
):
    """H128 with the CPU reference's one correctly-rounded terminal divide."""

    h0, h1, h2, h3 = _p8_had128_quad_unnormalized(v0, v1, v2, v3, lane)
    divisor = cutlass.Float32(_P8_SQRT128_F32)
    return (
        div_rn_f32(h0, divisor),
        div_rn_f32(h1, divisor),
        div_rn_f32(h2, divisor),
        div_rn_f32(h3, divisor),
    )


@cute.jit
def _p8_h512_mix_reference_order(
    x0: cutlass.Float32,
    x1: cutlass.Float32,
    x2: cutlass.Float32,
    x3: cutlass.Float32,
):
    """Finish H512 exactly as CPU butterfly strides 128 then 256."""

    r0 = x0 + x1
    r1 = x0 - x1
    r2 = x2 + x3
    r3 = x2 - x3
    divisor = cutlass.Float32(_P8_SQRT512_F32)
    return (
        div_rn_f32(r0 + r2, divisor),
        div_rn_f32(r1 + r3, divisor),
        div_rn_f32(r0 - r2, divisor),
        div_rn_f32(r1 - r3, divisor),
    )


@cute.jit
def _p8_scale_input_before_h128(
    value: cutlass.Float32,
    scale_component: cute.Tensor,
    input_col: Int32,
) -> cutlass.Float32:
    """Apply shared gate/up suh at Luke's exact pre-H128 FP16 boundary.

    This helper must run before both ``_w4a8_had128_quad`` and the E4M3 K32
    amax.  The scale component occupies the first H=4096 FP16 elements of the
    packed transform operand.  H512, when enabled by the coupled owner, runs
    before this helper and is not part of the scale component.
    """

    scaled = value * scale_component[input_col].to(cutlass.Float32)
    return cutlass.Float16(scaled).to(cutlass.Float32)


class P8DynamicKernelBackend(MoEDynamicKernelBackend):
    """The shared dynamic kernel with the P8 input boundary and phase owners.

    The P8 runtime builds this class; every flag it adds defaults off in the
    shared module, so the shared recipes keep the code they already ran.
    """

    # The P8 programs resolve routed ids in the raw form (the caller range-tests
    # them) and keep two input warps per shared-input token.
    p8_route_ids = True
    p8_producer_split = True

    @cute.jit
    def _store_p8_m1_full_coupled_input(
        self,
        a_input: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_lut: cute.Tensor,
        m1_lane_id: Int32,
        flat_tid: Int32,
        flat_stride: Int32,
        num_tokens: Int32,
        mx_blocks_per_row: Int32,
    ):
        """Materialize the exact H512*suh->H128 input for the M1 owners."""

        # Full Luke/QSRT input boundary. One warp owns all four
        # H128 quarters of a complete H512 block, so the outer
        # normalized H4 coupling happens in registers with no
        # cross-CTA dependency. The order is exactly:
        # BF16 -> FP16 -> H512 -> signed suh -> H128 (FP32),
        # followed by K32 amax/E4M3/UE8M0 packing.
        m1_warp = flat_tid >> Int32(5)
        m1_warp_stride = flat_stride >> Int32(5)
        m1_h512 = m1_warp
        while m1_h512 < num_tokens * (mx_blocks_per_row // Int32(16)):
            # One H512 unit per input row. Preserve the M1 order
            # while permitting multiple direct-route tokens.
            p8_input_row = m1_h512 // (mx_blocks_per_row // Int32(16))
            p8_local_h512 = m1_h512 % (mx_blocks_per_row // Int32(16))
            quarters = tuple(
                cute.make_rmem_tensor((4,), cutlass.Float32) for _quarter in range(4)
            )
            for quarter in cutlass.range_constexpr(4):
                m1_col = (
                    p8_local_h512 * Int32(512)
                    + Int32(quarter * 128)
                    + m1_lane_id * Int32(4)
                )
                quarters[quarter][0] = cutlass.Float16(
                    a_input[p8_input_row, m1_col].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][1] = cutlass.Float16(
                    a_input[p8_input_row, m1_col + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][2] = cutlass.Float16(
                    a_input[p8_input_row, m1_col + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][3] = cutlass.Float16(
                    a_input[p8_input_row, m1_col + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                q0, q1, q2, q3 = _p8_had128_quad_unnormalized(
                    quarters[quarter][0],
                    quarters[quarter][1],
                    quarters[quarter][2],
                    quarters[quarter][3],
                    m1_lane_id,
                )
                quarters[quarter][0] = q0
                quarters[quarter][1] = q1
                quarters[quarter][2] = q2
                quarters[quarter][3] = q3
            for component in cutlass.range_constexpr(4):
                x0 = quarters[0][component]
                x1 = quarters[1][component]
                x2 = quarters[2][component]
                x3 = quarters[3][component]
                y0, y1, y2, y3 = _p8_h512_mix_reference_order(x0, x1, x2, x3)
                quarters[0][component] = y0
                quarters[1][component] = y1
                quarters[2][component] = y2
                quarters[3][component] = y3
            for quarter in cutlass.range_constexpr(4):
                output_col = (
                    p8_local_h512 * Int32(512)
                    + Int32(quarter * 128)
                    + m1_lane_id * Int32(4)
                )
                h0 = _p8_mul_rn_f32(
                    quarters[quarter][0],
                    trellis_rotations[output_col].to(cutlass.Float32),
                )
                h1 = _p8_mul_rn_f32(
                    quarters[quarter][1],
                    trellis_rotations[output_col + Int32(1)].to(cutlass.Float32),
                )
                h2 = _p8_mul_rn_f32(
                    quarters[quarter][2],
                    trellis_rotations[output_col + Int32(2)].to(cutlass.Float32),
                )
                h3 = _p8_mul_rn_f32(
                    quarters[quarter][3],
                    trellis_rotations[output_col + Int32(3)].to(cutlass.Float32),
                )
                h0, h1, h2, h3 = _p8_had128_quad_reference_order(
                    h0, h1, h2, h3, m1_lane_id
                )
                m1_group = m1_lane_id >> Int32(3)
                m1_values = cute.make_rmem_tensor((32,), cutlass.Float32)
                m1_block_max = cutlass.Float32(0.0)
                for m1_elem in cutlass.range_constexpr(32):
                    m1_component = m1_elem & Int32(3)
                    m1_own = h0
                    if m1_component == Int32(1):
                        m1_own = h1
                    elif m1_component == Int32(2):
                        m1_own = h2
                    elif m1_component == Int32(3):
                        m1_own = h3
                    m1_source_lane = m1_group * Int32(8) + m1_elem // Int32(4)
                    m1_value = cute.arch.shuffle_sync(m1_own, m1_source_lane)
                    m1_values[m1_elem] = m1_value
                    m1_block_max = fmax_f32(m1_block_max, fabs_f32(m1_value))
                m1_payload, m1_scale_byte = quantize_block_fp8_mx(
                    _w4a8_trellis_permute_k32(m1_values), m1_block_max
                )
                if (m1_lane_id & Int32(7)) == Int32(0):
                    m1_blk_idx = m1_h512 * Int32(16) + Int32(quarter * 4) + m1_group
                    if cutlass.const_expr(self.p8_input_prequant_diagnostic):
                        # Exact diagnostic contract, logical order:
                        #   f32[0:32]   raw K32 block 40
                        #   f32[32:64]  block 40 * quantizer inv_scale
                        #   f32[64:96]  raw K32 block 62
                        #   f32[96:128] block 62 * quantizer inv_scale
                        # Capture happens after quantization so the
                        # observed payload arithmetic is not changed.
                        assert trellis_lut is not None
                        if m1_blk_idx == Int32(40) or m1_blk_idx == Int32(62):
                            trace_base = Int32(0)
                            if m1_blk_idx == Int32(62):
                                trace_base = Int32(64)
                            trace_inv_scale = ue8m0_to_output_scale(m1_scale_byte)
                            for trace_elem in cutlass.range_constexpr(32):
                                st_global_f32(
                                    get_ptr_as_int64(
                                        trellis_lut,
                                        (trace_base + Int32(trace_elem)) * Int32(4),
                                    ),
                                    m1_values[trace_elem],
                                )
                                st_global_f32(
                                    get_ptr_as_int64(
                                        trellis_lut,
                                        (trace_base + Int32(32 + trace_elem))
                                        * Int32(4),
                                    ),
                                    m1_values[trace_elem] * trace_inv_scale,
                                )
                    m1_block_start = m1_blk_idx * Int32(32)
                    for pair in cutlass.range_constexpr(4):
                        packed64 = (
                            Uint64(m1_payload[pair * 2 + 1]) << Uint64(32)
                        ) | Uint64(m1_payload[pair * 2])
                        st_global_u64(
                            get_ptr_as_int64(
                                packed_a_storage,
                                m1_block_start + Int32(pair * 8),
                            ),
                            packed64,
                        )
                    scale_storage[m1_blk_idx] = Uint8(m1_scale_byte & Uint32(0xFF))
                    if cutlass.const_expr(getattr(self, "p8_input_rowpack", False)):
                        # D-x2-RP2: x_lo = v - dq(x_hi) into token row + 16 (payload + UE8M0) for FC1 MMA rows q+8.
                        rp2_vals = _w4a8_trellis_permute_k32(m1_values)
                        rp2_scale, _ = pow2_ceil_ue8m0(
                            m1_block_max * cutlass.Float32(1.0 / 448.0)
                        )
                        rp2_lo = cute.make_rmem_tensor((32,), cutlass.Float32)
                        rp2_max = cutlass.Float32(0.0)
                        for rp2_w in cutlass.range_constexpr(8):
                            rp2_h0, rp2_h1, rp2_h2, rp2_h3 = cvt_e4m3x4_to_f32x4(
                                m1_payload[rp2_w]
                            )
                            rp2_lo[rp2_w * 4 + 0] = (
                                rp2_vals[rp2_w * 4 + 0] - rp2_h0 * rp2_scale
                            )
                            rp2_lo[rp2_w * 4 + 1] = (
                                rp2_vals[rp2_w * 4 + 1] - rp2_h1 * rp2_scale
                            )
                            rp2_lo[rp2_w * 4 + 2] = (
                                rp2_vals[rp2_w * 4 + 2] - rp2_h2 * rp2_scale
                            )
                            rp2_lo[rp2_w * 4 + 3] = (
                                rp2_vals[rp2_w * 4 + 3] - rp2_h3 * rp2_scale
                            )
                        for rp2_e in cutlass.range_constexpr(32):
                            rp2_max = fmax_f32(rp2_max, fabs_f32(rp2_lo[rp2_e]))
                        rp2_payload, rp2_byte = quantize_block_fp8_mx(rp2_lo, rp2_max)
                        rp2_blk = (
                            m1_blk_idx
                            + Int32(getattr(self, "p8_input_rowpack_stride", 16))
                            * mx_blocks_per_row
                        )
                        for rp2_pair in cutlass.range_constexpr(4):
                            rp2_packed = (
                                Uint64(rp2_payload[rp2_pair * 2 + 1]) << Uint64(32)
                            ) | Uint64(rp2_payload[rp2_pair * 2])
                            st_global_u64(
                                get_ptr_as_int64(
                                    packed_a_storage,
                                    rp2_blk * Int32(32) + Int32(rp2_pair * 8),
                                ),
                                rp2_packed,
                            )
                        scale_storage[rp2_blk] = Uint8(rp2_byte & Uint32(0xFF))
            m1_h512 += m1_warp_stride

    @cute.jit
    def _store_p8_m1_scaled_input(
        self,
        a_input: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        trellis_rotations: cute.Tensor,
        m1_lane_id: Int32,
        flat_tid: Int32,
        flat_stride: Int32,
        mx_blocks_per_row: Int32,
    ):
        """Materialize the M1 scale-sandwich input (suh before H128)."""

        # One warp owns one complete H128 input block.  Each lane
        # loads four adjacent BF16 channels, applies signed suh in
        # FP32, rounds to FP16, then participates in normalized
        # H128.  K32 amax is therefore computed only from the final
        # transformed values, never from the pre-suh input.
        m1_warp = flat_tid >> Int32(5)
        m1_warp_stride = flat_stride >> Int32(5)
        m1_h128 = m1_warp
        while m1_h128 < mx_blocks_per_row // Int32(4):
            m1_col = m1_h128 * Int32(128) + m1_lane_id * Int32(4)
            h0 = _p8_scale_input_before_h128(
                a_input[m1_col].to(cutlass.Float32),
                trellis_rotations,
                m1_col,
            )
            h1 = _p8_scale_input_before_h128(
                a_input[m1_col + Int32(1)].to(cutlass.Float32),
                trellis_rotations,
                m1_col + Int32(1),
            )
            h2 = _p8_scale_input_before_h128(
                a_input[m1_col + Int32(2)].to(cutlass.Float32),
                trellis_rotations,
                m1_col + Int32(2),
            )
            h3 = _p8_scale_input_before_h128(
                a_input[m1_col + Int32(3)].to(cutlass.Float32),
                trellis_rotations,
                m1_col + Int32(3),
            )
            h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, m1_lane_id)
            m1_group = m1_lane_id >> Int32(3)
            m1_values = cute.make_rmem_tensor((32,), cutlass.Float32)
            m1_block_max = cutlass.Float32(0.0)
            for m1_elem in cutlass.range_constexpr(32):
                m1_component = m1_elem & Int32(3)
                m1_own = h0
                if m1_component == Int32(1):
                    m1_own = h1
                elif m1_component == Int32(2):
                    m1_own = h2
                elif m1_component == Int32(3):
                    m1_own = h3
                m1_source_lane = m1_group * Int32(8) + m1_elem // Int32(4)
                m1_value = cute.arch.shuffle_sync(m1_own, m1_source_lane)
                m1_values[m1_elem] = m1_value
                m1_block_max = fmax_f32(m1_block_max, fabs_f32(m1_value))
            m1_payload, m1_mx_scale_byte = quantize_block_fp8_mx(
                _w4a8_trellis_permute_k32(m1_values),
                m1_block_max,
            )
            if (m1_lane_id & Int32(7)) == Int32(0):
                m1_blk_idx = m1_h128 * Int32(4) + m1_group
                m1_block_start = m1_blk_idx * Int32(32)
                for m1_payload_pair in cutlass.range_constexpr(4):
                    m1_packed64 = (
                        Uint64(m1_payload[m1_payload_pair * 2 + 1]) << Uint64(32)
                    ) | Uint64(m1_payload[m1_payload_pair * 2])
                    st_global_u64(
                        get_ptr_as_int64(
                            packed_a_storage,
                            m1_block_start + Int32(m1_payload_pair * 8),
                        ),
                        m1_packed64,
                    )
                scale_storage[m1_blk_idx] = Uint8(m1_mx_scale_byte & Uint32(0xFF))
            m1_h128 += m1_warp_stride

    @cute.jit
    def _store_p8_full_coupled_input_row(
        self,
        a_input: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        scale_component: cute.Tensor,
        token_idx: Int32,
        warp_partition: Int32,
        lane: Int32,
        mx_blocks_per_row: Int32,
    ):
        """Materialize one token's exact H512*suh->H128 P8 input.

        Each of the two producer warps owns alternating H512 units.  A unit is
        complete in one warp, so neither normalized H128 nor the outer H4
        crosses a CTA/warp boundary.  FP32 is retained from H512 through signed
        suh and H128 until the E4M3/UE8M0/32 quantizer.
        """

        # A scalar CuTe coordinate on rank-2 input is colexicographic, not
        # a row-major storage offset. Use explicit token/channel coordinates.
        h512 = warp_partition
        while h512 < mx_blocks_per_row // Int32(16):
            quarters = tuple(
                cute.make_rmem_tensor((4,), cutlass.Float32) for _quarter in range(4)
            )
            for quarter in cutlass.range_constexpr(4):
                col = h512 * Int32(512) + Int32(quarter * 128) + lane * Int32(4)
                quarters[quarter][0] = cutlass.Float16(
                    a_input[token_idx, col].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][1] = cutlass.Float16(
                    a_input[token_idx, col + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][2] = cutlass.Float16(
                    a_input[token_idx, col + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                quarters[quarter][3] = cutlass.Float16(
                    a_input[token_idx, col + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                q0, q1, q2, q3 = _p8_had128_quad_unnormalized(
                    quarters[quarter][0],
                    quarters[quarter][1],
                    quarters[quarter][2],
                    quarters[quarter][3],
                    lane,
                )
                quarters[quarter][0] = q0
                quarters[quarter][1] = q1
                quarters[quarter][2] = q2
                quarters[quarter][3] = q3
            for component in cutlass.range_constexpr(4):
                x0 = quarters[0][component]
                x1 = quarters[1][component]
                x2 = quarters[2][component]
                x3 = quarters[3][component]
                y0, y1, y2, y3 = _p8_h512_mix_reference_order(x0, x1, x2, x3)
                quarters[0][component] = y0
                quarters[1][component] = y1
                quarters[2][component] = y2
                quarters[3][component] = y3
            for quarter in cutlass.range_constexpr(4):
                output_col = h512 * Int32(512) + Int32(quarter * 128) + lane * Int32(4)
                h0 = _p8_mul_rn_f32(
                    quarters[quarter][0],
                    scale_component[output_col].to(cutlass.Float32),
                )
                h1 = _p8_mul_rn_f32(
                    quarters[quarter][1],
                    scale_component[output_col + Int32(1)].to(cutlass.Float32),
                )
                h2 = _p8_mul_rn_f32(
                    quarters[quarter][2],
                    scale_component[output_col + Int32(2)].to(cutlass.Float32),
                )
                h3 = _p8_mul_rn_f32(
                    quarters[quarter][3],
                    scale_component[output_col + Int32(3)].to(cutlass.Float32),
                )
                h0, h1, h2, h3 = _p8_had128_quad_reference_order(h0, h1, h2, h3, lane)
                group = lane >> Int32(3)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem in cutlass.range_constexpr(32):
                    component = elem & Int32(3)
                    owned = h0
                    if component == Int32(1):
                        owned = h1
                    elif component == Int32(2):
                        owned = h2
                    elif component == Int32(3):
                        owned = h3
                    source_lane = group * Int32(8) + elem // Int32(4)
                    value = cute.arch.shuffle_sync(owned, source_lane)
                    values[elem] = value
                    block_max = fmax_f32(block_max, fabs_f32(value))
                payload, scale_byte = quantize_block_fp8_mx(
                    _w4a8_trellis_permute_k32(values), block_max
                )
                if (lane & Int32(7)) == Int32(0):
                    block = h512 * Int32(16) + Int32(quarter * 4) + group
                    block_start = block * Int32(32)
                    payload_row = token_idx * Int32(a_input.shape[1])
                    for pair in cutlass.range_constexpr(4):
                        packed64 = (
                            Uint64(payload[pair * 2 + 1]) << Uint64(32)
                        ) | Uint64(payload[pair * 2])
                        st_global_u64(
                            get_ptr_as_int64(
                                packed_a_storage,
                                payload_row + block_start + Int32(pair * 8),
                            ),
                            packed64,
                        )
                    scale_storage[token_idx * mx_blocks_per_row + block] = Uint8(
                        scale_byte & Uint32(0xFF)
                    )
            h512 += Int32(self.input_warps_per_token)

    def _owner_phase_kernels(
        self,
        materialized_source_tile_m,
        trellis_bits,
        deterministic_output,
        phase1_split,
        phase2_split,
    ):
        """Install the P8 phase-1/phase-2 owners.

        The base kernel body is the shared one, so every recipe that is not a
        P8 arm keeps the phase kernels ``__init__`` already built. The base
        kernels constructed here are replaced by a P8 owner whenever one of the
        P8 paths is active, and a discarded instance is never launched.
        """

        self.materialized_phase1_kernel = P8W4A8MaterializedPhase1Kernel(
            fast_math=self.fast_math,
            source_tile_m=materialized_source_tile_m,
            deterministic_output=bool(deterministic_output),
            num_topk=self.num_topk,
            trellis_bits=(trellis_bits if self.w4a8_trellis and phase1_split else None),
            trellis_intermediate_hadamard=(
                self.trellis_intermediate_hadamard and phase1_split
            ),
            trellis_direct_lut=(self.trellis_direct_lut and phase1_split),
            # This helper is gated-only and is never launched unless the split
            # materialized path is active.  Use a valid inert specialization for
            # non-split activations (notably ReLU2) instead of rejecting them
            # during otherwise valid monolithic-kernel construction.
            activation=self.activation if phase1_split else "silu",
            trellis_codebook=(self.trellis_codebook if phase1_split else "lut_e4m3"),
            trellis_scaled=self.trellis_scaled and phase1_split,
            trellis_identity_boundary=(self.trellis_identity_boundary and phase1_split),
        )
        self.materialized_phase2_kernel = P8W4A8MaterializedPhase2Kernel(
            source_tile_m=materialized_source_tile_m,
            deterministic_output=bool(deterministic_output),
            trellis_bits=(trellis_bits if self.w4a8_trellis and phase2_split else None),
            trellis_direct_lut=(self.trellis_direct_lut and phase2_split),
            trellis_codebook=(self.trellis_codebook if phase2_split else "lut_e4m3"),
            trellis_scaled=self.trellis_scaled and phase2_split,
            trellis_identity_boundary=(self.trellis_identity_boundary and phase2_split),
        )
        if self.p8_small_m:
            from b12x.moe._shared.p8.kernels.fc2_k5_funnel import P8SmallMPhase2Kernel

            if self.mma_tiler_mn == (32, 128):
                from b12x.moe._shared.p8.kernels.direct_m32_fc2 import (
                    P8DirectM32FC2Kernel as P8SmallMPhase2Kernel,
                )
            self.materialized_phase2_kernel = P8SmallMPhase2Kernel(
                trellis_bits=trellis_bits
            )
            if self.p8_scale_sandwich:
                self.materialized_phase2_kernel = P8SmallMPhase2Kernel(
                    scale_sandwich=True,
                    full_coupled=self.p8_full_coupled,
                    trellis_bits=trellis_bits,
                )
        if self.p8_grouped_m32:
            from b12x.moe._shared.p8.kernels.policy_m32_fc1 import (
                P8CoupledPrefillFC1Kernel as P8GroupedM16FC1Kernel,
            )
            from b12x.moe._shared.p8.kernels.policy_m32_fc2 import (
                P8CoupledPrefillFC2Kernel as P8GroupedM16FC2Kernel,
            )

            self.materialized_phase1_kernel = P8GroupedM16FC1Kernel(
                trellis_bits=trellis_bits
            )
            self.materialized_phase2_kernel = P8GroupedM16FC2Kernel(
                trellis_bits=trellis_bits
            )
        elif self.p8_full_coupled and self.w4a8_m64_materialized:
            from b12x.moe._shared.p8.kernels.route_hoist_prefill import (
                P8CoupledPrefillFC1Kernel,
            )
            from b12x.moe._shared.p8.kernels.grouped_fc2_grid376 import (
                P8CoupledPrefillFC2Kernel,
            )

            self.materialized_phase1_kernel = P8CoupledPrefillFC1Kernel(
                trellis_bits=trellis_bits
            )
            self.materialized_phase2_kernel = P8CoupledPrefillFC2Kernel(
                trellis_bits=trellis_bits
            )
        elif self.p8_scale_sandwich:
            from b12x.moe._shared.p8.kernels.route_hoist_k5 import P8H128FC1Kernel

            if self.mma_tiler_mn == (32, 128):
                from b12x.moe._shared.p8.kernels.route_hoist_direct32 import (
                    P8DirectM32FC1Kernel as P8H128FC1Kernel,
                )
            self.materialized_phase1_kernel = P8H128FC1Kernel(
                full_coupled=self.p8_full_coupled,
                trellis_bits=trellis_bits,
            )
        elif self.p8_fc1_tile_n != 128:
            from b12x.moe._shared.p8.kernels.p8_narrow_fc1 import P8NarrowFC1Kernel

            self.materialized_phase1_kernel = P8NarrowFC1Kernel(
                self.p8_fc1_tile_n, trellis_bits=trellis_bits
            )
        if self.p8_small_m and self.p8_scale_sandwich:
            # Constructor invariant for the exact M1 boundary.  Keep this
            # fail-closed because falling back to the monolithic FC1 changes
            # the quantization transform while still producing finite output.
            if not self.external_materialized_fc1 or not isinstance(
                self.materialized_phase1_kernel, P8H128FC1Kernel
            ):
                raise AssertionError(
                    "P8 M1 scale sandwich requires external P8H128FC1Kernel"
                )
        return self.materialized_phase1_kernel, self.materialized_phase2_kernel


__all__ = ["P8DynamicKernelBackend"]
