"""Warp-specialized SM120 4D-TMA relative-position prefill attention with fused gate-sink.

Two lean kernels serve paged FP8 full-attention prefill over arbitrary context lengths up to 1M+:

1. ``RelativePrefillKernel``:
   - Producer warp issues 4D TMA loads of ``(STAGE, 128)`` FP8 E4M3 K and V tiles directly from
     ``[num_pages, page_size, Hkv, 128]`` (64-bit dynamic strides) with separate ``full_K`` and
     ``full_V`` mbarriers so V transfers overlap QK MMA and online softmax.
   - Compute warps keep Q in BF16 registers, widen FP8 K/V in registers via SM120 native
     ``cvt.rn.bf16x2.e4m3x2``, accumulate FP32 scores via side-effect-free ``bf16_mma_m16n8k16_f32``,
     and split the key loop into two static regions:
       * Region 0 (far prefix, ``key <= q_min_pos - rel_extent``): zero relative bias, zero
         causal/bounds masking, unmasked V MMA, log2-domain online softmax over ``BN_SUB`` keys
         per rescaling step.
       * Region 1 (biased/causal tail, last ``<= rel_extent + q_tile_tokens`` keys): per-query
         relative bias ``Bias[q_row, q_head, pos - k]``, causal masking, masked V MMA.
   - Supports both GQA-grouped CTAs (``grouped=True``, sharing each TMA K/V tile across all
     ``group_size`` query heads) and per-head CTAs (``grouped=False``), reverse Q-tile scheduling
     for causal load balance, and optional split-KV across context length.
   - When ``split_kv=False``, fuses per-token output gate and per-token sink sigmoid directly into
     a single BF16 round in the epilogue (zero intermediate FP32 output or LSE write).

2. ``RelativePrefillMergeKernel``:
   - Register-only warp-per-(token, head) reduction across ``active_splits`` FP32 partials with
     fused per-token gate and sink sigmoid into BF16 output and padded-row zeroing.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int64, T, dsl_user_op
from cutlass.cute.nvgpu import cpasync

from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m16n16x1_trans_b8,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v2_f32,
    st_global_v2_u32,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.intrinsics.packed import (
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    pack_f32x2_to_bfloat2,
)
from b12x.attention.paged.forward_paged import (
    _assume_paged_kv_tma_source_aligned,
    _assume_tensor_aligned,
    _exit_thread,
    _exp2_approx_ftz_f32,
    _get_memrange_tensor,
    _make_payload_memrange,
    _mask_pv_bf16_pair_windowed,
    _max_nan_f32,
    _paged_kv_tma_plane_layout,
    _paged_kv_tma_plane_stage_layout,
)

LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
SMEM_LIMIT = 99 * 1024


@dsl_user_op
def _ld_global_nc_u16(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load 16 bits from read-only global memory (.nc), zero-extended to u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _sigmoid_f32(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + _exp2_approx_ftz_f32(-x * Float32(LOG2_E)))


def _make_4d_kv_tma_source(t: cute.Tensor):
    """Permute ``[N, P, Hkv, 128]`` FP8 cache to ``[P, 128, Hkv, N]`` for 4D TMA."""
    t_4d = cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
    return _assume_paged_kv_tma_source_aligned(t_4d)


@cute.jit
def _mma_m16n16k16_pure(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    b2: Uint32,
    b3: Uint32,
):
    """Two side-effect-free ``bf16_mma_m16n8k16_f32`` instructions for one 16x16x16 tile."""
    r0, r1, r2, r3 = bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
    r4, r5, r6, r7 = bf16_mma_m16n8k16_f32(d4, d5, d6, d7, a0, a1, a2, a3, b2, b3)
    return r0, r1, r2, r3, r4, r5, r6, r7


@cute.jit
def _qk_fp8_multi(
    s_frag: cute.Tensor,
    q_regs: cute.Tensor,
    q_smem_warp: Int32,
    k_plane0: Int32,
    lane: Int32,
    key_base: Int32,
    n_sub: cutlass.Constexpr,
    q_in_smem: cutlass.Constexpr,
):
    """Compute ``s_frag[sub, 0..7] = Q @ K[sub]^T`` for ``n_sub`` 16-key subtiles."""
    mat = lane // Int32(8)
    row_in_sub = (mat % Int32(2)) * Int32(8) + lane % Int32(8)
    chunk_off = mat // Int32(2)
    for sub in cutlass.range_constexpr(n_sub):
        for r in cutlass.range_constexpr(8):
            s_frag[sub, r] = Float32(0.0)
    for pair in cutlass.range_constexpr(4):
        step = 2 * pair
        chunk = Int32(step) + chunk_off
        if const_expr(q_in_smem):
            a0_0, a1_0, a2_0, a3_0 = ld_shared_v4_u32(
                q_smem_warp + Int32(step * 512) + lane * Int32(16)
            )
            a0_1, a1_1, a2_1, a3_1 = ld_shared_v4_u32(
                q_smem_warp + Int32((step + 1) * 512) + lane * Int32(16)
            )
        else:
            a0_0 = q_regs[step, 0]
            a1_0 = q_regs[step, 1]
            a2_0 = q_regs[step, 2]
            a3_0 = q_regs[step, 3]
            a0_1 = q_regs[step + 1, 0]
            a1_1 = q_regs[step + 1, 1]
            a2_1 = q_regs[step + 1, 2]
            a3_1 = q_regs[step + 1, 3]
        for sub in cutlass.range_constexpr(n_sub):
            key_row = key_base + Int32(sub * 16) + row_in_sub
            row_swizzle = key_row % Int32(8)
            r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(
                k_plane0 + (key_row * Int32(8) + (chunk ^ row_swizzle)) * Int32(16)
            )
            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r1)
            d0, d1, d2, d3, d4, d5, d6, d7 = _mma_m16n16k16_pure(
                s_frag[sub, 0],
                s_frag[sub, 1],
                s_frag[sub, 2],
                s_frag[sub, 3],
                s_frag[sub, 4],
                s_frag[sub, 5],
                s_frag[sub, 6],
                s_frag[sub, 7],
                a0_0,
                a1_0,
                a2_0,
                a3_0,
                b0,
                b1,
                b2,
                b3,
            )
            c0, c1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r2)
            c2, c3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r3)
            d0, d1, d2, d3, d4, d5, d6, d7 = _mma_m16n16k16_pure(
                d0, d1, d2, d3, d4, d5, d6, d7,
                a0_1, a1_1, a2_1, a3_1,
                c0, c1, c2, c3,
            )
            s_frag[sub, 0] = d0
            s_frag[sub, 1] = d1
            s_frag[sub, 2] = d2
            s_frag[sub, 3] = d3
            s_frag[sub, 4] = d4
            s_frag[sub, 5] = d5
            s_frag[sub, 6] = d6
            s_frag[sub, 7] = d7


@cute.jit
def _online_softmax_far_prefix(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    sm_scale_log2: Float32,
    n_sub: cutlass.Constexpr,
):
    """Unmasked, unbiased log2-domain online softmax over ``16 * n_sub`` keys."""
    for row_slot in cutlass.range_constexpr(2):
        r0 = row_slot * 2 + 0
        r1 = row_slot * 2 + 1
        r4 = row_slot * 2 + 4
        r5 = row_slot * 2 + 5
        m_raw = _max_nan_f32(
            _max_nan_f32(s_frag[0, r0], s_frag[0, r1]),
            _max_nan_f32(s_frag[0, r4], s_frag[0, r5]),
        )
        for sub in cutlass.range_constexpr(1, n_sub):
            m_sub = _max_nan_f32(
                _max_nan_f32(s_frag[sub, r0], s_frag[sub, r1]),
                _max_nan_f32(s_frag[sub, r4], s_frag[sub, r5]),
            )
            m_raw = _max_nan_f32(m_raw, m_sub)
        m_local = m_raw * sm_scale_log2
        m_prev = Float32(m_frag[row_slot])
        m_new = _max_nan_f32(m_prev, m_local)
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        if m_new > m_prev and m_prev != -Float32.inf:
            scale_term = _exp2_approx_ftz_f32(m_prev - m_new)
            d_frag[row_slot] = Float32(d_frag[row_slot] * scale_term)
            scale_pair = (scale_term, scale_term)
            for mma_d in cutlass.range_constexpr(8):
                o0, o1 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r0], o_frag[mma_d, r1]), scale_pair
                )
                o4, o5 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r4], o_frag[mma_d, r5]), scale_pair
                )
                o_frag[mma_d, r0] = o0
                o_frag[mma_d, r1] = o1
                o_frag[mma_d, r4] = o4
                o_frag[mma_d, r5] = o5

        neg_m = -m_new
        scale_pair = (sm_scale_log2, sm_scale_log2)
        neg_m_pair = (neg_m, neg_m)
        p_sum = Float32(0.0)
        for sub in cutlass.range_constexpr(n_sub):
            a0, a1 = cute.arch.fma_packed_f32x2(
                (s_frag[sub, r0], s_frag[sub, r1]), scale_pair, neg_m_pair
            )
            a2, a3 = cute.arch.fma_packed_f32x2(
                (s_frag[sub, r4], s_frag[sub, r5]), scale_pair, neg_m_pair
            )
            p0 = _exp2_approx_ftz_f32(a0)
            p1 = _exp2_approx_ftz_f32(a1)
            p2 = _exp2_approx_ftz_f32(a2)
            p3 = _exp2_approx_ftz_f32(a3)
            p_sum += (p0 + p1) + (p2 + p3)
            p_frag[sub, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
            p_frag[sub, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=2)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=1)
        d_frag[row_slot] += p_sum
        m_frag[row_slot] = Float32(m_new)


@cute.jit
def _pv_fp8_unmasked_multi(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_plane0: Int32,
    lane: Int32,
    key_base: Int32,
    n_sub: cutlass.Constexpr,
):
    """Unmasked ``O += P @ V`` over ``n_sub`` 16-key subtiles (far prefix)."""
    lane16 = lane % Int32(16)
    lane_row = (
        (lane16 & Int32(0x1))
        | ((lane16 & Int32(0x4)) >> Int32(1))
        | ((lane16 & Int32(0x8)) >> Int32(1))
        | ((lane16 & Int32(0x2)) << Int32(2))
    )
    for sub in cutlass.range_constexpr(n_sub):
        v_row = key_base + Int32(sub * 16) + lane_row
        row_swizzle = v_row % Int32(8)
        p0 = p_frag[sub, 0]
        p1 = p_frag[sub, 1]
        p2 = p_frag[sub, 2]
        p3 = p_frag[sub, 3]
        for mma_d in cutlass.range_constexpr(8):
            f0, f1 = ldmatrix_m16n16x1_trans_b8(
                v_plane0 + (v_row * Int32(8) + (Int32(mma_d) ^ row_swizzle)) * Int32(16)
            )
            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
            d0, d1, d2, d3, d4, d5, d6, d7 = _mma_m16n16k16_pure(
                o_frag[mma_d, 0],
                o_frag[mma_d, 1],
                o_frag[mma_d, 2],
                o_frag[mma_d, 3],
                o_frag[mma_d, 4],
                o_frag[mma_d, 5],
                o_frag[mma_d, 6],
                o_frag[mma_d, 7],
                p0, p1, p2, p3,
                b0, b1, b2, b3,
            )
            o_frag[mma_d, 0] = d0
            o_frag[mma_d, 1] = d1
            o_frag[mma_d, 2] = d2
            o_frag[mma_d, 3] = d3
            o_frag[mma_d, 4] = d4
            o_frag[mma_d, 5] = d5
            o_frag[mma_d, 6] = d6
            o_frag[mma_d, 7] = d7


@cute.jit
def _online_softmax_biased_interior(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    tile_sub_base: Int32,
    lane_pair_base: Int32,
    row_valid: cute.Tensor,
    causal_hi: cute.Tensor,
    bias_ptr: cute.Tensor,
    sm_scale: Float32,
    n_sub: cutlass.Constexpr,
):
    """Unmasked, 100%-biased log2-domain online softmax over ``16 * n_sub`` interior keys."""
    log2_e = Float32(LOG2_E)
    log2_pair = (log2_e, log2_e)
    scale_pair = (sm_scale, sm_scale)
    for row_slot in cutlass.range_constexpr(2):
        r0 = row_slot * 2 + 0
        r1 = row_slot * 2 + 1
        r4 = row_slot * 2 + 4
        r5 = row_slot * 2 + 5
        valid_row = row_valid[row_slot] != Int32(0)
        m_local = Float32(-Float32.inf)
        if valid_row:
            bptr = bias_ptr[row_slot]
            d_base = causal_hi[row_slot] - (tile_sub_base + lane_pair_base)
            for sub in cutlass.range_constexpr(n_sub):
                d0 = d_base - Int32(sub * 16)
                b0 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0) * Int64(2)) << Uint32(16))
                b1 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(1)) * Int64(2)) << Uint32(16))
                b4 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(8)) * Int64(2)) << Uint32(16))
                b5 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(9)) * Int64(2)) << Uint32(16))
                x0, x1 = cute.arch.fma_packed_f32x2((s_frag[sub, r0], s_frag[sub, r1]), scale_pair, (b0, b1))
                x4, x5 = cute.arch.fma_packed_f32x2((s_frag[sub, r4], s_frag[sub, r5]), scale_pair, (b4, b5))
                x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), log2_pair)
                x4, x5 = cute.arch.mul_packed_f32x2((x4, x5), log2_pair)
                s_frag[sub, r0] = x0
                s_frag[sub, r1] = x1
                s_frag[sub, r4] = x4
                s_frag[sub, r5] = x5
                m_sub = _max_nan_f32(_max_nan_f32(x0, x1), _max_nan_f32(x4, x5))
                m_local = _max_nan_f32(m_local, m_sub)
        m_prev = Float32(m_frag[row_slot])
        m_new = _max_nan_f32(m_prev, m_local)
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        if valid_row and m_new > m_prev and m_prev != -Float32.inf:
            scale_term = _exp2_approx_ftz_f32(m_prev - m_new)
            d_frag[row_slot] = Float32(d_frag[row_slot] * scale_term)
            st_pair = (scale_term, scale_term)
            for mma_d in cutlass.range_constexpr(8):
                o0, o1 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r0], o_frag[mma_d, r1]), st_pair
                )
                o4, o5 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r4], o_frag[mma_d, r5]), st_pair
                )
                o_frag[mma_d, r0] = o0
                o_frag[mma_d, r1] = o1
                o_frag[mma_d, r4] = o4
                o_frag[mma_d, r5] = o5

        p_sum = Float32(0.0)
        if valid_row:
            for sub in cutlass.range_constexpr(n_sub):
                p0 = _exp2_approx_ftz_f32(s_frag[sub, r0] - m_new)
                p1 = _exp2_approx_ftz_f32(s_frag[sub, r1] - m_new)
                p2 = _exp2_approx_ftz_f32(s_frag[sub, r4] - m_new)
                p3 = _exp2_approx_ftz_f32(s_frag[sub, r5] - m_new)
                p_sum += (p0 + p1) + (p2 + p3)
                p_frag[sub, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
                p_frag[sub, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        else:
            for sub in cutlass.range_constexpr(n_sub):
                p_frag[sub, row_slot + 0] = Uint32(0)
                p_frag[sub, row_slot + 2] = Uint32(0)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=2)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=1)
        if valid_row:
            d_frag[row_slot] += p_sum
            m_frag[row_slot] = Float32(m_new)


@cute.jit
def _pv_fp8_masked_one(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_plane0: Int32,
    lane: Int32,
    key_base: Int32,
    valid_rows: Int32,
    win_lo_rel: Int32,
):
    """Masked ``O += P @ V`` over one 16-key subtile (causal/window/bounds tail)."""
    lane16 = lane % Int32(16)
    lane_row = (
        (lane16 & Int32(0x1))
        | ((lane16 & Int32(0x4)) >> Int32(1))
        | ((lane16 & Int32(0x8)) >> Int32(1))
        | ((lane16 & Int32(0x2)) << Int32(2))
    )
    v_row = key_base + lane_row
    row_swizzle = v_row % Int32(8)
    key = key_base + (lane % Int32(4)) * Int32(2)
    p0 = p_frag[0, 0]
    p1 = p_frag[0, 1]
    p2 = p_frag[0, 2]
    p3 = p_frag[0, 3]
    for mma_d in cutlass.range_constexpr(8):
        f0, f1 = ldmatrix_m16n16x1_trans_b8(
            v_plane0 + (v_row * Int32(8) + (Int32(mma_d) ^ row_swizzle)) * Int32(16)
        )
        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
        b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
        b0 = _mask_pv_bf16_pair_windowed(b0, key, valid_rows, win_lo_rel)
        b1 = _mask_pv_bf16_pair_windowed(b1, key + Int32(8), valid_rows, win_lo_rel)
        b2 = _mask_pv_bf16_pair_windowed(b2, key, valid_rows, win_lo_rel)
        b3 = _mask_pv_bf16_pair_windowed(b3, key + Int32(8), valid_rows, win_lo_rel)
        d0, d1, d2, d3, d4, d5, d6, d7 = _mma_m16n16k16_pure(
            o_frag[mma_d, 0],
            o_frag[mma_d, 1],
            o_frag[mma_d, 2],
            o_frag[mma_d, 3],
            o_frag[mma_d, 4],
            o_frag[mma_d, 5],
            o_frag[mma_d, 6],
            o_frag[mma_d, 7],
            p0, p1, p2, p3,
            b0, b1, b2, b3,
        )
        o_frag[mma_d, 0] = d0
        o_frag[mma_d, 1] = d1
        o_frag[mma_d, 2] = d2
        o_frag[mma_d, 3] = d3
        o_frag[mma_d, 4] = d4
        o_frag[mma_d, 5] = d5
        o_frag[mma_d, 6] = d6
        o_frag[mma_d, 7] = d7


@cute.jit
def _tail_scale_bias_mask_softmax(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    key_base: Int32,
    tile_base: Int32,
    tile_tokens: Int32,
    lane_pair_base: Int32,
    row_valid: cute.Tensor,
    causal_hi: cute.Tensor,
    window_lo: cute.Tensor,
    bias_ptr: cute.Tensor,
    sm_scale: Float32,
    interior: cutlass.Boolean,
    all_biased: cutlass.Boolean,
    any_biased: cutlass.Boolean,
    use_window: cutlass.Constexpr,
    rel_extent: cutlass.Constexpr,
):
    """Scale, apply relative bias and causal/window mask, and update log2-domain online softmax for 16 keys."""
    log2_e = Float32(LOG2_E)
    if interior and not any_biased:
        sm_scale_log2 = sm_scale * log2_e
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            if row_valid[row_slot] != Int32(0):
                s_frag[0, reg_id] = s_frag[0, reg_id] * sm_scale_log2
            else:
                s_frag[0, reg_id] = Float32(-Float32.inf)
    elif interior and all_biased:
        sub_base = tile_base + key_base
        for row_slot in cutlass.range_constexpr(2):
            r0 = row_slot * 2 + 0
            r1 = row_slot * 2 + 1
            r4 = row_slot * 2 + 4
            r5 = row_slot * 2 + 5
            if row_valid[row_slot] != Int32(0):
                bptr = bias_ptr[row_slot]
                d0 = causal_hi[row_slot] - (sub_base + lane_pair_base)
                b0 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0) * Int64(2)) << Uint32(16))
                b1 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(1)) * Int64(2)) << Uint32(16))
                b4 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(8)) * Int64(2)) << Uint32(16))
                b5 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(9)) * Int64(2)) << Uint32(16))
                s_frag[0, r0] = (s_frag[0, r0] * sm_scale + b0) * log2_e
                s_frag[0, r1] = (s_frag[0, r1] * sm_scale + b1) * log2_e
                s_frag[0, r4] = (s_frag[0, r4] * sm_scale + b4) * log2_e
                s_frag[0, r5] = (s_frag[0, r5] * sm_scale + b5) * log2_e
            else:
                s_frag[0, r0] = Float32(-Float32.inf)
                s_frag[0, r1] = Float32(-Float32.inf)
                s_frag[0, r4] = Float32(-Float32.inf)
                s_frag[0, r5] = Float32(-Float32.inf)
    else:
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            key_local = key_base + lane_pair_base + Int32(8 * (reg_id // 4) + (reg_id % 2))
            key_abs = tile_base + key_local
            valid = (
                (row_valid[row_slot] != Int32(0))
                and (key_local < tile_tokens)
                and (key_abs <= causal_hi[row_slot])
            )
            if const_expr(use_window):
                valid = valid and (key_abs >= window_lo[row_slot])
            if valid:
                s = s_frag[0, reg_id] * sm_scale
                if any_biased:
                    dist = causal_hi[row_slot] - key_abs
                    if dist < Int32(rel_extent):
                        b_u32 = _ld_global_nc_u16(bias_ptr[row_slot] + Int64(dist) * Int64(2))
                        s = s + u32_as_f32(b_u32 << Uint32(16))
                s_frag[0, reg_id] = s * log2_e
            else:
                s_frag[0, reg_id] = Float32(-Float32.inf)

    for row_slot in cutlass.range_constexpr(2):
        r0 = row_slot * 2 + 0
        r1 = row_slot * 2 + 1
        r4 = row_slot * 2 + 4
        r5 = row_slot * 2 + 5
        m_prev = Float32(m_frag[row_slot])
        m_local = _max_nan_f32(
            _max_nan_f32(s_frag[0, r0], s_frag[0, r1]),
            _max_nan_f32(s_frag[0, r4], s_frag[0, r5]),
        )
        m_new = _max_nan_f32(m_prev, m_local)
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        if m_new > m_prev and m_prev != -Float32.inf:
            scale_term = _exp2_approx_ftz_f32(m_prev - m_new)
            d_frag[row_slot] = Float32(d_frag[row_slot] * scale_term)
            st_pair = (scale_term, scale_term)
            for mma_d in cutlass.range_constexpr(8):
                o0, o1 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r0], o_frag[mma_d, r1]), st_pair
                )
                o4, o5 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r4], o_frag[mma_d, r5]), st_pair
                )
                o_frag[mma_d, r0] = o0
                o_frag[mma_d, r1] = o1
                o_frag[mma_d, r4] = o4
                o_frag[mma_d, r5] = o5

        p0 = Float32(0.0)
        p1 = Float32(0.0)
        p2 = Float32(0.0)
        p3 = Float32(0.0)
        if m_new != -Float32.inf:
            p0 = _exp2_approx_ftz_f32(s_frag[0, r0] - m_new)
            p1 = _exp2_approx_ftz_f32(s_frag[0, r1] - m_new)
            p2 = _exp2_approx_ftz_f32(s_frag[0, r4] - m_new)
            p3 = _exp2_approx_ftz_f32(s_frag[0, r5] - m_new)
        p_sum = Float32((p0 + p1) + (p2 + p3))
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=2)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=1)
        d_frag[row_slot] += p_sum
        p_frag[0, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
        p_frag[0, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        m_frag[row_slot] = Float32(m_new)


class RelativePrefillKernel:
    """Warp-specialized SM120 4D-TMA paged full-attention prefill kernel with relative bias and fused gate-sink."""

    def __init__(
        self,
        *,
        num_q_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 128,
        rel_extent: int = 1024,
        window_left: int = -1,
        q_tile_tokens: int = 64,
        grouped: bool = True,
        stage_rows: int = 64,
        bn_sub: int = 32,
        num_stages: int = 3,
        split_kv: bool = False,
        min_stages_per_split: int = 8,
        reverse_q_tiles: bool = True,
        q_in_smem: bool = False,
    ):
        if head_dim != 128:
            raise ValueError("RelativePrefillKernel requires head_dim == 128")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be a multiple of num_kv_heads")
        if q_tile_tokens < 16 or q_tile_tokens % 16 != 0:
            raise ValueError("q_tile_tokens must be a positive multiple of 16")
        if stage_rows not in (32, 64, 128) or page_size % stage_rows != 0:
            raise ValueError(f"unsupported stage_rows={stage_rows} for page_size={page_size}")
        if bn_sub not in (16, 32, 64) or stage_rows % bn_sub != 0:
            raise ValueError(f"bn_sub={bn_sub} must divide stage_rows={stage_rows}")
        if window_left < -1:
            raise ValueError(f"window_left must be -1 or non-negative, got {window_left}")
        group_size = num_q_heads // num_kv_heads
        warps_per_head = q_tile_tokens // 16
        ncw = (group_size * warps_per_head) if grouped else warps_per_head
        if ncw > 12:
            raise ValueError(f"num_compute_warps={ncw} exceeds 12")
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.group_size = int(group_size)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.rel_extent = int(rel_extent)
        self.window_left = int(window_left)
        self.is_window = self.window_left >= 0
        self.window_size = (self.window_left + 1) if self.is_window else 0
        self.q_tile_tokens = int(q_tile_tokens)
        self.warps_per_head = int(warps_per_head)
        self.grouped = bool(grouped)
        self.stage_rows = int(stage_rows)
        self.bn_sub = int(bn_sub)
        self.n_sub = int(bn_sub // 16)
        self.num_stages = int(num_stages)
        self.split_kv = bool(split_kv)
        self.min_stages_per_split = int(min_stages_per_split)
        self.reverse_q_tiles = bool(reverse_q_tiles)
        self.q_in_smem = bool(q_in_smem)
        self.ncw = int(ncw)
        self.num_warps = self.ncw
        self.kv_plane_stage_bytes = self.stage_rows * 128
        self.kv_plane_total_bytes = self.num_stages * self.kv_plane_stage_bytes
        self.q_smem_bytes = (self.ncw * 4096) if self.q_in_smem else 0
        self.shared_storage_bytes = 2 * self.kv_plane_total_bytes + self.q_smem_bytes
        if self.shared_storage_bytes + 2048 > SMEM_LIMIT:
            raise ValueError(f"shared storage {self.shared_storage_bytes} exceeds SM120 limit")

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "full_K": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "full_V": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "empty": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, int(self.shared_storage_bytes)],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [T, Hq, 128] bf16
        mKCache: cute.Tensor,  # [num_pages, page_size, Hkv, 128] u8 (e4m3)
        mVCache: cute.Tensor,  # [num_pages, page_size, Hkv, 128] u8 (e4m3)
        mPageTable: cute.Tensor,  # [B, max_pages] i32
        mLengths: cute.Tensor,  # [B] i32 (total KV length = start + qo_len)
        mCuSeqlensQ: cute.Tensor,  # [B + 1] i32
        mBias: cute.Tensor,  # [T, Hq, rel_extent] bf16
        mGate: cute.Tensor,  # [T, Hq, 128] bf16
        mSink: cute.Tensor,  # [T, Hq] bf16
        mGated: cute.Tensor,  # [T, Hq, 128] bf16
        mOPart: cute.Tensor,  # [T, Hq, S, 128] f32
        mLsePart: cute.Tensor,  # [T, Hq, S] f32
        sm_scale: Float32,
        stream: cuda.CUstream,
    ):
        mQ = _assume_tensor_aligned(mQ)
        mKCacheT = _make_4d_kv_tma_source(mKCache)
        mVCacheT = _make_4d_kv_tma_source(mVCache)
        plane_layout = _paged_kv_tma_plane_layout(self.stage_rows, 128)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op, mKCacheT, plane_layout, (self.stage_rows, 128), 1
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            tma_op, mVCacheT, plane_layout, (self.stage_rows, 128), 1
        )
        num_q_tiles = (mQ.shape[0] + Int32(self.q_tile_tokens - 1)) // Int32(self.q_tile_tokens)
        head_grid = Int32(self.num_kv_heads) if const_expr(self.grouped) else Int32(self.num_q_heads)
        num_splits = mLsePart.shape[2] if const_expr(self.split_kv) else Int32(1)
        num_reqs = mLengths.shape[0]
        min_blocks = 2 if (self.num_warps <= 4 and self.shared_storage_bytes <= 49152) else 1
        self.kernel(
            mQ,
            tma_tensor_K,
            tma_tensor_V,
            mPageTable,
            mLengths,
            mCuSeqlensQ,
            mBias,
            mGate,
            mSink,
            mGated,
            mOPart,
            mLsePart,
            tma_atom_K,
            tma_atom_V,
            sm_scale,
            num_q_tiles,
            num_splits,
        ).launch(
            grid=(num_q_tiles, head_grid, num_reqs * num_splits),
            block=[32, self.num_warps, 1],
            min_blocks_per_mp=min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCacheT: cute.Tensor,
        mVCacheT: cute.Tensor,
        mPageTable: cute.Tensor,
        mLengths: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mBias: cute.Tensor,
        mGate: cute.Tensor,
        mSink: cute.Tensor,
        mGated: cute.Tensor,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        sm_scale: Float32,
        num_q_tiles: Int32,
        num_splits: Int32,
    ):
        STAGE = self.stage_rows
        BN_SUB = self.bn_sub
        N_SUB = self.n_sub
        S = self.num_stages
        NCW = self.ncw
        lane, warp, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tidx = lane + warp * Int32(32)

        request_idx = bz // num_splits
        split_idx = bz - request_idx * num_splits
        q_tile_idx = (num_q_tiles - Int32(1) - bx) if const_expr(self.reverse_q_tiles) else bx
        kv_head_idx = by if const_expr(self.grouped) else (by // Int32(self.group_size))

        # Zero padded output rows once when split_kv=False.
        if const_expr(not self.split_kv):
            if bx == Int32(0) and request_idx == Int32(0) and split_idx == Int32(0):
                num_reqs = mLengths.shape[0]
                total_live = mCuSeqlensQ[num_reqs]
                total_cap = mGated.shape[0]
                if total_live < total_cap:
                    mGatedBytes0 = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                    g_t_s = Int64(mGated.stride[0]) * Int64(2)
                    g_h_s = Int64(mGated.stride[1]) * Int64(2)
                    rp = total_live + warp
                    while rp < total_cap:
                        if const_expr(self.grouped):
                            for hl in cutlass.range_constexpr(self.group_size):
                                hd = kv_head_idx * Int32(self.group_size) + Int32(hl)
                                st_global_v2_u32(
                                    get_ptr_as_int64(
                                        mGatedBytes0,
                                        Int64(rp) * g_t_s + Int64(hd) * g_h_s + Int64(lane * 8),
                                    ),
                                    Uint32(0),
                                    Uint32(0),
                                )
                        else:
                            st_global_v2_u32(
                                get_ptr_as_int64(
                                    mGatedBytes0,
                                    Int64(rp) * g_t_s + Int64(by) * g_h_s + Int64(lane * 8),
                                ),
                                Uint32(0),
                                Uint32(0),
                            )
                        rp += Int32(self.num_warps)

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        q_tile_lo = q_tile_idx * Int32(self.q_tile_tokens)
        if q_tile_lo >= qo_len:
            _exit_thread()

        length = mLengths[request_idx]
        start = length - qo_len
        q_tile_hi = cutlass.select_(
            q_tile_lo + Int32(self.q_tile_tokens) < qo_len,
            q_tile_lo + Int32(self.q_tile_tokens),
            qo_len,
        )
        valid_lo = Int32(0)
        seq_hi = length
        base_offset = start
        tile_start_all = Int32(0)
        if const_expr(self.is_window):
            # Staged window layout: history at [window - min(start, window), window),
            # new rows at [window, window + qo_len), length = min(start, window) + qo_len.
            valid_lo = Int32(self.window_size) + qo_len - length
            seq_hi = Int32(self.window_size) + qo_len
            base_offset = Int32(self.window_size)
            min_key = base_offset + q_tile_lo - Int32(self.window_left)
            if min_key < valid_lo:
                min_key = valid_lo
            tile_start_all = min_key // Int32(STAGE)
            max_key_excl = base_offset + q_tile_hi
            if max_key_excl > seq_hi:
                max_key_excl = seq_hi
            tile_end = (max_key_excl + Int32(STAGE - 1)) // Int32(STAGE)
        else:
            max_key_excl = start + q_tile_hi
            if max_key_excl > length:
                max_key_excl = length
            tile_end = (max_key_excl + Int32(STAGE - 1)) // Int32(STAGE)
        total_tiles_all = tile_end - tile_start_all
        if total_tiles_all <= Int32(0):
            _exit_thread()

        my_tile_lo = tile_start_all
        my_tile_hi = tile_end
        if const_expr(self.split_kv):
            tiles_per_split = (total_tiles_all + num_splits - Int32(1)) // num_splits
            tiles_per_split = cutlass.select_(
                tiles_per_split < Int32(self.min_stages_per_split),
                Int32(self.min_stages_per_split),
                tiles_per_split,
            )
            active_splits = (total_tiles_all + tiles_per_split - Int32(1)) // tiles_per_split
            if split_idx >= active_splits:
                _exit_thread()
            my_tile_lo = tile_start_all + split_idx * tiles_per_split
            my_tile_hi = cutlass.select_(
                my_tile_lo + tiles_per_split < tile_end,
                my_tile_lo + tiles_per_split,
                tile_end,
            )

        ntiles = my_tile_hi - my_tile_lo
        if ntiles <= Int32(0):
            _exit_thread()

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        full_K = storage.full_K.data_ptr()
        full_V = storage.full_V.data_ptr()
        empty = storage.empty.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_K + tidx, Int32(1))
            cute.arch.mbarrier_init(full_V + tidx, Int32(1))
            cute.arch.mbarrier_init(empty + tidx, Int32(NCW))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((self.shared_storage_bytes,), stride=(1,))
        )
        sKStageBytes = cute.make_tensor(
            payload_u8.iterator, cute.make_layout((self.kv_plane_total_bytes,), stride=(1,))
        )
        sVStageBytes = cute.make_tensor(
            payload_u8.iterator + Int32(self.kv_plane_total_bytes),
            cute.make_layout((self.kv_plane_total_bytes,), stride=(1,)),
        )

        # TMA views and warp-0 inline producer state (zero extra producer warp so
        # 8-warp CTAs keep full 255 regs/thread and 12-warp CTAs keep 168 regs/thread).
        plane_stage_layout = _paged_kv_tma_plane_stage_layout(STAGE, 128, S)
        plane_elems = S * STAGE * 128
        mKHead = mKCacheT[None, None, kv_head_idx, None]
        mVHead = mVCacheT[None, None, kv_head_idx, None]
        sK0 = _get_memrange_tensor(
            _make_payload_memrange(payload_u8, cutlass.Uint8, 0, plane_elems),
            plane_stage_layout,
        )
        sV0 = _get_memrange_tensor(
            _make_payload_memrange(
                payload_u8, cutlass.Uint8, self.kv_plane_total_bytes, plane_elems
            ),
            plane_stage_layout,
        )
        gK0 = cute.local_tile(mKHead, (STAGE, 128), (None, 0, None))
        gV0 = cute.local_tile(mVHead, (STAGE, 128), (None, 0, None))
        sK_part, gK_part = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK0, 0, 2),
            cute.group_modes(gK0, 0, 2),
        )
        sV_part, gV_part = cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV0, 0, 2),
            cute.group_modes(gV0, 0, 2),
        )
        page_table_width = mPageTable.shape[1]
        stages_per_page = Int32(self.page_size // STAGE)
        pt_base = my_tile_lo // stages_per_page
        pid_lane = Int32(0)

        # Pre-issue the first min(S, ntiles) TMA stages from warp 0 before Q loading
        # so initial global->shared KV transfers overlap with Q register loads.
        if warp == Int32(0):
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(not self.is_window):
                pt_idx = pt_base + lane
                if pt_idx < page_table_width:
                    pid_lane = mPageTable[request_idx, pt_idx]
            preload = cutlass.select_(ntiles < Int32(S), ntiles, Int32(S))
            tp = Int32(0)
            while tp < preload:
                abs_tile = my_tile_lo + tp
                if const_expr(self.is_window):
                    block_id = request_idx
                    tile_in_block = abs_tile
                else:
                    page_idx = abs_tile // stages_per_page
                    block_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
                    tile_in_block = abs_tile - page_idx * stages_per_page
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full_K + tp, self.kv_plane_stage_bytes)
                    cute.arch.mbarrier_arrive_and_expect_tx(full_V + tp, self.kv_plane_stage_bytes)
                cute.copy(
                    tma_atom_K,
                    gK_part[None, tile_in_block, block_id],
                    sK_part[None, tp],
                    tma_bar_ptr=full_K + tp,
                )
                cute.copy(
                    tma_atom_V,
                    gV_part[None, tile_in_block, block_id],
                    sV_part[None, tp],
                    tma_bar_ptr=full_V + tp,
                )
                tp += Int32(1)

        head_local = warp // Int32(self.warps_per_head) if const_expr(self.grouped) else Int32(0)
        tok_group = warp % Int32(self.warps_per_head) if const_expr(self.grouped) else warp
        q_head = (
            kv_head_idx * Int32(self.group_size) + head_local
            if const_expr(self.grouped)
            else by
        )
        warp_tok_base = q_tile_lo + tok_group * Int32(16)
        lane_group = lane // Int32(4)
        lane_quad = lane % Int32(4)
        lane_pair_base = Int32(2) * lane_quad

        o_frag = cute.make_rmem_tensor(cute.make_layout((8, 8), stride=(8, 1)), Float32)
        m_frag = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Float32)
        d_frag = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Float32)
        s_frag = cute.make_rmem_tensor(cute.make_layout((N_SUB, 8), stride=(8, 1)), Float32)
        p_frag = cute.make_rmem_tensor(cute.make_layout((N_SUB, 4), stride=(4, 1)), Uint32)
        q_regs_shape = (1, 1) if const_expr(self.q_in_smem) else (8, 4)
        q_regs_stride = (1, 1) if const_expr(self.q_in_smem) else (4, 1)
        q_regs = cute.make_rmem_tensor(cute.make_layout(q_regs_shape, stride=q_regs_stride), Uint32)
        row_valid = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_token = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)

        for row_slot in cutlass.range_constexpr(2):
            tok_loc = warp_tok_base + lane_group + Int32(8 * row_slot)
            row_valid[row_slot] = Int32(tok_loc < qo_len)
            q_token[row_slot] = tok_loc

        q_smem_warp = Int32(0)
        if const_expr(self.q_in_smem):
            q_smem_warp = shared_ptr_to_u32(
                payload_u8.iterator + Int32(2 * self.kv_plane_total_bytes) + warp * Int32(4096)
            )

        mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
        q_t_bytes = Int64(mQ.stride[0]) * Int64(2)
        q_h_bytes = Int64(mQ.stride[1]) * Int64(2)
        row_byte0 = Int64(q_start + q_token[0]) * q_t_bytes + Int64(q_head) * q_h_bytes
        row_byte1 = Int64(q_start + q_token[1]) * q_t_bytes + Int64(q_head) * q_h_bytes
        for mma_d in cutlass.range_constexpr(8):
            col = mma_d * 16 + 4 * lane_quad
            a0 = Uint32(0)
            a2 = Uint32(0)
            a1 = Uint32(0)
            a3 = Uint32(0)
            if row_valid[0] != Int32(0):
                a0, a2 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(mQBytes, row_byte0 + Int64(col * 2))
                )
            if row_valid[1] != Int32(0):
                a1, a3 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(mQBytes, row_byte1 + Int64(col * 2))
                )
            if const_expr(self.q_in_smem):
                st_shared_v4_u32(
                    q_smem_warp + Int32(mma_d * 512) + lane * Int32(16), a0, a1, a2, a3
                )
            else:
                q_regs[mma_d, 0] = a0
                q_regs[mma_d, 1] = a1
                q_regs[mma_d, 2] = a2
                q_regs[mma_d, 3] = a3

        for mma_d in cutlass.range_constexpr(8):
            for reg_id in cutlass.range_constexpr(8):
                o_frag[mma_d, reg_id] = Float32(0.0)
        for row_slot in cutlass.range_constexpr(2):
            m_frag[row_slot] = Float32(-Float32.inf)
            d_frag[row_slot] = Float32(0.0)

        # ---- Region 0: Far prefix (full attention only; unmasked, unbiased, full BN_SUB blocks) ----
        t = Int32(0)
        if const_expr(not self.is_window):
            min_cta_pos = start + q_tile_lo
            prefix_lim = min_cta_pos - Int32(self.rel_extent - 1)
            prefix_tile_end = cutlass.select_(
                prefix_lim > Int32(0), prefix_lim // Int32(STAGE), Int32(0)
            )
            if prefix_tile_end < my_tile_lo:
                prefix_tile_end = my_tile_lo
            if prefix_tile_end > my_tile_hi:
                prefix_tile_end = my_tile_hi
            prefix_count = prefix_tile_end - my_tile_lo
            sm_scale_log2 = sm_scale * Float32(LOG2_E)

            while t < prefix_count:
                buf = t % Int32(S)
                use = t // Int32(S)
                stage_off = buf * Int32(self.kv_plane_stage_bytes)
                k_smem = shared_ptr_to_u32(sKStageBytes.iterator + stage_off)
                v_smem = shared_ptr_to_u32(sVStageBytes.iterator + stage_off)
                cute.arch.mbarrier_wait(full_K + buf, phase=use & Int32(1))
                if const_expr(BN_SUB == STAGE):
                    _qk_fp8_multi(
                        s_frag, q_regs, q_smem_warp, k_smem, lane, Int32(0), N_SUB, self.q_in_smem
                    )
                    _online_softmax_far_prefix(
                        s_frag, o_frag, m_frag, d_frag, p_frag, sm_scale_log2, N_SUB
                    )
                    cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                    _pv_fp8_unmasked_multi(o_frag, p_frag, v_smem, lane, Int32(0), N_SUB)
                else:
                    for blk in cutlass.range_constexpr(STAGE // BN_SUB):
                        kb = Int32(blk * BN_SUB)
                        _qk_fp8_multi(
                            s_frag, q_regs, q_smem_warp, k_smem, lane, kb, N_SUB, self.q_in_smem
                        )
                        _online_softmax_far_prefix(
                            s_frag, o_frag, m_frag, d_frag, p_frag, sm_scale_log2, N_SUB
                        )
                        if const_expr(blk == 0):
                            cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                        _pv_fp8_unmasked_multi(o_frag, p_frag, v_smem, lane, kb, N_SUB)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(empty + buf)
                if warp == Int32(0):
                    t_next = t + Int32(S)
                    if t_next < ntiles:
                        cute.arch.mbarrier_wait(empty + buf, phase=use & Int32(1))
                        cute.arch.fence_view_async_shared()
                        abs_tile = my_tile_lo + t_next
                        page_idx = abs_tile // stages_per_page
                        if page_idx - pt_base >= Int32(32):
                            pt_base = pt_base + Int32(32)
                            pt_idx = pt_base + lane
                            pid_lane = Int32(0)
                            if pt_idx < page_table_width:
                                pid_lane = mPageTable[request_idx, pt_idx]
                        block_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
                        tile_in_block = abs_tile - page_idx * stages_per_page
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full_K + buf, self.kv_plane_stage_bytes)
                            cute.arch.mbarrier_arrive_and_expect_tx(full_V + buf, self.kv_plane_stage_bytes)
                        cute.copy(
                            tma_atom_K,
                            gK_part[None, tile_in_block, block_id],
                            sK_part[None, buf],
                            tma_bar_ptr=full_K + buf,
                        )
                        cute.copy(
                            tma_atom_V,
                            gV_part[None, tile_in_block, block_id],
                            sV_part[None, buf],
                            tma_bar_ptr=full_V + buf,
                        )
                t += Int32(1)

        # ---- Region 1: Biased / causal / windowed tail (bias pointers live only here) ----
        causal_hi = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        window_lo = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        bias_ptr = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int64)
        mBiasBytes = cute.flatten(cute.recast_tensor(mBias, cutlass.Uint8))
        bias_t_bytes = Int64(mBias.stride[0]) * Int64(2)
        bias_h_bytes = Int64(mBias.stride[1]) * Int64(2)
        for row_slot in cutlass.range_constexpr(2):
            tok_loc = q_token[row_slot]
            pos = base_offset + tok_loc
            causal_hi[row_slot] = cutlass.select_(row_valid[row_slot] != Int32(0), pos, Int32(-1))
            if const_expr(self.is_window):
                wlo = pos - Int32(self.window_left)
                window_lo[row_slot] = cutlass.select_(wlo > valid_lo, wlo, valid_lo)
            else:
                window_lo[row_slot] = Int32(0)
            bias_ptr[row_slot] = get_ptr_as_int64(
                mBiasBytes,
                Int64(q_start + tok_loc) * bias_t_bytes + Int64(q_head) * bias_h_bytes,
            )

        # Use the fixed 16-row warp upper bound for interior-tile predicates so
        # valid rows in a partial final warp take the same softmax reduction
        # schedule regardless of qo_len; invalid rows are guarded per row_slot.
        warp_tok_hi = warp_tok_base + Int32(15)
        warp_max_tok = cutlass.select_(
            warp_tok_hi < qo_len,
            warp_tok_hi,
            qo_len - Int32(1),
        )
        causal_hi_min = base_offset + warp_tok_base
        causal_hi_max = base_offset + warp_max_tok
        causal_hi_warp_max = base_offset + warp_tok_hi
        bias_min_key = causal_hi_min - Int32(self.rel_extent - 1)
        bias_all_key = causal_hi_warp_max - Int32(self.rel_extent - 1)
        warp_win_lo_min = Int32(0)
        warp_win_lo_max = Int32(0)
        if const_expr(self.is_window):
            w_min = causal_hi_min - Int32(self.window_left)
            warp_win_lo_min = cutlass.select_(w_min > valid_lo, w_min, valid_lo)
            w_max = causal_hi_warp_max - Int32(self.window_left)
            warp_win_lo_max = cutlass.select_(w_max > valid_lo, w_max, valid_lo)

        while t < ntiles:
            buf = t % Int32(S)
            use = t // Int32(S)
            stage_off = buf * Int32(self.kv_plane_stage_bytes)
            k_smem = shared_ptr_to_u32(sKStageBytes.iterator + stage_off)
            v_smem = shared_ptr_to_u32(sVStageBytes.iterator + stage_off)
            tile_base = (my_tile_lo + t) * Int32(STAGE)
            tile_tokens = cutlass.select_(
                tile_base + Int32(STAGE) <= seq_hi, Int32(STAGE), seq_hi - tile_base
            )
            tile_interior_biased = (
                (tile_tokens == Int32(STAGE))
                and (tile_base + Int32(STAGE - 1) <= causal_hi_min)
                and (tile_base >= bias_all_key)
            )
            if const_expr(self.is_window):
                tile_interior_biased = tile_interior_biased and (tile_base >= warp_win_lo_max)

            cute.arch.mbarrier_wait(full_K + buf, phase=use & Int32(1))
            if tile_interior_biased:
                if const_expr(BN_SUB == STAGE):
                    _qk_fp8_multi(
                        s_frag, q_regs, q_smem_warp, k_smem, lane, Int32(0), N_SUB, self.q_in_smem
                    )
                    _online_softmax_biased_interior(
                        s_frag,
                        o_frag,
                        m_frag,
                        d_frag,
                        p_frag,
                        tile_base,
                        lane_pair_base,
                        row_valid,
                        causal_hi,
                        bias_ptr,
                        sm_scale,
                        N_SUB,
                    )
                    cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                    _pv_fp8_unmasked_multi(o_frag, p_frag, v_smem, lane, Int32(0), N_SUB)
                else:
                    for blk in cutlass.range_constexpr(STAGE // BN_SUB):
                        kb = Int32(blk * BN_SUB)
                        _qk_fp8_multi(
                            s_frag, q_regs, q_smem_warp, k_smem, lane, kb, N_SUB, self.q_in_smem
                        )
                        _online_softmax_biased_interior(
                            s_frag,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            tile_base + kb,
                            lane_pair_base,
                            row_valid,
                            causal_hi,
                            bias_ptr,
                            sm_scale,
                            N_SUB,
                        )
                        if const_expr(blk == 0):
                            cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                        _pv_fp8_unmasked_multi(o_frag, p_frag, v_smem, lane, kb, N_SUB)
            else:
                cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                win_lo_rel = (warp_win_lo_min - tile_base) if const_expr(self.is_window) else Int32(0)
                for sub in cutlass.range_constexpr(STAGE // 16):
                    sub_key = Int32(sub * 16)
                    sub_base = tile_base + sub_key
                    sub_live = (sub_key < tile_tokens) and (sub_base <= causal_hi_max)
                    if const_expr(self.is_window):
                        sub_live = sub_live and (sub_base + Int32(15) >= warp_win_lo_min)
                    if sub_live:
                        _qk_fp8_multi(
                            s_frag, q_regs, q_smem_warp, k_smem, lane, sub_key, 1, self.q_in_smem
                        )
                        interior = (tile_tokens >= sub_key + Int32(16)) and (
                            sub_base + Int32(15) <= causal_hi_min
                        )
                        if const_expr(self.is_window):
                            interior = interior and (sub_base >= warp_win_lo_max)
                        any_biased = (sub_base + Int32(15)) >= bias_min_key
                        all_biased = sub_base >= bias_all_key
                        _tail_scale_bias_mask_softmax(
                            s_frag,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            sub_key,
                            tile_base,
                            tile_tokens,
                            lane_pair_base,
                            row_valid,
                            causal_hi,
                            window_lo,
                            bias_ptr,
                            sm_scale,
                            interior,
                            all_biased,
                            any_biased,
                            self.is_window,
                            self.rel_extent,
                        )
                        if interior:
                            _pv_fp8_unmasked_multi(o_frag, p_frag, v_smem, lane, sub_key, 1)
                        else:
                            _pv_fp8_masked_one(
                                o_frag, p_frag, v_smem, lane, sub_key, tile_tokens, win_lo_rel
                            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(empty + buf)
            if warp == Int32(0):
                t_next = t + Int32(S)
                if t_next < ntiles:
                    cute.arch.mbarrier_wait(empty + buf, phase=use & Int32(1))
                    cute.arch.fence_view_async_shared()
                    abs_tile = my_tile_lo + t_next
                    if const_expr(self.is_window):
                        block_id = request_idx
                        tile_in_block = abs_tile
                    else:
                        page_idx = abs_tile // stages_per_page
                        if page_idx - pt_base >= Int32(32):
                            pt_base = pt_base + Int32(32)
                            pt_idx = pt_base + lane
                            pid_lane = Int32(0)
                            if pt_idx < page_table_width:
                                pid_lane = mPageTable[request_idx, pt_idx]
                        block_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
                        tile_in_block = abs_tile - page_idx * stages_per_page
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full_K + buf, self.kv_plane_stage_bytes)
                        cute.arch.mbarrier_arrive_and_expect_tx(full_V + buf, self.kv_plane_stage_bytes)
                    cute.copy(
                        tma_atom_K,
                        gK_part[None, tile_in_block, block_id],
                        sK_part[None, buf],
                        tma_bar_ptr=full_K + buf,
                    )
                    cute.copy(
                        tma_atom_V,
                        gV_part[None, tile_in_block, block_id],
                        sV_part[None, buf],
                        tma_bar_ptr=full_V + buf,
                    )
            t += Int32(1)

        # ---- Epilogue: split partials or fused gate+sink BF16 output ----
        for row_slot in cutlass.range_constexpr(2):
            if row_valid[row_slot] != Int32(0):
                r0 = row_slot * 2 + 0
                r1 = row_slot * 2 + 1
                r4 = row_slot * 2 + 4
                r5 = row_slot * 2 + 5
                m_log2 = Float32(m_frag[row_slot])
                d_star = Float32(d_frag[row_slot])
                has_mass = m_log2 != -Float32.inf
                inv_d = Float32(0.0)
                lse_val = Float32(-Float32.inf)
                if has_mass and d_star > Float32(0.0):
                    inv_d = Float32(1.0) / d_star
                    lse_val = (m_log2 + cute.math.log2(d_star, fastmath=True)) * Float32(LN_2)
                q_row = q_start + q_token[row_slot]
                if const_expr(self.split_kv):
                    mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
                    part_base = (
                        Int64(q_row) * Int64(mOPart.stride[0])
                        + Int64(q_head) * Int64(mOPart.stride[1])
                        + Int64(split_idx) * Int64(mOPart.stride[2])
                    ) * Int64(4)
                    for mma_d in cutlass.range_constexpr(8):
                        dim_low = mma_d * 16 + lane_pair_base
                        st_global_v2_f32(
                            get_ptr_as_int64(mOPartBytes, part_base + Int64(dim_low * 4)),
                            o_frag[mma_d, r0] * inv_d,
                            o_frag[mma_d, r1] * inv_d,
                        )
                        st_global_v2_f32(
                            get_ptr_as_int64(mOPartBytes, part_base + Int64((dim_low + 8) * 4)),
                            o_frag[mma_d, r4] * inv_d,
                            o_frag[mma_d, r5] * inv_d,
                        )
                    if lane_quad == Int32(0):
                        mLsePart[q_row, q_head, split_idx] = lse_val
                else:
                    mGateBytes = cute.flatten(cute.recast_tensor(mGate, cutlass.Uint8))
                    mSinkBytes = cute.flatten(cute.recast_tensor(mSink, cutlass.Uint8))
                    mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                    sink_sig = Float32(0.0)
                    if has_mass and d_star > Float32(0.0):
                        s_u32 = _ld_global_nc_u16(
                            get_ptr_as_int64(
                                mSinkBytes,
                                Int64(q_row) * Int64(mSink.stride[0]) * Int64(2)
                                + Int64(q_head) * Int64(2),
                            )
                        )
                        sink_sig = _sigmoid_f32(lse_val - u32_as_f32(s_u32 << Uint32(16)))
                    gate_row = (
                        Int64(q_row) * Int64(mGate.stride[0])
                        + Int64(q_head) * Int64(mGate.stride[1])
                    ) * Int64(2)
                    gated_row = (
                        Int64(q_row) * Int64(mGated.stride[0])
                        + Int64(q_head) * Int64(mGated.stride[1])
                    ) * Int64(2)
                    top_mask = Uint32(0xFFFF0000)
                    for mma_d in cutlass.range_constexpr(8):
                        dim_low = mma_d * 16 + lane_pair_base
                        g01 = ld_global_nc_u32(
                            get_ptr_as_int64(mGateBytes, gate_row + Int64(dim_low * 2))
                        )
                        g89 = ld_global_nc_u32(
                            get_ptr_as_int64(mGateBytes, gate_row + Int64((dim_low + 8) * 2))
                        )
                        y0 = (o_frag[mma_d, r0] * inv_d) * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
                        y1 = (o_frag[mma_d, r1] * inv_d) * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
                        y2 = (o_frag[mma_d, r4] * inv_d) * _sigmoid_f32(u32_as_f32(g89 << Uint32(16))) * sink_sig
                        y3 = (o_frag[mma_d, r5] * inv_d) * _sigmoid_f32(u32_as_f32(g89 & top_mask)) * sink_sig
                        st_global_u32(
                            get_ptr_as_int64(mGatedBytes, gated_row + Int64(dim_low * 2)),
                            pack_f32x2_to_bfloat2(y0, y1),
                        )
                        st_global_u32(
                            get_ptr_as_int64(mGatedBytes, gated_row + Int64((dim_low + 8) * 2)),
                            pack_f32x2_to_bfloat2(y2, y3),
                        )


class RelativePrefillMergeKernel:
    """Register-only warp-per-(token, head) split-KV merge with fused per-token gate-sink."""

    def __init__(
        self,
        *,
        num_q_heads: int = 24,
        q_tile_tokens: int = 64,
        stage_rows: int = 64,
        min_stages_per_split: int = 8,
        warps_per_cta: int = 8,
    ):
        self.num_q_heads = int(num_q_heads)
        self.q_tile_tokens = int(q_tile_tokens)
        self.stage_rows = int(stage_rows)
        self.min_stages_per_split = int(min_stages_per_split)
        self.warps_per_cta = int(warps_per_cta)

    @cute.jit
    def __call__(
        self,
        mOPart: cute.Tensor,  # [T, Hq, S, 128] f32
        mLsePart: cute.Tensor,  # [T, Hq, S] f32
        mLengths: cute.Tensor,  # [B] i32
        mCuSeqlensQ: cute.Tensor,  # [B + 1] i32
        mGate: cute.Tensor,  # [T, Hq, 128] bf16
        mSink: cute.Tensor,  # [T, Hq] bf16
        mGated: cute.Tensor,  # [T, Hq, 128] bf16
        stream: cuda.CUstream,
    ):
        total_cap = mGated.shape[0]
        grid_x = (total_cap + Int32(self.warps_per_cta - 1)) // Int32(self.warps_per_cta)
        self.kernel(mOPart, mLsePart, mLengths, mCuSeqlensQ, mGate, mSink, mGated).launch(
            grid=(grid_x, self.num_q_heads, 1),
            block=[32, self.warps_per_cta, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mLengths: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mGate: cute.Tensor,
        mSink: cute.Tensor,
        mGated: cute.Tensor,
    ):
        lane, warp_y, _ = cute.arch.thread_idx()
        bx, head, _ = cute.arch.block_idx()
        q_row = bx * Int32(self.warps_per_cta) + warp_y
        total_cap = mGated.shape[0]
        if q_row >= total_cap:
            _exit_thread()

        num_reqs = mLengths.shape[0]
        total_live = mCuSeqlensQ[num_reqs]
        mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
        gated_ptr = get_ptr_as_int64(
            mGatedBytes,
            (Int64(q_row) * Int64(mGated.stride[0]) + Int64(head) * Int64(mGated.stride[1]))
            * Int64(2)
            + Int64(lane * 8),
        )
        if q_row >= total_live:
            st_global_v2_u32(gated_ptr, Uint32(0), Uint32(0))
            _exit_thread()

        # Locate request owning q_row (single iteration for prefill B=1).
        req = Int32(0)
        if num_reqs > Int32(1):
            r = Int32(0)
            while r < num_reqs:
                if q_row >= mCuSeqlensQ[r] and q_row < mCuSeqlensQ[r + 1]:
                    req = r
                r += Int32(1)
        q_start = mCuSeqlensQ[req]
        qo_len = mCuSeqlensQ[req + 1] - q_start
        length = mLengths[req]
        start = length - qo_len
        tok_loc = q_row - q_start
        q_tile_idx = tok_loc // Int32(self.q_tile_tokens)
        q_tile_hi = cutlass.select_(
            (q_tile_idx + Int32(1)) * Int32(self.q_tile_tokens) < qo_len,
            (q_tile_idx + Int32(1)) * Int32(self.q_tile_tokens),
            qo_len,
        )
        max_key_excl = start + q_tile_hi
        if max_key_excl > length:
            max_key_excl = length
        tile_end = (max_key_excl + Int32(self.stage_rows - 1)) // Int32(self.stage_rows)
        num_splits = mLsePart.shape[2]
        tiles_per_split = (tile_end + num_splits - Int32(1)) // num_splits
        tiles_per_split = cutlass.select_(
            tiles_per_split < Int32(self.min_stages_per_split),
            Int32(self.min_stages_per_split),
            tiles_per_split,
        )
        active = (tile_end + tiles_per_split - Int32(1)) // tiles_per_split

        log2_e = Float32(LOG2_E)
        lv_lane = Float32(-Float32.inf)
        if lane < active:
            lv_lane = mLsePart[q_row, head, lane]
        m_row = lv_lane
        for offset in cutlass.range_constexpr(5):
            m_row = _max_nan_f32(m_row, cute.arch.shuffle_sync_bfly(m_row, offset=1 << offset))
        w_lane = Float32(0.0)
        if m_row != -Float32.inf and lv_lane != -Float32.inf:
            w_lane = _exp2_approx_ftz_f32((lv_lane - m_row) * log2_e)
        s_row = cute.arch.warp_reduction_sum(w_lane)

        mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
        row_base = (
            Int64(q_row) * Int64(mOPart.stride[0]) + Int64(head) * Int64(mOPart.stride[1])
        ) * Int64(4) + Int64(lane * 16)
        split_stride_bytes = Int64(mOPart.stride[2]) * Int64(4)
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        if m_row != -Float32.inf:
            sp = Int32(0)
            while sp < active:
                w = cute.arch.shuffle_sync(w_lane, sp)
                o0, o1, o2, o3 = ld_global_v4_f32(
                    get_ptr_as_int64(mOPartBytes, row_base + Int64(sp) * split_stride_bytes)
                )
                acc0 += w * o0
                acc1 += w * o1
                acc2 += w * o2
                acc3 += w * o3
                sp += Int32(1)

        inv_s = Float32(0.0)
        sink_sig = Float32(0.0)
        if m_row != -Float32.inf and s_row > Float32(0.0):
            inv_s = Float32(1.0) / s_row
            lse = m_row + cute.math.log2(s_row, fastmath=True) * Float32(LN_2)
            mSinkBytes = cute.flatten(cute.recast_tensor(mSink, cutlass.Uint8))
            s_u32 = _ld_global_nc_u16(
                get_ptr_as_int64(
                    mSinkBytes,
                    Int64(q_row) * Int64(mSink.stride[0]) * Int64(2) + Int64(head) * Int64(2),
                )
            )
            sink_sig = _sigmoid_f32(lse - u32_as_f32(s_u32 << Uint32(16)))
        mGateBytes = cute.flatten(cute.recast_tensor(mGate, cutlass.Uint8))
        gate_ptr = get_ptr_as_int64(
            mGateBytes,
            (Int64(q_row) * Int64(mGate.stride[0]) + Int64(head) * Int64(mGate.stride[1]))
            * Int64(2)
            + Int64(lane * 8),
        )
        g01, g23 = ld_global_nc_v2_u32(gate_ptr)
        top_mask = Uint32(0xFFFF0000)
        y0 = (acc0 * inv_s) * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
        y1 = (acc1 * inv_s) * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
        y2 = (acc2 * inv_s) * _sigmoid_f32(u32_as_f32(g23 << Uint32(16))) * sink_sig
        y3 = (acc3 * inv_s) * _sigmoid_f32(u32_as_f32(g23 & top_mask)) * sink_sig
        st_global_v2_u32(
            gated_ptr,
            pack_f32x2_to_bfloat2(y0, y1),
            pack_f32x2_to_bfloat2(y2, y3),
        )


__all__ = ["RelativePrefillKernel", "RelativePrefillMergeKernel"]
