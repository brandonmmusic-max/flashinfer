"""Warp-specialized SM120 TMA relative-position decode/verify attention with fused KV write and gate-sink.

Two lean kernels cover both sliding-window rings and paged full-attention layers without ring staging:

1. ``RelativeDecodeKernel``:
   - Producer warp issues 4D TMA stages ``(STAGE, 128)`` directly from either the circular FP8 window
     ring ``[state_rows, window, Hkv, 128]`` (any 16B-aligned row stride) or paged FP8 cache
     ``[num_pages, page_size, Hkv, 128]``.
   - Consumer warps pack ``(qo_len * group_size)`` query rows into 16-row MMA row-groups and partition
     key tiles across key-groups, widening FP8 K/V via SM120 native ``cvt.rn.bf16x2.e4m3x2`` and
     accumulating FP32 scores with per-query relative bias ``Bias[q_row, q_head, pos - k]``.
   - The designated split (``split_idx == active_splits - 1``) quantizes the ``qo_len`` new BF16 K/V
     rows once to FP8 E4M3 (scale 1, RNE, saturating), writes them directly to the ring (sliding
     decode), compact staging slice (sliding verify), or KV pages (full decode/verify), and attends
     over them in the same CTA with zero data races against historical reads.
   - Merges consumer key-groups through shared memory and either writes FP32 split partials
     (``split_kv=True``) or directly applies per-token gate and sink into BF16 output
     (``split_kv=False``).

2. ``RelativeDecodeMergeKernel``:
   - Parallel split reduction across ``active_splits`` with 4 warps per head, fusing the per-token
     per-head output gate and sink sigmoid into a single BF16 round and zeroing padded rows.
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
    ld_global_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m16n16x1_trans_b8,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v2_f32,
    st_global_v2_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n16k16_f32
from b12x._lib.intrinsics.packed import (
    cvt_bf16x2x2_to_e4m3x4,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    pack_f32x2_to_bfloat2,
)
from b12x.attention._shared.cute import copy as cute_copy
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
    """Permute ``[N, P, Hkv, 128]`` FP8 cache to ``[P, 128, Hkv, N]`` for 4D TMA.

    Works for any 16B-aligned outer stride ``stride(0)`` (both circular window rings whose
    row stride is 256B-aligned and 128-token KV pages) without requiring ``stride(0)`` to be
    a multiple of ``stage_rows * stride(1)``.
    """
    t_4d = cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
    return _assume_paged_kv_tma_source_aligned(t_4d)


@cute.jit
def _qk_fp8_into_sfrag(frag_S, q_regs, k_plane0: Int32, lane: Int32, key_base: Int32):
    """S += Q K^T over one 16-key sub-tile of 128-byte FP8 K rows in swizzled SMEM."""
    mat = lane // Int32(8)
    key_row = key_base + (mat % Int32(2)) * Int32(8) + lane % Int32(8)
    row_swizzle = key_row % Int32(8)
    for pair in cutlass.range_constexpr(4):
        step = 2 * pair
        chunk = Int32(step) + mat // Int32(2)
        r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(
            k_plane0 + (key_row * Int32(8) + (chunk ^ row_swizzle)) * Int32(16)
        )
        for half in cutlass.range_constexpr(2):
            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r0 if half == 0 else r2)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r1 if half == 0 else r3)
            s_idx = step + half
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                frag_S[0, 0, 0],
                frag_S[0, 0, 1],
                frag_S[0, 0, 2],
                frag_S[0, 0, 3],
                frag_S[0, 0, 4],
                frag_S[0, 0, 5],
                frag_S[0, 0, 6],
                frag_S[0, 0, 7],
                q_regs[s_idx, 0, 0],
                q_regs[s_idx, 0, 1],
                q_regs[s_idx, 0, 2],
                q_regs[s_idx, 0, 3],
                b0,
                b1,
                b2,
                b3,
            )
            frag_S[0, 0, 0] = d0
            frag_S[0, 0, 1] = d1
            frag_S[0, 0, 2] = d2
            frag_S[0, 0, 3] = d3
            frag_S[0, 0, 4] = d4
            frag_S[0, 0, 5] = d5
            frag_S[0, 0, 6] = d6
            frag_S[0, 0, 7] = d7


@cute.jit
def _pv_fp8_into_ofrag(
    o_frag, p_frag, v_plane0: Int32, lane: Int32, key_base: Int32, valid_rows: Int32, window_lo: Int32
):
    """O += P V over one 16-key sub-tile of 128-byte FP8 V rows in swizzled SMEM."""
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
    for mma_d in cutlass.range_constexpr(8):
        f0, f1 = ldmatrix_m16n16x1_trans_b8(
            v_plane0 + (v_row * Int32(8) + (Int32(mma_d) ^ row_swizzle)) * Int32(16)
        )
        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
        b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
        b0 = _mask_pv_bf16_pair_windowed(b0, key, valid_rows, window_lo)
        b1 = _mask_pv_bf16_pair_windowed(b1, key + Int32(8), valid_rows, window_lo)
        b2 = _mask_pv_bf16_pair_windowed(b2, key, valid_rows, window_lo)
        b3 = _mask_pv_bf16_pair_windowed(b3, key + Int32(8), valid_rows, window_lo)
        d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            p_frag[0, 0, 0],
            p_frag[0, 0, 1],
            p_frag[0, 0, 2],
            p_frag[0, 0, 3],
            b0,
            b1,
            b2,
            b3,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7


@cute.jit
def _scale_bias_mask_subtile(
    frag_S: cute.Tensor,
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
    if interior and not any_biased:
        # Full-attention far prefix: no causal boundary, no window boundary, zero relative bias.
        for reg_id in cutlass.range_constexpr(8):
            frag_S[0, 0, reg_id] = frag_S[0, 0, reg_id] * sm_scale
    elif interior and all_biased:
        # Interior tile where every key has 0 <= dist < rel_extent for every valid row.
        sub_base = tile_base + key_base
        for row_slot in cutlass.range_constexpr(2):
            if row_valid[row_slot] != Int32(0):
                bptr = bias_ptr[row_slot]
                d0 = causal_hi[row_slot] - (sub_base + lane_pair_base)
                r0 = row_slot * 2 + 0
                r1 = row_slot * 2 + 1
                r4 = row_slot * 2 + 4
                r5 = row_slot * 2 + 5
                b0 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0) * Int64(2)) << Uint32(16))
                b1 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(1)) * Int64(2)) << Uint32(16))
                b4 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(8)) * Int64(2)) << Uint32(16))
                b5 = u32_as_f32(_ld_global_nc_u16(bptr + Int64(d0 - Int32(9)) * Int64(2)) << Uint32(16))
                frag_S[0, 0, r0] = frag_S[0, 0, r0] * sm_scale + b0
                frag_S[0, 0, r1] = frag_S[0, 0, r1] * sm_scale + b1
                frag_S[0, 0, r4] = frag_S[0, 0, r4] * sm_scale + b4
                frag_S[0, 0, r5] = frag_S[0, 0, r5] * sm_scale + b5
            else:
                frag_S[0, 0, row_slot * 2 + 0] = Float32(-Float32.inf)
                frag_S[0, 0, row_slot * 2 + 1] = Float32(-Float32.inf)
                frag_S[0, 0, row_slot * 2 + 4] = Float32(-Float32.inf)
                frag_S[0, 0, row_slot * 2 + 5] = Float32(-Float32.inf)
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
                s = frag_S[0, 0, reg_id] * sm_scale
                if any_biased:
                    dist = causal_hi[row_slot] - key_abs
                    if dist < Int32(rel_extent):
                        b_u32 = _ld_global_nc_u16(bias_ptr[row_slot] + Int64(dist) * Int64(2))
                        s = s + u32_as_f32(b_u32 << Uint32(16))
                frag_S[0, 0, reg_id] = s
            else:
                frag_S[0, 0, reg_id] = Float32(-Float32.inf)


@cute.jit
def _online_softmax_pack_p(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
):
    """Online FP32 softmax update on already-scaled-and-biased scores, packing BF16 P."""
    log2_e = Float32(LOG2_E)
    for row_slot in cutlass.range_constexpr(2):
        m_prev = Float32(m_frag[0, row_slot])
        m_local = _max_nan_f32(
            _max_nan_f32(s_frag[0, 0, row_slot * 2 + 0], s_frag[0, 0, row_slot * 2 + 1]),
            _max_nan_f32(s_frag[0, 0, row_slot * 2 + 4], s_frag[0, 0, row_slot * 2 + 5]),
        )
        m_new = _max_nan_f32(m_prev, m_local)
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        scale_term = Float32(1.0)
        if m_new != -Float32.inf:
            if m_prev == -Float32.inf:
                scale_term = Float32(0.0)
            else:
                scale_term = _exp2_approx_ftz_f32((m_prev - m_new) * log2_e)
        d_frag[0, row_slot] = Float32(d_frag[0, row_slot] * scale_term)
        for mma_d in cutlass.range_constexpr(8):
            o_frag[0, mma_d, row_slot * 2 + 0] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 1] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 4] *= scale_term
            o_frag[0, mma_d, row_slot * 2 + 5] *= scale_term

        p0 = Float32(0.0)
        p1 = Float32(0.0)
        p2 = Float32(0.0)
        p3 = Float32(0.0)
        if m_new != -Float32.inf:
            p0 = _exp2_approx_ftz_f32((s_frag[0, 0, row_slot * 2 + 0] - m_new) * log2_e)
            p1 = _exp2_approx_ftz_f32((s_frag[0, 0, row_slot * 2 + 1] - m_new) * log2_e)
            p2 = _exp2_approx_ftz_f32((s_frag[0, 0, row_slot * 2 + 4] - m_new) * log2_e)
            p3 = _exp2_approx_ftz_f32((s_frag[0, 0, row_slot * 2 + 5] - m_new) * log2_e)
        p_sum = Float32((p0 + p1) + (p2 + p3))
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=2)
        p_sum += cute.arch.shuffle_sync_bfly(p_sum, offset=1)
        d_frag[0, row_slot] += p_sum
        p_frag[0, 0, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
        p_frag[0, 0, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)
        m_frag[0, row_slot] = Float32(m_new)


class RelativeDecodeKernel:
    """Warp-specialized TMA split-KV relative decode/verify attention with fused KV write."""

    def __init__(
        self,
        *,
        group_size: int = 3,
        max_q_per_req: int = 1,
        head_dim: int = 128,
        is_window: bool = True,
        commit_ring: bool = True,
        page_or_window: int = 1024,
        window_left: int = 1023,
        rel_extent: int = 1024,
        split_kv: bool = True,
        stage_rows: int = 16,
        num_stages: int = 6,
        num_consumer_warps: int = 4,
        min_tiles_per_split: int = 2,
    ):
        if head_dim != 128:
            raise ValueError("RelativeDecodeKernel requires head_dim == 128")
        if stage_rows not in (16, 32) or page_or_window % stage_rows != 0:
            raise ValueError(f"unsupported stage_rows={stage_rows} for page_or_window={page_or_window}")
        if max_q_per_req < 1 or max_q_per_req > 16:
            raise ValueError("max_q_per_req must be in 1..16")
        rows_cap = group_size * max_q_per_req
        if (rows_cap + 15) // 16 > num_consumer_warps:
            raise ValueError(f"row capacity {rows_cap} exceeds {num_consumer_warps} consumer warps")
        self.group_size = int(group_size)
        self.max_q_per_req = int(max_q_per_req)
        self.head_dim = int(head_dim)
        self.is_window = bool(is_window)
        self.commit_ring = bool(commit_ring)
        self.page_or_window = int(page_or_window)
        self.window_left = int(window_left)
        self.rel_extent = int(rel_extent)
        self.split_kv = bool(split_kv)
        self.stage_rows = int(stage_rows)
        self.num_stages = int(num_stages)
        self.ncw = int(num_consumer_warps)
        self.num_warps = self.ncw + 1
        self.min_tiles_per_split = int(min_tiles_per_split)
        self.kv_plane_stage_bytes = self.stage_rows * 128
        self.kv_plane_total_bytes = self.num_stages * self.kv_plane_stage_bytes
        self.k_bytes = self.kv_plane_total_bytes
        self.v_bytes = self.kv_plane_total_bytes
        self.comb_o_floats = self.ncw * 8 * 8 * 32
        self.comb_md_floats = self.ncw * 2 * 32
        comb_bytes = 0 if self.ncw == 1 else (self.comb_o_floats + 2 * self.comb_md_floats) * 4
        # Stage 0 holds the up-to-16 new tokens (2048B K + 2048B V) after the TMA loop.
        self.shared_storage_bytes = max(self.k_bytes + self.v_bytes, comb_bytes, 4096)
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
        mNewK: cute.Tensor,  # [T, Hkv, 128] bf16
        mNewV: cute.Tensor,  # [T, Hkv, 128] bf16
        mKCache: cute.Tensor,  # [N, P, Hkv, 128] u8 (e4m3)
        mVCache: cute.Tensor,  # [N, P, Hkv, 128] u8 (e4m3)
        mStagedK: cute.Tensor,  # [B, cap, Hkv, 128] u8 (verify sliding; dummy otherwise)
        mStagedV: cute.Tensor,  # [B, cap, Hkv, 128] u8
        mPageTable: cute.Tensor,  # [B, max_pages] i32
        mStateRows: cute.Tensor,  # [B] i64
        mStarts: cute.Tensor,  # [B] i64
        mCuSeqlensQ: cute.Tensor,  # [B + 1] i32
        mBias: cute.Tensor,  # [T, Hq, rel_extent] bf16
        mGate: cute.Tensor,  # [T, Hq, 128] bf16 (used when split_kv=False)
        mSink: cute.Tensor,  # [T, Hq] bf16 (used when split_kv=False)
        mGated: cute.Tensor,  # [T, Hq, 128] bf16 (used when split_kv=False)
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
        num_splits = mLsePart.shape[2] if const_expr(self.split_kv) else Int32(1)
        self.kernel(
            mQ,
            mNewK,
            mNewV,
            mKCache,
            mVCache,
            tma_tensor_K,
            tma_tensor_V,
            mStagedK,
            mStagedV,
            mPageTable,
            mStateRows,
            mStarts,
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
        ).launch(
            grid=(mStarts.shape[0], mNewK.shape[1], num_splits),
            block=[32, self.num_warps, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mNewK: cute.Tensor,
        mNewV: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mKCacheT: cute.Tensor,
        mVCacheT: cute.Tensor,
        mStagedK: cute.Tensor,
        mStagedV: cute.Tensor,
        mPageTable: cute.Tensor,
        mStateRows: cute.Tensor,
        mStarts: cute.Tensor,
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
    ):
        STAGE = self.stage_rows
        S = self.num_stages
        NCW = self.ncw
        lane, warp, _ = cute.arch.thread_idx()
        request_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        num_reqs, _, num_splits = cute.arch.grid_dim()
        tidx = lane + warp * Int32(32)

        # Zero padded output rows if split_kv=False.
        if const_expr(not self.split_kv):
            if request_idx == Int32(0) and split_idx == Int32(0):
                total_live = mCuSeqlensQ[num_reqs]
                total_cap = mGated.shape[0]
                if total_live < total_cap:
                    mGatedBytes0 = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                    g_t_s = Int64(mGated.stride[0]) * Int64(2)
                    g_h_s = Int64(mGated.stride[1]) * Int64(2)
                    rp = total_live + warp
                    while rp < total_cap:
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
                        rp += Int32(self.num_warps)

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        start = Int32(mStarts[request_idx])

        if qo_len <= Int32(0):
            if const_expr(self.is_window and not self.commit_ring):
                if split_idx == Int32(0):
                    sk_base = (
                        get_ptr_as_int64(mStagedK, 0)
                        + Int64(request_idx) * Int64(mStagedK.stride[0])
                        + Int64(kv_head_idx) * Int64(mStagedK.stride[2])
                    )
                    sv_base = (
                        get_ptr_as_int64(mStagedV, 0)
                        + Int64(request_idx) * Int64(mStagedV.stride[0])
                        + Int64(kv_head_idx) * Int64(mStagedV.stride[2])
                    )
                    sk_s1 = Int64(mStagedK.stride[1])
                    sv_s1 = Int64(mStagedV.stride[1])
                    idx = tidx
                    while idx < Int32(self.max_q_per_req * 8):
                        tok = idx >> Int32(3)
                        chunk = idx & Int32(7)
                        pos_s = Int64(self.page_or_window) + Int64(tok)
                        col_b = Int64(chunk * Int32(16))
                        st_global_v4_u32(sk_base + pos_s * sk_s1 + col_b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        st_global_v4_u32(sv_base + pos_s * sv_s1 + col_b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        idx += Int32(self.num_warps * 32)
            _exit_thread()

        first_key = Int32(0)
        if const_expr(self.is_window):
            first_key = cutlass.select_(
                start - Int32(self.window_left) > Int32(0),
                start - Int32(self.window_left),
                Int32(0),
            )
        tile_begin = first_key // Int32(STAGE)
        tile_end = (start + Int32(STAGE - 1)) // Int32(STAGE)
        num_hist_tiles = cutlass.select_(tile_end > tile_begin, tile_end - tile_begin, Int32(0))

        active_splits = Int32(1)
        my_tile_lo = tile_begin
        my_tile_hi = tile_end
        if const_expr(self.split_kv):
            if num_hist_tiles <= Int32(0):
                active_splits = Int32(1)
                my_tile_lo = Int32(0)
                my_tile_hi = Int32(0)
            else:
                tiles_per_split = (num_hist_tiles + num_splits - Int32(1)) // num_splits
                tiles_per_split = cutlass.select_(
                    tiles_per_split < Int32(self.min_tiles_per_split),
                    Int32(self.min_tiles_per_split),
                    tiles_per_split,
                )
                active_splits = (num_hist_tiles + tiles_per_split - Int32(1)) // tiles_per_split
                my_tile_lo = tile_begin + split_idx * tiles_per_split
                my_tile_hi = cutlass.select_(
                    my_tile_lo + tiles_per_split < tile_end,
                    my_tile_lo + tiles_per_split,
                    tile_end,
                )
            if split_idx >= active_splits:
                _exit_thread()

        ntiles = cutlass.select_(my_tile_hi > my_tile_lo, my_tile_hi - my_tile_lo, Int32(0))
        owns_new = split_idx == active_splits - Int32(1)

        group_size = Int32(self.group_size)
        head_base = kv_head_idx * group_size
        packed_rows = qo_len * group_size
        num_row_groups = (packed_rows + Int32(15)) // Int32(16)
        num_key_groups = Int32(NCW) // num_row_groups
        if num_key_groups > Int32(S):
            num_key_groups = Int32(S)
        stages_per_group = Int32(S) // num_key_groups

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        full_K = storage.full_K.data_ptr()
        full_V = storage.full_V.data_ptr()
        empty = storage.empty.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_K + tidx, Int32(1))
            cute.arch.mbarrier_init(full_V + tidx, Int32(1))
            cute.arch.mbarrier_init(empty + tidx, num_row_groups)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((self.shared_storage_bytes,), stride=(1,))
        )
        sKStageBytes = cute.make_tensor(
            payload_u8.iterator, cute.make_layout((self.k_bytes,), stride=(1,))
        )
        sVStageBytes = cute.make_tensor(
            payload_u8.iterator + Int32(self.k_bytes),
            cute.make_layout((self.v_bytes,), stride=(1,)),
        )

        is_producer = warp == Int32(NCW)
        is_consumer = warp < num_row_groups * num_key_groups
        row_group = warp % num_row_groups
        key_group = warp // num_row_groups

        o_frag = cute.make_rmem_tensor(cute.make_layout((1, 8, 8), stride=(64, 8, 1)), Float32)
        m_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        d_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        row_valid = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_token = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_head = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        causal_hi = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        window_lo = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        bias_ptr = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int64)
        q_regs = cute.make_rmem_tensor(cute.make_layout((8, 1, 4), stride=(4, 4, 1)), Uint32)

        lane_group = lane // Int32(4)
        lane_quad = lane % Int32(4)
        lane_pair_base = Int32(2) * lane_quad
        warp_row_base = row_group * Int32(16)
        frag_s_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
        p_frag = cute.make_rmem_tensor(cute.make_layout((1, 1, 4), stride=(4, 4, 1)), Uint32)

        state_row_i32 = Int32(0)
        if const_expr(self.is_window):
            state_row_i32 = Int32(mStateRows[request_idx])

        if is_producer:
            plane_stage_layout = _paged_kv_tma_plane_stage_layout(STAGE, 128, S)
            plane_elems = S * STAGE * 128
            # Slice 4D TMA tensor [P, 128, Hkv, N] at kv_head_idx -> [P, 128, N]
            mKHead = mKCacheT[None, None, kv_head_idx, None]
            mVHead = mVCacheT[None, None, kv_head_idx, None]
            sK0 = _get_memrange_tensor(
                _make_payload_memrange(payload_u8, cutlass.Uint8, 0, plane_elems),
                plane_stage_layout,
            )
            sV0 = _get_memrange_tensor(
                _make_payload_memrange(payload_u8, cutlass.Uint8, self.k_bytes, plane_elems),
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
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)

            page_table_width = mPageTable.shape[1]
            pt_base = (my_tile_lo * Int32(STAGE)) // Int32(self.page_or_window)
            pid_lane = Int32(0)
            if const_expr(not self.is_window):
                pt_idx = pt_base + lane
                if pt_idx < page_table_width:
                    pid_lane = mPageTable[request_idx, pt_idx]

            t = Int32(0)
            while t < ntiles:
                kg_t = t % num_key_groups
                j = t // num_key_groups
                buf = kg_t * stages_per_group + j % stages_per_group
                use = j // stages_per_group
                if use > Int32(0):
                    cute.arch.mbarrier_wait(empty + buf, phase=(use - Int32(1)) & Int32(1))
                token_base = (my_tile_lo + t) * Int32(STAGE)
                tile_in_block = Int32(0)
                block_id = state_row_i32
                if const_expr(self.is_window):
                    tile_in_block = (token_base % Int32(self.page_or_window)) // Int32(STAGE)
                else:
                    page_idx = token_base // Int32(self.page_or_window)
                    if page_idx - pt_base >= Int32(32):
                        pt_base = pt_base + Int32(32)
                        pt_idx = pt_base + lane
                        pid_lane = Int32(0)
                        if pt_idx < page_table_width:
                            pid_lane = mPageTable[request_idx, pt_idx]
                    block_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
                    tile_in_block = (token_base - page_idx * Int32(self.page_or_window)) // Int32(STAGE)
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
        elif is_consumer:
            mBiasBytes = cute.flatten(cute.recast_tensor(mBias, cutlass.Uint8))
            bias_t_bytes = Int64(mBias.stride[0]) * Int64(2)
            bias_h_bytes = Int64(mBias.stride[1]) * Int64(2)
            for row_slot in cutlass.range_constexpr(2):
                packed_row = warp_row_base + lane_group + Int32(8 * row_slot)
                valid_row = packed_row < packed_rows
                row_valid[row_slot] = Int32(valid_row)
                tok_loc = packed_row // group_size
                q_token[row_slot] = tok_loc
                hd = head_base + (packed_row - tok_loc * group_size)
                q_head[row_slot] = hd
                pos = start + tok_loc
                causal_hi[row_slot] = cutlass.select_(valid_row, pos, Int32(-1))
                window_lo[row_slot] = pos - Int32(self.window_left)
                bias_ptr[row_slot] = get_ptr_as_int64(
                    mBiasBytes,
                    Int64(q_start + tok_loc) * bias_t_bytes + Int64(hd) * bias_h_bytes,
                )

            warp_min_tok = warp_row_base // group_size
            warp_max_tok = cutlass.select_(
                (warp_row_base + Int32(15)) // group_size < qo_len,
                (warp_row_base + Int32(15)) // group_size,
                qo_len - Int32(1),
            )
            warp_window_lo = Int32(0)
            if const_expr(self.is_window):
                warp_window_lo = start + warp_min_tok - Int32(self.window_left)
            window_lo_max = start + warp_max_tok - Int32(self.window_left)
            causal_hi_min = start + warp_min_tok
            # Keys >= bias_min_key have dist < rel_extent for at least one row in this warp.
            bias_min_key = start + warp_min_tok - Int32(self.rel_extent - 1)
            # Keys >= bias_all_key have dist < rel_extent for EVERY row in this warp.
            bias_all_key = start + warp_max_tok - Int32(self.rel_extent - 1)

            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            q_t_bytes = Int64(mQ.stride[0]) * Int64(2)
            q_h_bytes = Int64(mQ.stride[1]) * Int64(2)
            for row_slot in cutlass.range_constexpr(2):
                row_byte = (
                    Int64(q_start + q_token[row_slot]) * q_t_bytes
                    + Int64(q_head[row_slot]) * q_h_bytes
                )
                for mma_d in cutlass.range_constexpr(8):
                    for half in cutlass.range_constexpr(2):
                        col = mma_d * 16 + 4 * lane_quad + 2 * half
                        val = Uint32(0)
                        if row_valid[row_slot] != Int32(0):
                            val = ld_global_nc_u32(
                                get_ptr_as_int64(mQBytes, row_byte + Int64(col * 2))
                            )
                        q_regs[mma_d, 0, row_slot + 2 * half] = val

            for mma_d in cutlass.range_constexpr(8):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[0, mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[0, row_slot] = Float32(-Float32.inf)
                d_frag[0, row_slot] = Float32(0.0)

            t = key_group
            j = Int32(0)
            while t < ntiles:
                buf = key_group * stages_per_group + j % stages_per_group
                use = j // stages_per_group
                cute.arch.mbarrier_wait(full_K + buf, phase=use & Int32(1))
                cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                tile_base = (my_tile_lo + t) * Int32(STAGE)
                tile_tokens = cutlass.select_(
                    tile_base + Int32(STAGE) <= start, Int32(STAGE), start - tile_base
                )
                stage_off = buf * Int32(self.kv_plane_stage_bytes)
                k0 = shared_ptr_to_u32(sKStageBytes.iterator + stage_off)
                v0 = shared_ptr_to_u32(sVStageBytes.iterator + stage_off)
                for sub in cutlass.range_constexpr(STAGE // 16):
                    if Int32(sub * 16) < tile_tokens:
                        frag_S = cute.make_rmem_tensor(frag_s_layout, Float32)
                        frag_S.fill(0.0)
                        _qk_fp8_into_sfrag(frag_S, q_regs, k0, lane, Int32(sub * 16))
                        sub_base = tile_base + Int32(sub * 16)
                        interior = (tile_tokens >= Int32(sub * 16 + 16)) and (
                            sub_base + Int32(15) <= causal_hi_min
                        )
                        if const_expr(self.is_window):
                            interior = interior and (sub_base >= window_lo_max)
                        any_biased = (sub_base + Int32(15)) >= bias_min_key
                        all_biased = sub_base >= bias_all_key
                        _scale_bias_mask_subtile(
                            frag_S,
                            Int32(sub * 16),
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
                        _online_softmax_pack_p(frag_S, o_frag, m_frag, d_frag, p_frag)
                        _pv_fp8_into_ofrag(
                            o_frag, p_frag, v0, lane, Int32(sub * 16), tile_tokens, warp_window_lo - tile_base
                        )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(empty + buf)
                t += num_key_groups
                j += Int32(1)

        # All historical TMA reads are complete.
        cute.arch.sync_threads()

        # ---- Designated split quantizes new K/V, writes cache/staging, and attends ----
        if owns_new:
            k_smem0 = shared_ptr_to_u32(payload_u8.iterator)
            v_smem0 = k_smem0 + Int32(2048)
            nk_base = (
                get_ptr_as_int64(mNewK, 0)
                + Int64(kv_head_idx) * Int64(mNewK.stride[1]) * Int64(2)
            )
            nv_base = (
                get_ptr_as_int64(mNewV, 0)
                + Int64(kv_head_idx) * Int64(mNewV.stride[1]) * Int64(2)
            )
            nk_s0 = Int64(mNewK.stride[0]) * Int64(2)
            nv_s0 = Int64(mNewV.stride[0]) * Int64(2)
            k_cache_base = get_ptr_as_int64(mKCache, 0)
            v_cache_base = get_ptr_as_int64(mVCache, 0)
            k_s0 = Int64(mKCache.stride[0])
            k_s1 = Int64(mKCache.stride[1])
            v_s0 = Int64(mVCache.stride[0])
            v_s1 = Int64(mVCache.stride[1])
            k_h_off = Int64(kv_head_idx) * Int64(mKCache.stride[2])
            v_h_off = Int64(kv_head_idx) * Int64(mVCache.stride[2])
            state_row = Int64(state_row_i32)

            sk_base = Int64(0)
            sv_base = Int64(0)
            sk_s1 = Int64(0)
            sv_s1 = Int64(0)
            if const_expr(self.is_window and not self.commit_ring):
                sk_base = (
                    get_ptr_as_int64(mStagedK, 0)
                    + Int64(request_idx) * Int64(mStagedK.stride[0])
                    + Int64(kv_head_idx) * Int64(mStagedK.stride[2])
                )
                sv_base = (
                    get_ptr_as_int64(mStagedV, 0)
                    + Int64(request_idx) * Int64(mStagedV.stride[0])
                    + Int64(kv_head_idx) * Int64(mStagedV.stride[2])
                )
                sk_s1 = Int64(mStagedK.stride[1])
                sv_s1 = Int64(mStagedV.stride[1])

            # 128 chunks (16 tokens x 8 chunks of 16 bytes) across all CTA warps.
            idx = tidx
            while idx < Int32(128):
                tok = idx >> Int32(3)
                chunk = idx & Int32(7)
                swiz_off = (tok * Int32(8) + (chunk ^ (tok & Int32(7)))) * Int32(16)
                col_fp8 = Int64(chunk * Int32(16))
                col_bf16 = col_fp8 * Int64(2)
                if tok < qo_len:
                    src_k = nk_base + Int64(q_start + tok) * nk_s0 + col_bf16
                    src_v = nv_base + Int64(q_start + tok) * nv_s0 + col_bf16
                    ka0, ka1, ka2, ka3 = ld_global_v4_u32(src_k)
                    kb0, kb1, kb2, kb3 = ld_global_v4_u32(src_k + Int64(16))
                    va0, va1, va2, va3 = ld_global_v4_u32(src_v)
                    vb0, vb1, vb2, vb3 = ld_global_v4_u32(src_v + Int64(16))
                    ke0 = cvt_bf16x2x2_to_e4m3x4(ka0, ka1)
                    ke1 = cvt_bf16x2x2_to_e4m3x4(ka2, ka3)
                    ke2 = cvt_bf16x2x2_to_e4m3x4(kb0, kb1)
                    ke3 = cvt_bf16x2x2_to_e4m3x4(kb2, kb3)
                    ve0 = cvt_bf16x2x2_to_e4m3x4(va0, va1)
                    ve1 = cvt_bf16x2x2_to_e4m3x4(va2, va3)
                    ve2 = cvt_bf16x2x2_to_e4m3x4(vb0, vb1)
                    ve3 = cvt_bf16x2x2_to_e4m3x4(vb2, vb3)
                    st_shared_v4_u32(k_smem0 + swiz_off, ke0, ke1, ke2, ke3)
                    st_shared_v4_u32(v_smem0 + swiz_off, ve0, ve1, ve2, ve3)
                    if const_expr(self.is_window):
                        if const_expr(self.commit_ring):
                            if state_row > Int64(0):
                                rslot = Int64((start + tok) % Int32(self.page_or_window))
                                st_global_v4_u32(
                                    k_cache_base + state_row * k_s0 + rslot * k_s1 + k_h_off + col_fp8,
                                    ke0, ke1, ke2, ke3,
                                )
                                st_global_v4_u32(
                                    v_cache_base + state_row * v_s0 + rslot * v_s1 + v_h_off + col_fp8,
                                    ve0, ve1, ve2, ve3,
                                )
                        else:
                            pos_s = Int64(self.page_or_window) + Int64(tok)
                            st_global_v4_u32(sk_base + pos_s * sk_s1 + col_fp8, ke0, ke1, ke2, ke3)
                            st_global_v4_u32(sv_base + pos_s * sv_s1 + col_fp8, ve0, ve1, ve2, ve3)
                    else:
                        pos_abs = start + tok
                        p_idx = pos_abs // Int32(self.page_or_window)
                        p_off = Int64(pos_abs - p_idx * Int32(self.page_or_window))
                        p_id = Int64(mPageTable[request_idx, p_idx])
                        if p_id >= Int64(0):
                            st_global_v4_u32(
                                k_cache_base + p_id * k_s0 + p_off * k_s1 + k_h_off + col_fp8,
                                ke0, ke1, ke2, ke3,
                            )
                            st_global_v4_u32(
                                v_cache_base + p_id * v_s0 + p_off * v_s1 + v_h_off + col_fp8,
                                ve0, ve1, ve2, ve3,
                            )
                else:
                    st_shared_v4_u32(k_smem0 + swiz_off, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    st_shared_v4_u32(v_smem0 + swiz_off, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if const_expr(self.is_window and not self.commit_ring):
                        if tok < Int32(self.max_q_per_req):
                            pos_s = Int64(self.page_or_window) + Int64(tok)
                            st_global_v4_u32(sk_base + pos_s * sk_s1 + col_fp8, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                            st_global_v4_u32(sv_base + pos_s * sv_s1 + col_fp8, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                idx += Int32(self.num_warps * 32)
            cute.arch.sync_threads()
            if is_consumer and key_group == Int32(0):
                frag_S = cute.make_rmem_tensor(frag_s_layout, Float32)
                frag_S.fill(0.0)
                _qk_fp8_into_sfrag(frag_S, q_regs, k_smem0, lane, Int32(0))
                _scale_bias_mask_subtile(
                    frag_S,
                    Int32(0),
                    start,
                    qo_len,
                    lane_pair_base,
                    row_valid,
                    causal_hi,
                    window_lo,
                    bias_ptr,
                    sm_scale,
                    False,
                    False,
                    True,
                    self.is_window,
                    self.rel_extent,
                )
                _online_softmax_pack_p(frag_S, o_frag, m_frag, d_frag, p_frag)
                _pv_fp8_into_ofrag(o_frag, p_frag, v_smem0, lane, Int32(0), qo_len, Int32(0))
            cute.arch.sync_threads()

        # ---- Epilogue: direct register store when num_key_groups == 1, else SMEM combine ----
        if const_expr(NCW == 1):
            if is_consumer:
                mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
                for row_slot in cutlass.range_constexpr(2):
                    if row_valid[row_slot] != Int32(0):
                        m_star = Float32(m_frag[0, row_slot])
                        d_star = Float32(d_frag[0, row_slot])
                        has_mass = m_star != -Float32.inf
                        inv_d = Float32(0.0)
                        if has_mass and d_star > Float32(0.0):
                            inv_d = Float32(1.0) / d_star
                        q_row = q_start + q_token[row_slot]
                        head = q_head[row_slot]
                        if const_expr(self.split_kv):
                            part_base = (
                                Int64(q_row) * Int64(mOPart.stride[0])
                                + Int64(head) * Int64(mOPart.stride[1])
                                + Int64(split_idx) * Int64(mOPart.stride[2])
                            ) * Int64(4)
                            for mma_d in cutlass.range_constexpr(8):
                                dim_low = mma_d * 16 + lane_pair_base
                                st_global_v2_f32(
                                    get_ptr_as_int64(mOPartBytes, part_base + Int64(dim_low * 4)),
                                    o_frag[0, mma_d, row_slot * 2 + 0] * inv_d,
                                    o_frag[0, mma_d, row_slot * 2 + 1] * inv_d,
                                )
                                st_global_v2_f32(
                                    get_ptr_as_int64(mOPartBytes, part_base + Int64((dim_low + 8) * 4)),
                                    o_frag[0, mma_d, row_slot * 2 + 4] * inv_d,
                                    o_frag[0, mma_d, row_slot * 2 + 5] * inv_d,
                                )
                            if lane_quad == Int32(0):
                                lse = Float32(-Float32.inf)
                                if has_mass and d_star > Float32(0.0):
                                    lse = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                mLsePart[q_row, head, split_idx] = lse
                        else:
                            mGateBytes = cute.flatten(cute.recast_tensor(mGate, cutlass.Uint8))
                            mSinkBytes = cute.flatten(cute.recast_tensor(mSink, cutlass.Uint8))
                            mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                            sink_sig = Float32(0.0)
                            if has_mass and d_star > Float32(0.0):
                                lse_val = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                s_u32 = _ld_global_nc_u16(
                                    get_ptr_as_int64(
                                        mSinkBytes,
                                        Int64(q_row) * Int64(mSink.stride[0]) * Int64(2) + Int64(head) * Int64(2),
                                    )
                                )
                                sink_sig = _sigmoid_f32(lse_val - u32_as_f32(s_u32 << Uint32(16)))
                            gate_row = (
                                Int64(q_row) * Int64(mGate.stride[0]) + Int64(head) * Int64(mGate.stride[1])
                            ) * Int64(2)
                            gated_row = (
                                Int64(q_row) * Int64(mGated.stride[0]) + Int64(head) * Int64(mGated.stride[1])
                            ) * Int64(2)
                            top_mask = Uint32(0xFFFF0000)
                            for mma_d in cutlass.range_constexpr(8):
                                dim_low = mma_d * 16 + lane_pair_base
                                g01 = ld_global_nc_u32(get_ptr_as_int64(mGateBytes, gate_row + Int64(dim_low * 2)))
                                g89 = ld_global_nc_u32(
                                    get_ptr_as_int64(mGateBytes, gate_row + Int64((dim_low + 8) * 2))
                                )
                                y0 = (o_frag[0, mma_d, row_slot * 2 + 0] * inv_d) * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
                                y1 = (o_frag[0, mma_d, row_slot * 2 + 1] * inv_d) * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
                                y2 = (o_frag[0, mma_d, row_slot * 2 + 4] * inv_d) * _sigmoid_f32(u32_as_f32(g89 << Uint32(16))) * sink_sig
                                y3 = (o_frag[0, mma_d, row_slot * 2 + 5] * inv_d) * _sigmoid_f32(u32_as_f32(g89 & top_mask)) * sink_sig
                                st_global_u32(
                                    get_ptr_as_int64(mGatedBytes, gated_row + Int64(dim_low * 2)),
                                    pack_f32x2_to_bfloat2(y0, y1),
                                )
                                st_global_u32(
                                    get_ptr_as_int64(mGatedBytes, gated_row + Int64((dim_low + 8) * 2)),
                                    pack_f32x2_to_bfloat2(y2, y3),
                                )
        else:
            if num_key_groups == Int32(1):
                if is_consumer:
                    mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
                    for row_slot in cutlass.range_constexpr(2):
                        if row_valid[row_slot] != Int32(0):
                            m_star = Float32(m_frag[0, row_slot])
                            d_star = Float32(d_frag[0, row_slot])
                            has_mass = m_star != -Float32.inf
                            inv_d = Float32(0.0)
                            if has_mass and d_star > Float32(0.0):
                                inv_d = Float32(1.0) / d_star
                            q_row = q_start + q_token[row_slot]
                            head = q_head[row_slot]
                            if const_expr(self.split_kv):
                                part_base = (
                                    Int64(q_row) * Int64(mOPart.stride[0])
                                    + Int64(head) * Int64(mOPart.stride[1])
                                    + Int64(split_idx) * Int64(mOPart.stride[2])
                                ) * Int64(4)
                                for mma_d in cutlass.range_constexpr(8):
                                    dim_low = mma_d * 16 + lane_pair_base
                                    st_global_v2_f32(
                                        get_ptr_as_int64(mOPartBytes, part_base + Int64(dim_low * 4)),
                                        o_frag[0, mma_d, row_slot * 2 + 0] * inv_d,
                                        o_frag[0, mma_d, row_slot * 2 + 1] * inv_d,
                                    )
                                    st_global_v2_f32(
                                        get_ptr_as_int64(mOPartBytes, part_base + Int64((dim_low + 8) * 4)),
                                        o_frag[0, mma_d, row_slot * 2 + 4] * inv_d,
                                        o_frag[0, mma_d, row_slot * 2 + 5] * inv_d,
                                    )
                                if lane_quad == Int32(0):
                                    lse = Float32(-Float32.inf)
                                    if has_mass and d_star > Float32(0.0):
                                        lse = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                    mLsePart[q_row, head, split_idx] = lse
                            else:
                                mGateBytes = cute.flatten(cute.recast_tensor(mGate, cutlass.Uint8))
                                mSinkBytes = cute.flatten(cute.recast_tensor(mSink, cutlass.Uint8))
                                mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                                sink_sig = Float32(0.0)
                                if has_mass and d_star > Float32(0.0):
                                    lse_val = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                    s_u32 = _ld_global_nc_u16(
                                        get_ptr_as_int64(
                                            mSinkBytes,
                                            Int64(q_row) * Int64(mSink.stride[0]) * Int64(2) + Int64(head) * Int64(2),
                                        )
                                    )
                                    sink_sig = _sigmoid_f32(lse_val - u32_as_f32(s_u32 << Uint32(16)))
                                gate_row = (
                                    Int64(q_row) * Int64(mGate.stride[0]) + Int64(head) * Int64(mGate.stride[1])
                                ) * Int64(2)
                                gated_row = (
                                    Int64(q_row) * Int64(mGated.stride[0]) + Int64(head) * Int64(mGated.stride[1])
                                ) * Int64(2)
                                top_mask = Uint32(0xFFFF0000)
                                for mma_d in cutlass.range_constexpr(8):
                                    dim_low = mma_d * 16 + lane_pair_base
                                    g01 = ld_global_nc_u32(get_ptr_as_int64(mGateBytes, gate_row + Int64(dim_low * 2)))
                                    g89 = ld_global_nc_u32(
                                        get_ptr_as_int64(mGateBytes, gate_row + Int64((dim_low + 8) * 2))
                                    )
                                    y0 = (o_frag[0, mma_d, row_slot * 2 + 0] * inv_d) * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
                                    y1 = (o_frag[0, mma_d, row_slot * 2 + 1] * inv_d) * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
                                    y2 = (o_frag[0, mma_d, row_slot * 2 + 4] * inv_d) * _sigmoid_f32(u32_as_f32(g89 << Uint32(16))) * sink_sig
                                    y3 = (o_frag[0, mma_d, row_slot * 2 + 5] * inv_d) * _sigmoid_f32(u32_as_f32(g89 & top_mask)) * sink_sig
                                    st_global_u32(
                                        get_ptr_as_int64(mGatedBytes, gated_row + Int64(dim_low * 2)),
                                        pack_f32x2_to_bfloat2(y0, y1),
                                    )
                                    st_global_u32(
                                        get_ptr_as_int64(mGatedBytes, gated_row + Int64((dim_low + 8) * 2)),
                                        pack_f32x2_to_bfloat2(y2, y3),
                                    )
            else:
                sComb = cute.make_tensor(
                    cute.recast_ptr(payload_u8.iterator.align(16), dtype=Float32),
                    cute.make_layout((self.comb_o_floats + 2 * self.comb_md_floats,), stride=(1,)),
                )
                o_words = 8 * 8 * 32
                md_base = self.comb_o_floats
                if is_consumer:
                    for mma_d in cutlass.range_constexpr(8):
                        for reg_id in cutlass.range_constexpr(8):
                            sComb[warp * o_words + (mma_d * 8 + reg_id) * 32 + lane] = o_frag[0, mma_d, reg_id]
                    for row_slot in cutlass.range_constexpr(2):
                        sComb[md_base + (warp * 2 + row_slot) * 32 + lane] = m_frag[0, row_slot]
                        sComb[md_base + self.comb_md_floats + (warp * 2 + row_slot) * 32 + lane] = d_frag[0, row_slot]
                cute.arch.sync_threads()

                scales = cute.make_rmem_tensor(cute.make_layout((NCW,), stride=(1,)), Float32)
                if is_consumer:
                    mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
                    for row_slot in cutlass.range_constexpr(2):
                        if row_valid[row_slot] != Int32(0):
                            m_star = Float32(-Float32.inf)
                            for k in cutlass.range_constexpr(NCW):
                                if Int32(k) < num_key_groups:
                                    w = row_group + Int32(k) * num_row_groups
                                    m_star = _max_nan_f32(
                                        m_star, sComb[md_base + (w * 2 + row_slot) * 32 + lane]
                                    )
                            has_mass = m_star != -Float32.inf
                            d_star = Float32(0.0)
                            for k in cutlass.range_constexpr(NCW):
                                scales[k] = Float32(0.0)
                                if Int32(k) < num_key_groups and has_mass:
                                    w = row_group + Int32(k) * num_row_groups
                                    m_k = sComb[md_base + (w * 2 + row_slot) * 32 + lane]
                                    if m_k != -Float32.inf:
                                        scales[k] = _exp2_approx_ftz_f32((m_k - m_star) * Float32(LOG2_E))
                                    d_star += (
                                        scales[k]
                                        * sComb[md_base + self.comb_md_floats + (w * 2 + row_slot) * 32 + lane]
                                    )
                            inv_d = Float32(0.0)
                            if has_mass and d_star > Float32(0.0):
                                inv_d = Float32(1.0) / d_star
                            q_row = q_start + q_token[row_slot]
                            head = q_head[row_slot]
                            part_base = Int64(0)
                            sink_sig = Float32(0.0)
                            gate_row = Int64(0)
                            gated_row = Int64(0)
                            if const_expr(self.split_kv):
                                part_base = (
                                    Int64(q_row) * Int64(mOPart.stride[0])
                                    + Int64(head) * Int64(mOPart.stride[1])
                                    + Int64(split_idx) * Int64(mOPart.stride[2])
                                ) * Int64(4)
                            else:
                                mSinkBytes = cute.flatten(cute.recast_tensor(mSink, cutlass.Uint8))
                                if has_mass and d_star > Float32(0.0):
                                    lse_val = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                    s_u32 = _ld_global_nc_u16(
                                        get_ptr_as_int64(
                                            mSinkBytes,
                                            Int64(q_row) * Int64(mSink.stride[0]) * Int64(2) + Int64(head) * Int64(2),
                                        )
                                    )
                                    sink_sig = _sigmoid_f32(lse_val - u32_as_f32(s_u32 << Uint32(16)))
                                gate_row = (
                                    Int64(q_row) * Int64(mGate.stride[0]) + Int64(head) * Int64(mGate.stride[1])
                                ) * Int64(2)
                                gated_row = (
                                    Int64(q_row) * Int64(mGated.stride[0]) + Int64(head) * Int64(mGated.stride[1])
                                ) * Int64(2)

                            for mma_d in cutlass.range_constexpr(8):
                                if Int32(mma_d) % num_key_groups == key_group:
                                    a0 = Float32(0.0)
                                    a1 = Float32(0.0)
                                    a2 = Float32(0.0)
                                    a3 = Float32(0.0)
                                    for k in cutlass.range_constexpr(NCW):
                                        if Int32(k) < num_key_groups:
                                            w = row_group + Int32(k) * num_row_groups
                                            base = w * o_words + (mma_d * 8) * 32 + lane
                                            a0 += scales[k] * sComb[base + (row_slot * 2 + 0) * 32]
                                            a1 += scales[k] * sComb[base + (row_slot * 2 + 1) * 32]
                                            a2 += scales[k] * sComb[base + (row_slot * 2 + 4) * 32]
                                            a3 += scales[k] * sComb[base + (row_slot * 2 + 5) * 32]
                                    dim_low = mma_d * 16 + lane_pair_base
                                    o0 = a0 * inv_d
                                    o1 = a1 * inv_d
                                    o2 = a2 * inv_d
                                    o3 = a3 * inv_d
                                    if const_expr(self.split_kv):
                                        st_global_v2_f32(
                                            get_ptr_as_int64(mOPartBytes, part_base + Int64(dim_low * 4)),
                                            o0, o1,
                                        )
                                        st_global_v2_f32(
                                            get_ptr_as_int64(mOPartBytes, part_base + Int64((dim_low + 8) * 4)),
                                            o2, o3,
                                        )
                                    else:
                                        mGateBytes = cute.flatten(cute.recast_tensor(mGate, cutlass.Uint8))
                                        mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                                        top_mask = Uint32(0xFFFF0000)
                                        g01 = ld_global_nc_u32(
                                            get_ptr_as_int64(mGateBytes, gate_row + Int64(dim_low * 2))
                                        )
                                        g89 = ld_global_nc_u32(
                                            get_ptr_as_int64(mGateBytes, gate_row + Int64((dim_low + 8) * 2))
                                        )
                                        y0 = o0 * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
                                        y1 = o1 * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
                                        y2 = o2 * _sigmoid_f32(u32_as_f32(g89 << Uint32(16))) * sink_sig
                                        y3 = o3 * _sigmoid_f32(u32_as_f32(g89 & top_mask)) * sink_sig
                                        st_global_u32(
                                            get_ptr_as_int64(mGatedBytes, gated_row + Int64(dim_low * 2)),
                                            pack_f32x2_to_bfloat2(y0, y1),
                                        )
                                        st_global_u32(
                                            get_ptr_as_int64(mGatedBytes, gated_row + Int64((dim_low + 8) * 2)),
                                            pack_f32x2_to_bfloat2(y2, y3),
                                        )
                            if const_expr(self.split_kv):
                                if lane_quad == Int32(0) and key_group == Int32(0):
                                    lse = Float32(-Float32.inf)
                                    if has_mass and d_star > Float32(0.0):
                                        lse = m_star + cute.math.log2(d_star, fastmath=True) * Float32(LN_2)
                                    mLsePart[q_row, head, split_idx] = lse


class RelativeDecodeMergeKernel:
    """Four-warp-per-head split-KV reduction with fused per-token gate and per-token sink."""

    def __init__(
        self,
        *,
        num_q_heads: int = 24,
        max_q_per_req: int = 1,
        is_window: bool = True,
        window_left: int = 1023,
        stage_rows: int = 16,
        min_tiles_per_split: int = 2,
        warps_per_row: int = 4,
    ):
        self.num_q_heads = int(num_q_heads)
        self.max_q_per_req = int(max_q_per_req)
        self.is_window = bool(is_window)
        self.window_left = int(window_left)
        self.stage_rows = int(stage_rows)
        self.min_tiles_per_split = int(min_tiles_per_split)
        self.warps_per_row = int(warps_per_row)

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "acc": cute.struct.MemRange[Float32, self.warps_per_row * 128],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mOPart: cute.Tensor,  # [T, Hq, S, 128] f32
        mLsePart: cute.Tensor,  # [T, Hq, S] f32 (natural log)
        mStarts: cute.Tensor,  # [B] i64
        mCuSeqlensQ: cute.Tensor,  # [B + 1] i32
        mGate: cute.Tensor,  # [T, Hq, 128] bf16
        mSink: cute.Tensor,  # [T, Hq] bf16
        mGated: cute.Tensor,  # [T, Hq, 128] bf16
        stream: cuda.CUstream,
    ):
        self.kernel(mOPart, mLsePart, mStarts, mCuSeqlensQ, mGate, mSink, mGated).launch(
            grid=(mStarts.shape[0], self.max_q_per_req, self.num_q_heads),
            block=[32, self.warps_per_row, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mStarts: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mGate: cute.Tensor,
        mSink: cute.Tensor,
        mGated: cute.Tensor,
    ):
        STAGE = self.stage_rows
        WPR = self.warps_per_row
        lane, warp_y, _ = cute.arch.thread_idx()
        request_idx, token_local, head = cute.arch.block_idx()
        num_reqs, _, _ = cute.arch.grid_dim()

        # Zero padded output rows once (request 0, token_local 0, warp 0).
        if request_idx == Int32(0) and token_local == Int32(0) and warp_y == Int32(0):
            total_live = mCuSeqlensQ[num_reqs]
            total_cap = mGated.shape[0]
            if total_live < total_cap:
                mGatedBytes0 = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
                g_t_s = Int64(mGated.stride[0]) * Int64(2)
                g_h_s = Int64(mGated.stride[1]) * Int64(2)
                rp = total_live
                while rp < total_cap:
                    st_global_v2_u32(
                        get_ptr_as_int64(
                            mGatedBytes0,
                            Int64(rp) * g_t_s + Int64(head) * g_h_s + Int64(lane * 8),
                        ),
                        Uint32(0),
                        Uint32(0),
                    )
                    rp += Int32(1)

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        if token_local >= qo_len:
            _exit_thread()

        start = Int32(mStarts[request_idx])
        num_splits = mLsePart.shape[2]
        first_key = Int32(0)
        if const_expr(self.is_window):
            first_key = cutlass.select_(
                start - Int32(self.window_left) > Int32(0),
                start - Int32(self.window_left),
                Int32(0),
            )
        tile_begin = first_key // Int32(STAGE)
        tile_end = (start + Int32(STAGE - 1)) // Int32(STAGE)
        num_hist_tiles = cutlass.select_(tile_end > tile_begin, tile_end - tile_begin, Int32(0))
        active = Int32(1)
        if num_hist_tiles > Int32(0):
            tiles_per_split = (num_hist_tiles + num_splits - Int32(1)) // num_splits
            tiles_per_split = cutlass.select_(
                tiles_per_split < Int32(self.min_tiles_per_split),
                Int32(self.min_tiles_per_split),
                tiles_per_split,
            )
            active = (num_hist_tiles + tiles_per_split - Int32(1)) // tiles_per_split
        q_row = q_start + token_local
        log2_e = Float32(LOG2_E)

        mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
        row_base = (
            Int64(q_row) * Int64(mOPart.stride[0]) + Int64(head) * Int64(mOPart.stride[1])
        ) * Int64(4) + Int64(lane * 16)
        split_stride_bytes = Int64(mOPart.stride[2]) * Int64(4)
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        m_row = Float32(-Float32.inf)
        s_row = Float32(0.0)

        if active <= Int32(32):
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
            if m_row != -Float32.inf:
                sp = warp_y
                while sp + Int32(3 * WPR) < active:
                    w0 = cute.arch.shuffle_sync(w_lane, sp)
                    w1 = cute.arch.shuffle_sync(w_lane, sp + Int32(WPR))
                    w2 = cute.arch.shuffle_sync(w_lane, sp + Int32(2 * WPR))
                    w3 = cute.arch.shuffle_sync(w_lane, sp + Int32(3 * WPR))
                    x0, x1, x2, x3 = ld_global_v4_f32(
                        get_ptr_as_int64(mOPartBytes, row_base + Int64(sp) * split_stride_bytes)
                    )
                    y0, y1, y2, y3 = ld_global_v4_f32(
                        get_ptr_as_int64(mOPartBytes, row_base + Int64(sp + Int32(WPR)) * split_stride_bytes)
                    )
                    z0, z1, z2, z3 = ld_global_v4_f32(
                        get_ptr_as_int64(mOPartBytes, row_base + Int64(sp + Int32(2 * WPR)) * split_stride_bytes)
                    )
                    u0, u1, u2, u3 = ld_global_v4_f32(
                        get_ptr_as_int64(mOPartBytes, row_base + Int64(sp + Int32(3 * WPR)) * split_stride_bytes)
                    )
                    acc0 += w0 * x0 + w1 * y0 + w2 * z0 + w3 * u0
                    acc1 += w0 * x1 + w1 * y1 + w2 * z1 + w3 * u1
                    acc2 += w0 * x2 + w1 * y2 + w2 * z2 + w3 * u2
                    acc3 += w0 * x3 + w1 * y3 + w2 * z3 + w3 * u3
                    sp += Int32(4 * WPR)
                while sp < active:
                    w = cute.arch.shuffle_sync(w_lane, sp)
                    o0, o1, o2, o3 = ld_global_v4_f32(
                        get_ptr_as_int64(mOPartBytes, row_base + Int64(sp) * split_stride_bytes)
                    )
                    acc0 += w * o0
                    acc1 += w * o1
                    acc2 += w * o2
                    acc3 += w * o3
                    sp += Int32(WPR)
        else:
            m_local = Float32(-Float32.inf)
            sp = lane
            while sp < active:
                m_local = _max_nan_f32(m_local, mLsePart[q_row, head, sp])
                sp += Int32(32)
            m_row = m_local
            for offset in cutlass.range_constexpr(5):
                m_row = _max_nan_f32(m_row, cute.arch.shuffle_sync_bfly(m_row, offset=1 << offset))
            s_local = Float32(0.0)
            if m_row != -Float32.inf:
                sp = lane
                while sp < active:
                    lv = mLsePart[q_row, head, sp]
                    if lv != -Float32.inf:
                        s_local += _exp2_approx_ftz_f32((lv - m_row) * log2_e)
                    sp += Int32(32)
            s_row = cute.arch.warp_reduction_sum(s_local)
            if m_row != -Float32.inf:
                sp = warp_y
                while sp < active:
                    lv = mLsePart[q_row, head, sp]
                    if lv != -Float32.inf:
                        w = _exp2_approx_ftz_f32((lv - m_row) * log2_e)
                        o0, o1, o2, o3 = ld_global_v4_f32(
                            get_ptr_as_int64(mOPartBytes, row_base + Int64(sp) * split_stride_bytes)
                        )
                        acc0 += w * o0
                        acc1 += w * o1
                        acc2 += w * o2
                        acc3 += w * o3
                    sp += Int32(WPR)

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sAcc = storage.acc.get_tensor(cute.make_layout((WPR, 128), stride=(128, 1)))
        sAcc[warp_y, lane * 4 + 0] = acc0
        sAcc[warp_y, lane * 4 + 1] = acc1
        sAcc[warp_y, lane * 4 + 2] = acc2
        sAcc[warp_y, lane * 4 + 3] = acc3
        cute.arch.sync_threads()

        if warp_y == Int32(0):
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
            mGatedBytes = cute.flatten(cute.recast_tensor(mGated, cutlass.Uint8))
            gate_ptr = get_ptr_as_int64(
                mGateBytes,
                (Int64(q_row) * Int64(mGate.stride[0]) + Int64(head) * Int64(mGate.stride[1])) * Int64(2)
                + Int64(lane * 8),
            )
            g01, g23 = ld_global_nc_v2_u32(gate_ptr)
            top_mask = Uint32(0xFFFF0000)
            t0 = Float32(0.0)
            t1 = Float32(0.0)
            t2 = Float32(0.0)
            t3 = Float32(0.0)
            for wr in cutlass.range_constexpr(WPR):
                t0 += sAcc[wr, lane * 4 + 0]
                t1 += sAcc[wr, lane * 4 + 1]
                t2 += sAcc[wr, lane * 4 + 2]
                t3 += sAcc[wr, lane * 4 + 3]
            y0 = (t0 * inv_s) * _sigmoid_f32(u32_as_f32(g01 << Uint32(16))) * sink_sig
            y1 = (t1 * inv_s) * _sigmoid_f32(u32_as_f32(g01 & top_mask)) * sink_sig
            y2 = (t2 * inv_s) * _sigmoid_f32(u32_as_f32(g23 << Uint32(16))) * sink_sig
            y3 = (t3 * inv_s) * _sigmoid_f32(u32_as_f32(g23 & top_mask)) * sink_sig
            gated_ptr = get_ptr_as_int64(
                mGatedBytes,
                (Int64(q_row) * Int64(mGated.stride[0]) + Int64(head) * Int64(mGated.stride[1])) * Int64(2)
                + Int64(lane * 8),
            )
            st_global_v2_u32(
                gated_ptr,
                pack_f32x2_to_bfloat2(y0, y1),
                pack_f32x2_to_bfloat2(y2, y3),
            )


def default_max_splits(num_reqs: int, num_kv_heads: int, num_sms: int = 188, cap: int = 32) -> int:
    """Split budget that fits in one wave on SM120 without oversubscribing."""
    ctas = max(int(num_reqs) * int(num_kv_heads), 1)
    return int(max(1, min(cap, num_sms // ctas)))


__all__ = ["RelativeDecodeKernel", "RelativeDecodeMergeKernel", "default_max_splits"]
