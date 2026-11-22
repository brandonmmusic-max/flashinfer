"""Tensor-core GEMV for up to 32 BF16 rows with optional single-launch split-K.

At decode row counts the projection streams the weight once, so its cost is
the weight read.  Each CTA owns 16 output features and its warps split K into
64-wide (or 32-wide for K % 64 == 32 or K == 128) chunks.  Weights are the
``m16n8k16`` MMA's 16-row A operand and each 8-row slice of the live rows its
8-column B operand, so one weight fragment serves up to 32 rows (four 8-row
tiles).  For skinny N, ``split_k > 1`` fans K across multiple CTAs per 16-row
N tile and reduces FP32 partials inside the same launch via coalesced 128-bit
L2-coherent vector loads/stores and an atomic arrival counter reset by the last
CTA before exit (CUDA-graph safe).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_nc_v4_u32
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

TC_MAX_ROWS = 32
TC_MAX_WARPS = 8
TC_K_MULTIPLE = 32
TC_SMALL_K = 16
TC_SMALL_K_MAX_ROWS = 256
TC_SMALL_K_N_MULTIPLE = 64
TC_SPLIT_K_VALUES = (1, 2, 3, 4, 6, 7, 8, 12, 14, 16, 28, 56)
_LANES = 32

_TC_SPLITK_BUFFERS: dict[tuple[int, int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def tc_chunk_k(k: int) -> int:
    """Return K-elements per warp chunk (64 when K >= 256 and K % 64 == 0, else 32)."""
    return 64 if (int(k) >= 256 and int(k) % 64 == 0) else 32


def tc_splitk_scratch_bytes(max_rows: int, n: int, split_k: int) -> int:
    if split_k <= 1:
        return 0
    n_blocks = (int(n) + 15) // 16
    m_tiles = (int(max_rows) + 7) // 8
    return n_blocks * int(split_k) * m_tiles * 128 * 4


def tc_splitk_flags_bytes(n: int, split_k: int) -> int:
    if split_k <= 1:
        return 0
    return ((int(n) + 15) // 16) * 4


def tc_splitk_buffers(
    device: torch.device, max_rows: int, n: int, k: int, split_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reusable per-(device, shape, split_k) scratch and zero-initialized flags."""
    m_tiles = (int(max_rows) + 7) // 8
    key = (int(device.index or 0), int(n), int(k), m_tiles, int(split_k))
    cached = _TC_SPLITK_BUFFERS.get(key)
    if cached is not None:
        return cached
    n_blocks = (int(n) + 15) // 16
    scratch = torch.empty(
        (n_blocks * int(split_k) * m_tiles * 128,), dtype=torch.float32, device=device
    )
    flags = torch.zeros((n_blocks,), dtype=torch.uint32, device=device)
    _TC_SPLITK_BUFFERS[key] = (scratch, flags)
    return scratch, flags


@dsl_user_op
def _st_global_wb_v4_f32(
    addr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.wb.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_cg_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _atomic_add_gpu_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_release_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_nc_u32_pure(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


class TensorCoreGemvKernel:
    """``out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n])`` for up to 32 live rows."""

    def __init__(
        self,
        n: int,
        k: int,
        max_rows: int = 8,
        split_k: int = 1,
        *,
        has_bias: bool = False,
    ):
        if k <= 0 or k % TC_K_MULTIPLE:
            raise ValueError(
                f"tensor-core GEMV requires K to be a positive multiple of {TC_K_MULTIPLE}"
            )
        if not 1 <= max_rows <= TC_MAX_ROWS:
            raise ValueError(f"tensor-core GEMV supports 1..{TC_MAX_ROWS} rows")
        self.chunk_k = tc_chunk_k(k)
        chunks = int(k) // self.chunk_k
        if split_k <= 0 or chunks % int(split_k):
            raise ValueError(f"split_k={split_k} must divide K/{self.chunk_k}={chunks}")
        self.n, self.k = int(n), int(k)
        self.has_bias = bool(has_bias)
        self.max_m_tiles = (int(max_rows) + 7) // 8
        self.split_k = int(split_k)
        self.chunks_per_split = chunks // self.split_k
        self.warps = min(TC_MAX_WARPS, self.chunks_per_split)
        while self.chunks_per_split % self.warps:
            self.warps -= 1
        self.chunks_per_warp = self.chunks_per_split // self.warps

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.warps * _LANES * 4 * self.max_m_tiles
                ],
                16,
            ],
            "last": cute.struct.Align[cute.struct.MemRange[Uint32, 1], 4],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(bias),
            _flat(output),
            scratch,
            flags,
            rows,
            x_stride,
            weight_stride,
            output_stride,
        ).launch(
            grid=((self.n + 15) // 16, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
    ):
        result_dtype = output.element_type
        thread, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        first = Int32(n_tile) * 16
        row_a = cutlass.min(first + group, Int32(self.n - 1))
        row_b = cutlass.min(first + group + 8, Int32(self.n - 1))
        weight_base = get_ptr_as_int64(weight, Int64(0))
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        split_chunk_base = Int32(k_split) * self.chunks_per_split
        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = split_chunk_base + warp * self.chunks_per_warp + step
            if cutlass.const_expr(self.chunk_k == 64):
                offset = Int64(chunk * 128 + quad * 32)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
                wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
                wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
                    xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                    a0, a1, a2, a3 = acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                    for s in cutlass.range_constexpr(4):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3
            else:
                offset = Int64(chunk * 64 + quad * 16)
                pa = weight_base + Int64(row_a) * weight_stride * 2 + offset
                pb = weight_base + Int64(row_b) * weight_stride * 2 + offset
                wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
                wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
                wa = (wa0, wa1, wa2, wa3)
                wb = (wb0, wb1, wb2, wb3)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                    px = x_base + Int64(token) * x_stride * 2 + offset
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    xs = (x0, x1, x2, x3)
                    a0, a1, a2, a3 = acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                    for s in cutlass.range_constexpr(2):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wa[2 * s],
                            wb[2 * s],
                            wa[2 * s + 1],
                            wb[2 * s + 1],
                            xs[2 * s],
                            xs[2 * s + 1],
                        )
                    acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(
            cute.make_layout((self.warps * _LANES * 4 * self.max_m_tiles,))
        )
        if cutlass.const_expr(self.warps > 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                base = ((mt * self.warps + warp) * _LANES + lane) * 4
                for e in cutlass.range_constexpr(4):
                    partials[base + e] = acc[mt, e]
            cute.arch.sync_threads()

        if cutlass.const_expr(self.split_k == 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    for e in cutlass.range_constexpr(4):
                        total = (
                            acc[mt, e]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        if cutlass.const_expr(self.warps > 1):
                            for w in cutlass.range_constexpr(self.warps):
                                total = (
                                    total
                                    + partials[
                                        ((mt * self.warps + w) * _LANES + lane) * 4 + e
                                    ]
                                )
                        feature = first + group + (8 if e >= 2 else 0)
                        row = Int32(mt * 8) + quad * 2 + (e % 2)
                        if feature < self.n and row < rows:
                            if cutlass.const_expr(self.has_bias):
                                total = total + Float32(bias[Int64(feature)])
                            output[Int64(row) * output_stride + Int64(feature)] = (
                                total.to(result_dtype)
                            )
        else:
            tile_elems = self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    t0 = acc[mt, 0] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t1 = acc[mt, 1] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t2 = acc[mt, 2] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t3 = acc[mt, 3] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    if cutlass.const_expr(self.warps > 1):
                        for w in cutlass.range_constexpr(self.warps):
                            pbase = ((mt * self.warps + w) * _LANES + lane) * 4
                            t0 = t0 + partials[pbase + 0]
                            t1 = t1 + partials[pbase + 1]
                            t2 = t2 + partials[pbase + 2]
                            t3 = t3 + partials[pbase + 3]
                    vaddr = scratch_base + (sbase + Int64((mt * _LANES + lane) * 4)) * Int64(4)
                    _st_global_wb_v4_f32(vaddr, t0, t1, t2, t3)
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(n_tile) * Int64(4)
            if Int32(thread) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(n_tile) * self.split_k * tile_elems
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    if warp == Int32(mt % self.warps):
                        s0 = Float32(0.0)
                        s1 = Float32(0.0)
                        s2 = Float32(0.0)
                        s3 = Float32(0.0)
                        lane_off = Int64((mt * _LANES + lane) * 4)
                        for sp in cutlass.range_constexpr(self.split_k):
                            raddr = (
                                scratch_base
                                + (tbase + Int64(sp) * tile_elems + lane_off) * Int64(4)
                            )
                            v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                            s0 = s0 + v0
                            s1 = s1 + v1
                            s2 = s2 + v2
                            s3 = s3 + v3
                        f_lo = first + group
                        f_hi = first + group + 8
                        r0 = Int32(mt * 8) + quad * 2
                        r1 = r0 + 1
                        if f_lo < self.n:
                            if cutlass.const_expr(self.has_bias):
                                b_lo = Float32(bias[Int64(f_lo)])
                                s0 = s0 + b_lo
                                s1 = s1 + b_lo
                            if r0 < rows:
                                output[Int64(r0) * output_stride + Int64(f_lo)] = s0.to(result_dtype)
                            if r1 < rows:
                                output[Int64(r1) * output_stride + Int64(f_lo)] = s1.to(result_dtype)
                        if f_hi < self.n:
                            if cutlass.const_expr(self.has_bias):
                                b_hi = Float32(bias[Int64(f_hi)])
                                s2 = s2 + b_hi
                                s3 = s3 + b_hi
                            if r0 < rows:
                                output[Int64(r0) * output_stride + Int64(f_hi)] = s2.to(result_dtype)
                            if r1 < rows:
                                output[Int64(r1) * output_stride + Int64(f_hi)] = s3.to(result_dtype)
                cute.arch.sync_threads()
                if Int32(thread) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))


class SmallK16GemmKernel:
    """Single-step ``m16n8k16`` BF16 GEMM for ``K == 16``, ``N % 64 == 0``."""

    def __init__(self, n: int):
        if n <= 0 or n % TC_SMALL_K_N_MULTIPLE:
            raise ValueError(
                f"small-K16 GEMM requires N to be a positive multiple of {TC_SMALL_K_N_MULTIPLE}"
            )
        self.n = int(n)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(output),
            rows,
            x_stride,
            weight_stride,
            output_stride,
        ).launch(
            grid=((rows + Int32(15)) // Int32(16), self.n // 64, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bm, bn, _ = cute.arch.block_idx()
        warp = Int32(tid) // Int32(32)
        lane = Int32(tid) % Int32(32)
        group = lane // Int32(4)
        quad = lane % Int32(4)

        m0 = Int32(bm) * Int32(16)
        n0 = Int32(bn) * Int32(64) + warp * Int32(16)

        x_base = get_ptr_as_int64(x, Int64(0))
        w_base = get_ptr_as_int64(weight, Int64(0))

        rm_a = cutlass.min(m0 + group, rows - Int32(1))
        rm_b = cutlass.min(m0 + group + Int32(8), rows - Int32(1))
        x_row_bytes = x_stride * Int64(2)
        w_row_bytes = weight_stride * Int64(2)
        q_off = Int64(quad * Int32(4))

        a0 = _ld_nc_u32_pure(x_base + Int64(rm_a) * x_row_bytes + q_off)
        a2 = _ld_nc_u32_pure(x_base + Int64(rm_a) * x_row_bytes + q_off + Int64(16))
        a1 = _ld_nc_u32_pure(x_base + Int64(rm_b) * x_row_bytes + q_off)
        a3 = _ld_nc_u32_pure(x_base + Int64(rm_b) * x_row_bytes + q_off + Int64(16))

        rn0 = n0 + group
        b00 = _ld_nc_u32_pure(w_base + Int64(rn0) * w_row_bytes + q_off)
        b01 = _ld_nc_u32_pure(w_base + Int64(rn0) * w_row_bytes + q_off + Int64(16))
        rn1 = n0 + Int32(8) + group
        b10 = _ld_nc_u32_pure(w_base + Int64(rn1) * w_row_bytes + q_off)
        b11 = _ld_nc_u32_pure(w_base + Int64(rn1) * w_row_bytes + q_off + Int64(16))

        z = Float32(0.0)
        c00, c01, c02, c03 = bf16_mma_m16n8k16_f32(z, z, z, z, a0, a1, a2, a3, b00, b01)
        c10, c11, c12, c13 = bf16_mma_m16n8k16_f32(z, z, z, z, a0, a1, a2, a3, b10, b11)

        r0 = m0 + group
        r1 = m0 + group + Int32(8)
        col0 = n0 + quad * Int32(2)
        col1 = n0 + Int32(8) + quad * Int32(2)
        if r0 < rows:
            base0 = Int64(r0) * output_stride
            output[base0 + Int64(col0)] = c00.to(BFloat16)
            output[base0 + Int64(col0 + Int32(1))] = c01.to(BFloat16)
            output[base0 + Int64(col1)] = c10.to(BFloat16)
            output[base0 + Int64(col1 + Int32(1))] = c11.to(BFloat16)
        if r1 < rows:
            base1 = Int64(r1) * output_stride
            output[base1 + Int64(col0)] = c02.to(BFloat16)
            output[base1 + Int64(col0 + Int32(1))] = c03.to(BFloat16)
            output[base1 + Int64(col1)] = c12.to(BFloat16)
            output[base1 + Int64(col1 + Int32(1))] = c13.to(BFloat16)


@program_cache
def compile_small_k16_projection(ordinal: int, n: int):
    """Compile one SmallK16GemmKernel for BF16 K=16 projections."""
    kernel = SmallK16GemmKernel(n)
    key = (int(n),)
    fake_x = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_w = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_y = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            fake_x,
            fake_w,
            fake_y,
            Int32(1),
            Int64(TC_SMALL_K),
            Int64(TC_SMALL_K),
            Int64(n),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.bf16_projection.small_k16", 1, key
            ),
        )

    def run(x, weight, out, bias=None, *, scratch=None, flags=None):
        del bias, scratch, flags
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    make_ptr(BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=2),
                    Int32(x.shape[0]),
                    Int64(x.stride(0)),
                    Int64(weight.stride(0)),
                    Int64(out.stride(0)),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)


@program_cache
def compile_tc_projection(
    ordinal: int,
    max_rows: int,
    split_k: int,
    n: int,
    k: int,
    output_dtype: str,
    bias_dtype: str | None = None,
):
    """Compile one TensorCoreGemvKernel (or SmallK16GemmKernel for K=16)."""
    if int(k) == TC_SMALL_K:
        if output_dtype != "bfloat16" or bias_dtype is not None or int(split_k) != 1:
            raise ValueError("small-K16 GEMM requires BF16 output without bias or split_k")
        return compile_small_k16_projection(ordinal, n)
    y_type = getattr(torch, output_dtype)
    ct_y = Float32 if y_type == torch.float32 else BFloat16
    b_type = torch.bfloat16 if bias_dtype is None else getattr(torch, bias_dtype)
    ct_b = Float32 if b_type == torch.float32 else BFloat16
    max_m_tiles = (int(max_rows) + 7) // 8
    kernel = TensorCoreGemvKernel(
        n,
        k,
        max_rows=max_m_tiles * 8,
        split_k=split_k,
        has_bias=(bias_dtype is not None),
    )
    key = (max_m_tiles, int(split_k), int(n), int(k), output_dtype, bias_dtype)
    fake_x = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_w = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_b = make_ptr(ct_b, 16, cute.AddressSpace.gmem, assumed_align=b_type.itemsize)
    fake_y = make_ptr(ct_y, 16, cute.AddressSpace.gmem, assumed_align=y_type.itemsize)
    fake_sc = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_fl = make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            fake_x,
            fake_w,
            fake_b,
            fake_y,
            fake_sc,
            fake_fl,
            Int32(1),
            Int64(k),
            Int64(k),
            Int64(n),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.bf16_projection.tc_multirow", 3, key
            ),
        )

    def run(x, weight, out, bias=None, *, scratch=None, flags=None):
        b_tensor = x if bias is None else bias
        sc_ptr = (
            make_ptr(Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            if scratch is not None
            else make_ptr(Float32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        )
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    make_ptr(BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(ct_b, b_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=b_type.itemsize),
                    make_ptr(ct_y, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=y_type.itemsize),
                    sc_ptr,
                    fl_ptr,
                    Int32(x.shape[0]),
                    Int64(x.stride(0)),
                    Int64(weight.stride(0)),
                    Int64(out.stride(0)),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)
