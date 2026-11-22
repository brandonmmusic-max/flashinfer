"""Split-K tensor-core BF16 router GEMV kernel for small-batch MoE routing."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as _MLIR_T, dsl_user_op

from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


@dsl_user_op
def _ld_nc_v4_pure(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([_MLIR_T.i32(), _MLIR_T.i32(), _MLIR_T.i32(), _MLIR_T.i32()]),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(_MLIR_T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


class SplitKRouterGemvKernel:
    """Split-K ``m16n8k16`` GEMV writing FP32 ``[splits, rows, N]`` partials."""

    def __init__(self, n: int, k: int, splits: int):
        self.n, self.k, self.splits = int(n), int(k), int(splits)
        self.chunks = (self.k // 64) // self.splits

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        partials: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        w_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(_flat(x), _flat(w), _flat(partials), rows, x_stride, w_stride).launch(
            grid=(self.n // 16, self.splits, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        partials: cute.Tensor,
        rows: Int32,
        x_stride: Int64,
        w_stride: Int64,
    ):
        lane, _, _ = cute.arch.thread_idx()
        bn, split, _ = cute.arch.block_idx()
        lane = Int32(lane)
        group = lane // Int32(4)
        quad = lane % Int32(4)
        first = Int32(bn) * Int32(16)
        row_a = first + group
        row_b = first + group + Int32(8)
        token = cutlass.min(group, rows - Int32(1))
        w_base = get_ptr_as_int64(w, Int64(0))
        x_base = get_ptr_as_int64(x, Int64(0))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        for step in cutlass.range_constexpr(self.chunks):
            chunk = Int32(split) * Int32(self.chunks) + Int32(step)
            offset = Int64(chunk * Int32(128) + quad * Int32(32))
            pa = w_base + Int64(row_a) * w_stride * Int64(2) + offset
            pb = w_base + Int64(row_b) * w_stride * Int64(2) + offset
            px = x_base + Int64(token) * x_stride * Int64(2) + offset
            wa0, wa1, wa2, wa3 = _ld_nc_v4_pure(pa)
            wa4, wa5, wa6, wa7 = _ld_nc_v4_pure(pa + Int64(16))
            wb0, wb1, wb2, wb3 = _ld_nc_v4_pure(pb)
            wb4, wb5, wb6, wb7 = _ld_nc_v4_pure(pb + Int64(16))
            x0, x1, x2, x3 = _ld_nc_v4_pure(px)
            x4, x5, x6, x7 = _ld_nc_v4_pure(px + Int64(16))
            wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
            wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
            xs = (x0, x1, x2, x3, x4, x5, x6, x7)
            for s in cutlass.range_constexpr(4):
                acc0, acc1, acc2, acc3 = bf16_mma_m16n8k16_f32(
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    wa[2 * s],
                    wb[2 * s],
                    wa[2 * s + 1],
                    wb[2 * s + 1],
                    xs[2 * s],
                    xs[2 * s + 1],
                )
        accs = (acc0, acc1, acc2, acc3)
        for e in cutlass.range_constexpr(4):
            feature = first + group + (Int32(8) if e >= 2 else Int32(0))
            row = quad * Int32(2) + Int32(e % 2)
            if row < rows:
                partials[(Int64(split) * Int64(rows) + Int64(row)) * Int64(self.n) + Int64(feature)] = accs[e]
