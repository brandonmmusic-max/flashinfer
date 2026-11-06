"""Paired circular-window gather into FP8 staging storage."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from b12x._lib.intrinsics.conversion import cvt_f32x4_to_e4m3x4
from b12x._lib.intrinsics.memory import get_ptr_as_int64, st_global_u32


@dsl_user_op
def _invalid_row_trap(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class WindowStageKernel:
    def __init__(self, *, heads, dim, capacity, window, first_position=0):
        if dim % 4 or min(heads, dim, capacity, window) <= 0:
            raise ValueError(
                "window staging requires positive geometry and head dimensions divisible by four"
            )
        self.heads, self.dim = heads, dim
        self.capacity, self.window = capacity, window
        self.first_position = first_position

    @cute.jit
    def __call__(
        self,
        hk: cute.Tensor,
        hv: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        ok: cute.Tensor,
        ov: cute.Tensor,
        starts: cute.Tensor,
        states: cute.Tensor,
        offsets: cute.Tensor,
        lengths: cute.Tensor,
        counts: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        vectors = rows * self.capacity * self.heads * (self.dim // 4)
        self.kernel(
            hk, hv, key, value, ok, ov, starts, states, offsets, lengths, counts, rows
        ).launch(
            grid=(cute.ceil_div(vectors, 128), 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        hk: cute.Tensor,
        hv: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        ok: cute.Tensor,
        ov: cute.Tensor,
        starts: cute.Tensor,
        states: cute.Tensor,
        offsets: cute.Tensor,
        lengths: cute.Tensor,
        counts: cute.Tensor,
        rows: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        vector = Int64(bx) * 128 + Int64(tx)
        width = self.heads * self.dim
        element = vector * 4
        row = element // (self.capacity * width)
        if row < Int64(rows):
            position = (element // width) % self.capacity
            channel = element % width
            head = channel // self.dim
            col = channel % self.dim
            start = Int64(starts[row])
            state = Int64(states[row])
            state = cutlass.select_(state < 0, state + Int64(hk.shape[0]), state)
            if (state < 0) | (state >= Int64(hk.shape[0])):
                _invalid_row_trap()
            first = cutlass.max(start - self.window, Int64(self.first_position))
            absolute = first + position
            ring = absolute % self.window
            ring = cutlass.select_(ring < 0, ring + self.window, ring)
            current = cutlass.min(
                cutlass.max(Int64(offsets[row]) + absolute - start, Int64(0)),
                Int64(key.shape[0]) - 1,
            )
            valid = (lengths[row] > 0) & (absolute < start + Int64(counts[row]))
            kvals, vvals = [], []
            for i in cutlass.range_constexpr(4):
                k = Float32(0)
                v = Float32(0)
                if valid:
                    if absolute < start:
                        k = Float32(BFloat16(hk[state, ring, head, col + i]))
                        v = Float32(BFloat16(hv[state, ring, head, col + i]))
                    else:
                        k = Float32(key[current, head, col + i])
                        v = Float32(value[current, head, col + i])
                k = cutlass.select_(k < -448.0, Float32(-448), k)
                k = cutlass.select_(k > 448.0, Float32(448), k)
                v = cutlass.select_(v < -448.0, Float32(-448), v)
                v = cutlass.select_(v > 448.0, Float32(448), v)
                kvals.append(k)
                vvals.append(v)
            k_offset = (
                row * Int64(ok.stride[0])
                + position * Int64(ok.stride[1])
                + head * Int64(ok.stride[2])
                + col
            )
            v_offset = (
                row * Int64(ov.stride[0])
                + position * Int64(ov.stride[1])
                + head * Int64(ov.stride[2])
                + col
            )
            st_global_u32(
                get_ptr_as_int64(ok, 0) + k_offset, cvt_f32x4_to_e4m3x4(*kvals)
            )
            st_global_u32(
                get_ptr_as_int64(ov, 0) + v_offset, cvt_f32x4_to_e4m3x4(*vvals)
            )
