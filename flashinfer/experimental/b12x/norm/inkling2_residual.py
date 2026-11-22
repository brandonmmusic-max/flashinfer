"""CuTe DSL kernels for Inkling-2 residual block mix and parallel fused expert commit."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as _MLIR_T, dsl_user_op

from b12x._lib.intrinsics.memory import get_ptr_as_int64, u32_as_f32
from b12x._lib.intrinsics.reduction import warp_reduce


def _fadd(a, b):
    return a + b


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


@dsl_user_op
def _ld_nc_v4_u32(addr: Int64, *, loc=None, ip=None):
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


@dsl_user_op
def _st_v4_u32(addr: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            addr.ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_nc_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([_MLIR_T.f32(), _MLIR_T.f32(), _MLIR_T.f32(), _MLIR_T.f32()]),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(_MLIR_T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _pack_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            _MLIR_T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            _MLIR_T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            _MLIR_T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _unpack8(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32):
    return (
        u32_as_f32(w0 << Uint32(16)),
        u32_as_f32(w0 & Uint32(0xFFFF0000)),
        u32_as_f32(w1 << Uint32(16)),
        u32_as_f32(w1 & Uint32(0xFFFF0000)),
        u32_as_f32(w2 << Uint32(16)),
        u32_as_f32(w2 & Uint32(0xFFFF0000)),
        u32_as_f32(w3 << Uint32(16)),
        u32_as_f32(w3 & Uint32(0xFFFF0000)),
    )


class VecResidualMixKernel:
    """Single-pass vectorized residual mix with one barrier for all source statistics."""

    def __init__(self, d: int, s: int, has_addend: bool, normalize: bool, eps: float, norm_eps: float):
        self.d = int(d)
        self.n_threads = self.d // 8
        self.warps = self.n_threads // 32
        self.s = int(s)
        self.has_addend = bool(has_addend)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.norm_eps = float(norm_eps)

    def _smem_cls(self):
        class Smem:
            pass

        Smem.__annotations__ = {
            "red": cute.struct.Align[cute.struct.MemRange[Float32, 512], 16],
        }
        return cute.struct(Smem)

    @cute.jit
    def __call__(
        self,
        blocks: cute.Pointer,
        last: cute.Pointer,
        addend: cute.Pointer,
        queries: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        block_stride: Int64,
        row_stride: Int64,
        last_stride: Int64,
        addend_stride: Int64,
        out_stride: Int64,
        out_row_stride: Int64,
        tokens: Int32,
        query_rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(blocks),
            _flat(last),
            _flat(addend),
            _flat(queries),
            _flat(weight),
            _flat(output),
            block_stride,
            row_stride,
            last_stride,
            addend_stride,
            out_stride,
            out_row_stride,
        ).launch(grid=(tokens, query_rows, 1), block=(self.n_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        blocks: cute.Tensor,
        last: cute.Tensor,
        addend: cute.Tensor,
        queries: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        block_stride: Int64,
        row_stride: Int64,
        last_stride: Int64,
        addend_stride: Int64,
        out_stride: Int64,
        out_row_stride: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        token, query, _ = cute.arch.block_idx()
        tid = Int32(tid)
        warp = tid // Int32(32)
        lane = tid % Int32(32)
        smem = cutlass.utils.SmemAllocator().allocate(self._smem_cls())
        red = smem.red.get_tensor(cute.make_layout((512,)))

        col_b_bf16 = Int64(tid) * Int64(16)
        col_b_f32 = Int64(tid) * Int64(32)

        q_base = get_ptr_as_int64(queries, Int64(0)) + Int64(query) * Int64(self.d * 4) + col_b_f32
        qs = (*_ld_nc_v4_f32(q_base), *_ld_nc_v4_f32(q_base + Int64(16)))
        b_base = get_ptr_as_int64(blocks, Int64(0))
        l_base = get_ptr_as_int64(last, Int64(0))
        a_base = get_ptr_as_int64(addend, Int64(0))

        S = self.s
        vals = cute.make_rmem_tensor((S, 8), Float32)
        for s in cutlass.range_constexpr(S):
            if cutlass.const_expr(s < S - 1):
                vs = _unpack8(
                    *_ld_nc_v4_u32(
                        b_base
                        + (Int64(s) * block_stride + Int64(token) * row_stride) * Int64(2)
                        + col_b_bf16
                    )
                )
            else:
                vs = _unpack8(*_ld_nc_v4_u32(l_base + Int64(token) * last_stride * Int64(2) + col_b_bf16))
                if cutlass.const_expr(self.has_addend):
                    avs = _unpack8(
                        *_ld_nc_v4_u32(a_base + Int64(token) * addend_stride * Int64(2) + col_b_bf16)
                    )
                    vs = tuple(vs[i] + avs[i] for i in range(8))
            ss = Float32(0.0)
            dt = Float32(0.0)
            for i in cutlass.range_constexpr(8):
                vals[s, i] = vs[i]
                ss = ss + vs[i] * vs[i]
                dt = dt + qs[i] * vs[i]
            wss = warp_reduce(ss, _fadd)
            wdt = warp_reduce(dt, _fadd)
            if lane == Int32(0):
                red[Int32(s * 64) + warp] = wss
                red[Int32(s * 64 + 32) + warp] = wdt

        cute.arch.sync_threads()

        rstds = cute.make_rmem_tensor((S,), Float32)
        logits = cute.make_rmem_tensor((S,), Float32)
        for s in cutlass.range_constexpr(S):
            tss = Float32(0.0)
            tdt = Float32(0.0)
            for w in cutlass.range_constexpr(self.warps):
                tss = tss + red[Int32(s * 64 + w)]
                tdt = tdt + red[Int32(s * 64 + 32 + w)]
            rstd = cute.math.rsqrt(tss * Float32(1.0 / self.d) + Float32(self.eps))
            rstds[s] = rstd
            logits[s] = tdt * rstd

        peak = logits[0]
        for s in cutlass.range_constexpr(1, S):
            peak = cutlass.max(peak, logits[s])
        probs = cute.make_rmem_tensor((S,), Float32)
        tot_p = Float32(0.0)
        for s in cutlass.range_constexpr(S):
            ep = cute.math.exp(logits[s] - peak)
            probs[s] = ep
            tot_p = tot_p + ep
        inv_sum = Float32(1.0) / tot_p
        if cutlass.const_expr(self.normalize):
            for s in cutlass.range_constexpr(S):
                probs[s] = probs[s] * rstds[s]

        acc = cute.make_rmem_tensor((8,), Float32)
        acc_ss = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            a = Float32(0.0)
            for s in cutlass.range_constexpr(S):
                a = a + probs[s] * vals[s, i]
            a = a * inv_sum
            acc[i] = a
            acc_ss = acc_ss + a * a
        wass = warp_reduce(acc_ss, _fadd)
        cute.arch.sync_threads()
        if lane == Int32(0):
            red[warp] = wass
        cute.arch.sync_threads()
        tass = Float32(0.0)
        for w in cutlass.range_constexpr(self.warps):
            tass = tass + red[Int32(w)]
        scale = cute.math.rsqrt(tass / Float32(self.d) + Float32(self.norm_eps))

        wt_base = get_ptr_as_int64(weight, Int64(0)) + Int64(query) * Int64(self.d * 4) + col_b_f32
        wt = (*_ld_nc_v4_f32(wt_base), *_ld_nc_v4_f32(wt_base + Int64(16)))
        o_addr = (
            get_ptr_as_int64(output, Int64(0))
            + (Int64(query) * out_stride + Int64(token) * out_row_stride) * Int64(2)
            + col_b_bf16
        )
        _st_v4_u32(
            o_addr,
            _pack_bf16x2(acc[0] * scale * wt[0], acc[1] * scale * wt[1]),
            _pack_bf16x2(acc[2] * scale * wt[2], acc[3] * scale * wt[3]),
            _pack_bf16x2(acc[4] * scale * wt[4], acc[5] * scale * wt[5]),
            _pack_bf16x2(acc[6] * scale * wt[6], acc[7] * scale * wt[7]),
        )


class FusedExpertCommitParallelKernel:
    """Eight-warp shared-memory fused route sum, 4-tap conv, and residual block commit for 2..8 tokens."""

    def __init__(
        self,
        c: int,
        t: int,
        top_k: int,
        experts: int,
        batches: int,
        first: bool,
        store_moe: bool,
        acc_fp32: bool,
        scale: float,
    ):
        self.c, self.t, self.top_k = int(c), int(t), int(top_k)
        self.experts, self.batches = int(experts), int(batches)
        self.first, self.store_moe = bool(first), bool(store_moe)
        self.acc_fp32, self.scale = bool(acc_fp32), float(scale)

    def _smem_cls(self):
        class Smem:
            pass

        Smem.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[Float32, 11 * 32], 16],
        }
        return cute.struct(Smem)

    @cute.jit
    def __call__(
        self,
        routes: cute.Pointer,
        ids: cute.Pointer,
        w: cute.Pointer,
        h: cute.Pointer,
        states: cute.Pointer,
        starts: cute.Pointer,
        lengths: cute.Pointer,
        cu: cute.Pointer,
        residual: cute.Pointer,
        attn_out: cute.Pointer,
        moe_out: cute.Pointer,
        acc_out: cute.Pointer,
        out: cute.Pointer,
        ws: Int64,
        hs: Int64,
        hc: Int64,
        rs: Int64,
        a_s: Int64,
        mos: Int64,
        acs: Int64,
        os_: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(routes),
            _flat(ids),
            _flat(w),
            _flat(h),
            _flat(states),
            _flat(starts),
            _flat(lengths),
            _flat(cu),
            _flat(residual),
            _flat(attn_out),
            _flat(moe_out),
            _flat(acc_out),
            _flat(out),
            ws,
            hs,
            hc,
            rs,
            a_s,
            mos,
            acs,
            os_,
        ).launch(
            grid=(self.batches, (self.c + 31) // 32, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        routes: cute.Tensor,
        ids: cute.Tensor,
        w: cute.Tensor,
        h: cute.Tensor,
        states: cute.Tensor,
        starts: cute.Tensor,
        lengths: cute.Tensor,
        cu: cute.Tensor,
        residual: cute.Tensor,
        attn_out: cute.Tensor,
        moe_out: cute.Tensor,
        acc_out: cute.Tensor,
        out: cute.Tensor,
        ws: Int64,
        hs: Int64,
        hc: Int64,
        rs: Int64,
        a_s: Int64,
        mos: Int64,
        acs: Int64,
        os_: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch, ctile, _ = cute.arch.block_idx()
        warp = Int32(tid) // Int32(32)
        lane = Int32(tid) % Int32(32)
        col = Int32(ctile) * Int32(32) + lane
        valid_c = col < Int32(self.c)

        n = Int32(lengths[Int32(batch)])
        state = Int64(states[Int32(batch)])
        begin = Int32(cu[Int32(batch)])
        start = Int64(starts[Int32(batch)])

        smem = cutlass.utils.SmemAllocator().allocate(self._smem_cls())
        buf = smem.buf.get_tensor(cute.make_layout((11 * 32,)))

        if warp < Int32(3):
            hv = Float32(0.0)
            if valid_c and state > Int64(0) and start >= Int64(Int32(3) - warp):
                hv = Float32(h[state * hs + Int64(col) * hc + Int64(warp)])
            buf[warp * Int32(32) + lane] = hv

        if warp < n and valid_c:
            row = begin + warp
            acc_moe = Float32(0.0)
            for slot in cutlass.range_constexpr(self.top_k):
                rt = row * Int32(self.top_k) + Int32(slot)
                exp = Int32(ids[rt])
                if exp >= Int32(0) and exp < Int32(self.experts):
                    acc_moe = acc_moe + Float32(routes[Int64(rt) * Int64(self.c) + Int64(col)])
            moe_bf16 = acc_moe.to(BFloat16)
            if cutlass.const_expr(self.store_moe):
                moe_out[Int64(row) * mos + Int64(col)] = moe_bf16
            buf[(warp + Int32(3)) * Int32(32) + lane] = Float32(moe_bf16)

        cute.arch.sync_threads()

        if warp < n and valid_c:
            row = begin + warp
            w_base = Int64(col) * ws
            w0 = Float32(w[w_base + Int64(0)])
            w1 = Float32(w[w_base + Int64(1)])
            w2 = Float32(w[w_base + Int64(2)])
            w3 = Float32(w[w_base + Int64(3)])
            x0 = buf[warp * Int32(32) + lane]
            x1 = buf[(warp + Int32(1)) * Int32(32) + lane]
            x2 = buf[(warp + Int32(2)) * Int32(32) + lane]
            x3 = buf[(warp + Int32(3)) * Int32(32) + lane]
            acc_c = _add_rn(_add_rn(_add_rn(_mul_rn(x0, w0), _mul_rn(x1, w1)), _mul_rn(x2, w2)), _mul_rn(x3, w3))
            conv_f = Float32(0.0)
            if state > Int64(0):
                conv_f = _mul_rn(_add_rn(acc_c, x3), Float32(self.scale)).to(BFloat16).to(Float32)
            attn = Float32(attn_out[Int64(row) * a_s + Int64(col)])
            if cutlass.const_expr(self.first):
                out_val = attn + conv_f
            else:
                out_val = (Float32(residual[Int64(row) * rs + Int64(col)]) + attn) + conv_f
            if cutlass.const_expr(self.acc_fp32):
                acc_out[Int64(row) * acs + Int64(col)] = out_val
            else:
                acc_out[Int64(row) * acs + Int64(col)] = out_val.to(BFloat16)
            out[Int64(row) * os_ + Int64(col)] = out_val.to(BFloat16)

        if warp < Int32(3) and n > Int32(0) and state > Int64(0) and valid_c:
            h[state * hs + Int64(col) * hc + Int64(warp)] = buf[(n + warp) * Int32(32) + lane].to(BFloat16)

        if Int32(batch) == Int32(0) and valid_c:
            end = Int32(cu[Int32(self.batches)])
            if warp >= end and warp < Int32(self.t):
                attn = Float32(attn_out[Int64(warp) * a_s + Int64(col)])
                if cutlass.const_expr(self.first):
                    pv = attn
                else:
                    pv = Float32(residual[Int64(warp) * rs + Int64(col)]) + attn
                if cutlass.const_expr(self.acc_fp32):
                    acc_out[Int64(warp) * acs + Int64(col)] = pv
                else:
                    acc_out[Int64(warp) * acs + Int64(col)] = pv.to(BFloat16)
                out[Int64(warp) * os_ + Int64(col)] = pv.to(BFloat16)
