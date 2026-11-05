"""Fused TrellisMX boundary kernels: one launch for the whole input stage.

The reference input stage is ``bf16 -> cvt.rn fp16 -> H512 -> * gate_up_suh ->
H128 -> E4M3/UE8M0-k32``. As torch ops it is ~40 small kernel launches per call
(~153 us, flat in m); the same arithmetic fits in one CTA per
token, so this module implements it as a single launch with two output modes
selected at launch:

* ``mode="e4m3"`` -- the E4M3 payload (native K32 permutation) plus the UE8M0
  K32 scale bytes, byte-exact to
  ``trellismx_boundary.input_boundary_quantized`` (which was byte-verified
  against the release runtime's own operands). This is the operand pair the
  dynamic kernel's pre-quantized A arm consumes.
* ``mode="fp16"`` -- the same FP32 chain cast once to FP16, bit-exact to
  ``input_boundary(...).to(torch.float16)``, for the unquantized small-M arm.

``fused_output_stage`` is the second kernel: ``outer_h512(down_svh *
h128(reduced))``, bit-exact to the same composition of the torch ops.

Ordering and rounding belong to the format: FP32 continuity between the H512
and the quantizer, and one correctly-rounded ``div.rn.f32`` by the float32
encoding of sqrt(n) at each Hadamard's end (the release records that a
reciprocal multiply can move an E4M3 tie by one code).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# The reference ops the kernels below reproduce byte for byte.
from b12x.moe._shared.kernels.trellismx_boundary import (
    K32_PERMUTATION,
    SQRT128_F32,
    SQRT512_F32,
)

__all__ = [
    "BOUNDARY_KERNEL_VERSION",
    "TrellisMxFusedBoundary",
    "TrellisMxFusedOutputStage",
    "fused_input_boundary_quantized",
    "fused_input_boundary_fp16",
    "fused_output_stage",
]

#: Compile identity of the input-stage kernel. 4: exactly two modes (``e4m3``,
#: ``fp16``), and the K32 quantizer's exponent is exact and UE8M0-clamped for
#: every finite block maximum. 2: one CTA per 512-value chunk.
BOUNDARY_KERNEL_VERSION = 4

_THREADS = 128
#: Values one CTA owns: one H512 block, one `* gate_up_suh` window and one
#: K32 scale group of the quantizer, so no cross-CTA exchange is ever needed.
_CHUNK = 512
_E4M3_MAX = 448.0


@dsl_user_op
def _div_rn_f32(a, b, *, loc=None, ip=None):
    """One IEEE round-to-nearest FP32 division (the declared normalization)."""

    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _add_rn_f32(a, b, *, loc=None, ip=None):
    """One IEEE FP32 add that ptxas will not fuse into an FMA."""

    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _sub_rn_f32(a, b, *, loc=None, ip=None):
    """One IEEE FP32 subtract, opaque to contraction."""

    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _mul_rn_f32(a, b, *, loc=None, ip=None):
    """One IEEE FP32 multiply, opaque to contraction."""

    result = llvm.inline_asm(
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
    return cutlass.Float32(result)


@dsl_user_op
def _ue8m0_exponent_s32(value, *, loc=None, ip=None):
    """``clamp(ceil(log2(v)), -127, 127)`` for finite v >= 0, exactly, from the fp32 bits.

    A normal ``v = 2**e * m`` with ``1 <= m < 2`` gives ``e`` when the mantissa
    field is zero and ``e + 1`` otherwise. Below ``2**-126`` (zero or subnormal,
    ``v = mant * 2**-149``) ``ceil(log2(v))`` is -126 when ``v > 2**-127`` and at
    most -127 otherwise, which the UE8M0 range clamps to -127; ``v = 0`` (an
    ``amax / 448`` that underflowed) also gives -127, as the reference's
    ``ceil(log2(0)).clamp(-127, 127)`` does.

    The input must be finite. Infinity and NaN have undefined results here;
    callers derive ``value`` from a finite FP32 block maximum.
    """

    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 bits, expo, mant, small;
            .reg .pred has_mant, tiny, above;
            mov.b32 bits, $1;
            shr.u32 expo, bits, 23;
            and.b32 expo, expo, 0xff;
            setp.eq.u32 tiny, expo, 0;
            add.s32 expo, expo, -127;
            and.b32 mant, bits, 0x7fffff;
            setp.ne.b32 has_mant, mant, 0;
            selp.s32 $0, 1, 0, has_mant;
            add.s32 $0, $0, expo;
            min.s32 $0, $0, 127;
            setp.gt.u32 above, mant, 0x400000;
            selp.s32 small, -126, -127, above;
            selp.s32 $0, small, $0, tiny;
        }
        """,
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _pow2_s32(exponent, *, loc=None, ip=None):
    """``2**exponent`` for an integer exponent in [-149, 127] (exact bits).

    Exponents below -126 give the subnormal ``1 << (exponent + 149)``; in
    particular ``2**-127`` is ``0x00400000``, never zero.
    """

    result = llvm.inline_asm(
        T.f32(),
        [Int32(exponent).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 bits, shift, sub;
            .reg .pred subnormal;
            add.s32 bits, $1, 127;
            shl.b32 bits, bits, 23;
            add.s32 shift, $1, 149;
            shl.b32 sub, 1, shift;
            setp.lt.s32 subnormal, $1, -126;
            selp.b32 bits, sub, bits, subnormal;
            mov.b32 $0, bits;
        }
        """,
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _e4m3x2_rn_f32(a, b, *, loc=None, ip=None):
    """Two FP32 values -> two packed E4M3 bytes, RNE + satfinite, one rounding."""

    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 packed;
            cvt.rn.satfinite.e4m3x2.f32 packed, $1, $2;
            cvt.u32.u16 $0, packed;
        }
        """,
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _hadamard_shared(
    buf, hidden: cutlass.Constexpr, size: cutlass.Constexpr, tid: Int32
):
    """In-place normalized Sylvester Hadamard of ``size`` on ``buf``.

    Natural order, contiguous ``size``-blocks, one terminal ``div.rn.f32`` by
    the float32 encoding of sqrt(size).
    """

    threads = _THREADS
    stages = int(size).bit_length() - 1
    half = int(size) // 2
    for stage in cutlass.range_constexpr(stages):
        stride = 1 << stage
        for q in cutlass.range_constexpr((int(hidden) // 2) // threads):
            p = tid + Int32(q * threads)
            block = p // Int32(half)
            local = p - block * Int32(half)
            group = local // Int32(stride)
            j = (
                block * Int32(int(size))
                + group * Int32(2 * stride)
                + (local - group * Int32(stride))
            )
            left = buf[j]
            right = buf[j + Int32(stride)]
            buf[j] = _add_rn_f32(left, right)
            buf[j + Int32(stride)] = _sub_rn_f32(left, right)
        cute.arch.sync_threads()
    divisor = SQRT512_F32 if int(size) == 512 else SQRT128_F32
    for q in cutlass.range_constexpr(int(hidden) // threads):
        j = tid + Int32(q * threads)
        buf[j] = _div_rn_f32(buf[j], cutlass.Float32(divisor))
    cute.arch.sync_threads()


@cute.jit
def _quantize_k32_chunk(buf, payload, scales, bidx: Int32, cidx: Int32, tid: Int32):
    """E4M3 payload (native K32 order) and UE8M0 bytes of one 512-value chunk.

    ``buf`` holds the chunk's FP32 values in shared memory; the chunk's 512
    payload bytes and 16 scale bytes go to row ``bidx`` of ``payload`` and
    ``scales``. The scale exponent is the exact finite-FP32 ceiling log2 of
    ``amax / 448`` clamped to ``[-127, 127]``. A block whose quotient is
    subnormal or underflows to zero gets byte 0 and the subnormal scale
    ``2**-127``, and an all-zero block gets byte 127 and scale 1: the divisor
    is never zero. Non-finite quotients are outside the domain.

    Byte-exact to ``trellismx_boundary.quantize_e4m3_ue8m0_k32_codes`` on the
    CPU for every finite value: the reference derives the exponent from the
    quotient's exact FP32 bits rather than evaluating a float32 ``log2``.
    """

    blocks = _CHUNK // 32
    offset = cidx * Int32(_CHUNK)
    # One block per thread where the CTA is wide enough for the chunk, and a
    # strided pass otherwise; `blocks` is 16 here, so this is one iteration.
    for b in cutlass.range_constexpr((blocks + _THREADS - 1) // _THREADS):
        blk = tid + Int32(b * _THREADS)
        if blk < Int32(blocks):
            base = blk * Int32(32)
            amax = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(32):
                v = buf[base + Int32(e)]
                a = v if v >= cutlass.Float32(0.0) else -v
                amax = a if a > amax else amax
            exponent = _ue8m0_exponent_s32(
                _div_rn_f32(amax, cutlass.Float32(_E4M3_MAX))
            )
            scale = _pow2_s32(exponent)
            if amax <= cutlass.Float32(0.0):
                scale = cutlass.Float32(1.0)
                exponent = Int32(0)
            scales[bidx, cidx * Int32(blocks) + blk] = Uint8(exponent + Int32(127))
            for e in cutlass.range_constexpr(16):
                lo = _div_rn_f32(buf[base + Int32(K32_PERMUTATION[2 * e])], scale)
                hi = _div_rn_f32(buf[base + Int32(K32_PERMUTATION[2 * e + 1])], scale)
                # cvt.rn.satfinite.e4m3x2.f32 writes its first source
                # to the high byte, so pass (hi, lo) to get lo in 0..7.
                word = _e4m3x2_rn_f32(hi, lo)
                payload[bidx, offset + base + Int32(2 * e)] = Uint8(word & Uint32(0xFF))
                payload[bidx, offset + base + Int32(2 * e + 1)] = Uint8(
                    (word >> Uint32(8)) & Uint32(0xFF)
                )


class TrellisMxFusedBoundary:
    """Input stage: bf16 x [tokens, hidden] -> E4M3/UE8M0 or FP16 operands."""

    def __init__(self, hidden: int, mode: str):
        if hidden % 512:
            raise ValueError(f"hidden must be a multiple of 512, got {hidden}")
        if mode not in ("e4m3", "fp16"):
            raise ValueError(f"unsupported mode {mode!r}; expected 'e4m3' or 'fp16'")
        self.hidden = int(hidden)
        self.mode = str(mode)
        # Part of the compile key (instance state), so a changed kernel never
        # reuses a program compiled from an earlier version.
        self.version = BOUNDARY_KERNEL_VERSION

    @cute.jit
    def __call__(self, x, suh, payload, scales, values, stream):
        tokens = cute.size(x, 0)
        chunks = self.hidden // _CHUNK
        # Grid: one CTA per (token, 512-value chunk), so each token launches
        # hidden // 512 CTAs and decode sizes keep the GPU occupied.
        self.kernel(x, suh, payload, scales, values).launch(
            grid=(tokens, chunks, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, suh, payload, scales, values):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, cidx, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        offset = Int32(cidx) * Int32(_CHUNK)
        # `suh` is [hidden] from the reference op and [1, hidden] from the
        # small-M arm's plan; both are the same flat vector.
        suh_flat = cute.make_tensor(
            suh.iterator, cute.make_layout((self.hidden,), stride=(1,))
        )
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(_CHUNK), byte_alignment=16
        )
        for i in cutlass.range_constexpr(_CHUNK // _THREADS):
            j = tid + Int32(i * _THREADS)
            buf[j] = cutlass.Float32(cutlass.Float16(x[bidx, offset + j]))
        cute.arch.sync_threads()
        # H512, then `* gate_up_suh` and the H128: all three are block-local to
        # the 512 values this CTA owns, so the chunk never leaves the CTA.
        _hadamard_shared(buf, _CHUNK, _CHUNK, tid)
        for i in cutlass.range_constexpr(_CHUNK // _THREADS):
            j = tid + Int32(i * _THREADS)
            buf[j] = _mul_rn_f32(buf[j], cutlass.Float32(suh_flat[offset + j]))
        # The H128's first stage pairs values two threads wrote.
        cute.arch.sync_threads()
        _hadamard_shared(buf, _CHUNK, 128, tid)
        # Exactly one output per mode: FP16 values or the E4M3/UE8M0 operands.
        if cutlass.const_expr(self.mode == "fp16"):
            for i in cutlass.range_constexpr(_CHUNK // _THREADS):
                j = tid + Int32(i * _THREADS)
                values[bidx, offset + j] = cutlass.Float16(buf[j])
        else:
            _quantize_k32_chunk(buf, payload, scales, bidx, Int32(cidx), tid)


class TrellisMxFusedOutputStage:
    """Output stage: reduced [tokens, hidden] -> outer_h512(down_svh * h128)."""

    def __init__(self, hidden: int, output_dtype: str = "bf16"):
        if hidden % 512:
            raise ValueError(f"hidden must be a multiple of 512, got {hidden}")
        if output_dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError(
                f"unsupported output dtype {output_dtype!r}; expected bf16/fp16/fp32"
            )
        self.hidden = int(hidden)
        self.output_dtype = str(output_dtype)

    @cute.jit
    def __call__(self, reduced, down_svh, out, stream):
        tokens = cute.size(reduced, 0)
        chunks = self.hidden // _CHUNK
        self.kernel(reduced, down_svh, out).launch(
            grid=(tokens, chunks, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, reduced, down_svh, out):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, cidx, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        offset = Int32(cidx) * Int32(_CHUNK)
        # `down_svh` is [hidden] or [1, hidden]; read it as one flat vector.
        svh_flat = cute.make_tensor(
            down_svh.iterator, cute.make_layout((self.hidden,), stride=(1,))
        )
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(_CHUNK), byte_alignment=16
        )
        for i in cutlass.range_constexpr(_CHUNK // _THREADS):
            j = tid + Int32(i * _THREADS)
            buf[j] = reduced[bidx, offset + j]
        # H128, then `* down_svh`, then the outer H512 over the same chunk.
        # Each Hadamard's first stage pairs values two threads wrote, so a
        # barrier closes every producer before it.
        cute.arch.sync_threads()
        _hadamard_shared(buf, _CHUNK, 128, tid)
        for i in cutlass.range_constexpr(_CHUNK // _THREADS):
            j = tid + Int32(i * _THREADS)
            buf[j] = _mul_rn_f32(buf[j], cutlass.Float32(svh_flat[offset + j]))
        cute.arch.sync_threads()
        _hadamard_shared(buf, _CHUNK, _CHUNK, tid)
        for i in cutlass.range_constexpr(_CHUNK // _THREADS):
            j = tid + Int32(i * _THREADS)
            if cutlass.const_expr(self.output_dtype == "bf16"):
                out[bidx, offset + j] = cutlass.BFloat16(buf[j])
            elif cutlass.const_expr(self.output_dtype == "fp16"):
                out[bidx, offset + j] = cutlass.Float16(buf[j])
            else:
                out[bidx, offset + j] = cutlass.Float32(buf[j])


# ---------------------------------------------------------------------------
# Dispatch helpers
#
# The fused kernels are compiled per (hidden, mode, tokens, device) because the
# token count is a traced shape, and the first call for a key is made eagerly so
# a later call inside a CUDA graph finds a compiled program (the same rule the
# reference op's permutation cache follows). The torch reference stays the CPU
# path and the test oracle.
# ---------------------------------------------------------------------------


def _stream():
    from b12x._lib.utils.device import current_cuda_stream

    return current_cuda_stream()


_LAUNCHERS: dict[tuple, object] = {}
#: Zero-filled scratch for the operand the selected mode does not write. One
#: buffer per shape is enough: nothing reads it, and a per-call allocation
#: would add a launch inside a captured region.
_DUMMIES: dict[tuple, torch.Tensor] = {}


def _planning() -> bool:
    """Whether a b12x compile factory is describing programs, not compiling them."""

    from b12x._lib.compile_plan import planning

    return planning()


def _compile_input(hidden: int, mode: str, m: int, device: torch.device):
    key = (hidden, mode, m, str(device))
    compiled = _LAUNCHERS.get(key)
    if compiled is not None:
        return compiled
    import cutlass
    from cutlass.cute.runtime import make_fake_compact_tensor

    from b12x._lib.compiler import compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream

    # Fake compact tensors, never CUDA allocations: preparation's discovery
    # compiles this in workers that reject CUDA
    row_major = (1, 0)
    # The scope pins the target the program is compiled for: the boundary is
    # specialized per (hidden, mode, m) and launches on that device's stream.
    with torch.cuda.device(device):
        compiled = b12x_compile(
            TrellisMxFusedBoundary(hidden, mode),
            make_fake_compact_tensor(
                cutlass.BFloat16, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            make_fake_compact_tensor(cutlass.Float16, (hidden,), assumed_align=16),
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
        )
    # A planning pass describes the program instead of compiling it; caching
    # that description would hand the runtime an artifact it must compile.
    if not _planning():
        _LAUNCHERS[key] = compiled
    return compiled


def _compile_output(hidden: int, m: int, device: torch.device, dtype: torch.dtype):
    key = ("out", hidden, m, str(device), dtype)
    compiled = _LAUNCHERS.get(key)
    if compiled is not None:
        return compiled
    import cutlass
    from cutlass.cute.runtime import make_fake_compact_tensor

    from b12x._lib.compiler import compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream

    row_major = (1, 0)
    output_type = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }.get(dtype)
    if output_type is None:
        raise ValueError(f"unsupported output dtype {dtype}; expected bf16/fp16/fp32")
    output_dtype = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }[dtype]
    kernel = TrellisMxFusedOutputStage(hidden, output_dtype)
    # Same rule as the input stage: fake operands, one program per
    # (hidden, m, output dtype).
    with torch.cuda.device(device):
        compiled = b12x_compile(
            kernel,
            make_fake_compact_tensor(
                cutlass.Float32, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            make_fake_compact_tensor(cutlass.Float16, (hidden,), assumed_align=16),
            make_fake_compact_tensor(
                output_type, (m, hidden), stride_order=row_major, assumed_align=16
            ),
            current_cuda_stream(),
        )
    # A planning pass describes the program instead of compiling it; caching
    # that description would hand the runtime an artifact it must compile.
    if not _planning():
        _LAUNCHERS[key] = compiled
    return compiled


def _scratch(hidden: int, m: int, device: torch.device, dtype: torch.dtype):
    key = (hidden, m, str(device), dtype)
    buf = _DUMMIES.get(key)
    if buf is None:
        buf = torch.zeros(m, hidden, dtype=dtype, device=device)
        _DUMMIES[key] = buf
    return buf


def fused_input_boundary_quantized(
    x_bf16: torch.Tensor, gate_up_suh: torch.Tensor, *, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's A operands from the fused kernel (torch op on CPU)."""

    from b12x.moe._shared.kernels.trellismx_boundary import (
        input_boundary_quantized,
    )

    if x_bf16.device.type != "cuda":
        return input_boundary_quantized(x_bf16, gate_up_suh, hidden_size=hidden_size)
    hidden = int(hidden_size)
    m = int(x_bf16.shape[0])
    device = x_bf16.device
    compiled = _compile_input(hidden, "e4m3", m, device)
    payload = torch.empty((m, hidden), dtype=torch.uint8, device=device)
    scales = torch.empty((m, hidden // 32), dtype=torch.uint8, device=device)
    compiled(
        x_bf16,
        gate_up_suh,
        payload,
        scales,
        _scratch(hidden, m, device, torch.float16),
        _stream(),
    )
    return payload, scales


def fused_input_boundary_fp16(
    x_bf16: torch.Tensor, gate_up_suh: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """The unquantized input stage as FP16 values (torch chain on CPU)."""

    from b12x.moe._shared.kernels.trellismx_boundary import input_boundary

    if x_bf16.device.type != "cuda":
        return input_boundary(x_bf16, gate_up_suh, hidden_size=hidden_size).to(
            torch.float16
        )
    hidden = int(hidden_size)
    m = int(x_bf16.shape[0])
    device = x_bf16.device
    compiled = _compile_input(hidden, "fp16", m, device)
    values = torch.empty((m, hidden), dtype=torch.float16, device=device)
    compiled(
        x_bf16,
        gate_up_suh,
        _scratch(hidden, m, device, torch.uint8),
        torch.empty((m, hidden // 32), dtype=torch.uint8, device=device),
        values,
        _stream(),
    )
    return values


def fused_output_stage(
    reduced: torch.Tensor, down_svh: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """``outer_h512(down_svh * h128(reduced))`` -> the caller's dtype."""

    from b12x.moe._shared.kernels.trellismx_boundary import (
        h128_normalized,
        h512_normalized,
    )

    hidden = int(hidden_size)
    if reduced.device.type != "cuda":
        staged = h512_normalized(
            h128_normalized(reduced.to(torch.float32)) * down_svh.to(torch.float32)
        )
        return staged.to(reduced.dtype)
    m = int(reduced.shape[0])
    device = reduced.device
    compiled = _compile_output(hidden, m, device, reduced.dtype)
    out = torch.empty((m, hidden), dtype=reduced.dtype, device=device)
    compiled(reduced.to(torch.float32), down_svh, out, _stream())
    return out
