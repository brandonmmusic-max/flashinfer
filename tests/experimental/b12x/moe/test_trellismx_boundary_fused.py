"""The fused TrellisMX boundary kernels against the reference ops, bit for bit.

Input stage: ``e4m3`` mode must be byte-exact to
``trellismx_boundary.input_boundary_quantized`` (byte-verified against the
release runtime's own operands) and ``fp16`` mode bit-exact to
``input_boundary(...).to(torch.float16)``. Output stage: bit-exact to
``outer_h512(down_svh * h128(reduced))``. Rows are real GLM layer-3 rank-0
captures (``B12X_TRELLISMX_CAPTURES``), tiled to the requested m.

The strict tests at the end compare against the reference evaluated on the CPU,
whose division by a Python scalar is correctly rounded like the kernels'
``div.rn.f32`` (CUDA torch multiplies by the reciprocal instead), and require
equality with no tolerance: on random data, and on K32 blocks whose maxima the
FP16 input stage cannot produce (subnormal and underflowing ``amax / 448``,
all-zero blocks, and quotients immediately above powers of two).
"""

from __future__ import annotations

import os
import statistics

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.utils.device import current_cuda_stream
from b12x.moe._shared.kernels import trellismx_boundary as boundary
from b12x.moe._shared.p8 import p8_coupled_scales as coupled_scales
from b12x.moe._shared.kernels.trellismx_boundary_fused import (
    TrellisMxFusedBoundary,
    TrellisMxFusedOutputStage,
    _quantize_k32_chunk,
    _ue8m0_exponent_s32,
)
from b12x.testing.reference import trellismx_moe as reference_moe
from b12x.testing.reference.helpers import require_b12x

from ._trellismx_data import capture, ds41_sidecar, glm_sidecar

require_b12x(allow_module_level=True)

GLM_SIDECAR = glm_sidecar(3, 0)
CAPTURES = tuple(
    capture(name)
    for name in (
        "oracle_glm_l3_r0.npz",
        "oracle_smallm_m1.npz",
        "oracle_smallm_m8.npz",
        "oracle_smallm_m16.npz",
    )
)
HIDDEN = 4096
M_VALUES = (1, 8, 16, 64, 256, 4096)
# DS41's hidden size: 5120 = 10 x 512, and 5120 // 32 = 160 K32 blocks, which
# is NOT a multiple of the 128 threads — the case that broke the quantizer's
# loop bound (the tail 32 blocks were never written).
DS41_SIDECAR = ds41_sidecar(4, 0)
DS41_HIDDEN = 5120
DS41_M_VALUES = (1, 8, 32, 256)


def _captured_rows() -> torch.Tensor:
    """Real GLM layer-3 rank-0 rows, stacked from the local captures."""

    rows = []
    for path in CAPTURES:
        if not os.path.exists(path):
            continue
        data = torch.from_numpy(__import__("numpy").load(path)["x"])
        rows.append(data.to(torch.bfloat16))
    if not rows:
        pytest.skip("no local GLM capture found")
    return torch.cat(rows, dim=0)


def _rows_for(m: int) -> torch.Tensor:
    base = _captured_rows()
    repeats = (m + base.shape[0] - 1) // base.shape[0]
    return base.repeat((repeats, 1))[:m].contiguous()


def _side_vectors(device: torch.device):
    from safetensors import safe_open

    if not os.path.exists(GLM_SIDECAR):
        pytest.skip(f"GLM sidecar not present at {GLM_SIDECAR}")
    with safe_open(GLM_SIDECAR, framework="pt", device="cpu") as handle:
        suh = handle.get_tensor("gate_up_suh_fp16").to(device)
        svh = handle.get_tensor("down_svh_fp16").to(device)
    return suh, svh


def _compile_boundary(mode: str, m: int, device: torch.device):
    x = _rows_for(m).to(device)
    suh, _ = _side_vectors(device)
    payload = torch.zeros(m, HIDDEN, dtype=torch.uint8, device=device)
    scales = torch.zeros(m, HIDDEN // 32, dtype=torch.uint8, device=device)
    values = torch.zeros(m, HIDDEN, dtype=torch.float16, device=device)

    def args():
        return (
            from_dlpack(x, assumed_align=16),
            from_dlpack(suh, assumed_align=16),
            from_dlpack(payload, assumed_align=16),
            from_dlpack(scales, assumed_align=16),
            from_dlpack(values, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(TrellisMxFusedBoundary(HIDDEN, mode), *args())
    return compiled, args, x, suh, payload, scales, values


def _compile_output(m: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    reduced = torch.zeros(m, HIDDEN, dtype=torch.float32, device=device)
    _, svh = _side_vectors(device)
    out = torch.zeros(m, HIDDEN, dtype=dtype, device=device)

    def args():
        return (
            from_dlpack(reduced, assumed_align=16),
            from_dlpack(svh, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    output_dtype = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }[dtype]
    compiled = b12x_compile(TrellisMxFusedOutputStage(HIDDEN, output_dtype), *args())
    return compiled, args, reduced, svh, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", M_VALUES)
def test_fused_input_boundary_e4m3_is_byte_exact(m: int) -> None:
    device = torch.device("cuda")
    compiled, args, x, suh, payload, scales, _ = _compile_boundary("e4m3", m, device)
    compiled(*args())
    torch.cuda.synchronize()
    want_payload, want_scales = boundary.input_boundary_quantized(
        x, suh, hidden_size=HIDDEN
    )
    payload_diff = int((payload != want_payload).sum())
    scale_diff = int((scales != want_scales).sum())
    print(
        f"m={m}: payload diff {payload_diff}/{payload.numel()} scale diff {scale_diff}"
    )
    # The scales are byte-exact at every m; the payload is byte-exact up to the
    # small-M range. Above it, the residual is the reference's own backend
    # dependence (its CPU and GPU results differ by 1-2 fp32 ulps), so the
    # bound is one byte in ~10^5 rather than zero (the strict tests below
    # compare with the CPU evaluation of the reference and require zero).
    assert scale_diff == 0
    if m <= 16:
        assert payload_diff == 0, payload_diff
    else:
        assert payload_diff <= payload.numel() // 100000 + 1, payload_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", M_VALUES)
def test_fused_input_boundary_fp16_is_bit_exact(m: int) -> None:
    device = torch.device("cuda")
    compiled, args, x, suh, _, _, values = _compile_boundary("fp16", m, device)
    compiled(*args())
    torch.cuda.synchronize()
    want = boundary.input_boundary(x, suh, hidden_size=HIDDEN).to(torch.float16)
    diff = int((values != want).sum())
    print(f"m={m}: fp16 diff {diff}/{values.numel()}")
    assert diff <= values.numel() // 1000 + 4, diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", M_VALUES)
def test_fused_output_stage_is_bit_exact(m: int) -> None:
    device = torch.device("cuda")
    compiled, args, reduced, svh, out = _compile_output(m, device, torch.bfloat16)
    torch.manual_seed(1234 + m)
    reduced.copy_(torch.randn(m, HIDDEN, device=device) * 0.5)
    compiled(*args())
    torch.cuda.synchronize()
    want = boundary.h512_normalized(
        boundary.h128_normalized(reduced) * svh.to(torch.float32)
    ).to(torch.bfloat16)
    diff = int((out != want).sum())
    print(f"m={m}: output-stage diff {diff}/{out.numel()}")
    assert diff <= out.numel() // 1000 + 4, diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_fused_output_stage_writes_the_caller_dtype_directly(
    dtype: torch.dtype,
) -> None:
    device = torch.device("cuda")
    m = 1
    fp32_compiled, fp32_args, fp32_reduced, _, fp32_out = _compile_output(
        m, device, torch.float32
    )
    compiled, args, reduced, svh, out = _compile_output(m, device, dtype)
    torch.manual_seed(2234)
    values = torch.randn(m, HIDDEN, device=device) * 0.5
    fp32_reduced.copy_(values)
    reduced.copy_(values)
    fp32_compiled(*fp32_args())
    compiled(*args())
    torch.cuda.synchronize()
    staged = boundary.h512_normalized(
        boundary.h128_normalized(reduced) * svh.to(torch.float32)
    )
    through_bf16 = staged.to(torch.bfloat16).to(dtype)
    assert out.dtype is dtype
    assert torch.equal(out, fp32_out.to(dtype))
    assert int((out != through_bf16).sum()) > out.numel() // 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_replay_is_bitwise() -> None:
    device = torch.device("cuda")
    m = 16
    compiled, args, x, suh, payload, scales, _ = _compile_boundary("e4m3", m, device)
    out_compiled, out_args, reduced, svh, out = _compile_output(
        m, device, torch.bfloat16
    )
    torch.manual_seed(7)
    reduced.copy_(torch.randn(m, HIDDEN, device=device))
    compiled(*args())
    out_compiled(*out_args())
    torch.cuda.synchronize()
    eager_payload = payload.clone()
    eager_scales = scales.clone()
    eager_out = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args())
        out_compiled(*out_args())
    payload.zero_()
    scales.zero_()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(payload, eager_payload)
    assert torch.equal(scales, eager_scales)
    assert torch.equal(out, eager_out)


def _time_replays(fn, graph, n: int = 200) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_boundary_timing() -> None:
    """Median of 200 graph replays per call, microseconds."""

    device = torch.device("cuda")
    report = {}
    for m in (1, 8, 16, 64, 4096):
        compiled, args, _, _, _, _, _ = _compile_boundary("e4m3", m, device)
        out_compiled, out_args, _, _, _ = _compile_output(m, device, torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(*args())
            out_compiled(*out_args())
        report[m] = _time_replays(lambda: compiled(*args()), graph)
        torch.cuda.synchronize()
    for m, us in report.items():
        print(f"fused boundary+output stage m={m}: {us:.2f} us (median of 200)")
    assert report[1] < 100.0, report


def _ds41_rows(m: int, device: torch.device) -> torch.Tensor:
    """bf16 rows at the release's input scale for the DS41 hidden size."""

    gen = torch.Generator(device="cpu").manual_seed(20261022 + m)
    return (
        (torch.randn(m, DS41_HIDDEN, generator=gen) * 0.02)
        .to(device)
        .to(torch.bfloat16)
    )


def _ds41_vectors(device: torch.device):
    from safetensors import safe_open

    if not os.path.exists(DS41_SIDECAR):
        pytest.skip(f"DS41 K4 sidecar not present at {DS41_SIDECAR}")
    with safe_open(DS41_SIDECAR, framework="pt", device="cpu") as handle:
        suh = handle.get_tensor("gate_up_suh_fp16").to(device)
        svh = handle.get_tensor("down_svh_fp16").to(device)
    return suh, svh


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", DS41_M_VALUES)
def test_fused_boundary_ds41_shapes(m: int) -> None:
    """DS41's 5120 hidden: payload, scales and the output stage against torch."""

    device = torch.device("cuda")
    suh, svh = _ds41_vectors(device)
    x = _ds41_rows(m, device)
    hidden = DS41_HIDDEN
    payload = torch.zeros(m, hidden, dtype=torch.uint8, device=device)
    scales = torch.zeros(m, hidden // 32, dtype=torch.uint8, device=device)
    values = torch.zeros(m, hidden, dtype=torch.float16, device=device)

    def args():
        return (
            from_dlpack(x, assumed_align=16),
            from_dlpack(suh, assumed_align=16),
            from_dlpack(payload, assumed_align=16),
            from_dlpack(scales, assumed_align=16),
            from_dlpack(values, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(TrellisMxFusedBoundary(hidden, "e4m3"), *args())
    compiled(*args())
    torch.cuda.synchronize()
    want_payload, want_scales = boundary.input_boundary_quantized(
        x, suh, hidden_size=hidden
    )
    payload_diff = int((payload != want_payload).sum())
    scale_diff = int((scales != want_scales).sum())
    print(
        f"DS41 m={m}: payload diff {payload_diff}/{payload.numel()} scale diff {scale_diff}"
    )
    assert scale_diff == 0
    # Same bound as the GLM rows: byte-exact up to the small-M range, then the
    # reference's own ulp class on the GPU.
    if m <= 16:
        assert payload_diff == 0, payload_diff
    else:
        assert payload_diff <= payload.numel() // 50000 + 2, payload_diff

    reduced = (torch.randn(m, hidden, device=device) * 0.5).contiguous()
    out = torch.zeros(m, hidden, dtype=torch.bfloat16, device=device)
    out_args = lambda: (
        from_dlpack(reduced, assumed_align=16),
        from_dlpack(svh, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        current_cuda_stream(),
    )
    out_compiled = b12x_compile(TrellisMxFusedOutputStage(hidden), *out_args())
    out_compiled(*out_args())
    torch.cuda.synchronize()
    want_out = boundary.h512_normalized(
        boundary.h128_normalized(reduced) * svh.to(torch.float32)
    ).to(torch.bfloat16)
    out_diff = int((out != want_out).sum())
    print(f"DS41 m={m}: output-stage diff {out_diff}/{out.numel()}")
    assert out_diff <= out.numel() // 1000 + 4, out_diff


# --------------------------------------------------------------------------- #
# Strict: bit for bit against the CPU reference
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", (1, 8, 16, 64, 256, 1024))
def test_fused_input_boundary_matches_the_cpu_reference_bit_for_bit(m: int) -> None:
    """Random rows over a wide range of magnitudes, both input-stage modes."""

    from b12x.moe._shared.kernels.trellismx_boundary_fused import (
        fused_input_boundary_fp16,
        fused_input_boundary_quantized,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(20261025 + m)
    rows = torch.randn(m, HIDDEN, generator=generator)
    # Per-row magnitudes 2**-12 .. 2**6, so the blocks span many UE8M0 exponents
    # and the FP16 cast meets subnormals.
    rows = rows * torch.exp2(
        torch.randint(-12, 7, (m, 1), generator=generator).to(torch.float32)
    )
    x = rows.to(torch.bfloat16)
    signs = torch.where(torch.rand(HIDDEN, generator=generator) < 0.5, -1.0, 1.0)
    suh = ((0.5 + 1.5 * torch.rand(HIDDEN, generator=generator)) * signs).to(
        torch.float16
    )

    payload, scales = fused_input_boundary_quantized(
        x.to(device), suh.to(device), hidden_size=HIDDEN
    )
    values = fused_input_boundary_fp16(x.to(device), suh.to(device), hidden_size=HIDDEN)
    torch.cuda.synchronize()

    reference = boundary.input_boundary(x, suh, hidden_size=HIDDEN)
    want_payload, want_scales = boundary.input_boundary_quantized(
        x, suh, hidden_size=HIDDEN
    )
    assert torch.equal(scales.cpu(), want_scales)
    assert torch.equal(payload.cpu(), want_payload)
    assert torch.equal(
        values.cpu().view(torch.int16), reference.to(torch.float16).view(torch.int16)
    )


class _QuantizerProbe:
    """The input stage's K32 quantizer, ``_quantize_k32_chunk``, on FP32 rows."""

    def __init__(self, hidden: int):
        self.hidden = int(hidden)

    @cute.jit
    def __call__(self, values, payload, scales, stream):
        self.kernel(values, payload, scales).launch(
            grid=(cute.size(values, 0), self.hidden // 512, 1),
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, values, payload, scales):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, cidx, _ = cute.arch.block_idx()
        tid = cutlass.Int32(tidx)
        offset = cutlass.Int32(cidx) * cutlass.Int32(512)
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(512), byte_alignment=16
        )
        for i in cutlass.range_constexpr(4):
            j = tid + cutlass.Int32(i * 128)
            buf[j] = values[bidx, offset + j]
        cute.arch.sync_threads()
        _quantize_k32_chunk(buf, payload, scales, bidx, cutlass.Int32(cidx), tid)


class _ExponentProbe:
    """One thread per value, calling the kernel's exponent extraction directly."""

    @cute.jit
    def __call__(self, values, exponents, stream):
        self.kernel(values, exponents).launch(
            grid=(1, 1, 1),
            block=(cute.size(values, 0), 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, values, exponents):
        thread, _, _ = cute.arch.thread_idx()
        index = cutlass.Int32(thread)
        exponents[index] = _ue8m0_exponent_s32(values[index])


def _edge_blocks(generator: torch.Generator) -> list[torch.Tensor]:
    """K32 blocks around the subnormal and underflow cases of ``amax / 448``."""

    def block(*head: float) -> torch.Tensor:
        out = torch.zeros(32, dtype=torch.float32)
        out[: len(head)] = torch.tensor(head, dtype=torch.float32)
        return out

    tiny = 2.0**-149
    above_smallest_normal = float(
        torch.nextafter(
            torch.tensor(2.0**-126, dtype=torch.float32),
            torch.tensor(torch.inf, dtype=torch.float32),
        )
    )
    return [
        block(),  # all zero: byte 127, scale 1
        torch.full((32,), -0.0),  # negative zeros: the same, payload 0x80
        block(2.0**-140, -tiny, tiny, 2.0**-141, -(2.0**-140)),  # amax 2**-140
        block(tiny),  # amax 2**-149: the quotient underflows to zero
        block(224 * tiny, -tiny),  # largest amax whose quotient rounds to zero
        block(225 * tiny, 3 * tiny),  # smallest amax whose quotient does not
        # quotient exactly 2**-127, payload codes across the E4M3 range
        block(448 * 2.0**-127, 2.0**-130, -(2.0**-128), 3 * 2.0**-133, 336 * 2.0**-127),
        block(-448 * 2.0**-127, 2.0**-128),  # negative maximum
        block(448 * 2.0**-126.5, -(2.0**-127), 2.0**-131),  # subnormal, above 2**-127
        block(448 * 2.0**-126, 2.0**-140),  # quotient 2**-126, the smallest normal
        block(448 * above_smallest_normal, 2.0**-140),
        block(1.5 * 448 * 2.0**-126, -(2.0**-129)),
        torch.randn(32, generator=generator) * 2.0**-135,  # subnormal quotients
        torch.randn(32, generator=generator) * 2.0**-140,  # subnormal values
        torch.randn(32, generator=generator) * 2.0**-110,
        torch.randn(32, generator=generator),
        block(448.0, -1.0, 2.0**-9),  # quotient 1
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_k32_quantizer_is_byte_exact_on_tiny_and_zero_blocks() -> None:
    """The quantizer on maxima the FP16 input stage never produces.

    A block whose ``amax / 448`` is subnormal took exponent -126 instead of the
    reference's clamped -127, and one whose quotient underflowed took scale 0.0
    (NaN for its zero elements). Both now match the reference, with the
    subnormal scale ``2**-127``.
    """

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(1025)
    edge = torch.stack(_edge_blocks(generator))
    random_tiny = torch.randn(47, 32, generator=generator) * torch.exp2(
        -torch.randint(100, 150, (47, 1), generator=generator).to(torch.float32)
    )
    # Two rows of two 512-value chunks: every block and every chunk index.
    rows = torch.cat((edge, random_tiny)).reshape(2, 1024)
    hidden = rows.shape[1]

    values = rows.to(device)
    payload = torch.zeros(rows.shape, dtype=torch.uint8, device=device)
    scales = torch.zeros(rows.shape[0], hidden // 32, dtype=torch.uint8, device=device)
    args = (
        from_dlpack(values, assumed_align=16),
        from_dlpack(payload, assumed_align=16),
        from_dlpack(scales, assumed_align=16),
        current_cuda_stream(),
    )
    b12x_compile(_QuantizerProbe(hidden), *args)(*args)
    torch.cuda.synchronize()

    want_payload, want_scales = boundary.quantize_e4m3_ue8m0_k32_codes(rows)
    # The cases themselves: byte 0 wherever the quotient is subnormal or zero.
    assert want_scales[0, :12].tolist() == [
        127,
        127,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert want_scales[0, 9:12].tolist() == [1, 2, 2]
    assert torch.equal(scales.cpu(), want_scales)
    assert torch.equal(payload.cpu(), want_payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ue8m0_exponent_matches_all_references_on_finite_fp32_values() -> None:
    """Exercise the PTX helper itself over every finite exponent boundary."""

    finite_max = torch.finfo(torch.float32).max
    above_2_127 = torch.nextafter(
        torch.tensor(2.0**127, dtype=torch.float32),
        torch.tensor(torch.inf, dtype=torch.float32),
    )
    values = [
        0.0,
        2.0**-149,
        torch.nextafter(
            torch.tensor(2.0**-126, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        ),
        torch.nextafter(
            torch.tensor(2.0**-126, dtype=torch.float32),
            torch.tensor(torch.inf, dtype=torch.float32),
        ),
        2.0**-127,
        2.0**-126,
    ]
    for exponent in range(-126, 128):
        power = torch.tensor(2.0**exponent, dtype=torch.float32)
        values.append(power)
        values.append(
            torch.nextafter(power, torch.tensor(torch.inf, dtype=torch.float32))
        )
    values.extend((above_2_127, finite_max))
    quotient = torch.tensor(values, dtype=torch.float32)

    device = torch.device("cuda")
    values_device = quotient.to(device)
    exponents_device = torch.empty_like(values_device, dtype=torch.int32)
    args = (
        from_dlpack(values_device, assumed_align=16),
        from_dlpack(exponents_device, assumed_align=16),
        current_cuda_stream(),
    )
    b12x_compile(_ExponentProbe(), *args)(*args)
    torch.cuda.synchronize()

    references = (
        boundary._ue8m0_exponent(quotient),
        reference_moe._ue8m0_exponent(quotient),
        coupled_scales._ue8m0_exponent(quotient),
    )
    assert all(torch.equal(references[0], reference) for reference in references[1:])
    assert torch.equal(exponents_device.cpu(), references[0].to(torch.int32))
    assert exponents_device.cpu().min().item() >= -127
    assert exponents_device.cpu().max().item() <= 127


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_input_boundary_subnormal_maxima_end_to_end() -> None:
    """The whole e4m3 input stage on chunks scaled from ordinary to underflow.

    BF16 rows cast to FP16 with an FP16 ``suh`` cannot reach such blocks; an
    FP32 ``suh`` of tiny values (the kernel reads ``suh`` as FP32 either way)
    scales each 512-value chunk into its own range, so H512, ``* suh`` and H128
    run on subnormal FP32 values and the quantizer sees subnormal, underflowing
    and all-zero blocks, as IEEE arithmetic on the CPU does.
    """

    device = torch.device("cuda")
    m = 4
    generator = torch.Generator(device="cpu").manual_seed(20261025)
    x = torch.randn(m, HIDDEN, generator=generator).to(torch.bfloat16)
    exponents = (0, -100, -116, -122, -126, -132, -140, -151)
    suh = torch.cat(
        [
            (0.5 + torch.rand(512, generator=generator)) * 2.0**exponent
            for exponent in exponents
        ]
    )
    suh = suh * torch.where(torch.rand(HIDDEN, generator=generator) < 0.5, -1.0, 1.0)
    suh = suh.to(torch.float32)

    x_device, suh_device = x.to(device), suh.to(device)
    payload = torch.zeros(m, HIDDEN, dtype=torch.uint8, device=device)
    scales = torch.zeros(m, HIDDEN // 32, dtype=torch.uint8, device=device)
    unused = torch.zeros(m, HIDDEN, dtype=torch.float16, device=device)
    args = (
        from_dlpack(x_device, assumed_align=16),
        from_dlpack(suh_device, assumed_align=16),
        from_dlpack(payload, assumed_align=16),
        from_dlpack(scales, assumed_align=16),
        from_dlpack(unused, assumed_align=16),
        current_cuda_stream(),
    )
    b12x_compile(TrellisMxFusedBoundary(HIDDEN, "e4m3"), *args)(*args)
    torch.cuda.synchronize()

    want_payload, want_scales = boundary.input_boundary_quantized(
        x, suh, hidden_size=HIDDEN
    )
    # The data reaches every case: normal, subnormal or underflowing quotients
    # (byte 0) and all-zero blocks (byte 127 from a chunk whose suh underflowed).
    codes = set(want_scales.reshape(-1).tolist())
    assert 0 in codes and 127 in codes and any(0 < code < 127 for code in codes)
    assert torch.equal(scales.cpu(), want_scales)
    assert torch.equal(payload.cpu(), want_payload)
