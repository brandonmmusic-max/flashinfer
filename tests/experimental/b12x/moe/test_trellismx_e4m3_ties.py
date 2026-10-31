"""Does the Hadamard normalization divide, or multiply by a rounded reciprocal?

The TrellisMX declared orders end each Hadamard with a correctly-rounded
division: ``...-h512-fp32-suh-fp32-h128-fp32-e4m3-...``, and the release pins
the divisors as the float32 encodings of ``sqrt(512)`` and ``sqrt(128)``
(``_P8_SQRT512_F32`` / ``_P8_SQRT128_F32`` in the P8 kernel module) with the
explicit note that multiplying by a reciprocal instead "can move an E4M3 tie
by one code".  b12x's QSRT form multiplies by ``0.088388347648``.

The two arithmetics differ by at most one float32 ULP, so a random corpus almost
never sees a difference (measured: 8 discriminating 128-vectors in 86,535
tries).  This file builds a corpus that *does* discriminate, deterministically:

* the normalized output must land exactly on an E4M3 midpoint, where
  round-to-nearest-even is decided by the tie rule;
* an H128 input of ``u * e_0`` has the unnormalized transform output ``u`` in
  every coordinate (the first Sylvester column is all ones and the additions are
  exact), so the only rounding left is the normalization itself;
* choosing ``u = fl32(m * sqrt(128))`` for each E4M3 midpoint ``m`` makes the
  correctly-rounded quotient exactly ``m`` while the reciprocal multiply lands
  just below it.

The corpus is then run through the kernel's own H128 (``_w4a8_had128_quad`` and
``_w4a8_had128_quad_exact`` via ``_w4a8_had128_quad_dispatch``) in a one-warp
probe kernel, so the statement about which arithmetic the kernel does is
behavioural, not read off the source.
"""

from __future__ import annotations

import pytest
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from b12x.moe._shared.kernels import trellismx_boundary as boundary
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad_dispatch

# The release's divisors and the QSRT reciprocal, as float32.
SQRT128 = torch.tensor(11.313708305358887, dtype=torch.float32)
SQRT512 = torch.tensor(22.627416610717773, dtype=torch.float32)
RECIP128 = torch.tensor(0.088388347648, dtype=torch.float32)
RECIP512 = torch.tensor(0.04419417382415922, dtype=torch.float32)

_DIVISOR = {128: SQRT128, 512: SQRT512}
_RECIPROCAL = {128: RECIP128, 512: RECIP512}


def _butterfly(value: torch.Tensor, size: int) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard over the last axis (plain butterfly)."""

    work = value.reshape(-1, size).clone()
    stride = 1
    while stride < size:
        work = work.view(-1, size // (2 * stride), 2, stride)
        left = work[:, :, 0, :].clone()
        right = work[:, :, 1, :].clone()
        work[:, :, 0, :] = left + right
        work[:, :, 1, :] = left - right
        work = work.view(-1, size)
        stride *= 2
    return work


def _normalized_div(value: torch.Tensor, size: int) -> torch.Tensor:
    return _butterfly(value, size) / _DIVISOR[size]


def _normalized_mul(value: torch.Tensor, size: int) -> torch.Tensor:
    return _butterfly(value, size) * _RECIPROCAL[size]


def _e4m3(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float8_e4m3fn).view(torch.uint8)


def _midpoints() -> list[float]:
    """Every E4M3 midpoint a normalized output can plausibly land on."""

    return [
        (1.0 + (2 * k + 1) / 16.0) * 2.0**exponent
        for exponent in range(-8, 5)
        for k in range(8)
    ]


def _corpus(size: int) -> torch.Tensor:
    """Inputs whose normalized output sits exactly on an E4M3 midpoint.

    Each row is ``u * e_0``; only rows where the two arithmetics disagree on the
    E4M3 code are kept, so the corpus is a discriminator by construction.
    """

    divisor = _DIVISOR[size]
    kept = []
    for midpoint in _midpoints():
        unnormalized = torch.tensor(
            midpoint * float(divisor.double()), dtype=torch.float32
        )
        if float(unnormalized) == 0.0:
            continue
        row = torch.zeros(size, dtype=torch.float32)
        row[0] = unnormalized
        # The unnormalized transform must be exactly `unnormalized` everywhere.
        if not torch.equal(_butterfly(row, size), torch.full((1, size), unnormalized)):
            continue
        if not torch.equal(
            _e4m3(_normalized_div(row, size)), _e4m3(_normalized_mul(row, size))
        ):
            kept.append(row)
    return torch.stack(kept) if kept else torch.zeros(0, size)


@pytest.mark.parametrize("size", [128, 512])
def test_corpus_discriminates_the_two_arithmetics(size: int) -> None:
    corpus = _corpus(size)
    assert corpus.shape[0] >= 8, (size, corpus.shape)
    div_codes = _e4m3(_normalized_div(corpus, size))
    mul_codes = _e4m3(_normalized_mul(corpus, size))
    differing = int((div_codes != mul_codes).any(dim=1).sum())
    assert differing == corpus.shape[0], (size, differing)


@pytest.mark.parametrize("size", [128, 512])
def test_reference_normalization_is_the_division(size: int) -> None:
    corpus = _corpus(size)
    op = (
        boundary.h128_normalized(corpus)
        if size == 128
        else boundary.h512_normalized(corpus)
    )
    assert torch.equal(op, _normalized_div(corpus, size))
    # And it is *not* the reciprocal form: the corpus is what separates them.
    assert not torch.equal(op, _normalized_mul(corpus, size))
    assert not torch.equal(_e4m3(op), _e4m3(_normalized_mul(corpus, size)))


def test_e4m3_tie_survives_a_power_of_two_block_scale() -> None:
    """Justifies testing the raw E4M3 cast instead of the K32 quantizer.

    The UE8M0 block scale is a power of two, and dividing by a power of two is
    exact, so the block scale cannot move a value across an E4M3 rounding
    boundary.  The raw cast is therefore the same discriminator.
    """

    corpus = _corpus(128)
    div = _normalized_div(corpus, 128)
    mul = _normalized_mul(corpus, 128)
    for shift in (-4, -1, 0, 3):
        scale = 2.0**shift
        assert torch.equal(_e4m3(div / scale), _e4m3(mul / scale)) == torch.equal(
            _e4m3(div), _e4m3(mul)
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_h512_quad_order_is_bitwise_the_plain_butterfly(seed: int) -> None:
    """The release's H512 is H128-per-quarter then an H4 mix with one divide.

    ``_p8_h512_mix_reference_order`` finishes "exactly as CPU butterfly strides
    128 then 256"; this pins that the two are bitwise identical, which is what
    lets the boundary op use the plain butterfly.
    """

    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(16, 512, generator=generator)
    quarters = _butterfly(value, 128).reshape(-1, 4, 128)
    x0, x1, x2, x3 = quarters[:, 0], quarters[:, 1], quarters[:, 2], quarters[:, 3]
    r0, r1, r2, r3 = x0 + x1, x0 - x1, x2 + x3, x2 - x3
    explicit = torch.stack(
        [
            (r0 + r2) / SQRT512,
            (r1 + r3) / SQRT512,
            (r0 - r2) / SQRT512,
            (r1 - r3) / SQRT512,
        ],
        dim=1,
    ).reshape(-1, 512)
    assert torch.equal(explicit, boundary.h512_normalized(value))
    assert torch.equal(explicit, _normalized_div(value, 512))


def test_kernel_entry_points_and_default_are_pinned() -> None:
    """Both H128 entry points exist, the dispatcher selects them, and the
    kernel path keeps the reciprocal multiply by default.

    What each entry point computes is proven behaviourally by the GPU probe
    below (``test_kernel_h128_arithmetic_on_the_corpus``); this test only pins
    the API surface and the default that the probe cannot reach without a GPU.
    """

    from b12x.moe._shared.kernels import w4a8_trellis_decode as decode

    assert hasattr(decode, "_w4a8_had128_quad_exact"), (
        "the div.rn variant is missing; the kernel path can only multiply"
    )
    assert hasattr(decode, "_w4a8_had128_quad_dispatch")
    # The corpus below multiplies by the float32 nearest 1/sqrt(128) and
    # divides by the float32 sqrt(128).
    assert float(RECIP128) == float(torch.tensor(1.0 / 128.0**0.5, dtype=torch.float32))
    assert float(RECIP512) == float(torch.tensor(1.0 / 512.0**0.5, dtype=torch.float32))

    import inspect

    from b12x.moe._shared.kernels.dynamic.kernel import MoEDynamicKernelBackend

    parameters = inspect.signature(MoEDynamicKernelBackend.__init__).parameters
    assert "trellis_exact_hadamard" in parameters, tuple(parameters)
    assert parameters["trellis_exact_hadamard"].default is False, (
        "the QSRT default must stay the reciprocal multiply so existing bytes "
        "are unchanged; TrellisMX opts in with trellis_exact_hadamard=True"
    )


# --------------------------------------------------------------------------- #
# behavioural arm: run the kernel's own H128 on the corpus
# --------------------------------------------------------------------------- #
class _H128Probe:
    """One-warp probe: ``y = H128_dispatch(exact)(x)`` over 128-wide rows."""

    threads_per_cta = 32

    def __init__(self, exact: bool):
        # A compile-time attribute, so the dispatch specializes and the two
        # modes get distinct cache keys without a constexpr call argument.
        self.exact = bool(exact)

    @cute.jit
    def __call__(self, x_ptr, y_ptr, rows, stream: cuda.CUstream):
        x = cute.make_tensor(x_ptr, cute.make_layout((Int32(rows) * 128,), stride=(1,)))
        y = cute.make_tensor(y_ptr, cute.make_layout((Int32(rows) * 128,), stride=(1,)))
        self.kernel(x, y, rows).launch(
            grid=(Int32(rows), 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, x, y, rows):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = Int32(tid)
        row = Int32(bid)
        if row < rows:
            base = row * Int32(128) + lane * Int32(4)
            h0, h1, h2, h3 = _w4a8_had128_quad_dispatch(
                self.exact,
                x[base],
                x[base + Int32(1)],
                x[base + Int32(2)],
                x[base + Int32(3)],
                lane,
            )
            y[base] = h0
            y[base + Int32(1)] = h1
            y[base + Int32(2)] = h2
            y[base + Int32(3)] = h3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kernel_h128_arithmetic_on_the_corpus() -> None:
    """The kernel's H128, both modes, against the corpus — bitwise.

    ``exact=False`` must reproduce the reciprocal multiply and ``exact=True`` the
    correctly-rounded division; on this corpus those two also give different
    E4M3 codes, so the mode the kernel runs in is observable in a code.
    """

    from cutlass.cute.runtime import make_ptr

    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x.moe.fused_moe._impl import current_cuda_stream

    corpus = _corpus(128)
    rows = corpus.shape[0]
    device = torch.device("cuda")
    x = corpus.to(device).contiguous()
    y = torch.empty_like(x)

    compiled = {}
    for exact in (False, True):
        compiled[exact] = b12x_compile(
            _H128Probe(exact),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            rows,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_fields(
                "w2.trellismx.h128_probe", 1, ("rows", rows), ("exact", int(exact))
            ),
        )
        x_ptr = make_ptr(
            cutlass.Float32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        y_ptr = make_ptr(
            cutlass.Float32, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        compiled[exact](x_ptr, y_ptr, rows, current_cuda_stream())
        torch.cuda.synchronize()
        expected = (
            _normalized_div(corpus, 128) if exact else _normalized_mul(corpus, 128)
        )
        assert torch.equal(y.cpu(), expected), (
            f"exact={exact} kernel H128 disagrees with the "
            f"{'division' if exact else 'reciprocal'} reference"
        )
        # Record which E4M3 code the kernel produces, which is the observable
        # difference between the two modes.
        codes = _e4m3(y.cpu())
        other = _e4m3(
            _normalized_mul(corpus, 128) if exact else _normalized_div(corpus, 128)
        )
        assert not torch.equal(codes, other), (
            f"exact={exact} corpus stopped discriminating"
        )
