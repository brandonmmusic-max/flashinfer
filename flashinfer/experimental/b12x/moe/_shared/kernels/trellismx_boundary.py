"""TrellisMX per-token boundary ops for the W4A8 trellis MoE path.

TrellisMX = b12x's existing W4A8 trellis (QSRT) path plus a hidden-axis
boundary, ``silu-cap10`` and the procedural-NPK codebook.  The per-expert
intermediate-axis transform is *already* the same in both formats, so this
module implements only the parts that are identical for every expert and
therefore hoistable out of the routed kernel:

    input   : bf16 -> cvt.rn fp16 -> H512 -> * gate_up_suh -> H128 -> e4m3/ue8m0/k32
    output  : H128(FP16(MMA)) -> * down_svh, then the router-order FP32 top-k
              reduction and the outer H512.

Every op here is pure torch, has static shapes, performs no host synchronization
and writes no inputs, so it is CUDA-graph capturable as-is
(``tests/experimental/b12x/moe/test_trellismx_boundary.py`` captures and replays
these ops).

FP32 continuity is part of the format, not an implementation detail: the
declared orders are
``bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32`` and
``silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32``, so
nothing between the H512 and the E4M3 quantizer may round to bf16 or fp16.
``KERNEL_CONTRACT`` below states exactly what the routed kernel must (and must
not) do for that to hold; the dynamic kernel meets it through its
``trellis_pre_quantized_a``, ``trellis_exact_hadamard`` and
``trellis_fp32_boundary`` options.

Hadamard normalization is one correctly-rounded ``div.rn.f32`` by the float32
encoding of ``sqrt(n)`` -- the release's comment records that a reciprocal
multiply "can move an E4M3 tie by one code" (``route_hoist_dynamic.py:178-181``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "KERNEL_CONTRACT",
    "TrellisMxBoundarySpec",
    "h512_normalized",
    "h128_normalized",
    "input_boundary_pre_h128",
    "input_boundary",
    "input_boundary_quantized",
    "output_boundary",
    "coupled_topk_sum",
    "silu_cap10",
    "K32_PERMUTATION",
    "quantize_e4m3_ue8m0_k32",
    "quantize_e4m3_ue8m0_k32_codes",
]

# Exact float32 encodings used by the release's reference-order helpers.
SQRT512_F32 = 22.627416610717773  # 0x41b504f3
SQRT128_F32 = 11.313708305358887  # 0x413504f3
E4M3_MAX = 448.0

# The native trellis B lane order needs this within-K32 activation
# permutation; identical to
# b12x/moe/_shared/kernels/w4a8_trellis_decode.py::_W4A8_TRELLIS_A_K32_PERM and
# pinned there by tests/moe/test_w4a8_fragment_probe.py::_NATIVE_MMA_K32_PERM.
# It stays inside one K32 block, so the UE8M0 group is unchanged.
K32_PERMUTATION = (
    0,
    1,
    8,
    9,
    4,
    5,
    12,
    13,
    2,
    3,
    10,
    11,
    6,
    7,
    14,
    15,
    20,
    21,
    28,
    29,
    16,
    17,
    24,
    25,
    22,
    23,
    30,
    31,
    18,
    19,
    26,
    27,
)

_K32_PERM_CACHE: dict[torch.device, torch.Tensor] = {}


def _k32_permutation(device: torch.device) -> torch.Tensor:
    """Cached device-side K32 permutation index.

    Built once per device so the op allocates and copies nothing on the call
    path: a host-to-device copy inside a captured region is an error.  As with
    every b12x kernel, the first call for a device is made eagerly.
    """

    cached = _K32_PERM_CACHE.get(device)
    if cached is None:
        cached = torch.tensor(K32_PERMUTATION, dtype=torch.int64, device=device)
        _K32_PERM_CACHE[device] = cached
    return cached


KERNEL_CONTRACT = """\
What the routed W4A8 trellis kernel must do to serve TrellisMX exactly.

The per-token input stage runs OUTSIDE the kernel (input_boundary_pre_h128 or
input_boundary_quantized) because it is identical for every expert.  For the
declared FP32 order to survive, the kernel must satisfy all three of:

  (1) A operands are consumed at the precision the boundary produced.  The
      phase-1 path loads bf16 and re-casts
      `cutlass.Float16(a_input[...].to(f32)).to(f32)`; feeding it the boundary's
      fp32 output adds a rounding the order does not declare.  Either read the
      fp32 buffer directly, or consume `input_boundary_quantized`'s E4M3
      payload + UE8M0 bytes and skip the activation producer entirely
      (preferred: no new rounding, and the K32 permutation and sfb repack are
      already produced host-side).

  (2) No Hadamard is re-applied to A.  The boundary already ran the H128.  The
      kernel's `_w4a8_had128_quad` boundary (dynamic/kernel.py:5903-6055) is
      correct for QSRT, which supplies an identity-boundary input, and must be
      skipped when the boundary op supplied the activation.

  (3) The kernel must not re-normalize with a reciprocal multiply.  Where a
      Hadamard remains in-kernel, its terminal normalization must be
      `div.rn.f32` by SQRT128_F32/SQRT512_F32 rather than
      `_w4a8_had128_quad`'s `* 0.088388347648`.

Nothing else changes: the intermediate-axis boundary, the per-expert
rotations/scale layout, the E4M3 operand form, the UE8M0/K32 weight scales and
the sfb repack are already TrellisMX-compatible.  `silu-cap10` is already
reachable in-kernel as `activation="silu", swiglu_limit=10.0,
swiglu_clamp_after_silu=False` (dynamic/kernel.py:1192-1223), which is bitwise
the release's `_coupled_activation`.

The output stage is per-route and post-reduction, so it needs no kernel change
beyond emitting the per-route FC2 result at FP16 (the declared store) instead of
BF16; output_boundary + coupled_topk_sum then reproduce the release exactly.
"""


@dataclass(frozen=True)
class TrellisMxBoundarySpec:
    """Shape contract for one rank slice of one TrellisMX layer."""

    hidden_size: int
    intermediate_size: int
    bits: int = 4

    def __post_init__(self) -> None:
        if self.hidden_size % 512:
            raise ValueError(
                f"TrellisMX hidden_size must be a multiple of 512 (H512 outer "
                f"stage), got {self.hidden_size}"
            )
        if self.intermediate_size % 128:
            raise ValueError(
                f"TrellisMX local intermediate must be a multiple of 128 (H128 "
                f"activation axis), got {self.intermediate_size}"
            )
        if self.bits not in (2, 3, 4, 5):
            raise ValueError(f"TrellisMX trellis rate must be K2..K5, got K{self.bits}")


def _hadamard(value: torch.Tensor, size: int, divisor: float) -> torch.Tensor:
    """Normalized Sylvester Hadamard over the last axis with one terminal divide."""

    width = value.shape[-1]
    if width % size:
        raise ValueError(f"Hadamard width {width} is not a multiple of {size}")
    work = value.to(torch.float32).reshape(-1, size).clone()
    stride = 1
    while stride < size:
        work = work.view(-1, size // (2 * stride), 2, stride)
        left = work[:, :, 0, :].clone()
        right = work[:, :, 1, :].clone()
        work[:, :, 0, :] = left + right
        work[:, :, 1, :] = left - right
        work = work.view(-1, size)
        stride *= 2
    return (work / divisor).to(torch.float32).view_as(value)


def h512_normalized(value: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester H512 on the last axis, natural order, FP32."""

    return _hadamard(value, 512, SQRT512_F32)


def h128_normalized(value: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester H128 on the last axis, natural order, FP32."""

    return _hadamard(value, 128, SQRT128_F32)


def input_boundary_pre_h128(
    x_bf16: torch.Tensor, gate_up_suh: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """``bf16 -> cvt.rn fp16 -> H512 -> * gate_up_suh``, FP32 out.

    This is the exact hoistable prefix of the declared input order: everything
    here is per token and identical for every expert.  It is exact in FP32 given
    an FP32 input, so the only thing preventing it from being used as-is today
    is the kernel's re-cast (see ``KERNEL_CONTRACT`` point 1).

    ``x_bf16.to(torch.float16)`` is ``cvt.rn.f16`` from the bf16 value, which is
    lossless for every bf16 magnitude inside fp16's normal range
    (``2**-14 <= |x| <= 65504``); outside it fp16 subnormalizes or overflows, and
    that is what the release's ``cvt.rn`` does too.
    """

    if x_bf16.dtype != torch.bfloat16:
        raise ValueError(f"TrellisMX input must be bfloat16, got {x_bf16.dtype}")
    if x_bf16.shape[-1] != hidden_size:
        raise ValueError(
            f"TrellisMX input width {x_bf16.shape[-1]} != hidden_size {hidden_size}"
        )
    if gate_up_suh.numel() != hidden_size:
        raise ValueError(
            f"gate_up_suh has {gate_up_suh.numel()} entries, expected {hidden_size}"
        )
    source = x_bf16.to(torch.float16).to(torch.float32)
    source = h512_normalized(source)
    return source * gate_up_suh.to(torch.float32)


def input_boundary(
    x_bf16: torch.Tensor, gate_up_suh: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """The full per-token input stage, FP32 out (pre-quantization)."""

    return h128_normalized(
        input_boundary_pre_h128(x_bf16, gate_up_suh, hidden_size=hidden_size)
    )


def quantize_e4m3_ue8m0_k32(value: torch.Tensor) -> torch.Tensor:
    """UE8M0/K32 block quantize with a finite E4M3 payload, then reconstruct.

    ``scale = 2**ceil(log2(amax/448))`` clamped to the UE8M0 range
    ``[-127, 127]``, with the exponent taken exactly from the finite FP32 bits
    of the quotient. Zero blocks use scale 1 and the payload is the
    round-to-nearest-even finite E4M3 cast of ``value/scale``. Non-finite
    quotients are outside the domain. The testing and P8 CPU references
    implement the same exact-bits rule independently.
    """

    if value.shape[-1] % 32:
        raise ValueError("MXFP8 quantization needs a width divisible by 32")
    blocks = value.to(torch.float32).reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(maximum > 0, maximum / E4M3_MAX, torch.ones_like(maximum))
    exponent = _ue8m0_exponent(safe)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    payload = (blocks / scale).to(torch.float8_e4m3fn)
    return (payload.to(torch.float32) * scale).reshape_as(value)


def _ue8m0_exponent(quotient: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(quotient)
    exact = torch.where(mantissa == 0.5, exponent - 1, exponent)
    exact = torch.where(quotient > 0, exact, torch.full_like(exact, -127.0))
    return exact.clamp(-127.0, 127.0)


def input_boundary_quantized(
    x_bf16: torch.Tensor, gate_up_suh: torch.Tensor, *, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input stage -> the kernel's native A operands.

    Returns ``(payload, scales)``: ``payload`` is uint8 ``[tokens, hidden]`` in
    the native K32-permuted E4M3 order (``K32_PERMUTATION``), ``scales`` is the
    uint8 UE8M0 byte per K32 block, ``[tokens, hidden // 32]``.  Both are laid
    out so the existing sfb repack and MMA operand path can consume them
    unchanged.
    """

    return quantize_e4m3_ue8m0_k32_codes(
        input_boundary(x_bf16, gate_up_suh, hidden_size=hidden_size)
    )


def quantize_e4m3_ue8m0_k32_codes(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The UE8M0/K32 quantizer of ``input_boundary_quantized`` on FP32 values.

    Same arithmetic as ``quantize_e4m3_ue8m0_k32``, returned as the kernel's
    operand bytes: the K32-permuted E4M3 payload and one UE8M0 byte per block.
    A block whose ``amax / 448`` is subnormal or underflows to zero takes the
    clamped exponent -127 (byte 0, scale ``2**-127``). For every finite
    quotient, including one a few ulps above a power of two and the largest
    finite FP32 value, the exponent is exact and clamped to the UE8M0 range
    ``[-127, 127]``. Non-finite inputs are outside this domain.
    """

    if value.shape[-1] % 32:
        raise ValueError("MXFP8 quantization needs a width divisible by 32")
    value = value.to(torch.float32)
    blocks = value.reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(maximum > 0, maximum / E4M3_MAX, torch.ones_like(maximum))
    exponent = _ue8m0_exponent(safe)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    permuted = blocks.index_select(-1, _k32_permutation(value.device))
    payload = (permuted / scale).to(torch.float8_e4m3fn).view(torch.uint8)
    payload = payload.reshape(*value.shape[:-1], -1)
    scale_code = (exponent.squeeze(-1).to(torch.float32) + 127.0).to(torch.uint8)
    return payload, scale_code


def output_boundary(
    route: torch.Tensor, down_svh: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """Per-route ``H128(FP16(MMA)) * down_svh``; the outer H512 is separate."""

    if route.shape[-1] != hidden_size:
        raise ValueError(
            f"down output width {route.shape[-1]} != hidden_size {hidden_size}"
        )
    if down_svh.numel() != hidden_size:
        raise ValueError(
            f"down_svh has {down_svh.numel()} entries, expected {hidden_size}"
        )
    return h128_normalized(route.to(torch.float32)) * down_svh.to(torch.float32)


def coupled_topk_sum(
    routes: torch.Tensor, weights: torch.Tensor, *, hidden_size: int
) -> torch.Tensor:
    """Router-order FP32 weighted sum of per-route values, then normalized H512.

    ``routes`` is ``[tokens, topk, hidden]``.  Linearity lets the outer H512 be
    hoisted past the reduction, which is exactly what the release's coupled
    reducer does (``p8_coupled_topk.py:70-73``); the addition order over routes
    is the router order and is part of the result.
    """

    if routes.ndim != 3 or routes.shape[-1] != hidden_size:
        raise ValueError("routes must be [tokens, topk, hidden]")
    if weights.shape != routes.shape[:2]:
        raise ValueError("weights must be [tokens, topk]")
    total = torch.zeros(
        routes.shape[0], hidden_size, dtype=torch.float32, device=routes.device
    )
    for k in range(routes.shape[1]):
        total = total + weights[:, k].to(torch.float32)[:, None] * routes[:, k].to(
            torch.float32
        )
    return h512_normalized(total).to(torch.bfloat16)


def silu_cap10(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``min(gate,10) * sigmoid(min(gate,10)) * clamp(up,-10,10)``, no post-clamp."""

    gate_capped = torch.clamp(gate.to(torch.float32), max=10.0)
    up_capped = torch.clamp(up.to(torch.float32), min=-10.0, max=10.0)
    return gate_capped * torch.sigmoid(gate_capped) * up_capped
