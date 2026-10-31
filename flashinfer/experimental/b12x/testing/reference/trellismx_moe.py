"""Exact torch reference for one TrellisMX MoE layer on one TP rank slice.

TrellisMX is the GLM-5.3-Flash routed-expert format: procedural-NPK trellis
codewords over an E4M3 alphabet with UE8M0/K32 block scales, encoded in the
``coupled-h512-h128-suh-svh-v1`` basis.  This module is the CPU/GPU reference
every other TrellisMX worker tests against.  It computes, in FP32, exactly the
orders the release declares:

    quantized_input_order = bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32
    quantized_down_order  = silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32

and the hidden-axis output stage ``H128 -> * down_svh`` per route, the router
order top-k FP32 reduction, and the final normalized ``H512``.

Provenance of every step is cited in the module.  The
release ships the same topology as executable code (``p8_coupled_scales.py::
coupled_reference``); this module is an independent implementation of it, not a
copy, and it adds the sidecar weight decode, the E4M3/UE8M0/K32 activation
quantizer, the router reduction and the top-k weighting.

Declared storage casts this reference reproduces (they are part of the format,
not emulation slack):

* FC1 output  -> FP16 before the intermediate boundary (``_p8_pack_f32x2_to_half2``).
* FC2 output  -> FP16 before the output boundary.
* final output -> BF16.

Rounding conventions: every Hadamard normalizes with one correctly-rounded
``div.rn.f32`` by ``sqrt(n)`` (``_P8_SQRT512_F32`` / ``_P8_SQRT128_F32``), which
is what the release's "reference order" helpers do and what its comment calls
out as able to move an E4M3 tie by one code versus a reciprocal multiply.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from b12x.moe._shared.trellis_codebooks import PROCEDURAL_NPK

__all__ = [
    "TRELLISMX_INPUT_ORDER",
    "TRELLISMX_DOWN_ORDER",
    "trellismx_moe_reference",
    "trellismx_input_boundary",
    "trellismx_intermediate_boundary",
    "trellismx_output_boundary",
    "trellismx_coupled_topk_sum",
    "silu_cap10",
    "quantize_e4m3_ue8m0_k32",
    "decode_trellis_tile",
    "decode_trellis_matrix",
    "npk_decode_states",
    "rank_local_coupled_signs",
    "hadamard_normalized",
    "validate_sidecar_tensors",
]

TRELLISMX_INPUT_ORDER = "bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
TRELLISMX_DOWN_ORDER = (
    "silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32"
)

# Exact float32 encodings of sqrt(512) and sqrt(128); route_hoist_dynamic.py:182-183.
_P8_SQRT512_F32 = 22.627416610717773  # 0x41b504f3
_P8_SQRT128_F32 = 11.313708305358887  # 0x413504f3

# procedural-npk constants; the same three words appear in
# b12x/_lib/intrinsics/constants.py and in the release's decode asm.
_NPK_MULTIPLIER = 0xCBAC1FED
_NPK_MASK = 0x8FFF8FFF
_NPK_XOR = 0x3B603B60

_E4M3_MAX = 448.0

# Ring geometry of one t256 tile: 256 codes of `bits` bits = 8*bits uint32 words.
# trellis_ring.py:1-10 fixes the window for weight index w of lane l.
_TILE_SIDE = 16
_TILE_WEIGHTS = _TILE_SIDE * _TILE_SIDE


# --------------------------------------------------------------------------- #
# procedural NPK codec
# --------------------------------------------------------------------------- #
def _npk_states_to_e4m3_bytes(states: torch.Tensor) -> torch.Tensor:
    """Private implementation of the procedural-NPK law, uint8 E4M3FN bytes.

    ``w = (s * 0xCBAC1FED) mod 2**32; x = (w & 0x8FFF8FFF) ^ 0x3B603B60;
    h = fp16(x & 0xffff) + fp16(x >> 16)`` (add.rn.f16x2), then the alpha-2
    compander ``h = h + h``, then ``cvt.rn.satfinite.e4m3x2.f16x2``.

    This exists only so the reference has no hard dependency on the kernel
    module; when ``b12x.moe._shared.kernels.w4a8_npk_decode.npk_decode_states_reference``
    is importable it is used instead and must agree bit for bit.
    """

    s = states.to(torch.int64) & 0xFFFF
    w = (s * _NPK_MULTIPLIER) & 0xFFFFFFFF
    x = (w & _NPK_MASK) ^ _NPK_XOR
    lo = (x & 0xFFFF).to(torch.int32).to(torch.uint16).view(torch.float16)
    hi = ((x >> 16) & 0xFFFF).to(torch.int32).to(torch.uint16).view(torch.float16)
    h = (lo + hi).to(torch.float16)  # add.rn.f16x2
    h = (h + h).to(torch.float16)  # alpha-2 compander, add.rn.f16x2
    return h.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)


def npk_decode_states(states: torch.Tensor) -> torch.Tensor:
    """Decode 16-bit procedural-NPK states to E4M3FN bytes.

    Delegates to ``w4a8_npk_decode.npk_decode_states_reference`` when it is
    importable and cross-checks the two on the caller's states.
    """

    try:  # pragma: no cover - the kernel module may be unavailable
        from b12x.moe._shared.kernels.w4a8_npk_decode import (
            npk_decode_states_reference,
        )
    except Exception:
        return _npk_states_to_e4m3_bytes(states)
    # That reference is a CPU reference by contract (its ``_state_bits`` raises
    # on any non-CPU device), so decode on the host and move the bytes back.
    # The cross-check is between the two CPU results, which keeps it
    # independent of the caller's device.
    host = states.detach().cpu()
    reference = npk_decode_states_reference(host)
    if not torch.equal(reference.view(torch.uint8), _npk_states_to_e4m3_bytes(host)):
        raise AssertionError(
            "npk_decode_states_reference disagrees with the procedural-NPK law"
        )
    return reference.to(states.device)


def _e4m3_byte_to_value() -> torch.Tensor:
    """256-entry E4M3FN byte -> float32 table (finite values only)."""

    raw = torch.arange(256, dtype=torch.int32)
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    sign = (raw >> 7) & 1
    subnormal = mantissa.to(torch.float64) * 2.0**-9
    normal = (1.0 + mantissa.to(torch.float64) / 8.0) * torch.pow(
        2.0, (exponent - 7).to(torch.float64)
    )
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where(
        (exponent == 15) & (mantissa == 7), torch.zeros_like(value), value
    )
    value = torch.where(sign != 0, -value, value)
    return value.to(torch.float32)


_E4M3_LUT = _e4m3_byte_to_value()


# --------------------------------------------------------------------------- #
# sidecar payload -> dense weight matrix
# --------------------------------------------------------------------------- #
def _tile_position_map() -> tuple[torch.Tensor, torch.Tensor]:
    """State index (lane*8 + j) -> (row, col) inside one K16xN16 tile.

    ``row`` is the N (output) index and ``col`` the K (input) index.  This is
    the MMA B-fragment order the decoder produces; the identical table is used
    by ``b12x/testing/reference/trellis_moe.py::_tile_position_map``.
    """

    rows = torch.empty(_TILE_WEIGHTS, dtype=torch.int64)
    cols = torch.empty(_TILE_WEIGHTS, dtype=torch.int64)
    for lane in range(32):
        r = lane % 4
        n0 = 2 * (lane // 8) + ((lane >> 2) & 1)
        ks = (2 * r, 2 * r + 1, 2 * r + 8, 2 * r + 9)
        for j in range(8):
            rows[lane * 8 + j] = n0 + (8 if j >= 4 else 0)
            cols[lane * 8 + j] = ks[j % 4]
    return rows, cols


_TILE_ROWS, _TILE_COLS = _tile_position_map()


def _unpack_native_tiles(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of ``trellis_moe._pack_native_tiles``: payload -> [N, 256] symbols.

    ``_pack_native_tiles`` writes, per 16-symbol span, ``bits`` 16-bit words
    holding symbols ``4g..4g+3`` MSB-first, then swaps each consecutive pair of
    words.  Undoing the pair swap is the "int16 pair swap" step; the bit order
    inside each word is MSB-first per symbol; unpacking inverts packing
    exactly.
    """

    words = packed.to(torch.int64)
    words = words.reshape(-1, 16 * bits // 2, 2).flip(-1).reshape(-1, 16 * bits)
    spans = words.reshape(-1, 16, bits)
    bits_of = (spans[..., None] >> torch.arange(15, -1, -1)) & 1  # [N, 16, bits, 16]
    symbols = bits_of.reshape(-1, 16, 16, bits)
    edges = (symbols << torch.arange(bits - 1, -1, -1)).sum(-1)
    return edges.reshape(-1, _TILE_WEIGHTS)


def decode_trellis_tiles(blocks: torch.Tensor, bits: int) -> torch.Tensor:
    """Decode a batch of 16x16 tiles through b12x's established convention.

    ``blocks`` is ``[N, 16*bits]`` int16; the result is ``[N, 16, 16]`` FP32 in
    ``[out(row), in(col)]`` order, *not* yet scaled by the UE8M0 plane.

    This deliberately does **not** carry its own window arithmetic.  It decodes
    through the same three functions b12x's own reference uses on the payloads
    it builds and validates against the kernel --
    ``_unpack_native_tiles`` (the exact inverse of ``_pack_native_tiles``) then
    ``trellis_moe._cyclic_states`` then ``trellis_moe._decode_states`` with the
    ``procedural_npk`` codebook -- and places the values with the same
    ``_tile_position_map``.  A private re-derivation of the ring geometry
    disagreed with that convention (its cyclic and ring tiles differ), and the
    convention is the one the kernel tests pin, so the private one is gone.
    """

    from b12x.testing.reference import trellis_moe as _b12x_reference

    if blocks.dim() != 2 or blocks.shape[1] * 2 != _TILE_WEIGHTS * bits // 8:
        raise ValueError(
            f"trellis tile batch must be [N, {_TILE_WEIGHTS * bits // 16}] int16, "
            f"got {tuple(blocks.shape)}"
        )
    symbols = _unpack_native_tiles(blocks.to(torch.int16), bits)
    states = _b12x_reference._cyclic_states(symbols, bits)
    values = _b12x_reference._decode_states(
        states, bits, PROCEDURAL_NPK, blocks.device
    )  # [N, 256] float32
    tile = torch.zeros(
        (values.shape[0], _TILE_SIDE, _TILE_SIDE),
        dtype=torch.float32,
        device=blocks.device,
    )
    tile[:, _TILE_ROWS.to(blocks.device), _TILE_COLS.to(blocks.device)] = values
    return tile


def decode_trellis_tile(block: torch.Tensor, bits: int) -> torch.Tensor:
    """Decode one 16x16 tile: ``block`` is 16*bits int16 -> [16, 16] float32."""

    return decode_trellis_tiles(block.reshape(1, -1), bits)[0]


def decode_trellis_matrix(
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    scale_row_offset: int = 0,
    scale_row_stride: int = 1,
) -> torch.Tensor:
    """Decode a whole trellis matrix to dense FP32 weights in the stored basis.

    ``payload`` is the sidecar carrier with the trailing axis of ``16*bits``
    int16 and the two axes before it holding the 16-column (K) tile index and
    the 16-row (N) tile index, in either order -- the caller passes the axes
    already permuted so that ``payload[k16, n16]`` is the tile.

    ``scales`` is the UE8M0/K32 plane addressed as ``scales[row, col // 32]``
    after the caller's own row selection; ``scale_row_offset``/``_stride`` let
    the caller map this matrix's local row index onto the plane's row axis
    (TrellisMX stores ``w13``'s plane in ``up-gate`` order while the payload's
    matrix axis is ``gate-up``).
    """

    if payload.dtype != torch.int16:
        raise ValueError(f"trellis payload must be int16, got {payload.dtype}")
    k16_count, n16_count = payload.shape[0], payload.shape[1]
    if k16_count * 16 != in_features:
        raise ValueError(
            f"trellis payload has {k16_count} K16 tiles for in_features={in_features}"
        )
    if n16_count * 16 != out_features:
        raise ValueError(
            f"trellis payload has {n16_count} N16 tiles for out_features={out_features}"
        )
    blocks = payload.reshape(k16_count * n16_count, -1)
    # 512 tiles per chunk keeps the [N, 256, 16] bit gather under ~70 MB.
    tiles = torch.cat(
        [
            decode_trellis_tiles(blocks[start : start + 512], bits)
            for start in range(0, blocks.shape[0], 512)
        ]
    ).reshape(k16_count, n16_count, _TILE_SIDE, _TILE_SIDE)
    dense = tiles.permute(1, 2, 0, 3).reshape(out_features, in_features).contiguous()
    rows = scale_row_offset + torch.arange(out_features) * scale_row_stride
    block_scale = scales.index_select(0, rows.to(scales.device)).to(torch.float32)
    block_scale = torch.pow(2.0, block_scale - 127.0)
    expanded = block_scale.repeat_interleave(32, dim=1)
    if expanded.shape != dense.shape:
        raise ValueError(
            f"scale plane expands to {tuple(expanded.shape)}, weights are "
            f"{tuple(dense.shape)}"
        )
    return dense * expanded


# --------------------------------------------------------------------------- #
# boundary primitives
# --------------------------------------------------------------------------- #
def _sylvester(order: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < order:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h


def hadamard_normalized(
    value: torch.Tensor, size: int, *, divisor: float | None = None
) -> torch.Tensor:
    """Normalized Sylvester Hadamard over the last axis, FP32, one divide.

    ``divisor`` overrides ``sqrt(size)``; the release uses the float32 values
    ``_P8_SQRT512_F32``/``_P8_SQRT128_F32`` so the terminal divide is a
    ``div.rn.f32`` by exactly that constant.
    """

    if value.shape[-1] % size:
        raise ValueError(
            f"Hadamard width {value.shape[-1]} is not a multiple of {size}"
        )
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
    terminal = (
        float(divisor)
        if divisor is not None
        else float(
            _P8_SQRT512_F32
            if size == 512
            else _P8_SQRT128_F32
            if size == 128
            else math.sqrt(size)
        )
    )
    return (work / terminal).to(torch.float32).view_as(value)


def silu_cap10(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The release's ``silu-cap10``: capped SiLU times a two-sided capped up.

    ``gate = min(gate, 10); up = clamp(up, -10, 10); gate * sigmoid(gate) * up``
    -- ``route_hoist_k5.py:984-999``.  No post-sigmoid clamp.
    """

    gate_capped = torch.clamp(gate.to(torch.float32), max=10.0)
    up_capped = torch.clamp(up.to(torch.float32), min=-10.0, max=10.0)
    return gate_capped * torch.sigmoid(gate_capped) * up_capped


def quantize_e4m3_ue8m0_k32(value: torch.Tensor) -> torch.Tensor:
    """UE8M0/K32 block quantize with a finite E4M3 payload, then reconstruct.

    ``scale = 2**ceil(log2(amax/448))`` clamped to the UE8M0 range
    ``[-127, 127]``, with the exponent taken exactly from the finite FP32 bits
    of the quotient. Zero blocks use scale 1, payload is the round-to-nearest-
    even finite E4M3 cast of ``value/scale``. Non-finite quotients are outside
    the domain. This independent reference follows the same exact-bits rule as
    the boundary and P8 CPU references.
    """

    if value.shape[-1] % 32:
        raise ValueError("MXFP8 quantization needs a width divisible by 32")
    blocks = value.to(torch.float32).reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(maximum > 0, maximum / _E4M3_MAX, torch.ones_like(maximum))
    exponent = _ue8m0_exponent(safe)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    payload = (blocks / scale).to(torch.float8_e4m3fn)
    return (payload.to(torch.float32) * scale).reshape_as(value)


def _ue8m0_exponent(quotient: torch.Tensor) -> torch.Tensor:
    """Exact UE8M0 exponent for a finite FP32 quotient, without floating log2."""

    mantissa, exponent = torch.frexp(quotient)
    exact = torch.where(mantissa == 0.5, exponent - 1, exponent)
    exact = torch.where(quotient > 0, exact, torch.full_like(exact, -127.0))
    return exact.clamp(-127.0, 127.0)


def _interleave_atom32(slot0: torch.Tensor, slot1: torch.Tensor) -> torch.Tensor:
    """``slot0-atom32-slot1-atom32-v1`` on the last axis."""

    width = slot0.shape[-1]
    if width % 32 or slot1.shape != slot0.shape:
        raise ValueError("atom32 interleave needs equal widths divisible by 32")
    blocks = width // 32
    stacked = torch.stack(
        (
            slot0.reshape(-1, blocks, 32),
            slot1.reshape(-1, blocks, 32),
        ),
        dim=2,
    )
    return stacked.reshape(*slot0.shape[:-1], 2 * width)


def _deinterleave_atom32(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = value.shape[-1]
    blocks = width // 64
    shaped = value.reshape(-1, blocks, 2, 32)
    return shaped[:, :, 0, :].reshape(*value.shape[:-1], width // 2), shaped[
        :, :, 1, :
    ].reshape(*value.shape[:-1], width // 2)


# --------------------------------------------------------------------------- #
# the three boundaries
# --------------------------------------------------------------------------- #
def trellismx_input_boundary(
    x_bf16: torch.Tensor,
    gate_up_suh: torch.Tensor,
    *,
    hidden_size: int,
    quantize: bool = True,
) -> torch.Tensor:
    """``bf16 -> cvt.rn fp16 -> H512 -> *suh -> H128 -> e4m3/ue8m0/k32``.

    Returns the FP32 operand the FC1 MMA consumes (FP32 when ``quantize`` is
    False, the dequantized reconstruction when it is True).
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
    source = hadamard_normalized(source, 512, divisor=_P8_SQRT512_F32)
    source = source * gate_up_suh.to(torch.float32)
    source = hadamard_normalized(source, 128, divisor=_P8_SQRT128_F32)
    return quantize_e4m3_ue8m0_k32(source) if quantize else source


def trellismx_intermediate_boundary(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    down_suh: torch.Tensor,
    pre_signs: torch.Tensor,
    post_signs: torch.Tensor,
    quantize: bool = True,
    stage_sink: dict[str, torch.Tensor] | None = None,
    stage_prefix: str = "",
) -> torch.Tensor:
    """The per-expert intermediate boundary, ``gate``/``up`` -> down input.

    ``H128(interleave(g,u)) -> *(gate_svh|up_svh) -> H128 -> *pre_signs ->
    silu_cap10 on adjacent positions -> *post_signs -> H128 -> *down_suh ->
    H128 -> e4m3/ue8m0/k32``.
    """

    intermediate = gate.shape[-1]
    if up.shape != gate.shape:
        raise ValueError("gate and up must share shape")
    if gate_svh.numel() != intermediate or up_svh.numel() != intermediate:
        raise ValueError("gate_svh/up_svh must have one entry per intermediate channel")
    if down_suh.numel() != intermediate:
        raise ValueError("down_suh must have one entry per intermediate channel")
    if pre_signs.numel() != 2 * intermediate or post_signs.numel() != intermediate:
        raise ValueError("pre_signs must be 2I and post_signs I")

    raw = _interleave_atom32(gate.to(torch.float32), up.to(torch.float32))
    scale_raw = _interleave_atom32(gate_svh.to(torch.float32), up_svh.to(torch.float32))

    def record(name, tensor):
        if stage_sink is not None:
            stage_sink[stage_prefix + name] = tensor.detach().to(torch.float32).cpu()

    record("act_raw_interleaved", raw)
    pre = hadamard_normalized(raw, 128, divisor=_P8_SQRT128_F32)
    record("act_h128_1", pre)
    pre = hadamard_normalized(pre * scale_raw, 128, divisor=_P8_SQRT128_F32)
    record("act_h128_2_after_svh", pre)
    pre = pre * pre_signs.to(torch.float32)
    record("act_after_presigns", pre)
    gate_joint = pre[..., 0::2]
    up_joint = pre[..., 1::2]
    activated = silu_cap10(gate_joint, up_joint)
    record("act_silu_cap10", activated)
    activated = hadamard_normalized(
        activated * post_signs.to(torch.float32), 128, divisor=_P8_SQRT128_F32
    )
    record("act_h128_3_after_postsigns", activated)
    down_input = hadamard_normalized(
        activated * down_suh.to(torch.float32), 128, divisor=_P8_SQRT128_F32
    )
    record("fc2_input_pre_quant", down_input)
    if not quantize:
        return down_input
    blocks = down_input.reshape(*down_input.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(maximum > 0, maximum / _E4M3_MAX, torch.ones_like(maximum))
    exponent = _ue8m0_exponent(safe)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    payload = (blocks / scale).to(torch.float8_e4m3fn)
    if stage_sink is not None:
        stage_sink[stage_prefix + "fc2_input_e4m3"] = (
            payload.view(torch.uint8).reshape(down_input.shape).cpu()
        )
        stage_sink[stage_prefix + "fc2_input_ue8m0"] = (
            (exponent.squeeze(-1).to(torch.float32) + 127.0).to(torch.uint8).cpu()
        )
    return (payload.to(torch.float32) * scale).reshape_as(down_input)


def trellismx_output_boundary(
    down: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Per-route ``H128(FP16(MMA)) * down_svh``; the outer H512 is separate."""

    if down.shape[-1] != hidden_size:
        raise ValueError(
            f"down output width {down.shape[-1]} != hidden_size {hidden_size}"
        )
    if down_svh.numel() != hidden_size:
        raise ValueError(
            f"down_svh has {down_svh.numel()} entries, expected {hidden_size}"
        )
    route = hadamard_normalized(down.to(torch.float32), 128, divisor=_P8_SQRT128_F32)
    return route * down_svh.to(torch.float32)


def trellismx_coupled_topk_sum(
    routes: torch.Tensor,
    weights: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Router-order FP32 weighted sum of per-route values, then normalized H512.

    ``routes`` is ``[tokens, topk, hidden]``; the sum runs in router order and
    the H512 is applied afterwards, exactly as ``P8CoupledTopKSumKernel`` does
    (linearity lets the outer transform be hoisted past the reduction).
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
    return hadamard_normalized(total, 512, divisor=_P8_SQRT512_F32).to(torch.bfloat16)


def rank_local_coupled_signs(
    *,
    intermediate: int,
    rank: int,
    world_size: int = 4,
    draw: int = 0,
) -> torch.Tensor:
    """The TP-sliced ``pre[2I] | post[I]`` coupled sign vector, FP16.

    Draw 0 is all ones; other draws use ``qsrt-coupled-signs-v1`` with the
    ``0x6A09E667F3BCC909*draw + 0xBB67AE8584CAA73B*axis`` seed.  Pre signs are
    the axis-1 draw over the checkpoint-global ``2*I_global`` axis and post signs
    the axis-2 draw over ``I_global``, both sliced in atom32 coordinates.
    """

    if intermediate % 32 or rank not in range(world_size):
        raise ValueError("coupled signs need an atom32-aligned valid TP slice")
    global_intermediate = intermediate * world_size
    first = rank * (intermediate // 32) * 32

    def draw_axis(length: int, axis: int) -> torch.Tensor:
        if draw == 0:
            return torch.ones(length, dtype=torch.float16)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (0x6A09E667F3BCC909 * draw + 0xBB67AE8584CAA73B * axis) & ((1 << 63) - 1)
        )
        return (
            torch.randint(0, 2, (length,), generator=generator)
            .mul_(2)
            .sub_(1)
            .to(torch.float16)
        )

    pre = draw_axis(2 * global_intermediate, 1)[2 * first : 2 * (first + intermediate)]
    post = draw_axis(global_intermediate, 2)[first : first + intermediate]
    return torch.cat((pre, post)).contiguous()


# --------------------------------------------------------------------------- #
# sidecar validation
# --------------------------------------------------------------------------- #
_REQUIRED_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)


def validate_sidecar_tensors(
    tensors: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    bits: int,
) -> None:
    """Fail loudly on any missing or mis-shaped sidecar tensor."""

    missing = [name for name in _REQUIRED_TENSORS if name not in tensors]
    if missing:
        raise ValueError(f"TrellisMX sidecar is missing tensors: {missing}")
    hidden_tiles = hidden_size // 16
    slots = intermediate_size // 32
    expected = {
        "w13_trellis": (
            (2, num_experts, hidden_tiles, 2 * slots, 16 * bits),
            torch.int16,
        ),
        "w2_trellis": ((num_experts, 2 * slots, hidden_tiles, 16 * bits), torch.int16),
        "w13_scale_ue8m0": (
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            torch.uint8,
        ),
        "w2_scale_ue8m0": (
            (num_experts, hidden_size, intermediate_size // 32),
            torch.uint8,
        ),
        "intermediate_scales_fp16": (
            (num_experts, 3 * intermediate_size),
            torch.float16,
        ),
        "coupled_sign_draw_u8": ((num_experts,), torch.uint8),
        "gate_up_suh_fp16": ((hidden_size,), torch.float16),
        "down_svh_fp16": ((hidden_size,), torch.float16),
    }
    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"sidecar tensor {name} has shape {tuple(tensor.shape)}, expected {shape}"
            )
        if tensor.dtype != dtype:
            raise ValueError(
                f"sidecar tensor {name} has dtype {tensor.dtype}, expected {dtype}"
            )
    if meta.get("bits", bits) != bits:
        raise ValueError(f"meta bits {meta.get('bits')!r} disagrees with bits={bits}")
    patterns = tensors["coupled_sign_draw_u8"]
    if bool(torch.any(patterns > 7)):
        raise ValueError("coupled_sign_draw_u8 values must be in 0..7")


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def trellismx_moe_reference(
    x_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    *,
    stage_sink: dict[str, torch.Tensor] | None = None,
    stage_route: tuple[int, int] | None = None,
) -> torch.Tensor:
    """One rank slice of one TrellisMX MoE layer, FP32 internally, BF16 out.

    Parameters
    ----------
    x_bf16 : [tokens, hidden] bfloat16
    topk_ids : [tokens, top_k] integer expert ids in this rank's global range
    topk_weights : [tokens, top_k] float
    tensors : the eight sidecar tensors (see ``_REQUIRED_TENSORS``)
    meta : ``hidden_size``, ``intermediate_size`` (TP-local), ``num_experts``,
        ``top_k``, ``bits``, and optionally ``rank``/``world_size``,
        ``sign_draw``, ``quantize_activations`` (default True),
        ``emulate_fp16_stores`` (default True), ``expert_subset``,
        ``w13_scale_plane_order`` (default ``"up-gate"``).
    """

    required = ("hidden_size", "intermediate_size", "num_experts", "top_k", "bits")
    absent = [key for key in required if key not in meta]
    if absent:
        raise ValueError(f"meta is missing required keys: {absent}")
    hidden = int(meta["hidden_size"])
    intermediate = int(meta["intermediate_size"])
    num_experts = int(meta["num_experts"])
    top_k = int(meta["top_k"])
    bits = int(meta["bits"])
    rank = int(meta.get("rank", 0))
    world_size = int(meta.get("world_size", 4))
    sign_draw = int(meta.get("sign_draw", 0))
    quantize = bool(meta.get("quantize_activations", True))
    # The release quantizes the A operand and the FC2 input in *different* places (A in
    # the input-boundary producer, the down input in the FC1 epilogue), so the
    # two can be modelled separately.  `quantize_activations` remains the
    # umbrella and both default to it.
    quantize_a = bool(meta.get("quantize_a", quantize))
    quantize_down = bool(meta.get("quantize_down", quantize))
    fp16_stores = bool(meta.get("emulate_fp16_stores", True))
    # The two axis orders are checkpoint metadata, not constants: the sidecar
    # declares `fc1_scale_plane_order: up-gate` for the UE8M0 N axis against
    # `fc1_trellis_slot_order: gate-up` for the payload's matrix axis, and the
    # release's own kernel relies on exactly that mismatch
    # (route_hoist_k5.py:330-336 stages gate at output_tile + intermediate_tiles).
    # Read both so a future sidecar with the other order cannot be mis-decoded.
    plane_order = str(
        meta.get("w13_scale_plane_order")
        or meta.get("fc1_scale_plane_order")
        or "up-gate"
    )
    if plane_order not in ("up-gate", "gate-up"):
        raise ValueError(
            f"unsupported fc1_scale_plane_order {plane_order!r}; TrellisMX declares "
            "'up-gate' on the UE8M0 N axis"
        )
    slot_order = str(
        meta.get("w13_trellis_slot_order")
        or meta.get("fc1_trellis_slot_order")
        or "gate-up"
    )
    if slot_order not in ("gate-up", "up-gate"):
        raise ValueError(
            f"unsupported fc1_trellis_slot_order {slot_order!r}; TrellisMX declares "
            "'gate-up' on the payload matrix axis"
        )
    gate_matrix, up_matrix = (0, 1) if slot_order == "gate-up" else (1, 0)

    # Accept numpy carriers straight from safetensors' "np" backend.
    tensors = {name: torch.as_tensor(value) for name, value in tensors.items()}
    validate_sidecar_tensors(
        tensors,
        meta,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        bits=bits,
    )
    if top_k < 1 or top_k > 32:
        raise ValueError(f"top_k={top_k} is out of range")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x_bf16.shape[0]:
        raise ValueError(
            f"routing shapes disagree: ids {tuple(topk_ids.shape)}, "
            f"weights {tuple(topk_weights.shape)}, tokens {x_bf16.shape[0]}"
        )
    if topk_ids.shape[1] != top_k:
        raise ValueError(f"topk_ids has {topk_ids.shape[1]} routes, meta top_k={top_k}")

    device = x_bf16.device
    tensors = {name: value.to(device) for name, value in tensors.items()}

    # ---- shared, per-layer, expert-independent ----
    # Diagnostic only: "identity" models a kernel that has NOT been wired to the
    # hidden-axis boundary (no H512, no suh, no H128, no activation quantizer).
    # It exists so a kernel-vs-reference mismatch can be localised to the A
    # operands in one run instead of argued about; the declared format is
    # "trellismx" and that is the default.
    input_boundary_mode = str(meta.get("input_boundary", "trellismx"))
    if input_boundary_mode == "identity":
        source = x_bf16.to(torch.float32)
    elif input_boundary_mode == "trellismx":
        source = trellismx_input_boundary(
            x_bf16, tensors["gate_up_suh_fp16"], hidden_size=hidden, quantize=quantize_a
        )
    else:
        raise ValueError(
            f"unknown input_boundary {input_boundary_mode!r}; expected 'trellismx' "
            "or the diagnostic 'identity'"
        )
    down_svh = tensors["down_svh_fp16"].to(torch.float32)

    signs = rank_local_coupled_signs(
        intermediate=intermediate, rank=rank, world_size=world_size, draw=sign_draw
    ).to(device)
    if bool(torch.any(tensors["coupled_sign_draw_u8"] != sign_draw)):
        raise ValueError(
            "coupled_sign_draw_u8 is not uniform; the reference implements the "
            "per-expert path only through rank_local_coupled_signs(draw=...)"
        )
    pre_signs, post_signs = signs[: 2 * intermediate], signs[2 * intermediate :]

    scale13 = tensors["w13_scale_ue8m0"]
    intermediate_scales = tensors["intermediate_scales_fp16"].to(torch.float32)
    gate_svh_all = intermediate_scales[:, :intermediate]
    up_svh_all = intermediate_scales[:, intermediate : 2 * intermediate]
    down_suh_all = intermediate_scales[:, 2 * intermediate :]

    w13_payload = tensors["w13_trellis"]
    w2_payload = tensors["w2_trellis"]
    w2_scale = tensors["w2_scale_ue8m0"]

    subset = meta.get("expert_subset")
    wanted = {int(e) for e in subset} if subset is not None else set(range(num_experts))

    def decode_expert(expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # w13 payload axis 0 is the gate/up matrix (fc1_trellis_slot_order
        # gate-up); the UE8M0 plane's N axis is up-gate (route_hoist_k5.py:330-336).
        # Both payloads are already [K16 tile, N16 tile, 16*bits]:
        # w13 is (matrix, E, hidden//16, 2*slots, 16*bits) and w2 is
        # (E, 2*slots, hidden//16, 16*bits), i.e. K = intermediate for w2.
        gate_payload = w13_payload[gate_matrix, expert]
        up_payload = w13_payload[up_matrix, expert]
        if plane_order == "up-gate":
            up_offset, gate_offset = 0, intermediate
        else:
            up_offset, gate_offset = intermediate, 0
        gate = decode_trellis_matrix(
            gate_payload,
            scale13[expert],
            bits=bits,
            out_features=intermediate,
            in_features=hidden,
            scale_row_offset=gate_offset,
        )
        up = decode_trellis_matrix(
            up_payload,
            scale13[expert],
            bits=bits,
            out_features=intermediate,
            in_features=hidden,
            scale_row_offset=up_offset,
        )
        # w2_trellis: dim 1 is the intermediate (K) tile axis, dim 2 the hidden
        # (N) tile axis, so the payload is already [K16, N16, block].
        down = decode_trellis_matrix(
            w2_payload[expert],
            w2_scale[expert],
            bits=bits,
            out_features=hidden,
            in_features=intermediate,
        )
        return gate, up, down

    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def weights_of(expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if expert not in cache:
            cache.clear()
            cache[expert] = decode_expert(expert)
        return cache[expert]

    tokens = x_bf16.shape[0]
    routes = torch.zeros(tokens, top_k, hidden, dtype=torch.float32, device=device)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            if expert < 0:
                continue  # dropped route, matching the kernel's id<0 skip
            if expert >= num_experts:
                raise ValueError(
                    f"topk_ids[{token},{route}]={expert} outside 0..{num_experts - 1}"
                )
            if expert not in wanted:
                continue
            gate_w, up_w, down_w = weights_of(expert)
            xin = source[token : token + 1]
            g = xin @ gate_w.transpose(0, 1)
            u = xin @ up_w.transpose(0, 1)
            if fp16_stores:
                g = g.to(torch.float16).to(torch.float32)
                u = u.to(torch.float16).to(torch.float32)
            sink_here = (
                stage_sink
                if stage_sink is not None
                and (stage_route is None or stage_route == (token, route))
                else None
            )
            if sink_here is not None:
                sink_here["fc1_gate"] = g.detach().to(torch.float32).cpu()
                sink_here["fc1_up"] = u.detach().to(torch.float32).cpu()
                sink_here["expert"] = torch.tensor([expert], dtype=torch.int64)
            down_input = trellismx_intermediate_boundary(
                g,
                u,
                gate_svh=gate_svh_all[expert],
                up_svh=up_svh_all[expert],
                down_suh=down_suh_all[expert],
                pre_signs=pre_signs,
                post_signs=post_signs,
                quantize=quantize_down,
                stage_sink=sink_here,
            )
            y = down_input @ down_w.transpose(0, 1)
            if fp16_stores:
                y = y.to(torch.float16).to(torch.float32)
            if sink_here is not None:
                sink_here["fc2_route_pre_output_boundary"] = (
                    y.detach().to(torch.float32).cpu()
                )
            routes[token, route] = trellismx_output_boundary(
                y, down_svh, hidden_size=hidden
            )[0]
            if sink_here is not None:
                sink_here["route_after_output_boundary"] = (
                    routes[token, route].detach().to(torch.float32).cpu()
                )
    if str(meta.get("outer_h512", "trellismx")) == "identity":
        # Diagnostic only: the pre-H512 route-basis sum, which is what a kernel
        # without the release's coupled reducer produces.  The declared format
        # applies the H512 and that is the default.
        total = torch.zeros(tokens, hidden, dtype=torch.float32, device=routes.device)
        weights_f32 = topk_weights.to(torch.float32)
        for k in range(top_k):
            total = total + weights_f32[:, k][:, None] * routes[:, k]
        return total.to(torch.bfloat16)
    return trellismx_coupled_topk_sum(
        routes, topk_weights.to(torch.float32), hidden_size=hidden
    )
