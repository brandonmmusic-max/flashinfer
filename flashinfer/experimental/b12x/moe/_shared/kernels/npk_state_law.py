"""The procedural-NPK state law, its packed E4M3 decoder, and its CPU reference.

One 16-bit decode window (a t256 ring state) maps to one E4M3 operand byte with
integer arithmetic only: no value table, no shared-memory gather. The law, per
state ``s``:

    w = (s * 0xCBAC1FED) mod 2^32
    x = (w & 0x8FFF8FFF) ^ 0x3B603B60        # PTX lop3, immLut 0x6a
    h = fp16(x & 0xffff) + fp16(x >> 16)     # add.rn.f16x2
    h = h + h                                # alpha-2 compander
    e = cvt.rn.satfinite.e4m3x2.f16x2(h)     # RNE, saturating

The multiplier, mask and or-operand are the same three constants b12x's W4A16
decoder uses (``_lib/intrinsics/trellis.py``
``packed_dequant_trellis_to_half2x4``, ``_lib/intrinsics/constants.py``
``_TRELLIS_MCG``/``_TRELLIS_MASK``/``_TRELLIS_OR``); that decoder widens the
same state law to fp16x2, this one keeps the E4M3 output form the W4A8 QMMA
consumes, so the two share the arithmetic and differ only in the final
conversion.

The law is independent of the trellis rate; what changes with the rate is how a
lane's eight overlapping windows are cut out of the staged t256 ring. Two
extraction laws cover every rate b12x decodes:

``npk_funnel_windows``
    The two-word funnel: the lane's eight-window span (16 + 7*bits bits, so 30
    at K2, 37 at K3, 44 at K4) always fits in the two ring words ``ia``/``ib``.
    This is the same pair ``_w4a8_trellis_window_words`` hands the value-table
    arms, so both codebooks decode the identical windows.

``npk_ring3_windows``
    The K5 three-word ring: at K5 the span is 51 bits and half the lanes start
    deep enough into their first word that it crosses into a third word, which
    the two-word merge cannot see. This law reads the middle ring word too and
    takes each 32-bit window from the 64-bit pair that contains it. Two-word
    lanes reduce to the funnel law exactly (their middle word is their last
    word), so K3 and K4 decode through it unchanged.

``npk_decode_states_reference`` is the same law as a pure-torch CPU reference for
the same 16-bit states, used to pin the device decode bit for bit.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from b12x._lib.intrinsics.atomics import ld_shared_u32
from b12x._lib.intrinsics.constants import _TRELLIS_MASK, _TRELLIS_MCG, _TRELLIS_OR
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_trellis_window_words,
)

#: Every trellis rate the procedural-NPK law decodes.
NPK_RATES = (2, 3, 4, 5)

# E4M3 finite maximum; ``cvt.rn.satfinite`` clamps to it.
E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# CPU reference law
# ---------------------------------------------------------------------------


@functools.cache
def _e4m3_positive_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """The 128 non-negative E4M3 values and their payload bytes, ascending."""

    values: list[float] = []
    payloads: list[int] = []
    for exponent in range(16):
        max_mantissa = 6 if exponent == 15 else 7
        for mantissa in range(max_mantissa + 1):
            if exponent == 0:
                value = mantissa * (2.0**-9)
            else:
                value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
            values.append(value)
            payloads.append((exponent << 3) | mantissa)
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(payloads, dtype=torch.int64),
    )


def _e4m3_rne_satfinite(values: torch.Tensor) -> torch.Tensor:
    """Round float32 values to E4M3FN bytes: RNE, saturating, NaN preserved.

    Every input the NPK law produces is a finite fp16 value in [-8, 8], so the
    saturation and NaN arms are unreachable from ``npk_decode_states_reference``;
    they are implemented because ``cvt.rn.satfinite`` defines them.
    """

    grid, payloads = _e4m3_positive_grid()
    grid = grid.to(values.device)
    payloads = payloads.to(values.device)
    nan = torch.isnan(values)
    magnitude = torch.where(nan, torch.zeros_like(values), values.abs())
    clipped = torch.clamp(magnitude, max=E4M3_MAX)
    right = torch.searchsorted(grid, clipped.contiguous(), right=False)
    right = right.clamp(0, grid.numel() - 1)
    left = (right - 1).clamp(min=0)
    left_distance = clipped - grid[left]
    right_distance = grid[right] - clipped
    ties = left_distance == right_distance
    left_even = (payloads[left] & 1) == 0
    choose_left = (left_distance < right_distance) | (ties & left_even)
    payload = torch.where(choose_left, payloads[left], payloads[right])
    payload = payload | torch.where(
        values < 0, torch.full_like(payload, 0x80), torch.zeros_like(payload)
    )
    payload = torch.where(magnitude == 0, torch.zeros_like(payload), payload)
    payload = torch.where(nan, torch.full_like(payload, 0x7F), payload)
    return payload.to(torch.uint8)


def _state_bits(states: torch.Tensor) -> torch.Tensor:
    if states.is_floating_point() or states.is_complex():
        raise TypeError(f"trellis states must be an integer tensor, got {states.dtype}")
    if states.device.type != "cpu":
        raise ValueError(
            f"npk_decode_states_reference is a CPU reference; got {states.device}"
        )
    return states.to(torch.int64) & 0xFFFF


def _half_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """Reinterpret 16-bit patterns held in a wider integer tensor as fp16."""

    signed = bits.to(torch.int64)
    signed = signed - 65536 * (signed >= 32768).to(torch.int64)
    return signed.to(torch.int16).view(torch.float16)


def npk_decode_states_reference(states: torch.Tensor) -> torch.Tensor:
    """Decode 16-bit trellis states to E4M3FN bytes (CPU, pure torch).

    ``states`` is any integer tensor whose low 16 bits are the decode window
    (``torch.uint16``, ``torch.int16`` or ``torch.int32``); the result is a
    ``torch.uint8`` tensor of the same shape holding raw E4M3 bytes.
    """

    product = (_state_bits(states) * _TRELLIS_MCG) & 0xFFFFFFFF
    word = (product & _TRELLIS_MASK) ^ _TRELLIS_OR
    low = _half_from_bits(word & 0xFFFF)
    high = _half_from_bits((word >> 16) & 0xFFFF)
    combined = low + high
    combined = combined + combined
    return _e4m3_rne_satfinite(combined.float())


# ---------------------------------------------------------------------------
# Device decode
# ---------------------------------------------------------------------------


@dsl_user_op
def packed_decode_npk_to_e4m3x8(win_a, win_b, bits: int, *, loc=None, ip=None):
    """Decode eight sliding-window states to two packed E4M3x4 words.

    ``win_a``/``win_b`` are the two 32-bit funnel windows of one lane's
    eight-weight span. The returned byte order matches
    ``packed_decode_lut_e4m3_to_e4m3x8``: ``$0`` holds weights 0..3 (``win_b``
    at shifts ``3*bits, 2*bits, bits, 0``) and ``$1`` holds weights 4..7
    (``win_a`` at the same shifts).

    Two windows are converted per ``f16x2`` pair: the ``prmt`` pair splits the
    low and high halves of the two post-``lop3`` words so one ``add.rn.f16x2``
    performs both halves' sums and one ``cvt.rn.satfinite.e4m3x2.f16x2``
    converts both weights, halving the conversion count relative to a
    per-weight sequence. The shift schedule is the only rate-dependent part, so
    one body serves K2 through K5.
    """

    bits = int(bits)
    if bits not in NPK_RATES:
        raise ValueError(
            f"procedural_npk decodes K2/K3/K4/K5 streams only, got K{bits}"
        )
    multiplier = f"{_TRELLIS_MCG:#010x}"
    mask = f"{_TRELLIS_MASK:#010x}"
    or_operand = f"{_TRELLIS_OR:#010x}"
    asm = f"""
        {{
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi, M;
            .reg .b32 h01,h23,h45,h67;
            .reg .b16 e01,e23,e45,e67;
            mov.b32 M, {multiplier};
            and.b32 w7, $2, 0xffff;
            shr.u32 w6, $2, {bits}; and.b32 w6, w6, 0xffff;
            shr.u32 w5, $2, {2 * bits}; and.b32 w5, w5, 0xffff;
            shr.u32 w4, $2, {3 * bits}; and.b32 w4, w4, 0xffff;
            and.b32 w3, $3, 0xffff;
            shr.u32 w2, $3, {bits}; and.b32 w2, w2, 0xffff;
            shr.u32 w1, $3, {2 * bits}; and.b32 w1, w1, 0xffff;
            shr.u32 w0, $3, {3 * bits}; and.b32 w0, w0, 0xffff;
            mul.lo.u32 w0, w0, M; lop3.b32 w0, w0, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w1, w1, M; lop3.b32 w1, w1, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w2, w2, M; lop3.b32 w2, w2, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w3, w3, M; lop3.b32 w3, w3, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w4, w4, M; lop3.b32 w4, w4, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w5, w5, M; lop3.b32 w5, w5, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w6, w6, M; lop3.b32 w6, w6, {mask}, {or_operand}, 0x6a;
            mul.lo.u32 w7, w7, M; lop3.b32 w7, w7, {mask}, {or_operand}, 0x6a;
            prmt.b32 lo, w0, w1, 0x5410; prmt.b32 hi, w0, w1, 0x7632; add.rn.f16x2 h01, lo, hi;
            prmt.b32 lo, w2, w3, 0x5410; prmt.b32 hi, w2, w3, 0x7632; add.rn.f16x2 h23, lo, hi;
            prmt.b32 lo, w4, w5, 0x5410; prmt.b32 hi, w4, w5, 0x7632; add.rn.f16x2 h45, lo, hi;
            prmt.b32 lo, w6, w7, 0x5410; prmt.b32 hi, w6, w7, 0x7632; add.rn.f16x2 h67, lo, hi;
            add.rn.f16x2 h01, h01, h01;
            add.rn.f16x2 h23, h23, h23;
            add.rn.f16x2 h45, h45, h45;
            add.rn.f16x2 h67, h67, h67;
            cvt.rn.satfinite.e4m3x2.f16x2 e01, h01;
            cvt.rn.satfinite.e4m3x2.f16x2 e23, h23;
            cvt.rn.satfinite.e4m3x2.f16x2 e45, h45;
            cvt.rn.satfinite.e4m3x2.f16x2 e67, h67;
            mov.b32 $0, {{e01, e23}};
            mov.b32 $1, {{e45, e67}};
        }}
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


# ---------------------------------------------------------------------------
# Window extraction laws
# ---------------------------------------------------------------------------


@cute.jit
def npk_funnel_windows(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """Two-word funnel extraction (K2, K3, K4, and K5 lanes that fit).

    The merged read is the two ring words the lane's eight-window span touches,
    shifted so ``win_a`` starts at the final window and ``win_b`` four windows
    earlier. This is the pair the value-table arms decode, so both codebooks
    see identical windows.
    """

    return _w4a8_trellis_window_words(smem_base, base_u32, ia, ib, s2, bits)


@cute.jit
def npk_ring3_windows(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """Three-word ring extraction for the K5 span, and the funnel below it.

    At K5 a lane's eight overlapping L16 windows span 16 + 7*bits = 51 bits,
    and half the lanes start deep enough into their first ring word that the
    span crosses into a third. Read the middle ring word too and take each
    32-bit window from the 64-bit pair that contains it: ``(mid:last)`` covers
    bits [0, 64) of the 96-bit read and ``(first:mid)`` covers [32, 96). A
    two-word lane keeps ``(first:last)`` as the low pair with a zero top word,
    which reproduces the funnel arithmetic exactly, so K3 and K4 decode through
    this law unchanged.
    """

    if cutlass.const_expr(int(bits) < 5):
        return npk_funnel_windows(smem_base, base_u32, ia, ib, s2, bits)
    a = Uint32(ld_shared_u32(smem_base + ((base_u32 + ia) << Int32(2))))
    b = Uint32(ld_shared_u32(smem_base + ((base_u32 + ib) << Int32(2))))
    ring = Int32(8 * int(bits))
    im = ia + Int32(1)
    im = im - ring * (im >= ring).to(Int32)
    m = Uint32(ld_shared_u32(smem_base + ((base_u32 + im) << Int32(2))))
    top = Uint32(0)
    mid = a
    if ib != im:
        top = a
        mid = m
    lo64 = (Int64(mid) << Int64(32)) | Int64(b)
    hi64 = (Int64(top) << Int64(32)) | Int64(mid)
    win_a = Uint32(lo64 >> Int64(s2))
    t = s2 + Int32(4 * int(bits))
    win_b = Uint32(lo64 >> Int64(t))
    if t >= Int32(32):
        win_b = Uint32(hi64 >> Int64(t - Int32(32)))
    return win_a, win_b


@cute.jit
def npk_decode_ring3(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """One K16 window's E4M3x4 words for both n8 halves, ring extraction."""

    win_a, win_b = npk_ring3_windows(smem_base, base_u32, ia, ib, s2, bits)
    return packed_decode_npk_to_e4m3x8(win_a, win_b, int(bits))


@cute.jit
def npk_pair_words_both(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """(b0, b1) QMMA words for both n8 halves of one N16 tile and one K32.

    Same butterfly contract as ``_w4a8_trellis_pair_words``: the two adjacent
    K16 window blocks of the staged K32 are decoded and redistributed across
    the lane quad, pinned by tests/moe/test_w4a8_fragment_probe.py.
    """

    e0_lo, e0_hi = npk_decode_ring3(smem_base, base0_u32, ia, ib, s2, bits)
    e1_lo, e1_hi = npk_decode_ring3(smem_base, base1_u32, ia, ib, s2, bits)
    c = lane & Int32(3)
    own_lo, own_hi, send_lo, send_hi = e0_lo, e0_hi, e1_lo, e1_hi
    if c >= Int32(2):
        own_lo, own_hi, send_lo, send_hi = e1_lo, e1_hi, e0_lo, e0_hi
    peer_lo = Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2))
    peer_hi = Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2))
    return own_lo, peer_lo, own_hi, peer_hi


__all__ = [
    "E4M3_MAX",
    "NPK_RATES",
    "npk_decode_ring3",
    "npk_decode_states_reference",
    "npk_funnel_windows",
    "npk_pair_words_both",
    "npk_ring3_windows",
    "packed_decode_npk_to_e4m3x8",
]
