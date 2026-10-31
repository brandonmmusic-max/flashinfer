"""Procedural NPK decode for the W4A8 trellis path.

The ``procedural_npk`` codebook maps one 16-bit decode window (a t256 ring
state) to one E4M3 operand byte with integer arithmetic only: no value table,
no shared-memory gather. The state law itself, its CPU reference and the two
window-extraction laws live in one place,
:mod:`b12x.moe._shared.kernels.npk_state_law`; this module keeps the W4A8
phase and dynamic kernels' names and selects the ring extraction law, which
traces the same two-word funnel the ``lut_e4m3`` arm decodes for K2 to K4 and
reads a third ring word only at K5.

The packed byte order is ``packed_decode_lut_e4m3_to_e4m3x8``'s: the low word
holds weights 0..3 taken from ``win_b`` at shifts ``3*bits .. 0`` and the high
word holds weights 4..7 from ``win_a``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x.moe._shared.kernels.npk_state_law import (
    E4M3_MAX,
    NPK_RATES,
    npk_decode_ring3,
    npk_decode_states_reference,
    npk_pair_words_both,
    packed_decode_npk_to_e4m3x8,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_trellis_pair_words,
    _w4a8_trellis_pair_words_both,
)
from b12x.moe._shared.trellis_codebooks import LUT_E4M3, PROCEDURAL_NPK

#: Rates the W4A8 phase and dynamic kernels accept. K5 is a P8 arm; it decodes
#: through the same law with the ring extraction, which is what
#: ``npk_decode_ring3`` selects at that rate.
NPK_BITS = NPK_RATES


@cute.jit
def _w4a8_npk_decode_both(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """Decode one K16 window's E4M3x4 words for both n8 halves.

    The ring extraction law reads a third ring word only at K5 and traces the
    two-word funnel below it, so K2 to K4 compile to the same instructions they
    always did.
    """

    return npk_decode_ring3(smem_base, base_u32, ia, ib, s2, bits)


@cute.jit
def _w4a8_npk_decode_half(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    n_high: Int32,
    bits: cutlass.Constexpr,
) -> Uint32:
    """Decode one K16 window's E4M3x4 word for one n8 half of an N16 tile."""

    lo, hi = _w4a8_npk_decode_both(smem_base, base_u32, ia, ib, s2, bits)
    value = lo
    if n_high != Int32(0):
        value = hi
    return value


@cute.jit
def _w4a8_npk_pair_words(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    n_high: Int32,
    bits: cutlass.Constexpr,
):
    """(b0, b1) QMMA E4M3 words for one n8 half and one K32 block.

    Same butterfly contract as ``_w4a8_trellis_pair_words``: the two adjacent
    K16 window blocks of the staged K32 are decoded and redistributed across
    the lane quad, pinned by tests/moe/test_w4a8_fragment_probe.py.
    """

    e0 = _w4a8_npk_decode_half(smem_base, base0_u32, ia, ib, s2, n_high, bits)
    e1 = _w4a8_npk_decode_half(smem_base, base1_u32, ia, ib, s2, n_high, bits)
    c = lane & Int32(3)
    own = e0
    send = e1
    if c >= Int32(2):
        own = e1
        send = e0
    peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))
    return own, peer


@cute.jit
def _w4a8_npk_pair_words_both(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
):
    """(b0, b1) QMMA words for both n8 halves of one N16 tile and one K32."""

    return npk_pair_words_both(smem_base, lane, base0_u32, base1_u32, ia, ib, s2, bits)


__all__ = [
    "E4M3_MAX",
    "NPK_BITS",
    "NPK_RATES",
    "PROCEDURAL_NPK",
    "npk_decode_states_reference",
    "packed_decode_npk_to_e4m3x8",
]


# ---------------------------------------------------------------------------
# Compile-time codebook dispatch
#
# The decode sites select the arm from a ``cutlass.const_expr`` on the
# backend's ``trellis_codebook``, so a specialization traces exactly one arm
# and the unused one costs neither instructions nor shared memory. The
# ``lut_e4m3`` arm keeps its value-table arguments; the ``procedural_npk``
# arm reads no table at all.
# ---------------------------------------------------------------------------


@cute.jit
def _w4a8_trellis_pair_words_codebook(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    n_high: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    codebook: cutlass.Constexpr = LUT_E4M3,
    direct_lut: cutlass.Constexpr = False,
):
    """(b0, b1) QMMA E4M3 words for one n8 half, by codebook."""

    if cutlass.const_expr(codebook == PROCEDURAL_NPK):
        own, peer = _w4a8_npk_pair_words(
            smem_base, lane, base0_u32, base1_u32, ia, ib, s2, n_high, bits
        )
    else:
        own, peer = _w4a8_trellis_pair_words(
            smem_base,
            lane,
            base0_u32,
            base1_u32,
            ia,
            ib,
            s2,
            n_high,
            bits,
            lut_addr,
            direct_lut,
        )
    return own, peer


@cute.jit
def _w4a8_trellis_pair_words_both_codebook(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    codebook: cutlass.Constexpr = LUT_E4M3,
    lut_in_smem: cutlass.Constexpr = True,
    direct_lut: cutlass.Constexpr = False,
):
    """(b0, b1) QMMA words for both n8 halves of one N16 tile, by codebook."""

    if cutlass.const_expr(codebook == PROCEDURAL_NPK):
        own_lo, peer_lo, own_hi, peer_hi = _w4a8_npk_pair_words_both(
            smem_base, lane, base0_u32, base1_u32, ia, ib, s2, bits
        )
    else:
        own_lo, peer_lo, own_hi, peer_hi = _w4a8_trellis_pair_words_both(
            smem_base,
            lane,
            base0_u32,
            base1_u32,
            ia,
            ib,
            s2,
            bits,
            lut_addr,
            lut_in_smem,
            direct_lut,
        )
    return own_lo, peer_lo, own_hi, peer_hi
