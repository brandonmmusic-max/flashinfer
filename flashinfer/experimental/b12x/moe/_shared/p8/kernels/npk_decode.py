"""Procedural NPK to E4M3 decode shared by the monolithic and split P8 kernels.

The state law, its CPU reference and the two window-extraction laws live in
:mod:`b12x.moe._shared.kernels.npk_state_law`; this module keeps the P8
kernels' names and selects the three-word ring extraction, which K5 needs and
which reduces to the two-word funnel at K3 and K4.

The multiplier, mask and xor constants and the 16-bit window state follow
ExLlamaV3's procedural trellis decoder. The procedural NPK law adds the alpha-2
half2 compander followed by a native ties-to-even finite E4M3 conversion,
returned directly in QMMA B-register order. No weight matrix and no runtime
codebook table are materialized.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from b12x.moe._shared.kernels.npk_state_law import (
    npk_decode_ring3,
    npk_pair_words_both,
    packed_decode_npk_to_e4m3x8,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_trellis_pair_words_both as _sqg_pair_words_both,
)


#: Name the P8 dynamic kernel and the K5 funnel kernels import.
packed_decode_npk2_to_e4m3x8 = packed_decode_npk_to_e4m3x8


@cute.jit
def _npk_decode_both(smem_base, base, ia, ib, s2, bits):
    """Decode one K16 window's E4M3x4 words for both n8 halves.

    The ring law reads the third ring word at K5 and reproduces the two-word
    funnel at K3 and K4; see ``npk_state_law.npk_ring3_windows``.
    """

    return npk_decode_ring3(smem_base, base, ia, ib, s2, bits)


@cute.jit
def _npk_pair_words_both(smem_base, lane, base0, base1, ia, ib, s2, bits):
    return npk_pair_words_both(smem_base, lane, base0, base1, ia, ib, s2, bits)


@cute.jit
def w4a8_trellis_pair_words_dispatch(
    smem_base,
    lane,
    base0,
    base1,
    ia,
    ib,
    s2,
    bits,
    lut_addr,
    lut_in_smem=True,
    direct_lut=False,
):
    """Dispatch SQG or table-free NPK using compile-time LUT mode flags."""

    if cutlass.const_expr(not lut_in_smem and not direct_lut):
        return _npk_pair_words_both(smem_base, lane, base0, base1, ia, ib, s2, bits)
    return _sqg_pair_words_both(
        smem_base,
        lane,
        base0,
        base1,
        ia,
        ib,
        s2,
        bits,
        lut_addr,
        lut_in_smem,
        direct_lut,
    )


__all__ = [
    "packed_decode_npk_to_e4m3x8",
    "w4a8_trellis_pair_words_dispatch",
]
