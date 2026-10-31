"""P8 specialization of b12x's shared materialized-W4A8 FC2 kernel.

The compute body is the shared ``W4A8MaterializedPhase2Kernel``; the P8 arms
only decode their procedural NPK payload with the table-free decoder in
:mod:`b12x.moe._shared.p8.kernels.npk_decode` instead of the value-table law.
"""

from __future__ import annotations

import cutlass.cute as cute

from b12x.moe._shared.kernels.w4a8_phase2 import (
    W4A8MaterializedPhase2Kernel as _SharedPhase2Kernel,
)
from b12x.moe._shared.p8.kernels.npk_decode import w4a8_trellis_pair_words_dispatch


class P8W4A8MaterializedPhase2Kernel(_SharedPhase2Kernel):
    """The shared phase-2 kernel with the P8 arms' trellis decode law."""

    @cute.jit
    def _trellis_pair_words(self, smem_base, lane, base0, base1, ia, ib, s2, lut_addr):
        return w4a8_trellis_pair_words_dispatch(
            smem_base,
            lane,
            base0,
            base1,
            ia,
            ib,
            s2,
            self.trellis_bits,
            lut_addr,
            self.w4a8_trellis_lut,
            self.trellis_direct_lut,
        )
