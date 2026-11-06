"""Native MXFP8 QK reads each staged K row where the TMA producer wrote it.

With head_dim 256 the plane-TMA producer stages one 128-column plane per half
of a logical row.  The native QK helper must address the staged K through that
split-plane layout.  Addressing the staged K as a single raw
``[row][head_dim]`` region assembles a logical row from the last owned staged
row and the first unowned one on a partial tile, and packing the K32 A fragment
across the two K16 ldmatrix halves instead of within each half mixes the two
halves of the Q fragment.

These cases cover the plane-TMA decode and q16-extend configurations, with full
and partial last tiles, against the in-tree reference.  They also fill the
unowned K/V/scratch planes with poison bytes on the same configurations.  The
poison comparison is against the native path's own zero baseline, not against
the reference.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from . import _unowned_cache_bytes as ucb
from ._paged_attention_harness import (
    PagedCase,
    PagedHarness,
    check_poison_matrix,
    clean_reference,
    route,
)


NATIVE_PLANE_TMA_CASES = (
    PagedCase(
        "native_ref_decode_full",
        24,
        4,
        256,
        256,
        128,
        -1,
        True,
        1,
        (1,),
        (128,),
        "decode",
        turbo=True,
        reference_min_cos=0.995,
    ),
    PagedCase(
        "native_ref_decode_partial",
        24,
        4,
        256,
        256,
        128,
        -1,
        True,
        1,
        (1,),
        (129,),
        "decode",
        turbo=True,
        reference_min_cos=0.995,
    ),
    PagedCase(
        "native_ref_decode_partial_unsplit",
        24,
        4,
        256,
        256,
        128,
        -1,
        True,
        1,
        (1,),
        (129,),
        "decode",
        (("disable_split_kv", True),),
        turbo=True,
        reference_min_cos=0.995,
    ),
    PagedCase(
        "native_ref_decode_window",
        24,
        4,
        256,
        256,
        128,
        100,
        True,
        1,
        (1,),
        (200,),
        "decode",
        turbo=True,
        reference_min_cos=0.995,
    ),
    PagedCase(
        "native_ref_extend_q16_full",
        24,
        4,
        256,
        256,
        64,
        -1,
        True,
        1,
        (5,),
        (192,),
        "extend",
        turbo=True,
        reference_min_cos=0.995,
    ),
    PagedCase(
        "native_ref_extend_q16_partial",
        24,
        4,
        256,
        256,
        64,
        -1,
        True,
        1,
        (5,),
        (200,),
        "extend",
        turbo=True,
        reference_min_cos=0.995,
    ),
)


@pytest.mark.parametrize("case", NATIVE_PLANE_TMA_CASES, ids=lambda c: c.name)
def test_native_mxfp8_plane_tma_matches_reference(case):
    """Native MXFP8 plane-TMA addressing agrees with the in-tree reference.

    The bounds are the in-tree native checks' bounds
    (``test_attention_paged_forward.py``): max absolute error 0.02 and cosine
    0.995.
    """
    require_b12x()
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            out, _lse = harness.run("ZERO", "all")
            ref = clean_reference(harness, case.batch, case.window, True)
            rows = ucb.active_rows(case.q_lens)
            actual = out.index_select(0, rows).float()
            expected = ref.index_select(0, rows).float()
            assert bool(torch.isfinite(actual).all().item()), (
                f"{case.name}: non-finite output on the clean pool"
            )
            max_abs = float((actual - expected).abs().max().item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.reshape(-1), expected.reshape(-1), dim=0
                ).item()
            )
            print(
                f"{case.name}: max_abs={max_abs:.6g} cosine={cosine:.6g} "
                f"route={route(harness.plan, case)}",
                flush=True,
            )
        finally:
            torch.cuda.synchronize()
    assert max_abs <= 2e-2, f"{case.name}: max |diff| {max_abs:.6g} > 2e-2"
    assert cosine >= 0.995, f"{case.name}: cosine {cosine:.6g} < 0.995"


@pytest.mark.parametrize("case", NATIVE_PLANE_TMA_CASES, ids=lambda c: c.name)
def test_native_mxfp8_unowned_bytes(case):
    """Native MXFP8 outputs are independent of unowned K, V and scratch bytes."""
    require_b12x()
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            failures = check_poison_matrix(
                harness, patterns=("FF", "7F"), planes=("k", "v", "all", "scratch")
            )
        finally:
            torch.cuda.synchronize()
    assert not failures, f"{case.name}: " + "; ".join(failures)
