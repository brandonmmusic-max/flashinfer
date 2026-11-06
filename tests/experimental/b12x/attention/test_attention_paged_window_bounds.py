"""Windowed paged PV must ignore keys below the window start.

``PagedForwardKernel`` (decode/verify) and the generic extend kernel enumerate
KV tiles from a page-floored window start, so the first staged tile can begin
below the oldest key any query row of the tile attends.  The score mask sets
those keys' softmax weight to zero, but a zero weight does not suppress a NaN
in the staged V row, so the PV operands must be masked by key position too.

The unowned-byte cases fill exactly the rows below the window start with a byte
pattern the kernel must never read.  The straddle cases poison a band that is
live for an earlier Q warp and dead for a later one: the per-warp bound must
keep the band for the earlier warp and drop it for the later warp.
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


# Dense decode through PagedForwardKernel.  The exact window start is 99; the
# planner floors it to page 0, so keys 0..98 are attended by no query row.
PAGED_FORWARD_WINDOW_CASES = {
    "paged_forward_window_bf16": PagedCase(
        "paged_forward_window_bf16",
        24,
        4,
        256,
        256,
        128,
        100,
        False,
        1,
        (1,),
        (200,),
        "decode",
    ),
    "paged_forward_window_fp8": PagedCase(
        "paged_forward_window_fp8",
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
    ),
}

# The extend TMA staging path page-aligns the window start (the direct copy
# path starts at the exact boundary), so the first staged tile begins at 64
# while the exact start is 95.
PAGED_EXTEND_WINDOW_CASE = PagedCase(
    "paged_extend_window_bf16",
    24,
    4,
    256,
    256,
    64,
    100,
    False,
    1,
    (5,),
    (200,),
    "extend",
)

# A windowed verify request whose Q rows span several packed tiles.  With
# cta_tile_q=16 and group 8, tile 1 begins at query token 2 and its exact
# bound is 94 while the request-wide bound is 92.
TILE_LOCAL_VERIFY_CASE = PagedCase(
    "paged_forward_verify_window_tile",
    32,
    4,
    256,
    256,
    64,
    100,
    False,
    1,
    (8,),
    (200,),
    "verify",
)

# Extend geometry with four Q warps: cta_tile_q=128 packed rows, group 8, so a
# warp owns four query tokens and the window start advances four keys per warp.
EXTEND_STRADDLE_CASE = PagedCase(
    "paged_extend_window_straddle",
    32,
    4,
    128,
    128,
    64,
    100,
    False,
    1,
    (16,),
    (200,),
    "extend",
)

# The only planner-reachable decode/verify geometry with more than one Q warp
# uses a 64-row query tile (cta_tile_q=64).  A window keeps this case on the
# generic PagedForwardKernel instead of the dedicated verifier kernel.
FORWARD_STRADDLE_CASE = PagedCase(
    "paged_forward_window_straddle",
    24,
    4,
    128,
    128,
    128,
    100,
    True,
    1,
    (8,),
    (200,),
    "verify",
)


@pytest.mark.parametrize(
    "case",
    [pytest.param(value, id=key) for key, value in PAGED_FORWARD_WINDOW_CASES.items()],
)
def test_paged_forward_window_unowned_bytes(case):
    """Windowed PagedForwardKernel outputs ignore unowned bytes below the window."""
    require_b12x()
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            failures = check_poison_matrix(harness)
        finally:
            torch.cuda.synchronize()
    assert not failures, f"{case.name}: " + "; ".join(failures)


def test_paged_extend_window_unowned_bytes():
    """Windowed generic-extend outputs ignore unowned bytes below the window."""
    require_b12x()
    case = PAGED_EXTEND_WINDOW_CASE
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            failures = check_poison_matrix(harness)
        finally:
            torch.cuda.synchronize()
    assert not failures, f"{case.name}: " + "; ".join(failures)


def test_paged_verify_window_tile_local_lower_bound():
    """A verify tile's lower bound starts at its own first query token.

    The request-wide bound is 92, but tile 1 begins at query token 2, whose
    exact bound is 94.  Keys 92 and 93 are excluded by every query row from
    token 2 on (tiles 1..3) while tokens 0 and 1 attend them, so only rows
    2..7 are compared.
    """
    require_b12x()
    case = TILE_LOCAL_VERIFY_CASE
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            base, base_lse = harness.run("ZERO", "all")
            rows = torch.arange(2, case.q_lens[0], dtype=torch.long, device="cuda")
            out, lse = harness.run_with_v_poison(0, (92, 93))
            cmp = ucb.compare(base, out, rows=rows)
            lse_cmp = ucb.compare(base_lse, lse, rows=rows)
        finally:
            torch.cuda.synchronize()
    assert cmp.equal and lse_cmp.equal, f"{case.name}: {cmp.describe()}" + (
        "" if lse_cmp.equal else f"; lse: {lse_cmp.describe()}"
    )


def _straddle(case: PagedCase, split_token: int, early_tokens: int) -> None:
    """Poison the band [start(token 0), start(split_token)) in V only.

    Rows before ``split_token`` attend at least part of the band and must see
    the poison.  Rows from ``split_token`` on exclude it entirely and must stay
    bitwise identical to the clean result.
    """
    require_b12x()
    with torch.inference_mode():
        harness = PagedHarness(case)
        try:
            width = case.q_lens[0]
            length = case.lens[0]
            window_start = length - width - case.window
            band_lo = window_start
            band_hi = window_start + split_token
            assert band_lo >= 0 and band_hi <= length
            clean, clean_lse = harness.run("ZERO", "all")
            out, lse = harness.run_with_v_poison(0, range(band_lo, band_hi))
            shape = (width, *clean.shape[1:])
            early = out.view(shape)[:early_tokens]
            late = out.view(shape)[split_token:]
            clean_late = clean.view(shape)[split_token:]
            late_lse = lse.view(width, -1)[split_token:]
            clean_late_lse = clean_lse.view(width, -1)[split_token:]
            early_poison_visible = not bool(torch.isfinite(early).all())
            late_unchanged = bool(torch.equal(late, clean_late)) and bool(
                torch.equal(late_lse, clean_late_lse)
            )
            print(
                f"{case.name}: band=[{band_lo},{band_hi}) split_token={split_token} "
                f"early_poison_visible={early_poison_visible} "
                f"late_unchanged={late_unchanged}",
                flush=True,
            )
        finally:
            torch.cuda.synchronize()
    failures = []
    if not early_poison_visible:
        failures.append("band poison invisible to the early rows that attend it")
    if not late_unchanged:
        failures.append("band poison changed rows whose window excludes it")
    assert not failures, f"{case.name}: " + "; ".join(failures)


def test_paged_extend_window_straddle_between_warps():
    """A key dead for a later Q warp must not reach that warp's PV fragment.

    The extend tile has four Q warps of four query tokens each (cta_tile_q=128,
    group 8).  Keys [84, 88) are live for warp 0 (tokens 0..3) and dead for
    warps 1..3 (tokens 4..15).  Poisoning only V isolates the PV path.
    """
    _straddle(EXTEND_STRADDLE_CASE, split_token=4, early_tokens=1)


def test_paged_forward_window_straddle_between_warps():
    """A straddle on the multi-Q-warp decode/verify PagedForwardKernel.

    The only planner-reachable decode/verify geometry with more than one Q warp
    uses cta_tile_q=64.  A packed group of 6 means a warp boundary falls inside
    a token, so the first token owned entirely by warp 1 is token 6; tokens
    0..4 attend the band and tokens 6..7 exclude it.
    """
    from b12x.attention.paged.traits import select_paged_forward_traits_from_plan

    require_b12x()
    with torch.inference_mode():
        harness = PagedHarness(FORWARD_STRADDLE_CASE)
        try:
            traits = select_paged_forward_traits_from_plan(harness.plan)
            if traits.num_warps_q <= 1:
                pytest.skip(
                    "the planner gave this decode/verify geometry a single Q "
                    f"warp (cta_tile_q={traits.cta_tile_q}); there is no "
                    "inter-warp boundary to probe"
                )
            assert traits.num_warps_q == 4
            assert harness.plan.cta_tile_q == 64
            try:
                _straddle(FORWARD_STRADDLE_CASE, split_token=6, early_tokens=1)
            except NotImplementedError as error:
                pytest.skip(
                    "the only multi-Q-warp decode/verify geometry is the "
                    "dedicated verifier tile, and PagedForwardKernel rejects "
                    f"a windowed call that does not use TMA staging: {error}"
                )
        finally:
            torch.cuda.synchronize()
