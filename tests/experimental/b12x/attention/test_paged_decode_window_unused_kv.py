"""Prepared paged decode ignores non-finite KV below the window start.

The per-pair PV mask cuts keys at or past the tile's live-token bound, but a
sliding-window request also excludes live keys below the earliest query row's
window start. With ``window_left=W`` and width ``n`` query tokens per request,
keys at positions below ``length - n - W`` are unattended by every row, so
poisoning them must not change any output bit, while the first windowed key
must stay live. Each case runs in direct mode and again captured in a CUDA
graph and replayed; the live control poisons the value cache only, with K
finite, so the V path the fix targets is exercised on its own. Follows the
conventions of ``test_paged_unused_kv_paths.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x


PAGE = 128
PAGES_PER_REQUEST = 4
BATCH = 2
NUM_PAGES = 8
GEOMS = ((24, 4, 256), (16, 2, 64))  # (q_heads, kv_heads, head_dim)
DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
WINDOW_LEFT = 100


def _fill_nonfinite(tensor, value):
    if tensor.dtype == torch.float8_e4m3fn:
        tensor.view(torch.uint8).fill_(0x7F)
    else:
        tensor.fill_(value)


def _run_case(q_heads, kv_heads, head_dim, kv_dtype, width, length, replay):
    device = require_b12x()
    key = (torch.randn(NUM_PAGES, PAGE, kv_heads, head_dim, device=device) * 0.2).to(
        kv_dtype
    )
    value = (torch.randn(NUM_PAGES, PAGE, kv_heads, head_dim, device=device) * 0.2).to(
        kv_dtype
    )
    query = (
        torch.randn(BATCH * width, q_heads, head_dim, device=device) * 0.2
    ).bfloat16()
    output = torch.empty_like(query)
    # Identity page table (Luke's convention in test_paged_unused_kv_paths.py):
    # physical cache rows then equal logical KV positions, so poisoning flat
    # rows below the window start is exactly the below-window region. The
    # window lower-bound defect is page-layout independent.
    pages = torch.arange(NUM_PAGES, device=device, dtype=torch.int32).view(
        BATCH, PAGES_PER_REQUEST
    )
    lengths = torch.zeros(BATCH, device=device, dtype=torch.int32)
    cu = torch.arange(BATCH + 1, device=device, dtype=torch.int32) * width
    k_descale = (
        torch.ones((BATCH, kv_heads), dtype=torch.float32, device=device)
        if kv_dtype == torch.float8_e4m3fn
        else None
    )
    v_descale = torch.ones_like(k_descale) if k_descale is not None else None
    args = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        attention_sink_bias=None,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    caps = paged_decode.Caps(
        device=device,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=PAGE,
        max_batch=BATCH,
        max_q_per_req=width,
        causal=True,
        window_left=WINDOW_LEFT,
        has_sinks=False,
        kv_dtype=kv_dtype,
        # Per-head descales ([BATCH, kv_heads]) — the default "tensor" layout
        # makes bind reshape the descale to [1], which rejects the per-head
        # shape and failed every FP8 case at prepare on both trees (reg201b).
        descale_layout="head",
    )
    if not paged_decode.supports(
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=PAGE,
        kv_dtype=kv_dtype,
        max_q_per_req=width,
        window_left=WINDOW_LEFT,
        causal=True,
        has_sinks=False,
    ):
        pytest.skip("unsupported geometry")
    plan = paged_decode.plan(caps)
    scratch = None
    clean_key = key.clone()
    clean_value = value.clone()
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:

        def prepare(state):
            nonlocal scratch
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in plan.scratch_specs()
            ]
            binding = state.bind(plan=plan, scratch=scratch, **args)
            return PreparedCall(
                run=lambda: state.run(binding), output=output, owners=(scratch, binding)
            )

        session.prepare((plan.request(name="window-unused-kv", prepare_call=prepare),))
    binding = paged_decode.bind(plan, scratch=scratch, **args)

    window_start = length - width - WINDOW_LEFT
    assert window_start > 0, "case must expose a below-window region"

    if replay:
        # CUDA-graph replay mode: capture the prepared call once on clean
        # caches, then poison between replays. The graph reads the live cache
        # buffers, so below-window poison must stay invisible under replay
        # exactly as in direct mode. Binding happens outside the capture;
        # preparation documents that binding inside capture is also safe.
        lengths.fill_(length)
        torch.cuda.synchronize(device)
        warmup = torch.cuda.Stream(device)
        warmup.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup):
            paged_decode.run(binding)
        torch.cuda.current_stream(device).wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)

        def run():
            graph.replay()
            torch.cuda.synchronize(device)
            return output
    else:

        def run():
            lengths.fill_(length)
            paged_decode.run(binding)
            torch.cuda.synchronize(device)
            return output

    clean = run().clone()
    assert torch.equal(run(), clean), "non-repeatable clean result"

    reference, _ = paged_attention_reference(
        query,
        key,
        value,
        pages,
        lengths,
        cu,
        causal=True,
        window_left=WINDOW_LEFT,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    maximum_error = (clean - reference).abs().float().max().item()
    assert maximum_error <= 0.02, (
        f"clean windowed result off reference: {maximum_error}"
    )

    failures = []
    poisons = (
        (float("nan"),)
        if kv_dtype == torch.float8_e4m3fn
        else (float("nan"), float("inf"), -float("inf"))
    )
    for poison in poisons:
        for target in ("key", "value", "both"):
            # Reset to the CLEAN baseline, not fresh values: the assertion is
            # that below-window poison changes nothing, so the live region
            # must hold exactly the clean-run values (Luke's harness does
            # this; fresh resets would change the live region too).
            key.copy_(clean_key)
            value.copy_(clean_value)
            for name, cache in (("key", key), ("value", value)):
                if target in (name, "both"):
                    rows = cache.view(BATCH, -1, *cache.shape[-2:])
                    _fill_nonfinite(rows[:, :window_start], poison)
            observed = run()
            finite = bool(torch.isfinite(observed).all())
            unchanged = bool(torch.equal(observed, clean))
            case = f"length={length} width={width} target={target} poison={poison}"
            print(f"{case} finite={finite} unchanged={unchanged}", flush=True)
            if not finite or not unchanged:
                failures.append(f"{case}: finite={finite}, unchanged={unchanged}")

    # Live control: the first windowed key must stay live. Poison exactly that
    # position in the value cache only, with K kept finite: a V NaN must
    # surface on its own, since a K NaN reaches the output through the score
    # path and would prove nothing about the V mask this fix adds. A pass here
    # with the below-window cases failing would mean the window boundary
    # itself moved rather than the mask working.
    key.copy_(clean_key)
    value.copy_(clean_value)
    rows = value.view(BATCH, -1, *value.shape[-2:])
    _fill_nonfinite(rows[:, window_start], float("nan"))
    observed = run()
    live_nan_visible = not bool(torch.isfinite(observed).all())
    print(
        f"length={length} width={width} live_nan_visible={live_nan_visible}",
        flush=True,
    )
    if not live_nan_visible:
        failures.append(
            f"length={length} width={width}: a live windowed NaN was hidden"
        )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("replay", (False, True))
@pytest.mark.parametrize("length", (257, 129))
@pytest.mark.parametrize("width", (8, 1))
@pytest.mark.parametrize("kv_dtype", DTYPES)
@pytest.mark.parametrize("geom", GEOMS)
def test_paged_decode_window_unused_kv(geom, kv_dtype, width, length, replay):
    q_heads, kv_heads, head_dim = geom
    _run_case(q_heads, kv_heads, head_dim, kv_dtype, width, length, replay)


@pytest.mark.parametrize("replay", (False, True))
@pytest.mark.parametrize("aligned", (16, 128))
@pytest.mark.parametrize("kv_dtype", DTYPES)
@pytest.mark.parametrize("geom", GEOMS)
def test_paged_decode_window_aligned_boundary(geom, kv_dtype, aligned, replay):
    """Window starts exactly on 16-key and page/stage boundaries.

    The parametrized cases above use lengths whose window start falls mid-run;
    these place the earliest window start exactly on a multiple of 16 and on a
    multiple of 128 (the page size and the cp.async stage span), where an
    off-by-one in the bound would be invisible at unaligned offsets. Reuses
    the full assertion battery of ``_run_case`` unchanged.
    """
    q_heads, kv_heads, head_dim = geom
    width = 8
    window_start = 128 if aligned == 128 else 272
    length = window_start + width + WINDOW_LEFT
    _run_case(q_heads, kv_heads, head_dim, kv_dtype, width, length, replay)


def _run_straddle_case(q_heads, kv_heads, head_dim, kv_dtype, replay):
    device = require_b12x()
    width = 16
    length = 257
    key = (torch.randn(NUM_PAGES, PAGE, kv_heads, head_dim, device=device) * 0.2).to(
        kv_dtype
    )
    value = (torch.randn(NUM_PAGES, PAGE, kv_heads, head_dim, device=device) * 0.2).to(
        kv_dtype
    )
    query = (
        torch.randn(BATCH * width, q_heads, head_dim, device=device) * 0.2
    ).bfloat16()
    output = torch.empty_like(query)
    pages = torch.arange(NUM_PAGES, device=device, dtype=torch.int32).view(
        BATCH, PAGES_PER_REQUEST
    )
    lengths = torch.zeros(BATCH, device=device, dtype=torch.int32)
    cu = torch.arange(BATCH + 1, device=device, dtype=torch.int32) * width
    k_descale = (
        torch.ones((BATCH, kv_heads), dtype=torch.float32, device=device)
        if kv_dtype == torch.float8_e4m3fn
        else None
    )
    v_descale = torch.ones_like(k_descale) if k_descale is not None else None
    args = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        attention_sink_bias=None,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    caps = paged_decode.Caps(
        device=device,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=PAGE,
        max_batch=BATCH,
        max_q_per_req=width,
        causal=True,
        window_left=WINDOW_LEFT,
        has_sinks=False,
        kv_dtype=kv_dtype,
        descale_layout="head",
    )
    if not paged_decode.supports(
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=PAGE,
        kv_dtype=kv_dtype,
        max_q_per_req=width,
        window_left=WINDOW_LEFT,
        causal=True,
        has_sinks=False,
    ):
        pytest.skip("unsupported geometry")
    plan = paged_decode.plan(caps)
    scratch = None
    clean_key = key.clone()
    clean_value = value.clone()
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        head_splits_seen: list[int] = []

        def prepare(state):
            nonlocal scratch
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in plan.scratch_specs()
            ]
            binding = state.bind(plan=plan, scratch=scratch, **args)
            head_splits_seen.append(state.config.head_splits)
            return PreparedCall(
                run=lambda: state.run(binding), output=output, owners=(scratch, binding)
            )

        session.prepare((plan.request(name="window-straddle", prepare_call=prepare),))
    binding = paged_decode.bind(plan, scratch=scratch, **args)

    # Rows are packed in token order with sub_group packed rows per token,
    # where sub_group = group_size // head_splits and head_splits comes from
    # the plan's config (windowed decode splits heads by default). A warp
    # holds 16 packed rows, so token boundaries fall on warp boundaries only
    # at multiples of 16 // gcd(sub_group, 16) tokens. The band ends at the
    # first such boundary past token 0: keys below it are excluded for every
    # row from the containing warp on, while rows of earlier tokens attend
    # them. If no boundary exists within the request width the geometry has
    # no inter-warp straddle to probe and the case skips.
    sub_group = (q_heads // kv_heads) // head_splits_seen[0]
    split_token = 16 // math.gcd(sub_group, 16)
    if split_token >= width:
        pytest.skip(
            f"no warp-aligned token boundary within width={width} "
            f"(sub_group={sub_group})"
        )
    window_start = length - width - WINDOW_LEFT
    band_lo = window_start
    band_hi = window_start + split_token
    assert window_start > 0 and band_hi < length

    if replay:
        lengths.fill_(length)
        torch.cuda.synchronize(device)
        warmup = torch.cuda.Stream(device)
        warmup.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup):
            paged_decode.run(binding)
        torch.cuda.current_stream(device).wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)

        def run():
            graph.replay()
            torch.cuda.synchronize(device)
            return output

    else:

        def run():
            lengths.fill_(length)
            paged_decode.run(binding)
            torch.cuda.synchronize(device)
            return output

    clean = run().clone()
    assert torch.equal(run(), clean), "non-repeatable clean result"
    reference, _ = paged_attention_reference(
        query,
        key,
        value,
        pages,
        lengths,
        cu,
        causal=True,
        window_left=WINDOW_LEFT,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    maximum_error = (clean - reference).abs().float().max().item()
    assert maximum_error <= 0.02, (
        f"clean windowed result off reference: {maximum_error}"
    )

    # Poison exactly the band, in the value cache only, with K finite: rows
    # before the split attend the band and must surface the poison; rows at
    # or past the split exclude it and must stay bit-identical to clean.
    key.copy_(clean_key)
    value.copy_(clean_value)
    rows = value.view(BATCH, -1, *value.shape[-2:])
    _fill_nonfinite(rows[:, band_lo:band_hi], float("nan"))
    observed = run()
    shape = (BATCH, width, *observed.shape[1:])
    early = observed.view(shape)[:, :split_token]
    late = observed.view(shape)[:, split_token:]
    clean_late = clean.view(shape)[:, split_token:]
    early_poison_visible = not bool(torch.isfinite(early).all())
    late_unchanged = bool(torch.equal(late, clean_late))
    print(
        f"straddle width={width} length={length} split_token={split_token} "
        f"early_poison_visible={early_poison_visible} late_unchanged={late_unchanged}",
        flush=True,
    )
    failures = []
    if not early_poison_visible:
        failures.append("band poison invisible to the early rows that attend it")
    if not late_unchanged:
        failures.append("band poison changed rows whose window excludes it")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("replay", (False, True))
@pytest.mark.parametrize("kv_dtype", DTYPES)
@pytest.mark.parametrize("geom", GEOMS)
def test_paged_decode_window_straddle_between_warps(geom, kv_dtype, replay):
    """A key excluded for later warps but live for earlier warps.

    The per-warp lower bound is the earliest row's window start, so a key at
    or above an earlier warp's start but below a later warp's start must be
    read by the early rows and masked for the later ones. The poison band
    spans [token 0's window start, the first warp-aligned token boundary's
    window start); that boundary exists only for some head splits, so
    geometries without one within the request width skip (see
    ``_run_straddle_case``). On the unpatched base the band is read by every
    row, so the late rows change and the case fails.
    """
    q_heads, kv_heads, head_dim = geom
    _run_straddle_case(q_heads, kv_heads, head_dim, kv_dtype, replay)
