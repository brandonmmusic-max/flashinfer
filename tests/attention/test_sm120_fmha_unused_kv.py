# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unused FP8 K/V bytes must not affect captured FMHA outputs or live input bytes."""

import math

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import get_compute_capability

if not is_cute_dsl_experimental_available():
    pytest.skip("SM120 CuTe DSL dependencies are unavailable", allow_module_level=True)

_device = torch.device("cuda") if torch.cuda.is_available() else None
_capability = get_compute_capability(_device) if _device is not None else None
pytestmark = pytest.mark.skipif(
    _capability not in ((12, 0), (12, 1)),
    reason="Paged FP8 TMA FMHA requires SM120 or SM121",
)


def _finite_fp8(shape):
    data = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
    data.masked_fill_((data & 0x7F) == 0x7F, 0)
    return data.view(torch.float8_e4m3fn)


def _capture(run):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def _check_finite_reference(
    graph, q, k, v, o, lse, kv_length, cu_q, kv_tile, is_causal, threshold, gate=None
):
    from tests.attention.test_sm120_fmha import _ref_fmha_and_lse, _tol

    for operand in (k, v):
        operand.copy_(
            torch.randint(-2, 3, operand.shape, device="cuda", dtype=torch.int32).to(
                operand.dtype
            )
        )
    if k.ndim == 4:
        # The block table visits physical pages in reverse order.
        dense_k = k.float().flip(0).transpose(1, 2).reshape(-1, k.shape[1], k.shape[3])
        dense_v = v.float().flip(0).transpose(1, 2).reshape_as(dense_k)
    else:
        dense_k, dense_v = k.float(), v.float()
    if threshold is not None:
        # A zero skip threshold retains every tile for the dense reference.
        threshold.zero_()

    q_length = 17
    cu_q.copy_(
        torch.arange(cu_q.numel(), device=cu_q.device, dtype=cu_q.dtype) * q_length
    )
    for length in (17, 31, 32, 33, kv_tile - 1, kv_tile, kv_tile + 1, 2 * kv_tile + 1):
        kv_length.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        expected_o, expected_lse = _ref_fmha_and_lse(
            q[:q_length].unsqueeze(0),
            dense_k[:length].unsqueeze(0),
            dense_v[:length].unsqueeze(0),
            1.0 / math.sqrt(q.shape[-1]),
            is_causal,
        )
        expected_o = expected_o[0]
        if gate is not None:
            expected_o *= torch.sigmoid(gate[:q_length].float())
        torch.testing.assert_close(
            o[:q_length], expected_o.to(o.dtype), **_tol(o.dtype)
        )
        torch.testing.assert_close(
            lse[:q_length], expected_lse[0], atol=2e-3, rtol=2e-3
        )

    kv_length.fill_(1)
    if v.ndim == 4:
        v[-1, :, 0, :].fill_(1)
    else:
        v[0].fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    expected_o = torch.ones_like(o[:q_length])
    if is_causal:
        # Q=17 and KV=1 have sixteen fully masked rows under bottom-right alignment.
        expected_o[:-1].zero_()
        assert torch.count_nonzero(o[: q_length - 1]) == 0
        assert torch.isneginf(lse[: q_length - 1]).all()
    if gate is not None:
        expected_o = (expected_o.float() * torch.sigmoid(gate[:q_length].float())).to(
            o.dtype
        )
    torch.testing.assert_close(o[:q_length], expected_o, **_tol(o.dtype))


def _check_tail_replays(
    graph, k, v, o, lse, kv_length, cu_q, logical_rows, kv_tile, is_causal
):
    max_q_length = 17
    clean_k = k.view(torch.uint8).clone()
    clean_v = v.view(torch.uint8).clone()
    # E4M3FN converts infinities to NaNs. Include both signs and finite extrema.
    poison = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 448.0, -448.0],
        device="cuda",
    ).to(torch.float8_e4m3fn)
    poison = poison.view(torch.uint8)[
        torch.arange(k.numel(), device="cuda") % poison.numel()
    ].reshape(k.shape)
    lengths = sorted(
        {
            0,
            1,
            3,
            4,
            5,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            kv_tile - 1,
            kv_tile,
            kv_tile + 1,
            2 * kv_tile + 1,
        }
    )
    cases = [
        (length, min(max_q_length, length) if is_causal and length else max_q_length)
        for length in lengths
    ]
    if is_causal:
        # Retain rectangular cases whose leading Q rows have no visible keys.
        cases += [
            (length, max_q_length) for length in lengths if 0 < length < max_q_length
        ]
    following_q_offsets = (
        torch.arange(cu_q.numel() - 1, device=cu_q.device, dtype=cu_q.dtype)
        * max_q_length
    )

    def replay():
        o.fill_(-42)
        lse.fill_(42)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(o[active_q_rows:] == -42)
        assert torch.all(lse[active_q_rows:] == 42)

    for length, q_length in cases:
        kv_length.fill_(length)
        cu_q[1:].copy_(following_q_offsets + q_length)
        active_q_rows = q_length + (cu_q.numel() - 2) * max_q_length
        k.view(torch.uint8).copy_(clean_k)
        v.view(torch.uint8).copy_(clean_v)
        replay()
        clean_o = o[:q_length].view(torch.int16).clone()
        clean_lse = lse[:q_length].view(torch.int32).clone()
        assert torch.isfinite(o[:q_length]).all(), f"clean Q={q_length}, KV={length}"
        if is_causal and length < q_length:
            assert torch.count_nonzero(o[: q_length - length]) == 0
            assert torch.isneginf(lse[: q_length - length]).all()
        if length == 0:
            assert torch.count_nonzero(o[:q_length]) == 0
            assert torch.isneginf(lse[:q_length]).all()

        replay()
        torch.testing.assert_close(o[:q_length].view(torch.int16), clean_o)
        torch.testing.assert_close(lse[:q_length].view(torch.int32), clean_lse)

        tail = logical_rows >= length
        for operand in ("k", "v", "kv"):
            expected_k = (
                torch.where(tail, poison, clean_k) if "k" in operand else clean_k
            )
            expected_v = (
                torch.where(tail, poison, clean_v) if "v" in operand else clean_v
            )
            k.view(torch.uint8).copy_(expected_k)
            v.view(torch.uint8).copy_(expected_v)
            replay()
            context = f"Q={q_length}, KV={length}, poisoned={operand}"
            torch.testing.assert_close(
                o[:q_length].view(torch.int16), clean_o, msg=context
            )
            torch.testing.assert_close(
                lse[:q_length].view(torch.int32), clean_lse, msg=context
            )
            torch.testing.assert_close(k.view(torch.uint8), expected_k, msg=context)
            torch.testing.assert_close(v.view(torch.uint8), expected_v, msg=context)

    q_length = max_q_length
    cu_q[1:].copy_(following_q_offsets + q_length)
    active_q_rows = q_length + (cu_q.numel() - 2) * max_q_length
    length = kv_tile + q_length
    kv_length.fill_(length)
    live_row = logical_rows == length - 1
    for operand in ("k", "v", "kv"):
        expected_k = torch.where(live_row, 0x7F, clean_k) if "k" in operand else clean_k
        expected_v = torch.where(live_row, 0x7F, clean_v) if "v" in operand else clean_v
        k.view(torch.uint8).copy_(expected_k)
        v.view(torch.uint8).copy_(expected_v)
        replay()
        assert torch.isnan(o[q_length - 1]).all(), f"live NaN in {operand}"
        if operand == "v":
            # This live row remains a V operand even for Q rows that mask its score.
            assert torch.isnan(o[:q_length]).all()
        torch.testing.assert_close(k.view(torch.uint8), expected_k)
        torch.testing.assert_close(v.view(torch.uint8), expected_v)


@pytest.mark.parametrize(
    "head_dim,q_tile,kv_tile,page_size",
    [
        pytest.param(256, 128, 128, 128, id="d256-q128-k128-p128"),
        pytest.param(128, 128, 128, 64, id="d128-q128-k128-p64"),
        pytest.param(64, 64, 64, 32, id="d64-q64-k64-p32"),
        pytest.param(32, 64, 64, 16, id="d32-q64-k64-p16"),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["dense", "gate", "skip"])
def test_sm120_paged_unused_kv_graph(
    head_dim, q_tile, kv_tile, page_size, is_causal, layout, mode
):
    from cutlass import Float32, Int32

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
    )

    torch.manual_seed(0)
    q_length, num_qo_heads, num_kv_heads = 17, 24, 4
    capacity = 3 * kv_tile + page_size
    num_pages = capacity // page_size
    shape = (
        (num_pages, 2, num_kv_heads, page_size, head_dim)
        if layout == "HND"
        else (num_pages, 2, page_size, num_kv_heads, head_dim)
    )
    pool = _finite_fp8(shape)
    k, v = pool.unbind(dim=1)
    if layout == "NHD":
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    block_table = torch.arange(
        num_pages - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    logical_rows = (
        torch.arange(capacity, device="cuda")
        .reshape(num_pages, page_size)
        .flip(0)[:, None, :, None]
    )
    q = torch.randint(
        -2, 3, (q_tile, num_qo_heads, head_dim), device="cuda", dtype=torch.int32
    ).to(torch.float8_e4m3fn)
    o = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q_tile, num_qo_heads, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, q_length], device="cuda", dtype=torch.int32)
    seqlens = torch.tensor([kv_tile + 17], device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), 2.0, device="cuda") if mode == "skip" else None
    gate = (
        torch.randn(q_tile, num_qo_heads, 2 * head_dim, device="cuda", dtype=o.dtype)[
            ..., :head_dim
        ]
        if mode == "gate"
        else None
    )
    kernel = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        kv_tile=kv_tile,
        q_tile=q_tile,
        num_tokens_per_page=page_size,
        device=q.device,
        with_lse=True,
        balanced_scheduler=is_causal,
        enable_skip_softmax=threshold is not None,
        kv_layout=layout,
        with_output_gate=gate is not None,
    )

    def run():
        kernel(
            q,
            k,
            v,
            o,
            lse,
            Float32(math.log2(math.e) / math.sqrt(head_dim)),
            Float32(1.0),
            threshold,
            seqlens,
            cu_q,
            block_table,
            None,
            Int32(q_tile),
            False,
            *((gate,) if gate is not None else ()),
        )

    graph = _capture(run)
    _check_tail_replays(
        graph, k, v, o, lse, seqlens, cu_q, logical_rows, kv_tile, is_causal
    )
    _check_finite_reference(
        graph, q, k, v, o, lse, seqlens, cu_q, kv_tile, is_causal, threshold, gate
    )


@pytest.mark.parametrize(
    "head_dim,q_tile,kv_tile",
    [(256, 128, 128), (128, 128, 128), (64, 64, 64)],
)
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("skip_softmax", [False, True], ids=["dense", "skip"])
def test_sm120_ragged_unused_kv_graph(
    head_dim, q_tile, kv_tile, is_causal, skip_softmax
):
    if _capability != (12, 0):
        pytest.skip("The public ragged FMHA compiler requires SM120")

    from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_ragged_prefill

    torch.manual_seed(0)
    q_length, num_qo_heads, num_kv_heads = 17, 4, 2
    capacity = 3 * kv_tile
    q = torch.randint(
        -2, 3, (q_tile, num_qo_heads, head_dim), device="cuda", dtype=torch.int32
    ).to(torch.float8_e4m3fn)
    k = _finite_fp8((capacity, num_kv_heads, head_dim))
    v = _finite_fp8(k.shape)
    o = torch.empty_like(q, dtype=torch.float16)
    lse = torch.empty(q_tile, num_qo_heads, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, q_length, 2 * q_length], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_tile + 17, capacity], device="cuda", dtype=torch.int32)
    threshold = torch.full((2,), 2.0, device="cuda") if skip_softmax else None

    def run():
        sm120_fmha_fp8_ragged_prefill(
            q,
            k,
            v,
            o,
            cu_q,
            cu_k,
            max_seqlen_q=q_tile,
            is_causal=is_causal,
            q_tile=q_tile,
            kv_tile=kv_tile,
            lse=lse,
            skip_softmax_threshold=threshold,
            enable_pdl=True,
        )

    graph = _capture(run)
    # The second request's live K/V immediately follows the first request.
    logical_rows = torch.arange(capacity, device="cuda")[:, None, None]
    _check_tail_replays(
        graph, k, v, o, lse, cu_k[1:2], cu_q, logical_rows, kv_tile, is_causal
    )
    _check_finite_reference(
        graph, q, k, v, o, lse, cu_k[1:2], cu_q, kv_tile, is_causal, threshold
    )


@pytest.mark.parametrize(
    "head_dim,q_tile,kv_tile,page_size",
    [
        pytest.param(32, 64, 64, 16, id="d32-two-buffer"),
        pytest.param(64, 64, 64, 32, id="d64-two-buffer"),
        pytest.param(128, 128, 128, 64, id="d128-three-stage"),
        pytest.param(256, 128, 128, 128, id="d256-three-stage"),
    ],
)
def test_sm120_paged_causal_empty_cta_graph(head_dim, q_tile, kv_tile, page_size):
    from cutlass import Float32, Int32

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
    )
    from tests.attention.test_sm120_fmha import _tol

    num_qo_heads, num_kv_heads = 24, 4
    q_length = q_tile + 1
    q = torch.zeros(
        2 * q_tile, num_qo_heads, head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    pool = torch.zeros(
        1, 2, page_size, num_kv_heads, head_dim, device="cuda", dtype=q.dtype
    )
    k, v = (plane.permute(0, 2, 1, 3) for plane in pool.unbind(dim=1))
    v.fill_(16)
    v[:, :, 0].fill_(1)
    clean_k, clean_v = k.view(torch.uint8).clone(), v.view(torch.uint8).clone()
    o = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(1, kv_tile // page_size, device="cuda", dtype=torch.int32)
    kernel = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=True,
        kv_tile=kv_tile,
        q_tile=q_tile,
        num_tokens_per_page=page_size,
        device=q.device,
        with_lse=True,
        balanced_scheduler=True,
        kv_layout="NHD",
    )

    def run():
        kernel(
            q,
            k,
            v,
            o,
            lse,
            Float32(math.log2(math.e) / math.sqrt(head_dim)),
            Float32(1.0),
            None,
            seqlens,
            cu_q,
            block_table,
            None,
            Int32(q_length),
            False,
        )

    graph = _capture(run)
    # Expanding Q creates an entirely masked first CTA in the captured graph.
    cu_q[1].fill_(q_length)
    tail = (torch.arange(page_size, device="cuda") > 0)[None, None, :, None]
    for operand in ("finite", "k", "v", "kv"):
        expected_k = torch.where(tail, 0x7F, clean_k) if "k" in operand else clean_k
        expected_v = torch.where(tail, 0x7F, clean_v) if "v" in operand else clean_v
        k.view(torch.uint8).copy_(expected_k)
        v.view(torch.uint8).copy_(expected_v)
        for _ in range(4):
            o.fill_(-42)
            lse.fill_(42)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(o[:q_tile]) == 0, f"padding={operand}"
            assert torch.isneginf(lse[:q_tile]).all(), f"padding={operand}"
            torch.testing.assert_close(
                o[q_tile], torch.ones_like(o[q_tile]), **_tol(o.dtype)
            )
            torch.testing.assert_close(
                lse[q_tile], torch.zeros_like(lse[q_tile]), atol=2e-3, rtol=2e-3
            )
            assert torch.all(o[q_length:] == -42)
            assert torch.all(lse[q_length:] == 42)
            torch.testing.assert_close(k.view(torch.uint8), expected_k)
            torch.testing.assert_close(v.view(torch.uint8), expected_v)
