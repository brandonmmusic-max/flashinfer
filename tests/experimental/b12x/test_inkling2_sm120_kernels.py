# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 unit tests for b12x kernels supporting looped Inkling 2."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

CAPABILITY = torch.cuda.get_device_capability()
if CAPABILITY not in ((12, 0), (12, 1)):
    pytest.skip(f"requires SM120/SM121, found SM{CAPABILITY[0]}{CAPABILITY[1]}", allow_module_level=True)


def test_small_k16_gemm_and_tc_gemv_bias():
    from b12x.gemm import bf16_gemv
    from b12x.preparation import PreparationSession, PreparedCall

    torch.manual_seed(1)
    x = torch.randn((32, 16), dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn((1024, 16), dtype=torch.bfloat16, device="cuda") * 0.5
    out = torch.empty((32, 1024), dtype=torch.bfloat16, device="cuda")

    x2 = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.2
    w2 = torch.randn((256, 1024), dtype=torch.bfloat16, device="cuda") * 0.2
    b2 = torch.randn((256,), dtype=torch.bfloat16, device="cuda") * 0.1
    out2 = torch.empty((4, 256), dtype=torch.bfloat16, device="cuda")

    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        p1 = bf16_gemv.plan(
            bf16_gemv.query_from_call(x, w, out=out),
            override=bf16_gemv.GemvConfig(backend="tc", split_k=1),
        )
        p2 = bf16_gemv.plan(
            bf16_gemv.query_from_call(x2, w2, bias=b2, out=out2),
            override=bf16_gemv.GemvConfig(backend="tc", split_k=2),
        )
        session.prepare(
            (
                p1.request(
                    name="small_k16",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(x, w, out=out),
                    ),
                ),
                p2.request(
                    name="tc_gemv_bias",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(x2, w2, out=out2, bias=b2),
                    ),
                ),
            )
        )
        bf16_gemv.mm(x, w, plan=p1, out=out)
        bf16_gemv.mm(x2, w2, plan=p2, bias=b2, out=out2)

    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    assert (out.float() - ref.float()).abs().max().item() < 5e-2
    ref2 = (x2.float() @ w2.float().T + b2.float()).to(torch.bfloat16)
    assert (out2.float() - ref2.float()).abs().max().item() < 5e-2


def test_splitk_router_gemv_matches_reference():
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from b12x._lib.compiler import compile as b12x_compile, run_compiled
    from b12x._lib.utils.device import current_cuda_stream
    from b12x._lib.utils.pointers import make_ptr
    from b12x.moe._inkling2_splitk_router import SplitKRouterGemvKernel

    torch.manual_seed(2)
    tokens, k, n, splits = 4, 2304, 128, 6
    x = torch.randn((tokens, k), dtype=torch.bfloat16, device="cuda") * 0.25
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.25
    partials = torch.empty((splits, tokens, n), dtype=torch.float32, device="cuda")

    kernel = SplitKRouterGemvKernel(n, k, splits)
    compiled = b12x_compile(
        kernel,
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(tokens),
        Int64(k),
        Int64(k),
        current_cuda_stream(),
    )
    run_compiled(
        compiled,
        (
            make_ptr(BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, partials.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            Int32(tokens),
            Int64(x.stride(0)),
            Int64(w.stride(0)),
            current_cuda_stream(),
        ),
    )
    out = partials.sum(dim=0)
    ref = x.float() @ w.float().T
    assert (out - ref).abs().max().item() < 5e-2


def test_inkling2_residual_mix_matches_reference():
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from b12x._lib.compiler import compile as b12x_compile, run_compiled
    from b12x._lib.utils.device import current_cuda_stream
    from b12x._lib.utils.pointers import make_ptr
    from b12x.norm.inkling2_residual import VecResidualMixKernel

    torch.manual_seed(5)
    tokens, sources, hidden = 4, 4, 2304
    eps = 1e-6
    blocks = torch.randn((sources - 1, tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.5
    last = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.5
    addend = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.25
    queries = torch.randn((1, hidden), dtype=torch.float32, device="cuda") * 0.5
    weight = torch.randn((1, hidden), dtype=torch.float32, device="cuda") * 0.5 + 1.0
    out = torch.empty((1, tokens, hidden), dtype=torch.bfloat16, device="cuda")

    kernel = VecResidualMixKernel(hidden, sources, has_addend=True, normalize=True, eps=eps, norm_eps=eps)
    compiled = b12x_compile(
        kernel,
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int64(blocks.stride(0)),
        Int64(blocks.stride(1)),
        Int64(last.stride(0)),
        Int64(addend.stride(0)),
        Int64(out.stride(0)),
        Int64(out.stride(1)),
        Int32(tokens),
        Int32(1),
        current_cuda_stream(),
    )
    run_compiled(
        compiled,
        (
            make_ptr(BFloat16, blocks.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, last.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, addend.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, queries.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            Int64(blocks.stride(0)),
            Int64(blocks.stride(1)),
            Int64(last.stride(0)),
            Int64(addend.stride(0)),
            Int64(out.stride(0)),
            Int64(out.stride(1)),
            Int32(tokens),
            Int32(1),
            current_cuda_stream(),
        ),
    )

    stack = torch.cat([blocks.float(), (last.float() + addend.float()).unsqueeze(0)], dim=0)
    rstd = torch.rsqrt(stack.square().mean(dim=-1, keepdim=True) + eps)
    logits = torch.einsum("sth,h->st", stack * rstd, queries[0])
    probs = torch.softmax(logits, dim=0)
    mixed = torch.einsum("st,sth->th", probs, stack * rstd)
    out_rstd = torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + eps)
    ref = (mixed * out_rstd * weight[0]).to(torch.bfloat16)
    assert (out[0].float() - ref.float()).abs().max().item() < 5e-2


def test_relative_prefill_and_decode_attention_match_reference():
    from b12x.attention import relative_decode, relative_prefill
    from b12x.preparation import PreparationSession, PreparedCall

    torch.manual_seed(3)
    heads, kv_heads, dim, window = 32, 4, 128, 1024
    scale = dim ** -0.5

    seq = 128
    q = torch.randn((seq, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    k = torch.randn((seq, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    v = torch.randn((seq, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    rel = torch.randn((seq, heads, window), dtype=torch.bfloat16, device="cuda") * 0.05
    gate = torch.randn((seq, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    sink = torch.randn((seq, heads), dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.zeros_like(q)

    k_fp8 = k.to(torch.float8_e4m3fn)
    v_fp8 = v.to(torch.float8_e4m3fn)
    k_pages = k_fp8.view(1, seq, kv_heads, dim).contiguous()
    v_pages = v_fp8.view(1, seq, kv_heads, dim).contiguous()
    page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([seq], dtype=torch.int32, device="cuda")

    ring_k = torch.zeros((2, window, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda")
    ring_v = torch.zeros((2, window, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda")
    ring_k[1, :seq] = k_fp8
    ring_v[1, :seq] = v_fp8
    qd = torch.randn((1, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    kd = torch.randn((1, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    vd = torch.randn((1, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    reld = torch.randn((1, heads, window), dtype=torch.bfloat16, device="cuda") * 0.05
    gated_d = torch.randn((1, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    sink_d = torch.randn((1, heads), dtype=torch.bfloat16, device="cuda") * 0.1
    out_d = torch.zeros_like(qd)
    state_rows = torch.tensor([1], dtype=torch.int64, device="cuda")
    starts = torch.tensor([seq], dtype=torch.int64, device="cuda")
    d_cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        p_plan = relative_prefill.plan(
            relative_prefill.Caps(
                device="cuda",
                num_q_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=dim,
                page_size=seq,
                rel_extent=window,
                max_batch=1,
                max_total_q=seq,
                window_left=-1,
            )
        )
        d_plan = relative_decode.plan(
            relative_decode.Caps(
                device="cuda",
                num_q_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=dim,
                is_window=True,
                commit_ring=True,
                page_or_window=window,
                window_left=window - 1,
                rel_extent=window,
                max_batch=1,
                max_q_per_req=1,
                max_total_q=1,
            )
        )

        def _prep_prefill(state):
            from b12x.attention.relative_prefill._preparation import scratch_specs

            sc = [
                torch.empty(s.shape, dtype=s.dtype, device=s.device)
                for s in scratch_specs(state.query, state.config, q.device)
            ]
            b = state.bind(
                plan=p_plan,
                scratch=sc,
                q=q,
                k_cache=k_pages,
                v_cache=v_pages,
                page_table=page_table,
                lengths=lengths,
                cu_seqlens_q=cu,
                bias=rel,
                gate=gate,
                sink=sink,
                gated=out,
                softmax_scale=scale,
            )
            return PreparedCall(run=lambda: state.run(b), output=out, owners=(*sc, b))

        def _prep_decode(state):
            from b12x.attention.relative_decode._preparation import scratch_specs

            sc = [
                torch.empty(s.shape, dtype=s.dtype, device=s.device)
                for s in scratch_specs(state.query, state.config, qd.device)
            ]
            b = state.bind(
                plan=d_plan,
                scratch=sc,
                q=qd,
                new_k=kd,
                new_v=vd,
                k_cache=ring_k,
                v_cache=ring_v,
                starts=starts,
                cu_seqlens_q=d_cu,
                bias=reld,
                gate=gated_d,
                sink=sink_d,
                gated=out_d,
                state_rows=state_rows,
                softmax_scale=scale,
            )
            return PreparedCall(run=lambda: state.run(b), output=out_d, owners=(*sc, b))

        session.prepare(
            (
                p_plan.request(name="relative_prefill", prepare_call=_prep_prefill),
                d_plan.request(name="relative_decode", prepare_call=_prep_decode),
            )
        )
        p_scratch = [
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in p_plan.scratch_specs()
        ]
        d_scratch = [
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in d_plan.scratch_specs()
        ]
        p_bind = relative_prefill.bind(
            p_plan,
            scratch=p_scratch,
            q=q,
            k_cache=k_pages,
            v_cache=v_pages,
            page_table=page_table,
            lengths=lengths,
            cu_seqlens_q=cu,
            bias=rel,
            gate=gate,
            sink=sink,
            gated=out,
            softmax_scale=scale,
        )
        d_bind = relative_decode.bind(
            d_plan,
            scratch=d_scratch,
            q=qd,
            new_k=kd,
            new_v=vd,
            k_cache=ring_k,
            v_cache=ring_v,
            starts=starts,
            cu_seqlens_q=d_cu,
            bias=reld,
            gate=gated_d,
            sink=sink_d,
            gated=out_d,
            state_rows=state_rows,
            softmax_scale=scale,
        )
        relative_prefill.run(p_bind)
        relative_decode.run(d_bind)

    kf = k_fp8.float().repeat_interleave(heads // kv_heads, dim=1)
    vf = v_fp8.float().repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("ihd,jhd->hij", q.float(), kf) * scale
    rows = torch.arange(seq, device="cuda")
    dist = rows[:, None] - rows[None, :]
    inside = (dist >= 0) & (dist < window)
    bias = rel.float().permute(1, 0, 2).gather(2, dist.clamp(0, window - 1).expand(heads, seq, seq))
    scores = scores + torch.where(inside, bias, 0.0)
    scores = scores.masked_fill(dist < 0, float("-inf"))
    logits = torch.cat([scores, sink.float().t().unsqueeze(-1)], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    ref = (torch.einsum("hij,jhd->ihd", probs, vf) * torch.sigmoid(gate.float())).to(torch.bfloat16)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    assert torch.isfinite(out_d).all()
    assert torch.equal(ring_k[1, seq].view(torch.uint8), kd[0].to(torch.float8_e4m3fn).view(torch.uint8))


@pytest.mark.parametrize("window_left", [-1, 1023])
def test_relative_prefill_causal_invariance_across_16_token_boundaries(window_left):
    """Partial final 16-row warps must match full warps bitwise for every prefix length."""
    from b12x.attention import relative_prefill
    from b12x.attention.relative_prefill._preparation import scratch_specs
    from b12x.preparation import PreparationSession, PreparedCall

    torch.manual_seed(7)
    heads, kv_heads, dim, page_size, rel_extent, max_len = 24, 8, 128, 128, 1024, 128
    scale = dim ** -0.5
    window_size = (window_left + 1) if window_left >= 0 else 0

    q_full = torch.randn((max_len, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    k_full = (torch.randn((max_len, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    v_full = (torch.randn((max_len, kv_heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    bias_full = torch.randn((max_len, heads, rel_extent), dtype=torch.bfloat16, device="cuda") * 1.5
    gate_full = torch.randn((max_len, heads, dim), dtype=torch.bfloat16, device="cuda") * 0.25
    sink_full = torch.randn((max_len, heads), dtype=torch.bfloat16, device="cuda") * 0.1

    if window_left < 0:
        k_cache = torch.zeros((2, page_size, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda")
        v_cache = torch.zeros((2, page_size, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda")
        page_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    else:
        k_cache = torch.zeros(
            (1, window_size + max_len, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda"
        )
        v_cache = torch.zeros(
            (1, window_size + max_len, kv_heads, dim), dtype=torch.float8_e4m3fn, device="cuda"
        )
        page_table = torch.empty((1, 1), dtype=torch.int32, device="cuda")

    q_buf = torch.zeros_like(q_full)
    bias_buf = torch.zeros_like(bias_full)
    gate_buf = torch.zeros_like(gate_full)
    sink_buf = torch.zeros_like(sink_full)
    out_buf = torch.zeros_like(q_full)
    cu_buf = torch.tensor([0, max_len], dtype=torch.int32, device="cuda")
    len_buf = torch.tensor([max_len], dtype=torch.int32, device="cuda")

    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        p_plan = relative_prefill.plan(
            relative_prefill.Caps(
                device="cuda",
                num_q_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=dim,
                page_size=page_size,
                rel_extent=rel_extent,
                max_batch=1,
                max_total_q=max_len,
                window_left=window_left,
            )
        )

        def _prep(state):
            sc = [
                torch.empty(s.shape, dtype=s.dtype, device=s.device)
                for s in scratch_specs(state.query, state.config, q_buf.device)
            ]
            b = state.bind(
                plan=p_plan,
                scratch=sc,
                q=q_buf,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                lengths=len_buf,
                cu_seqlens_q=cu_buf,
                bias=bias_buf,
                gate=gate_buf,
                sink=sink_buf,
                gated=out_buf,
                softmax_scale=scale,
            )
            return PreparedCall(run=lambda: state.run(b), output=out_buf, owners=(*sc, b))

        session.prepare((p_plan.request(name=f"causal_{window_left}", prepare_call=_prep),))
        sc = [
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in p_plan.scratch_specs()
        ]
        binding = relative_prefill.bind(
            p_plan,
            scratch=sc,
            q=q_buf,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            lengths=len_buf,
            cu_seqlens_q=cu_buf,
            bias=bias_buf,
            gate=gate_buf,
            sink=sink_buf,
            gated=out_buf,
            softmax_scale=scale,
        )

        outputs = {}
        for seq_len in (63, 64, 79, 80, 95, 96, 106, 111, 112, 118, 127, 128):
            q_buf.zero_()
            q_buf[:seq_len] = q_full[:seq_len]
            bias_buf.zero_()
            bias_buf[:seq_len] = bias_full[:seq_len]
            gate_buf.zero_()
            gate_buf[:seq_len] = gate_full[:seq_len]
            sink_buf.zero_()
            sink_buf[:seq_len] = sink_full[:seq_len]
            k_cache.zero_()
            v_cache.zero_()
            if window_left < 0:
                k_cache[0, :seq_len] = k_full[:seq_len]
                v_cache[0, :seq_len] = v_full[:seq_len]
            else:
                k_cache[0, window_size : window_size + seq_len] = k_full[:seq_len]
                v_cache[0, window_size : window_size + seq_len] = v_full[:seq_len]
            cu_buf[0] = 0
            cu_buf[1] = seq_len
            len_buf[0] = seq_len
            out_buf.zero_()
            relative_prefill.run(binding)
            outputs[seq_len] = out_buf[:seq_len].clone()

    for short_len, long_len in ((63, 64), (79, 80), (95, 96), (106, 118), (111, 112), (127, 128)):
        assert torch.equal(outputs[short_len], outputs[long_len][:short_len]), (
            f"window_left={window_left}: causal drift between L={short_len} and L={long_len}"
        )
