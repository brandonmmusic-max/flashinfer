"""Paged head256 decode preserves mutable metadata and all value columns."""

from dataclasses import replace

import pytest
import torch

from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged_decode._tuning import default_config
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("width", [1, 16])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split_budget", [1, 12, None])
@pytest.mark.parametrize("causal,window,sinks", [(True, -1, False), (False, 127, True)])
@torch.no_grad()
def test_head256_mutable_captured_decode(
    width, kv_dtype, split_budget, causal, window, sinks
):
    device = require_b12x()
    torch.manual_seed(1713)
    key = (torch.randn(71, 128, 4, 256, device=device) * 0.1).to(kv_dtype)
    value = (torch.randn(71, 128, 4, 256, device=device) * 0.1).to(kv_dtype)
    query = (torch.randn(2 * width, 24, 256, device=device) * 0.1).bfloat16()
    output = torch.empty_like(query)
    table = torch.arange(66, device=device, dtype=torch.int32).view(2, 33)
    lengths = torch.full((2,), 4112, device=device, dtype=torch.int32)
    cu = torch.arange(3, device=device, dtype=torch.int32) * width
    sink = torch.randn(24, device=device) if sinks else None
    k_scale = (
        torch.tensor([0.5], device=device) if kv_dtype == torch.float8_e4m3fn else None
    )
    v_scale = (
        torch.tensor([0.25], device=device) if kv_dtype == torch.float8_e4m3fn else None
    )
    args = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        attention_sink_bias=sink,
        k_descale=k_scale,
        v_descale=v_scale,
    )
    caps = paged_decode.Caps(
        device=device,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=128,
        max_batch=2,
        max_q_per_req=width,
        causal=causal,
        window_left=window,
        has_sinks=sinks,
        kv_dtype=kv_dtype,
    )
    initial = paged_decode.plan(caps)
    config = default_config(initial.query)
    if split_budget is not None:
        config = replace(config, split_kv=split_budget != 1, max_splits=split_budget)
    plan = paged_decode.plan(caps, override=config)
    scratch = None
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

        session.prepare((plan.request(name="head256", prepare_call=prepare),))
        binding = paged_decode.bind(plan, scratch=scratch, **args)
        paged_decode.run(binding)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            paged_decode.run(binding)
        session.freeze()
        for endpoint, counts in (
            (4112, (width, width)),
            (127, (width, 0)),
            (4097, (1, max(1, width - 3))),
            (0, (0, 0)),
        ):
            cu.copy_(
                torch.tensor(
                    [0, counts[0], sum(counts)], device=device, dtype=torch.int32
                )
            )
            lengths.fill_(endpoint)
            table.copy_(torch.randperm(71, device=device)[:66].reshape(2, 33))
            query.normal_(std=0.1)
            output.fill_(37)
            for storage in scratch:
                storage.fill_(float("nan") if storage.is_floating_point() else 255)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[name] == after[name]
            live = sum(counts)
            if live:
                expected, _ = paged_attention_reference(
                    query,
                    key,
                    value,
                    table,
                    lengths,
                    cu,
                    causal=causal,
                    window_left=window,
                    attention_sink_bias=sink,
                    k_descale=None if k_scale is None else k_scale.expand(2),
                    v_descale=None if v_scale is None else v_scale.expand(2),
                )
                torch.testing.assert_close(
                    output[:live], expected[:live], atol=0.02, rtol=0.02
                )
                cosine = torch.nn.functional.cosine_similarity(
                    output[:live].float().flatten(),
                    expected[:live].float().flatten(),
                    dim=0,
                ).item()
                assert cosine >= 0.9999, cosine
            assert torch.all(output[live:] == 37)
            saved = output.clone()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(saved, output)
        graph.reset()
