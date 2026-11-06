"""Noncausal FP8 draft attention with mutable single-request blocks through sixteen rows."""

import pytest
import torch
from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged_decode._tuning import PagedDecodeConfig
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("width", [8, 13, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_wide_draft_capture_live_lengths_pages_and_poison(width, causal, split):
    require_b12x()
    generator = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn(
        (width, 32, 128), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k = (torch.randn((19, 128, 8, 128), generator=generator, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn((19, 128, 8, 128), generator=generator, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    pages = torch.arange(17, dtype=torch.int32, device="cuda")[None, :]
    lengths = torch.tensor([2048 + width], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, width], dtype=torch.int32, device="cuda")
    scale = torch.ones(1, device="cuda")
    output = torch.empty_like(q)
    caps = paged_decode.Caps(
        device="cuda",
        num_q_heads=32,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        max_batch=1,
        max_q_per_req=width,
        kv_dtype=torch.float8_e4m3fn,
        window_left=2047,
        causal=causal,
    )
    config = PagedDecodeConfig(
        split_kv=split,
        max_splits=4 if split else 1,
        head_splits=4 if split else 1,
        stage_rows=16,
        num_stages=16,
        min_tiles_per_split=4,
    )
    declaration = paged_decode.plan(caps, override=config)
    args = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        k_descale=scale,
        v_descale=scale,
    )
    scratch = [
        torch.empty(s.shape, dtype=s.dtype, device="cuda")
        for s in declaration.scratch_specs()
    ]

    def prepare_call(state):
        binding = state.bind(plan=declaration, scratch=scratch, output=output, **args)
        return PreparedCall(run=lambda: state.run(binding), output=output)

    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result = session.prepare(
            (declaration.request(name="wide_draft", prepare_call=prepare_call),)
        )
        plan = result.plans["wide_draft"]
        binding = paged_decode.bind(plan, scratch=scratch, output=output, **args)
        paged_decode.run(binding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)
        for live, context in (
            (width, 2048),
            (1, 127),
            (width - 1, 129),
            (0, 0),
            (width, 2048),
        ):
            cu[1] = live
            lengths[0] = context + live
            pages.copy_(
                torch.arange(17, dtype=torch.int32, device="cuda").flip(0)[None, :]
            )
            output.fill_(float("nan"))
            for tensor in scratch:
                tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            if live:
                reference, _ = paged_attention_reference(
                    q[:live].contiguous(),
                    k,
                    v,
                    pages,
                    lengths,
                    cu,
                    k_descale=scale,
                    v_descale=scale,
                    causal=causal,
                    window_left=2047,
                )
                actual = output[:live].float()
                assert torch.isfinite(actual).all()
                assert (actual - reference.float()).abs().max().item() <= 2e-2
                assert (
                    torch.nn.functional.cosine_similarity(
                        actual.reshape(-1), reference.float().reshape(-1), dim=0
                    ).item()
                    >= 0.9999
                )
            assert torch.isnan(output[live:]).all()
        graph.reset()
        result.close()
