"""Prepared dense verification adapts within its captured split capacity."""

import pytest
import torch

from b12x.attention import paged
from b12x.attention.paged.reference import attention_reference
from b12x.attention.paged.schedule import PrefillSchedule, schedule_nbytes
from b12x.preparation import PreparationSession, PreparedCall, require_prepared

from ..conftest import require_b12x


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("batch", [2, 8])
@torch.no_grad()
def test_prepared_adaptive_verify_preserves_long_tail_and_empty_replay(external, batch):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(431)
    q = torch.randn((batch * 8, 24, 256), device=device, dtype=torch.bfloat16) * 0.2
    key = (torch.randn((batch * 512, 128, 4, 256), device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    value = (torch.randn_like(key.float()) * 0.2).to(key.dtype)
    output = torch.empty_like(q)
    table = torch.zeros((batch, 2048), dtype=torch.int32, device=device)
    table[:, :512] = torch.arange(
        batch * 512, dtype=torch.int32, device=device
    ).reshape(batch, 512)
    lengths = torch.tensor(
        [4104, 3969] * (batch // 2), dtype=torch.int32, device=device
    )
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * 8
    scale = torch.ones(1, device=device)
    arguments = dict(
        q=q,
        k_cache=key,
        v_cache=value,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        output=output,
        k_descale=scale,
        v_descale=scale,
    )
    caps = paged.Caps(
        device=device,
        mode="verify",
        dtype=q.dtype,
        kv_dtype=key.dtype,
        num_q_heads=24,
        num_kv_heads=4,
        head_dim_qk=256,
        head_dim_vo=256,
        page_size=128,
        max_total_q=batch * 8,
        max_batch=batch,
        max_page_table_width=2048,
        num_cache_pages=batch * 512,
        max_work_items=512,
        max_partial_rows=4096,
        use_cuda_graph=True,
        copy_runtime_metadata=False,
    )
    invocation = dict(
        paged.invocation_from_tensors(caps, **arguments),
        query_len=8,
        graph_ctas_per_sm=6,
        force_split_kv=True,
    )
    plan = paged.plan(caps, invocation=invocation)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:

        def prepare(state):
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch_plan.scratch_specs()
            ]
            binding = state.bind(scratch=scratch, **arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        session.prepare((plan.request(name="adaptive.verify", prepare_call=prepare),))
        state = require_prepared(plan, "attention.gqa", device)
        scratch = [
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in plan.scratch_specs()
        ]
        schedule = PrefillSchedule.from_plan(plan)
        storage = torch.empty(schedule_nbytes(caps), dtype=torch.uint8, device=device)
        binding = paged.bind(plan, scratch=scratch, **arguments)
        chunks = binding.scratch._prefill_graph_max_chunks_per_q_tile
        assert chunks > 1
        assert schedule.geometry[-1] is True

        def run():
            extra = {}
            if external:
                schedule.update(storage, lengths, cu)
                extra["external_schedule"] = schedule.bind(storage)
            paged.run(
                plan=plan,
                binding=paged.bind(plan, scratch=scratch, **arguments, **extra),
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for live_lengths in (
            (4104, 3969),
            (3969, 8193),
            (32769, 0),
            (65535, 129),
            (0, 0),
            (129, 4104),
        ):
            live_lengths = live_lengths * (batch // 2)
            counts = [8 if length else 0 for length in live_lengths]
            boundaries = [0]
            for count in counts:
                boundaries.append(boundaries[-1] + count)
            lengths.copy_(torch.tensor(live_lengths, dtype=torch.int32, device=device))
            cu.copy_(torch.tensor(boundaries, dtype=torch.int32, device=device))
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            chunk_pointer = (
                schedule.views(storage)["kv_chunk_size_ptr"]
                if external
                else binding.scratch.kv_chunk_size_ptr
            )
            pages = max(max((length + 127) // 128, 1) for length in live_lengths)
            expected_chunk = ((pages + chunks - 1) // chunks) * 128
            assert int(chunk_pointer.item()) == expected_chunk
            for row, length in enumerate(live_lengths):
                if not length:
                    continue
                start, end = boundaries[row : row + 2]
                reference = attention_reference(
                    q[start:end],
                    key[row * 512 : (row + 1) * 512].flatten(0, 1)[:length].to(q.dtype),
                    value[row * 512 : (row + 1) * 512]
                    .flatten(0, 1)[:length]
                    .to(q.dtype),
                )[0]
                torch.testing.assert_close(
                    output[start:end].float(), reference.float(), rtol=0.05, atol=0.02
                )
            assert torch.isnan(output[boundaries[-1] :]).all()
        graph.reset()
