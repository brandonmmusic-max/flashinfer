"""Startup races of one runtime share eviction, timing and fixture storage across replays."""

from datetime import timedelta
from dataclasses import replace
import gc
import os
import weakref
from pathlib import Path

import pytest

SIZES = (4112, 163840, 327680, 1310720)
CAPACITY_BYTES = 2621440
PROFILES = (("payload_rsag", 42), ("bulk_push", 8), ("pull", 7))


def _exercise(rank, external_first):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import (
        CollectiveConfig,
        joint_startup_spec,
        prepare_joint_startup_tuner,
        StartupTuningStorage,
        DcpAllToAll,
        query_from_runtime,
        plan as native_plan,
    )
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import (
        CollectiveRequirement,
        PreparationSession,
        PreparedCall,
        FrozenMapping,
    )

    device = torch.device("cuda", rank)
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        joint=True,
        max_size=CAPACITY_BYTES,
        eager_buffer_bytes=CAPACITY_BYTES,
    )
    manager.prepare_channels(("startup-shared",))
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    gather_runtime = DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=256,
        total_heads=4,
        head_dim=8,
        stream_affine=False,
    )
    storage = StartupTuningStorage(device)
    resident = torch.ones(
        storage.eviction_bytes // 4, dtype=torch.float32, device=device
    )
    if external_first:
        before = torch.cuda.memory_allocated(device)
        storage.use_eviction(resident)
        assert torch.cuda.memory_allocated(device) == before
        assert storage._eviction_sum is None
    plans, requests, graphs, gather_plans = [], [], [], []
    try:
        for index, nbytes in enumerate((*SIZES, CAPACITY_BYTES)):
            source = torch.full(
                (nbytes // 2,), rank + 1, dtype=torch.bfloat16, device=device
            )
            out = source if index % 2 else torch.empty_like(source)
            plan = manager.plan(
                source,
                out=out,
                joint_only=True,
                joint_cluster_size=1,
                override=CollectiveConfig(
                    implementation="joint_bandwidth",
                    threads=512,
                    blocks=64,
                    submode="payload_rsag",
                    active_blocks=1,
                ),
            )
            name = f"startup-shared-{nbytes}"
            requests.append(
                plan.request(
                    name=name,
                    collective=CollectiveRequirement(name, tuple(range(4))),
                    prepare_call=lambda state, source=source, out=out: (
                        _prepare_plain_call(state, inp=source, out=out)
                    ),
                )
            )
            plans.append(plan)
        for rows in (3, 128, 256):
            source = torch.empty((rows, 1, 8), dtype=torch.bfloat16, device=device)
            out = torch.empty((rows, 4, 8), dtype=torch.bfloat16, device=device)
            query = query_from_runtime(
                gather_runtime,
                surface="DcpAllToAll.all_gather_heads",
                call={
                    "local_input": source,
                    "out": out,
                    "gather_transport": "joint",
                    "threads": 128,
                    "block_limit": 16,
                    "submode": "barrier",
                    "active_blocks": 2,
                },
            )
            query = replace(
                query,
                call=FrozenMapping(
                    {
                        **query.call,
                        "gather_transport": "joint",
                        "submode": "barrier",
                        "active_blocks": 2,
                    }
                ),
            )
            gather_plan = native_plan(query, runtime=gather_runtime)
            name = f"startup-shared-gather-{rows}"
            requests.append(
                gather_plan.request(
                    name=name,
                    collective=CollectiveRequirement(name, tuple(range(4))),
                    prepare_call=lambda state, source=source, out=out: PreparedCall(
                        run=lambda: state.run_selected(source, out),
                        owners=(source, out),
                    ),
                )
            )
            gather_plans.append(gather_plan)
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        producers = [
            prepare_joint_startup_tuner(
                (joint_startup_spec(plan, profiles=PROFILES),),
                cache_policy="cold",
                storage=storage,
            )
            for plan in plans[:-1]
        ]
        first = producers[0]
        assert {p.eviction.data_ptr() for p in producers} == {first.eviction.data_ptr()}
        assert {p.shared.local_ptr for p in producers} == {first.shared.local_ptr}
        if not external_first:
            assert int(torch.count_nonzero(first.eviction)) == 0
            previous = weakref.ref(storage._eviction)
            storage.use_eviction(resident)
            assert previous() is None
        assert {p.eviction.data_ptr() for p in producers} == {resident.data_ptr()}
        # Invalid sources must neither copy nor replace the registered input.
        invalid_sources = (
            (torch.empty(2048, dtype=torch.float32, device=device), "smaller"),
            (
                torch.empty(
                    storage.eviction_bytes // 2, dtype=torch.float32, device=device
                )[::2],
                "contiguous",
            ),
            (
                torch.empty(
                    storage.eviction_bytes + 1, dtype=torch.uint8, device=device
                )[1:],
                "aligned",
            ),
        )
        for source, message in invalid_sources:
            before = torch.cuda.memory_allocated(device)
            with pytest.raises(ValueError, match=message):
                storage.use_eviction(source)
            assert torch.cuda.memory_allocated(device) == before
        del invalid_sources, source
        assert (
            len({p.inputs.source.untyped_storage().data_ptr() for p in producers}) == 1
        )
        assert (
            len(
                {
                    p.inputs.reference.expected[0].untyped_storage().data_ptr()
                    for p in producers
                }
            )
            == 1
        )
        for plan in gather_plans[:-1]:
            producers.append(
                prepare_joint_startup_tuner(
                    (
                        joint_startup_spec(
                            plan, profiles=(("barrier", 2), ("word_exchange", 4))
                        ),
                    ),
                    cache_policy="cold",
                    storage=storage,
                )
            )
        assert (
            len({p.inputs.source.untyped_storage().data_ptr() for p in producers}) == 1
        )
        assert len(storage.fixtures.banks) == 2
        # Omitted storage preserves one cold eviction allocation per runtime.
        defaults = [
            prepare_joint_startup_tuner(
                (joint_startup_spec(plans[0], profiles=PROFILES),), cache_policy="cold"
            )
            for _ in range(2)
        ]
        assert defaults[0].storage is defaults[1].storage
        assert (
            defaults[0].storage
            is plans[0].prepared.state.runtime._joint_startup_storage
        )
        assert defaults[0].storage is not storage
        assert (
            defaults[0].inputs.source.data_ptr() == defaults[1].inputs.source.data_ptr()
        )
        assert defaults[0].eviction.data_ptr() == defaults[1].eviction.data_ptr()
        assert defaults[0].eviction.data_ptr() != resident.data_ptr()
        producers.extend(defaults)
        for producer in producers:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                producer.run()
            graphs.append(graph)
        with pytest.raises(RuntimeError, match="after capture"):
            storage.use_eviction(resident)
        captured = [
            (p.inputs.source.data_ptr(), p.inputs.reference.seeds[0].data_ptr())
            for p in producers
        ]
        late = [
            prepare_joint_startup_tuner(
                (joint_startup_spec(plans[-1], profiles=PROFILES),), storage=storage
            )
        ]
        late.append(
            prepare_joint_startup_tuner(
                (
                    joint_startup_spec(
                        gather_plans[-1],
                        profiles=(("barrier", 2), ("word_exchange", 4)),
                    ),
                ),
                storage=storage,
            )
        )
        assert captured == [
            (p.inputs.source.data_ptr(), p.inputs.reference.seeds[0].data_ptr())
            for p in producers
        ]
        shared_pointer = first.inputs.source.untyped_storage().data_ptr()
        assert all(
            p.inputs.source.untyped_storage().data_ptr() != shared_pointer for p in late
        )
        for producer in late:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                producer.run()
            graphs.append(graph)
        producers.extend(late)
        session.freeze()

        def replay(producer, graph):
            producer.status.fill_(-1)
            torch.cuda.synchronize()
            dist.barrier()
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats(device)
            for counter in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[counter] == after[counter]
            return producer.status.item()

        for round_index in range(3):
            for producer, graph in zip(producers, graphs):
                assert replay(producer, graph) == 0, round_index
                winner = int(producer.specs[0].selector.item())
                allowed = {
                    (mode << 16) | active for mode, active in producer.specs[0].profiles
                }
                assert winner in allowed
                winners = [None] * 4
                dist.all_gather_object(winners, winner)
                assert len(set(winners)) == 1
        # A wrong collective result keeps the race from publishing a profile.
        producer, graph = producers[1], graphs[1]
        expected = producer.inputs.reference.expected[0]
        saved = expected.clone()
        expected.view(-1)[0] += 1
        assert replay(producer, graph) == -1
        expected.copy_(saved)
        assert replay(producer, graph) == 0
        assert replay(producers[2], graphs[2]) == 0
        assert bool(torch.all(resident == 1))
        return tuple(
            weakref.ref(item)
            for item in (
                storage,
                storage.fixtures,
                resident,
                defaults[0].storage,
                defaults[0].storage.fixtures,
            )
        )
    finally:
        for graph in graphs:
            graph.reset()
        session.close()
        manager.close()
        gather_runtime.close()


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=240),
    )
    try:
        retained = []
        allocated = []
        for external_first in (True, False):
            retained.extend(_exercise(rank, external_first))
            gc.collect()
            torch.cuda.synchronize()
            assert all(reference() is None for reference in retained)
            allocated.append(torch.cuda.memory_allocated(rank))
        assert allocated[1] == allocated[0], allocated
    finally:
        gc.collect()
        dist.destroy_process_group()
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_CLUSTER_TEST") != "1",
    reason="requires an assigned TP4 SM120/SM121 group",
)
def test_startup_races_share_eviction_timing_and_fixture_storage(tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 4
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=True)
