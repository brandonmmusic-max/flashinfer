"""Production cluster plans preserve tails, profile transitions and graph ownership."""

import ctypes
from datetime import timedelta
import gc
import os
from pathlib import Path

import pytest

from .test_pcie_joint_gpu import (
    _operands,
    _poison_scratch,
    _reference,
    _assert_epoch_wrapped,
)

SIZES = (
    16,
    48,
    4112,
    162784,
    163840,
    169984,
    325568,
    327680,
    337920,
    339952,
    1310720,
    2293760,
)
CAPACITY_BYTES = 2293760
PROFILES = {
    "joint_latency": (
        ("payload_rsag_phased", 8),
        ("payload_rsag", 64),
        ("pull", 2),
        ("pull", 8),
        ("full_phased", 8),
        ("full", 25),
        ("pull", 7),
    ),
    "joint_bandwidth": (
        ("payload_rsag_phased", 8),
        ("bulk_pull", 11),
        ("payload_rsag", 42),
        ("payload_rsag", 64),
        ("bulk_push", 8),
        ("pull", 7),
    ),
}


def _geometry(graph, cluster):
    import cuda.bindings.driver as cuda

    raw = cuda.CUgraph(graph.raw_cuda_graph())
    status, _, count = cuda.cuGraphGetNodes(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS and count == 1
    status, nodes, _ = cuda.cuGraphGetNodes(raw, count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, params = cuda.cuGraphKernelNodeGetParams(nodes[0])
    assert status == cuda.CUresult.CUDA_SUCCESS
    assert (params.gridDimX, params.gridDimY, params.gridDimZ) == (64, 1, 1)
    assert (params.blockDimX, params.blockDimY, params.blockDimZ) == (512, 1, 1)
    status, attribute = cuda.cuGraphKernelNodeGetAttribute(
        nodes[0], cuda.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
    )
    assert status == cuda.CUresult.CUDA_SUCCESS and attribute.cooperative == 0
    status, attribute = cuda.cuGraphKernelNodeGetAttribute(
        nodes[0], cuda.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    )
    assert status == cuda.CUresult.CUDA_SUCCESS
    actual = (attribute.clusterDim.x, attribute.clusterDim.y, attribute.clusterDim.z)
    assert actual == (8, 1, 1) if cluster else actual in ((0, 0, 0), (1, 1, 1))


def _read_signal(state):
    import torch

    data = (ctypes.c_ubyte * state.workspace.signal_bytes)()
    stream = torch.cuda.current_stream()
    state.runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(data),
        state.workspace.shared.local_ptr,
        len(data),
        stream.cuda_stream,
    )
    stream.synchronize()
    return bytes(data)


def _exercise(rank, family, alias):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import (
        CollectiveConfig,
        joint_startup_spec,
        prepare_joint_startup_tuner,
    )
    from b12x.comm.pcie._joint_profile import JOINT_MODES
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import CollectiveRequirement, PreparationSession

    device = torch.device("cuda", rank)
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        joint=True,
        max_size=CAPACITY_BYTES,
        eager_buffer_bytes=CAPACITY_BYTES,
    )
    manager.prepare_channels(("cluster-profile-transitions",))
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    cases, graphs, startup_graph = [], [], None
    requests = []
    try:
        for nbytes in SIZES:
            source = torch.full(
                (nbytes // 2,), rank + 1, dtype=torch.bfloat16, device=device
            )
            out = source if alias else torch.empty_like(source)
            plans = {}
            for clustered in (False, True):
                mode = (
                    "payload_rsag_phased"
                    if clustered
                    else ("full" if family == "joint_latency" else "payload_rsag")
                )
                plan = manager.plan(
                    source,
                    out=out,
                    joint_only=True,
                    joint_cluster_size=8 if clustered else 1,
                    override=CollectiveConfig(
                        implementation=family,
                        threads=512,
                        blocks=64,
                        submode=mode,
                        active_blocks=8 if clustered else 1,
                    ),
                )
                name = f"{family}-{nbytes}-{alias}-{clustered}"
                requests.append(
                    plan.request(
                        name=name,
                        collective=CollectiveRequirement(name, tuple(range(4))),
                        prepare_call=lambda state, source=source, out=out: (
                            _prepare_plain_call(state, inp=source, out=out)
                        ),
                    )
                )
                plans[clustered] = plan
            cases.append((nbytes, source, out, plans))
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        for nbytes, source, out, plans in cases:
            for clustered, plan in plans.items():
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                dist.barrier()
                with session.capture(), manager.capture(), torch.cuda.graph(graph):
                    manager.all_reduce(source, out=out, plan=plan)
                graph.instantiate()
                _geometry(graph, clustered)
                graphs.append(
                    (nbytes, clustered, graph, source, out, plan.prepared.state)
                )
                assert (
                    plan.prepared.state.workspace.capacity_packs == CAPACITY_BYTES // 16
                )
        race_case = next(case for case in cases if case[0] == 327680)
        race_plan = race_case[3][True]
        race_profiles = PROFILES[family][: 4 if family == "joint_latency" else 3]
        spec = joint_startup_spec(race_plan, profiles=race_profiles)
        producer = prepare_joint_startup_tuner((spec,), cache_policy="warm")
        startup_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with session.capture(), manager.capture(), torch.cuda.graph(startup_graph):
            producer.run()
        session.freeze()
        for nbytes, clustered, graph, source, out, state in graphs:
            profiles = (
                PROFILES[family]
                if clustered
                else tuple(p for p in PROFILES[family] if not p[0].endswith("_phased"))
            )
            for position, (mode, active) in enumerate(profiles):
                state.select_profile(mode, active)
                rollover = position == 0
                _poison_scratch(state, rollover=rollover)
                values = _operands(4, source.numel(), position + SIZES.index(nbytes))
                expected = _reference(values, rank, mode.removesuffix("_phased"))
                for _ in range(3):
                    source.copy_(values[rank])
                    if not alias:
                        out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for counter in (
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ):
                        assert before[counter] == after[counter]
                    torch.testing.assert_close(
                        out.cpu(), expected, rtol=0, atol=0, equal_nan=True
                    )
                    if not alias:
                        torch.testing.assert_close(
                            source.cpu(), values[rank], rtol=0, atol=0, equal_nan=True
                        )
                if rollover:
                    _assert_epoch_wrapped(state)
            if clustered:
                for invalid in (1, 7, 9, 16, 64):
                    with pytest.raises(
                        RuntimeError,
                        match="phased joint profiles require exactly one active eight-CTA cluster",
                    ):
                        state.select_profile("payload_rsag_phased", invalid)
                # Even corrupted device selectors must not partially enter a barrier.
                source.fill_(3)
                out.fill_(3)
                state.selector.fill_((6 << 16) | 7)
                torch.cuda.synchronize()
                dist.barrier()
                before = _read_signal(state)
                graph.replay()
                torch.cuda.synchronize()
                assert _read_signal(state) == before
                assert torch.equal(out, torch.full_like(out, 3))
        startup_graph.replay()
        torch.cuda.synchronize()
        assert producer.status.item() == 0
        winner = int(race_plan.prepared.state.selector.item())
        allowed = {(JOINT_MODES[mode] << 16) | active for mode, active in race_profiles}
        assert winner in allowed
        winners = [None] * 4
        dist.all_gather_object(winners, winner)
        assert len(set(winners)) == 1
        _, _, graph, source, out, _ = next(g for g in graphs if g[0] == 327680 and g[1])
        mode = next(
            mode
            for mode, active in race_profiles
            if winner == (JOINT_MODES[mode] << 16) | active
        )
        values = _operands(4, source.numel(), 173)
        expected = _reference(values, rank, mode.removesuffix("_phased"))
        for _ in range(3):
            source.copy_(values[rank])
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out.cpu(), expected, rtol=0, atol=0, equal_nan=True
            )
    finally:
        if startup_graph is not None:
            startup_graph.reset()
        for _, _, graph, *_ in graphs:
            graph.reset()
        session.close()
        manager.close()


def _worker(rank, family, alias, directory):
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
        _exercise(rank, family, alias)
    finally:
        gc.collect()
        dist.destroy_process_group()
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_CLUSTER_TEST") != "1",
    reason="requires an assigned TP4 SM120/SM121 group",
)
@pytest.mark.parametrize("family", ("joint_latency", "joint_bandwidth"))
@pytest.mark.parametrize("alias", (False, True))
def test_joint_cluster_prepared_graph_transitions(family, alias, tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 4
    mp.spawn(_worker, args=(family, alias, str(tmp_path)), nprocs=4, join=True)
