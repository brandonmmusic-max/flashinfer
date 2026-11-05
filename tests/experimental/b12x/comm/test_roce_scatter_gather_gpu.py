"""Correctness tests for RoCEnante's pipelined reduce-scatter and shard gather.

Run with torchrun on two or more nodes that share a RoCE fabric, for example
from every node of a DGX Spark cluster::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        -m pytest -x tests/experimental/b12x/comm/test_roce_scatter_gather_gpu.py

Without a torchrun environment the tests skip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from b12x.preparation import PreparationSession, PreparedCall

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires a torchrun launch with WORLD_SIZE >= 2",
)

ONESHOT_MAX = 1 << 20
PIPE_BYTES = 32 << 20
HIDDEN = 4096
DTYPES = ("bfloat16", "float32")


@dataclass
class _Prepared:
    runtime: object
    plan: object

    def all_reduce(self, inp, **kwargs):
        return self.runtime.all_reduce(inp, plan=self.plan, **kwargs)

    def reduce_scatter(self, inp, **kwargs):
        return self.runtime.reduce_scatter(inp, plan=self.plan, **kwargs)

    def gather_shards(self, inp, **kwargs):
        return self.runtime.gather_shards(inp, plan=self.plan, **kwargs)


@pytest.fixture(scope="module")
def runtime():
    """A prepared runtime whose pipelined path holds a 32 MiB message, over the torchrun world."""
    from b12x.comm import roce
    from b12x.comm.roce import _preparation

    if not roce.is_supported():
        pytest.skip(
            "RoCE collectives need an integrated GPU with an active RDMA device"
        )
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rt = roce.AllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=ONESHOT_MAX,
        max_gather_bytes=1 << 20,
        pipe_bytes=PIPE_BYTES,
    )
    query = roce.query_from_runtime(
        rt,
        surface="AllReduce.all_reduce",
        call={"dtypes": DTYPES},
        topology="roce_rdma",
        peer_hosts=tuple(f"rank-{r}" for r in range(rt.world_size)),
    )
    plan = roce.plan(query, runtime=rt)
    world = rt.world_size
    seeds = [
        torch.zeros((world, 8), dtype=getattr(torch, d), device=device) for d in DTYPES
    ]

    def prepare(state):
        calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
        calls += [
            _preparation.prepared_reduce_scatter_call(state, inp=seed) for seed in seeds
        ]
        calls += [
            _preparation.prepared_gather_shards_call(state, inp=seed[:1])
            for seed in seeds
        ]
        return PreparedCall(run=lambda: [call.run() for call in calls])

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="roce", prepare_call=prepare),))
    prepared = _Prepared(rt, plan)
    yield prepared
    rt.close()
    dist.barrier()


def _tolerance(dtype: torch.dtype, world: int) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5 * world
    return 1e-2, 2e-2 * world


def _shard(tensor: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    rows = tensor.shape[0] // world
    return tensor[rank * rows : (rank + 1) * rows]


@pytest.mark.parametrize("dtype", [getattr(torch, d) for d in DTYPES])
# 128 rows: a 1 MiB BF16 message, the smallest prefill-like operand; 8192 rows: 32 MiB (BF16) or
# 2 x 8192 rows at FP32 would exceed the pipe, so FP32 uses 2048 rows (32 MiB).
@pytest.mark.parametrize("rows", [128, 2048, 8192])
def test_reduce_scatter_then_gather_matches_two_shot_all_reduce(runtime, dtype, rows):
    """Each rank's shard equals NCCL's sum within tolerance and the two-shot all-reduce bitwise."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    if rows * HIDDEN * dtype.itemsize > PIPE_BYTES:
        pytest.skip("message exceeds the pipelined capacity")
    for trial in range(3):
        torch.manual_seed(1000 * trial + rank)
        inp = torch.randn((rows, HIDDEN), dtype=dtype, device=rt.device)
        assert rt.should_reduce_scatter(inp)
        expected = inp.clone()
        dist.all_reduce(expected)
        shard = runtime.reduce_scatter(inp)
        torch.cuda.synchronize()
        assert shard.shape == (rows // world, HIDDEN)
        rtol, atol = _tolerance(dtype, world)
        torch.testing.assert_close(
            shard, _shard(expected, rank, world), rtol=rtol, atol=atol
        )
        if inp.numel() * inp.element_size() >= rt.pipe_threshold:
            # The same owner reduces the same shard in the same order as the two-shot path.
            reduced = runtime.all_reduce(inp)
            torch.cuda.synchronize()
            assert torch.equal(shard, _shard(reduced, rank, world))
        assert rt.should_gather_shards(shard)
        gathered = runtime.gather_shards(shard)
        torch.cuda.synchronize()
        assert gathered.shape == inp.shape
        torch.testing.assert_close(gathered, expected, rtol=rtol, atol=atol)
        peers = [torch.empty_like(gathered) for _ in range(world)]
        dist.all_gather(peers, gathered)
        for peer in peers:
            assert torch.equal(peer, gathered), "ranks must gather identical messages"


def test_gather_shards_matches_torch_cat(runtime):
    """A shard gather of distinct per-rank rows equals the rank-ordered concatenation."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    torch.manual_seed(31 + rank)
    shard = torch.randn((512, HIDDEN), dtype=torch.bfloat16, device=rt.device)
    peers = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(peers, shard)
    out = torch.empty((512 * world, HIDDEN), dtype=torch.bfloat16, device=rt.device)
    runtime.gather_shards(shard, out=out)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.cat(peers, dim=0))


def test_rejects_ineligible_operands(runtime):
    """Eligibility and output validation happen before any launch."""
    world = dist.get_world_size()
    rt = runtime.runtime
    device = rt.device
    ragged = torch.zeros((world + 1, HIDDEN), dtype=torch.bfloat16, device=device)
    assert not rt.should_reduce_scatter(ragged)
    narrow = torch.zeros((world, 4), dtype=torch.bfloat16, device=device)
    assert not rt.should_reduce_scatter(narrow)
    assert not rt.should_gather_shards(narrow[:1])
    too_large = torch.zeros(
        (PIPE_BYTES // (HIDDEN * 2) + world, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    assert not rt.should_reduce_scatter(too_large)
    assert not rt.should_gather_shards(too_large)
    with pytest.raises(ValueError):
        runtime.reduce_scatter(ragged)
    good = torch.zeros((world * 8, HIDDEN), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        runtime.reduce_scatter(
            good, out=torch.empty((8, HIDDEN - 8), dtype=torch.bfloat16, device=device)
        )
    with pytest.raises(ValueError):
        runtime.gather_shards(
            good[:8],
            out=torch.empty((world * 8, HIDDEN), dtype=torch.float32, device=device),
        )
    dist.barrier()


def test_graph_interleaves_every_kind(runtime):
    """One captured graph alternating one-shot, two-shot, reduce-scatter and gather ops replays correctly."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    torch.manual_seed(77 + rank)
    small = torch.randn((16, HIDDEN), dtype=torch.bfloat16, device=rt.device)
    large = torch.randn((2048, HIDDEN), dtype=torch.bfloat16, device=rt.device)
    small_out, large_out = torch.empty_like(small), torch.empty_like(large)
    shard = torch.empty((2048 // world, HIDDEN), dtype=torch.bfloat16, device=rt.device)
    gathered = torch.empty_like(large)
    expected_small, expected_large = small.clone(), large.clone()
    dist.all_reduce(expected_small)
    dist.all_reduce(expected_large)
    stream = torch.cuda.Stream(device=rt.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.all_reduce(small, out=small_out)
        runtime.reduce_scatter(large, out=shard)
        runtime.gather_shards(shard, out=gathered)
        runtime.all_reduce(large, out=large_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
        for _ in range(3):
            runtime.all_reduce(small, out=small_out)
            runtime.reduce_scatter(large, out=shard)
            runtime.gather_shards(shard, out=gathered)
            runtime.all_reduce(large, out=large_out)
            runtime.reduce_scatter(large, out=shard)
    for _ in range(5):
        small_out.zero_()
        large_out.zero_()
        shard.zero_()
        gathered.zero_()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        rt.check_health()
        rtol, atol = _tolerance(torch.bfloat16, world)
        torch.testing.assert_close(small_out, expected_small, rtol=rtol, atol=atol)
        torch.testing.assert_close(large_out, expected_large, rtol=rtol, atol=atol)
        torch.testing.assert_close(gathered, expected_large, rtol=rtol, atol=atol)
        assert torch.equal(shard, _shard(large_out, rank, world))
        assert torch.equal(gathered, large_out)
    dist.barrier()
