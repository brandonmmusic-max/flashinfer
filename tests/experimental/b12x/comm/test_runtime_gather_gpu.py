"""Exact column gathers while captured kernels switch protocol and CTA count."""

import ctypes
from datetime import timedelta
import os
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_WORD_EXCHANGE_TEST") != "1",
    reason="requires explicitly assigned peer-access GPUs",
)


def _rank(rank, world, folder, row_bytes, threads):
    from b12x.comm.pcie import DcpAllToAll
    from b12x.comm.pcie._runtime_gather_cute import compile_selected_gather
    from b12x.comm.pcie._dcp_a2a_cute import _GRAPH_EPOCH_INDEX, _GRAPH_ARRIVED_INDEX

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(folder) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=3),
    )
    capacity = 16384 // row_bytes
    runtime = DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=capacity,
        total_heads=world,
        head_dim=row_bytes // 2,
        stream_affine=False,
    )
    selector = torch.ones(1, dtype=torch.uint32, device=device)
    launcher = compile_selected_gather(
        world, rank, row_bytes // 4, threads, 64, runtime._exchange_chunks, "joint"
    )
    graphs, data = {}, {}
    stream = int(torch.cuda.current_stream().cuda_stream)

    def copy_host(address, values, upload=False):
        host = ctypes.addressof(values)
        runtime._ipc.cudaMemcpyAsync(
            address if upload else host,
            host if upload else address,
            ctypes.sizeof(values),
            stream,
        )
        torch.cuda.synchronize()
        return list(values)

    def agree(errors):
        gathered = [None] * world
        dist.all_gather_object(gathered, errors)
        assert not any(gathered), gathered

    try:
        for rows in (1, 3, capacity):
            source = torch.zeros(
                (rows, row_bytes // 4), dtype=torch.int32, device=device
            )
            storage = torch.full(
                (rows * world * row_bytes // 4 + 8,),
                0x12345678,
                dtype=torch.int32,
                device=device,
            )
            output = storage[4:-4].view(rows, world, row_bytes // 4)
            graphs[rows] = torch.cuda.CUDAGraph()
            data[rows] = source, output, storage
            dist.barrier()
            with torch.cuda.graph(graphs[rows]):
                launcher(
                    source.data_ptr(),
                    output.data_ptr(),
                    selector.data_ptr(),
                    runtime._staging0_ptrs,
                    runtime._signal_ptrs,
                    rows,
                    runtime._slot_bytes,
                    runtime._exchange_offset,
                    runtime._exchange_counter_offset,
                )
        counter_address = (
            runtime._staging0_ptrs[rank] + runtime._exchange_counter_offset
        )
        expected_counters = [0xFFFFFFFD] * runtime._exchange_chunks
        copy_host(
            counter_address,
            (ctypes.c_uint32 * len(expected_counters))(*expected_counters),
            True,
        )
        stale = (ctypes.c_uint64 * (world * runtime._exchange_chunks * 64))(
            *([2 << 32 | 0xDEADCAFE] * (world * runtime._exchange_chunks * 64))
        )
        for bank in (0, 1):
            copy_host(
                runtime._staging_ptrs[bank][rank] + runtime._exchange_offset,
                stale,
                True,
            )
        epoch_address = runtime._signal_ptrs[rank] + 4 * _GRAPH_EPOCH_INDEX
        epoch = copy_host(epoch_address, (ctypes.c_uint32 * 1)())[0]
        schedule = [
            (1, 1, 1),
            (3, 0, 2),
            (capacity, 1, 64),
            (1, 0, 1),
            (3, 1, 4),
            (capacity, 0, 64),
            (capacity, 1, 1),
            (3, 0, 8),
        ] * 6
        history = []
        for step, (rows, mode, active) in enumerate(schedule):
            bits = torch.randint(
                -(2**31),
                2**31,
                (world, rows, row_bytes // 4),
                dtype=torch.int32,
                generator=torch.Generator().manual_seed(4193 + step),
            )
            bits[:, 0, :4] = torch.tensor(
                [0, -2147483648, -1, 2143289345], dtype=torch.int32
            )
            history.append(
                (
                    bits[rank].to(device),
                    bits.permute(1, 0, 2).contiguous(),
                    torch.empty_like(data[rows][1]),
                )
            )
        torch.cuda.synchronize()
        dist.barrier()
        before = torch.cuda.memory_stats()
        for (rows, mode, active), (payload, _, saved) in zip(schedule, history):
            source, output, _ = data[rows]
            source.copy_(payload)
            selector.fill_((mode << 16) | active)
            graphs[rows].replay()
            saved.copy_(output)
            if mode:
                for chunk in range((rows * row_bytes + 255) // 256):
                    previous = expected_counters[chunk]
                    expected_counters[chunk] = (
                        2 if previous == 0xFFFFFFFF else previous + 1
                    )
            else:
                epoch += 1
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        errors = []
        for key in (
            "allocation.all.allocated",
            "allocation.all.freed",
            "segment.all.allocated",
            "segment.all.freed",
        ):
            if before[key] != after[key]:
                errors.append((key, before[key], after[key]))
        for step, (_, expected, saved) in enumerate(history):
            if not torch.equal(saved.cpu(), expected):
                errors.append((step, "payload mismatch"))
        for rows, (source, _, storage) in data.items():
            last = max(i for i, (n, _, _) in enumerate(schedule) if n == rows)
            if not torch.equal(source, history[last][0]):
                errors.append((rows, "source modified"))
            if not torch.all(storage[:4] == 0x12345678) or not torch.all(
                storage[-4:] == 0x12345678
            ):
                errors.append((rows, "output guard overwritten"))
        actual = copy_host(
            counter_address, (ctypes.c_uint32 * len(expected_counters))()
        )
        if actual != expected_counters:
            errors.append(("word counters", actual, expected_counters))
        actual = copy_host(epoch_address, (ctypes.c_uint32 * 1)())[0]
        if actual != epoch:
            errors.append(("barrier epoch", actual, epoch))
        arrived = copy_host(
            runtime._signal_ptrs[rank] + 4 * _GRAPH_ARRIVED_INDEX,
            (ctypes.c_uint32 * 1)(),
        )[0]
        if arrived:
            errors.append(("unretired CTAs", arrived))
        agree(errors)
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "row_bytes,threads", [(16, 32), (128, 64), (512, 128), (1024, 256), (64, 512)]
)
def test_protocol_switching_replay(world, row_bytes, threads):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(
            _rank, args=(world, folder, row_bytes, threads), nprocs=world, join=True
        )
