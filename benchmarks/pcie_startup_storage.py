#!/usr/bin/env python3
"""Compare retained startup storage with identical prepared collectives and weights."""

import argparse
from datetime import timedelta
import gc
import json
from pathlib import Path
import subprocess
import tempfile

SIZES = (4112, 163840, 327680, 1310720)
PROFILES = (("payload_rsag", 42), ("bulk_push", 8), ("pull", 7))
WEIGHT_BYTES = 512 << 20


def worker(rank, args, rendezvous):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import (
        CollectiveConfig,
        joint_startup_spec,
        prepare_joint_startup_tuner,
    )
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import (
        CollectiveRequirement,
        MemoryRequirements,
        PreparationSession,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(rendezvous) / "group").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=300),
    )
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        joint=True,
        max_size=max(SIZES),
        eager_buffer_bytes=max(SIZES),
    )
    manager.prepare_channels(("startup-memory-comparison",))
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    graphs = []
    try:
        plans, requests = [], []
        for nbytes in SIZES:
            source = torch.full(
                (nbytes // 2,), rank + 1, dtype=torch.bfloat16, device=device
            )
            out = torch.empty_like(source)
            plan = manager.plan(
                source,
                out=out,
                joint_only=True,
                override=CollectiveConfig(
                    implementation="joint_bandwidth",
                    threads=512,
                    blocks=64,
                    submode="payload_rsag",
                    active_blocks=1,
                ),
            )
            name = f"startup-memory-{nbytes}"
            requests.append(
                plan.request(
                    name=name,
                    collective=CollectiveRequirement(name, tuple(range(4))),
                    prepare_call=lambda state,
                    source=source,
                    out=out: _prepare_plain_call(state, inp=source, out=out),
                )
            )
            plans.append(plan)
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives
            else None,
        )
        weights = torch.ones(WEIGHT_BYTES // 4, dtype=torch.float32, device=device)

        def snapshot():
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            return {
                "torch_live_bytes": torch.cuda.memory_allocated(device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_free_bytes": free,
                "cuda_total_bytes": total,
            }

        before = snapshot()
        storage = None
        if args.arm == "candidate":
            from b12x.comm.pcie import StartupTuningStorage

            storage = StartupTuningStorage(device)
            storage.use_eviction(weights)
            assert storage._eviction.data_ptr() == weights.data_ptr()
        producers = [
            prepare_joint_startup_tuner(
                (joint_startup_spec(plan, profiles=PROFILES),),
                cache_policy="cold",
                **({"storage": storage} if storage is not None else {}),
            )
            for plan in plans
        ]
        for producer in producers:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                producer.run()
            graphs.append(graph)
        session.freeze()
        winners = []
        for producer, graph in zip(producers, graphs, strict=True):
            for _ in range(2):
                producer.status.fill_(-1)
                torch.cuda.synchronize(device)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                assert producer.status.item() == 0
                winner = int(producer.specs[0].selector.item())
                assert winner in {
                    (mode << 16) | active for mode, active in producer.specs[0].profiles
                }
                rank_winners = [None] * 4
                dist.all_gather_object(rank_winners, winner)
                assert len(set(rank_winners)) == 1
            winners.append(winner)
        assert bool(torch.all(weights == 1))
        after = snapshot()
        memory = MemoryRequirements.sequential(
            p.memory_requirements() for p in producers
        )
        result = {
            "rank": rank,
            "before": before,
            "after": after,
            "torch_live_delta_bytes": after["torch_live_bytes"]
            - before["torch_live_bytes"],
            "cuda_free_decrease_bytes": before["cuda_free_bytes"]
            - after["cuda_free_bytes"],
            "producer_deduplicated_resident_bytes": sum(
                entry.resident_nbytes for entry in memory.persistent
            ),
            "producer_persistent_allocations": len(memory.persistent),
            "graph_replay_status": [int(p.status.item()) for p in producers],
            "winners": winners,
            "eviction_aliases_weights": producers[0].eviction.data_ptr()
            == weights.data_ptr(),
            "eviction_allocations": len(
                {p.eviction.untyped_storage().data_ptr() for p in producers}
            ),
            "fixture_arenas": len(
                {p.inputs.source.untyped_storage().data_ptr() for p in producers}
            ),
        }
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        dist.barrier()
    finally:
        for graph in graphs:
            graph.reset()
        session.close()
        manager.close()
        gc.collect()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "candidate"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    import torch
    import torch.multiprocessing as mp
    import b12x

    assert torch.cuda.device_count() == 4
    assert Path(b12x.__file__).resolve().is_relative_to(args.source.resolve())
    revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="startup-memory-") as rendezvous:
        mp.spawn(worker, args=(args, rendezvous), nprocs=4, join=True)
    results = [
        json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(4)
    ]
    document = {
        "arm": args.arm,
        "revision": revision,
        "source": str(args.source),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sizes_bytes": SIZES,
        "profiles": PROFILES,
        "threads": 512,
        "physical_blocks": 64,
        "world_size": 4,
        "resident_weight_bytes_per_rank": WEIGHT_BYTES,
        "measurement": "live graphs after validated replay, synchronization, GC, and empty_cache; resident weights precede baseline",
        "ranks": results,
    }
    (args.output / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2), flush=True)


if __name__ == "__main__":
    main()
