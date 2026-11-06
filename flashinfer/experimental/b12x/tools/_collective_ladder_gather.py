"""Column all-gather candidates measured with the collective ladder protocol."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import json
import time

from ._collective_ladder_gpu import JointCollector


def gather_catalog(size, row_bytes, world, kernel="static"):
    if kernel not in ("static", "runtime", "compare"):
        raise ValueError("gather kernel must be static, runtime, or compare")
    if (
        world not in (2, 4, 8)
        or row_bytes < 16
        or row_bytes % 16
        or size < row_bytes
        or size % row_bytes
    ):
        raise ValueError(
            "gather sizes must contain complete aligned rows on TP2/TP4/TP8"
        )
    rows = size // row_bytes
    barrier_work = rows * ((row_bytes + 511) // 512) * world
    configs = []
    for transport in ("barrier", "word_exchange"):
        if transport == "word_exchange" and (world not in (2, 4) or size > 16384):
            continue
        for threads in (
            (32, 64, 128, 256, 512) if transport == "barrier" else (32, 64, 128, 256)
        ):
            warps = threads // 32
            work = barrier_work if transport == "barrier" else (size + 255) // 256
            limit = min(64, (work + warps - 1) // warps)
            counts = {min(limit, value) for value in (1, 2, 4, 8, 16, 32, 64)}
            if kernel in ("static", "compare"):
                for count in sorted(counts):
                    configs.append(
                        dict(
                            implementation="gather_" + transport,
                            threads=threads,
                            blocks=64,
                            submode=transport,
                            active_blocks=count,
                        )
                    )
            if kernel in ("runtime", "compare"):
                families = ["gather_selected_" + transport]
                for family in families:
                    for blocks in sorted({1, 2, 4, 8, 16, 32, 64, limit}):
                        for count in sorted(counts):
                            if count <= blocks:
                                configs.append(
                                    dict(
                                        implementation=family,
                                        threads=threads,
                                        blocks=blocks,
                                        submode=transport,
                                        active_blocks=count,
                                    )
                                )
    if kernel in ("runtime", "compare") and world in (2, 4) and size <= 16384:
        for threads in (32, 64, 128, 256, 512):
            limits = {
                "barrier": min(
                    64, (barrier_work + threads // 32 - 1) // (threads // 32)
                ),
                "word_exchange": min(64, (size + 255) // 256),
            }
            grids = {1, 2, 4, 8, 16, 32, 64} | set(limits.values())
            for mode, limit in limits.items():
                counts = {min(limit, value) for value in (1, 2, 4, 8, 16, 32, 64)}
                for blocks in sorted(grids):
                    for count in sorted(counts):
                        if count <= blocks:
                            configs.append(
                                dict(
                                    implementation="gather_joint",
                                    threads=threads,
                                    blocks=blocks,
                                    submode=mode,
                                    active_blocks=count,
                                )
                            )
    return tuple(configs)


def proposal_catalog(proposal, size, row_bytes, world):
    """Expand an inclusive bucket into exact profiles, retaining measured references."""
    if proposal.get("operation") != "all_gather" or proposal.get("world_size") != world:
        raise ValueError("gather proposal operation and TP size must match collection")
    if not proposal["min_bytes"] <= size <= proposal["buckets"][-1]["max_bytes"]:
        raise ValueError("gather proposal does not cover the payload")
    row = next(row for row in proposal["buckets"] if size <= row["max_bytes"])
    gather_catalog(size, row_bytes, world, "compare")
    configs = [
        dict(
            implementation=row["implementation"],
            threads=row["threads"],
            blocks=row["blocks"],
            submode=mode,
            active_blocks=count,
        )
        for mode, counts in row["profiles"].items()
        for count in counts
    ]
    configs.extend(dict(c) for c in row.get("references", ()))

    def legal(config):
        family, mode = config["implementation"], config["submode"]
        modes = {
            "gather_joint": ("barrier", "word_exchange"),
            "gather_barrier": ("barrier",),
            "gather_selected_barrier": ("barrier",),
            "gather_word_exchange": ("word_exchange",),
            "gather_selected_word_exchange": ("word_exchange",),
        }
        return (
            mode in modes.get(family, ())
            and config["threads"] in (32, 64, 128, 256, 512)
            and type(config["blocks"]) is int
            and type(config["active_blocks"]) is int
            and 1 <= config["active_blocks"] <= config["blocks"] <= 64
            and (family != "gather_joint" or (world in (2, 4) and size <= 16384))
            and (
                mode != "word_exchange"
                or (
                    world in (2, 4)
                    and size <= 16384
                    and (family == "gather_joint" or config["threads"] <= 256)
                )
            )
        )

    if not configs or any(not legal(config) for config in configs):
        raise ValueError(f"gather proposal contains an illegal profile at {size} bytes")
    return tuple({tuple(sorted(c.items())): c for c in configs}.values())


class GatherCollector(JointCollector):
    """Race static and device-selected gather kernels with identical fixtures."""

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        from b12x.comm.pcie import DcpAllToAll, query_from_runtime
        from b12x.comm.pcie._dcp_preparation import plan
        from b12x.preparation import PreparationSession, PreparedCall, FrozenMapping

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.row_bytes, self.max_bytes = args.row_bytes, args.max_bytes
        self.inplace, self.cache_policy = False, args.cache_policy
        self.compilation_seconds, self._closed = 0.0, False
        proposal_path = getattr(args, "gather_proposal", None)
        proposal = (
            json.loads(proposal_path.read_text()) if proposal_path is not None else None
        )
        if proposal is not None and proposal.get("cache_policy") != args.cache_policy:
            raise ValueError("gather proposal cache policy differs from collection")
        self._catalogs = {
            size: (
                proposal_catalog(proposal, size, self.row_bytes, self.world)
                if proposal is not None
                else gather_catalog(
                    size, self.row_bytes, self.world, args.gather_kernel
                )
            )
            for size in args.sizes
        }
        self._states = {}
        self.session = PreparationSession(
            device=self.device,
            autotune=False,
            prime=False,
            compile_workers=args.compile_workers,
        )
        self.session.__enter__()
        rows, columns = self.max_bytes // self.row_bytes, self.row_bytes // 2
        self.manager = DcpAllToAll.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=self.device,
            max_batch_size=rows,
            total_heads=self.world,
            head_dim=columns,
            stream_affine=False,
        )
        values = (
            torch.arange(self.world * rows * columns).reshape(
                self.world, rows, 1, columns
            )
            % 251
        ).bfloat16()
        seeds = (values, torch.zeros_like(values), -values)
        self.inputs = SimpleNamespace(
            source=torch.empty(
                self.max_bytes // 2, dtype=torch.bfloat16, device=self.device
            ),
            out=torch.empty(
                self.world * self.max_bytes // 2,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            reference=SimpleNamespace(
                seeds=tuple(
                    value[self.rank].flatten().to(self.device) for value in seeds
                ),
                expected=tuple(
                    value.permute(1, 0, 2, 3).flatten().to(self.device)
                    for value in seeds
                ),
            ),
        )
        configs = {
            self._geometry(config): config
            for catalog in self._catalogs.values()
            for config in catalog
        }
        plans, requests, jobs = {}, [], []
        from b12x._lib.compile_pool import CompileJob

        selected = {}
        for key, config in sorted(configs.items()):
            if config["implementation"] not in (
                "gather_barrier",
                "gather_word_exchange",
            ):
                transport = (
                    "joint"
                    if config["implementation"] == "gather_joint"
                    else config["submode"]
                )
                selected[key] = (
                    self.world,
                    self.rank,
                    self.row_bytes // 4,
                    config["threads"],
                    config["blocks"],
                    self.manager._exchange_chunks,
                    transport,
                )
                jobs.append(
                    CompileJob.create(
                        "b12x.comm.pcie._runtime_gather_cute:compile_selected_gather",
                        *selected[key],
                    )
                )
                continue
            size = (
                min(self.max_bytes, 16384)
                if config["submode"] == "word_exchange"
                else self.max_bytes
            )
            count = size // 2
            source = self.inputs.source[:count].view(-1, 1, columns)
            output = self.inputs.out[: count * self.world].view(-1, self.world, columns)
            query = query_from_runtime(
                self.manager,
                surface="DcpAllToAll.all_gather_heads",
                call=dict(local_input=source, out=output),
            )
            query = replace(
                query,
                call=FrozenMapping(
                    dict(
                        query.call,
                        threads=config["threads"],
                        block_limit=config["active_blocks"],
                        gather_transport=config["submode"],
                    )
                ),
            )
            prepared = plan(query, runtime=self.manager)
            plans[key] = prepared
            jobs.extend(prepared._compile_jobs(None, self.session.device))
            requests.append(
                prepared.request(
                    name="gather." + ".".join(map(str, key)),
                    prepare_call=lambda state, source=source, output=output: (
                        PreparedCall(
                            run=lambda: state.run(local_input=source, out=output),
                            owners=(state, source, output),
                        )
                    ),
                )
            )
        for action in ("reset", "validate", "begin", "end"):
            jobs.append(
                CompileJob.create(
                    "b12x.comm.pcie._joint_startup_cute:timing_launcher",
                    self.world,
                    self.rank,
                    action,
                    1,
                    CompileJob.DEVICE_ORDINAL,
                )
            )
        started = time.monotonic()
        if self.session.compile_workers:
            self.session._compiler().compile(tuple(jobs))
        else:
            for job in dict.fromkeys(jobs):
                job.invoke()
        self.compilation_seconds += time.monotonic() - started
        self.session.prepare(requests)
        self._plans = plans
        self._states = {
            key: SimpleNamespace(
                state=prepared.prepared.state,
                selector=torch.zeros(1, dtype=torch.uint32, device=self.device),
            )
            for key, prepared in plans.items()
        }
        if selected:
            from b12x.comm.pcie._runtime_gather_cute import compile_selected_gather

            self._states.update(
                {
                    key: SimpleNamespace(
                        launcher=compile_selected_gather(*values),
                        selector=torch.ones(1, dtype=torch.uint32, device=self.device),
                    )
                    for key, values in selected.items()
                }
            )
        self._initialize_timing(self.manager)

    @staticmethod
    def _geometry(config):
        grid = (
            config["active_blocks"]
            if config["implementation"] in ("gather_barrier", "gather_word_exchange")
            else config["blocks"]
        )
        return config["implementation"], config["threads"], grid

    def catalog(self, size):
        return tuple(dict(config) for config in self._catalogs[size])

    def _select(self, config):
        entry = self._states[self._geometry(config)]
        entry.selector.fill_(
            (int(config["submode"] == "word_exchange") << 16) | config["active_blocks"]
        )
        return entry

    def _output_count(self, count):
        return count * self.world

    def _capture(self):
        return nullcontext()

    def _invoke(self, entry, size):
        runtime = self.manager
        rows = size // self.row_bytes
        if hasattr(entry, "launcher"):
            entry.launcher(
                self.inputs.source.data_ptr(),
                self.inputs.out.data_ptr(),
                entry.selector.data_ptr(),
                runtime._staging0_ptrs,
                runtime._signal_ptrs,
                rows,
                runtime._slot_bytes,
                runtime._exchange_offset,
                runtime._exchange_counter_offset,
            )
            return
        state = entry.state
        if state.query.call["gather_transport"] == "word_exchange":
            state.launcher(False)(
                self.inputs.source.data_ptr(),
                self.inputs.out.data_ptr(),
                runtime._staging0_ptrs,
                runtime._slot_bytes,
                runtime._exchange_offset,
                runtime._exchange_counter_offset,
                rows,
            )
        else:
            state.launcher(True)(
                self.inputs.source.data_ptr(),
                self.inputs.out.data_ptr(),
                runtime._staging0_ptrs,
                runtime._signal_ptrs,
                rows,
                1,
                self.row_bytes // 2,
                2,
                runtime._slot_bytes // 256,
                state.query.call["block_limit"],
            )
