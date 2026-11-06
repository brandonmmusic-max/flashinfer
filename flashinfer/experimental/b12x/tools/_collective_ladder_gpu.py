"""Fixed-storage GPU measurements for the offline joint collective ladder."""

from dataclasses import asdict
import csv
import io
import socket
import subprocess
import time

from ._collective_ladder_sampling import MAX_ROUNDS, SAMPLES_PER_ROUND


class JointCollector:
    """Measure legal joint profiles after preparing every compiled geometry.

    The caller initializes a Gloo process group and selects its local CUDA device.
    Each recorded sample is the maximum rank duration of one reduction, with
    same-profile preconditioning, producer copies, and optional cache eviction
    outside the timed interval.
    """

    _BATCH = 8
    _MAX_ROUNDS = MAX_ROUNDS
    _CHUNK_ROUNDS = 8

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        from b12x.comm.pcie import AllReduce
        from b12x.preparation import PreparationSession, PreparedCall

        if not dist.is_initialized() or dist.get_backend() != "gloo":
            raise ValueError(
                "joint ladder collection requires an initialized Gloo group"
            )
        if args.cache_policy not in ("cold", "warm"):
            raise ValueError("cache policy must be cold or warm")
        if type(args.inplace) is not bool:
            raise TypeError("inplace must be a boolean")
        if (
            type(args.max_bytes) is not int
            or args.max_bytes <= 0
            or args.max_bytes % 16
        ):
            raise ValueError("maximum payload must contain positive 16-byte packs")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.max_bytes, self.inplace, self.cache_policy = (
            args.max_bytes,
            args.inplace,
            args.cache_policy,
        )
        self.compilation_seconds = 0.0
        self._closed = False
        self._catalogs = {}
        self._states = {}
        self.inputs = None
        self.session = PreparationSession(
            device=self.device,
            autotune=False,
            prime=False,
            compile_workers=args.compile_workers,
        )
        self.session.__enter__()
        self.manager = AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=self.device,
            single_channel=True,
            max_size=self.max_bytes,
            eager_buffer_bytes=self.max_bytes,
            joint=True,
        )
        prototype = torch.empty(
            self.max_bytes // 2, dtype=torch.bfloat16, device=self.device
        )
        output = prototype if self.inplace else torch.empty_like(prototype)
        maximum = self._query(prototype, output)
        configs = self._eligible(maximum)
        geometry = {}
        for config in configs:
            geometry.setdefault(self._geometry(config), config)
        plans, requests, jobs = {}, [], []

        def prepare(state):
            from b12x.comm.pcie._benchmark import _ReductionInputs

            if self.inputs is None:
                self.inputs = _ReductionInputs(state, inplace=self.inplace)
            state.bind_input(self.inputs.source)
            return PreparedCall(
                run=lambda: state.run_plain(self.inputs.source, self.inputs.out),
                owners=(state, self.inputs),
            )

        from b12x.comm.pcie import CollectiveConfig
        from b12x._lib.compile_pool import CompileJob

        for key, config in sorted(geometry.items()):
            plan = self.manager.plan(
                prototype,
                out=output,
                joint_only=True,
                override=CollectiveConfig(**config),
            )
            plans[key] = plan
            jobs.extend(
                plan._compile_jobs(CollectiveConfig(**config), self.session.device)
            )
            requests.append(
                plan.request(
                    name="ladder." + ".".join(map(str, key)), prepare_call=prepare
                )
            )
        actions = ("reset", "validate", "begin", "end")
        for action in actions:
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
        self._states = {key: plan.prepared.state for key, plan in plans.items()}
        self._plans = plans
        self._catalogs[self.max_bytes] = configs

        self._initialize_timing(next(iter(self._states.values())).runtime)

    def _initialize_timing(self, runtime):
        import torch
        from b12x.comm.pcie._joint_startup_cute import SIGNAL_BYTES, SCRATCH_BYTES
        from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        actions = ("reset", "validate", "begin", "end")
        from b12x.comm.pcie._joint_startup_cute import timing_launcher

        self._timers = {
            action: timing_launcher(self.world, self.rank, action, 1, self.device.index)
            for action in actions
        }
        self._shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            runtime.exchange_group, SCRATCH_BYTES, zero_fill=True, ipc=runtime._ipc
        )
        runtime._owned_buffers.append(self._shared)
        self._peers = torch.tensor(
            self._shared.peer_ptrs, dtype=torch.uint64, device=self.device
        )
        self._local = _tensor_from_pointer(
            self._shared.local_ptr + SIGNAL_BYTES,
            shape=(4,),
            dtype=torch.int64,
            device=self.device,
            nbytes=32,
        )
        self._samples = torch.empty(
            (self._BATCH, self._CHUNK_ROUNDS * SAMPLES_PER_ROUND),
            dtype=torch.int64,
            device=self.device,
        )
        self._status = torch.full((1,), -1, dtype=torch.int32, device=self.device)
        self._invalid = torch.zeros((), dtype=torch.int32, device=self.device)
        self._mismatch = torch.empty_like(self.inputs.out, dtype=torch.bool)
        self._profiles = torch.zeros(1, dtype=torch.uint32, device=self.device)
        self._eviction = self._eviction_sum = None
        if self.cache_policy == "cold":
            props = torch.cuda.get_device_properties(self.device)
            self._eviction = torch.ones(
                max(2 * props.L2_cache_size, 64 << 20) // 4,
                dtype=torch.float32,
                device=self.device,
            )
            self._eviction_sum = torch.empty(
                (), dtype=torch.float32, device=self.device
            )
        self._stream = torch.cuda.Stream(device=self.device)
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        self.metadata = self._metadata()

    @staticmethod
    def _geometry(config):
        return config["implementation"], config["threads"], config["blocks"]

    def _query(self, source, output):
        return self.manager.plan(source, out=output, joint_only=True).query

    def _eligible(self, query):
        from b12x.comm.pcie._collective_tuning import TUNING
        from b12x.preparation import autotune_effort

        with autotune_effort("audit"):
            return tuple(
                asdict(config)
                for _, config in TUNING.choices(query, self.session.device.identity)
            )

    def catalog(self, size):
        """Return the actual legal, equivalence-deduplicated joint configurations."""
        if type(size) is not int or not 0 < size <= self.max_bytes or size % 16:
            raise ValueError("payload must fit the prepared capacity in 16-byte packs")
        if self._closed:
            raise RuntimeError("joint collector is closed")
        if size not in self._catalogs:
            source, output = (
                self.inputs.source[: size // 2],
                self.inputs.out[: size // 2],
            )
            configs = self._eligible(self._query(source, output))
            if any(self._geometry(config) not in self._states for config in configs):
                raise RuntimeError(
                    "payload catalog requires an unprepared joint geometry"
                )
            self._catalogs[size] = configs
        return tuple(dict(config) for config in self._catalogs[size])

    def _metadata(self):
        import torch
        import torch.distributed as dist
        import cuda.bindings.driver as cuda

        props = torch.cuda.get_device_properties(self.device)
        local = dict(
            rank=self.rank,
            host=socket.gethostname(),
            uuid=str(props.uuid),
            name=props.name,
            compute_capability=[props.major, props.minor],
            sm_count=props.multi_processor_count,
        )
        devices = [None] * self.world
        dist.all_gather_object(devices, local)
        status, version = cuda.cuDriverGetVersion()
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("cannot query the CUDA driver version")
        topology = self._host_topology()
        topologies = [None] * self.world
        dist.all_gather_object(topologies, dict(host=local["host"], **topology))
        return dict(
            world_size=self.world,
            devices=devices,
            driver_version=version,
            cuda_version=torch.version.cuda,
            topologies=topologies,
            cache_policy=self.cache_policy,
            inplace=self.inplace,
            max_bytes=self.max_bytes,
        )

    @staticmethod
    def _host_topology():
        commands = (
            ("topology", ["nvidia-smi", "topo", "-m"]),
            (
                "GPU index map",
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id",
                    "--format=csv,noheader",
                ],
            ),
        )
        results = []
        for label, command in commands:
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=10
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"nvidia-smi {label} query timed out after 10s; stderr={error.stderr!r}"
                ) from error
            except OSError as error:
                raise RuntimeError(
                    f"cannot execute nvidia-smi {label} query: {error}"
                ) from error
            if result.returncode != 0:
                raise RuntimeError(
                    f"nvidia-smi {label} query failed: returncode={result.returncode}; "
                    f"stderr={result.stderr!r}; stdout={result.stdout!r}"
                )
            results.append(result)
        topology, device_map = results
        if not topology.stdout.strip():
            raise RuntimeError("nvidia-smi topology query returned an empty matrix")
        devices = []
        try:
            for row in csv.reader(io.StringIO(device_map.stdout)):
                if len(row) != 3:
                    raise ValueError("expected index, UUID, and PCI bus ID columns")
                index, uuid, pci_bus_id = (value.strip() for value in row)
                index = int(index)
                uuid = uuid.removeprefix("GPU-")
                if index < 0 or not uuid or not pci_bus_id:
                    raise ValueError("GPU index, UUID, and PCI bus ID must be present")
                devices.append(dict(index=index, uuid=uuid, pci_bus_id=pci_bus_id))
            if (
                not devices
                or len({d["index"] for d in devices}) != len(devices)
                or len({d["uuid"] for d in devices}) != len(devices)
            ):
                raise ValueError(
                    "GPU index map must contain distinct indices and UUIDs"
                )
        except (ValueError, csv.Error) as error:
            raise RuntimeError(
                f"invalid nvidia-smi GPU index map: {error}; returncode={device_map.returncode}; "
                f"stderr={device_map.stderr!r}; stdout={device_map.stdout!r}"
            ) from error
        return dict(
            returncode=topology.returncode,
            text=topology.stdout,
            stderr=topology.stderr,
            gpu_map=devices,
        )

    def _timing(self, action, sample, selector):
        self._timers[action](
            self._peers.data_ptr(),
            self._local.data_ptr(),
            self._samples.data_ptr(),
            self._invalid.data_ptr(),
            self._profiles.data_ptr(),
            selector.data_ptr(),
            self._status.data_ptr(),
            sample,
        )

    def _invoke(self, state, size):
        workspace = state.workspace
        state.launcher(
            workspace.peer_table.data_ptr(),
            workspace.signal_table.data_ptr(),
            workspace.shared.local_ptr,
            state.selector.data_ptr(),
            self.inputs.source.data_ptr(),
            self.inputs.out.data_ptr(),
            size // 16,
            workspace.capacity_packs,
        )

    def _select(self, config):
        from b12x.comm.pcie._joint_profile import selected_profile

        state = self._states[self._geometry(config)]
        state.selector.fill_(
            selected_profile(
                config["implementation"],
                config["blocks"],
                config["submode"],
                config["active_blocks"],
            )
        )
        return state

    def _output_count(self, count):
        return count

    def _capture(self):
        return self.manager.capture()

    def measure(self, size, configs, rounds=3):
        """Return six per-invocation rank maxima per round from three fixtures."""
        import torch
        from b12x.comm.pcie.pcie_oneshot import _require_collective_contract

        if type(rounds) is not int or not 1 <= rounds <= self._MAX_ROUNDS:
            raise ValueError("ladder measurements require one to 512 rounds")
        legal = {tuple(sorted(value.items())) for value in self.catalog(size)}
        configs = tuple(dict(value) for value in configs)
        if not configs or any(
            tuple(sorted(value.items())) not in legal for value in configs
        ):
            raise ValueError(
                "measurement configurations must belong to the payload's legal catalog"
            )
        _require_collective_contract(
            owner="joint ladder measurement",
            exchange_group=self.manager.exchange_group,
            contract=(
                size,
                rounds,
                tuple(tuple(sorted(value.items())) for value in configs),
            ),
        )
        output = []
        count = size // 2
        output_count = self._output_count(count)
        source, target = self.inputs.source[:count], self.inputs.out[:output_count]
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        for begin in range(0, len(configs), self._BATCH):
            batch = configs[begin : begin + self._BATCH]
            values_by_config = [[] for _ in batch]

            def produce(case, state=None):
                source.copy_(self.inputs.reference.seeds[case][:count])
                if state is not None:
                    self._invoke(state, size)
                if state is None and self._eviction is not None:
                    torch.sum(self._eviction, dim=0, out=self._eviction_sum)

            def enqueue(chunk_rounds):
                self._timing("reset", 0, self._profiles)
                for config in batch:
                    state = self._select(config)
                    for case in range(3):
                        produce(case)
                        self._invoke(state, size)
                        torch.ne(
                            target,
                            self.inputs.reference.expected[case][:output_count],
                            out=self._mismatch[:output_count],
                        )
                        torch.sum(
                            self._mismatch[:output_count],
                            dim=0,
                            dtype=torch.int32,
                            out=self._invalid,
                        )
                        self._timing("validate", 0, state.selector)
                width = chunk_rounds * SAMPLES_PER_ROUND
                for turn in range(chunk_rounds):
                    order = list(range(len(batch)))
                    if turn % 2:
                        order.reverse()
                    offset = (turn // 2) % len(order)
                    order = order[offset:] + order[:offset]
                    for case in range(3):
                        for index in order:
                            state = self._select(batch[index])
                            produce(case, state)
                            # Consecutive reductions exercise both shared-buffer slots.
                            for invocation in range(2):
                                produce(case)
                                sample = (
                                    index * width
                                    + turn * SAMPLES_PER_ROUND
                                    + case * 2
                                    + invocation
                                )
                                self._timing("begin", sample, state.selector)
                                self._invoke(state, size)
                                self._timing("end", sample, state.selector)

            with torch.cuda.stream(self._stream):
                produce(0)
                torch.ne(
                    target,
                    self.inputs.reference.expected[0][:output_count],
                    out=self._mismatch[:output_count],
                )
                torch.sum(
                    self._mismatch[:output_count],
                    dim=0,
                    dtype=torch.int32,
                    out=self._invalid,
                )
            self._stream.synchronize()
            full, remainder = divmod(rounds, self._CHUNK_ROUNDS)
            chunks = []
            if full:
                chunks.append((self._CHUNK_ROUNDS, full))
            if remainder:
                chunks.append((remainder, 1))
            for chunk_rounds, replays in chunks:
                graph = torch.cuda.CUDAGraph()
                width = chunk_rounds * SAMPLES_PER_ROUND
                try:
                    with (
                        self.session.capture(),
                        self._capture(),
                        torch.cuda.graph(graph, stream=self._stream),
                    ):
                        enqueue(chunk_rounds)
                    for _ in range(replays):
                        with torch.cuda.stream(self._stream):
                            graph.replay()
                        self._stream.synchronize()
                        if self._local[3].item() != 0:
                            raise RuntimeError(
                                "joint ladder configuration failed exact fixture validation"
                            )
                        samples = (
                            self._samples.view(-1)[: len(batch) * width]
                            .cpu()
                            .view(len(batch), width)
                            .tolist()
                        )
                        for collected, values in zip(values_by_config, samples):
                            if any(value <= 0 for value in values):
                                raise RuntimeError(
                                    "joint ladder timing produced a nonpositive duration"
                                )
                            collected.extend(value / 1000.0 for value in values)
                finally:
                    graph.reset()
            for config, values in zip(batch, values_by_config):
                output.append(
                    dict(
                        bytes=size,
                        config=config,
                        samples_us=values,
                        correctness="passed",
                    )
                )
        return output

    def close(self):
        if self._closed:
            return
        import torch

        torch.cuda.synchronize(self.device)
        self._closed = True
        try:
            self.session.__exit__(None, None, None)
        finally:
            self.manager.close()
