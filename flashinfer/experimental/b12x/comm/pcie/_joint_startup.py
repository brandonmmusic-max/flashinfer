"""Prepare captured selection among supplied or measured joint collective profiles."""

from dataclasses import dataclass, field
import weakref

import torch

from b12x.preparation import FinalistPolicy
from ._joint_profile import (
    JOINT_MODES,
    joint_modes,
    selected_profile,
    validate_cluster_geometry,
)


def joint_finalist_policy():
    """Retain the best measured CTA count for each mode and compiled geometry."""
    return FinalistPolicy(
        family_fields=("implementation", "threads", "blocks", "submode"),
        per_family=1,
        max_total=35,
    )


class StartupTuningStorage:
    """Caller-owned fixtures and eviction input for serial startup races.

    Pass one instance to every startup producer in a preparation session. All
    graphs using it must replay serially. Retain the producers until their
    graphs drain; dropping the storage and producers releases their tensors.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("startup tuning storage requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._fixtures = None
        self._eviction = self._eviction_sum = None
        self.external_eviction = False
        self.captured = False
        self._cold_users = weakref.WeakSet()

    @property
    def eviction_bytes(self):
        """Minimum contiguous resident bytes needed to displace the device's L2."""
        l2 = torch.cuda.get_device_properties(self.device).L2_cache_size
        return max(2 * l2, 64 * 1024 * 1024)

    @property
    def fixtures(self):
        from ._joint_startup_fixtures import StartupFixtures

        if self._fixtures is None:
            self._fixtures = StartupFixtures(self.device)
        return self._fixtures

    def _cold_eviction(self, producer):
        if self._eviction is None:
            self._eviction = torch.zeros(
                self.eviction_bytes // 4, dtype=torch.float32, device=self.device
            )
        if self._eviction_sum is None:
            self._eviction_sum = torch.empty(
                (), dtype=torch.float32, device=self.device
            )
        self._cold_users.add(producer)
        return self._eviction, self._eviction_sum

    def use_eviction(self, source):
        """Read caller-owned resident storage during cold races without copying it.

        The source must be contiguous CUDA storage on this device, aligned to
        four bytes, with at least ``eviction_bytes`` bytes. Configure it before
        capture. Its contents are read as FP32 bits solely for cache eviction.
        """
        if self.captured:
            raise RuntimeError(
                "the startup eviction source cannot change after capture"
            )
        if not isinstance(source, torch.Tensor) or source.device != self.device:
            raise ValueError(
                "startup eviction source must be a CUDA tensor on the storage device"
            )
        if not source.is_contiguous():
            raise ValueError("startup eviction source must be contiguous")
        if source.data_ptr() % 4:
            raise ValueError("startup eviction source must be four-byte aligned")
        nbytes = self.eviction_bytes
        if source.numel() * source.element_size() < nbytes:
            raise ValueError(
                "startup eviction source is smaller than the required L2 eviction input"
            )
        # Contiguity and alignment make these views aliases, never copies.
        eviction = source.view(torch.uint8).view(-1)[:nbytes].view(torch.float32)
        previous = self._eviction
        self._eviction = eviction
        self.external_eviction = True
        for producer in list(self._cold_users):
            producer.eviction = eviction
            producer.owners = tuple(
                eviction if item is previous else item for item in producer.owners
            )

    def _freeze(self):
        self.captured = True
        if self._fixtures is not None:
            self._fixtures.frozen = True


def _runtime_startup_storage(runtime):
    storage = getattr(runtime, "_joint_startup_storage", None)
    if storage is None:
        storage = StartupTuningStorage(runtime.device)
        runtime._joint_startup_storage = storage
    return storage


@dataclass(frozen=True)
class JointStartupSpec:
    key: tuple
    profiles: tuple[tuple[int, int], ...]
    selector: torch.Tensor = field(compare=False, repr=False)
    state: object = field(compare=False, repr=False)
    inplace: bool
    rank: int


def joint_startup_spec(plan, *, profiles=None):
    """Describe supplied runtime profiles or measured finalists of a prepared launcher."""
    from ._joint_preparation import _JointExecutionState
    from ._gather_preparation import SelectedGatherState, startup_spec

    prepared = plan.prepared
    if prepared is not None and isinstance(prepared.state, SelectedGatherState):
        return startup_spec(plan, profiles)
    if prepared is None or not isinstance(prepared.state, _JointExecutionState):
        raise ValueError("startup tuning requires a prepared joint collective")
    state = prepared.state
    call = state.query.call
    family = call["transport"]
    geometry = (family, call["threads"], call["blocks"])
    cluster_size = call.get("joint_cluster_size", 1)
    validate_cluster_geometry(call["blocks"], call["threads"], cluster_size)
    if profiles is not None:
        profiles = tuple(profiles)
        if not profiles or any(
            not isinstance(profile, (tuple, list)) or len(profile) != 2
            for profile in profiles
        ):
            raise ValueError(
                "runtime profiles must be nonempty (mode, active CTA) pairs"
            )
        encoded = []
        for mode, active in profiles:
            selected_profile(
                family, call["blocks"], mode, active, cluster_size=cluster_size
            )
            encoded.append((JOINT_MODES[mode], active))
        if len(set(encoded)) != len(encoded):
            raise ValueError("runtime profiles must be distinct")
        profiles = tuple(encoded)
    else:
        measured = {}
        for record in prepared.finalists:
            assignment = record.assignment
            if (
                tuple(
                    assignment[name] for name in ("implementation", "threads", "blocks")
                )
                == geometry
            ):
                mode = assignment["submode"]
                active = assignment["active_blocks"]
                selected_profile(
                    family, call["blocks"], mode, active, cluster_size=cluster_size
                )
                measured[mode] = (JOINT_MODES[mode], active)
        modes = joint_modes(family, cluster_size=cluster_size)
        if set(measured) != set(modes):
            raise ValueError(
                "startup tuning requires measured finalists for every mode of the selected geometry"
            )
        profiles = tuple(measured[mode] for mode in modes)
    inplace = bool(plan.query.inplace)
    import torch.distributed as dist

    members = tuple(dist.get_process_group_ranks(state.runtime.exchange_group))
    key = (
        "joint_startup",
        1,
        state.query.world_size,
        members,
        *geometry,
        tuple(state.query.setup["shape"]),
        call["dtype"],
        call["size_packs"],
        state.workspace.capacity_packs,
        inplace,
        profiles,
    )
    if cluster_size != 1:
        key += (("cluster_size", cluster_size),)
    return JointStartupSpec(
        key, profiles, state.selector, state, inplace, state.query.rank
    )


class JointStartupTuner:
    """Captureable bounded race; retained owners must outlive its CUDA graph."""

    def __init__(self, specs, cache_policy, storage=None):
        from ._benchmark import _DTYPES, _ReductionInputs, _reduction_repetitions
        from ._gather_preparation import SelectedGatherState
        from ._joint_startup_fixtures import (
            StartupGatherInputs,
            StartupReductionInputs,
        )
        from ._joint_startup_cute import SCRATCH_BYTES, SIGNAL_BYTES, timing_launcher
        from .pcie_oneshot import (
            PCIeOneshotAllReduce,
            _require_collective_contract,
            _run_collective_preallocation_setup,
            _finish_collective_runtime_setup,
        )

        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("startup tuning requires at least one consumer")
        first = self.specs[0]
        state, runtime = first.state, first.state.runtime
        if any(
            spec.key != first.key
            or spec.rank != first.rank
            or spec.selector.device != first.selector.device
            or spec.state.runtime is not runtime
            for spec in self.specs
        ):
            raise ValueError(
                "startup consumers must share geometry, profiles, rank, and device"
            )
        if cache_policy not in ("cold", "warm"):
            raise ValueError("cache_policy must be cold or warm")
        self.key = first.key
        self.state, self.runtime = state, runtime
        self.gather = isinstance(state, SelectedGatherState)
        self.cache_policy = cache_policy
        if storage is None:
            storage = _runtime_startup_storage(runtime)
        if (
            not isinstance(storage, StartupTuningStorage)
            or storage.device != runtime.device
        ):
            raise ValueError("startup storage must belong to the runtime CUDA device")
        self.storage = storage
        _require_collective_contract(
            owner="joint startup tuner",
            exchange_group=runtime.exchange_group,
            contract=(self.key, cache_policy),
        )

        def prepare_local():
            with torch.cuda.device(runtime.device):
                self.peers = torch.empty(
                    runtime.world_size, dtype=torch.uint64, device=runtime.device
                )
                self.status = torch.full(
                    (1,), -1, dtype=torch.int32, device=runtime.device
                )
                self.samples = torch.empty(
                    (len(first.profiles), 21), dtype=torch.int64, device=runtime.device
                )
                self.profiles = torch.tensor(
                    [(mode << 16) | active for mode, active in first.profiles],
                    dtype=torch.uint32,
                    device=runtime.device,
                )
                query = state.query
                self.fixtures = storage.fixtures
                if self.gather:
                    self.inputs = StartupGatherInputs(self.fixtures, state)
                elif query.surface.endswith(
                    ".all_reduce"
                ) and not query.surface.startswith("PCIeTwoShotBF16."):
                    self.inputs = StartupReductionInputs(
                        self.fixtures,
                        state,
                        inplace=first.inplace,
                        dtype=_DTYPES[query.call["dtype"]],
                        repetitions=_reduction_repetitions(tuple(query.setup["shape"])),
                    )
                else:
                    self.fixtures = None
                    self.inputs = _ReductionInputs(state, inplace=first.inplace)
                    self.inputs.checked = self.inputs.out
                    self.inputs.mismatch = torch.empty_like(
                        self.inputs.out, dtype=torch.bool
                    )
                self.invalid = torch.zeros(
                    (1,), dtype=torch.int32, device=runtime.device
                )
                # The validator reads the int32 word; only its first byte is ever written.
                self.invalid_any = self.invalid.view(torch.bool)[0]
                self.eviction = self.eviction_sum = None
                if cache_policy == "cold":
                    self.eviction, self.eviction_sum = storage._cold_eviction(self)
                self.launchers = {
                    action: timing_launcher(
                        runtime.world_size,
                        runtime.rank,
                        action,
                        len(first.profiles),
                        runtime.device.index,
                    )
                    for action in ("reset", "validate", "begin", "end", "publish")
                }
            return True

        _run_collective_preallocation_setup(
            owner="joint startup tuner",
            exchange_group=runtime.exchange_group,
            setup=prepare_local,
        )
        with torch.cuda.device(runtime.device):
            # Races of one runtime replay one at a time on every rank, so they share
            # its timing exchange: its barrier counters only ever advance.
            shared = getattr(runtime, "_joint_startup_shared", None)
            if shared is None:
                shared = PCIeOneshotAllReduce._allocate_shared_buffer(
                    runtime.exchange_group,
                    SCRATCH_BYTES,
                    zero_fill=True,
                    ipc=runtime._ipc,
                )
                local_error = None
                try:
                    self.peers.copy_(
                        torch.tensor(
                            shared.peer_ptrs, dtype=torch.uint64, device=runtime.device
                        )
                    )
                except Exception as error:
                    local_error = error
                _finish_collective_runtime_setup(
                    owner="joint startup tuner",
                    exchange_group=runtime.exchange_group,
                    ipc=runtime._ipc,
                    shared=shared,
                    local_error=local_error,
                    detach_shared_ownership=lambda: None,
                )
                runtime._owned_buffers.append(shared)
                runtime._joint_startup_shared = shared
            else:
                self.peers.copy_(
                    torch.tensor(
                        shared.peer_ptrs, dtype=torch.uint64, device=runtime.device
                    )
                )
            self.shared = shared
        self.local_address = shared.local_ptr + SIGNAL_BYTES
        self.owners = (
            self.specs,
            storage,
            runtime,
            shared,
            self.inputs,
            self.fixtures,
            self.peers,
            self.status,
            self.samples,
            self.profiles,
            self.invalid,
            self.eviction,
            self.eviction_sum,
            self.launchers,
        )

    @property
    def checked_output(self):
        return self.inputs.checked

    @property
    def mismatch(self):
        return self.inputs.mismatch

    def memory_requirements(self):
        """Resident startup storage, with shared fixtures and eviction deduplicated."""
        from b12x.preparation import MemoryRequirements, PersistentMemory

        tensors = (
            self.inputs.source,
            self.inputs.out,
            *self.inputs.reference.seeds,
            *self.inputs.reference.expected,
            self.peers,
            self.status,
            self.samples,
            self.profiles,
            self.mismatch,
            self.invalid,
            None if self.storage.external_eviction else self.storage._eviction,
            self.storage._eviction_sum,
        )
        fixtures = self.storage._fixtures
        if fixtures is not None:
            # Frozen owners retain these allocations even when this producer
            # uses larger private fixtures and every shared-fixture user is gone.
            tensors += (fixtures.scratch,)
            for bank in fixtures.banks.values():
                tensors += (*bank.seeds, *bank.expected)
        entries = {}
        for tensor in tensors:
            if tensor is not None:
                storage = tensor.untyped_storage()
                key = (
                    "pcie-joint-startup-tensor",
                    str(tensor.device),
                    storage.data_ptr(),
                )
                entries[key] = PersistentMemory(key, storage.nbytes(), storage.nbytes())
        key = ("pcie-joint-startup-ipc", self.shared.local_ptr)
        entries[key] = PersistentMemory(key, self.shared.nbytes, self.shared.nbytes)
        return MemoryRequirements(persistent=tuple(entries.values()))

    def _timing(self, action, sample, selector):
        self.launchers[action](
            self.peers.data_ptr(),
            self.local_address,
            self.samples.data_ptr(),
            self.invalid.data_ptr(),
            self.profiles.data_ptr(),
            selector.data_ptr(),
            self.status.data_ptr(),
            sample,
        )

    def _collective(self):
        if self.gather:
            self.state.run_selected(self.inputs.source, self.inputs.out)
            return
        work = self.state.workspace
        self.state.launcher(
            work.peer_table.data_ptr(),
            work.signal_table.data_ptr(),
            work.shared.local_ptr,
            self.state.selector.data_ptr(),
            self.inputs.source.data_ptr(),
            self.inputs.out.data_ptr(),
            self.state.query.call["size_packs"],
            work.capacity_packs,
        )

    def run(self):
        """Enqueue the race without allocation, compilation, or host decisions."""
        with torch.cuda.device(self.runtime.device):
            if torch.cuda.is_current_stream_capturing():
                # A captured race reads its fixtures and eviction source by address from now on.
                self.storage._freeze()
            self._timing("reset", 0, self.state.selector)
            for mode, active in self.specs[0].profiles:
                self.state.selector.fill_((mode << 16) | active)
                for producer in range(3):
                    self.inputs.produce(producer)
                    self._collective()
                    torch.ne(
                        self.checked_output,
                        self.inputs.reference.expected[producer],
                        out=self.mismatch,
                    )
                    # A bool reduction needs no int32 copy of the mask, which a
                    # captured race would otherwise keep in the graph pool.
                    torch.any(self.mismatch, out=self.invalid_any)
                    self._timing("validate", 0, self.state.selector)
            for round_index in range(7):
                for producer in range(3):
                    for offset in range(len(self.specs[0].profiles)):
                        mode_index = (offset + round_index) % len(
                            self.specs[0].profiles
                        )
                        mode, active = self.specs[0].profiles[mode_index]
                        self.state.selector.fill_((mode << 16) | active)
                        if self.eviction is not None:
                            torch.sum(self.eviction, dim=0, out=self.eviction_sum)
                        self.inputs.produce(producer)
                        sample = mode_index * 21 + round_index * 3 + producer
                        self._timing("begin", sample, self.state.selector)
                        for _ in range(self.inputs.repetitions):
                            self._collective()
                        self._timing("end", sample, self.state.selector)
            for spec in self.specs:
                self._timing("publish", 0, spec.selector)


def prepare_joint_startup_tuner(specs, *, cache_policy="cold", storage=None):
    """Collectively allocate and compile a startup producer before capture freeze."""
    producer = JointStartupTuner(specs, cache_policy, storage)
    producer.run()
    torch.cuda.synchronize(producer.runtime.device)
    if producer.status.item() != 0:
        raise RuntimeError("joint startup fixture validation failed during preparation")
    for spec in producer.specs:
        from ._gather_preparation import SelectedGatherState

        if isinstance(spec.state, SelectedGatherState):
            spec.selector.fill_(spec.state.initial_profile)
            continue
        call = spec.state.query.call
        spec.selector.fill_(
            selected_profile(
                call["transport"],
                call["blocks"],
                call["submode"],
                call["active_blocks"],
                cluster_size=call.get("joint_cluster_size", 1),
            )
        )
    producer.status.fill_(-1)
    return producer
