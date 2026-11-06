"""Fixed-storage joint PCIe launches through an established oneshot channel."""

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan

from ._joint_profile import profile_index, profile_records, selected_profile
from ._oneshot_preparation import _OneshotExecutionState
from ._tuning import TUNING
from .pcie_oneshot import (
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _align_up,
    _finish_collective_runtime_setup,
    _is_current_stream_capturing,
    _require_collective_contract,
    _run_collective_preallocation_setup,
    _validate_plain_io,
)


def _layout(capacity, world):
    from ._joint_profile import JOINT_SIGNAL_BYTES, joint_slot_packs

    signals = _align_up(JOINT_SIGNAL_BYTES, IPC_SLAB_ALIGNMENT)
    slot = _align_up(joint_slot_packs(capacity, world) * 16, IPC_SLAB_ALIGNMENT)
    return signals, slot, signals + 2 * slot


@dataclass
class _JointWorkspace:
    shared: object
    peer_table: torch.Tensor
    signal_table: torch.Tensor
    capacity_packs: int
    signal_bytes: int
    slot_bytes: int


def prepare_workspace(runtime):
    """Collectively provision the channel before asynchronous preparation begins."""
    existing = getattr(runtime, "_joint_workspace", None)
    if existing is not None:
        return existing
    if _is_current_stream_capturing(runtime.device):
        raise RuntimeError("joint workspace must be prepared before graph capture")
    capacity = runtime.max_size // 16
    signals, slot, nbytes = _layout(capacity, runtime.world_size)
    _require_collective_contract(
        owner="joint PCIe workspace",
        exchange_group=runtime.exchange_group,
        contract=(runtime.world_size, capacity, signals, slot),
    )
    with torch.cuda.device(runtime.device):
        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            runtime.exchange_group,
            nbytes,
            zero_fill=True,
            ipc=runtime._ipc,
        )
    workspace, error = None, None
    try:
        pointers = tuple(shared.peer_ptrs)
        pointers += (pointers[runtime.rank],) * (16 - runtime.world_size)
        with torch.cuda.device(runtime.device):
            workspace = _JointWorkspace(
                shared,
                torch.tensor(
                    [
                        [p + signals + index * slot for p in pointers]
                        for index in range(2)
                    ],
                    dtype=torch.uint64,
                    device=runtime.device,
                ),
                torch.tensor(pointers, dtype=torch.uint64, device=runtime.device),
                capacity,
                signals,
                slot,
            )
    except Exception as exc:
        error = exc
    _finish_collective_runtime_setup(
        owner="joint PCIe workspace",
        exchange_group=runtime.exchange_group,
        ipc=runtime._ipc,
        shared=shared,
        local_error=error,
        detach_shared_ownership=lambda: None,
    )
    runtime._owned_buffers.append(shared)
    runtime._joint_workspace = workspace
    return workspace


def _workspace(runtime):
    workspace = getattr(runtime, "_joint_workspace", None)
    if workspace is None:
        raise RuntimeError(
            "joint workspace must be provisioned by coordinated channel setup"
        )
    return workspace


@program_cache(scope="preparation")
def compile_joint(family, world, rank, threads, max_blocks, ordinal):
    from ._joint_cute import get_joint_launcher

    return get_joint_launcher(family, world, rank, threads, max_blocks, ordinal)


@program_cache(scope="preparation")
def compile_joint_cluster(
    family, world, rank, threads, max_blocks, cluster_size, ordinal
):
    from ._joint_cluster_cute import get_joint_cluster_launcher

    return get_joint_cluster_launcher(
        family, world, rank, threads, max_blocks, cluster_size, ordinal
    )


class _JointExecutionState(_OneshotExecutionState):
    def __init__(self, query, runtime, launcher, workspace):
        self.query, self.runtime, self.launcher = query, runtime, launcher
        self.launchers = {tuple(slot): launcher for slot in query.setup["variants"]}
        self._bound_input = None
        self.workspace = workspace
        call = query.call
        self.cluster_size = call.get("joint_cluster_size", 1)
        self.profiles = profile_records(
            call["transport"], call["blocks"], cluster_size=self.cluster_size
        )
        selected = selected_profile(
            call["transport"],
            call["blocks"],
            call["submode"],
            call["active_blocks"],
            cluster_size=self.cluster_size,
        )
        self.selector = torch.tensor(
            [selected], dtype=torch.uint32, device=runtime.device
        )
        self._bound = False

    def bind_input(self, inp):
        if _is_current_stream_capturing(self.runtime.device):
            raise RuntimeError("joint input must be bound before graph capture")
        self._check_input(inp)
        self._bound = True

    def select_profile(self, submode, active_blocks):
        """Collectively drain the channel and select a prepared mode and CTA count.

        All ranks must call with identical arguments after stopping submissions.
        Captured graphs retain the selector address and observe its updated value.
        """
        if self.runtime._closed:
            raise RuntimeError("runtime is closed")
        if _is_current_stream_capturing(self.runtime.device):
            raise RuntimeError("joint profiles cannot change during graph capture")
        call = self.query.call
        selected = _run_collective_preallocation_setup(
            owner="joint PCIe profile validation",
            exchange_group=self.runtime.exchange_group,
            setup=lambda: selected_profile(
                call["transport"],
                call["blocks"],
                submode,
                active_blocks,
                cluster_size=self.cluster_size,
            ),
        )
        torch.cuda.synchronize(self.runtime.device)
        _require_collective_contract(
            owner="joint PCIe profile",
            exchange_group=self.runtime.exchange_group,
            contract=(
                call["transport"],
                call["threads"],
                call["blocks"],
                submode,
                active_blocks,
            )
            + (
                () if self.cluster_size == 1 else (("cluster_size", self.cluster_size),)
            ),
        )
        self.selector.fill_(selected)
        torch.cuda.synchronize(self.runtime.device)

    def run_plain(self, inp, out):
        if self.runtime._closed:
            raise RuntimeError("runtime is closed")
        if not self._bound:
            raise RuntimeError("joint input was not bound during preparation")
        self.runtime._check_stream()
        self._check_input(inp)
        _validate_plain_io(inp, out)
        work = self.workspace
        with torch.cuda.device(self.runtime.device):
            self.launcher(
                work.peer_table.data_ptr(),
                work.signal_table.data_ptr(),
                work.shared.local_ptr,
                self.selector.data_ptr(),
                inp.data_ptr(),
                out.data_ptr(),
                self.query.call["size_packs"],
                work.capacity_packs,
            )


def plan(query, *, runtime, override=None):
    call = query.call
    cluster_size = call.get("joint_cluster_size", 1)
    from ._joint_profile import validate_cluster_geometry

    validate_cluster_geometry(call["blocks"], call["threads"], cluster_size)
    profile_index(
        call["transport"],
        call["blocks"],
        call["submode"],
        call["active_blocks"],
        cluster_size=cluster_size,
    )
    if call["dtype"] != "bfloat16" or not call["stage_input"]:
        raise ValueError("joint PCIe all-reduce requires staged BF16 input")
    if not 1 <= call["blocks"] <= query.setup.get("joint_max_blocks", 0):
        raise ValueError("joint PCIe all-reduce grid exceeds resident CTA capacity")
    family = call["transport"].removeprefix("joint_")
    args = (family, query.world_size, query.rank, call["threads"], call["blocks"])
    resident = (
        int(query.setup["rank_data_nbytes"])
        + int(query.setup["owned_slab_nbytes"]) * int(query.setup["owned_slab_count"])
        + int(query.setup["joint_workspace_nbytes"])
    )
    profiles_bytes = 4

    def materialize(selection, device):
        if cluster_size == 1:
            launcher = load_programs(compile_joint(*args, device.ordinal))
        else:
            from ._joint_cluster_cute import validate_cluster_residency

            validate_cluster_geometry(
                call["blocks"], call["threads"], cluster_size, device.identity
            )
            launcher = load_programs(
                compile_joint_cluster(*args, cluster_size, device.ordinal)
            )
            validate_cluster_residency(
                launcher, call["blocks"], call["threads"], cluster_size
            )
        return _JointExecutionState(query, runtime, launcher, _workspace(runtime))

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._joint_preparation:compile_joint"
                if cluster_size == 1
                else "b12x.comm.pcie._joint_preparation:compile_joint_cluster",
                *args,
                *((cluster_size,) if cluster_size != 1 else ()),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(
                    ("pcie-oneshot-runtime", id(runtime)), resident, resident
                ),
                PersistentMemory(
                    ("pcie-joint-profile", id(query)), profiles_bytes, profiles_bytes
                ),
            )
        ),
        _materialize=materialize,
        _device=runtime.device,
    )
