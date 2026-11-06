"""Fixed-geometry column gathers with persistent device-selected profiles."""

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import MemoryRequirements, PersistentMemory, Plan
from ._tuning import TUNING

GATHER_MODES = {
    "joint": ("barrier", "word_exchange"),
    "selected_barrier": ("barrier",),
    "selected_word_exchange": ("word_exchange",),
}


def selected_gather_profile(transport, blocks, mode, active):
    if transport not in GATHER_MODES or mode not in GATHER_MODES[transport]:
        raise ValueError("profile does not belong to the selected gather transport")
    if (
        type(active) is not int
        or type(blocks) is not int
        or not 1 <= active <= blocks <= 64
    ):
        raise ValueError("active gather CTAs must fit the physical grid")
    return (int(mode == "word_exchange") << 16) | active


def validate_query(query):
    call = query.call
    shape = call["shape"]
    if (
        len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or call["dtype"]
        not in ("torch.float16", "torch.bfloat16", "torch.float8_e4m3fn")
    ):
        raise ValueError(
            "selected gather requires a positive 3D FP16, BF16, or E4M3 input"
        )
    element = 1 if call["dtype"] == "torch.float8_e4m3fn" else 2
    row_bytes = prod(shape[1:]) * element
    size = shape[0] * row_bytes
    if (
        shape[0] > query.setup["max_batch_size"]
        or shape[1] * query.world_size != query.setup["total_heads"]
        or shape[2] != query.setup["query_head_dim"]
    ):
        raise ValueError(
            "selected gather shape exceeds or differs from its channel geometry"
        )
    transport = call["gather_transport"]
    selected_gather_profile(
        transport, call["block_limit"], call["submode"], call["active_blocks"]
    )
    if (
        call["threads"] not in (32, 64, 128, 256, 512)
        or row_bytes % 16
        or row_bytes < 16
        or shape[0] <= 0
        or not call.get("disjoint_output", False)
        or not call.get("pack_aligned", False)
        or tuple(call["stride"]) != (prod(shape[1:]), shape[2], 1)
        or tuple(call["out_shape"]) != (shape[0], shape[1] * query.world_size, shape[2])
        or tuple(call["out_stride"])
        != (prod(shape[1:]) * query.world_size, shape[2], 1)
    ):
        raise ValueError(
            "selected gather requires disjoint contiguous 16-byte-aligned rows"
        )
    if transport != "selected_barrier":
        chunks = query.setup["exchange_chunks"]
        if (
            query.world_size not in (2, 4)
            or not 1 <= chunks <= 64
            or size > chunks * 256
            or (transport == "selected_word_exchange" and call["threads"] > 256)
            or query.setup["exchange_offset"] < 0
            or query.setup["exchange_offset"] % 256
            or query.setup["exchange_offset"] + query.world_size * chunks * 512
            > query.setup["slot_bytes"]
            or query.setup["exchange_counter_offset"] != 2 * query.setup["slot_bytes"]
        ):
            raise ValueError(
                "selected word exchange requires owned TP2/TP4 receive storage within 16 KiB"
            )


@dataclass
class SelectedGatherState:
    query: object
    runtime: object
    launcher: object
    selector: torch.Tensor

    @property
    def initial_profile(self):
        c = self.query.call
        return selected_gather_profile(
            c["gather_transport"], c["block_limit"], c["submode"], c["active_blocks"]
        )

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("selected gather plan belongs to a different channel")

    def run_selected(self, source, output):
        c, runtime = self.query.call, self.runtime
        if tuple(source.shape) != tuple(c["shape"]) or str(source.dtype) != c["dtype"]:
            raise ValueError(
                "selected gather input differs from the prepared shape or dtype"
            )
        self.launcher(
            source.data_ptr(),
            output.data_ptr(),
            self.selector.data_ptr(),
            runtime._staging0_ptrs,
            runtime._signal_ptrs,
            source.shape[0],
            runtime._slot_bytes,
            runtime._exchange_offset,
            runtime._exchange_counter_offset,
        )
        return output

    def run(self, *, local_input, out):
        return self.runtime._all_gather_heads_on_device(
            local_input,
            out,
            state=self,
            threads=self.query.call["threads"],
            block_limit=self.query.call["block_limit"],
        )


def plan(query, *, runtime, invocation, override):
    from ._runtime_gather_cute import compile_selected_gather

    validate_query(query)
    if invocation:
        raise ValueError("selected gather invocation semantics belong in its query")
    for field in (
        "max_batch_size",
        "total_heads",
        "query_head_dim",
        "slot_bytes",
        "exchange_chunks",
        "exchange_offset",
        "exchange_counter_offset",
    ):
        name = (
            field
            if field in ("max_batch_size", "total_heads", "query_head_dim")
            else "_" + field
        )
        if query.setup[field] != getattr(runtime, name):
            raise ValueError(f"selected gather {field} differs from its runtime owner")
    c = query.call
    element = 1 if c["dtype"] == "torch.float8_e4m3fn" else 2
    args = (
        query.world_size,
        query.rank,
        prod(c["shape"][1:]) * element // 4,
        c["threads"],
        c["block_limit"],
        query.setup["exchange_chunks"],
        c["gather_transport"].removeprefix("selected_"),
    )

    def materialize(selection, device):
        launcher = load_programs(compile_selected_gather(*args))
        value = selected_gather_profile(
            c["gather_transport"], c["block_limit"], c["submode"], c["active_blocks"]
        )
        selector = torch.tensor([value], dtype=torch.uint32, device=runtime.device)
        return SelectedGatherState(query, runtime, launcher, selector)

    resident = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._runtime_gather_cute:compile_selected_gather", *args
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(("pcie-dcp-runtime", id(runtime)), resident, resident),
                PersistentMemory(("pcie-gather-selector", id(query)), 4, 4),
            )
        ),
        _materialize=materialize,
        _device=runtime.device,
    )


class GatherInputs:
    """Exact rank-interleaved fixtures with caller-independent persistent storage."""

    def __init__(self, state):
        q = state.query
        shape, dtype = (
            tuple(q.call["shape"]),
            getattr(torch, q.call["dtype"].removeprefix("torch.")),
        )
        # Byte fixtures exercise FP8 payloads and FP32 bit views without arithmetic conversion.
        rows, row_bytes = shape[0], prod(shape[1:]) * dtype.itemsize
        values = (
            torch.arange(
                q.world_size * rows * row_bytes, device=state.runtime.device
            ).reshape(q.world_size, rows, row_bytes)
            % 251
        ).to(torch.uint8)
        cases = (values, torch.zeros_like(values), 255 - values)
        self.source = torch.empty(shape, dtype=dtype, device=state.runtime.device)
        self.out = torch.empty(
            tuple(q.call["out_shape"]), dtype=dtype, device=state.runtime.device
        )
        self.reference = SimpleNamespace(
            seeds=tuple(case[q.rank].clone().view(dtype).view(shape) for case in cases),
            expected=tuple(
                case.permute(1, 0, 2).contiguous().view(torch.uint8).reshape(-1)
                for case in cases
            ),
        )
        self.repetitions = 1

    def produce(self, index):
        self.source.copy_(self.reference.seeds[index])


def startup_spec(plan, profiles):
    from ._joint_startup import JointStartupSpec
    import torch.distributed as dist

    state = plan.prepared.state
    c = state.query.call
    if profiles is None or not profiles:
        raise ValueError("gather startup tuning requires explicit offline profiles")
    encoded = tuple(
        selected_gather_profile(c["gather_transport"], c["block_limit"], mode, active)
        for mode, active in profiles
    )
    if len(set(encoded)) != len(encoded):
        raise ValueError("runtime gather profiles must be distinct")
    pairs = tuple((value >> 16, value & 65535) for value in encoded)
    key = (
        "gather_startup",
        1,
        state.query.world_size,
        tuple(dist.get_process_group_ranks(state.runtime.exchange_group)),
        c["gather_transport"],
        c["threads"],
        c["block_limit"],
        tuple(c["shape"]),
        c["dtype"],
        state.runtime._slot_bytes,
        state.runtime._exchange_chunks,
        pairs,
    )
    return JointStartupSpec(key, pairs, state.selector, state, False, state.query.rank)
