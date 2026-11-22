"""Prepared SM120 TMA relative decode/verify attention: declaration, compilation, execution."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.utils.device import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import TUNING, RelativeDecodeConfig, RelativeDecodeQuery

_FORWARD_ID = "attention.relative_decode.forward"
_MERGE_ID = "attention.relative_decode.merge"
_COMPILE_VERSION = 3

_FORWARD_OPERANDS = (
    ("q", 3, cutlass.BFloat16, 16),
    ("new_k", 3, cutlass.BFloat16, 16),
    ("new_v", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, cutlass.Uint8, 16),
    ("v_cache", 4, cutlass.Uint8, 16),
    ("staged_k", 4, cutlass.Uint8, 16),
    ("staged_v", 4, cutlass.Uint8, 16),
    ("page_table", 2, cutlass.Int32, 4),
    ("state_rows", 1, cutlass.Int64, 8),
    ("starts", 1, cutlass.Int64, 8),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("bias", 3, cutlass.BFloat16, 2),
    ("gate", 3, cutlass.BFloat16, 8),
    ("sink", 2, cutlass.BFloat16, 2),
    ("gated", 3, cutlass.BFloat16, 8),
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
)

_MERGE_OPERANDS = (
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
    ("starts", 1, cutlass.Int64, 8),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("gate", 3, cutlass.BFloat16, 8),
    ("sink", 2, cutlass.BFloat16, 2),
    ("gated", 3, cutlass.BFloat16, 8),
)

_LAUNCH_ONLY_FIELDS = frozenset(("max_batch", "max_total_q", "sm_count", "max_splits"))


def _fake_operand(dtype, rank: int, align: int):
    from cutlass.cute.runtime import make_fake_tensor

    shape = tuple(cute.sym_int(32) for _ in range(rank))
    strides = tuple(1 if idx == rank - 1 else cute.sym_int(64) for idx in range(rank))
    return make_fake_tensor(dtype, shape, strides, assumed_align=align)


def _kernel_operand(tensor: torch.Tensor, dtype, align: int):
    storage = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
    converted = from_dlpack(storage, assumed_align=align)
    converted.element_type = dtype
    return converted.mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _validate_operand(name, tensor, *, device, rank, align, aligned_strides=False):
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.ndim != rank or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have rank {rank} and a unit inner stride")
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")
    if aligned_strides and any(
        stride * tensor.element_size() % align for stride in tensor.stride()[:-1]
    ):
        raise ValueError(f"{name} outer strides must be {align}-byte aligned")


def _forward_kernel(query: RelativeDecodeQuery, config: RelativeDecodeConfig):
    from ._kernel import RelativeDecodeKernel

    return RelativeDecodeKernel(
        group_size=query.group_size,
        max_q_per_req=query.max_q_per_req,
        head_dim=query.head_dim,
        is_window=query.is_window,
        commit_ring=query.commit_ring,
        page_or_window=query.page_or_window,
        window_left=query.window_left,
        rel_extent=query.rel_extent,
        split_kv=config.split_kv,
        stage_rows=config.stage_rows,
        num_stages=config.num_stages,
        num_consumer_warps=config.num_consumer_warps,
        min_tiles_per_split=config.min_tiles_per_split,
    )


def _merge_kernel(query: RelativeDecodeQuery, config: RelativeDecodeConfig):
    from ._kernel import RelativeDecodeMergeKernel

    return RelativeDecodeMergeKernel(
        num_q_heads=query.num_q_heads,
        max_q_per_req=query.max_q_per_req,
        is_window=query.is_window,
        window_left=query.window_left,
        stage_rows=config.stage_rows,
        min_tiles_per_split=config.min_tiles_per_split,
    )


def _compile_key(query: RelativeDecodeQuery, config: RelativeDecodeConfig) -> tuple:
    fields = {**dict(TUNING.encode_query(query)), **config.to_dict()}
    excluded = _LAUNCH_ONLY_FIELDS | {"autotune_effort"}
    return tuple(sorted((k, v) for k, v in fields.items() if k not in excluded))


@program_cache(scope="preparation")
def compile_relative_decode(query_payload, config_payload, ordinal):
    """Compile the forward kernel (and split merge) from metadata only."""
    query = RelativeDecodeQuery(**dict(query_payload))
    config = RelativeDecodeConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    key = _compile_key(query, config)
    programs = {}
    with torch.cuda.device(ordinal):
        stream = current_cuda_stream()
        fwd_args = [
            _fake_operand(dt, rank, align) for _, rank, dt, align in _FORWARD_OPERANDS
        ]
        programs["forward"] = b12x_compile(
            _forward_kernel(query, config),
            *fwd_args,
            Float32(1.0),
            stream,
            compile_spec=KernelCompileSpec.from_key(_FORWARD_ID, _COMPILE_VERSION, key),
        )
        if config.split_kv:
            merge_args = [
                _fake_operand(dt, rank, align) for _, rank, dt, align in _MERGE_OPERANDS
            ]
            programs["merge"] = b12x_compile(
                _merge_kernel(query, config),
                *merge_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    _MERGE_ID, _COMPILE_VERSION, key
                ),
            )
    return programs


def scratch_specs(query: RelativeDecodeQuery, config: RelativeDecodeConfig, device):
    """FP32 split partials sized for the layer family's row capacity."""
    if not config.split_kv:
        return ()
    rows, heads, splits = query.max_total_q, query.num_q_heads, config.max_splits
    return (
        ScratchBufferSpec(
            name="relative_decode.o_part",
            shape=(rows, heads, splits, query.head_dim),
            dtype=torch.float32,
            device=device,
        ),
        ScratchBufferSpec(
            name="relative_decode.lse_part",
            shape=(rows, heads, splits),
            dtype=torch.float32,
            device=device,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    q: torch.Tensor
    new_k: torch.Tensor
    new_v: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    staged_k: torch.Tensor
    staged_v: torch.Tensor
    page_table: torch.Tensor
    state_rows: torch.Tensor
    starts: torch.Tensor
    cu_seqlens_q: torch.Tensor
    bias: torch.Tensor
    gate: torch.Tensor
    sink: torch.Tensor
    gated: torch.Tensor
    o_part: torch.Tensor
    lse_part: torch.Tensor
    softmax_scale: float


@dataclass(frozen=True)
class PreparedRelativeDecode:
    query: RelativeDecodeQuery
    config: RelativeDecodeConfig
    device: torch.device
    programs: dict
    dummy_part: tuple[torch.Tensor, torch.Tensor]
    dummy_staged: torch.Tensor
    dummy_page_table: torch.Tensor
    dummy_state_rows: torch.Tensor

    def bind(
        self,
        *,
        plan: Plan,
        scratch,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        starts: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        bias: torch.Tensor,
        gate: torch.Tensor,
        sink: torch.Tensor,
        gated: torch.Tensor,
        staged_k: torch.Tensor | None = None,
        staged_v: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        state_rows: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ) -> Binding:
        query, cfg = self.query, self.config
        batch = int(starts.shape[0])
        if not 0 < batch <= query.max_batch:
            raise ValueError(f"batch {batch} exceeds planned max_batch {query.max_batch}")
        if tuple(cu_seqlens_q.shape) != (batch + 1,):
            raise ValueError("cu_seqlens_q shape disagrees with starts")
        if query.is_window:
            if state_rows is None or tuple(state_rows.shape) != (batch,):
                raise ValueError("windowed relative_decode requires state_rows [B]")
            if page_table is None:
                page_table = self.dummy_page_table
            if not query.commit_ring:
                if staged_k is None or staged_v is None:
                    raise ValueError("uncommitted windowed relative_decode requires staged_k/staged_v")
            else:
                staged_k = self.dummy_staged if staged_k is None else staged_k
                staged_v = self.dummy_staged if staged_v is None else staged_v
        else:
            if page_table is None or page_table.shape[0] != batch:
                raise ValueError("paged relative_decode requires page_table [B, max_pages]")
            state_rows = self.dummy_state_rows if state_rows is None else state_rows
            staged_k = self.dummy_staged if staged_k is None else staged_k
            staged_v = self.dummy_staged if staged_v is None else staged_v

        if sink.ndim == 1:
            sink = sink.view(-1, query.num_q_heads)

        tensors = {
            "q": q,
            "new_k": new_k,
            "new_v": new_v,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "staged_k": staged_k,
            "staged_v": staged_v,
            "page_table": page_table,
            "state_rows": state_rows,
            "starts": starts,
            "cu_seqlens_q": cu_seqlens_q,
            "bias": bias,
            "gate": gate,
            "sink": sink,
            "gated": gated,
        }
        aligned_stride_names = {"q", "new_k", "new_v", "k_cache", "v_cache", "gate", "gated"}
        if query.is_window and not query.commit_ring:
            aligned_stride_names.update(("staged_k", "staged_v"))
        for name, rank, _, align in _FORWARD_OPERANDS:
            if name in tensors:
                _validate_operand(
                    name,
                    tensors[name],
                    device=self.device,
                    rank=rank,
                    align=align,
                    aligned_strides=name in aligned_stride_names,
                )

        if k_cache.dtype not in (torch.float8_e4m3fn, torch.uint8) or v_cache.dtype not in (
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            raise ValueError("relative_decode requires FP8 E4M3 (or uint8 storage) KV caches")
        if tuple(k_cache.shape[1:]) != (query.page_or_window, query.num_kv_heads, query.head_dim):
            raise ValueError("k_cache shape disagrees with plan")
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache shapes must match")
        if q.shape[0] > query.max_total_q:
            raise ValueError(f"q rows {q.shape[0]} exceed planned max_total_q {query.max_total_q}")

        if cfg.split_kv:
            buffers = tuple(scratch)
            if len(buffers) != 2:
                raise ValueError("split relative_decode binds two scratch buffers")
            shapes = [spec.shape for spec in scratch_specs(query, cfg, self.device)]
            o_part, lse_part = (
                buf.view(torch.float32).view(shape)
                for buf, shape in zip(buffers, shapes, strict=True)
            )
        else:
            o_part, lse_part = self.dummy_part

        scale = float(softmax_scale) if softmax_scale is not None else query.head_dim**-0.5
        return Binding(
            plan=plan,
            q=q,
            new_k=new_k,
            new_v=new_v,
            k_cache=k_cache,
            v_cache=v_cache,
            staged_k=staged_k,
            staged_v=staged_v,
            page_table=page_table,
            state_rows=state_rows,
            starts=starts,
            cu_seqlens_q=cu_seqlens_q,
            bias=bias,
            gate=gate,
            sink=sink,
            gated=gated,
            o_part=o_part,
            lse_part=lse_part,
            softmax_scale=scale,
        )

    def run(self, binding: Binding) -> torch.Tensor:
        cfg = self.config
        tensors = {
            "q": binding.q,
            "new_k": binding.new_k,
            "new_v": binding.new_v,
            "k_cache": binding.k_cache,
            "v_cache": binding.v_cache,
            "staged_k": binding.staged_k,
            "staged_v": binding.staged_v,
            "page_table": binding.page_table,
            "state_rows": binding.state_rows,
            "starts": binding.starts,
            "cu_seqlens_q": binding.cu_seqlens_q,
            "bias": binding.bias,
            "gate": binding.gate,
            "sink": binding.sink,
            "gated": binding.gated,
            "o_part": binding.o_part,
            "lse_part": binding.lse_part,
        }
        with torch.cuda.device(self.device):
            stream = current_cuda_stream()
            run_compiled(
                self.programs["forward"],
                (
                    *(
                        _kernel_operand(tensors[name], dt, align)
                        for name, _, dt, align in _FORWARD_OPERANDS
                    ),
                    Float32(binding.softmax_scale),
                    stream,
                ),
            )
            if cfg.split_kv:
                run_compiled(
                    self.programs["merge"],
                    (
                        *(
                            _kernel_operand(tensors[name], dt, align)
                            for name, _, dt, align in _MERGE_OPERANDS
                        ),
                        stream,
                    ),
                )
        return binding.gated


def make_plan(caps, *, invocation=None, override=None) -> Plan:
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be relative_decode.Caps")
    invocation = FrozenMapping(invocation or {})
    if invocation:
        raise ValueError("relative_decode declarations carry no invocation metadata")
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    max_total_q = (
        int(caps.max_total_q)
        if caps.max_total_q is not None
        else int(caps.max_batch) * int(caps.max_q_per_req)
    )
    query = RelativeDecodeQuery(
        num_q_heads=int(caps.num_q_heads),
        num_kv_heads=int(caps.num_kv_heads),
        head_dim=int(caps.head_dim),
        is_window=bool(caps.is_window),
        commit_ring=bool(caps.commit_ring),
        page_or_window=int(caps.page_or_window),
        window_left=int(caps.window_left),
        rel_extent=int(caps.rel_extent),
        max_batch=int(caps.max_batch),
        max_q_per_req=int(caps.max_q_per_req),
        max_total_q=max_total_q,
        sm_count=int(sm_count),
    )
    TUNING.validate_query(query, None)
    compile_query = TUNING.encode_query(replace(query, autotune_effort="default"))

    def compile_jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.relative_decode._preparation:compile_relative_decode",
                compile_query,
                config.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def memory(config, detected):
        TUNING.validate_config(query, config, None)
        return MemoryRequirements(
            scratch=scratch_specs(query, config, torch.device("cuda", detected.ordinal))
        )

    def materialize(selection, detected):
        config = selection.config
        dev = torch.device("cuda", detected.ordinal)
        programs = compile_relative_decode(
            compile_query, config.to_dict(), detected.ordinal
        )
        load_programs(programs)
        dummy_part = (
            torch.zeros(
                (1, query.num_q_heads, 1, query.head_dim),
                dtype=torch.float32,
                device=dev,
            ),
            torch.zeros((1, query.num_q_heads, 1), dtype=torch.float32, device=dev),
        )
        state = PreparedRelativeDecode(
            query=query,
            config=config,
            device=dev,
            programs=programs,
            dummy_part=dummy_part,
            dummy_staged=torch.zeros((1, 1, query.num_kv_heads, query.head_dim), dtype=torch.uint8, device=dev),
            dummy_page_table=torch.zeros((1, 1), dtype=torch.int32, device=dev),
            dummy_state_rows=torch.zeros((1,), dtype=torch.int64, device=dev),
        )
        return attach_programs(state, *programs.values())

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=device,
    )


__all__ = [
    "Binding",
    "PreparedRelativeDecode",
    "compile_relative_decode",
    "make_plan",
    "scratch_specs",
]
