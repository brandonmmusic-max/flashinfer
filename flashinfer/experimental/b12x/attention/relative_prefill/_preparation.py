"""Prepared SM120 4D-TMA relative prefill attention: declaration, compilation, execution."""

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

from ._tuning import TUNING, RelativePrefillConfig, RelativePrefillQuery

_FORWARD_ID = "attention.relative_prefill.forward"
_MERGE_ID = "attention.relative_prefill.merge"
_COMPILE_VERSION = 5

_FORWARD_OPERANDS = (
    ("q", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, cutlass.Uint8, 16),
    ("v_cache", 4, cutlass.Uint8, 16),
    ("page_table", 2, cutlass.Int32, 4),
    ("lengths", 1, cutlass.Int32, 4),
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
    ("lengths", 1, cutlass.Int32, 4),
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


def _forward_kernel(query: RelativePrefillQuery, config: RelativePrefillConfig):
    from ._kernel import RelativePrefillKernel

    return RelativePrefillKernel(
        num_q_heads=query.num_q_heads,
        num_kv_heads=query.num_kv_heads,
        head_dim=query.head_dim,
        page_size=query.page_size,
        rel_extent=query.rel_extent,
        window_left=query.window_left,
        q_tile_tokens=config.q_tile_tokens,
        grouped=config.grouped,
        stage_rows=config.stage_rows,
        bn_sub=config.bn_sub,
        num_stages=config.num_stages,
        split_kv=config.split_kv,
        min_stages_per_split=config.min_stages_per_split,
        q_in_smem=config.q_in_smem,
    )


def _merge_kernel(query: RelativePrefillQuery, config: RelativePrefillConfig):
    from ._kernel import RelativePrefillMergeKernel

    return RelativePrefillMergeKernel(
        num_q_heads=query.num_q_heads,
        q_tile_tokens=config.q_tile_tokens,
        stage_rows=config.stage_rows,
        min_stages_per_split=config.min_stages_per_split,
    )


def _compile_key(query: RelativePrefillQuery, config: RelativePrefillConfig) -> tuple:
    fields = {**dict(TUNING.encode_query(query)), **config.to_dict()}
    excluded = _LAUNCH_ONLY_FIELDS | {"autotune_effort"}
    return tuple(sorted((k, v) for k, v in fields.items() if k not in excluded))


@program_cache(scope="preparation")
def compile_relative_prefill(query_payload, config_payload, ordinal):
    query = RelativePrefillQuery(**dict(query_payload))
    config = RelativePrefillConfig.from_config(FrozenMapping(config_payload))
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


def scratch_specs(query: RelativePrefillQuery, config: RelativePrefillConfig, device):
    if not config.split_kv:
        return ()
    rows, heads, splits = query.max_total_q, query.num_q_heads, config.max_splits
    return (
        ScratchBufferSpec(
            name="relative_prefill.o_part",
            shape=(rows, heads, splits, query.head_dim),
            dtype=torch.float32,
            device=device,
        ),
        ScratchBufferSpec(
            name="relative_prefill.lse_part",
            shape=(rows, heads, splits),
            dtype=torch.float32,
            device=device,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    lengths: torch.Tensor
    cu_seqlens_q: torch.Tensor
    bias: torch.Tensor
    gate: torch.Tensor
    sink: torch.Tensor
    gated: torch.Tensor
    o_part: torch.Tensor
    lse_part: torch.Tensor
    softmax_scale: float


@dataclass(frozen=True)
class PreparedRelativePrefill:
    query: RelativePrefillQuery
    config: RelativePrefillConfig
    device: torch.device
    programs: dict
    dummy_part: tuple[torch.Tensor, torch.Tensor]

    def bind(
        self,
        *,
        plan: Plan,
        scratch,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        lengths: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        bias: torch.Tensor,
        gate: torch.Tensor,
        sink: torch.Tensor,
        gated: torch.Tensor,
        softmax_scale: float | None = None,
    ) -> Binding:
        query, cfg = self.query, self.config
        batch = int(lengths.shape[0])
        if not 0 < batch <= query.max_batch:
            raise ValueError(f"batch {batch} exceeds planned max_batch {query.max_batch}")
        if tuple(cu_seqlens_q.shape) != (batch + 1,):
            raise ValueError("cu_seqlens_q shape disagrees with lengths")
        if query.window_left < 0 and page_table.shape[0] != batch:
            raise ValueError("page_table batch disagrees with lengths")
        if sink.ndim == 1:
            sink = sink.view(-1, query.num_q_heads)

        tensors = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "page_table": page_table,
            "lengths": lengths,
            "cu_seqlens_q": cu_seqlens_q,
            "bias": bias,
            "gate": gate,
            "sink": sink,
            "gated": gated,
        }
        aligned_stride_names = {"q", "k_cache", "v_cache", "gate", "gated"}
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
            raise ValueError("relative_prefill requires FP8 E4M3 (or uint8 storage) KV caches")
        if query.window_left >= 0:
            if (
                k_cache.shape[1] <= 0
                or k_cache.shape[1] % cfg.stage_rows != 0
                or tuple(k_cache.shape[2:]) != (query.num_kv_heads, query.head_dim)
            ):
                raise ValueError("staged k_cache shape disagrees with plan")
        elif tuple(k_cache.shape[1:]) != (query.page_size, query.num_kv_heads, query.head_dim):
            raise ValueError("k_cache shape disagrees with plan")
        if k_cache.shape != v_cache.shape:
            raise ValueError("k_cache and v_cache shapes must match")
        if q.shape[0] > query.max_total_q:
            raise ValueError(f"q rows {q.shape[0]} exceed planned max_total_q {query.max_total_q}")

        if cfg.split_kv:
            buffers = tuple(scratch)
            if len(buffers) != 2:
                raise ValueError("split relative_prefill binds two scratch buffers")
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
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            lengths=lengths,
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
            "k_cache": binding.k_cache,
            "v_cache": binding.v_cache,
            "page_table": binding.page_table,
            "lengths": binding.lengths,
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
        raise TypeError("caps must be relative_prefill.Caps")
    invocation = FrozenMapping(invocation or {})
    if invocation:
        raise ValueError("relative_prefill declarations carry no invocation metadata")
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    query = RelativePrefillQuery(
        num_q_heads=int(caps.num_q_heads),
        num_kv_heads=int(caps.num_kv_heads),
        head_dim=int(caps.head_dim),
        page_size=int(caps.page_size),
        rel_extent=int(caps.rel_extent),
        max_batch=int(caps.max_batch),
        max_total_q=int(caps.max_total_q),
        sm_count=int(sm_count),
        window_left=int(caps.window_left),
    )
    TUNING.validate_query(query, None)
    compile_query = TUNING.encode_query(replace(query, autotune_effort="default"))

    def compile_jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.relative_prefill._preparation:compile_relative_prefill",
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
        programs = compile_relative_prefill(
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
        state = PreparedRelativePrefill(
            query=query,
            config=config,
            device=dev,
            programs=programs,
            dummy_part=dummy_part,
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
    "PreparedRelativePrefill",
    "compile_relative_prefill",
    "make_plan",
    "scratch_specs",
]
