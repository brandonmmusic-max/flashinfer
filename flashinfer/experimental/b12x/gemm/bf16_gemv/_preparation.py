"""Metadata-only preparation and retained native unquantized projections."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from b12x.preparation.types import plan_from_handle, require_prepared

from ._tuning import TUNING, GemvConfig, GemvQuery


def _is_aligned(tensor):
    return tensor.data_ptr() % 16 == 0


def query_from_call(x, weight, *, bias=None, out=None, output_dtype=None):
    """Capture geometry, numerical types and layout eligibility without GPU work."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("projection requires source[M,K] and weight[N,K]")
    if weight.device != x.device or (bias is not None and bias.device != x.device):
        raise ValueError("projection operands must share a device")
    dtype = x.dtype if output_dtype is None else output_dtype
    if out is not None:
        if output_dtype is not None and out.dtype != output_dtype:
            raise ValueError("output_dtype disagrees with caller-owned output")
        dtype = out.dtype
    query = GemvQuery(
        source_dtype=str(x.dtype).removeprefix("torch."),
        weight_dtype=str(weight.dtype).removeprefix("torch."),
        max_rows=int(x.shape[0]),
        in_features=int(x.shape[1]),
        out_features=int(weight.shape[0]),
        source_contiguous=x.is_contiguous(),
        source_aligned=_is_aligned(x),
        weight_contiguous=weight.is_contiguous(),
        weight_aligned=_is_aligned(weight),
        output_dtype=str(dtype).removeprefix("torch."),
        output_contiguous=out is None or out.is_contiguous(),
        output_aligned=out is None or _is_aligned(out),
        bias_dtype=None if bias is None else str(bias.dtype).removeprefix("torch."),
    )
    TUNING.validate_query(query, None)
    return query


@program_cache(scope="preparation")
def compile_gemv(query_payload, config_payload, ordinal):
    """Compile only the selected kernel and retain its runtime launcher."""
    query = GemvQuery(**dict(query_payload))
    config = GemvConfig(**dict(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    if config.backend == "torch":
        return {}
    if config.backend == "prefill":
        from ._prefill import compile_prefill

        launcher = compile_prefill(
            ordinal,
            query.max_rows,
            query.out_features,
            query.in_features,
            query.output_dtype,
        )
    elif config.backend == "tc":
        from ._tensor_core import compile_tc_projection

        launcher = compile_tc_projection(
            ordinal,
            query.max_rows,
            config.split_k,
            query.out_features,
            query.in_features,
            query.output_dtype,
            query.bias_dtype,
        )
    else:
        from ._kernel import compile_projection

        launcher = compile_projection(
            ordinal,
            config.backend,
            config.rows_per_tile,
            query.out_features,
            query.in_features,
            query.source_dtype,
            query.weight_dtype,
            query.output_dtype,
            query.bias_dtype,
            config.block_threads,
        )
    return {"gemv": launcher}


def _torch_projection(x, weight, out, bias):
    if out.dtype == torch.float32:
        torch.mm(x, weight.T, out_dtype=torch.float32, out=out)
    elif bias is None:
        torch.mm(x, weight.T, out=out)
    else:
        torch.addmm(bias, x, weight.T, out=out)


@dataclass(frozen=True)
class _GemvExecutionState:
    query: GemvQuery
    device: torch.device
    launcher: object
    tc_scratch: torch.Tensor | None = None
    tc_flags: torch.Tensor | None = None

    def run(self, x, weight, *, out=None, bias=None):
        from ._kernel import _validate

        query = self.query
        if x.device != self.device or weight.device != self.device:
            raise ValueError("projection operands differ from the prepared device")
        if (
            x.ndim != 2
            or x.shape[1] != query.in_features
            or x.shape[0] > query.max_rows
        ):
            raise ValueError("projection source exceeds its declared geometry")
        if tuple(weight.shape) != (query.out_features, query.in_features):
            raise ValueError("projection weight geometry differs from preparation")
        if (
            str(x.dtype).removeprefix("torch.") != query.source_dtype
            or str(weight.dtype).removeprefix("torch.") != query.weight_dtype
        ):
            raise ValueError("projection operand dtypes differ from preparation")
        if (
            None if bias is None else str(bias.dtype).removeprefix("torch.")
        ) != query.bias_dtype:
            raise ValueError("projection bias differs from preparation")
        if out is None:
            out = torch.empty(
                (x.shape[0], query.out_features),
                dtype=getattr(torch, query.output_dtype),
                device=self.device,
            )
        if str(out.dtype).removeprefix("torch.") != query.output_dtype:
            raise ValueError("projection output dtype differs from preparation")
        for name, tensor in (("source", x), ("weight", weight), ("output", out)):
            if getattr(query, f"{name}_contiguous") and not tensor.is_contiguous():
                raise ValueError(f"projection {name} must retain contiguous layout")
            if getattr(query, f"{name}_aligned") and not _is_aligned(tensor):
                raise ValueError(f"projection {name} must retain 16-byte alignment")
        _validate(x, weight, out, bias)
        if x.shape[0]:
            if self.tc_scratch is not None:
                self.launcher(
                    x, weight, out, bias, scratch=self.tc_scratch, flags=self.tc_flags
                )
            else:
                self.launcher(x, weight, out, bias)
        return out


def plan(query: GemvQuery, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, GemvQuery):
        raise TypeError("projection plan requires GemvQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("projection invocation semantics belong in GemvQuery")
    compile_query = TUNING.encode_query(replace(query, autotune_effort="default"))

    def compile_jobs(config, device):
        if config.backend == "torch":
            return ()
        return (
            CompileJob.create(
                "b12x.gemm.bf16_gemv._preparation:compile_gemv",
                compile_query,
                TUNING.encode_config(config),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def memory(config, device):
        TUNING.validate_config(query, config, device)
        if config.backend == "tc" and config.split_k > 1:
            from ._tensor_core import (
                _TC_SPLITK_BUFFERS,
                tc_splitk_flags_bytes,
                tc_splitk_scratch_bytes,
            )

            m_tiles = (query.max_rows + 7) // 8
            total = tc_splitk_scratch_bytes(
                query.max_rows, query.out_features, config.split_k
            ) + tc_splitk_flags_bytes(query.out_features, config.split_k)
            key = (
                device.ordinal,
                query.out_features,
                query.in_features,
                m_tiles,
                config.split_k,
            )
            resident = total if key in _TC_SPLITK_BUFFERS else 0
            return MemoryRequirements(
                (),
                (PersistentMemory(("bf16_gemv.tc_splitk", *key), total, resident),),
            )
        return MemoryRequirements()

    def materialize(selection, device):
        if selection.config.backend == "torch":
            return _GemvExecutionState(
                query, torch.device("cuda", device.ordinal), _torch_projection
            )
        programs = compile_gemv(
            compile_query,
            TUNING.encode_config(selection.config),
            device.ordinal,
        )
        load_programs(programs)
        resolved = torch.device("cuda", device.ordinal)
        tc_scratch = tc_flags = None
        if selection.config.backend == "tc" and selection.config.split_k > 1:
            from ._tensor_core import tc_splitk_buffers

            tc_scratch, tc_flags = tc_splitk_buffers(
                resolved,
                query.max_rows,
                query.out_features,
                query.in_features,
                selection.config.split_k,
            )
        return attach_programs(
            _GemvExecutionState(
                query, resolved, programs["gemv"], tc_scratch, tc_flags
            ),
            programs,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=True,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(
    x: torch.Tensor,
    weight: torch.Tensor,
    plan_handle: int,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.bf16_gemv", x.device)
    if output_dtype is not None and output_dtype != getattr(
        torch, state.query.output_dtype
    ):
        raise ValueError("output_dtype differs from preparation")
    return state.run(x, weight, bias=bias)


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    plan_handle: int,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype if output_dtype is None else output_dtype,
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n_out", mutates_args=("out",))
def bf16_gemv_small_n_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
    bias: torch.Tensor | None = None,
) -> None:
    require_prepared(plan_from_handle(plan_handle), "gemm.bf16_gemv", x.device).run(
        x, weight, out=out, bias=bias
    )


@bf16_gemv_small_n_out.register_fake
def _bf16_gemv_small_n_out_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
    bias: torch.Tensor | None = None,
) -> None:
    return None
