"""Prepared paged kernels ignore non-finite values beyond the live KV extent."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import cutlass

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.utils.device import current_cuda_stream
from b12x.attention import paged
from b12x.attention.paged._forward import _to_kernel_tensor
from b12x.attention.paged._scratch import variant_key
from b12x.attention.paged.forward_extend_generic import (
    PagedForwardKernel as ExtendForwardKernel,
)
from b12x.attention.paged.forward_paged import PagedFp8ExtendRawForwardKernel
from b12x.attention.paged.merge import (
    LagunaVerifierMergeKernel,
    PagedPersistentMergeKernel,
)
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged.traits import select_paged_forward_traits
from b12x.preparation import (
    PreparationSession,
    PreparedCall,
    detect_device,
    require_prepared,
)
from b12x.testing.reference.helpers import require_b12x


def _fill_nonfinite(tensor, value):
    if tensor.dtype == torch.float8_e4m3fn:
        tensor.view(torch.uint8).fill_(0x7F)
    else:
        tensor.fill_(value)


def _check_replays(
    *,
    graph,
    q,
    key,
    value,
    pages,
    lengths,
    cu,
    output,
    query_len,
    k_descale=None,
    v_descale=None,
    partial_output=None,
    partial_lse=None,
):
    batch, capacity = pages.shape[0], pages.shape[1] * key.shape[1]
    clean_key, clean_value = key.clone(), value.clone()
    failures = []

    def replay():
        allocated = torch.cuda.memory_allocated(q.device)
        reserved = torch.cuda.memory_reserved(q.device)
        graph.replay()
        torch.cuda.synchronize(q.device)
        assert torch.cuda.memory_allocated(q.device) == allocated
        assert torch.cuda.memory_reserved(q.device) == reserved
        return output

    for length in (15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257):
        lengths.fill_(length)
        key.copy_(clean_key)
        value.copy_(clean_value)
        clean = replay().clone()
        if not torch.isfinite(clean).all():
            print(
                f"nonfinite clean length={length} "
                f"per_query_nonfinite={(~torch.isfinite(clean)).sum(dim=(1, 2)).tolist()} "
                f"per_head_nonfinite={(~torch.isfinite(clean)).sum(dim=(0, 2)).tolist()}",
                flush=True,
            )
            if partial_output is not None:
                num_chunks = partial_output.shape[0] // q.shape[0]
                partial = partial_output.view(
                    batch, query_len, num_chunks, *partial_output.shape[-2:]
                )
                print(
                    f"partial_nonfinite_by_request_chunk={(~torch.isfinite(partial)).sum(dim=(1, 3, 4)).tolist()} "
                    f"first_request_head_partial_lse={partial_lse.view(batch, query_len, num_chunks, -1)[0, :, :, 0].tolist()}",
                    flush=True,
                )
        assert torch.isfinite(clean).all(), (
            f"non-finite clean result at length={length}"
        )
        assert torch.equal(replay(), clean), (
            f"non-repeatable clean result at length={length}"
        )
        reference, _ = paged_attention_reference(
            q,
            key,
            value,
            pages,
            lengths,
            cu,
            causal=True,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        error = (clean - reference).abs().float()
        maximum_error = error.max().item()
        if maximum_error > 0.02:
            print(
                f"clean length={length} max_error={maximum_error} "
                f"per_query_error={error.amax(dim=(1, 2)).tolist()} "
                f"per_head_error={error.amax(dim=(0, 2)).tolist()} "
                f"per_8_columns_error={error.reshape(*error.shape[:-1], -1, 8).amax(dim=(0, 1, 3)).tolist()} "
                f"first_head_output={clean[0, 0].tolist()} "
                f"first_head_reference={reference[0, 0].tolist()}",
                flush=True,
            )
        assert maximum_error <= 0.02
        poisons = (
            (float("nan"),)
            if key.dtype == torch.float8_e4m3fn
            else (
                float("nan"),
                float("inf"),
                -float("inf"),
            )
        )
        for poison in poisons:
            for target in ("key", "value", "both"):
                key.copy_(clean_key)
                value.copy_(clean_value)
                for name, cache in (("key", key), ("value", value)):
                    if target in (name, "both"):
                        rows = cache.view(batch, capacity, *cache.shape[-2:])
                        _fill_nonfinite(rows[:, length:], poison)
                observed = replay()
                finite = bool(torch.isfinite(observed).all())
                unchanged = bool(torch.equal(observed, clean))
                case = f"length={length} target={target} poison={poison}"
                print(f"{case} finite={finite} unchanged={unchanged}", flush=True)
                if not finite or not unchanged:
                    failures.append(f"{case}: finite={finite}, unchanged={unchanged}")

    for target, cache in (("key", key), ("value", value)):
        key.copy_(clean_key)
        value.copy_(clean_value)
        rows = cache.view(batch, capacity, *cache.shape[-2:])
        _fill_nonfinite(rows[0, 256], float("nan"))
        live_nan_visible = not bool(torch.isfinite(replay()[query_len - 1]).all())
        print(f"live_{target}_nan_visible={live_nan_visible}", flush=True)
        if not live_nan_visible:
            failures.append(f"a committed NaN {target} was hidden")
    assert not failures, "\n".join(failures)


def _run_public_path(*, mode, batch, query_len, page_size, head_dim, kv_dtype):
    device = require_b12x()
    capacity, q_heads, kv_heads = 512, 24, 4
    q = (
        torch.randn((batch * query_len, q_heads, head_dim), device=device) * 0.2
    ).bfloat16()
    shape = (batch * capacity // page_size, page_size, kv_heads, head_dim)
    key = (torch.randn(shape, device=device) * 0.2).to(kv_dtype)
    value = (torch.randn(shape, device=device) * 0.2).to(kv_dtype)
    pages = torch.arange(shape[0], dtype=torch.int32, device=device).view(batch, -1)
    lengths = torch.full((batch,), 257, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_len
    output = torch.empty_like(q)
    k_descale = (
        torch.ones((batch, kv_heads), dtype=torch.float32, device=device)
        if kv_dtype == torch.float8_e4m3fn
        else None
    )
    v_descale = torch.ones_like(k_descale) if k_descale is not None else None
    caps = paged.Caps(
        device=device,
        mode=mode,
        dtype=q.dtype,
        kv_dtype=kv_dtype,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        max_total_q=q.shape[0],
        max_batch=batch,
        max_page_table_width=pages.shape[1],
        max_work_items=1024,
        max_partial_rows=4096,
        num_cache_pages=shape[0],
        use_cuda_graph=True,
    )
    invocation = dict(
        paged.invocation_from_tensors(
            caps,
            q=q,
            k_cache=key,
            v_cache=value,
            output=output,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            k_descale=k_descale,
            v_descale=v_descale,
        )
    )
    if mode == "verify":
        invocation["force_split_kv"] = True
    declaration = paged.plan(caps, invocation=invocation)
    if mode == "verify":
        configured = declaration.contract.configure(
            declaration.query,
            device=detect_device(device).identity,
            search=False,
        )
        declaration = paged.plan(
            caps,
            invocation=invocation,
            override=replace(configured.default, cta_tile_q=64),
        )

    def prepare_call(state):
        spec = state.scratch_plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        prime_output = torch.empty_like(output)
        binding = state.bind(
            scratch=scratch,
            q=q,
            k_cache=key,
            v_cache=value,
            output=prime_output,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            active_total_q=q.shape[0],
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return PreparedCall(
            run=lambda: state.run(binding),
            output=prime_output,
            owners=(scratch, binding),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        result = session.prepare(
            (declaration.request(name="paged-tail", prepare_call=prepare_call),)
        )
        graph = torch.cuda.CUDAGraph()
        try:
            plan = result.plans["paged-tail"]
            state = require_prepared(plan, "attention.gqa")
            spec = state.scratch_plan.scratch_specs()[0]
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

            def bind():
                return paged.bind(
                    plan,
                    scratch=scratch,
                    q=q,
                    k_cache=key,
                    v_cache=value,
                    output=output,
                    page_table=pages,
                    cache_seqlens=lengths,
                    cu_seqlens_q=cu,
                    active_total_q=q.shape[0],
                    k_descale=k_descale,
                    v_descale=v_descale,
                )

            binding = bind()
            launchers = state.programs[variant_key(binding)]
            kernel, traits = launchers.forward_kernel, launchers.traits
            if mode == "verify":
                assert type(kernel) is PagedFp8ExtendRawForwardKernel
                assert binding.scratch.plan.split_kv
                assert (traits.cta_tile_q, traits.cta_tile_kv) == (64, 64)
            else:
                assert isinstance(kernel, ExtendForwardKernel)
                assert not binding.scratch.plan.split_kv
                assert (traits.cta_tile_q, traits.cta_tile_kv) == (16, 64)
                assert kernel.use_paged_v_tma == (page_size == 64)
                assert kernel.use_paged_k_tma == (page_size == 64)
            print(
                f"kernel={type(kernel).__module__}.{type(kernel).__name__} "
                f"mode={mode} page={page_size} dtype={kv_dtype} "
                f"tile_q={traits.cta_tile_q} tile_kv={traits.cta_tile_kv} "
                f"warps_q={traits.num_warps_q} warps_kv={traits.num_warps_kv} "
                f"split_kv={binding.scratch.plan.split_kv}",
                flush=True,
            )
            paged.run(binding=binding, plan=plan)
            with session.capture(), torch.cuda.graph(graph):
                paged.run(binding=bind(), plan=plan)
            _check_replays(
                graph=graph,
                q=q,
                key=key,
                value=value,
                pages=pages,
                lengths=lengths,
                cu=cu,
                output=output,
                query_len=query_len,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        finally:
            graph.reset()
            result.close()


@torch.inference_mode()
def test_raw_fp8_verifier_unused_kv_graph():
    device = require_b12x()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("the public raw verifier route requires SM120 KV-stage selection")
    _run_public_path(
        mode="verify",
        batch=4,
        query_len=8,
        page_size=128,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("page_size", (64, 128))
@pytest.mark.parametrize("kv_dtype", (torch.bfloat16, torch.float8_e4m3fn))
@torch.inference_mode()
def test_generic_extend_unused_kv_graph(page_size, kv_dtype):
    _run_public_path(
        mode="extend",
        batch=1,
        query_len=4,
        page_size=page_size,
        head_dim=256,
        kv_dtype=kv_dtype,
    )


@pytest.mark.parametrize(
    "family,split_kv",
    (
        pytest.param("native-extend", False, id="native-extend-direct"),
        pytest.param("raw-q64", True, id="raw-verifier-d128-b4-split-tma"),
        pytest.param("raw-q32", False, id="raw-q32-direct"),
        pytest.param("raw-q32", True, id="raw-q32-split"),
        pytest.param("raw-q48", False, id="raw-q48-direct"),
        pytest.param("raw-q48", True, id="raw-q48-split"),
    ),
)
@torch.inference_mode()
def test_explicit_paged_configuration_unused_kv_graph(family, split_kv):
    """Compile native FP8 and raw tile configurations without routing controls."""
    device = require_b12x()
    native = family.startswith("native")
    head_dim = 128 if native or family == "raw-q64" else 256
    cta_tile_q = 64 if native else int(family.removeprefix("raw-q"))
    query_len = 8 if cta_tile_q == 64 else 4 if cta_tile_q == 32 else 6
    capacity, q_heads, kv_heads = 512, 24, 4
    batch = 4 if family == "raw-q64" else 1
    page_size = 128 if family == "raw-q64" else 64
    total_q = batch * query_len
    q = (torch.randn((total_q, q_heads, head_dim), device=device) * 0.2).bfloat16()
    shape = (batch * capacity // page_size, page_size, kv_heads, head_dim)
    key = (torch.randn(shape, device=device) * 0.2).to(torch.float8_e4m3fn)
    value = (torch.randn(shape, device=device) * 0.2).to(torch.float8_e4m3fn)
    pages = torch.arange(shape[0], dtype=torch.int32, device=device).view(batch, -1)
    lengths = torch.full((batch,), 257, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_len
    num_chunks = 4 if split_kv else 1
    chunk_size = capacity // num_chunks
    requests = torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(
        num_chunks
    )
    qo_tiles = torch.zeros_like(requests)
    kv_tiles = torch.arange(num_chunks, dtype=torch.int32, device=device).repeat(batch)
    o_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device)
        * query_len
        * num_chunks
    )
    chunk = torch.tensor([chunk_size], dtype=torch.int32, device=device)
    window_starts = torch.zeros(batch, dtype=torch.int32, device=device)
    valid = torch.ones(batch * num_chunks, dtype=torch.int32, device=device)
    raw_output = torch.empty(
        (total_q * num_chunks, q_heads, head_dim), dtype=q.dtype, device=device
    )
    raw_lse = torch.empty(
        (total_q * num_chunks, q_heads) if split_kv else (q_heads, total_q),
        dtype=torch.float32,
        device=device,
    )

    def tensor(value, dtype=cutlass.Int32):
        return _to_kernel_tensor(
            value, dtype, assumed_align=4 if dtype == cutlass.Int32 else 16
        )

    common = (
        tensor(q, cutlass.BFloat16),
        tensor(key.view(torch.uint8), cutlass.Uint8),
        tensor(value.view(torch.uint8), cutlass.Uint8),
        tensor(pages),
        tensor(lengths),
        tensor(cu),
        tensor(requests),
        tensor(qo_tiles),
        tensor(kv_tiles),
        tensor(o_indptr),
        tensor(chunk),
    )
    outputs = (
        tensor(raw_output, cutlass.BFloat16),
        tensor(raw_lse, cutlass.Float32),
        None,
        None,
    )
    dummy_descs = torch.zeros(kv_heads, dtype=torch.int64, device=device)
    if native:
        traits = select_paged_forward_traits(
            cta_tile_q=64,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            q_dtype=q.dtype,
            kv_dtype=key.dtype,
            device=device,
            exact_num_mma_kv=2,
        )
        options = dict(
            traits=traits,
            use_native_fp8_qk=True,
            use_native_fp8_pv=True,
            page_size=page_size,
        )
        kernel = ExtendForwardKernel(
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Uint8,
            cutlass.BFloat16,
            **options,
        )
        args = (
            *common,
            tensor(window_starts),
            tensor(valid),
            None,
            None,
            None,
            None,
            None,
            None,
            *outputs,
            tensor(dummy_descs, cutlass.Int64),
            tensor(dummy_descs, cutlass.Int64),
        )
        assert kernel.use_native_fp8_qk_mma and kernel.use_native_fp8_pv_mma
        assert (traits.num_warps_kv, traits.num_mma_kv, traits.cta_tile_kv) == (
            1,
            2,
            32,
        )
    else:
        kernel = PagedFp8ExtendRawForwardKernel(
            split_kv=split_kv,
            cta_tile_q=cta_tile_q,
            page_size=page_size,
            head_dim=head_dim,
            analytic_verify_batch=batch,
            analytic_verify_max_chunks=num_chunks,
            use_tma_kv_load=family == "raw-q64",
        )
        assert (
            kernel.stage_tile_rows,
            kernel.compute_tile_rows,
            kernel.num_stages,
        ) == ((32, 16, 2) if cta_tile_q == 48 else (64, 32, 1))
        if family == "raw-q64":
            assert kernel.use_q64_laguna_verifier and kernel.use_overlapped_tma_kv_load
        args = (*common, tensor(valid), *outputs)
    spec = KernelCompileSpec.from_facts(
        "diagnostic.attention.paged.unused_kv",
        1,
        ("family", family),
        ("split_kv", split_kv),
        ("batch", batch),
        ("page_size", page_size),
        ("head_dim", head_dim),
        ("cta_tile_q", cta_tile_q),
        ("num_chunks", num_chunks),
    )
    compiled = b12x_compile(kernel, *args, current_cuda_stream(), compile_spec=spec)
    output = raw_output
    merge = None
    merge_args = ()
    if split_kv:
        output = torch.empty_like(q)
        lse = torch.empty((q_heads, total_q), dtype=torch.float32, device=device)
        merge_indptr = (
            torch.arange(total_q + 1, dtype=torch.int32, device=device) * num_chunks
        )
        total_q_ptr = torch.tensor([total_q], dtype=torch.int32, device=device)
        if family == "raw-q64":
            merge_kernel = LagunaVerifierMergeKernel(persistent_ctas=total_q * q_heads)
            merge_args = (
                tensor(raw_output, cutlass.BFloat16),
                tensor(raw_lse, cutlass.Float32),
                tensor(lengths),
                tensor(output, cutlass.BFloat16),
                tensor(lse, cutlass.Float32),
            )
        else:
            merge_kernel = PagedPersistentMergeKernel(
                cutlass.BFloat16,
                cutlass.BFloat16,
                head_dim=head_dim,
                persistent_ctas=2,
            )
            merge_args = (
                tensor(raw_output, cutlass.BFloat16),
                tensor(raw_lse, cutlass.Float32),
                tensor(merge_indptr),
                tensor(lengths),
                tensor(chunk),
                tensor(output, cutlass.BFloat16),
                tensor(lse, cutlass.Float32),
                tensor(total_q_ptr),
            )
        merge = b12x_compile(
            merge_kernel,
            *merge_args,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "diagnostic.attention.paged.unused_kv.merge",
                1,
                ("head_dim", head_dim),
                ("family", family),
                ("persistent_ctas", merge_kernel.persistent_ctas),
            ),
        )

    def run():
        run_compiled(compiled, (*args, current_cuda_stream()))
        if merge is not None:
            run_compiled(merge, (*merge_args, current_cuda_stream()))

    print(
        f"kernel={type(kernel).__module__}.{type(kernel).__name__} "
        f"configuration={family} split_kv={split_kv} stage_rows={kernel.stage_tile_rows} "
        f"native_fp8_pv={getattr(kernel, 'use_native_fp8_pv_mma', False)}",
        flush=True,
    )
    run()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        _check_replays(
            graph=graph,
            q=q,
            key=key,
            value=value,
            pages=pages,
            lengths=lengths,
            cu=cu,
            output=output,
            query_len=query_len,
            partial_output=raw_output if split_kv else None,
            partial_lse=raw_lse if split_kv else None,
        )
    finally:
        graph.reset()
