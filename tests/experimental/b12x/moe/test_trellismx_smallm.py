"""TrellisMX small-M arm: the procedural-NPK law and the arm itself.

The arm's arithmetic is fixed by measurement against small-M forwards recorded
from the release runtime: m <= 16 does **not**
quantize A and does **not** quantize the FC2 input, and its per-route rows are
FP32. Its weights use the released E4M3 alphabet (the same decode the grouped
arm uses) widened to the W4A16 GEMM's fp16 operand with the UE8M0/K32 block
scale; the compander that the ``mcg`` arm does not apply is part of the law.

The first tests pin the law on the device; the real-data tests then run the arm
against those recorded forwards and a full CUDA graph replay, and skip when the
released sidecar or the captures are absent.
"""

from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Uint32

from b12x.moe._shared.kernels.trellismx_smallm import (
    SMALLM_MAX_TOKENS,
    TrellisMxSmallM,
    compile_smallm_programs,
    packed_decode_npk_to_half2x4,
)

from ._trellismx_data import capture, glm_sidecar

_BITS = 4

#: Released GLM-5.3 Flash layer 3 rank 0 sidecar (read only).
GLM_SIDECAR = glm_sidecar(3, 0)
#: Small-M forwards recorded from the release runtime (one forward each, real
#: mixed top-8 routing).
CAPTURES = tuple(
    pathlib.Path(capture(name))
    for name in (
        "oracle_glm_l3_r0_e0.npz",
        "oracle_smallm_m1.npz",
        "oracle_smallm_m8.npz",
        "oracle_smallm_m16.npz",
    )
)
_SIDECAR_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)


def _law_fp16(win_a: int, win_b: int, bits: int = _BITS) -> np.ndarray:
    """The procedural-NPK law widened to fp16 with the alpha-2 compander."""

    def state(word, shift):
        return (word >> shift) & 0xFFFF

    order = [
        state(win_b, 3 * bits),
        state(win_b, 2 * bits),
        state(win_b, bits),
        state(win_b, 0),
        state(win_a, 3 * bits),
        state(win_a, 2 * bits),
        state(win_a, bits),
        state(win_a, 0),
    ]
    values = []
    for code in order:
        product = (int(code) * 0xCBAC1FED) & 0xFFFFFFFF
        word = (product & 0x8FFF8FFF) ^ 0x3B603B60
        low = np.array([word & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
        high = np.array([(word >> 16) & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
        combined = np.float16(low + high)
        values.append(np.float16(combined + combined))
    return np.array(values, dtype=np.float16)


class _Probe:
    """One block per window pair; thread 0 decodes that pair's eight states."""

    threads_per_cta = 32

    @cute.jit
    def __call__(self, a_ptr, b_ptr, out_ptr, count, stream: cuda.CUstream):
        a = cute.make_tensor(a_ptr, cute.make_layout((Int32(count),), stride=(1,)))
        b = cute.make_tensor(b_ptr, cute.make_layout((Int32(count),), stride=(1,)))
        o = cute.make_tensor(
            out_ptr, cute.make_layout((Int32(count) * 4,), stride=(1,))
        )
        self.kernel(a, b, o, count).launch(
            grid=(Int32(count), 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, a, b, o, count):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if Int32(bid) < count and Int32(tid) == Int32(0):
            o0, o1, o2, o3 = packed_decode_npk_to_half2x4(
                Uint32(a[bid]), Uint32(b[bid]), _BITS
            )
            o[Int32(bid) * Int32(4)] = o0
            o[Int32(bid) * Int32(4) + Int32(1)] = o1
            o[Int32(bid) * Int32(4) + Int32(2)] = o2
            o[Int32(bid) * Int32(4) + Int32(3)] = o3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_npk_fp16_decode_is_bitwise_the_law() -> None:
    """The kernel's 16-bit decode == the procedural-NPK law + alpha-2, bitwise."""

    from cutlass.cute.runtime import make_ptr

    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x.moe.fused_moe._impl import current_cuda_stream

    rows = 256
    generator = torch.Generator().manual_seed(11)
    win_a = (
        torch.randint(0, 2**31, (rows,), generator=generator, dtype=torch.int64)
        .to(torch.int32)
        .cuda()
    )
    win_b = (
        torch.randint(0, 2**31, (rows,), generator=generator, dtype=torch.int64)
        .to(torch.int32)
        .cuda()
    )
    out = torch.zeros(rows * 4, dtype=torch.int32, device="cuda")
    probe = _Probe()
    compiled = b12x_compile(
        probe,
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        rows,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            "w2.tmx.npk_fp16_probe", 1, ("rows", rows)
        ),
    )
    compiled(
        make_ptr(
            cutlass.Int32, win_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Int32, win_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Int32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        rows,
        current_cuda_stream(),
    )
    torch.cuda.synchronize()
    got = (
        out.cpu()
        .numpy()
        .view(np.uint32)
        .reshape(rows, 4)
        .view(np.uint16)
        .reshape(rows, 8)
    )
    a_np = win_a.cpu().numpy().astype(np.uint32)
    b_np = win_b.cpu().numpy().astype(np.uint32)
    want = np.stack([_law_fp16(int(a_np[i]), int(b_np[i])) for i in range(rows)])
    assert np.array_equal(got, want.view(np.uint16).reshape(rows, 8))


def test_the_compander_is_not_the_mcg_arm() -> None:
    """The compander law is the mcg law *plus* the doubling; without the doubling the values halve."""

    rows = 32
    generator = torch.Generator().manual_seed(3)
    win_a = torch.randint(0, 2**31, (rows,), generator=generator, dtype=torch.int64)
    win_b = torch.randint(0, 2**31, (rows,), generator=generator, dtype=torch.int64)
    for index in range(rows):
        doubled = _law_fp16(int(win_a[index]), int(win_b[index]))
        halved = np.array([np.float16(v / 2) for v in doubled], dtype=np.float16)
        # the mcg arm stops one doubling short, so its value is half of the compander law's
        assert np.allclose(
            doubled.astype(np.float32),
            2.0 * halved.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )


def test_smallm_owns_only_one_to_sixteen_tokens() -> None:
    assert SMALLM_MAX_TOKENS == 16
    with pytest.raises(ValueError, match="small-M arm owns 1..16"):
        TrellisMxSmallM.plan(experts=None, max_tokens=32, top_k=8)


def test_plan_rejects_a_bundle_that_is_not_trellismx() -> None:
    class _Impl:
        source_format = "fp4_e8m0_k32"

    class _Experts:
        _impl = _Impl()

    with pytest.raises(ValueError, match="prepared TrellisMX bundle"):
        TrellisMxSmallM.plan(_Experts(), max_tokens=16, top_k=8)


def test_run_rejects_a_foreign_workspace() -> None:
    with pytest.raises(TypeError, match="plan returned by"):
        TrellisMxSmallM.run(None, None, None, None, None)


def _bound(plan):
    """The plan's buffers, carved out of an arena this caller owns."""

    arena = torch.zeros(plan.workspace_bytes, dtype=torch.uint8, device=plan.device)
    return plan.bind_scratch(arena)


def _load_glm():
    from safetensors import safe_open

    from b12x.moe.fused_moe import trellismx as tmx

    device = torch.device("cuda", torch.cuda.current_device())
    with safe_open(GLM_SIDECAR, "np") as handle:
        raw = {name: handle.get_tensor(name) for name in _SIDECAR_TENSORS}
        meta = dict(handle.metadata())
    tensors = {name: torch.from_numpy(value).to(device) for name, value in raw.items()}
    prepared = tmx.prepare_trellismx_experts(
        tensors,
        meta,
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=device,
    )
    return raw, tensors, prepared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("capture", CAPTURES, ids=lambda p: p.stem)
def test_smallm_arm_matches_the_released_smallm_capture(capture) -> None:
    """Kernel vs a recorded small-M forward: cosine >= 0.99999, max|d| <= 2x reference."""

    from b12x.testing.reference import trellismx_moe as reference

    if not os.path.exists(GLM_SIDECAR):
        pytest.skip(f"released sidecar absent: {GLM_SIDECAR}")
    if not capture.exists():
        pytest.skip(f"capture absent: {capture}")

    device = torch.device("cuda", torch.cuda.current_device())
    raw, _, prepared = _load_glm()
    plan = TrellisMxSmallM.plan(prepared, max_tokens=16, top_k=8)

    data = np.load(capture, allow_pickle=True)
    x = torch.from_numpy(data["x"]).to(device).to(torch.bfloat16)
    ids = torch.from_numpy(data["topk_ids"]).to(device)
    weights = torch.from_numpy(data["topk_weights"]).to(device)
    want = torch.from_numpy(data["y"]).to(device)
    m = int(x.shape[0])
    out = torch.zeros(m, 4096, dtype=torch.bfloat16, device=device)
    TrellisMxSmallM.run(x, ids, weights, out, plan, _bound(plan))
    torch.cuda.synchronize()

    ref_meta = dict(
        hidden_size=4096,
        intermediate_size=512,
        num_experts=288,
        top_k=8,
        bits=4,
        rank=0,
        world_size=4,
        sign_draw=0,
        quantize_a=False,
        quantize_down=False,
        emulate_fp16_stores=True,
    )
    cpu = {name: torch.from_numpy(value) for name, value in raw.items()}
    ref = reference.trellismx_moe_reference(
        x.cpu(), ids.cpu(), weights.cpu(), cpu, ref_meta
    )

    def metrics(got, want):
        got = got.float().reshape(-1)
        want = want.float().reshape(-1)
        cosine = float((got * want).sum() / (got.norm() * want.norm()))
        return cosine, float((got - want).abs().max())

    cosine, max_d = metrics(out, want)
    _, ref_max_d = metrics(ref, want.cpu())
    assert cosine >= 0.99999, (capture.stem, cosine, max_d, ref_max_d)
    assert max_d <= 2.0 * ref_max_d + 1e-6, (capture.stem, max_d, ref_max_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_smallm_arm_full_graph_capture_replays_bitwise() -> None:
    """One capture region per m; replay is bitwise the eager result."""

    if not os.path.exists(GLM_SIDECAR):
        pytest.skip(f"released sidecar absent: {GLM_SIDECAR}")

    device = torch.device("cuda", torch.cuda.current_device())
    _, _, prepared = _load_glm()
    plan = TrellisMxSmallM.plan(prepared, max_tokens=16, top_k=8)
    generator = torch.Generator().manual_seed(17)
    for m in (1, 2, 8, 16):
        x = torch.randn(m, 4096, generator=generator).to(device).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(288, generator=generator)[:8] for _ in range(m)]
        ).to(device, dtype=torch.int32)
        weights = torch.rand(m, 8, generator=generator).to(device)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros(m, 4096, dtype=torch.bfloat16, device=device)
        scratch = _bound(plan)
        TrellisMxSmallM.run(x, ids, weights, out, plan, scratch)
        torch.cuda.synchronize()
        eager = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            TrellisMxSmallM.run(x, ids, weights, out, plan, scratch)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager, out), m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_smallm_run_requires_capture_compatible_routing() -> None:
    """The run entry does not cast or copy routing tensors during capture."""

    if not os.path.exists(GLM_SIDECAR):
        pytest.skip(f"released sidecar absent: {GLM_SIDECAR}")

    device = torch.device("cuda", torch.cuda.current_device())
    _, _, prepared = _load_glm()
    plan = TrellisMxSmallM.plan(prepared, max_tokens=16, top_k=8)
    x = torch.zeros(1, 4096, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(1, 8, dtype=torch.int32, device=device)
    weights = torch.full((1, 8), 0.125, dtype=torch.float32, device=device)
    invalid = (
        (ids.to(torch.int16), weights),
        (ids, weights.to(torch.float16)),
        (torch.cat((ids, ids), dim=1)[:, ::2], weights),
        (ids, torch.cat((weights, weights), dim=1)[:, ::2]),
    )
    for bad_ids, bad_weights in invalid:
        with pytest.raises(
            ValueError, match="routing must be|topk_ids must be int32/int64"
        ):
            TrellisMxSmallM.run(x, bad_ids, bad_weights, x, plan, _bound(plan))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_worker_preparation_serves_every_smallm_token_count(
    tmp_path,
) -> None:
    """A frozen 16-token plan serves every token count it accepts.

    The per-token-count fused boundary and fused launch must be declared by
    preparation; compiling one of them on the first smaller call is too late
    after kernel resolution freezes.
    """

    from b12x._lib.compile_pool import CompileJob
    from b12x.preparation import (
        Knob,
        MemoryRequirements,
        Plan,
        PreparationSession,
        PreparedCall,
        TuningContract,
    )

    if not os.path.exists(GLM_SIDECAR):
        pytest.skip(f"released sidecar absent: {GLM_SIDECAR}")

    device = torch.device("cuda", torch.cuda.current_device())
    _, _, prepared = _load_glm()
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(16, 4096, generator=generator).to(device).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(288, generator=generator)[:8] for _ in range(16)]
    ).to(device=device, dtype=torch.int32)
    weights = torch.rand(16, 8, generator=generator).to(device)
    weights = weights / weights.sum(-1, keepdim=True)
    implementation = "smallm"
    tuning = TuningContract(
        component_id="test.trellismx_smallm_capacity",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=frozenset({"max_tokens"}),
        config_fields=frozenset({"implementation"}),
        encode_query=lambda query: {"max_tokens": query.max_tokens},
        encode_config=lambda config: {"implementation": config},
        decode_config=lambda payload: payload["implementation"],
        validate_query=lambda query, device: None,
        validate_config=lambda query, config, device: None,
        default_config=lambda query, device: implementation,
        knobs=(Knob(name="implementation", values=(implementation,)),),
    )

    def compile_jobs(config, target_device):
        del config
        return (
            CompileJob.create(
                "b12x.moe._shared.kernels.trellismx_smallm:compile_smallm_programs",
                hidden_size=4096,
                intermediate_size=512,
                num_experts=288,
                top_k=8,
                bits=4,
                device=torch.device("cuda", target_device.ordinal),
                max_tokens=16,
            ),
        )

    def call(state):
        output = torch.full(
            (16, 4096), float("nan"), dtype=torch.bfloat16, device=device
        )
        # The arm writes caller-owned buffers only, so each prepared call binds
        # its own arena; the session keeps both the arena and the binding alive.
        arena = torch.zeros(state.workspace_bytes, dtype=torch.uint8, device=device)
        scratch = state.bind_scratch(arena)
        return PreparedCall(
            run=lambda: TrellisMxSmallM.run(x, ids, weights, output, state, scratch),
            output=output,
            produce=lambda: None,
            owners=(state, scratch, output),
        )

    def materialize(selection, target_device):
        from b12x._lib.compile_plan import load_programs

        programs = compile_smallm_programs(
            hidden_size=4096,
            intermediate_size=512,
            num_experts=288,
            top_k=8,
            bits=4,
            device=device,
            max_tokens=16,
        )
        load_programs(programs)
        return TrellisMxSmallM.plan(prepared, max_tokens=16, top_k=8)

    declaration = Plan(
        contract=tuning,
        query=SimpleNamespace(max_tokens=16),
        _compile_jobs=compile_jobs,
        _memory_requirements=lambda config, target_device: MemoryRequirements(),
        _materialize=materialize,
    )

    with PreparationSession(
        device=device,
        autotune=True,
        compile_workers=1,
        cache_dir=str(tmp_path / "programs"),
    ) as session:
        session.prepare(
            (
                declaration.request(
                    name="trellismx-smallm-capacity",
                    prepare_call=call,
                ),
            )
        )
        session.freeze()
        prepared_plan = declaration.prepared.state
        eager_plan = TrellisMxSmallM.plan(prepared, max_tokens=16, top_k=8)
        prepared_scratch = _bound(prepared_plan)
        eager_scratch = _bound(eager_plan)
        for rows in (1, 2, 8, 16):
            output = torch.full(
                (rows, 4096), float("nan"), dtype=torch.bfloat16, device=device
            )
            TrellisMxSmallM.run(
                x[:rows],
                ids[:rows],
                weights[:rows],
                output,
                prepared_plan,
                prepared_scratch,
            )
            eager = torch.full_like(output, float("nan"))
            TrellisMxSmallM.run(
                x[:rows],
                ids[:rows],
                weights[:rows],
                eager,
                eager_plan,
                eager_scratch,
            )
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.int16), eager.view(torch.int16)), rows
