"""The TrellisMX small-M arm's buffers are the caller's, one set per binding.

The arm used to keep its activation, route and output buffers on the prepared
experts, so every call and every captured replay of one prepared layer wrote the
same storage and had to be serialized. They are caller-owned scratch now: the
plan declares the bytes (``smallm_scratch_layout``), the fused-MoE arena
reserves them, and ``bind`` carves one set per binding out of the buffer it was
given. These tests pin that contract: two bindings of one prepared layer hold
distinct buffers, each replays its own capture to its own eager result and to
the reference, a capture allocates nothing, and a missing or short arena raises.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import trellismx

from ._trellismx_data import glm_sidecar

_GLM_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)

#: The arm's carved buffers, as the binding carries them.
_ARM_BUFFERS = (
    "boundary_payload",
    "boundary_scales",
    "x_fp16",
    "out_f32",
    "intermediate_cache13",
    "intermediate_cache2",
    "fc1_c_tmp",
    "fc2_c_tmp",
    "packed_route_indices",
    "block_expert_ids",
    "packed_route_count",
    "expert_offsets",
    "expert_counts",
)


def _load(path: pathlib.Path, names):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in names}
        metadata = dict(handle.metadata() or {})
    return tensors, metadata


def _layer(device):
    """GLM-5.3 Flash layer 3 rank 0, prepared on ``device``."""

    path = pathlib.Path(glm_sidecar(3, 0))
    if not path.exists():
        pytest.skip(f"released sidecar absent: {path}")
    tensors, metadata = _load(path, _GLM_TENSORS)
    tensors = {name: value.to(device) for name, value in tensors.items()}
    prepared = trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=device,
    )
    return tensors, prepared


def _routing(tokens, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tokens, 4096, generator=generator).bfloat16().to(device)
    ids = torch.stack(
        [torch.randperm(288, generator=generator)[:8] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    weights = torch.rand(tokens, 8, generator=generator).to(device)
    weights = weights / weights.sum(-1, keepdim=True)
    return x, ids, weights


def _arena(plan, device):
    specs = plan.scratch_specs()
    assert len(specs) == 1
    return torch.zeros(int(specs[0].nbytes), dtype=torch.uint8, device=device)


def _reference(x, ids, weights, tensors, device):
    from b12x.testing.reference import trellismx_moe as reference

    want = reference.trellismx_moe_reference(
        x.cpu(),
        ids.cpu(),
        weights.cpu(),
        {name: value.cpu() for name, value in tensors.items()},
        dict(
            hidden_size=4096,
            intermediate_size=512,
            num_experts=288,
            top_k=8,
            bits=4,
            rank=0,
            world_size=4,
            sign_draw=0,
            quantize_activations=False,
            expert_subset=sorted({int(v) for v in ids.reshape(-1).cpu()}),
        ),
    ).to(device)
    return want


def _cosine(got, want) -> float:
    return torch.nn.functional.cosine_similarity(
        got.double().reshape(1, -1), want.double().reshape(1, -1)
    ).item()


# --------------------------------------------------------------------------- #
# the layout and the carve: device-free, so these run everywhere
# --------------------------------------------------------------------------- #
def _layout():
    from b12x.moe._shared.kernels.trellismx_smallm import smallm_scratch_layout

    return smallm_scratch_layout(
        hidden_size=512,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        max_tokens=16,
        sms=148,
    )


def test_the_layout_is_aligned_disjoint_and_complete() -> None:
    """Every region starts on a 16-byte boundary and the total covers them."""

    layout = _layout()
    end = 0
    for spec in layout.specs:
        assert spec.offset_bytes % 16 == 0, spec.name
        assert spec.offset_bytes >= end, (spec.name, "regions overlap")
        assert spec.numel * spec.dtype.itemsize == spec.nbytes
        end = spec.offset_bytes + spec.nbytes
    assert layout.nbytes % 16 == 0
    assert layout.nbytes >= end
    assert {spec.name for spec in layout.specs} == set(_ARM_BUFFERS)


def test_carve_views_the_caller_arena_and_rejects_a_bad_one() -> None:
    """The views alias the arena; a missing, short or misaligned arena raises."""

    from b12x.moe._shared.kernels.trellismx_smallm import carve_smallm_scratch

    layout = _layout()
    arena = torch.zeros(layout.nbytes + 64, dtype=torch.uint8)
    scratch = carve_smallm_scratch(arena, layout, offset_bytes=16)
    for spec in layout.specs:
        view = getattr(scratch, spec.name)
        assert tuple(view.shape) == spec.shape, spec.name
        assert view.dtype == spec.dtype, spec.name
        assert view.data_ptr() == arena.data_ptr() + 16 + spec.offset_bytes, spec.name
        assert scratch.device == arena.device

    with pytest.raises(RuntimeError, match="caller-owned scratch"):
        carve_smallm_scratch(None, layout)
    with pytest.raises(ValueError, match="requires"):
        carve_smallm_scratch(torch.zeros(layout.nbytes - 1, dtype=torch.uint8), layout)
    with pytest.raises(TypeError, match="uint8"):
        carve_smallm_scratch(torch.zeros(layout.nbytes, dtype=torch.int32), layout)
    # An unaligned base cannot back 16-byte aligned launch pointers.
    with pytest.raises(ValueError, match="16-byte"):
        carve_smallm_scratch(arena[1:], layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_two_bindings_of_one_prepared_layer_own_their_buffers() -> None:
    """Two bindings, two arenas: distinct buffers, independent captures.

    Both bindings come from one prepared plan, so they share the arm's compiled
    programs. They must not share a byte of mutable storage: each binding's
    carved buffers live in the arena that binding was given, and each captured
    graph replays to its own eager result (different inputs per binding, so a
    shared buffer would show up as the other call's activations).
    """

    from b12x._lib.runtime_control import kernel_resolution_guard

    device = torch.device("cuda", torch.cuda.current_device())
    tensors, prepared = _layer(device)
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=16, top_k=8, route_num_experts=0
        ),
    )

    x1, ids1, weights1 = _routing(16, 20261031, device)
    x2, ids2, weights2 = _routing(16, 20261101, device)
    out1 = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    out2 = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    arena1, arena2 = _arena(plan, device), _arena(plan, device)
    binding1 = fused_moe.bind(
        plan, scratch=arena1, a=x1, topk_ids=ids1, topk_weights=weights1, output=out1
    )
    binding2 = fused_moe.bind(
        plan, scratch=arena2, a=x2, topk_ids=ids2, topk_weights=weights2, output=out2
    )

    # The arm's buffers are part of what the plan declares, and each binding
    # carved its own set out of the arena it was given.
    for binding, arena in ((binding1, arena1), (binding2, arena2)):
        scratch = binding.trellismx_smallm_scratch
        assert scratch is not None, "bind must carve the arm's buffers"
        base, end = arena.data_ptr(), arena.data_ptr() + arena.numel()
        for name in _ARM_BUFFERS:
            view = getattr(scratch, name)
            assert base <= view.data_ptr() < end, (name, "outside the bound arena")
    first = binding1.trellismx_smallm_scratch
    second = binding2.trellismx_smallm_scratch
    for name in _ARM_BUFFERS:
        assert getattr(first, name).data_ptr() != getattr(second, name).data_ptr(), (
            name,
            "two bindings of one plan must not share a buffer",
        )

    eager1 = fused_moe.run(binding=binding1).clone()
    eager2 = fused_moe.run(binding=binding2).clone()
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    graph1, graph2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with kernel_resolution_guard("test: two small-M bindings, no eager call"):
        with torch.cuda.graph(graph1):
            fused_moe.run(binding=binding1)
        with torch.cuda.graph(graph2):
            fused_moe.run(binding=binding2)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, (
        "capture allocated device memory; the arm's buffers are the caller's"
    )

    # Interleaved replays: each graph writes only its own binding's storage.
    out1.zero_()
    out2.zero_()
    graph1.replay()
    graph2.replay()
    graph1.replay()
    torch.cuda.synchronize()
    assert torch.equal(out1, eager1), "replay 1 changed when binding 2 replayed"
    assert torch.equal(out2, eager2), "replay 2 changed when binding 1 replayed"
    assert not torch.equal(eager1, eager2), "the two bindings must see different inputs"

    want1 = _reference(x1, ids1, weights1, tensors, device)
    want2 = _reference(x2, ids2, weights2, tensors, device)
    assert _cosine(eager1, want1) >= 0.99995, _cosine(eager1, want1)
    assert _cosine(eager2, want2) >= 0.99995, _cosine(eager2, want2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_the_arm_reserves_its_own_bytes_and_rejects_a_short_arena() -> None:
    """The plan's arena holds the arm's buffers; a short one is rejected at bind."""

    device = torch.device("cuda", torch.cuda.current_device())
    _, prepared = _layer(device)
    capacity = dict(max_tokens=16, top_k=8, route_num_experts=0)
    with_arm = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(**capacity),
        smallm_max_tokens=16,
    )
    grouped_only = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(**capacity),
        smallm_max_tokens=0,
    )
    arm_bytes = int(with_arm.scratch_specs()[0].nbytes)
    grouped_bytes = int(grouped_only.scratch_specs()[0].nbytes)
    assert arm_bytes > grouped_bytes, (
        "a plan that serves the arm must reserve its buffers in the arena",
        arm_bytes,
        grouped_bytes,
    )

    x, ids, weights = _routing(16, 20261102, device)
    output = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    short = torch.zeros(grouped_bytes, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="requires"):
        fused_moe.bind(
            with_arm,
            scratch=short,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
        )

    # The grouped-only plan never reaches the arm, so its smaller arena binds.
    fused_moe.bind(
        grouped_only,
        scratch=short,
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_a_call_without_the_arm_scratch_raises() -> None:
    """No plan-owned fallback: the arm and its dispatch refuse a missing arena."""

    from dataclasses import replace

    from b12x.moe._shared.kernels.trellismx_smallm import TrellisMxSmallM

    device = torch.device("cuda", torch.cuda.current_device())
    _, prepared = _layer(device)
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=16, top_k=8, route_num_experts=0
        ),
    )
    x, ids, weights = _routing(16, 20261103, device)
    output = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    binding = fused_moe.bind(
        plan,
        scratch=_arena(plan, device),
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
    )

    # The arm's own entry point.
    arm = binding.trellismx_smallm_plan
    with pytest.raises(RuntimeError, match="caller-owned scratch"):
        TrellisMxSmallM.run(x, ids, weights, output, arm)
    # A binding that carries no carved buffers (a plan that did not declare the
    # arm, or a caller that dropped them) cannot run it either.
    with pytest.raises(RuntimeError, match="caller-owned scratch"):
        fused_moe.run(binding=replace(binding, trellismx_smallm_scratch=None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_the_direct_entry_point_rejects_a_foreign_or_misaligned_call() -> None:
    """``TrellisMxSmallM.run`` checks the device and the alignment itself.

    The fused wrapper validates the binding before dispatch, but the arm is
    also exported on its own. A scratch carved for the same layout on another
    device compares equal to the plan's layout, and a contiguous activation or
    output can still miss the 16-byte alignment the launches assume; both must
    be rejected here instead of launching over storage the call does not own.
    """

    from b12x.moe._shared.kernels.trellismx_smallm import (
        TrellisMxSmallM,
        carve_smallm_scratch,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    _, prepared = _layer(device)
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=16, top_k=8, route_num_experts=0
        ),
    )
    x, ids, weights = _routing(16, 20261105, device)
    output = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    binding = fused_moe.bind(
        plan,
        scratch=_arena(plan, device),
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
    )
    arm = binding.trellismx_smallm_plan
    scratch = binding.trellismx_smallm_scratch
    assert arm is not None and scratch is not None

    # A same-layout arena on another device: the layout compares equal, so only
    # the device check can tell the two apart.
    foreign = carve_smallm_scratch(
        torch.zeros(int(scratch.layout.nbytes), dtype=torch.uint8), scratch.layout
    )
    assert foreign.layout == scratch.layout
    assert foreign.device != scratch.device
    with pytest.raises(ValueError, match="scratch lives on"):
        TrellisMxSmallM.run(x, ids, weights, output, workspace=arm, scratch=foreign)

    # Contiguous views whose base is not 16-byte aligned (one bf16 element off).
    flat = torch.randn(16 * 4096 + 1, dtype=torch.bfloat16, device=device)
    misaligned = flat[1:].view(16, 4096)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        TrellisMxSmallM.run(
            misaligned, ids, weights, output, workspace=arm, scratch=scratch
        )
    with pytest.raises(ValueError, match="16-byte aligned"):
        TrellisMxSmallM.run(x, ids, weights, misaligned, workspace=arm, scratch=scratch)
    # The aligned call the same plan accepts still runs.
    TrellisMxSmallM.run(x, ids, weights, output, workspace=arm, scratch=scratch)
    torch.cuda.synchronize()
