"""TrellisMX per-token boundary ops vs the exact CPU reference.

Two arms:

* **Boundary parity** -- the runtime ops in
  ``b12x/moe/_shared/kernels/trellismx_boundary.py`` against the independent
  implementation in ``b12x/testing/reference/trellismx_moe.py``.  Both are pure
  FP32 torch, so the assertion is bitwise.  The two modules intentionally do not
  share code: the reference is the oracle.
* **Declared-order closure** -- the end-to-end reference against a float64 dense
  computation that builds the transforms as explicit matrices and the atom32
  interleave as an explicit permutation.  That arm can only be a bound: the
  reference accumulates in FP32 and the output is BF16, so the bound is one
  BF16 ULP.

Rows use distinct inputs, mixed routing (top-8 over six experts, so experts
repeat inside a row) and M in {1, 8, 32, 256}.  The CUDA-graph arm captures and
replays the ops and requires bitwise equality, because capture must not change
the arithmetic.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.moe._shared.kernels import trellismx_boundary as boundary
from b12x.moe._shared.kernels.w4a8_trellis_decode import _W4A8_TRELLIS_A_K32_PERM
from b12x.testing.reference import trellismx_moe as reference

_M_SWEEP = (1, 8, 32, 256)
_HIDDEN = 4096
_INTERMEDIATE = 512
_TOP_K = 8


def _sylvester(order: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < order:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h


def _block_diag(base: torch.Tensor, width: int, count: int) -> torch.Tensor:
    matrix = torch.zeros(width, width, dtype=torch.float64)
    step = base.shape[0]
    for index in range(count):
        matrix[index * step : (index + 1) * step, index * step : (index + 1) * step] = (
            base
        )
    return matrix


def _interleave_matrix(intermediate: int) -> torch.Tensor:
    """Explicit ``slot0-atom32-slot1-atom32`` permutation on the 2I axis."""

    perm = torch.zeros(2 * intermediate, 2 * intermediate, dtype=torch.float64)
    for block in range(intermediate // 32):
        for element in range(32):
            perm[block * 64 + element, block * 32 + element] = 1.0
            perm[block * 64 + 32 + element, intermediate + block * 32 + element] = 1.0
    return perm


def _interleave_vector(slot0: torch.Tensor, slot1: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(2 * slot0.numel(), dtype=torch.float64)
    for block in range(slot0.numel() // 32):
        out[block * 64 : block * 64 + 32] = slot0[block * 32 : block * 32 + 32]
        out[block * 64 + 32 : block * 64 + 64] = slot1[block * 32 : block * 32 + 32]
    return out


def _synthetic_sidecar(
    *, hidden: int, intermediate: int, experts: int, bits: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    k16, n16 = hidden // 16, intermediate // 16
    return {
        "w13_trellis": torch.randint(
            -32768,
            32767,
            (2, experts, k16, 2 * (intermediate // 32), 16 * bits),
            generator=generator,
            dtype=torch.int16,
        ),
        "w2_trellis": torch.randint(
            -32768,
            32767,
            (experts, 2 * (intermediate // 32), k16, 16 * bits),
            generator=generator,
            dtype=torch.int16,
        ),
        "w13_scale_ue8m0": torch.randint(
            120,
            130,
            (experts, 2 * intermediate, hidden // 32),
            generator=generator,
            dtype=torch.uint8,
        ),
        "w2_scale_ue8m0": torch.randint(
            120,
            130,
            (experts, hidden, intermediate // 32),
            generator=generator,
            dtype=torch.uint8,
        ),
        "intermediate_scales_fp16": (
            torch.randn(experts, 3 * intermediate, generator=generator) * 0.1 + 1.0
        ).half(),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=torch.uint8),
        "gate_up_suh_fp16": (
            torch.randn(hidden, generator=generator) * 0.1 + 1.0
        ).half(),
        "down_svh_fp16": (torch.randn(hidden, generator=generator) * 0.1 + 1.0).half(),
    }


# --------------------------------------------------------------------------- #
# Hadamard parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [128, 512])
@pytest.mark.parametrize("rows", [1, 7, 33])
def test_hadamard_matches_explicit_matrix(size: int, rows: int) -> None:
    torch.manual_seed(1000 + size + rows)
    value = torch.randn(rows, size, dtype=torch.float32)
    explicit = _sylvester(size) / math.sqrt(size)
    expected = (value.double() @ explicit.T).float()
    op = (
        boundary.h512_normalized(value)
        if size == 512
        else boundary.h128_normalized(value)
    )
    # Bound, not bitwise: the butterfly sums in a different order from the
    # explicit float64 matmul, so agreement is within a couple of float32 ULPs
    # of the largest magnitude (measured: <= 1 ULP).
    ulp = float(torch.abs(expected).max()) * 2.0**-23
    delta = float((op - expected).abs().max())
    assert delta <= 2.0 * ulp, (delta, ulp)


def test_hadamard_divisor_is_the_release_constant() -> None:
    # The P8 kernel's _P8_SQRT512_F32 / _P8_SQRT128_F32 pin these two
    # float32 encodings; the terminal divide is div.rn.f32 by exactly them,
    # not a reciprocal multiply.
    assert boundary.SQRT512_F32 == 22.627416610717773
    assert boundary.SQRT128_F32 == 11.313708305358887
    assert boundary.SQRT128_F32**2 == pytest.approx(128.0, rel=1e-7)
    assert boundary.SQRT512_F32**2 == pytest.approx(512.0, rel=1e-7)


# --------------------------------------------------------------------------- #
# boundary ops vs the reference, bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows", _M_SWEEP)
def test_input_boundary_bitwise_vs_reference(rows: int) -> None:
    torch.manual_seed(rows)
    x = torch.randn(rows, _HIDDEN).bfloat16()
    suh = (torch.randn(_HIDDEN) * 0.25 + 1.0).half()
    op = boundary.input_boundary(x, suh, hidden_size=_HIDDEN)
    ref = reference.trellismx_input_boundary(
        x, suh, hidden_size=_HIDDEN, quantize=False
    )
    assert torch.equal(op, ref)
    assert op.dtype == torch.float32


def test_input_boundary_pre_h128_is_the_declared_prefix() -> None:
    torch.manual_seed(11)
    x = torch.randn(5, _HIDDEN).bfloat16()
    suh = (torch.randn(_HIDDEN) * 0.25 + 1.0).half()
    prefix = boundary.input_boundary_pre_h128(x, suh, hidden_size=_HIDDEN)
    full = boundary.input_boundary(x, suh, hidden_size=_HIDDEN)
    assert torch.equal(full, boundary.h128_normalized(prefix))
    # The prefix must differ from a bf16 round-trip of itself, which is exactly
    # why it cannot be stored between the H512 and the H128.
    assert not torch.equal(prefix, prefix.bfloat16().float())


@pytest.mark.parametrize("rows", [1, 8, 256])
def test_input_boundary_quantized_roundtrip(rows: int) -> None:
    torch.manual_seed(200 + rows)
    x = torch.randn(rows, _HIDDEN).bfloat16()
    suh = (torch.randn(_HIDDEN) * 0.25 + 1.0).half()
    payload, scales = boundary.input_boundary_quantized(x, suh, hidden_size=_HIDDEN)
    assert payload.shape == (rows, _HIDDEN) and payload.dtype == torch.uint8
    assert scales.shape == (rows, _HIDDEN // 32) and scales.dtype == torch.uint8
    perm = torch.tensor(boundary.K32_PERMUTATION)
    blocks = payload.view(torch.float8_e4m3fn).float().reshape(rows, -1, 32)
    blocks = blocks.index_select(-1, torch.argsort(perm)).reshape(rows, _HIDDEN)
    recon = blocks * torch.pow(
        torch.tensor(2.0), (scales.float() - 127.0)
    ).repeat_interleave(32, dim=1)
    expected = boundary.quantize_e4m3_ue8m0_k32(
        boundary.input_boundary(x, suh, hidden_size=_HIDDEN)
    )
    assert torch.equal(recon, expected)


def test_k32_codes_on_tiny_and_zero_blocks() -> None:
    """The reference bytes the fused quantizer must reproduce at the low end.

    A block whose ``amax / 448`` is subnormal or underflows to zero clamps to
    exponent -127 (byte 0, scale ``2**-127``); an all-zero block keeps byte 127.
    """

    tiny = 2.0**-149
    value = torch.zeros(5, 32)
    value[1, :2] = torch.tensor([2.0**-140, -tiny])  # quotient 2**-149
    value[2, 0] = 224 * tiny  # quotient rounds to zero
    value[3, :2] = torch.tensor([448 * 2.0**-127, 2.0**-130])  # quotient 2**-127
    value[4, 0] = 448 * 2.0**-126  # quotient 2**-126, the smallest normal
    payload, scales = boundary.quantize_e4m3_ue8m0_k32_codes(value.reshape(1, -1))
    assert scales.tolist() == [[127, 0, 0, 0, 1]]
    natural = torch.argsort(torch.tensor(boundary.K32_PERMUTATION))
    codes = payload.reshape(5, 32).index_select(-1, natural)
    expected = torch.zeros(5, 32, dtype=torch.uint8)
    expected[1, 1] = 0x80  # -2**-22 rounds to -0
    expected[3, :2] = torch.tensor([0x7E, 0x20])  # 448 and 2**-3
    expected[4, 0] = 0x7E
    assert torch.equal(codes, expected)


def test_fused_boundary_has_exactly_two_modes() -> None:
    """``e4m3`` and ``fp16`` are the only input-stage outputs; nothing falls through."""

    from b12x.moe._shared.kernels.trellismx_boundary_fused import (
        BOUNDARY_KERNEL_VERSION,
        TrellisMxFusedBoundary,
    )

    for mode in ("e4m3", "fp16"):
        kernel = TrellisMxFusedBoundary(4096, mode)
        # Instance state is part of the compile key, so the version is too.
        assert vars(kernel)["version"] == BOUNDARY_KERNEL_VERSION
    for mode in ("pre", "fp32", "bf16"):
        with pytest.raises(ValueError, match="unsupported mode"):
            TrellisMxFusedBoundary(4096, mode)


@pytest.mark.parametrize("rows", [1, 32])
def test_output_boundary_bitwise_vs_reference(rows: int) -> None:
    torch.manual_seed(300 + rows)
    down = torch.randn(rows, _HIDDEN) * 2.0
    svh = (torch.randn(_HIDDEN) * 0.25 + 1.0).half()
    assert torch.equal(
        boundary.output_boundary(down, svh, hidden_size=_HIDDEN),
        reference.trellismx_output_boundary(down, svh, hidden_size=_HIDDEN),
    )


@pytest.mark.parametrize("rows", _M_SWEEP)
def test_coupled_topk_sum_bitwise_vs_reference(rows: int) -> None:
    torch.manual_seed(400 + rows)
    routes = torch.randn(rows, _TOP_K, _HIDDEN)
    weights = torch.rand(rows, _TOP_K)
    weights = weights / weights.sum(-1, keepdim=True)
    op = boundary.coupled_topk_sum(routes, weights, hidden_size=_HIDDEN)
    ref = reference.trellismx_coupled_topk_sum(routes, weights, hidden_size=_HIDDEN)
    assert torch.equal(op, ref)
    assert op.dtype == torch.bfloat16


def test_topk_sum_order_is_router_order() -> None:
    # The reduction is left-to-right over routes; a different order can change
    # the last FP32 bit, so pin that the implementation is order-faithful.
    routes = torch.zeros(1, 3, _HIDDEN)
    routes[0, 0] = 1.0
    routes[0, 1] = 2.0**-24
    routes[0, 2] = 2.0**-24
    weights = torch.ones(1, 3)
    total = (routes[0, 0] * 1.0 + routes[0, 1] * 1.0) + routes[0, 2] * 1.0
    expected = boundary.h512_normalized(total[None, :]).to(torch.bfloat16)
    assert torch.equal(
        boundary.coupled_topk_sum(routes, weights, hidden_size=_HIDDEN), expected
    )


@pytest.mark.parametrize("rows", [1, 8])
def test_silu_cap10_bitwise_vs_reference(rows: int) -> None:
    torch.manual_seed(500 + rows)
    gate = torch.randn(rows, _INTERMEDIATE) * 6.0
    up = torch.randn(rows, _INTERMEDIATE) * 6.0
    assert torch.equal(boundary.silu_cap10(gate, up), reference.silu_cap10(gate, up))


def test_silu_cap10_caps_gate_up_and_only_gate() -> None:
    gate = torch.tensor([[12.0, -12.0, 0.0]])
    up = torch.tensor([[12.0, -12.0, 5.0]])
    out = boundary.silu_cap10(gate, up)
    # gate capped at 10 one-sided, up capped two-sided, no post-sigmoid clamp.
    assert float(out[0, 0]) == pytest.approx(
        10.0 * torch.sigmoid(torch.tensor(10.0)).item() * 10.0
    )
    assert float(out[0, 1]) == pytest.approx(
        (-12.0 * torch.sigmoid(torch.tensor(-12.0)).item()) * -10.0
    )
    assert float(out[0, 2]) == 0.0


def test_k32_permutation_matches_trellis_decode() -> None:
    assert tuple(boundary.K32_PERMUTATION) == tuple(_W4A8_TRELLIS_A_K32_PERM)
    assert sorted(boundary.K32_PERMUTATION) == list(range(32))


def test_compact_activation_rejects_trellismx() -> None:
    # The compact (split-materialized) producer has no coupled boundary; asking
    # it for TrellisMX must fail closed rather than compute the ordinary one.
    from b12x.moe._shared.kernels.w4a8_compact_activation import (
        W4A8CompactMicroActivationKernel,
    )

    W4A8CompactMicroActivationKernel(
        512, 8, swiglu_limit=10.0, fast_math=False
    )  # unchanged default path still constructs
    with pytest.raises(NotImplementedError, match="coupled intermediate boundary"):
        W4A8CompactMicroActivationKernel(
            512, 8, swiglu_limit=10.0, fast_math=False, trellis_coupled=True
        )


def test_spec_fails_loudly_on_unsupported_geometry() -> None:
    boundary.TrellisMxBoundarySpec(hidden_size=4096, intermediate_size=512)
    with pytest.raises(ValueError, match="multiple of 512"):
        boundary.TrellisMxBoundarySpec(hidden_size=4032, intermediate_size=512)
    with pytest.raises(ValueError, match="multiple of 128"):
        boundary.TrellisMxBoundarySpec(hidden_size=4096, intermediate_size=96)
    with pytest.raises(ValueError, match="K2..K5"):
        boundary.TrellisMxBoundarySpec(hidden_size=4096, intermediate_size=512, bits=6)


# --------------------------------------------------------------------------- #
# declared-order closure: reference vs an explicit float64 dense computation
# --------------------------------------------------------------------------- #
def _explicit_float64_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    *,
    hidden: int,
    intermediate: int,
    experts: int,
    bits: int,
) -> torch.Tensor:
    h512 = _block_diag(_sylvester(512) / math.sqrt(512), hidden, hidden // 512)
    h128_h = _block_diag(_sylvester(128) / math.sqrt(128), hidden, hidden // 128)
    h128_i = _block_diag(
        _sylvester(128) / math.sqrt(128), intermediate, intermediate // 128
    )
    h128_2i = _block_diag(
        _sylvester(128) / math.sqrt(128), 2 * intermediate, 2 * intermediate // 128
    )
    perm = _interleave_matrix(intermediate)

    suh = tensors["gate_up_suh_fp16"].double()
    dsvh = tensors["down_svh_fp16"].double()
    scales = tensors["intermediate_scales_fp16"].double()
    gate_svh, up_svh, down_suh = scales.split(intermediate, dim=1)
    signs = reference.rank_local_coupled_signs(
        intermediate=intermediate, rank=0, world_size=4, draw=0
    ).double()
    pre, post = signs[: 2 * intermediate], signs[2 * intermediate :]

    source = (x.double().half().double() @ h512.T) * suh
    source = source @ h128_h.T

    decoded = {}
    for expert in range(experts):
        decoded[expert] = (
            reference.decode_trellis_matrix(
                tensors["w13_trellis"][0, expert],
                tensors["w13_scale_ue8m0"][expert],
                bits=bits,
                out_features=intermediate,
                in_features=hidden,
                scale_row_offset=intermediate,
            ).double(),
            reference.decode_trellis_matrix(
                tensors["w13_trellis"][1, expert],
                tensors["w13_scale_ue8m0"][expert],
                bits=bits,
                out_features=intermediate,
                in_features=hidden,
                scale_row_offset=0,
            ).double(),
            reference.decode_trellis_matrix(
                tensors["w2_trellis"][expert],
                tensors["w2_scale_ue8m0"][expert],
                bits=bits,
                out_features=hidden,
                in_features=intermediate,
            ).double(),
        )

    routes = torch.zeros(x.shape[0], topk_ids.shape[1], hidden, dtype=torch.float64)
    for token in range(x.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            gate_w, up_w, down_w = decoded[expert]
            g = (source[token : token + 1] @ gate_w.T).half().double()
            u = (source[token : token + 1] @ up_w.T).half().double()
            p = torch.cat([g, u], dim=1) @ perm.T @ h128_2i.T
            p = p * _interleave_vector(gate_svh[expert], up_svh[expert]) @ h128_2i.T
            p = p * pre
            gate_joint, up_joint = p[:, 0::2], p[:, 1::2]
            act = (
                torch.clamp(gate_joint, max=10.0)
                * torch.sigmoid(torch.clamp(gate_joint, max=10.0))
                * torch.clamp(up_joint, min=-10.0, max=10.0)
            )
            act = (act * post) @ h128_i.T
            act = act * down_suh[expert] @ h128_i.T
            y = (act @ down_w.T).half().double()
            routes[token, route] = ((y @ h128_h.T) * dsvh)[0]

    total = torch.zeros(x.shape[0], hidden, dtype=torch.float64)
    for k in range(topk_ids.shape[1]):
        total = total + topk_weights[:, k].double()[:, None] * routes[:, k]
    return (total @ h512.T).to(torch.bfloat16)


@pytest.mark.parametrize("rows", _M_SWEEP)
def test_reference_matches_float64_dense(rows: int) -> None:
    hidden, intermediate, experts, bits = 512, 128, 6, 4
    torch.manual_seed(600 + rows)
    tensors = _synthetic_sidecar(
        hidden=hidden, intermediate=intermediate, experts=experts, bits=bits, seed=rows
    )
    x = torch.randn(rows, hidden).bfloat16()
    # Mixed routing with repeats: six experts, top-8, so a row cannot be a
    # permutation and experts recur across and inside rows.
    ids = torch.randint(
        0, experts, (rows, _TOP_K), generator=torch.Generator().manual_seed(rows)
    ).to(torch.int64)
    weights = torch.rand(rows, _TOP_K)
    weights = weights / weights.sum(-1, keepdim=True)
    meta = {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_experts": experts,
        "top_k": _TOP_K,
        "bits": bits,
        "rank": 0,
        "world_size": 4,
        "quantize_activations": False,
        "emulate_fp16_stores": True,
    }
    got = reference.trellismx_moe_reference(x, ids, weights, tensors, meta)
    want = _explicit_float64_moe(
        x,
        ids,
        weights,
        tensors,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        bits=bits,
    )
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    # Bound: the reference accumulates in FP32 and stores BF16; the oracle is
    # float64. One BF16 ULP of the largest magnitude is the correct bound.
    ulp = torch.abs(want.float()).max() * 2.0**-8
    delta = (got.float() - want.float()).abs().max()
    assert float(delta) <= float(ulp), (float(delta), float(ulp))


def test_reference_rejects_misshaped_sidecar() -> None:
    tensors = _synthetic_sidecar(
        hidden=512, intermediate=128, experts=2, bits=4, seed=1
    )
    tensors["gate_up_suh_fp16"] = tensors["gate_up_suh_fp16"][:256]
    meta = {
        "hidden_size": 512,
        "intermediate_size": 128,
        "num_experts": 2,
        "top_k": 2,
        "bits": 4,
    }
    x = torch.randn(1, 512).bfloat16()
    ids = torch.zeros(1, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="gate_up_suh_fp16"):
        reference.trellismx_moe_reference(x, ids, torch.ones(1, 2), tensors, meta)


# --------------------------------------------------------------------------- #
# CUDA graph replay
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_replay_is_bitwise() -> None:
    device = torch.device("cuda")
    rows = 32
    torch.manual_seed(700)
    x_static = torch.randn(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
    suh = (torch.randn(_HIDDEN, device=device) * 0.25 + 1.0).half()
    routes_static = torch.randn(rows, _TOP_K, _HIDDEN, device=device)
    weights = torch.rand(rows, _TOP_K, device=device)
    weights = weights / weights.sum(-1, keepdim=True)

    def run(x, routes):
        source = boundary.input_boundary(x, suh, hidden_size=_HIDDEN)
        payload, scales = boundary.input_boundary_quantized(x, suh, hidden_size=_HIDDEN)
        out = boundary.coupled_topk_sum(routes, weights, hidden_size=_HIDDEN)
        return source, payload, scales, out

    eager = run(x_static, routes_static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(x_static, routes_static)
    graph.replay()
    for got, want in zip(captured, eager):
        assert torch.equal(got, want)

    # Replay with fresh contents through the same static buffers.
    x_static.copy_(torch.randn_like(x_static))
    routes_static.copy_(torch.randn_like(routes_static))
    graph.replay()
    fresh = run(x_static, routes_static)
    for got, want in zip(captured, fresh):
        assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# the axis orders come from checkpoint metadata, not from a constant
# --------------------------------------------------------------------------- #
def test_released_sidecar_declares_the_two_axis_orders() -> None:
    """The reference must read these, so pin the released values.

    `fc1_scale_plane_order: up-gate` (the UE8M0 N axis) against
    `fc1_trellis_slot_order: gate-up` (the payload matrix axis) is the mismatch
    the release's own staging relies on (route_hoist_k5.py:330-336).
    """

    from pathlib import Path

    from ._trellismx_data import glm_sidecar

    sidecar = Path(glm_sidecar(3, 0))
    if not sidecar.exists():
        pytest.skip("GLM TrellisMX release sidecar is not present")
    from safetensors import safe_open

    with safe_open(str(sidecar), "np") as handle:
        metadata = dict(handle.metadata())
    assert metadata["fc1_scale_plane_order"] == "up-gate"
    assert metadata["fc1_trellis_slot_order"] == "gate-up"


@pytest.mark.parametrize(
    "override,expected_in_message",
    [
        ({"fc1_scale_plane_order": "sideways"}, "up-gate"),
        ({"fc1_trellis_slot_order": "sideways"}, "gate-up"),
    ],
)
def test_unknown_axis_orders_raise(override, expected_in_message) -> None:
    tensors = _synthetic_sidecar(
        hidden=512, intermediate=128, experts=2, bits=4, seed=3
    )
    meta = {
        "hidden_size": 512,
        "intermediate_size": 128,
        "num_experts": 2,
        "top_k": 2,
        "bits": 4,
        **override,
    }
    x = torch.randn(1, 512).bfloat16()
    ids = torch.zeros(1, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match=expected_in_message):
        reference.trellismx_moe_reference(x, ids, torch.ones(1, 2), tensors, meta)


@pytest.mark.parametrize(
    "override",
    [
        {"fc1_scale_plane_order": "gate-up"},
        {"fc1_trellis_slot_order": "up-gate"},
    ],
)
def test_flipping_either_order_changes_the_result(override) -> None:
    """A hard-coded order would be caught: both flips move the output.

    The scale-plane flip only perturbs rows whose two plane halves differ, and
    the slot flip swaps gate and up; both change the arithmetic, which is why
    the reference reads them instead of assuming them.
    """

    hidden, intermediate, experts, bits = 512, 128, 2, 4
    tensors = _synthetic_sidecar(
        hidden=hidden, intermediate=intermediate, experts=experts, bits=bits, seed=5
    )
    base = {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_experts": experts,
        "top_k": 2,
        "bits": bits,
        "quantize_activations": False,
    }
    torch.manual_seed(31)
    x = torch.randn(2, hidden).bfloat16()
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.full((2, 2), 0.5)
    declared = reference.trellismx_moe_reference(x, ids, weights, tensors, dict(base))
    flipped = reference.trellismx_moe_reference(
        x, ids, weights, tensors, {**base, **override}
    )
    assert not torch.equal(declared, flipped), override
