"""``source_format="trellismx"``: preparation, gates and the plan/bind/run route.

TrellisMX is another *weight source format* under the existing fused-MoE API,
not a new operator. These tests cover, on CPU, the whole preparation contract
(metadata gates, shape gates, carrier preservation, scale-plane mapping, the
lossless CSF expansion) and, on GPU, one small synthetic layer driven through
``plan_execution`` / ``bind`` / ``run`` against the reference.

The synthetic fixture uses unit UE8M0 weight scales (byte 127) on purpose: it
isolates the container/routing path from the kernel's weight-scale arm, whose
real-data evidence is ``test_trellismx_realdata.py``.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch

from b12x.moe.fused_moe import planning as fused_planning
from b12x.moe.fused_moe.config import TrellisCodebook
from b12x.moe.fused_moe.source import TrellisSource
from b12x.moe.fused_moe.trellismx import (
    COMPANDER_SCALE,
    TrellisMxMetadata,
    build_trellismx_weights,
    decode_csf_scale_plane,
    encode_csf_scale_plane,
    prepare_trellismx_experts,
)
from b12x.moe.fused_moe.weights import WeightPacking

from ._trellismx_data import ds41_sidecar, glm_sidecar

_DS41_K2 = ds41_sidecar(2, 0)

# The coupled boundary's outer stage is H512, so the smallest legal synthetic
# layer is hidden=512 with a 128-channel rank slice.
_E, _H, _I, _BITS, _TOP_K = 32, 512, 256, 4, 4


def _sidecar(
    *,
    experts: int = _E,
    hidden: int = _H,
    intermediate: int = _I,
    bits: int = _BITS,
    top_k: int = _TOP_K,
    seed: int = 20261022,
    scale_byte: int = 127,
    csf: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """A small released-shaped sidecar (plain or CSF scales)."""

    generator = torch.Generator().manual_seed(seed)
    hidden_tiles = hidden // 16
    local_tiles = intermediate // 16
    words = 16 * bits
    tensors = {
        "w13_trellis": torch.randint(
            0,
            1 << 15,
            (2, experts, hidden_tiles, local_tiles, words),
            generator=generator,
            dtype=torch.int16,
        ),
        "w2_trellis": torch.randint(
            0,
            1 << 15,
            (experts, local_tiles, hidden_tiles, words),
            generator=generator,
            dtype=torch.int16,
        ),
        "intermediate_scales_fp16": (
            torch.randint(0, 2, (experts, 3 * intermediate), generator=generator)
            .mul_(2)
            .sub_(1)
            .to(torch.float16)
            * 1.0625
        ),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=torch.uint8),
        "gate_up_suh_fp16": torch.ones(hidden, dtype=torch.float16),
        "down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
    }
    w13_rows, w13_cols = 2 * intermediate, hidden // 32
    w2_rows, w2_cols = hidden, intermediate // 32
    if csf:
        grid13 = torch.randint(
            scale_byte,
            scale_byte + 3,
            (experts, w13_rows, w13_cols),
            generator=generator,
            dtype=torch.uint8,
        )
        grid2 = torch.randint(
            scale_byte,
            scale_byte + 3,
            (experts, w2_rows, w2_cols),
            generator=generator,
            dtype=torch.uint8,
        )
        bases13 = grid13.amin(dim=-1)
        bases2 = grid2.amin(dim=-1)
        plane13, records13, offsets13 = _encode_per_expert(grid13, bases13)
        plane2, records2, offsets2 = _encode_per_expert(grid2, bases2)
        # One record arena per side, shared by every expert segment.
        tensors.update(
            {
                "w13_scale.mxfp4_csf_fixed": plane13,
                "w2_scale.mxfp4_csf_fixed": plane2,
                "w13_exceptions": records13.view(torch.int32),
                "w2_exceptions": records2.view(torch.int32),
                "w13_exception_offsets": offsets13,
                "w2_exception_offsets": offsets2,
            }
        )
    else:
        tensors["w13_scale_ue8m0"] = torch.full(
            (experts, w13_rows, w13_cols), scale_byte, dtype=torch.uint8
        )
        tensors["w2_scale_ue8m0"] = torch.full(
            (experts, w2_rows, w2_cols), scale_byte, dtype=torch.uint8
        )
    metadata = {
        "law": "procedural-npk",
        "compander_scale": "2.0",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "boundary": "coupled-h512-h128-suh-svh-v1",
        "activation": "silu-cap10",
        "bits": str(bits),
        "hidden": str(hidden),
        "experts_start": "0",
        "experts_end": str(experts),
        "intermediate_per_rank": str(intermediate),
        "topk": str(top_k),
        "world_size": "1",
        "slice_begin": "0",
        "slice_end": str(intermediate),
        "schema": "test-trellismx-sidecar.v1",
    }
    return tensors, metadata


def _prepare(tensors, metadata, **overrides):
    arguments = dict(
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        top_k=_TOP_K,
        device=torch.device("cpu"),
    )
    arguments.update(overrides)
    return prepare_trellismx_experts(tensors, metadata, **arguments)


def _encode_per_expert(grid: torch.Tensor, bases: torch.Tensor):
    """Encode a grid with one exception segment per expert (as a release does)."""

    planes, records, offsets = [], [], [0]
    for expert in range(int(grid.shape[0])):
        plane, arena = encode_csf_scale_plane(
            grid[expert : expert + 1], bases[expert : expert + 1]
        )
        planes.append(plane)
        records.append(arena)
        offsets.append(offsets[-1] + int(arena.numel()))
    return (
        torch.cat(planes),
        torch.cat(records) if records else torch.zeros(0, dtype=torch.uint32),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# preparation: the released tensors reach the prepared layout unchanged
# --------------------------------------------------------------------------- #


def test_preparation_preserves_every_codeword() -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    value = prepared._impl.representation.value

    assert _sha(value.w13) == _sha(tensors["w13_trellis"])
    assert _sha(value.w2) == _sha(tensors["w2_trellis"])
    # No permutation, no dtype change: the carrier is the checkpoint's own
    # assembled int16 tile stream.
    assert value.w13.dtype == torch.int16 and value.w2.dtype == torch.int16
    assert tuple(value.w13.shape) == (2, _E, _H // 16, _I // 16, 16 * _BITS)
    assert tuple(value.w2.shape) == (_E, _I // 16, _H // 16, 16 * _BITS)


def test_glm_rank_atoms_map_to_channel_offsets() -> None:
    """Every GLM rank's ``local_atom_begin`` is a 32-channel atom index.

    Layer 3 carries 0/16/32/48 for ranks 0..3, i.e. first slots 0/16/32/48 and
    first channels 0/512/1024/1536. Reading it as a channel index gives rank 2
    the window at channel 32 and nothing raises.
    """

    from safetensors import safe_open

    if not os.path.exists(glm_sidecar(3, 0)):
        pytest.skip("GLM-5.3 release is not mounted")
    for rank in range(4):
        path = glm_sidecar(3, rank)
        with safe_open(path, framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        assert int(metadata["rank"]) == rank
        assert int(metadata["local_atom_begin"]) == 16 * rank
        meta = TrellisMxMetadata.from_sidecar(
            metadata,
            num_experts=288,
            hidden_size=4096,
            intermediate_size=512,
            top_k=8,
        )
        assert meta.first_slot == 16 * rank, (rank, meta.first_slot)
        assert 32 * meta.first_slot == 512 * rank
        assert meta.slot_count == 16
        assert meta.global_intermediate_size == 2048
        extent = meta.source().extent
        assert extent.first_slot == 16 * rank


def test_trellismx_activation_is_silu_cap10() -> None:
    """The release is silu-cap10: SiLU, clamp 10.0, before the nonlinearity."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    activation = prepared.plan.activation
    assert activation.nonlinearity == "silu"
    assert activation.swiglu_limit == 10.0
    assert activation.swiglu_clamp_after_silu is False
    # A different activation is a different release and is rejected.
    with pytest.raises(ValueError, match="activation"):
        _prepare(tensors, dict(metadata, activation="situ-cap10"))


def test_a_scale_plane_stride_follows_the_recipe() -> None:
    """The A scale plane is one byte per K32 for the MX recipes.

    ``_DynamicMoEW4A8Launch`` declares ``rows_padded * (k // 32)`` and
    ``w4a8_phase1`` reads the row at ``tok * input_k128_tiles * 4`` = k/32
    bytes. Sizing the MX recipes at the FP4 width (one scale per 16 values)
    allocated twice the bytes the launch reads and halved
    ``_dynamic_rows_padded_limit``'s row budget for them; the launches take the
    plane flat, so the extra bytes were never read. ``tests/.../test_activation_scale_plane.py``
    checks the planned plane against the launch that consumes it.
    """

    from b12x.moe.fused_moe._impl import _activation_scale_cols_pad

    k = 4096
    for quant_mode in ("w4a8_mx", "w6a8_mx", "w8a8_mx"):
        assert _activation_scale_cols_pad(k, quant_mode) == k // 32 == 128, quant_mode
    # The FP4 recipes keep one E4M3 scale per 16 values, padded to the 4-block
    # atom column; w4a8_nvfp4 is an A4 recipe and must not take 32.
    for quant_mode in ("nvfp4", "w4a8_nvfp4"):
        assert _activation_scale_cols_pad(k, quant_mode) == k // 16 == 256, quant_mode
    # A capacity bound must agree with the allocation it bounds.
    from b12x.moe.fused_moe._impl import _dynamic_rows_padded_limit

    limit = _dynamic_rows_padded_limit(k, quant_mode="w4a8_mx")
    assert limit > 0
    assert limit * (k // 32) <= 0x7FFFFFFF


def test_prepared_boundary_ops_are_the_declared_stages() -> None:
    """The plan carries the coupled boundary as prepared, capturable ops.

    The arithmetic is `trellismx_boundary`'s; what this asserts is that
    preparation binds it to *this* layer's released vectors and that the
    composition is the declared order: input ``H512 -> suh -> H128 -> E4M3``
    before the kernel, output ``H128 -> svh``, router-order FP32 sum, ``H512``.
    """

    from b12x.testing.reference.trellismx_moe import (
        trellismx_input_boundary,
        trellismx_output_boundary,
    )

    tensors, metadata = _sidecar()
    tensors["gate_up_suh_fp16"] = (
        torch.rand(_H, generator=torch.Generator().manual_seed(7))
        .add_(0.5)
        .to(torch.float16)
    )
    tensors["down_svh_fp16"] = (
        torch.rand(_H, generator=torch.Generator().manual_seed(8))
        .add_(0.5)
        .to(torch.float16)
    )
    prepared = _prepare(tensors, metadata)
    boundary = prepared._impl.representation.value.boundary

    x = (torch.randn(4, _H, generator=torch.Generator().manual_seed(9)) * 0.02).to(
        torch.bfloat16
    )
    # input: FP32 continuity through both Hadamard stages and the side vector.
    want = trellismx_input_boundary(
        x, tensors["gate_up_suh_fp16"], hidden_size=_H, quantize=False
    )
    got = boundary.input_boundary(x)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    # and the quantized operands are one byte per value / per K32 block.
    payload, scales = boundary.input_boundary_quantized(x)
    assert payload.dtype == torch.uint8 and tuple(payload.shape) == (4, _H)
    assert scales.dtype == torch.uint8 and tuple(scales.shape) == (4, _H // 32)

    # output: per-route H128 then *down_svh in FP32, then the coupled sum.
    routes_fp16 = (
        torch.randn(4 * _TOP_K, _H, generator=torch.Generator().manual_seed(10)) * 0.5
    ).to(torch.float16)
    want_route = trellismx_output_boundary(
        routes_fp16, tensors["down_svh_fp16"], hidden_size=_H
    )
    got_route = boundary.output_boundary(routes_fp16)
    torch.testing.assert_close(got_route, want_route, rtol=1e-6, atol=1e-6)
    weights = torch.rand(4, _TOP_K, generator=torch.Generator().manual_seed(11))
    weights = weights / weights.sum(-1, keepdim=True)
    out = boundary.apply_output(routes_fp16, weights, top_k=_TOP_K)
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (4, _H)
    # The FC2 result must arrive at the declared FP16 store.
    with pytest.raises(TypeError, match="fp16"):
        boundary.apply_output(routes_fp16.float(), weights, top_k=_TOP_K)


#: The released geometries a trellismx plan must accept at every M: the
#: synthetic layer, GLM TP4, and DS41 K4's two uneven rank widths.
_PLAN_GEOMETRIES = (
    (_E, _H, _I, _TOP_K),
    (288, 4096, 512, 8),
    (384, 5120, 640, 6),
    (384, 5120, 512, 6),
)


def _trellismx_query(
    num_tokens: int,
    geometry: tuple[int, int, int, int] | None = None,
):
    """The decode query a trellismx plan produces, at one token capacity."""

    from b12x.moe.fused_moe._tuning import MoeDecodeQuery
    from b12x.preparation.types import FrozenMapping

    experts, hidden, intermediate, top_k = geometry or (_E, _H, _I, _TOP_K)
    return MoeDecodeQuery(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="trellismx",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        num_tokens=num_tokens,
        routed_rows=num_tokens * top_k,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        w13_layout="w13",
        weight_layouts=("trellis_native",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=False,
        numerical_recipe=None,
        controls=FrozenMapping(),
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 256])
@pytest.mark.parametrize("geometry", _PLAN_GEOMETRIES)
def test_plans_select_the_monolithic_trellis_arm(
    num_tokens: int, geometry: tuple[int, int, int, int]
) -> None:
    """No M and no released geometry reaches micro or split-materialized.

    Covers the DS41 K4 widths (640/512) as well as GLM's, because a trellismx
    plan must pick a monolithic-supported tile/route at *every* M and width.
    """

    from b12x.moe.fused_moe._impl import (
        _heuristic_dynamic_route_mode,
        _heuristic_moe_decode_config,
        _w4a8_dynamic_materialized_enabled,
        _w4a8_mx_micro_preferred,
    )

    query = _trellismx_query(num_tokens, geometry)
    config = _heuristic_moe_decode_config(query, device=None)
    assert config.backend == "dynamic", (num_tokens, config.backend)
    assert config.dynamic_route_mode == "grouped", config.dynamic_route_mode
    assert not _w4a8_mx_micro_preferred(query, device=None)
    assert _heuristic_dynamic_route_mode(query, None, planned_tile_m=16) == "grouped"
    # The monolithic arm at every M: no materialized-intermediate
    # specialization for the procedural-NPK law.
    for planned_tile_m in (16, 32, 64, 128):
        assert not _w4a8_dynamic_materialized_enabled(
            quant_mode="w4a8_mx",
            activation="silu",
            num_tokens=num_tokens,
            routed_rows=num_tokens * _TOP_K,
            num_experts=_E,
            k=_H,
            n=_I,
            w4a8_repacked=True,
            share_input_across_experts=True,
            deterministic_output=False,
            planned_tile_m=planned_tile_m,
            trellis_codebook="procedural_npk",
        ), (num_tokens, planned_tile_m)
    # The gate is specific to the codebook, not to the geometry: at a tile
    # config the dense candidate qualifies for, the same call returns True
    # without the procedural-NPK declaration and False with it.
    arguments = dict(
        quant_mode="w4a8_mx",
        activation="silu",
        num_tokens=8,
        routed_rows=8 * _TOP_K,
        num_experts=_E,
        k=_H,
        n=_I,
        w4a8_repacked=True,
        share_input_across_experts=True,
        deterministic_output=False,
        planned_tile_m=64,
    )
    assert _w4a8_dynamic_materialized_enabled(**arguments, trellis_codebook="lut_e4m3")
    assert not _w4a8_dynamic_materialized_enabled(
        **arguments, trellis_codebook="procedural_npk"
    )
    # Omitted codebook (the W4A16/QMMA callers' default) keeps the old answer.
    assert _w4a8_dynamic_materialized_enabled(**arguments)


def test_prepared_contract_is_the_w4a8_trellis_recipe() -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    plan = prepared.plan

    assert plan.source.source_format == "trellismx"
    assert plan._impl.source_format == "trellismx"
    assert plan._impl.quant_modes == frozenset({"w4a8_mx"})
    assert plan.prepared_format.packing is WeightPacking.TRELLIS_NATIVE
    assert plan.prepared_format.weights.value == "trellis"
    assert plan.prepared_format.scales.value == "e8m0_k32"
    assert plan._impl.trellis_codebook == "procedural_npk"
    assert plan._impl.trellis_bits == _BITS
    assert plan._impl.intermediate_hadamard
    assert plan._impl.intermediate_hadamard_blocks == (128, 128)
    assert plan.activation.mode.value == "a8"
    assert plan.activation.swiglu_limit == 10.0


def test_scale_planes_reach_the_sfb_layout_as_a_permutation() -> None:
    tensors, metadata = _sidecar()
    tensors["w13_scale_ue8m0"] = torch.randint(
        100, 200, tensors["w13_scale_ue8m0"].shape, dtype=torch.uint8
    )
    tensors["w2_scale_ue8m0"] = torch.randint(
        100, 200, tensors["w2_scale_ue8m0"].shape, dtype=torch.uint8
    )
    prepared = _prepare(tensors, metadata)
    value = prepared._impl.representation.value

    # The repack is in-place, so the logical planes must be byte-preserved as a
    # multiset: same bytes, different order. Nothing is requantized or clamped.
    for plane, sfb in (
        (value.w13_scale, value.w13_sfb),
        (value.w2_scale, value.w2_sfb),
    ):
        assert plane.dtype == torch.uint8
        assert torch.equal(
            plane.flatten().sort().values, sfb.view(torch.uint8).flatten().sort().values
        )
    assert tuple(value.w13_sfb.shape)[1:] == (
        (2 * _I) // 256,
        _H // 128,
        32,
        8,
    )
    assert tuple(value.w2_sfb.shape)[1:] == (_H // 256, _I // 128, 32, 8)


def test_boundary_tensors_are_the_released_ones() -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    value = prepared._impl.representation.value

    # [E, 6*I]: the released per-expert values followed by the appended sign
    # draw, exactly what the coupled boundary consumes.
    assert tuple(value.intermediate_rotations.shape) == (_E, 6 * _I)
    assert value.intermediate_rotations.dtype == torch.float16
    assert torch.equal(
        value.intermediate_rotations[:, : 3 * _I],
        tensors["intermediate_scales_fp16"],
    )
    # sign draw 0 is all +1.
    assert torch.equal(
        value.intermediate_rotations[:, 3 * _I :],
        torch.ones(_E, 3 * _I, dtype=torch.float16),
    )
    assert torch.equal(value.gate_suh[0], tensors["gate_up_suh_fp16"])
    assert torch.equal(value.up_suh[0], tensors["gate_up_suh_fp16"])
    assert torch.equal(value.down_svh[0], tensors["down_svh_fp16"])


def test_preparation_is_a_fused_moe_source_format_not_a_new_operator() -> None:
    """The plan/bind/run lifecycle consumes it through ``prepare_weights``."""

    from b12x.moe.fused_moe.weights import PreparedExperts

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    assert isinstance(prepared, PreparedExperts)
    assert prepared.num_experts == _E
    assert prepared.hidden_size == _H
    assert prepared.intermediate_size == _I
    assert prepared.plan is prepared.plan  # canonical WeightPlan
    assert prepared._impl.plan is prepared.plan._impl


# --------------------------------------------------------------------------- #
# metadata gates: every one raises, naming its field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("law", "procedural-mcg-alpha3", "law"),
        ("compander_scale", "1.5", "compander_scale"),
        ("alphabet", "e5m2", "alphabet"),
        ("scale", "e4m3-k16", "scale"),
        ("boundary", "coupled-h128-h128-suh-svh-v1", "boundary"),
        ("activation", "silu", "activation"),
        ("bits", "5", "bits"),
        ("hidden", "1024", "hidden"),
        ("experts_end", "8", "experts_end"),
        ("intermediate_per_rank", "512", "intermediate_per_rank"),
        ("topk", "3", "topk"),
        ("experts_start", "1", "experts_start"),
        ("slice_end", "512", "slice_begin"),
    ],
)
def test_metadata_gates_raise(field: str, value: str, match: str) -> None:
    tensors, metadata = _sidecar()
    metadata = dict(metadata)
    metadata[field] = value
    with pytest.raises(ValueError, match=match):
        _prepare(tensors, metadata)


def test_missing_compander_is_required_for_the_canonical_law() -> None:
    tensors, metadata = _sidecar()
    metadata = dict(metadata)
    del metadata["compander_scale"]
    with pytest.raises(ValueError, match="compander_scale"):
        _prepare(tensors, metadata)


def test_alias_law_implies_the_compander_and_takes_declared_geometry() -> None:
    """GLM-5.3 sidecars: the old label, and no geometry fields at all."""

    tensors, metadata = _sidecar()
    glm = {
        "law": "procedural-mcg-alpha2",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "boundary": "coupled-h512-h128-suh-svh-v1",
        "activation": "silu-cap10",
        "bits": str(_BITS),
        "world_size": "1",
        "local_atom_begin": "0",
        "global_intermediate": str(_I),
    }
    prepared = _prepare(tensors, glm)
    assert prepared._impl.representation.value.trellis_bits == _BITS
    # A present value other than 2.0 is still rejected under the alias.
    with pytest.raises(ValueError, match="compander_scale"):
        _prepare(tensors, dict(glm, compander_scale="2.5"))
    # A metadata geometry field that disagrees with the declaration is
    # rejected; a silent one is simply taken from the declaration.
    with pytest.raises(ValueError, match="hidden"):
        _prepare(tensors, dict(glm, hidden="1024"))
    with pytest.raises(ValueError, match="intermediate_per_rank"):
        _prepare(tensors, dict(glm, intermediate_per_rank="512"))


# --------------------------------------------------------------------------- #
# tensor gates
# --------------------------------------------------------------------------- #


def test_missing_tensor_raises_naming_it() -> None:
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    del tensors["w2_trellis"]
    with pytest.raises(ValueError, match="w2_trellis"):
        _prepare(tensors, metadata)


def test_wrong_dtype_raises_naming_it() -> None:
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    tensors["w13_trellis"] = tensors["w13_trellis"].to(torch.int32)
    with pytest.raises(ValueError, match="w13_trellis"):
        _prepare(tensors, metadata)


def test_wrong_shape_raises_naming_it() -> None:
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    tensors["w2_trellis"] = tensors["w2_trellis"][:, :-1]
    with pytest.raises(ValueError, match="w2_trellis"):
        _prepare(tensors, metadata)


def test_unknown_tensor_raises() -> None:
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    tensors["w13_scale_fp8"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="w13_scale_fp8"):
        _prepare(tensors, metadata)


def test_both_scale_families_raise() -> None:
    tensors, metadata = _sidecar(csf=True)
    tensors = dict(tensors)
    tensors["w13_scale_ue8m0"] = torch.full(
        (_E, 2 * _I, _H // 32), 127, dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="plain and CSF"):
        _prepare(tensors, metadata)


def test_scale_above_the_repack_range_is_refused() -> None:
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    tensors["w13_scale_ue8m0"] = torch.full(
        (_E, 2 * _I, _H // 32), 250, dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="247"):
        _prepare(tensors, metadata)


def test_planning_rejects_a16_and_foreign_codebooks() -> None:
    tensors, metadata = _sidecar()
    meta = TrellisMxMetadata.from_sidecar(
        metadata,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        top_k=_TOP_K,
    )
    with pytest.raises(ValueError, match="A8"):
        fused_planning.plan_weights(
            source=meta.source(),
            activation=fused_planning.ActivationSpec(
                mode=fused_planning.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_planning.MoEGeometry(
                num_experts=_E, hidden_size=_H, intermediate_size=_I
            ),
        )
    with pytest.raises(ValueError, match="procedural_npk"):
        TrellisSource(
            config=meta.config().__class__(
                **{**meta.config().__dict__, "codebook": TrellisCodebook.LUT_E4M3}
            ),
            uniform_bits=_BITS,
            extent=meta.source().extent,
            source_format="trellismx",
        )


def test_execution_registration() -> None:
    from b12x.moe._shared.execution import (
        WeightPreparationTransform,
        make_moe_spec,
        plan_moe_weight_preparation,
        validate_moe_source_quant,
    )

    validate_moe_source_quant(source_format="trellismx", quant_mode="w4a8_mx")
    with pytest.raises(ValueError, match="incompatible"):
        validate_moe_source_quant(source_format="trellismx", quant_mode="w4a16")
    spec = make_moe_spec(
        quant_mode="w4a8_mx",
        source_format="trellismx",
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )
    assert spec.source_weight_scale.value == "e8m0_k32"
    plan = plan_moe_weight_preparation(
        spec,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        trellis_bits=_BITS,
        trellis_codebook="procedural_npk",
        intermediate_hadamard=True,
        intermediate_hadamard_blocks=(128, 128),
    )
    assert WeightPreparationTransform.W4A8_TRELLIS in plan.transforms
    assert plan.w4a8_weight_layout == "trellis_t256"
    assert plan.required_weight_layout("w4a8_mx").value == "trellis_native"


def test_execution_rejects_a_foreign_trellismx_codebook() -> None:
    from b12x.moe._shared.execution import make_moe_spec, plan_moe_weight_preparation

    spec = make_moe_spec(
        quant_mode="w4a8_mx",
        source_format="trellismx",
        activation="silu",
        io_dtype="bfloat16",
        w13_layout="w13",
    )
    with pytest.raises(ValueError, match="procedural_npk"):
        plan_moe_weight_preparation(
            spec,
            num_experts=_E,
            hidden_size=_H,
            intermediate_size=_I,
            trellis_bits=_BITS,
            trellis_codebook="lut_e4m3",
        )


# --------------------------------------------------------------------------- #
# CSF: lossless storage expansion, byte-exact round trip
# --------------------------------------------------------------------------- #


def test_csf_sidecar_expands_to_the_identical_ue8m0_bytes() -> None:
    grid13 = torch.randint(100, 200, (_E, 2 * _I, _H // 32), dtype=torch.uint8)
    grid2 = torch.randint(100, 200, (_E, _H, _I // 32), dtype=torch.uint8)
    bases13 = grid13.amin(dim=-1)
    bases2 = grid2.amin(dim=-1)
    plane13, records13, offsets13 = _encode_per_expert(grid13, bases13)
    plane2, records2, offsets2 = _encode_per_expert(grid2, bases2)
    tensors, metadata = _sidecar()
    tensors = dict(tensors)
    for name in ("w13_scale_ue8m0", "w2_scale_ue8m0"):
        del tensors[name]
    tensors.update(
        {
            "w13_scale.mxfp4_csf_fixed": plane13,
            "w2_scale.mxfp4_csf_fixed": plane2,
            "w13_exceptions": records13.view(torch.int32),
            "w2_exceptions": records2.view(torch.int32),
            "w13_exception_offsets": offsets13,
            "w2_exception_offsets": offsets2,
        }
    )
    prepared = _prepare(tensors, metadata)
    value = prepared._impl.representation.value
    # CSF -> UE8M0 is a storage expansion only: the expanded bytes are the
    # grid the slab encoded, bit for bit.
    assert torch.equal(value.w13_scale, grid13)
    assert torch.equal(value.w2_scale, grid2)


def test_csf_expansion_rejects_a_corrupt_container() -> None:
    from b12x.moe.fused_moe.trellismx import csf_scale_source

    grid = torch.full((2, 64, 8), 120, dtype=torch.uint8)
    bases = torch.full((2, 64), 120, dtype=torch.uint8)
    plane, records = encode_csf_scale_plane(grid, bases)
    starts = torch.tensor([0, 0, records.numel()], dtype=torch.int64)
    source = csf_scale_source(plane, records, starts, rows=64, columns=8)
    expanded, _ = decode_csf_scale_plane(source)
    assert torch.equal(expanded, grid)
    # Positions must be strictly increasing inside one expert segment.
    if records.numel() >= 2:
        bad = torch.stack((records[0], records[0]))
        corrupt = csf_scale_source(
            plane, bad, torch.tensor([0, 0, 2], dtype=torch.int64), rows=64, columns=8
        )
        with pytest.raises(ValueError, match="sorted"):
            decode_csf_scale_plane(corrupt)


@pytest.mark.skipif(
    not os.path.exists(_DS41_K2), reason="DS41 K2 test data is not mounted"
)
def test_csf_round_trip_is_byte_exact_on_a_real_ds41_plane() -> None:
    """The released slab decodes and re-encodes byte for byte."""

    from safetensors import safe_open

    with safe_open(_DS41_K2, framework="pt") as handle:
        plane = handle.get_tensor("w13_scale.mxfp4_csf_fixed")
        records = handle.get_tensor("w13_exceptions")
        starts = handle.get_tensor("w13_exception_offsets")
        metadata = dict(handle.metadata() or {})
    rows, columns = 2 * 1152, 5120 // 32
    assert metadata["scale_storage"] == ("mxfp4-csf-row-base-offset1-u24-exceptions-1")
    from b12x.moe.fused_moe.trellismx import csf_scale_source

    source = csf_scale_source(
        plane, records.view(torch.uint32), starts, rows=rows, columns=columns
    )
    grid, bases = decode_csf_scale_plane(source, name="w13_scale")
    assert tuple(grid.shape) == (384, rows, columns)
    assert int(grid.min()) > 0 and int(grid.max()) < 248
    # Exceptions are exactly the non-representable positions: no record is
    # redundant and none is missing.
    delta = grid.to(torch.int16) - bases.to(torch.int16).unsqueeze(-1)
    non_representable = int(((delta < 0) | (delta > 1)).sum())
    assert non_representable == int(records.numel())
    again, again_records = encode_csf_scale_plane(grid, bases)
    assert torch.equal(again, plane), "CSF plane bytes are not reproducible"
    assert torch.equal(again_records, records.view(torch.uint32))


def test_real_ds41_sidecar_prepares() -> None:
    if not os.path.exists(_DS41_K2):
        pytest.skip("DS41 K2 test data is not mounted")
    from safetensors import safe_open

    with safe_open(_DS41_K2, framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    prepared = prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=384,
        hidden_size=5120,
        intermediate_size=1152,
        top_k=6,
        device=torch.device("cpu"),
        global_intermediate_size=2304,
    )
    value = prepared._impl.representation.value
    assert tuple(value.w13.shape) == (2, 384, 320, 72, 32)
    assert tuple(value.w13_scale.shape) == (384, 2304, 160)
    assert tuple(value.w13_sfb.shape)[1:] == (9, 40, 32, 8)
    assert value.trellis_bits == 2


# --------------------------------------------------------------------------- #
# GPU: one synthetic layer through plan/bind/run against the reference
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_trellismx_plan_serves_only_its_planned_token_counts() -> None:
    """Boundary and small-M programs are compiled per token count, so binding refuses
    a count the plan did not prepare rather than reach a program it never declared;
    the counts it did prepare run and match the reference."""

    from b12x.moe.fused_moe import ExecutionCapacity, bind, plan_execution, run
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    device = torch.device("cuda", torch.cuda.current_device())
    tensors, metadata = _sidecar()
    prepared = prepare_trellismx_experts(
        {name: value.to(device) for name, value in tensors.items()},
        metadata,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        top_k=_TOP_K,
        device=device,
    )
    torch.manual_seed(4100)

    def bind_rows(execution, rows):
        x = (torch.randn(rows, _H, device=device) * 0.02).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(_E, device=device)[:_TOP_K] for _ in range(rows)]
        ).to(dtype=torch.int32)
        weights = torch.full((rows, _TOP_K), 1.0 / _TOP_K, device=device)
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in execution.scratch_specs()
        )
        binding = bind(
            execution,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            scratch=scratch,
            output=torch.empty_like(x),
        )
        return binding, x, ids, weights

    def serve(execution, rows):
        binding, x, ids, weights = bind_rows(execution, rows)
        result = run(binding=binding).float()
        torch.cuda.synchronize()
        expected = trellismx_moe_reference(
            x.cpu(),
            ids.cpu(),
            weights.cpu(),
            dict(tensors),
            {
                "hidden_size": _H,
                "intermediate_size": _I,
                "num_experts": _E,
                "top_k": _TOP_K,
                "bits": _BITS,
                "world_size": 1,
                "rank": 0,
                "sign_draw": 0,
                "quantize_activations": rows > 16,
            },
        ).to(device=device, dtype=torch.float32)
        unit = float(expected.abs().max()) * 2**-8
        assert torch.isfinite(result).all(), rows
        assert float((result - expected).abs().max()) <= 8 * unit, rows

    single = plan_execution(
        experts=prepared, capacity=ExecutionCapacity(max_tokens=32, top_k=_TOP_K)
    )
    serve(single, 32)
    for rows in (31, 8):
        with pytest.raises(ValueError, match="cannot serve"):
            bind_rows(single, rows)
    warmed = plan_execution(
        experts=prepared,
        capacity=ExecutionCapacity(
            max_tokens=32, top_k=_TOP_K, warmup_token_counts=(8,)
        ),
    )
    serve(warmed, 8)
    serve(warmed, 32)
    with pytest.raises(ValueError, match="cannot serve"):
        bind_rows(warmed, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [8, 32])
def test_synthetic_layer_plan_bind_run_matches_reference(rows: int) -> None:
    """One small synthetic layer through plan/bind/run, eager and graph replay.

    ``rows=8`` takes the small-M arm (16-bit activation and FC2 input),
    ``rows=32`` the grouped arm (E4M3/UE8M0-32 activations); each is compared
    with the reference in its own activation regime.
    """

    from b12x.moe.fused_moe import ExecutionCapacity, bind, plan_execution, run
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    device = torch.device("cuda", torch.cuda.current_device())
    tensors, metadata = _sidecar()
    on_device = {name: value.to(device) for name, value in tensors.items()}
    prepared = prepare_trellismx_experts(
        on_device,
        metadata,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        top_k=_TOP_K,
        device=device,
    )
    torch.manual_seed(4242 + rows)
    x = (torch.randn(rows, _H, device=device) * 0.02).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(_E, device=device)[:_TOP_K] for _ in range(rows)]
    ).to(dtype=torch.int32)
    weights = torch.rand(rows, _TOP_K, device=device, dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    smallm = rows <= 16
    # The reference is a CPU reference (its procedural-NPK decode is a CPU
    # law), so it consumes the host copies of the same released tensors.
    reference = trellismx_moe_reference(
        x.cpu(),
        ids.cpu(),
        weights.cpu(),
        {name: value for name, value in tensors.items()},
        {
            "hidden_size": _H,
            "intermediate_size": _I,
            "num_experts": _E,
            "top_k": _TOP_K,
            "bits": _BITS,
            "world_size": 1,
            "rank": 0,
            "sign_draw": 0,
            "quantize_activations": not smallm,
        },
    )

    execution = plan_execution(
        experts=prepared,
        capacity=ExecutionCapacity(max_tokens=rows, top_k=_TOP_K),
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in execution.scratch_specs()
    )
    output = torch.empty_like(x)
    binding = bind(
        execution,
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        scratch=scratch,
        output=output,
    )
    eager = run(binding=binding).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(binding=binding)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()

    expected = reference.to(device=device, dtype=torch.float32)
    unit = float(expected.abs().max()) * 2**-8
    if smallm:
        assert torch.equal(eager, output), "graph replay changed the result"
    else:
        # Atomic scatter: replay may reorder the FP32 additions.
        assert float((eager.float() - output.float()).abs().max()) <= 4 * unit
    difference = (eager.float() - expected).abs()
    assert float(difference.max()) <= 8 * unit, (float(difference.max()), 8 * unit)
    assert torch.isfinite(eager.float()).all()
    assert torch.count_nonzero(eager) > 0
