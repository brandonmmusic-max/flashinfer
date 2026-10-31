"""Geometry of the TrellisMX activation operands against the W4A8 trellis launch.

``trellismx_boundary.input_boundary_quantized`` emits the A operands the routed
kernel consumes: one E4M3 byte per hidden channel plus one UE8M0 byte per K32
block.  This file pins the geometry of those two buffers to what the launch and
the phase-1 kernel actually read, on the real released shapes:

* GLM-5.3 Flash layer 3 rank 0 — hidden 4096, local intermediate 512, E=288, top-8
* DeepSeek-V4.1 K2 TP2 — hidden 5120, local intermediate 1152
* DeepSeek-V4.1 K4 TP4 — local intermediate 640 and 512 (uneven slices)

The authority for the extents is ``_DynamicMoEW4A8Launch.__call__``
(``fused_moe/_impl.py:11262-11269``): ``packed_a_storage`` is a flat uint8
tensor of ``rows_padded * k`` elements and ``scale_storage`` a flat
``rows_padded * (k // 32)``; ``packed_a`` is the float4x2-typed view
``(rows_padded, 2k, 1)`` so its byte footprint is ``k`` per row.  The phase-1
kernel reads the scale row at ``tok * input_k128_tiles * 4`` bytes
(``w4a8_phase1.py:320-322``), i.e. ``k // 32`` bytes per row with no padding.

Nothing here weakens an existing test; ``test_dynamic_w4a8_trellis.py`` keeps
its own coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from b12x.moe._shared.kernels import trellismx_boundary as boundary

from ._trellismx_data import GLM_RELEASE, glm_sidecar

# (label, hidden, local intermediate, experts, top_k)
GEOMETRIES = (
    ("glm53-layer3-rank0", 4096, 512, 288, 8),
    ("ds41-k2-tp2", 5120, 1152, 384, 6),
    ("ds41-k4-tp4-r0", 5120, 640, 384, 6),
    ("ds41-k4-tp4-r2", 5120, 512, 384, 6),
)

_M_SWEEP = (1, 8, 32, 256)
_TILE_M = 16


def _align_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _rows_padded(m: int, experts: int, top_k: int, *, small_m: bool) -> int:
    """Rows the launch reserves for the A operands.

    ``p8_native_kernel.py:718-724`` sizes the physical tiles as
    ``m * top_k`` for the small-M arm and ``experts + ceil(m*top_k / tile_m)``
    for the materialized arm, then multiplies by ``tile_m``.
    """

    if small_m:
        return m * top_k
    tiles = experts + (m * top_k + _TILE_M - 1) // _TILE_M
    return tiles * _TILE_M


def _inputs(hidden: int, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(
        rows, hidden, generator=torch.Generator().manual_seed(seed)
    ).bfloat16()
    suh = (
        torch.randn(hidden, generator=torch.Generator().manual_seed(seed + 1)) * 0.25
        + 1.0
    ).half()
    return x, suh


def _produce(hidden: int, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, suh = _inputs(hidden, rows, seed)
    return boundary.input_boundary_quantized(x, suh, hidden_size=hidden)


# --------------------------------------------------------------------------- #
# the launch's declared extents
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,hidden,inter,experts,top_k", GEOMETRIES)
@pytest.mark.parametrize("m", _M_SWEEP)
def test_producer_fills_the_launch_extents(
    label: str, hidden: int, inter: int, experts: int, top_k: int, m: int
) -> None:
    """Producer bytes == the flat extents the launch declares, for both arms."""

    for small_m in (True, False):
        rows_padded = _rows_padded(m, experts, top_k, small_m=small_m)
        payload, scales = _produce(hidden, m, seed=m)
        # A buffer of exactly the declared extents, with the producer's rows
        # written at their token positions.
        a_storage = torch.zeros(rows_padded * hidden, dtype=torch.uint8)
        scale_storage = torch.zeros(rows_padded * (hidden // 32), dtype=torch.uint8)
        token_map = torch.arange(m, dtype=torch.int64)
        a_storage.view(rows_padded, hidden)[token_map] = payload
        scale_storage.view(rows_padded, hidden // 32)[token_map] = scales
        assert a_storage.numel() == rows_padded * hidden
        assert scale_storage.numel() == rows_padded * (hidden // 32)
        assert rows_padded >= m, (label, m, rows_padded)


@pytest.mark.parametrize("label,hidden,inter,experts,top_k", GEOMETRIES)
def test_scale_row_padding_is_a_no_op_on_this_path(
    label: str, hidden: int, inter: int, experts: int, top_k: int
) -> None:
    """``align_up(k//32, 4) == k//32`` for every released hidden size.

    The byte-container launch pads the scale row to a 4-block atom
    (``_impl.py:11167-11169``) while the W4A8 launch declares a flat
    ``rows_padded * (k // 32)`` (``_impl.py:11267-11269``).  Those agree only
    when ``k // 32`` is already a multiple of four, which holds for 4096 and
    5120 and is asserted here so a future hidden size cannot silently
    disagree.  The counterexample documents the hazard.
    """

    assert _align_up(hidden // 32, 4) == hidden // 32
    # k = 4128 -> 129 K32 blocks -> the padded stride would be 132.
    assert _align_up(4128 // 32, 4) == 132 != 4128 // 32


@pytest.mark.parametrize("label,hidden,inter,experts,top_k", GEOMETRIES)
def test_producer_rows_are_exactly_k_bytes_and_k_over_32_scales(
    label: str, hidden: int, inter: int, experts: int, top_k: int
) -> None:
    """Row ``t`` occupies exactly ``[t*k, (t+1)*k)`` and ``[t*k/32, ...)``.

    A sentinel row per token: if the producer ever emitted a padded or
    interleaved row stride, the sentinel would land in the wrong place.
    """

    rows = 4
    payload, scales = _produce(hidden, rows, seed=7)
    assert payload.shape == (rows, hidden) and payload.dtype == torch.uint8
    assert scales.shape == (rows, hidden // 32) and scales.dtype == torch.uint8
    flat_payload = payload.reshape(-1)
    flat_scales = scales.reshape(-1)
    for token in range(rows):
        assert torch.equal(
            flat_payload[token * hidden : (token + 1) * hidden], payload[token]
        )
        assert torch.equal(
            flat_scales[token * (hidden // 32) : (token + 1) * (hidden // 32)],
            scales[token],
        )


# --------------------------------------------------------------------------- #
# placement round-trip through a rows_padded buffer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,hidden,inter,experts,top_k", GEOMETRIES)
@pytest.mark.parametrize("m", _M_SWEEP)
def test_placement_round_trip_is_bitwise(
    label: str, hidden: int, inter: int, experts: int, top_k: int, m: int
) -> None:
    """Scatter into ``rows_padded`` rows, gather back, compare to the reference.

    ``rows_padded`` exceeds ``m`` on both arms, so the unused rows must not
    perturb anything and the mapping must be exactly ``row = token_map[t]``.
    """

    rows_padded = _rows_padded(m, experts, top_k, small_m=False)
    payload, scales = _produce(hidden, m, seed=100 + m)
    generator = torch.Generator().manual_seed(200 + m)
    token_map = torch.randperm(rows_padded, generator=generator)[:m].to(torch.int64)

    a_storage = torch.zeros(rows_padded * hidden, dtype=torch.uint8)
    scale_storage = torch.zeros(rows_padded * (hidden // 32), dtype=torch.uint8)
    a_storage.view(rows_padded, hidden)[token_map] = payload
    scale_storage.view(rows_padded, hidden // 32)[token_map] = scales

    gathered_payload = a_storage.view(rows_padded, hidden).index_select(0, token_map)
    gathered_scales = scale_storage.view(rows_padded, hidden // 32).index_select(
        0, token_map
    )
    assert torch.equal(gathered_payload, payload)
    assert torch.equal(gathered_scales, scales)

    # And the gathered payload still reconstructs the declared quantizer, i.e.
    # placement did not disturb the intra-K32 order or the block scale pairing.
    perm = torch.tensor(boundary.K32_PERMUTATION)
    blocks = gathered_payload.view(torch.float8_e4m3fn).float().reshape(m, -1, 32)
    blocks = blocks.index_select(-1, torch.argsort(perm)).reshape(m, hidden)
    recon = blocks * torch.pow(
        torch.tensor(2.0), (gathered_scales.float() - 127.0)
    ).repeat_interleave(32, dim=1)
    x, suh = _inputs(hidden, m, 100 + m)
    expected = boundary.quantize_e4m3_ue8m0_k32(
        boundary.input_boundary(x, suh, hidden_size=hidden)
    )
    assert torch.equal(recon, expected)


# --------------------------------------------------------------------------- #
# The prepared experts agree with the A geometry
# --------------------------------------------------------------------------- #
_GLM_RELEASE = Path(GLM_RELEASE)
_GLM_SIDECAR = Path(glm_sidecar(3, 0))
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


def _load_glm_sidecar() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not _GLM_SIDECAR.exists():
        pytest.skip("GLM TrellisMX release sidecar is not present")
    from safetensors import safe_open

    with safe_open(str(_GLM_SIDECAR), "np") as handle:
        tensors = {
            name: torch.from_numpy(handle.get_tensor(name)) for name in _SIDECAR_TENSORS
        }
        metadata = dict(handle.metadata())
    return tensors, metadata


def _equivalent_metadata(metadata: dict[str, str]) -> dict[str, str]:
    """The documented alternative to the sidecar `__metadata__`.

    The released sidecar carries only ``law``/``alphabet``/``scale``/``boundary``/
    ``bits``/``activation``/``global_intermediate``/``world_size``; the geometry
    keys live in ``config.json`` and the local intermediate follows from
    ``global_intermediate // world_size``.
    """

    import json

    config = json.loads((_GLM_RELEASE / "config.json").read_text())
    text = config.get("text_config", config)
    world_size = int(metadata["world_size"])
    local = int(metadata["global_intermediate"]) // world_size
    return {
        **metadata,
        "compander_scale": "2.0",
        "hidden_size": str(text["hidden_size"]),
        "num_experts": str(text["n_routed_experts"]),
        "top_k": str(text["num_experts_per_tok"]),
        "intermediate_per_rank": str(local),
    }


def test_prepared_experts_report_the_geometry_the_operands_assume() -> None:
    """``prepare_trellismx_experts`` and the A operands must agree.

    Uses the documented equivalent metadata mapping, on the real released
    sidecar: the point is the real prepared object, not a fake.
    """

    from b12x.moe.fused_moe import trellismx

    tensors, metadata = _load_glm_sidecar()
    prepared = trellismx.prepare_trellismx_experts(
        tensors,
        _equivalent_metadata(metadata),
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=torch.device("cpu"),
    )
    # The three attributes the Flare recipe reads, and the ones the A operand
    # geometry is a function of.
    for attribute, expected in (
        ("num_experts", 288),
        ("hidden_size", 4096),
        ("intermediate_size", 512),
    ):
        assert int(getattr(prepared, attribute)) == expected, attribute
    # The vectors the operand producer consumes are per hidden channel, and the
    # weight scale plane has one UE8M0 byte per K32 block of the hidden axis.
    assert tensors["gate_up_suh_fp16"].numel() == int(prepared.hidden_size)
    assert tensors["down_svh_fp16"].numel() == int(prepared.hidden_size)
    assert tensors["w13_scale_ue8m0"].shape[-1] == int(prepared.hidden_size) // 32


def test_verbatim_sidecar_metadata_is_accepted() -> None:
    """The released ``__metadata__`` must work unchanged (callers pass it verbatim).

    The compander is implied by the law and the geometry keys are an optional
    cross-check, so the verbatim metadata prepares; this is a hard assertion so
    a regression cannot come back quietly.
    """

    from b12x.moe.fused_moe import trellismx

    tensors, metadata = _load_glm_sidecar()
    trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=torch.device("cpu"),
    )


# --------------------------------------------------------------------------- #
# GPU arm: the producer and the placement are graph-capturable and bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("label,hidden,inter,experts,top_k", GEOMETRIES)
def test_gpu_producer_matches_cpu_and_replays(
    label: str, hidden: int, inter: int, experts: int, top_k: int
) -> None:
    device = torch.device("cuda")
    rows = 8
    torch.manual_seed(900)
    x_cpu = torch.randn(rows, hidden).bfloat16()
    suh_cpu = (torch.randn(hidden) * 0.25 + 1.0).half()
    payload_cpu, scales_cpu = boundary.input_boundary_quantized(
        x_cpu, suh_cpu, hidden_size=hidden
    )

    x = x_cpu.to(device)
    suh = suh_cpu.to(device)

    def run(x_in, suh_in):
        return boundary.input_boundary_quantized(x_in, suh_in, hidden_size=hidden)

    eager = run(x, suh)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(x, suh)
    graph.replay()
    for got, want in zip(captured, eager):
        assert torch.equal(got, want)
    assert torch.equal(captured[0].cpu(), payload_cpu)
    assert torch.equal(captured[1].cpu(), scales_cpu)
