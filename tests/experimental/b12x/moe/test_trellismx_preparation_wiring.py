"""The fused-MoE preparation path must hand TrellisMX's options to the kernel.

``plan.trellis_codebook`` is validated against the source format when the plan is
built, but until this package nothing passed it to a kernel constructor: a
``trellismx`` plan compiled a kernel whose ``trellis_codebook`` stayed at the
``lut_e4m3`` default, so a procedural-NPK payload was decoded through the LUT
table.  That is a silent wrong answer, not a crash,
and ``test_trellismx_e4m3_ties.py`` shows the coupled-boundary flags change real
E4M3 codes, so they matter too.

This file is CPU-only except for the last test, which runs real forwards on a
GPU against ``trellismx_moe_reference``.  The wiring is checked two ways: on the
real released sidecar (the plan really does carry ``procedural_npk``), and at the
call site with a spy on the kernel constructor (so the assertion is about what
``_preparation.py`` passes, independent of what the constructor currently
accepts).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe import _impl, _preparation, trellismx
from b12x.moe._shared.trellis_codebooks import LUT_E4M3, PROCEDURAL_NPK

from ._trellismx_data import ds41_sidecar, glm_sidecar

_GLM_SIDECAR = Path(glm_sidecar(3, 0))
_DS41_K4 = Path(ds41_sidecar(4, 0))
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
_COUPLED_FLAGS = ("trellis_exact_hadamard", "trellis_fp32_boundary")


def _load(path: Path, names) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not path.exists():
        pytest.skip(f"{path} is not present")
    from safetensors import safe_open

    with safe_open(str(path), "np") as handle:
        tensors = {name: torch.from_numpy(handle.get_tensor(name)) for name in names}
        metadata = dict(handle.metadata())
    return tensors, metadata


def _glm_prepared():
    tensors, metadata = _load(_GLM_SIDECAR, _GLM_TENSORS)
    return trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=288,
        hidden_size=4096,
        intermediate_size=512,
        top_k=8,
        device=torch.device("cpu"),
    )


# --------------------------------------------------------------------------- #
# the plan really does carry procedural_npk
# --------------------------------------------------------------------------- #
def test_glm_plan_declares_procedural_npk() -> None:
    prepared = _glm_prepared()
    plan = prepared.plan._impl
    assert plan.source_format == "trellismx"
    assert plan.trellis_codebook == PROCEDURAL_NPK
    assert int(plan.trellis_bits) == 4
    # The 6I rotation extent is what selects the coupled intermediate boundary.
    assert bool(plan.intermediate_hadamard) is True
    assert (
        int(plan.num_experts),
        int(plan.hidden_size),
        int(plan.intermediate_size),
    ) == (
        288,
        4096,
        512,
    )


def test_ds41_k4_plan_declares_procedural_npk() -> None:
    if not _DS41_K4.exists():
        pytest.skip("DS41 K4 TrellisMX test sidecar is not present")
    from safetensors import safe_open

    with safe_open(str(_DS41_K4), "np") as handle:
        names = tuple(handle.keys())
        tensors = {name: torch.from_numpy(handle.get_tensor(name)) for name in names}
        metadata = dict(handle.metadata())
    # The K4 TP4 layer-0 rank-0 slice is 640 channels (uneven 640/640/512/512).
    prepared = trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=384,
        hidden_size=5120,
        intermediate_size=640,
        top_k=6,
        device=torch.device("cpu"),
    )
    plan = prepared.plan._impl
    assert plan.source_format == "trellismx"
    assert plan.trellis_codebook == PROCEDURAL_NPK
    assert (int(plan.hidden_size), int(plan.intermediate_size)) == (5120, 640)


# --------------------------------------------------------------------------- #
# the option derivation
# --------------------------------------------------------------------------- #
def _stub_plan(**overrides):
    values = dict(
        implementation="dynamic",
        source_format="trellismx",
        trellis_codebook=PROCEDURAL_NPK,
        weight_E=4,
        routed_rows=64,
        num_topk=8,
        k=512,
        n=128,
        max_rows=64,
        activation="silu",
        quant_mode="w4a8_mx",
        deterministic_output=False,
        swiglu_limit=10.0,
        swiglu_clamp_after_silu=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        device=torch.device("cpu"),
        max_tokens_per_launch=8,
        decode_config=SimpleNamespace(nvfp4_materialize_intermediate=False),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_options_for_trellismx() -> None:
    codebook, exact, fp32 = _preparation._trellis_kernel_options(_stub_plan())
    assert codebook == PROCEDURAL_NPK
    assert exact is True and fp32 is True


def test_options_for_other_source_formats_are_untouched() -> None:
    plan = _stub_plan(source_format="b12x_trellis", trellis_codebook=LUT_E4M3)
    assert _preparation._trellis_kernel_options(plan) == (LUT_E4M3, False, False)
    # No codebook declared, not a trellismx plan: unchanged, no invention.
    plan = _stub_plan(source_format="modelopt_nvfp4", trellis_codebook=None)
    assert _preparation._trellis_kernel_options(plan) == (None, False, False)


def test_options_reject_a_conflicting_explicit_codebook() -> None:
    plan = _stub_plan(trellis_codebook=LUT_E4M3)
    with pytest.raises(ValueError, match="trellismx requires trellis_codebook"):
        _preparation._trellis_kernel_options(plan)


# --------------------------------------------------------------------------- #
# the call sites
# --------------------------------------------------------------------------- #
def _stub_caps():
    return SimpleNamespace(
        w4a16_fast_math=False,
        split_fc1_scales=False,
        output_layout="token",
        weight_plan=SimpleNamespace(trellis_bits=4, intermediate_hadamard=True),
    )


def _stub_scratch(plan):
    return SimpleNamespace(
        launch_plan=plan,
        full_rotation=False,
        _prewarmed_fused_launches=(),
        _prewarmed_topk_sum_launches=(),
        _mixed_trellis_launches=(),
        _prewarmed_route_pack_launches=None,
    )


def _patched_dynamic_arguments(monkeypatch):
    monkeypatch.setattr(
        _preparation,
        "_dynamic_program_arguments",
        lambda plan, caps: dict(
            n=plan.n,
            w4a8_repacked=True,
            w4a8_n64_repacked=False,
            csf_inline_scales=False,
            direct_routing=False,
            external_route_plan=False,
            share_input_across_experts=False,
            planned_tile_m=16,
            mma_warps=4,
            materialized_fc1_cta_budget=2,
            raw_routes=False,
        ),
    )


def _spy_with_full_signature(monkeypatch, record):
    """A stand-in constructor that accepts the coupled flags and records them."""

    def spy(
        E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        *,
        topk_ids_dtype,
        fast_math,
        mma_warps=None,
        materialized_fc1_cta_budget=2,
        activation="silu",
        quant_mode="nvfp4",
        w4a8_repacked=False,
        w4a8_n64_repacked=False,
        nvfp4_materialize_intermediate=False,
        csf_inline_scales=False,
        separate_w13_halves=None,
        direct_routing=False,
        external_route_plan=False,
        share_input_across_experts=False,
        deterministic_output=False,
        raw_routes=False,
        split_fc1_scales=False,
        swiglu_limit=None,
        swiglu_clamp_after_silu=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        trellis_bits=0,
        trellis_intermediate_hadamard=False,
        trellis_codebook=LUT_E4M3,
        trellis_exact_hadamard=False,
        trellis_fp32_boundary=False,
        trellis_pre_quantized_a=False,
        planned_tile_m=None,
        planned_num_tokens=None,
    ):
        record.append(
            dict(
                topk_ids_dtype=topk_ids_dtype,
                trellis_codebook=trellis_codebook,
                trellis_exact_hadamard=trellis_exact_hadamard,
                trellis_fp32_boundary=trellis_fp32_boundary,
                trellis_bits=trellis_bits,
                trellis_intermediate_hadamard=trellis_intermediate_hadamard,
                trellis_pre_quantized_a=trellis_pre_quantized_a,
            )
        )
        return object(), None

    monkeypatch.setattr(_impl, "_get_dynamic_kernel", spy)


def test_dynamic_site_passes_codebook_and_coupled_flags(monkeypatch) -> None:
    record: list[dict] = []
    _patched_dynamic_arguments(monkeypatch)
    _spy_with_full_signature(monkeypatch, record)
    plan = _stub_plan()
    _preparation._program_carriers(
        _stub_scratch(plan),
        _stub_caps(),
        None,
        input_scale_count=1,
        intermediate_scale_count=1,
    )
    assert record, "the dynamic constructor was never called"
    for call in record:
        assert call["trellis_codebook"] == PROCEDURAL_NPK
        assert call["trellis_exact_hadamard"] is True
        assert call["trellis_fp32_boundary"] is True
        # The plan's own rate and coupled intermediate boundary still arrive.
        assert call["trellis_bits"] == 4
        # Planning compiles the same pre-quantized-activation kernel the prepared payload launches.
        assert call["trellis_pre_quantized_a"] is True
        assert _impl._TRELLISMX_PRE_QUANTIZED_A is True
        assert call["trellis_intermediate_hadamard"] is True


def test_dynamic_site_passes_nothing_extra_for_other_formats(monkeypatch) -> None:
    record: list[dict] = []
    _patched_dynamic_arguments(monkeypatch)
    _spy_with_full_signature(monkeypatch, record)
    plan = _stub_plan(source_format="b12x_trellis", trellis_codebook=LUT_E4M3)
    _preparation._program_carriers(
        _stub_scratch(plan),
        _stub_caps(),
        None,
        input_scale_count=1,
        intermediate_scale_count=1,
    )
    assert record
    for call in record:
        assert call["trellis_codebook"] == LUT_E4M3
        assert call["trellis_exact_hadamard"] is False
        assert call["trellis_fp32_boundary"] is False


def test_micro_site_refuses_a_trellismx_plan(monkeypatch) -> None:
    # The micro backend has no procedural-NPK arm and no coupled boundary, so it
    # must fail before compiling rather than run the default decode.
    plan = _stub_plan(implementation="micro")
    with pytest.raises(ValueError, match="micro backend cannot serve a trellismx plan"):
        _preparation._program_carriers(
            _stub_scratch(plan),
            _stub_caps(),
            None,
            input_scale_count=1,
            intermediate_scale_count=1,
        )


def test_dynamic_site_against_the_real_constructor() -> None:
    """The real constructor takes the coupled flags with off-by-default values."""

    parameters = inspect.signature(_impl._get_dynamic_kernel).parameters
    for name in _COUPLED_FLAGS:
        assert parameters[name].default is False
    assert parameters["trellis_codebook"].default == LUT_E4M3
    assert parameters["trellis_pre_quantized_a"].default is False


# --------------------------------------------------------------------------- #
# discovery runs where preparation's compiler workers run
# --------------------------------------------------------------------------- #

# Runs in a fresh interpreter because the offline compile target replaces the
# process-global CUDA queries exactly as a compiler worker's initializer does.
# argv: device ordinal, device UUID, SM count of the described target.
_OFFLINE_DISCOVERY = r"""
import json, multiprocessing, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from b12x._lib import compile_pool
ordinal, uuid, sm_count = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
compile_pool._initialize_worker(
    ordinal, (12, 0), uuid, "synthetic SM120", sm_count, 101376, 102400,
    multiprocessing.get_context("spawn").Array("q", (0, 0)),
)
from b12x._lib.compile_pool import CompileJob, describe_compilation
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe import trellismx
from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._discovery import discovery_key
from b12x.preparation.device import DetectedDevice

E, H, I, BITS, TOP_K = 4, 512, 128, 4, 2
generator = torch.Generator().manual_seed(5)
words = (2, E, H // 16, I // 16, 16 * BITS)
tensors = {
    "w13_trellis": torch.randint(0, 1 << 15, words, generator=generator, dtype=torch.int16),
    "w2_trellis": torch.randint(
        0, 1 << 15, (E, I // 16, H // 16, 16 * BITS), generator=generator, dtype=torch.int16
    ),
    "w13_scale_ue8m0": torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8),
    "w2_scale_ue8m0": torch.full((E, H, I // 32), 127, dtype=torch.uint8),
    "intermediate_scales_fp16": torch.ones(E, 3 * I, dtype=torch.float16),
    "coupled_sign_draw_u8": torch.zeros(E, dtype=torch.uint8),
    "gate_up_suh_fp16": torch.ones(H, dtype=torch.float16),
    "down_svh_fp16": torch.ones(H, dtype=torch.float16),
}
metadata = {
    "law": "procedural-npk", "compander_scale": "2.0", "alphabet": "e4m3",
    "scale": "ue8m0-k32", "boundary": "coupled-h512-h128-suh-svh-v1",
    "activation": "silu-cap10", "bits": str(BITS), "hidden": str(H),
    "experts_start": "0", "experts_end": str(E), "intermediate_per_rank": str(I),
    "topk": str(TOP_K), "world_size": "1", "slice_begin": "0", "slice_end": str(I),
}
prepared = trellismx.prepare_trellismx_experts(
    tensors, metadata, num_experts=E, hidden_size=H, intermediate_size=I, top_k=TOP_K,
    device=torch.device("cpu"),
)
identity = DeviceIdentity("nvidia", (12, 0), sm_count, "synthetic SM120")
device = DetectedDevice(
    ordinal=ordinal, identity=identity, uuid=uuid,
    max_shared_memory_per_block=101376, max_shared_memory_per_multiprocessor=102400,
)
report = {}
for m in (8, 32):
    plan = moe.plan_execution(
        experts=prepared, capacity=moe.ExecutionCapacity(max_tokens=m, top_k=TOP_K)
    )
    jobs = plan._compile_jobs(TUNING.default_config(plan.query, identity), device)
    programs = {
        (program.dialect, program.key, program.name)
        for job in jobs
        for program in describe_compilation(job).programs
    }
    report[str(m)] = {
        "names": sorted({name for _, _, name in programs}),
        "keys": sorted(programs),
        # Placement is bound when a job runs, never stored in its metadata (a
        # stored ordinal would also show up as a per-device discovery key).
        "placement": sorted(
            {
                "sentinel" if value is CompileJob.DEVICE_ORDINAL else repr(value)
                for job in jobs
                for value in (*job.args, *dict(job.kwargs).values())
                if value is CompileJob.DEVICE_ORDINAL or isinstance(value, torch.device)
            }
        ),
        "discovery": discovery_key(
            [(FrozenMapping({}), FrozenMapping({"tokens": m}), tuple(jobs))], device
        ),
    }
report["cuda_initialized"] = torch.cuda.is_initialized()
print(json.dumps(report))
"""


def _offline_discovery(ordinal: int, uuid: str, sm_count: int):
    import os
    import subprocess
    import sys

    return subprocess.Popen(
        [sys.executable, "-c", _OFFLINE_DISCOVERY, str(ordinal), uuid, str(sm_count)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # The discovery key hashes pickled jobs, and a fixed contract's encoded
        # query (the route job's) is built in its field set's iteration order,
        # which string hashing varies per process. One seed for every worker
        # keeps this a test of device independence.
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0"),
        cwd=Path(__file__).resolve().parents[4],
    )


def test_discovery_runs_in_offline_compile_workers() -> None:
    """TrellisMX discovery needs no CUDA device, only the worker's target description.

    Preparation's compiler workers hide CUDA and answer device queries from the
    target the parent describes; the small-M and fused-boundary factories ask
    only those queries, so describing m = 8 (small-M arm) and m = 32 (grouped
    arm with the fused input boundary and output stage) never initializes CUDA.

    Discovery results are cached across ranks by target geometry, not by device
    (``preparation._discovery``): the jobs carry ``CompileJob.DEVICE_ORDINAL``
    instead of an ordinal, and two devices of the same geometry (ordinals 0 and
    7, different UUIDs) get the same program keys and discovery key, while a
    different SM count changes the discovery key and every CuTe program key.
    """

    pytest.importorskip("cutlass")
    import json

    targets = ((0, "GPU-first", 188), (7, "GPU-second", 188), (0, "GPU-first", 170))
    processes = [_offline_discovery(*target) for target in targets]
    reports = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=900)
        assert process.returncode == 0, stderr[-4000:]
        reports.append(json.loads(stdout.strip().splitlines()[-1]))
    first, second, other = reports
    for report in reports:
        assert report["cuda_initialized"] is False
        for m in ("8", "32"):
            assert report[m]["placement"] == ["sentinel"]
            assert report[m]["discovery"] is not None
    assert {
        "moe.trellismx.smallm.input_boundary_fp16",
        "moe.w4a16.fused_moe",
        "moe.w4a16.topk_sum",
    } <= set(first["8"]["names"])
    fused = "b12x.moe._shared.kernels.trellismx_boundary_fused."
    assert {
        f"{fused}TrellisMxFusedBoundary",
        f"{fused}TrellisMxFusedOutputStage",
    } <= set(first["32"]["names"])
    assert not any("smallm" in name for name in first["32"]["names"])
    for m in ("8", "32"):
        assert first[m]["keys"] == second[m]["keys"]
        assert first[m]["discovery"] == second[m]["discovery"]
        assert first[m]["discovery"] != other[m]["discovery"]
        # Every CuTe program is specialized to the hardware geometry (the shared
        # Triton route kernels are not).
        cute = {tuple(key) for key in first[m]["keys"] if key[0] == "cute"}
        assert cute and not cute & {tuple(key) for key in other[m]["keys"]}


def test_prepared_payload_selects_pre_quantized_a() -> None:
    """The prepared record and program planning agree on the pre-quantized-A mode.

    The reviewed head derived both from ``inspect.signature`` of the activation
    wrapper; the mode is now one constant, so the two cannot diverge.
    """

    tensors, metadata = _synthetic_sidecar()
    prepared = trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=4,
        hidden_size=512,
        intermediate_size=128,
        top_k=2,
        device=torch.device("cpu"),
    )
    value = prepared._impl.representation.value
    assert value.trellis_pre_quantized_a is _impl._TRELLISMX_PRE_QUANTIZED_A is True


def _synthetic_sidecar():
    experts, hidden, local, bits = 4, 512, 128, 4
    generator = torch.Generator().manual_seed(5)
    tensors = {
        "w13_trellis": torch.randint(
            0,
            1 << 15,
            (2, experts, hidden // 16, local // 16, 16 * bits),
            generator=generator,
            dtype=torch.int16,
        ),
        "w2_trellis": torch.randint(
            0,
            1 << 15,
            (experts, local // 16, hidden // 16, 16 * bits),
            generator=generator,
            dtype=torch.int16,
        ),
        "w13_scale_ue8m0": torch.full(
            (experts, 2 * local, hidden // 32), 127, dtype=torch.uint8
        ),
        "w2_scale_ue8m0": torch.full(
            (experts, hidden, local // 32), 127, dtype=torch.uint8
        ),
        "intermediate_scales_fp16": torch.ones(experts, 3 * local, dtype=torch.float16),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=torch.uint8),
        "gate_up_suh_fp16": torch.ones(hidden, dtype=torch.float16),
        "down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
    }
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
        "intermediate_per_rank": str(local),
        "topk": "2",
        "world_size": "1",
        "slice_begin": "0",
        "slice_end": str(local),
    }
    return tensors, metadata


# --------------------------------------------------------------------------- #
# the small-M arm is the plan's declaration, not a process setting
# --------------------------------------------------------------------------- #
def _trellismx_bundle():
    tensors, metadata = _synthetic_sidecar()
    return trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=4,
        hidden_size=512,
        intermediate_size=128,
        top_k=2,
        device=torch.device("cpu"),
    )


def _plan_with(prepared, **kwargs):
    from b12x.moe import fused_moe as moe

    return moe.plan_execution(
        experts=prepared,
        capacity=moe.ExecutionCapacity(max_tokens=8, top_k=2),
        **kwargs,
    )


def test_smallm_limit_is_the_plans_declaration() -> None:
    """``smallm_max_tokens`` selects the arm; the plan and its tuning query carry it.

    The declaration, not a process-wide default, decides how much of the arm's
    1..16 token range a plan serves and which programs discovery compiles.
    """

    from b12x.moe._shared.kernels.trellismx_smallm import SMALLM_MAX_TOKENS
    from b12x.moe.fused_moe._tuning import TUNING

    prepared = _trellismx_bundle()
    default = _plan_with(prepared)
    assert default.query.controls["trellismx_smallm"] is True
    assert default.query.controls["trellismx_smallm_limit"] == SMALLM_MAX_TOKENS == 16
    assert _preparation._query_smallm_tokens(default.query) == 16

    narrowed = _plan_with(prepared, smallm_max_tokens=8)
    assert narrowed.query.controls["trellismx_smallm"] is True
    assert narrowed.query.controls["trellismx_smallm_limit"] == 8
    assert _preparation._query_smallm_tokens(narrowed.query) == 8

    grouped = _plan_with(prepared, smallm_max_tokens=0)
    assert grouped.query.controls["trellismx_smallm"] is False
    assert grouped.query.controls["trellismx_smallm_limit"] == 0
    assert _preparation._query_smallm_tokens(grouped.query) == 0

    # The tuning query is the encoded plan query, so each declaration is its own
    # tuning identity and the arm's programs are compiled and raced per value.
    encoded = [TUNING.encode_query(plan.query) for plan in (default, narrowed, grouped)]
    assert encoded[0] != encoded[1]
    assert encoded[1] != encoded[2]
    assert encoded[0] != encoded[2]


@pytest.mark.parametrize(
    "value,error",
    [
        (17, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (8.0, TypeError),
        ("16", TypeError),
    ],
)
def test_smallm_limit_is_validated(value, error) -> None:
    """The declaration names programs the arm has: 0..16 tokens, as an integer."""

    with pytest.raises(error, match="smallm_max_tokens"):
        _plan_with(_trellismx_bundle(), smallm_max_tokens=value)


def test_the_arm_is_only_built_for_token_counts_it_serves() -> None:
    """Preparation builds the arm for the counts it serves, and for no others."""

    from b12x.moe import fused_moe as moe

    prepared = _trellismx_bundle()
    above = moe.plan_execution(
        experts=prepared, capacity=moe.ExecutionCapacity(max_tokens=17, top_k=2)
    )
    assert _preparation._query_smallm_tokens(above.query) == 16
    assert _preparation._trellismx_smallm_arm(prepared._impl, above.query) is None

    disabled = _plan_with(prepared, smallm_max_tokens=0)
    assert _preparation._trellismx_smallm_arm(prepared._impl, disabled.query) is None
    # A count the arm serves is built (on the device); the GPU test below covers it.


@pytest.mark.parametrize(
    "change,error",
    [
        ({"trellismx_smallm": True, "trellismx_smallm_limit": 17}, ValueError),
        ({"trellismx_smallm": 1, "trellismx_smallm_limit": 16}, TypeError),
        ("drop_limit", ValueError),
        ("drop_choice", ValueError),
    ],
)
def test_the_tuning_query_rejects_a_smallm_pair_the_arm_cannot_serve(
    change, error
) -> None:
    """The resolved pair is part of the query identity, so it is validated there too."""

    import dataclasses

    from b12x.moe.fused_moe._tuning import TUNING
    from b12x.preparation import FrozenMapping

    query = _plan_with(_trellismx_bundle()).query
    controls = query.controls.to_dict()
    if change == "drop_limit":
        del controls["trellismx_smallm_limit"]
    elif change == "drop_choice":
        del controls["trellismx_smallm"]
    else:
        controls.update(change)
    broken = dataclasses.replace(query, controls=FrozenMapping(controls))
    with pytest.raises(error, match="small-M"):
        TUNING.validate_query(broken, None)


# Runs in a fresh interpreter so the environment is read (or not) at import,
# exactly as a serving process would read it. CUDA stays hidden: planning a
# TrellisMX declaration from a CPU-prepared bundle touches no device.
_SMALLM_ENV_PROBE = r"""
import json
import torch
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe import trellismx

E, H, I, BITS, TOP_K = 4, 512, 128, 4, 2
generator = torch.Generator().manual_seed(5)
tensors = {
    "w13_trellis": torch.randint(
        0, 1 << 15, (2, E, H // 16, I // 16, 16 * BITS), generator=generator,
        dtype=torch.int16,
    ),
    "w2_trellis": torch.randint(
        0, 1 << 15, (E, I // 16, H // 16, 16 * BITS), generator=generator,
        dtype=torch.int16,
    ),
    "w13_scale_ue8m0": torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8),
    "w2_scale_ue8m0": torch.full((E, H, I // 32), 127, dtype=torch.uint8),
    "intermediate_scales_fp16": torch.ones(E, 3 * I, dtype=torch.float16),
    "coupled_sign_draw_u8": torch.zeros(E, dtype=torch.uint8),
    "gate_up_suh_fp16": torch.ones(H, dtype=torch.float16),
    "down_svh_fp16": torch.ones(H, dtype=torch.float16),
}
metadata = {
    "law": "procedural-npk", "compander_scale": "2.0", "alphabet": "e4m3",
    "scale": "ue8m0-k32", "boundary": "coupled-h512-h128-suh-svh-v1",
    "activation": "silu-cap10", "bits": str(BITS), "hidden": str(H),
    "experts_start": "0", "experts_end": str(E), "intermediate_per_rank": str(I),
    "topk": str(TOP_K), "world_size": "1", "slice_begin": "0", "slice_end": str(I),
}
prepared = trellismx.prepare_trellismx_experts(
    tensors, metadata, num_experts=E, hidden_size=H, intermediate_size=I,
    top_k=TOP_K, device=torch.device("cpu"),
)
plan = moe.plan_execution(
    experts=prepared, capacity=moe.ExecutionCapacity(max_tokens=8, top_k=TOP_K)
)
print(json.dumps({
    "trellismx_smallm": plan.query.controls["trellismx_smallm"],
    "trellismx_smallm_limit": plan.query.controls["trellismx_smallm_limit"],
}))
"""


def test_no_environment_variable_selects_the_smallm_arm() -> None:
    """The arm choice is the plan's ``smallm_max_tokens``, never the environment.

    A fresh process that sets the retired ``B12X_TRELLISMX_SMALLM`` and
    ``B12X_TRELLISMX_SMALLM_LIMIT`` still plans the declared default, so no
    environment value can change which arm serves m <= 16 or which programs
    discovery compiles for it.
    """

    import json
    import os
    import subprocess
    import sys

    process = subprocess.run(
        [sys.executable, "-c", _SMALLM_ENV_PROBE],
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            B12X_TRELLISMX_SMALLM="0",
            B12X_TRELLISMX_SMALLM_LIMIT="1",
        ),
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path(__file__).resolve().parents[4],
    )
    assert process.returncode == 0, process.stderr[-4000:]
    controls = json.loads(process.stdout.strip().splitlines()[-1])
    assert controls == {"trellismx_smallm": True, "trellismx_smallm_limit": 16}


# --------------------------------------------------------------------------- #
# GPU rows: real forwards through plan/bind/run against the reference
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layer,rank", [(3, 0), (3, 3), (44, 1), (45, 2)])
@pytest.mark.parametrize("tokens", [1, 8, 16, 17, 32, 129])
def test_trellismx_forward_matches_the_reference(
    layer: int, rank: int, tokens: int
) -> None:
    """GLM forwards through each arm and dispatch boundary, against its own reference.

    Covers routed layers 3 and 44 and the MTP layer 45 on three ranks (different
    slices of the intermediate axis and of the coupled signs). m <= 16 takes the
    small-M arm, which keeps the activation and the FC2 input 16-bit, so its
    reference runs without activation quantization; m > 16 takes the grouped
    arm (M32 tiles to 128 rows, M64 above), which quantizes both to
    E4M3/UE8M0-32 like the reference's ``quantize_activations``. The bounds come
    from the recorded session runs (small-M 0.9999990, grouped 0.99990-0.99991,
    measured earlier) with margin; eager and graph replay must agree
    bit for bit at m <= 16 (the arm's reduction is deterministic) and to 4 BF16
    units above it (atomic scatter).
    """

    from b12x.moe import fused_moe
    from b12x.testing.reference import trellismx_moe as reference

    sidecar = Path(glm_sidecar(layer, rank))
    tensors, metadata = _load(sidecar, _GLM_TENSORS)
    # An indexed device: trellis.py:94 compares `tensor.device != device`, and a
    # bare torch.device("cuda") never equals torch.device("cuda:0").
    device = torch.device("cuda", torch.cuda.current_device())
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
    assert prepared._impl.representation.value.smallm_exact_scales

    generator = torch.Generator(device="cpu").manual_seed(
        808 + 1000 * layer + 10 * rank + tokens
    )
    x = torch.randn(tokens, 4096, generator=generator).bfloat16().to(device)
    ids = torch.stack(
        [torch.randperm(288, generator=generator)[:8] for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    weights = torch.rand(tokens, 8, generator=generator).to(device)
    weights = weights / weights.sum(-1, keepdim=True)

    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=tokens, top_k=8, route_num_experts=0
        ),
    )
    specs = plan.scratch_specs()
    assert len(specs) == 1
    scratch = torch.empty(int(specs[0].nbytes), dtype=torch.uint8, device=device)
    output = torch.empty(tokens, 4096, dtype=torch.bfloat16, device=device)
    binding = fused_moe.bind(
        plan, scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=output
    )
    eager = fused_moe.run(binding=binding).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_moe.run(binding=binding)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()

    smallm = tokens <= 16
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
            rank=rank,
            world_size=4,
            sign_draw=0,
            quantize_activations=not smallm,
            expert_subset=sorted({int(v) for v in ids.reshape(-1).cpu()}),
        ),
    ).to(device)

    got = eager.double()
    expected = want.double()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), expected.reshape(1, -1)
    ).item()
    assert torch.isfinite(eager.float()).all()
    assert cosine >= (0.99995 if smallm else 0.9998), (tokens, cosine)
    if smallm:
        assert torch.equal(eager, output), "small-M graph replay changed the result"
    else:
        unit = float(expected.abs().max()) * 2**-8
        assert float((eager.double() - output.double()).abs().max()) <= 4 * unit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_plan_bind_and_capture_need_no_eager_call() -> None:
    """Preparation builds the small-M arm, so capture neither compiles nor allocates.

    The arm's plan and buffers used to be built on the first eager call, so a
    caller that planned, bound and captured without one failed at capture. The
    capture here is the first use of the binding, and it runs with kernel
    resolution frozen, where any lazily resolved program raises. It also must
    not allocate: preparation owns the immutable plan and programs, and binding
    carves the mutable arm storage out of the caller's scratch.
    """

    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe
    from b12x.testing.reference import trellismx_moe as reference

    tensors, metadata = _load(Path(glm_sidecar(3, 0)), _GLM_TENSORS)
    device = torch.device("cuda", torch.cuda.current_device())
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
    generator = torch.Generator(device="cpu").manual_seed(20261030)
    x = torch.randn(16, 4096, generator=generator).bfloat16().to(device)
    ids = torch.stack(
        [torch.randperm(288, generator=generator)[:8] for _ in range(16)]
    ).to(device=device, dtype=torch.int32)
    weights = torch.rand(16, 8, generator=generator).to(device)
    weights = weights / weights.sum(-1, keepdim=True)

    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=16, top_k=8, route_num_experts=0
        ),
    )
    specs = plan.scratch_specs()
    scratch = torch.empty(int(specs[0].nbytes), dtype=torch.uint8, device=device)
    output = torch.empty(16, 4096, dtype=torch.bfloat16, device=device)
    # bind materializes the plan; nothing has run yet.
    binding = fused_moe.bind(
        plan, scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=output
    )
    state = plan.prepared.state
    assert state.trellismx_smallm_tokens == 16
    arm = state.trellismx_smallm_plan
    assert arm is not None, "preparation must build the small-M arm"
    assert binding.trellismx_smallm_plan is arm
    assert prepared._impl._trellismx_smallm_plans == {(8, 16): arm}

    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("test: TrellisMX small-M capture without a run"):
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, (
        "capture allocated device memory; binding carves the arm's buffers from "
        "the caller's scratch"
    )

    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    replayed = output.clone()
    # The first eager call, now that the graph exists, is the bitwise reference.
    fused_moe.run(binding=binding)
    torch.cuda.synchronize()
    assert torch.equal(replayed, output), "capture and eager disagree"

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
    cosine = torch.nn.functional.cosine_similarity(
        replayed.double().reshape(1, -1), want.double().reshape(1, -1)
    ).item()
    assert cosine >= 0.99995, cosine


def test_a_raw_scratch_plan_carries_its_declared_smallm_tokens() -> None:
    """The legacy raw path's caps declare the arm, and the binding carries it.

    ``TPMoEScratchCaps.smallm_tokens`` defaults to 0: the grouped arm serves
    every m, so the arena reserves no arm region and bind carves none. A
    binding that left the declaration unset made dispatch resolve the bundle's
    default instead, build the small-M arm for m <= 16 and fail for the missing
    scratch; it must carry the caps' zero. A caps that does reserve the arm
    stamps the arm's plan and its carved views on the binding.
    """

    from b12x.moe.fused_moe._impl import (
        TPMoEScratchCaps,
        _trellismx_smallm_limit,
        plan_tp_moe_scratch,
    )
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    prepared = _trellismx_bundle()
    impl = prepared._impl
    caps = TPMoEScratchCaps(
        device="cpu",
        max_tokens=8,
        num_topk=2,
        weight_plan=impl.plan,
        quant_mode="w4a8_mx",
        decode_config=MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_route_mode="grouped",
        ),
    )
    assert caps.smallm_tokens == 0
    plan = plan_tp_moe_scratch(caps, prewarm_launches=False)
    assert plan.layout.smallm_layout is None
    binding = plan.bind(
        scratch=torch.zeros(int(plan.scratch_specs()[0].nbytes), dtype=torch.uint8),
        a=torch.zeros(4, 512, dtype=torch.bfloat16),
        experts=impl,
        topk_weights=torch.full((4, 2), 0.5),
        topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        output=torch.empty(4, 512, dtype=torch.bfloat16),
    )
    assert binding.trellismx_smallm_tokens == 0
    assert binding.trellismx_smallm_scratch is None
    # Dispatch's decision, not the bundle's default (16): the grouped arm.
    assert _trellismx_smallm_limit(binding) == 0
