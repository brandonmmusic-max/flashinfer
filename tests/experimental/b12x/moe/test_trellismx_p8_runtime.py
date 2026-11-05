"""The in-tree P8 runtime (``b12x.moe._shared.p8``) and its attachment.

The P8 kernels are Brandon M. Music's (local-inference-lab/b12x#342). CPU parts
check the options object, the in-tree backend declaration, the compile cache's
source closure and, on the GLM-5.3 Flash release when it is present, that the
runtime built from a real sidecar resolves the recorded serving options and is
accepted by ``attach_trellismx_p8_runtime``. GPU parts compare the runtime's
outputs with digests recorded from the serving image's runtime, check eager,
caller-owned output and CUDA graph replay against each other, dispatch through
``fused_moe``, compare each arm with the reference, check that the programs the
runtime declares are the programs it launches, and prepare the attached experts
under b12x's preparation session.

Data (read only): the GLM release, located by ``B12X_TRELLISMX_GLM_RELEASE``
(see ``_trellismx_data.py``); the release-backed cases skip without it.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import os
import pickle
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import b12x
from b12x._lib.compile_pool import CompileJob
from b12x.moe._shared.kernels import trellismx_boundary_fused
from b12x.moe._shared.kernels.trellismx_boundary_fused import fused_output_stage
from b12x.moe._shared.p8.p8_native_kernel import _arm_key, _select_arm
from b12x.moe._shared.p8.plan import p8_capacity_launch_rows, p8_launch_rows
from b12x.moe._shared.p8.tile_policy import select_tile
from b12x.moe.fused_moe.trellismx_p8 import (
    P8_OPTION_ATTRIBUTES,
    TrellisMxP8Backend,
    TrellisMxP8Options,
    attach_trellismx_p8_runtime,
    trellismx_p8_attachment,
    verify_trellismx_p8_attachment,
)
from b12x.moe.fused_moe.trellismx_p8_runtime import (
    IN_TREE_P8_PACKAGE,
    GLM53_SERVING_OPTIONS,
    P8_PROGRAM_FACTORY,
    P8NativeTPMoE,
    P8ProgramSpec,
    P8RuntimeOptions,
    P8TensorSource,
    compile_p8_programs,
    in_tree_p8_backend,
)

from ._trellismx_data import GLM_RELEASE as _GLM_RELEASE
from ._trellismx_data import glm_sidecar


GLM_RELEASE = Path(_GLM_RELEASE)
_HIDDEN, _LOCAL, _EXPERTS, _TOP_K, _WORLD = 4096, 512, 288, 8, 4
_TENSORS = (
    "w13_trellis",
    "w2_trellis",
    "w13_scale_ue8m0",
    "w2_scale_ue8m0",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
    "gate_up_suh_fp16",
    "down_svh_fp16",
)
_SCALE_TENSORS = ("gate_up_suh_fp16", "intermediate_scales_fp16", "down_svh_fp16")
_P8_ROOT = Path(b12x.__file__).resolve().parent / "moe" / "_shared" / "p8"

#: What the serving configuration resolves to in the runtime: the values the
#: P8 interface records for the GLM recipe, plus the constructor defaults.
SERVING_RESOLVED = {
    "p8_down_remainder": False,
    "_dx2_phases": "both",
    "p8_dx2_rowpack": True,
    "p8_input_rowpack": True,
    "rp2_direct_max_tokens": 16,
    "rp2_direct_tile_m": 0,
    "rp2_skip_input_clear": True,
    "p8_epi_par": True,
    "small_m_scheduler": True,
    "fc1_tile_n": 128,
    "fc1_pipeline_stages": 2,
    "fc1_warps": 4,
    "fc1_warp_quant": False,
    "fc1_broadcast_a": True,
    "fc1_a_swizzle_rotate": False,
    "fc1_exact_staging": False,
    "prefill_chunk_tokens": 0,
    "grid_policy": True,
    "tile_major_tasks": False,
    "compact_input_storage": False,
    "force_materialized": None,
    "mac_override": None,
}


def _same(actual: dict, expected: dict) -> bool:
    return actual == expected and all(
        type(actual[name]) is type(expected[name]) for name in expected
    )


def _resolved(runtime) -> dict:
    return {name: getattr(runtime, name) for name in P8_OPTION_ATTRIBUTES}


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #


def test_options_default_to_the_serving_configuration() -> None:
    options = P8RuntimeOptions()
    assert options == GLM53_SERVING_OPTIONS
    values = options.as_dict()
    for name, value in {
        "down_remainder": "rp2",
        "down_remainder_phases": "both",
        "epi_par": True,
        "rp2_direct_max_tokens": 16,
        "rp2_skip_input_clear": True,
        "small_m_scheduler": True,
        "fc1_tile_n": 128,
        "prefill_chunk_tokens": 0,
        "grid_policy": True,
        "fc1_warp_quant": False,
        "fc1_broadcast_a": True,
        "deterministic_output": True,
        "force_materialized": None,
        "mac_override": None,
    }.items():
        assert values[name] == value and type(values[name]) is type(value), name
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.epi_par = False


@pytest.mark.parametrize(
    "option, axis, value",
    [
        ("rp2_direct_tile_m", "direct_tile_m", 32),
        ("fc1_pipeline_stages", "fc1_pipeline_stages", 3),
        ("fc1_warps", "fc1_warps", 8),
    ],
)
def test_the_launch_geometry_is_not_a_constructor_option(option, axis, value) -> None:
    """The launch configuration owns the geometry; the options cannot pin it.

    A constructor value here would be silently replaced by the plan's serving
    configuration, so the field does not exist: a caller that wants other
    geometry names it in a ``P8LaunchConfig``.
    """

    from b12x.moe._shared.p8.launch import P8LaunchConfig

    with pytest.raises(TypeError, match=f"unexpected keyword argument '{option}'"):
        P8RuntimeOptions(**{option: value})
    assert getattr(P8LaunchConfig(**{axis: value}), axis) == value


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("down_remainder", "1", ValueError),
        ("down_remainder", "", ValueError),
        ("down_remainder_phases", "all", ValueError),
        ("epi_par", 1, TypeError),
        ("grid_policy", None, TypeError),
        ("rp2_direct_max_tokens", True, TypeError),
        ("fc1_tile_n", "128", TypeError),
        ("force_materialized", 0, TypeError),
        ("mac_override", 1.5, TypeError),
    ],
)
def test_options_reject_wrong_types_and_values(field, value, error) -> None:
    with pytest.raises(error, match=field):
        P8RuntimeOptions(**{field: value})


def test_direct_rp2_auto_selection_is_one_tile_policy_arm() -> None:
    """Direct RP2 keeps one fixed geometry across its enabled row counts."""

    spec = SimpleNamespace(
        rp2_direct_tile_m=0,
        grouped_tile_m=0,
        rp2_stride=192,
        fc1_pipeline_stages=2,
        fc1_warps=4,
    )
    rows = range(1, int(P8RuntimeOptions().rp2_direct_max_tokens) + 1)
    keys = {_arm_key(spec, True, True, row) for row in rows}
    assert keys == {(True, True, select_tile(1)[0], 192, 2, 4)}


@pytest.mark.parametrize(
    "options, match",
    [
        (P8RuntimeOptions(rp2_direct_max_tokens=193), "0..192"),
        (P8RuntimeOptions(prefill_chunk_tokens=100), "multiple of 64"),
        (P8RuntimeOptions(fc1_tile_n=96), "FC1 tile N"),
        (
            P8RuntimeOptions(fc1_tile_n=64, small_m_scheduler=False),
            "Narrow FC1 requires small-M",
        ),
        (P8RuntimeOptions(force_materialized=True), "automatic fallback"),
        (P8RuntimeOptions(mac_override=-1), "mac_override must be positive"),
    ],
)
def test_runtime_rejects_what_the_original_rejected(options, match) -> None:
    """The runtime's own checks run on the resolved options before any I/O."""

    with pytest.raises(ValueError, match=match):
        P8NativeTPMoE(
            None,
            device=torch.device("cpu"),
            tp_rank=0,
            layer=3,
            intermediate=_LOCAL,
            options=options,
        )
    with pytest.raises(TypeError, match="P8RuntimeOptions"):
        P8NativeTPMoE(
            None,
            device=torch.device("cpu"),
            tp_rank=0,
            options={"epi_par": True},
        )


def test_the_p8_package_reads_no_environment() -> None:
    """Every behavior setting is an option or an argument: no module of the P8
    package imports ``os`` or touches ``environ``/``getenv``."""

    offenders = []
    for path in sorted(_P8_ROOT.rglob("*.py")):
        text = path.read_text()
        for marker in ("os.environ", "getenv"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import) and any(
                alias.name == "os" for alias in node.names
            ):
                offenders.append(f"{path.name}:{node.lineno}: import os")
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                offenders.append(f"{path.name}:{node.lineno}: from os import")
            elif isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                offenders.append(f"{path.name}:{node.lineno}: .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
                offenders.append(f"{path.name}:{node.lineno}: {node.id}")
    assert offenders == []


def _stand_in_sidecar(experts=4, hidden=256, intermediate=128, bits=4):
    """A small identity-boundary sidecar (the historical v2 schema): enough for
    the runtime to construct without the release, not enough to attach."""

    generator = torch.Generator().manual_seed(7)
    words = 16 * bits
    tensors = {
        "w13_trellis": torch.randint(
            -(2**31),
            2**31 - 1,
            (2, experts, hidden // 16, intermediate // 16, words),
            dtype=torch.int32,
            generator=generator,
        ),
        "w2_trellis": torch.randint(
            -(2**31),
            2**31 - 1,
            (experts, intermediate // 16, hidden // 16, words),
            dtype=torch.int32,
            generator=generator,
        ),
        "w13_scale_ue8m0": torch.full(
            (experts, 2 * intermediate, hidden // 32), 127, dtype=torch.uint8
        ),
        "w2_scale_ue8m0": torch.full(
            (experts, hidden, intermediate // 32), 127, dtype=torch.uint8
        ),
    }
    metadata = {
        "schema": "glm53-p8-mcg-tp4-rank.v2",
        "layer": "3",
        "rank": "1",
        "world_size": "4",
        "bits": str(bits),
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "ldlq": "false",
        "boundary": "identity",
        "source_design_sha256": "ab" * 32,
    }
    return tensors, metadata


def test_runtime_constructs_from_a_stand_in_sidecar() -> None:
    """Construction, in-place tensors and recorded options without the release."""

    tensors, metadata = _stand_in_sidecar()
    # The small-M owner and its fused scratch need the GLM TP4 geometry.
    options = P8RuntimeOptions(small_m_scheduler=False)
    runtime = P8NativeTPMoE(
        P8TensorSource(tensors, metadata),
        device=torch.device("cpu"),
        tp_rank=1,
        layer=3,
        hidden=256,
        intermediate=128,
        options=options,
    )
    assert not runtime.full_coupled and runtime.scale_component is None
    assert (runtime.experts, runtime.trellis_bits) == (4, 4)
    for name, source in (("w13_stream", "w13_trellis"), ("w2_stream", "w2_trellis")):
        assert (
            getattr(runtime, name).untyped_storage().data_ptr()
            == tensors[source].untyped_storage().data_ptr()
        )
    expected = dict(SERVING_RESOLVED, small_m_scheduler=False)
    assert _same(TrellisMxP8Options.from_runtime(runtime).as_dict(), expected)
    with pytest.raises(RuntimeError, match="invalid P8 native sidecar metadata"):
        P8NativeTPMoE(
            P8TensorSource(tensors, dict(metadata, layer="4")),
            device=torch.device("cpu"),
            tp_rank=1,
            layer=3,
            hidden=256,
            intermediate=128,
            options=options,
        )


def test_the_runtime_declares_its_programs_as_a_compile_job() -> None:
    """The spec is plain compile-job metadata taken from the resolved settings."""

    tensors, metadata = _stand_in_sidecar()
    options = P8RuntimeOptions(small_m_scheduler=False, prefill_chunk_tokens=128)
    runtime = P8NativeTPMoE(
        P8TensorSource(tensors, metadata),
        device=torch.device("cpu"),
        tp_rank=1,
        layer=3,
        hidden=256,
        intermediate=128,
        options=options,
    )
    factory, spec = runtime.program_declaration
    assert factory == P8_PROGRAM_FACTORY and spec is runtime.program_spec
    assert isinstance(spec, P8ProgramSpec)
    for field in dataclasses.fields(spec):
        name = {
            "scale_sandwich": None,
            "dx2_phases": "_dx2_phases",
            # Launch axes: the runtime's spec carries their serving values,
            # which the launch configurations replace per candidate.
            "grouped_tile_m": None,
            "rp2_stride": None,
        }.get(field.name, field.name)
        if name is not None:
            assert getattr(spec, field.name) == getattr(runtime, name), field.name
    assert spec.scale_sandwich is (runtime.scale_component is not None)
    job = CompileJob.create(factory, spec, 300, CompileJob.DEVICE_ORDINAL)
    assert pickle.loads(pickle.dumps(job)) == job
    # A bounded prefill launches full chunks and a tail; other calls launch once.
    assert p8_launch_rows(spec, 300) == (44, 128)
    assert p8_launch_rows(spec, 256) == (128,)
    assert p8_launch_rows(spec, 100) == (100,)
    # A capacity declares one launch per arm that its calls reach: every arm of
    # every call from 1 to the capacity rows, and no launch beyond a chunk.
    coupled = dataclasses.replace(spec, full_coupled=True)

    def arm(rows):
        return _arm_key(coupled, *_select_arm(coupled, rows), rows)

    for capacity in (1, 2, 16, 17, 100, 128, 300):
        declared = p8_capacity_launch_rows(coupled, capacity)
        assert max(declared) <= min(capacity, coupled.prefill_chunk_tokens)
        assert len({arm(rows) for rows in declared}) == len(declared)
        reached = {
            arm(rows)
            for tokens in range(1, capacity + 1)
            for rows in p8_launch_rows(coupled, tokens)
        }
        assert {arm(rows) for rows in declared} == reached, capacity
    # Only the full-coupled deterministic runtimes b12x attaches declare programs.
    with pytest.raises(ValueError, match="full-coupled deterministic"):
        compile_p8_programs(spec, 1, 0)


def test_the_one_row_arena_holds_the_residual_plane() -> None:
    """A one-row direct launch declares the input carrier's residual plane.

    The RP2 prologue writes token ``t``'s residual plane at row
    ``rp2_stride + t`` of the input carrier, while the direct arm's tiles cover
    ``tokens * top_k * tile_m`` rows. At one row the serving tile covers 128
    rows, so a carrier sized from the tiles alone would place the residual
    plane outside ``packed_a``: the call would write it into the scale plane
    the same call reads.
    """

    from b12x.moe._shared.p8.plan import p8_scratch_layout_for

    tensors, metadata = _stand_in_sidecar()
    runtime = P8NativeTPMoE(
        P8TensorSource(tensors, metadata),
        device=torch.device("cpu"),
        tp_rank=1,
        layer=3,
        hidden=256,
        intermediate=128,
        options=P8RuntimeOptions(small_m_scheduler=False),
    )
    # The small-M owner is the direct RP2 arm, which is what stages the
    # residual plane; the stand-in sidecar is identity-boundary, so the plan
    # is made full-coupled to reach the same arm as the GLM runtime.
    spec = dataclasses.replace(
        runtime.program_spec,
        small_m_scheduler=True,
        full_coupled=True,
        p8_input_rowpack=True,
        rp2_direct_max_tokens=16,
    )
    for stride in (128, 192):
        layout = p8_scratch_layout_for(dataclasses.replace(spec, rp2_stride=stride), 1)
        rows = layout.region("packed_a").nbytes // spec.hidden
        assert rows >= stride + 1, (stride, rows)
    # Two rows and above already cover the residual plane from the tiles alone,
    # so the serving arena sizes do not move.
    assert (
        p8_scratch_layout_for(dataclasses.replace(spec, rp2_stride=192), 16)
        .region("packed_a")
        .nbytes
        // spec.hidden
        == 16 * spec.topk * 16
    )


def test_the_clear_spans_skip_only_the_route_partial_sum() -> None:
    """A deterministic call clears every arena byte but the per-route partial sum.

    The runtime clears one span per gap between the regions it must clear, and
    the per-route partial sum is the only region a deterministic call leaves
    alone: its split FC2 owners write every element of it before the reducer
    reads it. The complement is one span, so a call still issues one fill.
    """

    from b12x.moe._shared.p8.plan import p8_scratch_layout_for
    from b12x.moe._shared.p8.scratch import INTERMEDIATE, ROUTE_OUTPUT, p8_clear_spans

    spec = dataclasses.replace(
        _stand_in_runtime().program_spec,
        full_coupled=True,
        deterministic_output=True,
        small_m_scheduler=True,
        p8_input_rowpack=True,
        rp2_direct_max_tokens=16,
    )
    for rows in (1, 16, 17, 129):
        layout = p8_scratch_layout_for(spec, rows)
        route = layout.region(ROUTE_OUTPUT)
        assert layout.names()[-1] == ROUTE_OUTPUT
        # Without a skipped region the complement is the whole arena, exactly
        # as the single fill covered it before.
        assert p8_clear_spans(layout) == ((0, layout.nbytes),)
        assert p8_clear_spans(layout, skip=(ROUTE_OUTPUT,)) == ((0, route.offset),)
        start = layout.region(INTERMEDIATE).offset
        assert p8_clear_spans(layout, start=start) == ((start, layout.nbytes - start),)
        assert p8_clear_spans(layout, start=start, skip=(ROUTE_OUTPUT,)) == (
            (start, route.offset - start),
        )


def test_the_clear_spans_include_the_tail_after_a_skipped_final_region() -> None:
    """A skipped final region does not take the arena's trailing padding with it.

    ``p8_clear_spans`` returns the complement of the skipped regions, so
    alignment padding between a skipped final region and the arena's declared
    size is still cleared. The serving geometry has no such tail (its per-route
    partial sum is a whole number of 16-byte rows), so the layout below states
    one directly.
    """

    from b12x.moe._shared.p8.scratch import (
        P8ScratchLayout,
        P8ScratchRegion,
        p8_clear_spans,
    )

    layout = P8ScratchLayout(
        regions=(
            P8ScratchRegion("packed_a", "uint8", (32,), 0, 32),
            P8ScratchRegion("route_output", "uint8", (8,), 32, 8),
        ),
        nbytes=48,
    )
    route = layout.region("route_output")
    assert p8_clear_spans(layout) == ((0, layout.nbytes),)
    assert p8_clear_spans(layout, skip=("route_output",)) == (
        (0, route.offset),
        (route.offset + route.nbytes, layout.nbytes - route.offset - route.nbytes),
    )


def _stand_in_runtime(**options):
    tensors, metadata = _stand_in_sidecar()
    return P8NativeTPMoE(
        P8TensorSource(tensors, metadata),
        device=torch.device("cpu"),
        tp_rank=1,
        layer=3,
        hidden=256,
        intermediate=128,
        options=P8RuntimeOptions(small_m_scheduler=False, **options),
    )


def test_a_misaligned_p8_arena_is_rejected() -> None:
    """The launch assumes 16-byte alignment, so the arena must carry it.

    ``_gptr`` creates every region's pointer with ``assumed_align=16`` and each
    region offset is 16-byte aligned, so a one-byte-offset slice of a properly
    sized buffer would hand the kernels misaligned pointers that they are told
    are aligned. The arena specification states the requirement and the
    workspace check enforces it.
    """

    from b12x.moe._shared.p8.plan import p8_scratch_layout_for
    from b12x.moe._shared.p8.scratch import P8_ARENA_ALIGNMENT

    runtime = _stand_in_runtime()
    assert runtime.scratch_specs(torch.device("cpu"), 1)[0].alignment == (
        P8_ARENA_ALIGNMENT
    )
    spec = dataclasses.replace(
        runtime.program_spec,
        small_m_scheduler=True,
        full_coupled=True,
        p8_input_rowpack=True,
    )
    layout = p8_scratch_layout_for(spec, 1)
    base = torch.empty(layout.nbytes + P8_ARENA_ALIGNMENT, dtype=torch.uint8)
    shift = (P8_ARENA_ALIGNMENT - base.data_ptr() % P8_ARENA_ALIGNMENT) % (
        P8_ARENA_ALIGNMENT
    )
    aligned = base[shift : shift + layout.nbytes]
    assert aligned.data_ptr() % P8_ARENA_ALIGNMENT == 0
    assert runtime._workspace(aligned, layout).numel() == layout.nbytes
    misaligned = base[shift + 1 : shift + 1 + layout.nbytes]
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % P8_ARENA_ALIGNMENT == 1
    with pytest.raises(ValueError, match="16-byte aligned"):
        runtime._workspace(misaligned, layout)


def test_the_resident_grid_cap_has_one_owner_per_value() -> None:
    """An explicit cap pin wins; otherwise a raced cap, then the policy."""

    from b12x.moe._shared.p8.tile_policy import (
        direct_grid_capacity,
        resolve_grid_cap,
    )

    # A constructor pin is the only owner of the cap.
    assert resolve_grid_cap(16, arm=192, config=64, policy=True, pin=96) == 96
    # Without a pin, the raced configuration's nonzero cap wins.
    assert resolve_grid_cap(16, arm=192, config=64, policy=True, pin=None) == 64
    # ``mac=0`` means the runtime's grid policy ...
    assert resolve_grid_cap(16, arm=192, config=0, policy=True, pin=None) == (
        direct_grid_capacity(16, 192)
    )
    # ... or, with the policy off, the arm's own cap.
    assert resolve_grid_cap(16, arm=192, config=0, policy=False, pin=None) == 192
    runtime = _stand_in_runtime(mac_override=96)
    assert runtime.mac_override == 96
    assert runtime.program_spec.mac_override == 96


# --------------------------------------------------------------------------- #
# backend declaration and compile-cache identity
# --------------------------------------------------------------------------- #


def test_in_tree_backend_pins_every_p8_file() -> None:
    backend = in_tree_p8_backend()
    assert isinstance(backend, TrellisMxP8Backend)
    assert backend.package == IN_TREE_P8_PACKAGE == "b12x"
    assert backend.root() == _P8_ROOT.parents[2]
    files = dict(backend.files)
    on_disk = {
        path.relative_to(_P8_ROOT.parents[2]).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in _P8_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert files == on_disk
    assert backend.relative_path(P8NativeTPMoE.__module__) in files
    assert (
        files["moe/_shared/p8/p8_native_kernel.py"]
        == on_disk["moe/_shared/p8/p8_native_kernel.py"]
    )
    backend.verify()
    assert in_tree_p8_backend() is backend
    assert (
        TrellisMxP8Backend(package="b12x", files=files, upstream="").identity
        == backend.identity
    )
    assert "b12x#342" in backend.upstream and "Brandon M. Music" in backend.upstream


def test_compile_cache_closure_covers_the_p8_kernels() -> None:
    """The launch and reducer fingerprints reach every kernel module they run."""

    from b12x._lib import compiler
    from b12x._lib.source_dependencies import callable_sources, source_dependencies
    from b12x.moe._shared.p8.kernels.p8_coupled_topk import P8CoupledTopKSumKernel
    from b12x.moe._shared.p8.kernels.w4a8_launch import P8DynamicMoEW4A8Launch

    root = compiler._PACKAGE_ROOT
    deps = source_dependencies(root)

    def closure(target) -> set[str]:
        names = set()
        for module, path in callable_sources(target, root):
            names.update(name for name, _ in deps.closure(module, path))
        return names

    launch = closure(P8DynamicMoEW4A8Launch)
    kernels = "b12x.moe._shared.p8.kernels."
    for leaf in (
        "route_hoist_dynamic",
        "route_hoist_k5",
        "route_hoist_direct32",
        "route_hoist_fc1",
        "route_hoist_prefill",
        "row_alias376_fc1",
        "fc2_k5_funnel",
        "direct_m32_fc2",
        "grouped_fc2_grid376",
        "policy_m32_fc1",
        "policy_m32_fc2",
        "p8_small_m",
        "p8_narrow_fc1",
        "npk_decode",
        "w4a8_phase1",
        "w4a8_phase2",
    ):
        assert kernels + leaf in launch, leaf
    # The H128 helper is the shared one, not a copy under the P8 tree.
    assert "b12x.moe._shared.kernels.w4a8_trellis_decode" in launch
    # The kernel body and the NPK state law are the shared modules, so the P8
    # programs' identity follows them.
    assert "b12x.moe._shared.kernels.dynamic.kernel" in launch
    assert "b12x.moe._shared.kernels.npk_state_law" in launch
    assert kernels + "w4a8_had128" not in launch
    assert kernels + "p8_coupled_topk" in closure(P8CoupledTopKSumKernel)
    # The runtime module's closure includes the computed-import helper, which
    # must declare its targets for the fingerprint to resolve.
    from b12x.moe._shared.p8 import p8_native_kernel

    runtime_path = Path(p8_native_kernel.__file__).resolve()
    names = {
        name
        for name, _ in deps.closure(
            "b12x.moe._shared.p8.p8_native_kernel", runtime_path
        )
    }
    assert kernels + "p8_down_remainder" in names
    for target in (P8DynamicMoEW4A8Launch, P8CoupledTopKSumKernel):
        assert len(compiler._target_source_fingerprint(target)) == 64


# --------------------------------------------------------------------------- #
# the runtime on a released sidecar (CPU)
# --------------------------------------------------------------------------- #


def _sidecar_path(layer: int, rank: int) -> Path:
    return Path(glm_sidecar(layer, rank))


def _load(layer: int, rank: int, device: torch.device):
    from safetensors import safe_open

    path = _sidecar_path(layer, rank)
    if not path.exists():
        pytest.skip(f"released data absent: {path}")
    manifest = json.loads((GLM_RELEASE / "trellismx-manifest.json").read_text())
    record = next(
        r for r in manifest["files"] if r["layer"] == layer and r["rank"] == rank
    )
    with safe_open(str(path), framework="pt") as handle:
        host = {name: handle.get_tensor(name) for name in _TENSORS}
        metadata = dict(handle.metadata())
    transform = hashlib.sha256(
        (GLM_RELEASE / "design" / "transform.json").read_bytes()
    ).hexdigest()
    payload = {name: value.to(device) for name, value in host.items()}
    return SimpleNamespace(
        host=host,
        payload=payload,
        metadata=metadata,
        design=record["source_design_sha256"],
        transform=transform,
        layer=layer,
        rank=rank,
        device=device,
    )


def _runtime(data, options: P8RuntimeOptions | None = None) -> P8NativeTPMoE:
    """Built as the GLM recipe builds it: the payload in place, the scale component on the CPU."""

    tensors = {
        **data.payload,
        **{name: data.payload[name].cpu() for name in _SCALE_TENSORS},
    }
    return P8NativeTPMoE(
        P8TensorSource(tensors, data.metadata),
        device=data.device,
        tp_rank=data.rank,
        world_size=_WORLD,
        layer=data.layer,
        topk=_TOP_K,
        hidden=_HIDDEN,
        intermediate=_LOCAL,
        expected_design_sha256=data.design,
        expected_transform_sha256=data.transform,
        options=options,
    )


def _prepare(data):
    from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts

    return prepare_trellismx_experts(
        data.payload,
        data.metadata,
        num_experts=_EXPERTS,
        hidden_size=_HIDDEN,
        intermediate_size=_LOCAL,
        top_k=_TOP_K,
        device=data.device,
    )


@pytest.fixture(scope="module")
def cpu_layer3():
    return _load(3, 0, torch.device("cpu"))


def test_runtime_resolves_the_recorded_serving_options(cpu_layer3, monkeypatch) -> None:
    for name, value in {
        "B12X_P8_DOWN_REMAINDER": "1",
        "B12X_P8_EPI_PAR": "0",
        "B12X_P8_RP2_DIRECT_MAX_TOKENS": "0",
        "B12X_P8_RP2_SKIP_INPUT_CLEAR": "0",
        "GLM53_P8_GRID_POLICY": "0",
        "GLM53_P8_FC1_BROADCAST_A": "0",
    }.items():
        monkeypatch.setenv(name, value)
    runtime = _runtime(cpu_layer3)
    assert _same(_resolved(runtime), SERVING_RESOLVED)
    assert _same(TrellisMxP8Options.from_runtime(runtime).as_dict(), SERVING_RESOLVED)
    assert runtime.options is GLM53_SERVING_OPTIONS
    assert runtime.full_coupled and runtime.trellis_bits == 4


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("none", (False, False, False, 1)),
        ("dx2", (True, False, False, 2)),
        ("rp", (False, True, False, 1)),
        ("rp2", (False, True, True, 1)),
    ],
)
def test_down_remainder_modes_resolve_as_the_environment_values_did(
    cpu_layer3, mode, expected
) -> None:
    runtime = _runtime(cpu_layer3, P8RuntimeOptions(down_remainder=mode))
    assert (
        runtime.p8_down_remainder,
        runtime.p8_dx2_rowpack,
        runtime.p8_input_rowpack,
        runtime._dx2_planes,
    ) == expected


def test_in_tree_runtime_attaches_with_the_in_tree_backend(cpu_layer3) -> None:
    prepared = _prepare(cpu_layer3)
    runtime = _runtime(cpu_layer3)
    backend = in_tree_p8_backend()
    attached = attach_trellismx_p8_runtime(
        prepared, runtime, backend=backend, options=SERVING_RESOLVED
    )
    verify_trellismx_p8_attachment(attached)
    entry = trellismx_p8_attachment(attached)
    assert entry.live and entry.runtime is runtime
    assert entry.backend == backend and entry.backend.identity == backend.identity
    assert _same(entry.options.as_dict(), SERVING_RESOLVED)
    record = attached._impl.representation.value
    for name in ("w13_sfb", "w2_sfb"):
        assert (
            getattr(record, name).untyped_storage().data_ptr()
            == getattr(runtime, name).untyped_storage().data_ptr()
        )


def test_the_attachment_declares_the_runtime_programs(cpu_layer3) -> None:
    """Each capacity's plan carries the runtime's compile job, pickled or not."""

    runtime = _runtime(cpu_layer3)
    attached = attach_trellismx_p8_runtime(
        _prepare(cpu_layer3),
        runtime,
        backend=in_tree_p8_backend(),
        options=SERVING_RESOLVED,
    )
    entry = trellismx_p8_attachment(attached)
    assert entry.programs == runtime.program_declaration
    expected = (
        CompileJob.create(
            P8_PROGRAM_FACTORY, runtime.program_spec, 17, CompileJob.DEVICE_ORDINAL
        ),
    )
    assert entry.compile_jobs(17) == expected
    descriptor = pickle.loads(pickle.dumps(entry))
    assert not descriptor.live and descriptor.programs == entry.programs
    assert descriptor.compile_jobs(17) == expected


# --------------------------------------------------------------------------- #
# the race identity across tensor-parallel ranks (CPU)
# --------------------------------------------------------------------------- #


def _finish(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as done:
            return done.value


def _p8_race(plan):
    """The race key a real preparation session derives for one P8 plan.

    The declaration's device is cleared first. This process holds no CUDA
    device (the compiler's offline description stubs are installed, the bundle
    is on the cpu and the session has no ordinal), and the declaration/session
    device match is a placement check that the key does not hash. Everything
    else is the session's own path: enumeration, program discovery through the
    compiler's metadata-only description, and ``_choice_key``.
    """

    from b12x.preparation import DetectedDevice, PreparationSession
    from b12x.preparation.session import _coalesce_requests

    plan = dataclasses.replace(plan, _device=None)
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=True, compile_workers=0
    ) as session:
        request = plan.request(name="moe.decode", prepare_call=lambda state: None)
        job = session.begin((request,))
        (obligation,) = _finish(job._configure(_coalesce_requests((request,))))
        _finish(job._describe_candidates(obligation))
        return job._choice_key(obligation, {}), obligation.candidate_program_identity


def _rank_identity_probe(connection):
    """Race and describe released layer 3 ranks 0 and 1 without a CUDA device."""

    try:
        import multiprocessing

        from b12x._lib.compile_pool import _initialize_worker

        context = multiprocessing.get_context("spawn")
        # The production offline description the compiler workers run: the
        # target device is described, never initialized.
        _initialize_worker(
            0,
            (12, 0),
            "GPU-synthetic",
            "NVIDIA synthetic",
            188,
            101376,
            102400,
            context.Array("q", (0, 0)),
        )
        from b12x._lib.compile_pool import describe_compilation
        from b12x.moe import fused_moe as moe
        from b12x.moe._shared.p8.launch import P8LaunchConfig

        cases = (
            ("rows1", 1, None),
            ("rows17", 17, None),
            ("tile32", 1, P8LaunchConfig(direct_tile_m=32)),
        )
        results = {}
        for rank in (0, 1):
            data = _load(3, rank, torch.device("cpu"))
            attached = attach_trellismx_p8_runtime(
                _prepare(data),
                _runtime(data),
                backend=in_tree_p8_backend(),
                options=SERVING_RESOLVED,
            )
            if rank == 0:
                # A full-coupled layer has exactly one geometry, so the
                # "different geometry" control is the stored rate: a K5
                # program must not share a K4 program's identity.
                spec = trellismx_p8_attachment(attached).program_spec()

                def programs_of(program_spec):
                    return sorted(
                        (program.dialect, program.key)
                        for program in describe_compilation(
                            CompileJob.create(
                                P8_PROGRAM_FACTORY,
                                program_spec,
                                1,
                                CompileJob.DEVICE_ORDINAL,
                            )
                        ).programs
                    )

                results["programs"] = programs_of(spec)
                results["rate5_programs"] = programs_of(
                    dataclasses.replace(spec, trellis_bits=5)
                )
            for name, rows, override in cases:
                plan = moe.plan_execution(
                    experts=attached,
                    capacity=moe.ExecutionCapacity(
                        max_tokens=rows, top_k=_TOP_K, route_num_experts=0
                    ),
                    override=override,
                )
                key, identity = _p8_race(plan)
                results[f"{rank}:{name}"] = {
                    "key": key,
                    "identity": identity,
                    "jobs": [
                        pickle.dumps(job) for job in plan._compile_jobs(override, None)
                    ],
                    "query": plan.contract.encode_query(plan.query),
                    "invocation": plan.invocation.to_dict(),
                }
        connection.send((results, None))
    except BaseException:
        connection.send((None, traceback.format_exc()))
    finally:
        connection.close()


def test_the_p8_race_identity_does_not_depend_on_the_tensor_parallel_rank():
    """The four ranks of a layer declare one program set and one race key.

    b12x's collective preparation requires every rank of a tensor-parallel
    group to declare the same race keys, so the candidates' program identity
    may not carry anything a rank selects. The P8 kernels receive no rank; the
    rank chooses which sidecar slice a runtime holds and is verified against
    that slice, and nothing else. Each rank's runtime is therefore described
    and raced here through the session's own discovery, and the compile jobs,
    encoded queries, invocations, program identities and race keys must agree.
    A different row geometry, a different launch configuration and a different
    stored rate still separate keys and programs, so the shared identity does
    not collapse distinct programs.
    """

    missing = [
        path
        for path in (_sidecar_path(3, rank) for rank in (0, 1))
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"released data absent: {missing[0]}")

    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_rank_identity_probe, args=(writer,))
    process.start()
    writer.close()
    try:
        assert reader.poll(1800), "rank identity probe timed out"
        results, error = reader.recv()
    finally:
        process.join(timeout=60)
        reader.close()
        if process.is_alive():
            process.kill()
            process.join()
    assert error is None, error
    assert process.exitcode == 0

    first, second = results["0:rows1"], results["1:rows1"]
    assert first["jobs"] == second["jobs"]
    assert first["query"] == second["query"]
    assert first["invocation"] == second["invocation"]
    assert first["identity"] == second["identity"]
    assert first["key"] == second["key"]
    assert results["0:rows17"]["key"] != first["key"]
    assert results["0:tile32"]["key"] != first["key"]
    assert results["programs"] != results["rate5_programs"]


# --------------------------------------------------------------------------- #
# GPU: the serving runtime's outputs, replay, dispatch and the reference
# --------------------------------------------------------------------------- #

#: SHA-256 of the BF16 output bytes of the runtime as the GLM-5.3 Flash serving
#: image runs it (b12x#342's ``P8NativeTPMoE`` with the serving settings in the
#: environment), GLM layer 3 rank 0, on ``_inputs(3, 0, m)``.
SERVING_DIGESTS = {
    1: "c2e7ffae54030ea03d04399c6f40a1936d7cd9cc57e48a86c175d4b618b232f0",
    2: "dd7d8e1190201b26063cb32ed4bf6c71501acd2bfb82d803183ac06f767377fa",
    16: "6d1570ae145747e7a5aa0968bd4ff8c1e591a9f8048d1486e674fffa4d9b0046",
    17: "fcd7834b4202320cd87a0fa56398309d5245806ba58dd8679d58892137357aa9",
    128: "a10d85a5420e14b9967bf47e12ed0a6c67e200e294d844a222f683676a515473",
    129: "e280530b2b227cc61301205b46ae6936e263f8a40baed475423e1f94332853c2",
    1024: "f49ada11093ee715e9e1df7a7f708e60dd82bef93d24dedae3021265f47639bb",
}


def _gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((12, 0), (12, 1)):
        pytest.skip(f"SM120 or SM121 GPU required, found sm_{major}{minor}")
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(layer: int, rank: int, rows: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(
        20261025 + 100000 * layer + 1000 * rank + rows
    )
    x = torch.randn(rows, _HIDDEN, generator=generator).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(_EXPERTS, generator=generator)[:_TOP_K] for _ in range(rows)]
    )
    weights = torch.rand(rows, _TOP_K, generator=generator, dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    return x, ids, weights


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


@pytest.fixture(scope="module")
def gpu_layer3():
    device = _gpu()
    data = _load(3, 0, device)
    runtime = _runtime(data)
    with torch.cuda.device(device):
        runtime.install_compiled_programs(
            compile_p8_programs(
                runtime.program_spec, 129, device.index or torch.cuda.current_device()
            ),
            129,
        )
    attached = attach_trellismx_p8_runtime(
        _prepare(data), runtime, backend=in_tree_p8_backend(), options=SERVING_RESOLVED
    )
    verify_trellismx_p8_attachment(attached)
    return SimpleNamespace(data=data, runtime=runtime, attached=attached, device=device)


def _producer(tensor):
    """A producer that rewrites the activations, as a race sample requires."""

    fixture = tensor.clone()

    def produce():
        tensor.copy_(fixture)

    return produce


def _arena(runtime, rows, device, config=None):
    """The caller-owned scratch a call of ``rows`` rows carves its buffers from."""

    return tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in runtime.scratch_specs(device, rows, config)
    )


def _run_runtime(runtime, x, weights, ids, rows, device, config=None) -> torch.Tensor:
    """One standalone runtime call: a caller-owned output and the declared arena."""

    out = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
    runtime(
        x,
        weights,
        ids,
        out,
        scratch=_arena(runtime, rows, device, config),
        config=config,
    )
    return out


def _graph_output(runtime, x, weights, ids, config=None) -> torch.Tensor:
    out = torch.empty(x.shape[0], _HIDDEN, dtype=torch.bfloat16, device=x.device)
    scratch = _arena(runtime, x.shape[0], x.device, config)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            runtime(x, weights, ids, out, scratch=scratch, config=config)
    torch.cuda.current_stream().wait_stream(stream)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    return out


def _dispatch(attached, x, ids, weights) -> torch.Tensor:
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe import _impl

    rows = x.shape[0]
    plan = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=rows, top_k=_TOP_K, route_num_experts=0
        ),
    )
    assert plan._compile_jobs(None, None) == trellismx_p8_attachment(
        attached
    ).compile_jobs(rows)
    state = plan._materialize(None, SimpleNamespace(ordinal=x.device.index))
    out = torch.full(
        (rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=x.device
    )
    binding = state.bind(
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=out,
        scratch=torch.empty(
            state.scratch.scratch_specs()[0].shape, dtype=torch.uint8, device=x.device
        ),
        input_scales_static=True,
    )
    assert binding.implementation == "trellismx_p8"
    assert _impl.b12x_moe_fp4(binding=binding) is out
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("rows", [1, 2, 16, 17, 128, 129, 1024])
def test_in_tree_runtime_matches_the_serving_runtime(gpu_layer3, rows) -> None:
    """Bit for bit: the serving runtime's digest, out=, graph replay and fused_moe."""

    device = gpu_layer3.device
    x, ids, weights = (t.to(device) for t in _inputs(3, 0, rows, torch.device("cpu")))
    ids = ids.to(torch.int32)
    eager = _run_runtime(gpu_layer3.runtime, x, weights, ids, rows, device)
    out = torch.full((rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device)
    gpu_layer3.runtime(
        x, weights, ids, out, scratch=_arena(gpu_layer3.runtime, rows, device)
    )
    replay = _graph_output(gpu_layer3.runtime, x, weights, ids)
    dispatched = _dispatch(gpu_layer3.attached, x, ids, weights)
    torch.cuda.synchronize()
    assert torch.isfinite(eager.float()).all()
    for other in (out, replay, dispatched):
        assert torch.equal(eager.view(torch.int16), other.view(torch.int16))
    assert _digest(eager) == SERVING_DIGESTS[rows]


@pytest.mark.parametrize("rows", [1, 2, 16, 17, 129])
def test_in_tree_runtime_against_the_reference(gpu_layer3, rows) -> None:
    """Each arm against the reference in its own activation regime."""

    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    x, ids, weights = _inputs(3, 0, rows, torch.device("cpu"))
    device = gpu_layer3.device
    produced = _run_runtime(
        gpu_layer3.runtime,
        x.to(device),
        weights.to(device),
        ids.to(device, torch.int32),
        rows,
        device,
    )
    direct = rows <= SERVING_RESOLVED["rp2_direct_max_tokens"]
    count = min(rows, 32)
    meta = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _LOCAL,
        "num_experts": _EXPERTS,
        "top_k": _TOP_K,
        "bits": 4,
        "rank": 0,
        "world_size": _WORLD,
        "sign_draw": 0,
        # Direct RP2 rows carry a two-term input; grouped rows a quantized one.
        "quantize_activations": not direct,
    }
    reference = trellismx_moe_reference(
        x[:count], ids[:count], weights[:count], dict(gpu_layer3.data.host), meta
    ).double()
    cosine = torch.nn.functional.cosine_similarity(
        produced[:count].double().cpu().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine >= 0.9999, cosine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_in_tree_runtime_rejects_unusable_routing_tensors(gpu_layer3) -> None:
    """Routing inputs must already match the raw-pointer ABI."""

    device = gpu_layer3.device
    x, ids, weights = _inputs(3, 0, 1, device)
    x, weights = x.to(device), weights.to(device)
    ids = ids.to(device=device, dtype=torch.int32)
    invalid = (
        (ids.cpu(), weights),
        (ids, weights.cpu()),
        (ids.to(torch.int16), weights),
        (ids.to(torch.int64), weights),
        (ids, weights.to(torch.bfloat16)),
        (torch.cat((ids, ids), dim=1)[:, ::2], weights),
        (ids, torch.cat((weights, weights), dim=1)[:, ::2]),
    )
    scratch = _arena(gpu_layer3.runtime, 1, device)
    out = torch.empty(1, _HIDDEN, dtype=torch.bfloat16, device=device)
    for bad_ids, bad_weights in invalid:
        with pytest.raises(RuntimeError, match="P8 native routing contract mismatch"):
            gpu_layer3.runtime(x, bad_weights, bad_ids, out, scratch=scratch)


@pytest.mark.parametrize(
    "capacity,rows",
    [(1, 1), (17, 2), (17, 16), (17, 17), (129, 17), (129, 129), (4096, 129)],
)
def test_the_route_partial_sum_needs_no_clear(gpu_layer3, capacity, rows) -> None:
    """A poisoned arena serves the same bytes as a zeroed one.

    The call does not clear the m * top-k * hidden FP32 per-route partial
    sum, because the split FC2 owners write every element of it before the
    reducer reads it. Poisoning the whole arena with 0xFF first leaves nothing
    to read but what the call itself writes, so the output must be
    bit-identical to a call on a zeroed arena: for the direct arm (two and
    sixteen rows), the grouped arm (seventeen and 129 rows), and a call that
    runs in a larger capacity's arena.
    """

    device = gpu_layer3.device
    runtime = gpu_layer3.runtime
    x, ids, weights = (t.to(device) for t in _inputs(3, 0, rows, torch.device("cpu")))
    ids = ids.to(torch.int32)

    def call(poison: bool):
        out = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        scratch = _arena(runtime, capacity, device)
        if poison:
            scratch[0].fill_(0xFF)
        runtime(x, weights, ids, out, scratch=scratch)
        torch.cuda.synchronize()
        return out

    zeroed = call(False)
    assert bool(torch.isfinite(zeroed.float()).all())
    assert torch.equal(zeroed.view(torch.int16), call(True).view(torch.int16))


@pytest.fixture(scope="module")
def gpu_rank_layer3(gpu_layer3):
    """Released layer 3 as each tensor-parallel rank holds it, on one device."""

    device = gpu_layer3.device
    ranks = [
        SimpleNamespace(
            rank=0,
            data=gpu_layer3.data,
            runtime=gpu_layer3.runtime,
            attached=gpu_layer3.attached,
            device=device,
        )
    ]
    for rank in range(1, _WORLD):
        data = _load(3, rank, device)
        runtime = _runtime(data)
        attached = attach_trellismx_p8_runtime(
            _prepare(data),
            runtime,
            backend=in_tree_p8_backend(),
            options=SERVING_RESOLVED,
        )
        verify_trellismx_p8_attachment(attached)
        ranks.append(
            SimpleNamespace(
                rank=rank, data=data, runtime=runtime, attached=attached, device=device
            )
        )
    return ranks


def test_the_p8_race_key_is_the_same_on_every_tensor_parallel_rank(gpu_rank_layer3):
    """Four rank runtimes of one layer declare one program set and one race key.

    Each runtime is built from its own rank's sidecar slice, so the four differ
    in everything a rank selects. The kernels they launch are the same, so a
    real preparation session must derive one race key for all four; otherwise
    the ranks cannot agree on a winner and the collective preparation fails.
    """

    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession
    from b12x.preparation.session import _coalesce_requests

    keys = {}
    for entry in gpu_rank_layer3:
        plan = moe.plan_execution(
            experts=entry.attached,
            capacity=moe.ExecutionCapacity(
                max_tokens=17, top_k=_TOP_K, route_num_experts=0
            ),
        )
        with PreparationSession(
            device=entry.device, autotune=True, compile_workers=0
        ) as session:
            request = plan.request(name="moe.decode", prepare_call=lambda state: None)
            job = session.begin((request,))
            (obligation,) = _finish(job._configure(_coalesce_requests((request,))))
            _finish(job._describe_candidates(obligation))
            keys[entry.rank] = (
                job._choice_key(obligation, {}),
                obligation.candidate_program_identity,
            )
    assert len({key for key, _ in keys.values()}) == 1, keys
    assert len({identity for _, identity in keys.values()}) == 1, keys


def test_every_rank_runtime_installs_and_runs_its_arms(gpu_rank_layer3) -> None:
    """Each rank's declared programs install, and both arms serve its slice."""

    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    for entry in gpu_rank_layer3:
        with torch.cuda.device(entry.device):
            entry.runtime.install_compiled_programs(
                compile_p8_programs(
                    entry.runtime.program_spec, 17, entry.device.index or 0
                ),
                17,
            )
        for rows in (1, 17):
            x, ids, weights = (
                t.to(entry.device)
                for t in _inputs(3, entry.rank, rows, torch.device("cpu"))
            )
            ids = ids.to(torch.int32)
            produced = _run_runtime(entry.runtime, x, weights, ids, rows, entry.device)
            torch.cuda.synchronize()
            assert torch.isfinite(produced.float()).all(), (entry.rank, rows)
            meta = {
                "hidden_size": _HIDDEN,
                "intermediate_size": _LOCAL,
                "num_experts": _EXPERTS,
                "top_k": _TOP_K,
                "bits": 4,
                "rank": entry.rank,
                "world_size": _WORLD,
                "sign_draw": 0,
                "quantize_activations": rows
                > SERVING_RESOLVED["rp2_direct_max_tokens"],
            }
            reference = trellismx_moe_reference(
                x.cpu(), ids.cpu(), weights.cpu(), dict(entry.data.host), meta
            ).double()
            cosine = torch.nn.functional.cosine_similarity(
                produced.double().cpu().flatten(), reference.flatten(), dim=0
            ).item()
            assert cosine >= 0.9999, (entry.rank, rows, cosine)


@pytest.mark.parametrize("capacity", [1, 2, 16, 17, 129, 4096])
def test_a_capacity_declares_the_programs_of_every_call_it_accepts(
    gpu_layer3, capacity
) -> None:
    """b12x describes the compile job without lowering. Every call the capacity
    accepts launches only declared programs, and the declared arms' calls launch
    all of them."""

    from b12x._lib.compile_plan import observe_programs
    from b12x._lib.compile_pool import describe_compilation

    (job,) = trellismx_p8_attachment(gpu_layer3.attached).compile_jobs(capacity)
    declared = set(describe_compilation(job).programs)
    device = gpu_layer3.device
    x, ids, weights = (
        t.to(device) for t in _inputs(3, 0, capacity, torch.device("cpu"))
    )
    ids = ids.to(torch.int32)
    spec = gpu_layer3.runtime.program_spec
    calls = {*p8_capacity_launch_rows(spec, capacity), capacity}
    calls |= {rows for rows in (1, 2, 16, 17, 128, 129) if rows <= capacity}
    launched_by_calls = set()
    for rows in sorted(calls):
        out = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        with observe_programs() as launched:
            gpu_layer3.runtime(
                x[:rows],
                weights[:rows],
                ids[:rows],
                out,
                scratch=_arena(gpu_layer3.runtime, capacity, device),
            )
        torch.cuda.synchronize()
        assert launched and launched <= declared, rows
        launched_by_calls |= launched
    assert launched_by_calls == declared


def test_one_capacity_serves_every_smaller_call_after_freezing(gpu_layer3) -> None:
    """Experts prepared for one 4096-row capacity, with no warmup counts, serve calls
    of any size up to it after kernel resolution is frozen, through fused_moe
    dispatch, and match the runtime's own result bit for bit."""

    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    capacity = 4096
    device = gpu_layer3.device
    x, ids, weights = (
        t.to(device) for t in _inputs(3, 0, capacity, torch.device("cpu"))
    )
    ids = ids.to(torch.int32)
    plan = moe.plan_execution(
        experts=gpu_layer3.attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=capacity, top_k=_TOP_K, route_num_experts=0
        ),
    )

    def bind(target, rows, specs):
        output = torch.full(
            (rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        )
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in specs
        )
        binding = target(
            a=x[:rows],
            topk_ids=ids[:rows],
            topk_weights=weights[:rows],
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return binding, output, scratch

    def call(state):
        binding, output, scratch = bind(
            state.bind, capacity, state.scratch.scratch_specs()
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=_producer(x),
            benchmark_producers=(_producer(x),),
            owners=(binding, scratch),
        )

    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        session.prepare(
            (
                plan.request(
                    name="trellismx-p8-capacity",
                    prepare_call=call,
                    benchmark_call=call,
                ),
            )
        )
        session.freeze()
        selected = plan.selection.config
        for rows in (1, 2, 16, 17, 128, 129, capacity):
            binding, output, _ = bind(
                functools.partial(moe.bind, plan), rows, plan.scratch_specs()
            )
            binding.run()
            torch.cuda.synchronize()
            expected = _run_runtime(
                gpu_layer3.runtime,
                x[:rows],
                weights[:rows],
                ids[:rows],
                rows,
                device,
                selected,
            )
            assert torch.equal(expected.view(torch.int16), output.view(torch.int16)), (
                rows
            )


def test_worker_compiled_programs_are_installed_before_freeze(
    gpu_layer3, tmp_path
) -> None:
    """A compile worker's artifacts must populate the runtime that will serve."""

    from b12x.moe import fused_moe as moe
    from b12x._lib.compiler import clear_compile_cache
    from b12x.preparation import PreparationSession, PreparedCall

    capacity = 4096
    device = gpu_layer3.device
    clear_compile_cache()
    runtime = _runtime(gpu_layer3.data)
    attached = attach_trellismx_p8_runtime(
        _prepare(gpu_layer3.data),
        runtime,
        backend=in_tree_p8_backend(),
        options=SERVING_RESOLVED,
    )
    x, ids, weights = (
        tensor.to(device) for tensor in _inputs(3, 0, capacity, torch.device("cpu"))
    )
    ids = ids.to(torch.int32)
    plan = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=capacity,
            top_k=_TOP_K,
            route_num_experts=0,
            warmup_token_counts=(),
        ),
    )

    def bind(target, rows, specs):
        output = torch.full(
            (rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        )
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in specs
        )
        binding = target(
            a=x[:rows],
            topk_ids=ids[:rows],
            topk_weights=weights[:rows],
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return binding, output, scratch

    def call(state):
        binding, output, scratch = bind(
            state.bind, capacity, state.scratch.scratch_specs()
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=_producer(x),
            benchmark_producers=(_producer(x),),
            owners=(binding, scratch),
        )

    with PreparationSession(
        device=device,
        autotune=True,
        compile_workers=1,
        cache_dir=str(tmp_path / "programs"),
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="trellismx-p8-worker-capacity",
                    prepare_call=call,
                    benchmark_call=call,
                ),
            )
        )
        session.freeze()
        selected = plan.selection.config
        rows_to_check = (*range(1, 17), 17, 128, 129, 1024, capacity)
        for rows in rows_to_check:
            binding, output, _ = bind(
                functools.partial(moe.bind, plan), rows, plan.scratch_specs()
            )
            binding.run()
            torch.cuda.synchronize()
            expected = _run_runtime(
                gpu_layer3.runtime,
                x[:rows],
                weights[:rows],
                ids[:rows],
                rows,
                device,
                selected,
            )
            assert torch.equal(expected.view(torch.int16), output.view(torch.int16)), (
                rows
            )


def test_frozen_grouped_capacity_serves_every_output_dtype(
    gpu_layer3, tmp_path
) -> None:
    """A frozen grouped capacity serves, and its output stage matches the CPU
    reference bit for bit at every output dtype."""

    from b12x.moe import fused_moe as moe
    from b12x._lib.compiler import clear_compile_cache
    from b12x.preparation import PreparationSession, PreparedCall

    rows = 17
    device = gpu_layer3.device
    clear_compile_cache()
    trellismx_boundary_fused._LAUNCHERS.clear()
    prepared = _prepare(gpu_layer3.data)
    x, ids, weights = (
        tensor.to(device) for tensor in _inputs(3, 0, rows, torch.device("cpu"))
    )
    ids = ids.to(torch.int32)
    plan = moe.plan_execution(
        experts=prepared,
        capacity=moe.ExecutionCapacity(
            max_tokens=rows,
            top_k=_TOP_K,
            route_num_experts=0,
            warmup_token_counts=(),
        ),
    )
    output = torch.full(
        (rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )

    def call(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        binding = state.bind(
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=_producer(x),
            benchmark_producers=(_producer(x),),
            owners=(binding, *scratch),
        )

    with PreparationSession(
        device=device,
        autotune=True,
        compile_workers=1,
        cache_dir=str(tmp_path / "programs"),
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="trellismx-output-dtypes",
                    prepare_call=call,
                    benchmark_call=call,
                ),
            )
        )
        session.freeze()
        # The prepared call still serves after the freeze, allocating nothing
        # preparation did not declare.
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        binding = moe.bind(
            plan,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        binding.run()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(output.float()).all())
        svh = gpu_layer3.data.payload["down_svh_fp16"]
        generator = torch.Generator().manual_seed(20261030)
        reduced = (torch.randn(rows, _HIDDEN, generator=generator) * 0.5).to(
            torch.float32
        )
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            got = fused_output_stage(
                reduced.to(device).to(dtype),
                svh,
                hidden_size=_HIDDEN,
            )
            expected = fused_output_stage(
                reduced.to(dtype), svh.cpu(), hidden_size=_HIDDEN
            )
            torch.cuda.synchronize()
            assert got.dtype is dtype
            assert torch.equal(got.cpu().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 17, 4096])
def test_attached_experts_prepare_under_b12x_preparation(gpu_layer3, rows) -> None:
    """A tuning preparation session, the path a serving stack prepares through,
    accepts the attached experts' plan, and a graph captured after freezing
    replays the eager result bit for bit. (Without autotuning the session only
    warms heuristics and adopts whatever programs a call launches; with it, a
    launched program the plan did not declare fails preparation.)"""

    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    device = gpu_layer3.device
    x, ids, weights = (t.to(device) for t in _inputs(3, 0, rows, torch.device("cpu")))
    ids = ids.to(torch.int32)
    plan = moe.plan_execution(
        experts=gpu_layer3.attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=rows, top_k=_TOP_K, route_num_experts=0
        ),
    )

    def bind(target, specs):
        output = torch.full(
            (rows, _HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
        )
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in specs
        )
        binding = target(
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return binding, output, scratch

    def call(state):
        binding, output, scratch = bind(state.bind, state.scratch.scratch_specs())
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=_producer(x),
            benchmark_producers=(_producer(x),),
            owners=(binding, scratch),
        )

    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        session.prepare(
            (
                plan.request(
                    name=f"trellismx-p8-{rows}",
                    prepare_call=call,
                    benchmark_call=call,
                ),
            )
        )
        # The race fixes the launch configuration; the serving runtime built
        # from the same sidecar must reproduce the selected one bit for bit.
        selected = plan.selection.config
        expected = _run_runtime(
            gpu_layer3.runtime,
            x,
            weights,
            ids,
            rows,
            device,
            selected,
        )
        binding, output, _ = bind(
            functools.partial(moe.bind, plan), plan.scratch_specs()
        )
        binding.run()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.run()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.int16), output.view(torch.int16))


# --------------------------------------------------------------------------- #
# caller-owned scratch and the launch-configuration race
# --------------------------------------------------------------------------- #


class _AllocationRecorder:
    """Record every tensor factory call made while it is active."""

    FACTORIES = ("zeros", "ones", "empty", "full")

    def __init__(self, monkeypatch):
        self.calls = []
        self._monkeypatch = monkeypatch

    def __enter__(self):
        for name in self.FACTORIES:
            original = getattr(torch, name)

            def record(*args, _name=name, _original=original, **kwargs):
                self.calls.append((_name, args[:2]))
                return _original(*args, **kwargs)

            self._monkeypatch.setattr(torch, name, record)
        return self

    def __exit__(self, *exception):
        for name in self.FACTORIES:
            self._monkeypatch.undo()
        return False


def test_a_captured_p8_call_allocates_nothing(gpu_layer3, monkeypatch) -> None:
    """The runtime carves the caller's arena, so capture allocates no tensor.

    A prepared call writes the caller's output out of a caller-owned arena:
    recording every ``torch.zeros``/``ones``/``empty``/``full`` made while the
    call is captured finds none, and the replayed graph reproduces the eager
    result bit for bit.
    """

    from b12x.moe import fused_moe as moe

    rows = 129
    device = gpu_layer3.device
    x, ids, weights = (t.to(device) for t in _inputs(3, 0, rows, torch.device("cpu")))
    ids = ids.to(torch.int32)
    plan = moe.plan_execution(
        experts=gpu_layer3.attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=rows, top_k=_TOP_K, route_num_experts=0
        ),
    )
    output = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in plan.scratch_specs()
    )
    binding = moe.bind(
        plan,
        a=x,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
        scratch=scratch,
        input_scales_static=True,
    )
    # Eager first use, outside capture: the reference the replay must match.
    binding.run()
    torch.cuda.synchronize()
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    recorder = _AllocationRecorder(monkeypatch)
    with recorder:
        with torch.cuda.graph(graph):
            binding.run()
    assert recorder.calls == []
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int16), output.view(torch.int16))


def test_a_p8_call_refuses_a_missing_short_or_misaligned_arena(gpu_layer3) -> None:
    """The runtime never allocates: an absent, too-small or misaligned arena raises.

    The output is the caller's: a call without one raises, and a served call
    writes the tensor it was handed rather than a region of its arena.
    """

    rows = 129
    device = gpu_layer3.device
    x, ids, weights = (t.to(device) for t in _inputs(3, 0, rows, torch.device("cpu")))
    ids = ids.to(torch.int32)
    needed = gpu_layer3.runtime.scratch_specs(device, rows)[0].shape[0]
    assert needed > 256
    out = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="scratch from the caller"):
        gpu_layer3.runtime(x, weights, ids, out)
    with pytest.raises(ValueError, match="requires"):
        gpu_layer3.runtime(
            x,
            weights,
            ids,
            out,
            scratch=(torch.empty(256, dtype=torch.uint8, device=device),),
        )
    # A contiguous one-byte-offset slice of a large enough buffer is refused:
    # the launch would hand every region a misaligned pointer.
    base = torch.empty(needed + 16, dtype=torch.uint8, device=device)
    shift = (16 - base.data_ptr() % 16) % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        gpu_layer3.runtime(x, weights, ids, out, scratch=(base[shift + 1 :][:needed],))
    # The declared arena serves the call, and the result is the caller's output.
    arena = torch.empty(needed, dtype=torch.uint8, device=device)
    result = gpu_layer3.runtime(x, weights, ids, out, scratch=(arena,))
    torch.cuda.synchronize()
    assert result is out
    assert bool(torch.isfinite(result.float()).all())
    # The arena holds no result: a call without an output is refused.
    with pytest.raises(RuntimeError, match="caller-owned output"):
        gpu_layer3.runtime(x, weights, ids, None, scratch=(arena,))


def _grid_cap_ladder(sm_count: int) -> list[int]:
    """The resident-grid caps the P8 race may try, as the component declares them.

    The serving policy (``0``), the powers of two up to the residency bound, and
    the bound's halves, three quarters and full value.
    """

    rungs = {1 << exponent for exponent in range(sm_count.bit_length())}
    rungs |= {
        max(1, sm_count // 2),
        max(1, 3 * sm_count // 4),
        max(1, 7 * sm_count // 8),
        sm_count,
    }
    return sorted({0, *(rung for rung in rungs if rung <= sm_count)})


def test_the_autotuner_races_the_p8_launch_configurations(gpu_layer3, tmp_path) -> None:
    """The P8 component's candidates are measured, and every one of them serves.

    The capacity's parameter space is exactly the direct arm's geometry
    (two direct tiles, two FC1 pipeline depths, two warp counts), the serving
    carrier stride, and the resident-grid axis: the serving policy and the full
    residency bound at the default effort, every rung between them at the audit
    effort. The session benchmarks that set, the winner is one of them, and each
    candidate is bit-identical to the standalone runtime at the same
    configuration and reproduces the reference in its activation regime. The
    race is read back from b12x's own ledger and selection record.
    """

    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.p8.launch import P8LaunchConfig
    from b12x.preparation import PreparationSession, PreparedCall, autotune_effort
    from b12x.preparation.device import detect_device
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference
    from b12x.tools.tuning_ledger import load_records as load_ledger_records

    cache = tmp_path / "tuning"

    rows = 16
    device = gpu_layer3.device
    identity = detect_device(device).identity
    ladder = _grid_cap_ladder(identity.sm_count)
    top = ladder[-1]
    x_cpu, ids_cpu, weights_cpu = _inputs(3, 0, rows, torch.device("cpu"))
    x, ids, weights = (t.to(device) for t in (x_cpu, ids_cpu, weights_cpu))
    ids = ids.to(torch.int32)

    def capacity():
        return moe.ExecutionCapacity(
            max_tokens=rows,
            top_k=_TOP_K,
            route_num_experts=0,
            warmup_token_counts=(),
        )

    plan = moe.plan_execution(experts=gpu_layer3.attached, capacity=capacity())
    candidates = [config for _, config in plan.contract.choices(plan.query, identity)]
    # The direct arm's 16-row call reaches 128 routed rows, so the 128-row
    # carrier stride fails the correctness predicate and only 192 survives.
    expected = {
        P8LaunchConfig(
            direct_tile_m=tile,
            fc1_pipeline_stages=stages,
            fc1_warps=warps,
            rp2_stride=192,
            mac=mac,
        )
        for tile in (0, 32)
        for stages in (2, 3)
        for warps in (4, 8)
        for mac in (0, top)
    }
    assert len(expected) == 16
    assert len(candidates) == 16
    assert set(candidates) == expected
    assert P8LaunchConfig.serving() in candidates
    # The audit effort races every rung of the resident-grid axis.
    with autotune_effort("audit"):
        audit_plan = moe.plan_execution(
            experts=gpu_layer3.attached, capacity=capacity()
        )
    audit = [
        config for _, config in audit_plan.contract.choices(audit_plan.query, identity)
    ]
    assert len(audit) == 8 * len(ladder)
    assert {config.mac for config in audit} == set(ladder)
    assert set(audit) == {
        P8LaunchConfig(
            direct_tile_m=tile,
            fc1_pipeline_stages=stages,
            fc1_warps=warps,
            rp2_stride=192,
            mac=mac,
        )
        for tile in (0, 32)
        for stages in (2, 3)
        for warps in (4, 8)
        for mac in ladder
    }

    def call(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        binding = state.bind(
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device),
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=binding.output,
            produce=_producer(x),
            benchmark_producers=(_producer(x),),
            owners=(binding, *scratch),
        )

    # Every candidate serves the capacity and holds the direct arm's reference
    # (unquantized activations restore BF16-level precision).
    meta = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _LOCAL,
        "num_experts": _EXPERTS,
        "top_k": _TOP_K,
        "bits": 4,
        "rank": 0,
        "world_size": _WORLD,
        "sign_draw": 0,
        "quantize_activations": False,
    }
    reference = trellismx_moe_reference(
        x_cpu, ids_cpu, weights_cpu, dict(gpu_layer3.data.host), meta
    ).double()
    for config in candidates:
        pinned = moe.plan_execution(
            experts=gpu_layer3.attached,
            capacity=moe.ExecutionCapacity(
                max_tokens=rows,
                top_k=_TOP_K,
                route_num_experts=0,
                warmup_token_counts=(),
            ),
            override=config,
        )
        output = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in pinned.scratch_specs()
        )
        binding = moe.bind(
            pinned,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        moe.run(binding=binding)
        torch.cuda.synchronize()
        assert pinned.selection.config == config
        direct = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        gpu_layer3.runtime(
            x,
            weights,
            ids,
            direct,
            scratch=_arena(gpu_layer3.runtime, rows, device, config),
            config=config,
        )
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), direct.view(torch.int16))
        cosine = torch.nn.functional.cosine_similarity(
            output.double().cpu().flatten(), reference.flatten(), dim=0
        ).item()
        assert cosine >= 0.9999, (config, cosine)

    # The grouped capacity's candidates hold its own contract: the quantized
    # activation regime's 0.9998 bound.
    grouped_rows = 17
    grouped_x, grouped_ids, grouped_weights = (
        tensor.to(device) for tensor in _inputs(3, 0, grouped_rows, torch.device("cpu"))
    )
    grouped_ids = grouped_ids.to(torch.int32)
    grouped_plan = moe.plan_execution(
        experts=gpu_layer3.attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=grouped_rows,
            top_k=_TOP_K,
            route_num_experts=0,
            warmup_token_counts=(),
        ),
    )
    grouped_candidates = [
        config
        for _, config in grouped_plan.contract.choices(grouped_plan.query, identity)
    ]
    # A 17-row call lands on the grouped arm, so only that arm's tile races:
    # the policy (0) against the tile the kernels accept for this row band.
    assert set(grouped_candidates) == {
        P8LaunchConfig(grouped_tile_m=tile, mac=mac)
        for tile in (0, 64)
        for mac in (0, top)
    }
    assert len(grouped_candidates) == 4
    grouped_reference = trellismx_moe_reference(
        grouped_x.cpu(),
        grouped_ids.cpu(),
        grouped_weights.cpu(),
        dict(gpu_layer3.data.host),
        dict(meta, quantize_activations=True),
    ).double()
    for config in grouped_candidates:
        pinned = moe.plan_execution(
            experts=gpu_layer3.attached,
            capacity=moe.ExecutionCapacity(
                max_tokens=grouped_rows,
                top_k=_TOP_K,
                route_num_experts=0,
                warmup_token_counts=(),
            ),
            override=config,
        )
        output = torch.empty(grouped_rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in pinned.scratch_specs()
        )
        binding = moe.bind(
            pinned,
            a=grouped_x,
            topk_ids=grouped_ids,
            topk_weights=grouped_weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        moe.run(binding=binding)
        torch.cuda.synchronize()
        assert pinned.selection.config == config
        # The same configuration through the standalone runtime must produce
        # the same bytes, as it does for the direct arm.
        direct = torch.empty(grouped_rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        gpu_layer3.runtime(
            grouped_x,
            grouped_weights,
            grouped_ids,
            direct,
            scratch=_arena(gpu_layer3.runtime, grouped_rows, device, config),
            config=config,
        )
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), direct.view(torch.int16))
        cosine = torch.nn.functional.cosine_similarity(
            output.double().cpu().flatten(), grouped_reference.flatten(), dim=0
        ).item()
        assert cosine >= 0.9998, (config, cosine)

    # The session's own selection cache keeps this race cold: a shared cache
    # would restore a recorded winner and measure nothing.
    with PreparationSession(
        device=device, autotune=True, compile_workers=0, cache_dir=cache
    ) as session:
        # The subject is the race itself, so an environment that turns the
        # autotuner off is a precondition failure rather than a pass.
        assert session.autotune, (
            "the P8 launch race needs an enabled autotune tier; "
            "B12X_AUTOTUNE=0 or B12X_AUTOTUNE_EFFORT=off disables it"
        )
        result = session.prepare(
            (
                plan.request(
                    name="trellismx-p8-launch-race",
                    prepare_call=call,
                    benchmark_call=call,
                ),
            )
        )
        assert result.cache_hits == 0
        assert result.restored_candidates == 0
        assert result.benchmarked_candidates >= len(candidates), (
            f"the session measured {result.benchmarked_candidates} of "
            f"{len(candidates)} P8 candidates (cache hits {result.cache_hits}, "
            f"restored {result.restored_candidates}); a race needs a cold cache "
            "and an enabled tier (B12X_AUTOTUNE, B12X_AUTOTUNE_EFFORT)"
        )
        assert plan.selection.config in candidates
        # b12x's own record of the race: every candidate was measured, the
        # winner is one of them, and the installed selection is the winner.
        (record,) = [
            item
            for item in load_ledger_records(cache)
            if item["request"] == "trellismx-p8-launch-race"
        ]
        assert record["effort"] != "off"
        assert record["rejected"] == []
        assert len(record["candidates"]) == len(candidates)
        assert record["coverage"]["measured_count"] == len(candidates)
        # The space the race enumerated: every geometry combination of the
        # direct arm, both carrier strides before the correctness predicate,
        # and every rung of the resident-grid axis. The default effort keeps
        # two rungs (the policy and the bound); the audit effort admits all.
        assert record["coverage"]["cartesian_count"] == 16 * len(ladder)
        assert record["coverage"]["legal_count"] == 16
        assert record["coverage"]["effective_count"] == 16
        assert record["coverage"]["legal_by_effort"] == {
            "default": 16,
            "audit": 8 * len(ladder),
        }
        assert {item["admission"] for item in record["candidates"]} == {"default"}
        assert {item["config"]["mac"] for item in record["candidates"]} == {0, top}
        measured = [
            item for item in record["candidates"] if item["latency_us"] is not None
        ]
        assert len(measured) == len(candidates)
        assert all(item["rounds"] for item in measured)
        champion = record["champion"]["index"]
        assert champion in {item["index"] for item in measured}
        assert record["champion"]["latency_us"] == min(
            item["latency_us"] for item in measured
        )
        payload = plan.contract.config_payload(plan.selection.config)
        assert record["candidates"][champion]["config"] == payload.to_dict()
        selected = record["_ledger"]["selected"]
        assert selected["config"] == payload.to_dict()
        assert selected["coverage"]["measured_count"] == len(candidates)
        session.freeze()


def test_every_one_row_launch_configuration_serves(gpu_layer3) -> None:
    """At one row every direct launch configuration serves the serving bytes.

    The RP2 carrier stages each direct row's residual plane a fixed distance
    after its high plane, and a one-row launch is the only row count whose
    tiles cover fewer carrier rows than that distance. Every tile and stride
    the component races is therefore served here, and each must reproduce the
    serving configuration's bytes instead of reading past the input carrier.
    """

    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.p8.launch import P8LaunchConfig

    rows = 1
    device = gpu_layer3.device
    x_cpu, ids_cpu, weights_cpu = _inputs(3, 0, rows, torch.device("cpu"))
    x, ids, weights = (t.to(device) for t in (x_cpu, ids_cpu, weights_cpu))
    ids = ids.to(torch.int32)

    def served(config):
        plan = moe.plan_execution(
            experts=gpu_layer3.attached,
            capacity=moe.ExecutionCapacity(
                max_tokens=rows,
                top_k=_TOP_K,
                route_num_experts=0,
                warmup_token_counts=(),
            ),
            override=config,
        )
        output = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=device)
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        binding = moe.bind(
            plan,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        moe.run(binding=binding)
        torch.cuda.synchronize()
        return output

    serving = served(P8LaunchConfig.serving())
    assert bool(torch.isfinite(serving.float()).all())
    for direct_tile_m in (0, 32):
        for rp2_stride in (128, 192):
            config = P8LaunchConfig(direct_tile_m=direct_tile_m, rp2_stride=rp2_stride)
            output = served(config)
            assert bool(torch.isfinite(output.float()).all()), config
            assert torch.equal(output.view(torch.int16), serving.view(torch.int16)), (
                config
            )
