"""TrellisMX P8 runtime attachment, small-M ownership and execution contracts.

CPU-only. The P8 runtime is a test double that exposes the attributes the
attachment reads from ``P8NativeTPMoE`` (``fused_moe.trellismx_p8``); its module
lives in a temporary package so the backend pin covers a real file.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import hashlib
import importlib
import os
import pickle
import sys
import textwrap
import weakref
from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe import _impl, trellismx, trellismx_p8
from b12x.moe.fused_moe.trellismx_p8 import (
    P8_OPTION_ATTRIBUTES,
    TrellisMxP8Attachment,
    TrellisMxP8Backend,
    TrellisMxP8Descriptor,
    TrellisMxP8IncompatibleError,
    TrellisMxP8Options,
    TrellisMxP8Unavailable,
    attach_trellismx_p8_runtime,
    detach_trellismx_p8_runtime,
    release_trellismx_execution_state,
    trellismx_p8_attachment,
    verify_trellismx_p8_attachment,
)

_E, _H, _I, _BITS, _TOP_K = 4, 512, 128, 4, 2

_RUNTIME_SOURCE = textwrap.dedent(
    '''
    import torch

    from b12x._lib.scratch import scratch_buffer_spec
    from b12x.moe._shared.p8.plan import P8ProgramSpec, launch_spec, p8_capacity_scratch_nbytes


    class FakeP8Runtime:
        """Exposes the P8NativeTPMoE attributes the b12x attachment verifies."""

        def __init__(self, tensors, record, *, top_k, **overrides):
            local = int(record.intermediate_size)
            rotations = record.intermediate_rotations
            self.device = record.w13.device
            self.experts = int(record.num_experts)
            self.hidden = int(record.hidden_size)
            self.intermediate = local
            self.topk = int(top_k)
            self.trellis_bits = int(record.trellis_bits)
            self.swiglu_limit = 10.0
            self.full_coupled = True
            self.deterministic_output = True
            self.debug_capture = False
            self.compact_scale_storage = False
            # Resolved behavior settings (P8_OPTION_ATTRIBUTES), as the GLM recipe sets them.
            self.p8_down_remainder = False
            self._dx2_phases = "both"
            self.p8_dx2_rowpack = True
            self.p8_input_rowpack = True
            self.rp2_direct_max_tokens = 16
            self.rp2_direct_tile_m = 0
            self.rp2_skip_input_clear = True
            self.p8_epi_par = True
            self.small_m_scheduler = True
            self.fc1_tile_n = 128
            self.fc1_pipeline_stages = 2
            self.fc1_warps = 4
            self.fc1_warp_quant = False
            self.fc1_broadcast_a = True
            self.fc1_a_swizzle_rotate = False
            self.fc1_exact_staging = False
            self.prefill_chunk_tokens = 0
            self.grid_policy = True
            self.tile_major_tasks = False
            self.compact_input_storage = False
            self.force_materialized = None
            self.mac_override = None
            # In place, as the runtime reads a resident payload.
            self.w13_stream = tensors["w13_trellis"].view(torch.int32).reshape(-1)
            self.w2_stream = tensors["w2_trellis"].view(torch.int32).reshape(-1)
            self.w13_scale_mx = tensors["w13_scale_ue8m0"].reshape(-1)
            self.w2_scale_mx = tensors["w2_scale_ue8m0"].reshape(-1)
            # Private copies, as the runtime repacks its own.
            self.w13_sfb = record.w13_sfb.clone().reshape(-1)
            self.w2_sfb = record.w2_sfb.clone().reshape(-1)
            self.scale_component_packed = torch.cat(
                (
                    tensors["gate_up_suh_fp16"].reshape(-1),
                    tensors["intermediate_scales_fp16"].reshape(-1),
                    tensors["down_svh_fp16"].reshape(-1),
                    rotations[0, 3 * local :],
                )
            )
            self.calls = []
            self.installed = []
            for name, value in overrides.items():
                setattr(self, name, value)
            # The declaration the plan sizes its arena and compile jobs from.
            self.program_spec = P8ProgramSpec(
                trellis_bits=int(record.trellis_bits),
                experts=int(record.num_experts),
                hidden=int(record.hidden_size),
                intermediate=local,
                topk=int(top_k),
                swiglu_limit=10.0,
                full_coupled=True,
                scale_sandwich=True,
                mac_override=self.mac_override,
                deterministic_output=True,
                small_m_scheduler=True,
                force_materialized=None,
                rp2_direct_max_tokens=int(self.rp2_direct_max_tokens),
                rp2_skip_input_clear=True,
                prefill_chunk_tokens=0,
                compact_input_storage=False,
                p8_down_remainder=False,
                dx2_phases="both",
                p8_dx2_rowpack=True,
                p8_input_rowpack=True,
                p8_epi_par=True,
                tile_major_tasks=False,
                fc1_tile_n=128,
                rp2_direct_tile_m=0,
                grouped_tile_m=0,
                rp2_stride=192,
                fc1_warps=4,
                fc1_pipeline_stages=2,
                fc1_a_swizzle_rotate=False,
                fc1_warp_quant=False,
                fc1_exact_staging=False,
                fc1_broadcast_a=True,
            )
            self.program_declaration = (
                "fakep8_pkg.runtime:compile_fake_p8_programs",
                self.program_spec,
            )

        def install_compiled_programs(self, programs, tokens, config=None):
            self.installed.append((programs, int(tokens), config))

        def scratch_specs(self, device, tokens, config=None):
            return (
                scratch_buffer_spec(
                    "trellismx_p8",
                    nbytes=p8_capacity_scratch_nbytes(
                        launch_spec(self.program_spec, config), int(tokens)
                    ),
                    device=torch.device(device),
                ),
            )

        def __call__(
            self, x, topk_weights, topk_ids, out=None, *, scratch=None, config=None
        ):
            self.calls.append((x, topk_weights, topk_ids, out))
            out.fill_(1.5)
            return out


    def compile_fake_p8_programs(spec, tokens, ordinal):
        """The double compiles elsewhere; preparation only declares the job."""
        return None
    '''
)


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    package = tmp_path / "fakep8_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "runtime.py").write_text(_RUNTIME_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("fakep8_pkg.runtime")
    digest = hashlib.sha256((package / "runtime.py").read_bytes()).hexdigest()
    backend = TrellisMxP8Backend(
        package="fakep8_pkg", files={"runtime.py": digest}, upstream="test double"
    )
    yield SimpleNamespace(
        backend=backend, runtime_class=module.FakeP8Runtime, root=package
    )
    for name in [name for name in sys.modules if name.startswith("fakep8_pkg")]:
        del sys.modules[name]


def _sidecar(*, seed=11, scale_bytes=(127, 127), down_svh=None, signs=0):
    generator = torch.Generator().manual_seed(seed)
    w13_byte, w2_byte = scale_bytes
    tensors = {
        "w13_trellis": torch.randint(
            0,
            1 << 15,
            (2, _E, _H // 16, _I // 16, 16 * _BITS),
            generator=generator,
            dtype=torch.int16,
        ),
        "w2_trellis": torch.randint(
            0,
            1 << 15,
            (_E, _I // 16, _H // 16, 16 * _BITS),
            generator=generator,
            dtype=torch.int16,
        ),
        "w13_scale_ue8m0": torch.full(
            (_E, 2 * _I, _H // 32), w13_byte, dtype=torch.uint8
        ),
        "w2_scale_ue8m0": torch.full((_E, _H, _I // 32), w2_byte, dtype=torch.uint8),
        "intermediate_scales_fp16": torch.randint(
            0, 2, (_E, 3 * _I), generator=generator
        )
        .mul_(2)
        .sub_(1)
        .to(torch.float16),
        "coupled_sign_draw_u8": torch.full((_E,), signs, dtype=torch.uint8),
        "gate_up_suh_fp16": torch.ones(_H, dtype=torch.float16),
        "down_svh_fp16": (
            torch.ones(_H, dtype=torch.float16) if down_svh is None else down_svh
        ),
    }
    metadata = {
        "law": "procedural-npk",
        "compander_scale": "2.0",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "boundary": "coupled-h512-h128-suh-svh-v1",
        "activation": "silu-cap10",
        "bits": str(_BITS),
        "hidden": str(_H),
        "experts_start": "0",
        "experts_end": str(_E),
        "intermediate_per_rank": str(_I),
        "topk": str(_TOP_K),
        "world_size": "1",
        "slice_begin": "0",
        "slice_end": str(_I),
    }
    return tensors, metadata


def _prepare(tensors, metadata):
    return trellismx.prepare_trellismx_experts(
        tensors,
        metadata,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size=_I,
        top_k=_TOP_K,
        device=torch.device("cpu"),
    )


def _record(prepared):
    return prepared._impl.representation.value


def _runtime(fake_backend, tensors, prepared, **overrides):
    return fake_backend.runtime_class(
        tensors, _record(prepared), top_k=_TOP_K, **overrides
    )


def _attached(fake_backend, seed=11):
    tensors, metadata = _sidecar(seed=seed)
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    return (
        tensors,
        prepared,
        runtime,
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend),
    )


def _attached_with(fake_backend, *, seed=11, **overrides):
    """``_attached`` with runtime constructor overrides."""

    tensors, metadata = _sidecar(seed=seed)
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared, **overrides)
    return (
        tensors,
        prepared,
        runtime,
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend),
    )


def _runtime_from_record(fake_backend, prepared, **overrides):
    """A runtime double reading ``prepared``'s own tensors (no sidecar at hand)."""

    record = _record(prepared)
    local = int(record.intermediate_size)
    tensors = {
        "w13_trellis": record.w13,
        "w2_trellis": record.w2,
        "w13_scale_ue8m0": record.w13_scale,
        "w2_scale_ue8m0": record.w2_scale,
        "gate_up_suh_fp16": record.gate_suh.reshape(-1),
        "intermediate_scales_fp16": record.intermediate_rotations[:, : 3 * local],
        "down_svh_fp16": record.down_svh.reshape(-1),
    }
    return fake_backend.runtime_class(tensors, record, top_k=_TOP_K, **overrides)


def _binding(experts, *, m=3, top_k=_TOP_K, **overrides):
    a = torch.randn(m, _H).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(_E)[:top_k] for _ in range(m)]).to(torch.int32)
    weights = torch.full((m, top_k), 1.0 / top_k)
    values = dict(
        a=a,
        experts=experts,
        topk_weights=weights,
        topk_ids=ids,
        implementation="dynamic",
        state_E=_E,
        weight_E=_E,
        max_rows=16 * top_k,
        k=_H,
        n=_I,
        num_topk=top_k,
        device=a.device,
        dtype=torch.bfloat16,
        output=torch.zeros(m, _H, dtype=torch.bfloat16),
        quant_mode="w4a8_mx",
        swiglu_limit=10.0,
        routed_rows_capacity=16 * top_k,
        compiler_scratch=torch.empty(1 << 16, dtype=torch.uint8),
    )
    values.update(overrides)
    return _impl.TPMoEFP4Binding(**values)


# --------------------------------------------------------------------------- #
# attach: ownership and the complete identity
# --------------------------------------------------------------------------- #


def test_the_address_keyed_registry_is_gone() -> None:
    for name in (
        "register_trellismx_p8_runtime",
        "_trellismx_p8_runtime",
        "_TRELLISMX_P8_RUNTIMES",
        "_TRELLISMX_SMALLM_PLANS",
    ):
        assert not hasattr(_impl, name), name


def test_attach_returns_a_new_bundle_that_owns_the_runtime(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    assert attached is not prepared
    assert trellismx_p8_attachment(prepared) is None
    attachment = trellismx_p8_attachment(attached)
    assert attachment.runtime is runtime
    assert attachment.backend.identity == fake_backend.backend.identity
    assert attachment.contract.top_k == _TOP_K and attachment.contract.bits == _BITS
    assert trellismx_p8_attachment(attached._impl) is attachment
    with pytest.raises(TrellisMxP8IncompatibleError, match="already owns"):
        attach_trellismx_p8_runtime(attached, runtime, backend=fake_backend.backend)
    assert trellismx_p8_attachment(detach_trellismx_p8_runtime(attached)) is None


@pytest.mark.parametrize(
    "changed",
    [
        "w2_trellis",
        "w13_scale_ue8m0",
        "w2_scale_ue8m0",
        "down_svh_fp16",
        "intermediate_scales_fp16",
    ],
)
def test_shared_w13_with_a_different_bundle_is_rejected(fake_backend, changed) -> None:
    """A runtime verified for one bundle never serves another that shares W13."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    other = dict(tensors)
    if changed == "w2_trellis":
        other[changed] = tensors[changed] ^ 1
    elif changed.endswith("ue8m0"):
        other[changed] = torch.full_like(tensors[changed], 126)
    else:
        other[changed] = -tensors[changed]
    sibling = _prepare(other, metadata)
    assert _record(sibling).w13.untyped_storage().data_ptr() == (
        _record(prepared).w13.untyped_storage().data_ptr()
    ), "the two bundles must share the W13 carrier"
    with pytest.raises(TrellisMxP8IncompatibleError):
        attach_trellismx_p8_runtime(sibling, runtime, backend=fake_backend.backend)
    # The prepared bundle still attaches, and the sibling keeps its own arms.
    attached = attach_trellismx_p8_runtime(
        prepared, runtime, backend=fake_backend.backend
    )
    assert trellismx_p8_attachment(attached) is not None
    assert trellismx_p8_attachment(sibling) is None


def test_a_runtime_built_for_other_coupled_signs_is_rejected(fake_backend) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    runtime.scale_component_packed = runtime.scale_component_packed.clone()
    runtime.scale_component_packed[-1] = -runtime.scale_component_packed[-1]
    with pytest.raises(TrellisMxP8IncompatibleError, match="scale_component_packed"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"swiglu_limit": 7.0}, "10.0"),
        ({"trellis_bits": 3}, "trellis_bits"),
        ({"experts": _E + 1}, "experts"),
        ({"hidden": 2 * _H}, "hidden"),
        ({"full_coupled": False}, "full_coupled"),
        ({"deterministic_output": False}, "deterministic_output"),
        ({"debug_capture": True}, "debug_capture"),
        ({"compact_scale_storage": True}, "compact_scale_storage"),
        ({"device": torch.device("meta")}, "lives on"),
    ],
)
def test_incompatible_runtime_contracts_are_rejected(
    fake_backend, override, match
) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared, **override)
    with pytest.raises(TrellisMxP8IncompatibleError, match=match):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


@pytest.mark.parametrize(
    "name", ["w13_stream", "w2_stream", "w13_scale_mx", "w2_scale_mx"]
)
def test_a_runtime_that_copied_the_payload_is_rejected(fake_backend, name) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    setattr(runtime, name, getattr(runtime, name).clone())
    with pytest.raises(TrellisMxP8IncompatibleError, match=name):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


@pytest.mark.parametrize("name", ["w13_sfb", "w2_sfb"])
def test_a_runtime_with_other_repacked_scales_is_rejected(fake_backend, name) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    value = getattr(runtime, name).clone()
    value.view(torch.uint8)[0] ^= 1
    setattr(runtime, name, value)
    with pytest.raises(TrellisMxP8IncompatibleError, match=name):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


def _strided(tensor):
    storage = torch.zeros(2 * tensor.numel(), dtype=tensor.dtype)
    storage[::2].copy_(tensor.reshape(-1))
    return storage[::2]


def _misaligned(tensor):
    storage = torch.zeros(tensor.numel() + 1, dtype=tensor.dtype)
    storage[1:].copy_(tensor.reshape(-1))
    return storage[1:]


@pytest.mark.parametrize("name", trellismx_p8._RUNTIME_TENSORS)
@pytest.mark.parametrize(
    "layout,match", [(_strided, "contiguous"), (_misaligned, "16-byte aligned")]
)
def test_raw_pointer_tensors_need_their_layout(
    fake_backend, name, layout, match
) -> None:
    """Equal values in a strided or misaligned tensor are not what the kernels read."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    moved = layout(getattr(runtime, name))
    assert torch.equal(moved, getattr(runtime, name).reshape(-1))
    setattr(runtime, name, moved)
    with pytest.raises(TrellisMxP8IncompatibleError, match=match):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


def test_a_longer_packed_boundary_is_rejected(fake_backend) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    packed = runtime.scale_component_packed
    runtime.scale_component_packed = torch.cat((packed, packed[:8]))
    with pytest.raises(TrellisMxP8IncompatibleError, match="scale_component_packed"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


def test_other_source_formats_are_rejected(fake_backend) -> None:
    """A W4A8-MX bundle from a non-TrellisMX source never takes a P8 runtime."""

    plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout=moe.W13Layout.W13),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=_E, hidden_size=_H, intermediate_size=_I),
    )
    weights = moe.PackedWeights(
        w13=torch.zeros(_E, 2 * _I, _H // 2, dtype=torch.uint8),
        w2=torch.zeros(_E, _H, _I // 2, dtype=torch.uint8),
        w13_block_scales=torch.full((_E, 2 * _I, _H // 32), 127, dtype=torch.uint8),
        w2_block_scales=torch.full((_E, _H, _I // 32), 127, dtype=torch.uint8),
        w13_global_scales=torch.ones(_E),
        w2_global_scales=torch.ones(_E),
    )
    packed = moe.prepare_weights(plan=plan, weights=weights)
    assert packed._impl.source_format == "fp4_e8m0_k32"
    with pytest.raises(TrellisMxP8IncompatibleError, match="TrellisMX experts only"):
        attach_trellismx_p8_runtime(packed, object(), backend=fake_backend.backend)
    with pytest.raises(TypeError):
        attach_trellismx_p8_runtime(
            packed._impl, object(), backend=fake_backend.backend
        )


# --------------------------------------------------------------------------- #
# the backend pin
# --------------------------------------------------------------------------- #


def test_the_backend_pin_is_verified(fake_backend) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    wrong = TrellisMxP8Backend(package="fakep8_pkg", files={"runtime.py": "0" * 64})
    with pytest.raises(TrellisMxP8IncompatibleError, match="hashes to"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=wrong)
    (fake_backend.root / "other.py").write_text("")
    unpinned = TrellisMxP8Backend(
        package="fakep8_pkg",
        files={"other.py": hashlib.sha256(b"").hexdigest()},
    )
    with pytest.raises(TrellisMxP8IncompatibleError, match="not among the pinned"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=unpinned)
    missing = TrellisMxP8Backend(package="no_such_p8_backend", files={"x.py": "0" * 64})
    with pytest.raises(
        TrellisMxP8IncompatibleError, match="not importable|not an importable"
    ):
        attach_trellismx_p8_runtime(prepared, runtime, backend=missing)
    with pytest.raises(ValueError, match="package-relative"):
        TrellisMxP8Backend(package="fakep8_pkg", files={"../runtime.py": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256"):
        TrellisMxP8Backend(package="fakep8_pkg", files={"runtime.py": "abc"})
    assert (
        TrellisMxP8Backend(
            package="fakep8_pkg", files={"runtime.py": "1" * 64}
        ).identity
        != fake_backend.backend.identity
    )


def test_every_attach_hashes_the_backend_files(fake_backend) -> None:
    """A same-size replacement with the old modification time is still seen."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)
    path = fake_backend.root / "runtime.py"
    status = path.stat()
    data = path.read_bytes()
    at = data.index(b"Exposes")
    changed = data[:at] + b"exposes" + data[at + len(b"Exposes") :]
    assert len(changed) == len(data) and changed != data
    path.write_bytes(changed)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (
        status.st_size,
        status.st_mtime_ns,
    )
    with pytest.raises(TrellisMxP8IncompatibleError, match="hashes to"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


def test_the_resolved_options_are_part_of_the_identity(fake_backend) -> None:
    """Identical backend files, different settings: different plan identities."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    first = attach_trellismx_p8_runtime(
        prepared,
        _runtime(fake_backend, tensors, prepared),
        backend=fake_backend.backend,
    )
    second = attach_trellismx_p8_runtime(
        prepared,
        _runtime(fake_backend, tensors, prepared, rp2_direct_max_tokens=0),
        backend=fake_backend.backend,
    )
    a, b = trellismx_p8_attachment(first), trellismx_p8_attachment(second)
    assert set(a.options.as_dict()) == set(P8_OPTION_ATTRIBUTES)
    assert a.options.as_dict()["rp2_direct_max_tokens"] == 16
    assert b.options.as_dict()["rp2_direct_max_tokens"] == 0
    assert a.backend.identity == b.backend.identity
    assert a.identity != b.identity
    capacity = moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K)
    plan_a = moe.plan_execution(experts=first, capacity=capacity)
    plan_b = moe.plan_execution(experts=second, capacity=capacity)
    assert plan_a.query.backend == plan_b.query.backend
    assert plan_a.query.options == a.options.identity != plan_b.query.options
    assert plan_a.query != plan_b.query


def test_declared_options_must_match_the_runtime(fake_backend) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    runtime = _runtime(fake_backend, tensors, prepared)
    # What the GLM recipe's B12X_P8_* settings resolve to in the runtime.
    declared = {
        "p8_down_remainder": False,
        "p8_dx2_rowpack": True,
        "p8_input_rowpack": True,
        "_dx2_phases": "both",
        "p8_epi_par": True,
        "rp2_direct_max_tokens": 16,
        "rp2_direct_tile_m": 0,
        "rp2_skip_input_clear": True,
    }
    attached = attach_trellismx_p8_runtime(
        prepared, runtime, backend=fake_backend.backend, options=declared
    )
    assert trellismx_p8_attachment(attached).options.as_dict().items() >= (
        declared.items()
    )
    for options, error, match in (
        ({"rp2_direct_max_tokens": 8}, TrellisMxP8IncompatibleError, "rp2_direct"),
        (
            TrellisMxP8Options({"p8_epi_par": False}),
            TrellisMxP8IncompatibleError,
            "epi",
        ),
        ({"rp2_direct_tile_m": False}, TrellisMxP8IncompatibleError, "tile_m"),
        ({"B12X_P8_EPI_PAR": "1"}, ValueError, "unknown P8 options"),
    ):
        with pytest.raises(error, match=match):
            attach_trellismx_p8_runtime(
                prepared, runtime, backend=fake_backend.backend, options=options
            )
    with pytest.raises(TypeError, match="fc1_tile_n"):
        TrellisMxP8Options({"fc1_tile_n": torch.tensor(128)})
    # A runtime that does not expose a setting cannot be identified.
    del runtime.grid_policy
    with pytest.raises(TrellisMxP8IncompatibleError, match="grid_policy"):
        attach_trellismx_p8_runtime(prepared, runtime, backend=fake_backend.backend)


# --------------------------------------------------------------------------- #
# dataclasses.replace keeps the runtime only while compatible
# --------------------------------------------------------------------------- #


def test_replace_keeps_the_runtime_only_while_compatible(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    impl = attached._impl
    record = _record(attached)

    kept = dataclasses.replace(impl, immutable_input_scales=True)
    assert trellismx_p8_attachment(kept).runtime is runtime
    kept_record = dataclasses.replace(record, trellis_exact_hadamard=True)
    assert kept_record.p8 is record.p8

    for name in ("w2_sfb", "w13_scale", "down_svh", "intermediate_rotations"):
        with pytest.raises(TrellisMxP8IncompatibleError, match=name):
            dataclasses.replace(record, **{name: getattr(record, name).clone()})
    with pytest.raises(TrellisMxP8IncompatibleError, match="trellis_bits"):
        dataclasses.replace(record, trellis_bits=3)
    # The bundle's own canonical-storage guard or the attachment rejects it.
    with pytest.raises(ValueError, match="canonical scale storage|w1_blockscale"):
        dataclasses.replace(impl, w1_blockscale=record.w13_sfb.clone())

    detached = dataclasses.replace(record, p8=None)
    assert detached.p8 is None


def test_the_owner_plan_must_match_the_attachment(fake_backend) -> None:
    """A bundle whose plan declares another bitrate or boundary drops nothing silently."""

    *_, attached = _attached(fake_backend)
    impl = attached._impl
    k3 = dataclasses.replace(impl.plan, trellis_bits=3)
    with pytest.raises(TrellisMxP8IncompatibleError, match="trellis_bits"):
        dataclasses.replace(impl, plan=k3)
    plain = dataclasses.replace(
        impl.plan, intermediate_hadamard=False, intermediate_hadamard_blocks=None
    )
    with pytest.raises(TrellisMxP8IncompatibleError, match="intermediate_hadamard"):
        dataclasses.replace(impl, plan=plain)
    # The same plan is accepted.
    assert (
        trellismx_p8_attachment(dataclasses.replace(impl, plan=impl.plan)) is not None
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("trellis_exact_hadamard", False),
        ("trellis_fp32_boundary", False),
        ("trellis_pre_quantized_a", False),
        ("trellis_codebook", "lut_e4m3"),
        ("weight_layout", "trellis_t256_proj"),
        ("scale_format", "e4m3_k32"),
        ("w13_layout", "w31"),
        ("params_dtype", torch.float16),
        ("source_format", "b12x_trellis"),
    ],
)
def test_the_record_invariants_are_not_configurable(fake_backend, name, value) -> None:
    *_, attached = _attached(fake_backend)
    for record in (_record(attached), _record(_prepare(*_sidecar()))):
        with pytest.raises(ValueError, match=f"fix {name}"):
            dataclasses.replace(record, **{name: value})


def test_the_attached_bundle_shares_the_runtime_scale_planes(fake_backend) -> None:
    """One repacked UE8M0 copy stays resident: the bundle adopts the runtime's."""

    tensors, prepared, runtime, attached = _attached(fake_backend)
    record = _record(attached)
    for name in ("w13_sfb", "w2_sfb"):
        assert getattr(record, name).untyped_storage().data_ptr() == (
            getattr(runtime, name).untyped_storage().data_ptr()
        )
        assert torch.equal(getattr(record, name), getattr(_record(prepared), name))
    assert attached._impl.w1_blockscale is record.w13_sfb
    assert attached._impl.w2_blockscale is record.w2_sfb
    packed = runtime.scale_component_packed
    assert trellismx_p8_attachment(attached).resident_nbytes == (
        packed.untyped_storage().nbytes()
    )
    # The unattached bundle's own copies are released with it.
    original = weakref.ref(_record(prepared).w13_sfb)
    del prepared
    gc.collect()
    assert original() is None


# --------------------------------------------------------------------------- #
# immutability while attached
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,value",
    [
        ("w2_stream", "clone"),
        ("w13_sfb", "clone"),
        ("scale_component_packed", "clone"),
        ("swiglu_limit", 3.0),
        ("deterministic_output", False),
        ("topk", _TOP_K + 1),
        ("rp2_direct_max_tokens", 8),
        ("grid_policy", False),
        ("_dx2_phases", "fc1"),
    ],
)
def test_a_runtime_changed_after_attach_is_rejected(fake_backend, name, value) -> None:
    """Planning, binding and dispatch see a replaced handle or a changed setting."""

    tensors, prepared, runtime, attached = _attached(fake_backend)
    capacity = moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K)
    state = moe.plan_execution(experts=attached, capacity=capacity)._materialize(
        None, SimpleNamespace(ordinal=0)
    )
    current = getattr(runtime, name)
    setattr(runtime, name, current.clone() if value == "clone" else value)
    match = "replaced after attachment" if value == "clone" else f"{name} changed"
    with pytest.raises(TrellisMxP8IncompatibleError, match=match):
        _impl.b12x_moe_fp4(binding=_binding(attached._impl))
    with pytest.raises(TrellisMxP8IncompatibleError, match=match):
        moe.plan_execution(experts=attached, capacity=capacity)
    with pytest.raises(TrellisMxP8IncompatibleError, match=match):
        state.bind(
            a=torch.zeros(2, _H, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
            topk_weights=torch.full((2, _TOP_K), 0.5),
            scratch=tuple(
                torch.empty(spec.shape, dtype=spec.dtype)
                for spec in state.scratch.scratch_specs()
            ),
        )
    assert runtime.calls == []
    setattr(runtime, name, current)
    assert _impl.b12x_moe_fp4(binding=_binding(attached._impl)) is not None


def test_explicit_verification_reads_the_contents(fake_backend) -> None:
    """In-place writes are invisible to the host-side checks; verify reads them."""

    tensors, prepared, runtime, attached = _attached(fake_backend)
    verify_trellismx_p8_attachment(attached)
    record = _record(attached)
    attachment = trellismx_p8_attachment(attached)
    for tensor, change, undo, match in (
        (record.down_svh, lambda t: t.mul_(2), lambda t: t.div_(2), "scale_component"),
        (
            record.intermediate_rotations,
            lambda t: t.mul_(-1),
            lambda t: t.mul_(-1),
            "scale_component",
        ),
        (
            record.w13_scale,
            lambda t: t[0, 0].add_(1),
            lambda t: t[0, 0].sub_(1),
            "w13_sfb",
        ),
        (
            record.w2_scale,
            lambda t: t[-1, -1].add_(1),
            lambda t: t[-1, -1].sub_(1),
            "w2_sfb",
        ),
    ):
        change(tensor)
        attachment.check_record(record)  # same storage, offset, shape: identity holds
        with pytest.raises(TrellisMxP8IncompatibleError, match=match):
            verify_trellismx_p8_attachment(attached)
        undo(tensor)
        verify_trellismx_p8_attachment(attached)
    with pytest.raises(TrellisMxP8IncompatibleError, match="owns no P8"):
        verify_trellismx_p8_attachment(prepared)


# --------------------------------------------------------------------------- #
# serialization: an inert descriptor crosses the process boundary
# --------------------------------------------------------------------------- #


def test_pickling_an_attached_bundle_yields_an_inert_descriptor(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    live = trellismx_p8_attachment(attached)
    data = pickle.dumps(attached)
    assert b"FakeP8Runtime" not in data, "the runtime must never be serialized"
    back = pickle.loads(data)
    descriptor = trellismx_p8_attachment(back)
    assert isinstance(descriptor, TrellisMxP8Descriptor)
    assert not isinstance(descriptor, TrellisMxP8Attachment)
    assert descriptor.live is False and live.live is True
    assert descriptor.identity == live.identity
    assert descriptor.token == live.token
    assert descriptor.backend == live.backend
    assert descriptor.options == live.options
    assert descriptor.contract == live.contract
    assert descriptor.resident_nbytes == live.resident_nbytes
    with pytest.raises(TrellisMxP8Unavailable, match="never launches"):
        descriptor.runtime
    # Round trips again, and copies of a live attachment behave the same way.
    again = trellismx_p8_attachment(pickle.loads(pickle.dumps(back)))
    assert isinstance(again, TrellisMxP8Descriptor) and again.identity == live.identity
    assert isinstance(
        trellismx_p8_attachment(copy.deepcopy(attached)), TrellisMxP8Descriptor
    )
    assert copy.copy(live) is live
    # The bundle around it is intact: same tensors and fields, unattached arms untouched.
    assert torch.equal(_record(back).w2, _record(attached).w2)
    assert back._impl.w1_blockscale is _record(back).w13_sfb


def test_a_deserialized_bundle_plans_the_same_p8_plan(fake_backend) -> None:
    """The descriptor declares the same compile job and arena as the live one."""

    *_, attached = _attached(fake_backend)
    back = pickle.loads(pickle.dumps(attached))
    capacity = moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K)
    original = moe.plan_execution(experts=attached, capacity=capacity)
    plan = moe.plan_execution(experts=back, capacity=capacity)
    assert plan.contract is trellismx_p8.P8_TUNING
    assert plan.query == original.query
    (job,) = plan._compile_jobs(None, None)
    assert (job,) == original._compile_jobs(None, None)
    device = SimpleNamespace(ordinal=0)
    live = original._memory_requirements(None, device)
    assert plan._memory_requirements(None, device).persistent == live.persistent
    assert plan._memory_requirements(None, device).scratch == live.scratch
    assert live.scratch_nbytes > 256


def test_a_deserialized_binding_never_launches(
    fake_backend, fake_smallm, monkeypatch
) -> None:
    """No launch and no fallback to the in-tree arms, even where they would serve."""

    tensors, prepared, runtime, attached = _attached(fake_backend)
    monkeypatch.setattr(_impl, "_on_cuda", lambda tensor: True)
    for binding in (
        _binding(attached._impl),
        _binding(attached._impl, implementation="trellismx_p8"),
    ):
        back = pickle.loads(pickle.dumps(binding))
        with pytest.raises(TrellisMxP8Unavailable):
            _impl.b12x_moe_fp4(binding=back)
    assert runtime.calls == []
    assert fake_smallm == []


def test_a_deserialized_p8_state_binds_but_never_launches(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    plan = moe.plan_execution(
        experts=attached, capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K)
    )
    state = pickle.loads(
        pickle.dumps(plan._materialize(None, SimpleNamespace(ordinal=0)))
    )
    assert isinstance(state.attachment, TrellisMxP8Descriptor)
    binding = state.bind(
        a=torch.zeros(4, _H, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, _TOP_K, dtype=torch.int32),
        topk_weights=torch.full((4, _TOP_K), 0.5),
        output=torch.zeros(4, _H, dtype=torch.bfloat16),
        scratch=tuple(
            torch.empty(spec.shape, dtype=spec.dtype)
            for spec in state.scratch.scratch_specs()
        ),
    )
    assert binding.implementation == "trellismx_p8"
    with pytest.raises(TrellisMxP8Unavailable):
        _impl.b12x_moe_fp4(binding=binding)
    assert runtime.calls == []


def test_rebinding_a_deserialized_bundle_requires_the_same_identity(
    fake_backend,
) -> None:
    *_, attached = _attached(fake_backend)
    back = pickle.loads(pickle.dumps(attached))
    descriptor = trellismx_p8_attachment(back)
    other = _runtime_from_record(fake_backend, back, rp2_direct_max_tokens=0)
    with pytest.raises(TrellisMxP8IncompatibleError, match="deserialized bundle"):
        attach_trellismx_p8_runtime(back, other, backend=fake_backend.backend)
    runtime = _runtime_from_record(fake_backend, back)
    rebound = attach_trellismx_p8_runtime(back, runtime, backend=fake_backend.backend)
    attachment = trellismx_p8_attachment(rebound)
    assert isinstance(attachment, TrellisMxP8Attachment)
    assert attachment.runtime is runtime and attachment.identity == descriptor.identity
    binding = _binding(rebound._impl)
    assert _impl.b12x_moe_fp4(binding=binding) is binding.output
    assert len(runtime.calls) == 1


def test_the_descriptor_keeps_the_record_contract(fake_backend) -> None:
    *_, attached = _attached(fake_backend)
    record = _record(pickle.loads(pickle.dumps(attached)))
    with pytest.raises(TrellisMxP8IncompatibleError, match="trellis_bits"):
        dataclasses.replace(record, trellis_bits=3)
    with pytest.raises(TrellisMxP8IncompatibleError, match="down_svh"):
        dataclasses.replace(record, down_svh=record.down_svh[..., :8].clone())
    # Addresses and contents are not visible to a descriptor; rebinding verifies them.
    assert dataclasses.replace(record, down_svh=record.down_svh.clone()).p8 is record.p8


# --------------------------------------------------------------------------- #
# lifetimes
# --------------------------------------------------------------------------- #


def test_the_runtime_lives_exactly_as_long_as_its_owners(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    alive = weakref.ref(runtime)
    binding = _binding(attached._impl)
    del runtime, attached
    gc.collect()
    assert alive() is not None, "a live binding must keep the runtime for its graphs"
    del binding
    gc.collect()
    assert alive() is None, "nothing global may retain the runtime"


class _FakePlan:
    pass


@pytest.fixture
def fake_smallm(monkeypatch):
    from b12x.moe._shared.kernels import trellismx_smallm

    made = []

    def plan(experts, *, max_tokens, top_k, tensors=None, meta=None):
        result = _FakePlan()
        result.args = (max_tokens, top_k)
        made.append(weakref.ref(result))  # weak: the fixture must not own plans
        return result

    monkeypatch.setattr(trellismx_smallm.TrellisMxSmallM, "plan", staticmethod(plan))
    return made


def test_smallm_plans_are_owned_by_the_bundle(fake_smallm) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    impl = prepared._impl
    first = _impl._trellismx_smallm_plan(impl, top_k=_TOP_K, max_tokens=16)
    assert first.args == (16, _TOP_K)
    assert _impl._trellismx_smallm_plan(impl, top_k=_TOP_K, max_tokens=16) is first
    assert impl._trellismx_smallm_plans == {(_TOP_K, 16): first}
    # Another bundle, or a replaced one, plans its own.
    replaced = dataclasses.replace(impl, immutable_input_scales=True)
    assert replaced._trellismx_smallm_plans == {}
    assert (
        _impl._trellismx_smallm_plan(replaced, top_k=_TOP_K, max_tokens=16) is not first
    )
    # The plan holds no reference back to the bundle.
    from b12x.moe._shared.kernels.trellismx_smallm import TrellisMxSmallMPlan

    assert "experts" not in TrellisMxSmallMPlan.__dataclass_fields__
    alive = weakref.ref(first)
    release_trellismx_execution_state(prepared)
    assert impl._trellismx_smallm_plans == {}
    del first
    gc.collect()
    assert alive() is None


def test_smallm_plans_die_with_the_bundle_and_live_with_bindings(fake_smallm) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    plan = _impl._trellismx_smallm_plan(prepared._impl, top_k=_TOP_K, max_tokens=16)
    alive = weakref.ref(plan)
    binding = _binding(prepared._impl)
    del plan, prepared
    gc.collect()
    assert alive() is not None, "a live binding keeps the plan its graphs replay"
    del binding
    gc.collect()
    assert alive() is None


def test_smallm_planning_refuses_capture(fake_smallm, monkeypatch) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        _impl._trellismx_smallm_plan(prepared._impl, top_k=_TOP_K, max_tokens=16)


# --------------------------------------------------------------------------- #
# dispatch: validation before the runtime runs
# --------------------------------------------------------------------------- #


def test_a_bound_call_runs_the_attached_runtime(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    binding = _binding(attached._impl)
    result = _impl.b12x_moe_fp4(binding=binding)
    assert result is binding.output
    assert float(result[0, 0]) == 1.5
    ((x, weights, ids, out),) = runtime.calls
    assert x is binding.a and ids is binding.topk_ids and out is binding.output
    # A bundle without the runtime does not reach it.
    with pytest.raises(RuntimeError, match="do not own"):
        _impl.b12x_moe_fp4(
            binding=_binding(prepared._impl, implementation="trellismx_p8")
        )


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        ({"a": torch.zeros(3, _H, dtype=torch.float16)}, ValueError, "BF16"),
        (
            {
                "topk_ids": torch.zeros(3, _TOP_K + 1, dtype=torch.int32),
                "topk_weights": torch.zeros(3, _TOP_K + 1),
            },
            ValueError,
            "routing",
        ),
        ({"topk_ids": torch.zeros(3, _TOP_K, dtype=torch.int16)}, TypeError, "int32"),
        ({"apply_router_weight_on_input": True}, NotImplementedError, "router weights"),
        (
            {"route_expert_map": torch.zeros(_E, dtype=torch.int32)},
            ValueError,
            "route maps",
        ),
        (
            {"output_expert_map": torch.zeros(_E, dtype=torch.int32)},
            ValueError,
            "route maps",
        ),
        ({"activation_amax": torch.zeros(1)}, NotImplementedError, "amax"),
        ({"swiglu_limit": 7.0}, ValueError, "silu-cap10"),
        ({"swiglu_limit": None}, ValueError, "silu-cap10"),
        ({"swiglu_clamp_after_silu": True}, ValueError, "silu-cap10"),
        ({"swiglu_alpha": 2.0}, ValueError, "swiglu_alpha"),
        ({"swiglu_beta": 1.0}, ValueError, "swiglu_alpha"),
        ({"output": torch.zeros(3, _H, dtype=torch.float32)}, ValueError, "output"),
        (
            {"output": torch.zeros(_H, 3, dtype=torch.bfloat16).t()},
            ValueError,
            "output",
        ),
        (
            {"output_cast_target": torch.zeros(3, _H, dtype=torch.bfloat16)},
            ValueError,
            "cast target",
        ),
        ({"routed_rows_capacity": 2, "max_rows": 2}, ValueError, "capacity"),
        ({"quant_mode": "nvfp4"}, ValueError, "w4a8_mx"),
        (
            {"a": torch.zeros(3 * _H + 1, dtype=torch.bfloat16)[1:].view(3, _H)},
            ValueError,
            "16-byte aligned",
        ),
        ({"a": torch.zeros(_H, 3, dtype=torch.bfloat16).t()}, ValueError, "contiguous"),
        (
            {"a": torch.empty(3, _H, dtype=torch.bfloat16, device="meta")},
            ValueError,
            "lives? on",
        ),
        ({"fast_math": False}, ValueError, "fast_math"),
    ],
)
def test_invalid_calls_are_rejected_before_the_runtime(
    fake_backend, overrides, error, match
) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    with pytest.raises(error, match=match):
        _impl.b12x_moe_fp4(binding=_binding(attached._impl, **overrides))
    assert runtime.calls == []


def test_plain_silu_parameters_are_accepted(fake_backend) -> None:
    tensors, prepared, runtime, attached = _attached(fake_backend)
    binding = _binding(attached._impl, swiglu_alpha=1.0, swiglu_beta=0.0)
    assert _impl.b12x_moe_fp4(binding=binding) is binding.output
    assert len(runtime.calls) == 1


def test_invalid_calls_are_rejected_before_the_smallm_arm(fake_smallm) -> None:
    """The arm without a runtime validates the same contract before any work."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    for overrides, error in (
        ({"apply_router_weight_on_input": True}, NotImplementedError),
        ({"route_expert_map": torch.zeros(_E, dtype=torch.int32)}, ValueError),
        ({"activation_amax": torch.zeros(1)}, NotImplementedError),
        ({"swiglu_limit": 7.0}, ValueError),
        ({"swiglu_alpha": 2.0}, ValueError),
        ({"swiglu_beta": 1.0}, ValueError),
        ({"a": torch.zeros(3, _H, dtype=torch.float16)}, ValueError),
        ({"a": torch.empty(3, _H, dtype=torch.bfloat16, device="meta")}, ValueError),
        (
            {"a": torch.zeros(3 * _H + 1, dtype=torch.bfloat16)[1:].view(3, _H)},
            ValueError,
        ),
    ):
        with pytest.raises(error):
            _impl.b12x_moe_fp4(binding=_binding(prepared._impl, **overrides))
    assert fake_smallm == []


# --------------------------------------------------------------------------- #
# preparation: the P8 plan and TrellisMX declaration checks
# --------------------------------------------------------------------------- #


def test_plan_execution_declares_the_attached_runtime(fake_backend) -> None:
    from b12x.preparation.types import Plan, _CompositePlan

    tensors, prepared, runtime, attached = _attached(fake_backend)
    plan = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=16, top_k=_TOP_K, route_num_experts=0
        ),
    )
    assert isinstance(plan, Plan)
    assert plan.contract is trellismx_p8.P8_TUNING
    assert plan.component_id == "moe.decode"
    assert plan.query.backend == fake_backend.backend.identity
    assert plan.query.num_tokens == 16 and plan.query.top_k == _TOP_K
    (job,) = plan._compile_jobs(None, None)
    assert job == trellismx_p8_attachment(attached).compile_jobs(16)[0]
    state = plan._materialize(None, SimpleNamespace(ordinal=0))
    specs = state.scratch.scratch_specs()
    assert len(specs) == 1 and specs[0].nbytes > 256
    a = torch.randn(5, _H).to(torch.bfloat16)
    binding = state.bind(
        a=a,
        topk_ids=torch.zeros(5, _TOP_K, dtype=torch.int32),
        topk_weights=torch.full((5, _TOP_K), 0.5),
        output=torch.zeros(5, _H, dtype=torch.bfloat16),
        scratch=tuple(torch.empty(spec.shape, dtype=spec.dtype) for spec in specs),
        input_scales_static=True,
    )
    assert binding.implementation == "trellismx_p8"
    assert _impl.b12x_moe_fp4(binding=binding) is binding.output
    assert len(runtime.calls) == 1
    with pytest.raises(ValueError, match="capacity"):
        state.bind(
            a=torch.zeros(17, _H, dtype=torch.bfloat16),
            topk_ids=torch.zeros(17, _TOP_K, dtype=torch.int32),
            topk_weights=torch.zeros(17, _TOP_K),
        )
    with pytest.raises(ValueError, match="activation_amax"):
        state.bind(
            a=a,
            topk_ids=binding.topk_ids,
            topk_weights=binding.topk_weights,
            activation_amax=torch.zeros(1),
        )
    with pytest.raises(ValueError, match="fast_math"):
        state.bind(
            a=a,
            topk_ids=binding.topk_ids,
            topk_weights=binding.topk_weights,
            fast_math=False,
        )
    with pytest.raises(TypeError, match="unexpected"):
        state.bind(
            a=a,
            topk_ids=binding.topk_ids,
            topk_weights=binding.topk_weights,
            apply_router_weight_on_input=True,
        )
    with pytest.raises(NotImplementedError):
        state.route(binding)
    composite = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(
            max_tokens=64, top_k=_TOP_K, warmup_token_counts=(1, 8)
        ),
    )
    assert isinstance(composite, _CompositePlan)
    assert composite.token_counts == (1, 8, 64)


def test_a_pinned_resident_grid_cap_keeps_the_race_off_the_cap_axis(
    fake_backend,
) -> None:
    """A constructor cap pin is the only owner of the cap: no race replaces it."""

    from b12x.moe._shared.p8.launch import P8LaunchConfig

    *_, attached = _attached_with(fake_backend, mac_override=96)
    plan = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(max_tokens=16, top_k=_TOP_K),
    )
    assert plan.query.mac_pin == 96
    space = plan.contract.parameter_space(plan.query, None)
    (mac,) = [knob for knob in space.knobs if knob.name == "mac"]
    assert mac.values == (0,)
    assert {config.mac for _, config in plan.contract.choices(plan.query, None)} == {0}
    with pytest.raises(ValueError, match="pins its resident-grid cap"):
        plan.contract.validate_config(plan.query, P8LaunchConfig(mac=64), None)
    plan.contract.validate_config(plan.query, P8LaunchConfig.serving(), None)


def test_a_p8_bind_rejects_a_misaligned_arena(fake_backend) -> None:
    """The launch assumes 16-byte alignment for every region pointer.

    The plan's arena specification carries that requirement, so a contiguous
    one-byte-offset slice of a properly sized buffer is refused at bind time
    rather than handed to the kernels as an aligned arena.
    """

    *_, attached = _attached(fake_backend)
    plan = moe.plan_execution(
        experts=attached,
        capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K),
    )
    state = plan._materialize(None, SimpleNamespace(ordinal=0))
    (spec,) = state.scratch.scratch_specs()
    assert spec.alignment == 16
    base = torch.empty(spec.shape[0] + 16, dtype=torch.uint8)
    shift = (16 - base.data_ptr() % 16) % 16
    arena = base[shift + 1 : shift + 1 + spec.shape[0]]
    assert arena.is_contiguous() and arena.data_ptr() % 16
    a = torch.zeros(2, _H, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="16-byte aligned"):
        state.bind(
            a=a,
            topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
            topk_weights=torch.full((2, _TOP_K), 0.5),
            output=torch.zeros(2, _H, dtype=torch.bfloat16),
            scratch=(arena,),
        )
    state.bind(
        a=a,
        topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
        topk_weights=torch.full((2, _TOP_K), 0.5),
        output=torch.zeros(2, _H, dtype=torch.bfloat16),
        scratch=(base[shift : shift + spec.shape[0]],),
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"capacity": moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K + 1)}, "top_k"),
        (
            {
                "capacity": moe.ExecutionCapacity(
                    max_tokens=8, top_k=_TOP_K, route_num_experts=7
                )
            },
            "route maps",
        ),
        (
            {"routing": moe.RoutingSpec(apply_router_weight_on_input=True)},
            "router weights",
        ),
        ({"routing": moe.RoutingSpec(collect_activation_amax=True)}, "amax"),
        ({"invocation": {"a4_prefill": True}}, "A4"),
        ({"invocation": {"fast_math": False}}, "fast_math"),
        ({"override": object()}, "configs"),
        ({"scale_expansion": object()}, "expand"),
    ],
)
def test_plan_execution_rejects_what_the_runtime_cannot_serve(
    fake_backend, kwargs, match
) -> None:
    *_, attached = _attached(fake_backend)
    arguments = dict(capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K))
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        moe.plan_execution(experts=attached, **arguments)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"routing": moe.RoutingSpec(apply_router_weight_on_input=True)},
            "router weights",
        ),
        ({"routing": moe.RoutingSpec(collect_activation_amax=True)}, "amax"),
        (
            {
                "capacity": moe.ExecutionCapacity(
                    max_tokens=8, top_k=_TOP_K, route_num_experts=7
                )
            },
            "route maps",
        ),
        ({"invocation": {"a4_prefill": True}}, "A4"),
    ],
)
def test_trellismx_plans_reject_unsupported_routing(kwargs, match) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    arguments = dict(capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K))
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        moe.plan_execution(experts=prepared, **arguments)


@pytest.mark.parametrize(
    "scale_bytes,invocation,expected",
    [
        ((127, 127), {}, True),
        ((127, 127), {"fast_math": False}, False),
        ((111, 127), {}, False),
    ],
)
def test_trellismx_plans_resolve_the_smallm_policy_once(
    scale_bytes, invocation, expected
) -> None:
    """plan_execution resolves the small-M decision once, into the controls."""

    from b12x.moe._shared.kernels.trellismx_smallm import SMALLM_MAX_TOKENS

    tensors, metadata = _sidecar(scale_bytes=scale_bytes)
    prepared = _prepare(tensors, metadata)
    plan = moe.plan_execution(
        experts=prepared,
        capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K),
        invocation=invocation,
    )
    controls = plan.query.controls
    assert controls["trellismx_smallm"] is expected
    assert controls["trellismx_smallm_limit"] == SMALLM_MAX_TOKENS
    assert _impl._trellismx_smallm_policy(
        prepared._impl, fast_math=invocation.get("fast_math", True)
    ) == (controls["trellismx_smallm"], controls["trellismx_smallm_limit"])


def test_discovery_refuses_a_query_without_the_policy() -> None:
    from b12x.moe.fused_moe._preparation import _precompile_trellismx_programs

    query = SimpleNamespace(num_tokens=8, hidden_size=_H, controls={})
    with pytest.raises(ValueError, match="small-M policy"):
        _precompile_trellismx_programs(query, SimpleNamespace(trellis_bits=_BITS), 0)


class _FakeScratch:
    """The scratch plan side of ``_FusedMoeState.bind``: builds the binding."""

    # Planned for the three rows `_state_binding` binds.
    caps = SimpleNamespace(w4a16_fast_math=True, max_tokens=3, output_layout="token")

    def bind(self, *, experts, a, topk_ids, topk_weights, output, fast_math, **_):
        return _binding(
            experts,
            m=int(a.shape[0]),
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            fast_math=fast_math,
            # A real scratch plan carves the arm's buffers out of the arena it
            # was bound; this test is about which arm dispatch chooses.
            trellismx_smallm_scratch=SimpleNamespace(),
        )


@pytest.fixture
def smallm_runs(monkeypatch):
    """Record small-M runs; any CUDA check passes, so only the decision matters."""

    from b12x.moe._shared.kernels import trellismx_smallm

    runs = []
    monkeypatch.setattr(_impl, "_on_cuda", lambda tensor: True)
    monkeypatch.setattr(
        _impl,
        "_trellismx_smallm_plan",
        lambda experts, *, top_k, max_tokens: ("plan", max_tokens),
    )
    monkeypatch.setattr(
        trellismx_smallm.TrellisMxSmallM,
        "run",
        staticmethod(
            lambda a, ids, weights, out, workspace, scratch=None: runs.append(workspace)
        ),
    )
    return runs


def _state_binding(prepared, tokens):
    from b12x.moe.fused_moe._preparation import _FusedMoeState

    state = _FusedMoeState(
        prepared,
        _FakeScratch(),
        SimpleNamespace(nvfp4_share_input=False, backend="dynamic"),
        None,
        (),
        None,
        trellismx_smallm_tokens=tokens,
    )
    return state.bind(
        a=torch.randn(3, _H).to(torch.bfloat16),
        topk_ids=torch.zeros(3, _TOP_K, dtype=torch.int32),
        topk_weights=torch.full((3, _TOP_K), 0.5),
        output=torch.zeros(3, _H, dtype=torch.bfloat16),
    )


def test_dispatch_runs_the_arm_the_plan_resolved(smallm_runs, monkeypatch) -> None:
    """The prepared state stamps the plan's decision; dispatch follows it even
    when resolving the policy again would now decide otherwise."""

    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)

    # Re-resolving the policy now would say "grouped arm only"; the prepared
    # binding still runs the arm it was made for.
    monkeypatch.setattr(
        _impl, "_trellismx_smallm_policy", lambda *args, **kwargs: (False, 16)
    )
    binding = _state_binding(prepared, 16)
    assert binding.trellismx_smallm_tokens == 16
    assert _impl.b12x_moe_fp4(binding=binding) is binding.output
    assert smallm_runs == [("plan", 16)]

    # And the other way: the binding says 0, so the grouped arm serves the call.
    monkeypatch.setattr(
        _impl, "_trellismx_smallm_policy", lambda *args, **kwargs: (True, 16)
    )
    binding = _state_binding(prepared, 0)
    assert binding.trellismx_smallm_tokens == 0
    assert _impl._run_trellismx_fast_path(binding) is None
    # The grouped launch is what runs next (and needs a real plan on CPU).
    with pytest.raises(RuntimeError, match="execution plan"):
        _impl.b12x_moe_fp4(binding=binding)
    assert smallm_runs == [("plan", 16)]


def test_a_binding_without_a_plan_resolves_the_arm_at_dispatch(
    smallm_runs, monkeypatch
) -> None:
    tensors, metadata = _sidecar()
    prepared = _prepare(tensors, metadata)
    binding = _binding(prepared._impl, trellismx_smallm_scratch=SimpleNamespace())
    assert binding.trellismx_smallm_tokens is None
    monkeypatch.setattr(
        _impl, "_trellismx_smallm_policy", lambda *args, **kwargs: (True, 16)
    )
    assert _impl.b12x_moe_fp4(binding=binding) is binding.output
    assert len(smallm_runs) == 1
    monkeypatch.setattr(
        _impl, "_trellismx_smallm_policy", lambda *args, **kwargs: (False, 16)
    )
    assert _impl._run_trellismx_fast_path(binding) is None
    assert len(smallm_runs) == 1


@pytest.mark.parametrize(
    "scale_bytes,invocation,enabled",
    [
        ((127, 127), {}, True),
        ((127, 127), {"fast_math": False}, False),
        ((111, 127), {}, False),
    ],
)
def test_the_prepared_state_carries_the_query_decision(
    scale_bytes, invocation, enabled
) -> None:
    from b12x.moe._shared.kernels.trellismx_smallm import SMALLM_MAX_TOKENS
    from b12x.moe.fused_moe._preparation import _query_smallm_tokens

    tensors, metadata = _sidecar(scale_bytes=scale_bytes)
    plan = moe.plan_execution(
        experts=_prepare(tensors, metadata),
        capacity=moe.ExecutionCapacity(max_tokens=8, top_k=_TOP_K),
        invocation=invocation,
    )
    expected = SMALLM_MAX_TOKENS if enabled else 0
    assert _query_smallm_tokens(plan.query) == expected


# --------------------------------------------------------------------------- #
# activation and dtype contract
# --------------------------------------------------------------------------- #


def _meta():
    tensors, metadata = _sidecar()
    return trellismx.TrellisMxMetadata.from_sidecar(
        metadata, num_experts=_E, hidden_size=_H, intermediate_size=_I, top_k=_TOP_K
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"io_dtype": torch.float16}, "BF16"),
        ({"swiglu_limit": 7.0}, "silu-cap10"),
        ({"swiglu_limit": None}, "silu-cap10"),
        ({"nonlinearity": "situ", "swiglu_limit": None}, "silu-cap10"),
        ({"rotation_dtype": torch.float16}, "rotation_dtype"),
    ],
)
def test_planning_rejects_any_other_activation_or_dtype(overrides, match) -> None:
    values = dict(
        mode=moe.ActivationMode.A8,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
        swiglu_limit=10.0,
    )
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        moe.plan_weights(
            source=_meta().source(),
            activation=moe.ActivationSpec(**values),
            geometry=moe.MoEGeometry(
                num_experts=_E, hidden_size=_H, intermediate_size=_I
            ),
        )


def test_the_clamp_and_dtype_are_not_configurable() -> None:
    tensors, metadata = _sidecar()
    for name, value in (("io_dtype", torch.float16), ("swiglu_limit", 7.0)):
        with pytest.raises(TypeError):
            trellismx.prepare_trellismx_experts(
                tensors,
                metadata,
                num_experts=_E,
                hidden_size=_H,
                intermediate_size=_I,
                top_k=_TOP_K,
                device=torch.device("cpu"),
                **{name: value},
            )
        with pytest.raises(TypeError):
            trellismx.plan_trellismx(_meta(), **{name: value})
    activation = _prepare(tensors, metadata).plan.activation
    assert (activation.io_dtype, activation.swiglu_limit) == (torch.bfloat16, 10.0)
    assert activation.swiglu_clamp_after_silu is False


# --------------------------------------------------------------------------- #
# small-M eligibility: the fp16 weight-scale product must be exact
# --------------------------------------------------------------------------- #


def test_the_exact_scale_range_follows_from_the_decode_law() -> None:
    """Enumerate every NPK state: bytes 112..139 keep w * 2**(b-127) exact in fp16."""

    from b12x.moe._shared.kernels.w4a8_npk_decode import npk_decode_states_reference

    values = npk_decode_states_reference(torch.arange(1 << 16, dtype=torch.int32))
    weights = values.view(torch.float8_e4m3fn).to(torch.float32)
    exact = []
    for byte in range(100, 146):
        scale = torch.tensor(2.0 ** (byte - 127), dtype=torch.float64).to(torch.float16)
        product = weights.to(torch.float16) * scale
        if torch.equal(product.double(), weights.double() * 2.0 ** (byte - 127)):
            exact.append(byte)
    assert (exact[0], exact[-1]) == _impl._TRELLISMX_SMALLM_EXACT_SCALE_BYTES
    assert exact == list(range(exact[0], exact[-1] + 1))


@pytest.mark.parametrize(
    "scale_bytes,eligible",
    [((127, 127), True), ((112, 139), True), ((111, 127), False), ((127, 140), False)],
)
def test_smallm_eligibility_is_decided_at_preparation(scale_bytes, eligible) -> None:
    tensors, metadata = _sidecar(scale_bytes=scale_bytes)
    impl = _prepare(tensors, metadata)._impl
    assert impl.representation.value.smallm_exact_scales is eligible
    enabled, limit = _impl._trellismx_smallm_policy(impl, fast_math=True)
    assert enabled is eligible
    assert limit >= 1
    # The arm is compiled with fast math only; exact-math calls take the grouped arm.
    assert _impl._trellismx_smallm_policy(impl, fast_math=False)[0] is False


@pytest.mark.parametrize(
    "byte,accepted", [(111, False), (112, True), (139, True), (140, False)]
)
def test_the_smallm_scale_validator_requires_exact_products(byte, accepted) -> None:
    from b12x.moe._shared.kernels.w4a16.prepare import _validate_trellis256_block_scale

    scale = torch.full((2, 1, 1, 32, 8), 0, dtype=torch.int32)
    scale.view(torch.uint8).fill_(byte)
    arguments = dict(
        name="w13_block_scale",
        num_experts=2,
        rows=256,
        k_dim=128,
        device=torch.device("cpu"),
    )
    if accepted:
        assert _validate_trellis256_block_scale(scale, **arguments) is scale
    else:
        with pytest.raises(ValueError, match="112..139"):
            _validate_trellis256_block_scale(scale, **arguments)


def test_the_released_w13_plane_is_handed_over_unswapped() -> None:
    tensors, metadata = _sidecar()
    plane = tensors["w13_scale_ue8m0"]
    plane[:, :_I] = 125  # gate/up halves differ, so a swap would show
    record = _record(_prepare(tensors, metadata))
    assert torch.equal(record.w13_scale, plane)


# --------------------------------------------------------------------------- #
# launch policy: the TrellisMX bits round-trip without disturbing the rest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("clusters", [-1, 0, 188])
@pytest.mark.parametrize("flags", range(16))
def test_trellis_launch_policy_bits_round_trip(clusters, flags) -> None:
    exact, fp32, npk, prequant = (bool(flags & (1 << bit)) for bit in range(4))
    policy = _impl._encode_dynamic_launch_policy(
        volatile_launch_state=True,
        external_route_plan_requested=False,
        planned_tile_m=32,
        planned_direct_routing=False,
        w4a8_n64_repacked=False,
        mma_warps=None,
        policy_max_active_clusters=clusters,
        trellis_exact_hadamard=exact,
        trellis_fp32_boundary=fp32,
        trellis_procedural_npk=npk,
        trellis_pre_quantized_a=prequant,
    )
    assert _impl._decode_dynamic_launch_policy(policy) == (
        True,
        False,
        32,
        False,
        False,
        False,
        None,
        clusters,
    )
    assert _impl._decode_dynamic_trellis_policy(policy) == (exact, fp32, npk, prequant)
