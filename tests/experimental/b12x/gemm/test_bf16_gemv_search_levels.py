"""Host-only effort and backend legality contracts for BF16 projection search."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.gemm.bf16_gemv._tuning import GemvConfig, GemvQuery, TUNING


def query(**changes):
    return GemvQuery(
        source_dtype="bfloat16",
        weight_dtype="bfloat16",
        max_rows=256,
        in_features=5120,
        out_features=32,
        source_contiguous=True,
        source_aligned=True,
        weight_contiguous=True,
        weight_aligned=True,
        **changes,
    )


def choices(q):
    return {config for _, config in TUNING.choices(q, None)}


def test_effort_capture_and_selection_identity(monkeypatch):
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
    low = query()
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    high = query()
    assert low.autotune_effort == "default"
    assert high.autotune_effort == "audit"
    assert TUNING.encode_query(low) != TUNING.encode_query(high)
    assert TUNING.parameter_space(low, None).autotune_effort == "default"
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "invalid")
    with pytest.raises(ValueError):
        query()


@pytest.mark.parametrize("backend", ["simt", "mma", "prefill"])
def test_effort_does_not_change_compile_jobs(backend):
    from b12x.gemm.bf16_gemv._preparation import plan

    low = query(autotune_effort="off")
    maximum = replace(low, autotune_effort="audit")
    low_plan, maximum_plan = plan(low), plan(maximum)
    config = GemvConfig(backend=backend)
    device = SimpleNamespace(ordinal=0)
    assert TUNING.encode_query(low) != TUNING.encode_query(maximum)
    jobs = low_plan._compile_jobs(config, device)
    assert jobs == maximum_plan._compile_jobs(config, device)
    assert jobs[0].args[0]["autotune_effort"] == "default"


@pytest.mark.parametrize(
    "n,k", [(32, 5120), (384, 5120), (512, 5120), (1024, 5120), (17, 8)]
)
@pytest.mark.parametrize("rows", [1, 8, 256])
def test_nested_tiers_and_natural_heuristic_membership(n, k, rows):
    q = replace(query(), out_features=n, in_features=k, max_rows=rows)
    default = TUNING.default_config(q, None)
    previous = {default}
    for effort in ("off", "default", "audit"):
        active = replace(q, autotune_effort=effort)
        space = TUNING.parameter_space(active, None)
        assignment = TUNING.config_payload(default)
        space.validate(assignment)
        assert assignment in list(space.configurations())
        selected = choices(active)
        assert previous <= selected
        if effort == "off":
            assert selected == {default}
        previous = selected


def test_audit_restores_small_projection_backends_and_preserves_pins():
    q = query()
    for backend in ("mma", "prefill"):
        config = GemvConfig(backend=backend)
        TUNING.validate_config(q, config, None)
        assert config not in choices(q)
        pinned = TUNING.configure(q, device=None, override=config)
        assert pinned.default == pinned.pinned == config
        assert config in choices(replace(q, autotune_effort="audit"))


@pytest.mark.parametrize(
    "backend,changes",
    [
        ("mma", {"source_dtype": "float32"}),
        ("mma", {"weight_dtype": "float32"}),
        ("prefill", {"in_features": 513}),
        ("prefill", {"source_dtype": "float32"}),
        ("prefill", {"bias_dtype": "bfloat16"}),
        *[
            ("prefill", {name: False})
            for name in (
                "source_contiguous",
                "source_aligned",
                "weight_contiguous",
                "weight_aligned",
                "output_contiguous",
                "output_aligned",
            )
        ],
        ("tc", {"max_rows": 9}),
        ("tc", {"max_rows": 8, "in_features": 65}),
        ("tc", {"max_rows": 8, "source_dtype": "float32"}),
        ("tc", {"max_rows": 8, "weight_dtype": "float32"}),
        *[
            ("tc", {"max_rows": 8, name: False})
            for name in (
                "source_contiguous",
                "source_aligned",
                "weight_contiguous",
                "weight_aligned",
                "output_contiguous",
            )
        ],
    ],
)
@pytest.mark.parametrize("output_dtype", ["bfloat16", "float32"])
def test_audit_keeps_backend_support_mandatory(backend, changes, output_dtype):
    q = replace(query(), autotune_effort="audit", output_dtype=output_dtype, **changes)
    config = GemvConfig(backend=backend)
    with pytest.raises(ValueError, match="ineligible"):
        TUNING.validate_config(q, config, None)
    assert config not in choices(q)


@pytest.mark.parametrize("n", [240, 304, 368, 432, 496])
@pytest.mark.parametrize("rows", [1, 8])
def test_tensor_core_fp32_joins_existing_projection_candidates(n, rows):
    import torch

    q = replace(query(), max_rows=rows, in_features=4096, out_features=n)
    expected = {
        GemvConfig(backend="simt", rows_per_tile=tile, block_threads=threads)
        for tile in (1, 2, 4, 8)
        for threads in (128, 256)
    }
    expected.add(GemvConfig(backend="tc"))
    if n >= 256:
        expected.add(GemvConfig(backend="mma"))
    for output_dtype in ("bfloat16", "float32"):
        active = replace(q, output_dtype=output_dtype, autotune_effort="default")
        dtype_expected = set(expected)
        if output_dtype == "bfloat16" or hasattr(torch.ops.aten.mm, "dtype_out"):
            dtype_expected.add(GemvConfig(backend="torch"))
        TUNING.validate_config(active, GemvConfig(backend="tc"), None)
        assert choices(active) == dtype_expected
        assert choices(replace(active, autotune_effort="audit")) == dtype_expected | {
            GemvConfig(backend="mma"),
            GemvConfig(backend="prefill"),
        }
        assert TUNING.default_config(active, None).backend == "simt"


def test_default_preserves_projection_defaults():
    q = query()
    assert TUNING.default_config(q, None).backend == "torch"
    # The prefill GEMM lost every race on the shapes it used to be chosen for.
    assert TUNING.default_config(replace(q, out_features=384), None).backend == "torch"
    assert TUNING.default_config(replace(q, max_rows=1), None) == GemvConfig(
        backend="simt", rows_per_tile=2
    )


@pytest.mark.parametrize("output_dtype", ["bfloat16", "float32"])
def test_tensor_core_projection_needs_no_workspace(output_dtype):
    from b12x.gemm.bf16_gemv._preparation import plan

    q = replace(query(output_dtype=output_dtype), max_rows=8)
    requirements = plan(q)._memory_requirements(
        GemvConfig(backend="tc"), SimpleNamespace(ordinal=0)
    )
    assert requirements.scratch_nbytes == 0
    assert not requirements.persistent
