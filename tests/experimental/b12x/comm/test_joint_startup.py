"""Prepared mode retention uses the established finalist cache contract."""

from b12x.comm.pcie import joint_finalist_policy, joint_startup_spec
import pytest


def test_joint_finalist_policy_preserves_mode_and_compiled_geometry():
    policy = joint_finalist_policy()
    assert policy.family_fields == ("implementation", "threads", "blocks", "submode")
    assert policy.per_family == 1
    assert policy.max_total == 35


def test_unprepared_joint_startup_rejected():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="prepared joint"):
        joint_startup_spec(SimpleNamespace(prepared=None))


@pytest.fixture
def prepared_joint(monkeypatch):
    from types import SimpleNamespace

    import torch
    from b12x.comm.pcie._joint_preparation import _JointExecutionState

    state = _JointExecutionState.__new__(_JointExecutionState)
    state.query = SimpleNamespace(
        call=dict(
            transport="joint_bandwidth",
            threads=512,
            blocks=64,
            dtype="bfloat16",
            size_packs=65536,
        ),
        setup=dict(shape=(512, 1024)),
        world_size=4,
        rank=0,
    )
    state.workspace = SimpleNamespace(capacity_packs=65536)
    state.selector = torch.zeros(1, dtype=torch.uint32)
    state.runtime = SimpleNamespace(exchange_group=object())
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )
    return SimpleNamespace(
        prepared=SimpleNamespace(state=state, finalists=()),
        query=SimpleNamespace(inplace=False),
    )


def test_explicit_runtime_profiles_need_no_measured_finalists(prepared_joint):
    spec = joint_startup_spec(
        prepared_joint,
        profiles=(("bulk_push", 32), ("bulk_push", 64), ("bulk_pull", 8)),
    )
    assert spec.profiles == ((2, 32), (2, 64), (3, 8))
    assert spec.selector is prepared_joint.prepared.state.selector
    assert spec.state is prepared_joint.prepared.state
    other = joint_startup_spec(prepared_joint, profiles=(("bulk_push", 32),))
    assert spec.key != other.key
    with pytest.raises(ValueError, match="measured finalists"):
        joint_startup_spec(prepared_joint)


@pytest.mark.parametrize(
    "profiles",
    [
        (),
        (("bulk_push", 0),),
        (("bulk_push", 65),),
        (("bulk_push", True),),
        (("full", 1),),
        (("bulk_pull", 8), ("bulk_pull", 8)),
        ("bulk_pull",),
        (("bulk_pull",),),
    ],
)
def test_invalid_explicit_runtime_profiles(prepared_joint, profiles):
    with pytest.raises(ValueError):
        joint_startup_spec(prepared_joint, profiles=profiles)


@pytest.mark.parametrize("kind", ("reduction", "gather"))
def test_startup_references_of_fewer_rows_are_leading_rows(kind):
    import torch
    from b12x.comm.pcie._joint_startup_fixtures import _gather_bank, _reduction_bank

    device = torch.device("cpu")
    if kind == "reduction":
        small, large = (
            _reduction_bank(4, 1, rows, (16,), torch.bfloat16, device)
            for rows in (3, 8)
        )
        leading = lambda tensor, rows: tensor[:rows]
    else:
        small, large = (_gather_bank(4, 2, rows, 24, device) for rows in (3, 8))
        leading = lambda tensor, rows: tensor[: rows * tensor.numel() // 8]
    for case in range(3):
        assert torch.equal(small.seeds[case], leading(large.seeds[case], 3))
        assert torch.equal(small.expected[case], leading(large.expected[case], 3))
    assert large.seeds[0].abs().sum() > 0


@pytest.mark.parametrize(
    "kind,inplace", [("reduction", False), ("reduction", True), ("gather", False)]
)
def test_fixture_growth_preserves_captured_addresses(kind, inplace):
    from types import SimpleNamespace
    import torch
    from b12x.comm.pcie._joint_startup_fixtures import (
        StartupFixtures,
        StartupGatherInputs,
        StartupReductionInputs,
    )

    fixtures = StartupFixtures(torch.device("cpu"))

    def inputs(rows):
        state = SimpleNamespace(
            runtime=SimpleNamespace(device=torch.device("cpu")),
            query=SimpleNamespace(
                world_size=4,
                rank=1,
                setup={"shape": (rows, 16)},
                call={
                    "shape": (rows, 2, 8),
                    "out_shape": (rows, 8, 8),
                    "dtype": "torch.float32",
                },
            ),
        )
        if kind == "gather":
            return StartupGatherInputs(fixtures, state)
        return StartupReductionInputs(
            fixtures, state, inplace=inplace, dtype=torch.bfloat16, repetitions=1
        )

    small = inputs(3)
    source = small.reference.seeds[0].clone()
    previous = small.source.data_ptr()
    large = inputs(32)
    assert small.source.data_ptr() != previous
    assert small.source.data_ptr() == large.source.data_ptr()
    assert torch.equal(small.reference.seeds[0], source)
    assert (small.source.data_ptr() == small.out.data_ptr()) is inplace
    fixtures.frozen = True
    pointers = (
        small.source.data_ptr(),
        small.reference.seeds[0].data_ptr(),
        large.source.data_ptr(),
        large.reference.expected[0].data_ptr(),
    )
    larger = inputs(64)
    assert larger.source.data_ptr() != small.source.data_ptr()
    assert pointers == (
        small.source.data_ptr(),
        small.reference.seeds[0].data_ptr(),
        large.source.data_ptr(),
        large.reference.expected[0].data_ptr(),
    )
    for current in (larger, small, large):
        current.produce(2)
        assert torch.equal(
            current.source.view(torch.uint8),
            current.reference.seeds[2].view(torch.uint8),
        )


def test_startup_storage_is_independent_and_lazy():
    import torch
    from b12x.comm.pcie import StartupTuningStorage

    first, second = (StartupTuningStorage(torch.device("cuda", 0)) for _ in range(2))
    assert first is not second
    assert first._eviction is second._eviction is None
    assert first._fixtures is second._fixtures is None
    first._freeze()
    assert first.captured and not second.captured
    with pytest.raises(ValueError, match="CUDA device"):
        StartupTuningStorage("cpu")


def test_default_startup_storage_shares_one_runtime_and_releases_with_it():
    import gc
    from types import SimpleNamespace
    import weakref
    import torch
    from b12x.comm.pcie import StartupTuningStorage
    from b12x.comm.pcie._joint_startup import _runtime_startup_storage

    runtime = SimpleNamespace(device=torch.device("cuda:0"))
    first = _runtime_startup_storage(runtime)
    assert _runtime_startup_storage(runtime) is first
    other_runtime = SimpleNamespace(device=torch.device("cuda:0"))
    assert _runtime_startup_storage(other_runtime) is not first
    assert StartupTuningStorage("cuda:0") is not first
    assert first._fixtures is first._eviction is None
    retained = weakref.ref(first)
    del first, runtime
    gc.collect()
    assert retained() is None


@pytest.mark.parametrize(
    "case,match",
    [
        ("cpu", "CUDA tensor"),
        ("device", "CUDA tensor"),
        ("strided", "contiguous"),
        ("alignment", "aligned"),
        ("small", "smaller"),
    ],
)
def test_invalid_eviction_source_fails_without_allocation(monkeypatch, case, match):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    from b12x.comm.pcie import StartupTuningStorage

    storage = StartupTuningStorage("cuda:0")
    source = Mock(spec=torch.Tensor)
    source.device = torch.device(
        "cpu" if case == "cpu" else "cuda:1" if case == "device" else "cuda:0"
    )
    source.is_contiguous.return_value = case != "strided"
    source.data_ptr.return_value = 1 if case == "alignment" else 256
    source.numel.return_value = 1
    source.element_size.return_value = 4
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(L2_cache_size=128 << 20),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid eviction input allocated storage")

    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError, match=match):
        storage.use_eviction(source)
    assert storage._eviction is storage._eviction_sum is None
    assert not storage.external_eviction
    source.view.assert_not_called()


def test_captured_eviction_source_is_rejected_before_inspecting_input():
    from b12x.comm.pcie import StartupTuningStorage

    storage = StartupTuningStorage("cuda:0")
    storage._freeze()
    with pytest.raises(RuntimeError, match="after capture"):
        storage.use_eviction(object())
