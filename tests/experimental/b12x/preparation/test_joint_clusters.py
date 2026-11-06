"""Cluster geometry is opt-in metadata; ordinary profiles retain their identity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import FrozenMapping
from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
from b12x.comm.pcie._joint_profile import (
    JOINT_FAMILIES,
    JOINT_SIGNAL_BYTES,
    joint_modes,
    profile_records,
    selected_profile,
    validate_cluster_geometry,
)
from b12x.comm.pcie._tuning import PcieQuery, TUNING as NATIVE


def native_query(family="joint_latency", *, cluster_size=1, mode="full", active=1):
    call = dict(
        transport=family,
        threads=512,
        blocks=64,
        submode=mode,
        active_blocks=active,
        dtype="bfloat16",
        stage_input=True,
        size_packs=10240,
    )
    if cluster_size != 1:
        call["joint_cluster_size"] = cluster_size
    return PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(call),
        setup=FrozenMapping(
            dict(
                shape=(81920,),
                joint_max_blocks=64,
                rank_data_nbytes=256,
                owned_slab_nbytes=512,
                owned_slab_count=1,
                joint_workspace_nbytes=4096,
            )
        ),
    )


def collective_query(native):
    c = native.call
    return CollectiveQuery(
        operation="all_reduce",
        world_size=4,
        rank=0,
        alternatives=FrozenMapping(
            {
                c["transport"]: dict(
                    surface=native.surface,
                    world_size=native.world_size,
                    rank=native.rank,
                    topology=native.topology,
                    call=native.call.to_dict(),
                    setup=native.setup.to_dict(),
                )
            }
        ),
        default=CollectiveConfig(
            implementation=c["transport"],
            threads=c["threads"],
            blocks=c["blocks"],
            submode=c["submode"],
            active_blocks=c["active_blocks"],
        ),
    )


def test_ordinary_contract_and_exact_profile_records_are_unchanged():
    assert (
        TUNING.query_schema_version,
        TUNING.candidate_contract_version,
        TUNING.config_schema_version,
    ) == (7, 3, 2)
    assert NATIVE.query_schema_version == 42
    assert tuple(CollectiveConfig.__dataclass_fields__) == (
        "implementation",
        "threads",
        "blocks",
        "submode",
        "active_blocks",
    )
    assert JOINT_FAMILIES == {
        "joint_latency": ("full", "payload_rsag", "pull"),
        "joint_bandwidth": ("payload_rsag", "bulk_push", "bulk_pull", "pull"),
    }
    for family, modes in {
        "joint_latency": (0, 1, 4),
        "joint_bandwidth": (1, 2, 3, 4),
    }.items():
        expected = tuple((mode, active) for mode in modes for active in range(1, 65))
        assert profile_records(family, 64) == expected
        assert profile_records(family, 64, cluster_size=1) == expected
        assert profile_records(family, 64, cluster_size=8)[: len(expected)] == expected


@pytest.mark.parametrize(
    "family,mode,active",
    [
        ("joint_latency", "full", 25),
        ("joint_latency", "pull", 7),
        ("joint_latency", "payload_rsag", 64),
        ("joint_bandwidth", "bulk_pull", 11),
        ("joint_bandwidth", "payload_rsag", 42),
        ("joint_bandwidth", "bulk_pull", 37),
        ("joint_latency", "full_phased", 8),
        ("joint_latency", "payload_rsag_phased", 8),
        ("joint_bandwidth", "payload_rsag_phased", 8),
    ],
)
def test_cluster_config_admits_runtime_profiles_without_candidate_axis_loss(
    family, mode, active
):
    q = collective_query(native_query(family, cluster_size=8, mode=mode, active=active))
    TUNING.validate_query(q, None)
    TUNING.validate_config(q, q.default, None)
    assert selected_profile(family, 64, mode, active, cluster_size=8) & 0xFFFF == active


@pytest.mark.parametrize("active", [0, 1, 7, 9, 16, 32, 64, True])
def test_partial_or_multiple_phase_clusters_are_rejected(active):
    for family in JOINT_FAMILIES:
        with pytest.raises(ValueError):
            selected_profile(family, 64, "payload_rsag_phased", active, cluster_size=8)


def test_phase_profiles_are_opt_in_and_family_specific():
    for family in JOINT_FAMILIES:
        with pytest.raises(ValueError):
            selected_profile(family, 64, "payload_rsag_phased", 8)
    with pytest.raises(ValueError):
        selected_profile("joint_bandwidth", 64, "full_phased", 8, cluster_size=8)
    assert joint_modes("joint_latency", cluster_size=8) == (
        *JOINT_FAMILIES["joint_latency"],
        "full_phased",
        "payload_rsag_phased",
    )


@pytest.mark.parametrize(
    "blocks,threads,cluster",
    [
        (8, 512, 8),
        (32, 512, 8),
        (63, 512, 8),
        (64, 256, 8),
        (64, 512, 16),
        (64, 512, True),
    ],
)
def test_unsupported_cluster_geometry_rejected(blocks, threads, cluster):
    with pytest.raises(ValueError):
        validate_cluster_geometry(blocks, threads, cluster)


@pytest.mark.parametrize("cc,sms", [((9, 0), 132), ((12, 0), 63), ((12, 2), 188)])
def test_unsupported_cluster_devices_rejected(cc, sms):
    with pytest.raises(ValueError):
        validate_cluster_geometry(
            64, 512, 8, SimpleNamespace(compute_capability=cc, sm_count=sms)
        )


@pytest.mark.parametrize(
    "surface,transport",
    [
        ("OneshotAllReduce.all_reduce_fused_add", "joint_latency"),
        ("OneshotAllReduce.all_reduce", "pull"),
    ],
)
def test_cluster_metadata_cannot_escape_joint_plain_surface(surface, transport):
    q = native_query(cluster_size=8)
    q = replace(
        q, surface=surface, call=FrozenMapping({**q.call, "transport": transport})
    )
    with pytest.raises(ValueError, match="staged joint BF16"):
        NATIVE.validate_query(q, None)


def test_cluster_compile_job_and_workspace_keep_standard_preparation_ownership():
    from b12x.comm.pcie import _joint_preparation as preparation
    from b12x._lib.compile_pool import CompileJob

    runtime = SimpleNamespace(device=torch.device("cpu"))
    ordinary = preparation.plan(native_query(), runtime=runtime)
    clustered = preparation.plan(native_query(cluster_size=8), runtime=runtime)
    (old,) = ordinary._compile_jobs(None, None)
    (new,) = clustered._compile_jobs(None, None)
    assert old == CompileJob.create(
        "b12x.comm.pcie._joint_preparation:compile_joint",
        "latency",
        4,
        0,
        512,
        64,
        CompileJob.DEVICE_ORDINAL,
    )
    assert new == CompileJob.create(
        "b12x.comm.pcie._joint_preparation:compile_joint_cluster",
        "latency",
        4,
        0,
        512,
        64,
        8,
        CompileJob.DEVICE_ORDINAL,
    )
    assert [
        x.resident_nbytes for x in ordinary._memory_requirements(None, None).persistent
    ] == [
        x.resident_nbytes for x in clustered._memory_requirements(None, None).persistent
    ]
    signals, slot, total = preparation._layout(2293760 // 16, 4)
    assert JOINT_SIGNAL_BYTES == 266248
    assert signals == 266496 and total == signals + 2 * slot


def test_cluster_materialization_unwraps_detected_device_identity(monkeypatch):
    from b12x.preparation import DetectedDevice, DeviceIdentity
    from b12x.comm.pcie import _joint_preparation as preparation
    from b12x.comm.pcie import _joint_cluster_cute as kernels

    identity = DeviceIdentity("nvidia", (12, 0), 188, "SM120 test device")
    device = DetectedDevice(ordinal=3, identity=identity)
    workspace, launcher = object(), object()
    runtime = SimpleNamespace(device=torch.device("cpu"), _joint_workspace=workspace)
    compiled, checked = [], []
    monkeypatch.setattr(
        preparation,
        "compile_joint_cluster",
        lambda *args: compiled.append(args) or launcher,
    )
    monkeypatch.setattr(preparation, "load_programs", lambda value: value)
    monkeypatch.setattr(
        kernels, "validate_cluster_residency", lambda *args: checked.append(args)
    )
    monkeypatch.setattr(preparation, "_JointExecutionState", lambda *args: args)
    query = native_query(cluster_size=8)
    plan = preparation.plan(query, runtime=runtime)
    assert plan._materialize(None, device) == (query, runtime, launcher, workspace)
    assert compiled == [("latency", 4, 0, 512, 64, 8, 3)]
    assert checked == [(launcher, 64, 512, 8)]


def test_startup_keys_and_phase_admission_use_immutable_cluster_geometry(monkeypatch):
    from b12x.comm.pcie._joint_preparation import _JointExecutionState
    from b12x.comm.pcie import joint_startup_spec

    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )

    def prepared(cluster):
        state = _JointExecutionState.__new__(_JointExecutionState)
        state.query = native_query(cluster_size=cluster)
        state.workspace = SimpleNamespace(capacity_packs=143360)
        state.selector = torch.zeros(1, dtype=torch.uint32)
        state.runtime = SimpleNamespace(exchange_group=object())
        return SimpleNamespace(
            prepared=SimpleNamespace(state=state, finalists=()),
            query=SimpleNamespace(inplace=False),
        )

    plain = prepared(1)
    clustered = prepared(8)
    a = joint_startup_spec(plain, profiles=(("pull", 7),))
    b = joint_startup_spec(clustered, profiles=(("pull", 7),))
    assert a.key == (
        "joint_startup",
        1,
        4,
        (0, 1, 2, 3),
        "joint_latency",
        512,
        64,
        (81920,),
        "bfloat16",
        10240,
        143360,
        False,
        ((4, 7),),
    )
    assert b.key == (*a.key, ("cluster_size", 8))
    assert joint_startup_spec(
        clustered, profiles=(("payload_rsag_phased", 8), ("payload_rsag", 64))
    ).profiles == ((6, 8), (1, 64))
    with pytest.raises(ValueError):
        joint_startup_spec(clustered, profiles=(("payload_rsag_phased", 7),))


def test_public_plan_omits_default_cluster_field(monkeypatch):
    from b12x.comm.pcie import _preparation, _collective_tuning
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce

    native = native_query()
    runtime = SimpleNamespace(device=torch.device("cpu"))
    initial = replace(native, call=FrozenMapping({**native.call, "transport": "pull"}))
    monkeypatch.setattr(_preparation, "query_from_runtime", lambda *a, **k: initial)
    monkeypatch.setattr(
        _collective_tuning,
        "oneshot_alternatives",
        lambda *a: {"joint_latency": (native, runtime)},
    )
    monkeypatch.setattr(
        _collective_tuning, "plan", lambda alternatives, **kwargs: alternatives
    )
    manager = PCIeAllReduce.__new__(PCIeAllReduce)
    manager.algorithm = "oneshot"
    manager._bf16_twoshot = None
    manager._runtime = SimpleNamespace(
        single_channel=True, _prepared_channel_for_stream=lambda *a: runtime
    )
    inp = torch.zeros(16, dtype=torch.bfloat16)
    omitted = manager.plan(inp, joint_only=True)
    explicit = manager.plan(inp, joint_only=True, joint_cluster_size=1)
    assert omitted == explicit == {"joint_latency": (native, runtime)}
    clustered = manager.plan(inp, joint_only=True, joint_cluster_size=8)[
        "joint_latency"
    ][0]
    assert clustered.call == FrozenMapping({**native.call, "joint_cluster_size": 8})
    for kwargs in [
        dict(joint_only=False),
        dict(joint_only=True, operation="all_reduce_fused_add"),
    ]:
        with pytest.raises(ValueError, match="plain all_reduce"):
            manager.plan(inp, joint_cluster_size=8, **kwargs)
