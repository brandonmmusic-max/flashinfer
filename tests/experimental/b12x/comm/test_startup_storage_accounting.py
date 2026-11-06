"""Startup reports include retained owner storage and deduplicate aliases."""

import gc
from types import SimpleNamespace
import weakref

import torch

from b12x.comm.pcie import StartupTuningStorage
from b12x.comm.pcie._joint_startup import JointStartupTuner
from b12x.comm.pcie._joint_startup_fixtures import (
    StartupFixtures,
    StartupGatherInputs,
    StartupReductionInputs,
)
from b12x.preparation import MemoryRequirements


def owner():
    storage = StartupTuningStorage("cuda:0")
    # Accounting only inspects tensor storage; CPU tensors exercise the exact
    # storage-key rules without constructing CUDA resources or a communicator.
    storage._fixtures = StartupFixtures(torch.device("cpu"))
    return storage


def producer(storage, rows, *, gather=False, cold=False):
    state = SimpleNamespace(
        runtime=SimpleNamespace(device=torch.device("cpu")),
        query=SimpleNamespace(
            world_size=4,
            rank=1,
            setup={"shape": (rows, 16)},
            call={
                "shape": (rows, 2, 8),
                "out_shape": (rows, 8, 8),
                "dtype": "torch.bfloat16",
            },
        ),
    )
    result = JointStartupTuner.__new__(JointStartupTuner)
    result.storage = storage
    result.fixtures = storage._fixtures
    result.inputs = (
        StartupGatherInputs(result.fixtures, state)
        if gather
        else StartupReductionInputs(
            result.fixtures, state, inplace=False, dtype=torch.bfloat16, repetitions=1
        )
    )
    result.peers = torch.empty(4, dtype=torch.uint64)
    result.status = torch.empty(1, dtype=torch.int32)
    result.samples = torch.empty((2, 21), dtype=torch.int64)
    result.profiles = torch.empty(2, dtype=torch.uint32)
    result.invalid = torch.empty(1, dtype=torch.int32)
    result.eviction = storage._eviction if cold else None
    result.eviction_sum = storage._eviction_sum if cold else None
    result.shared = SimpleNamespace(local_ptr=1234, nbytes=64)
    return result


def visible_tensors(producer):
    inputs = producer.inputs
    return (
        inputs.source,
        inputs.out,
        *inputs.reference.seeds,
        *inputs.reference.expected,
        producer.peers,
        producer.status,
        producer.samples,
        producer.profiles,
        producer.invalid,
        producer.mismatch,
        producer.eviction,
        producer.eviction_sum,
    )


def retained_fixture_tensors(storage):
    fixtures = storage._fixtures
    return (
        fixtures.scratch,
        *(
            tensor
            for bank in fixtures.banks.values()
            for tensor in (*bank.seeds, *bank.expected)
        ),
    )


def allocations(tensors):
    return {
        (
            "pcie-joint-startup-tensor",
            str(tensor.device),
            tensor.untyped_storage().data_ptr(),
        ): tensor.untyped_storage().nbytes()
        for tensor in tensors
        if tensor is not None
    }


def reported(producer):
    return {
        entry.key: entry.resident_nbytes
        for entry in producer.memory_requirements().persistent
        if entry.key[0] == "pcie-joint-startup-tensor"
    }


def test_shared_fixtures_and_precapture_aggregate_keep_existing_totals():
    storage = owner()
    storage._eviction = torch.empty(128, dtype=torch.float32)
    storage._eviction_sum = torch.empty((), dtype=torch.float32)
    producers = [
        producer(storage, 3, cold=True),
        producer(storage, 32, cold=True),
        producer(storage, 3, gather=True, cold=True),
        producer(storage, 32, gather=True, cold=True),
    ]
    retained = allocations(retained_fixture_tensors(storage))
    for current in producers:
        report = reported(current)
        assert all(report[key] == size for key, size in retained.items())
        assert len(report) == len(current.memory_requirements().persistent) - 1
    # Flare reports all producers together before capture. Their inputs already
    # cover every shared bank, so adding owner-retained storage changes no total.
    expected = allocations(
        tensor for current in producers for tensor in visible_tensors(current)
    )
    expected[("pcie-joint-startup-ipc", 1234)] = 64
    combined = MemoryRequirements.sequential(
        current.memory_requirements() for current in producers
    )
    assert {
        entry.key: entry.resident_nbytes for entry in combined.persistent
    } == expected


def test_frozen_private_growth_reports_owner_after_shared_users_disappear():
    storage = owner()
    small = producer(storage, 32)
    old_arena = weakref.ref(small.inputs.source.untyped_storage())
    retained = allocations(retained_fixture_tensors(storage))
    storage._freeze()
    large = producer(storage, 64)
    del small
    gc.collect()
    assert len(storage._fixtures.users) == 0
    assert old_arena() is not None
    private = allocations(visible_tensors(large))
    assert not retained.keys() & private.keys()
    assert reported(large) == private | retained


def test_warm_producer_reports_owner_internal_eviction_after_cold_user_drops():
    storage = owner()
    storage._eviction = torch.empty(128, dtype=torch.float32)
    storage._eviction_sum = torch.empty((), dtype=torch.float32)
    cold = producer(storage, 32, cold=True)
    warm = producer(storage, 3)
    del cold
    gc.collect()
    assert warm.eviction is warm.eviction_sum is None
    expected = allocations(
        (
            *visible_tensors(warm),
            *retained_fixture_tensors(storage),
            storage._eviction,
            storage._eviction_sum,
        )
    )
    assert reported(warm) == expected


def test_external_eviction_is_excluded_and_its_internal_scalar_is_reported():
    storage = owner()
    source = torch.empty(256, dtype=torch.float32)
    storage._eviction = source[:128]
    storage._eviction_sum = torch.empty((), dtype=torch.float32)
    storage.external_eviction = True
    current = producer(storage, 32, cold=True)
    source_key = next(iter(allocations((source,))))
    scalar_key = next(iter(allocations((storage._eviction_sum,))))
    report = reported(current)
    assert source_key not in report
    assert report[scalar_key] == storage._eviction_sum.untyped_storage().nbytes()
    assert report == allocations(
        (
            *visible_tensors(current)[:-2],
            *retained_fixture_tensors(storage),
            storage._eviction_sum,
        )
    )
