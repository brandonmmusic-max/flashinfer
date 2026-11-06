"""Selected gather plans validate exact layouts and channel-owned receive storage."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.comm.pcie._gather_preparation import plan, selected_gather_profile
from b12x.comm.pcie._tuning import PcieQuery, TUNING
from b12x.preparation import FrozenMapping


def query():
    return PcieQuery(
        surface="DcpAllToAll.all_gather_heads",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                shape=(64, 1, 64),
                stride=(64, 64, 1),
                out_shape=(64, 4, 64),
                out_stride=(256, 64, 1),
                dtype="torch.bfloat16",
                threads=256,
                block_limit=32,
                active_blocks=4,
                submode="barrier",
                gather_transport="joint",
                disjoint_output=True,
                pack_aligned=True,
            )
        ),
        setup=FrozenMapping(
            dict(
                max_batch_size=256,
                total_heads=4,
                query_head_dim=64,
                exchange_chunks=64,
                exchange_offset=131072,
                slot_bytes=262144,
                exchange_counter_offset=524288,
            )
        ),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (64, 64)},
        {"shape": (0, 1, 64)},
        {"dtype": "torch.float32"},
        {"out_stride": (256, 1, 4)},
        {"pack_aligned": False},
        {"disjoint_output": False},
        {"active_blocks": 33},
        {"submode": "pull"},
        {"threads": 96},
        {"shape": (512, 1, 64)},
        {"shape": (64, 2, 64)},
    ],
)
def test_selected_gather_rejects_incompatible_call(changes):
    q = query()
    with pytest.raises(ValueError):
        TUNING.validate_query(
            replace(q, call=FrozenMapping(dict(q.call, **changes))), None
        )


def test_selected_gather_requires_matching_receive_owner():
    q = query()
    TUNING.validate_query(q, None)
    runtime = SimpleNamespace(
        max_batch_size=256,
        total_heads=4,
        query_head_dim=64,
        _exchange_chunks=32,
        _exchange_offset=131072,
        _slot_bytes=262144,
        _exchange_counter_offset=524288,
    )
    with pytest.raises(ValueError, match="exchange_chunks differs"):
        plan(q, runtime=runtime, invocation=FrozenMapping(), override=None)


@pytest.mark.parametrize(
    "transport,mode",
    [
        ("joint", "barrier"),
        ("joint", "word_exchange"),
        ("selected_barrier", "barrier"),
        ("selected_word_exchange", "word_exchange"),
    ],
)
def test_selected_gather_profiles_encode_mode_and_active_grid(transport, mode):
    assert (
        selected_gather_profile(transport, 32, mode, 8)
        == (int(mode == "word_exchange") << 16) | 8
    )
    with pytest.raises(ValueError):
        selected_gather_profile(transport, 32, mode, 33)
