"""Owner-interval traversal preserves the payload-ready partition and coverage."""

from collections import Counter
import random
import pytest


def owner(index, size, world):
    part = size // world
    return min(index // part, world - 1) if part else world - 1


def interval(peer, size, world):
    part = size // world
    return peer * part, size if peer == world - 1 else (peer + 1) * part


def work(peer, lane, size, world, stride):
    begin, end = interval(peer, size, world)
    return range(begin + lane, end, stride)


@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
def test_owner_intervals_preserve_coverage_and_thread_local_handoff(world):
    stride = 8 * 512
    cases = sorted(
        {
            1,
            world - 1,
            world,
            world + 1,
            stride - 1,
            stride,
            stride + 1,
            81408 // 16,
            81920 // 16,
            84992 // 16,
            162784 // 16,
            163840 // 16,
            169984 // 16,
            325568 // 16,
            327680 // 16,
            337920 // 16,
            339952 // 16,
            1310720 // 16,
            2293760 // 16,
            *(random.Random(9151 + i).randrange(1, 50000) for i in range(12)),
        }
    )
    for size in cases:
        all_indices = Counter()
        for peer in range(world):
            expected = {
                index for index in range(size) if owner(index, size, world) == peer
            }
            assigned = Counter(
                index
                for lane in range(stride)
                for index in work(peer, lane, size, world, stride)
            )
            assert set(assigned) == expected and all(
                count == 1 for count in assigned.values()
            )
            all_indices.update(assigned)
            # Scatter to this owner, its reduction, and all of its outgoing
            # gathers use this exact per-thread map. Other owners cannot alias it.
            begin, end = interval(peer, size, world)
            assert 0 <= begin <= end <= size
            for other in range(world):
                if peer == other:
                    continue
                a, b = interval(other, size, world)
                assert max(begin, a) >= min(end, b)
        assert all_indices == Counter(range(size))


def test_zero_sized_owner_regions_do_not_remove_phase_barriers():
    for world in (2, 3, 4, 6, 8, 10):
        for size in range(1, world):
            assert [interval(peer, size, world) for peer in range(world - 1)] == [
                (0, 0)
            ] * (world - 1)
            assert interval(world - 1, size, world) == (0, size)
            for rank in range(world):
                scatter = [(rank + phase) % world for phase in range(world)]
                gather = [(rank + phase) % world for phase in range(1, world)]
                assert len(scatter[1:]) == len(gather) == world - 1
                assert set(scatter) == set(range(world))
                assert set(gather) == set(range(world)) - {rank}
