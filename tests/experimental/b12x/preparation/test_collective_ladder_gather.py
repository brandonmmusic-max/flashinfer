"""Gather catalogs preserve row layout, protocol limits, and effective grids."""

import pytest
from b12x.tools._collective_ladder_gather import gather_catalog
from b12x.tools.collective_ladder import arguments


@pytest.mark.parametrize("row_bytes", [128, 512, 1024, 4096, 65536])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_gather_catalog_covers_legal_protocols(row_bytes, world):
    for rows in (1, 7, 64):
        size = rows * row_bytes
        configs = gather_catalog(size, row_bytes, world)
        protocols = {c["submode"] for c in configs}
        assert protocols == (
            {"barrier", "word_exchange"}
            if world in (2, 4) and size <= 16384
            else {"barrier"}
        )
        assert len(configs) == len({tuple(c.items()) for c in configs})
        for config in configs:
            work = (
                rows * ((row_bytes + 511) // 512) * world
                if config["submode"] == "barrier"
                else (size + 255) // 256
            )
            warps = config["threads"] // 32
            assert 1 <= config["active_blocks"] <= min(64, (work + warps - 1) // warps)


def test_gather_cli_requires_exact_rows_and_explicit_geometry(tmp_path):
    base = ["--output", str(tmp_path / "report.jsonl"), "--operation", "all_gather"]
    for flags in (
        [],
        ["--row-bytes", "128", "--alias"],
        ["--row-bytes", "128", "--sizes", "144"],
    ):
        with pytest.raises(SystemExit):
            arguments(base + flags)
    args = arguments(base + ["--row-bytes", "128", "--sizes", "128", "8192"])
    assert args.sizes == (128, 8192) and args.operation == "all_gather"


@pytest.mark.parametrize("kernel", ["runtime", "compare"])
def test_gather_device_profiles_fit_compiled_grid(kernel):
    for size in (128, 8192, 16384, 32768):
        configs = gather_catalog(size, 128, 4, kernel)
        assert len(configs) == len({tuple(c.items()) for c in configs})
        joint = [c for c in configs if c["implementation"] == "gather_joint"]
        assert bool(joint) == (size <= 16384)
        if joint:
            assert {c["submode"] for c in joint} == {"barrier", "word_exchange"}
        for config in configs:
            assert config["active_blocks"] <= config["blocks"]
            if kernel == "runtime":
                assert config["implementation"] not in (
                    "gather_barrier",
                    "gather_word_exchange",
                )


def test_joint_recommendation_retains_topology_specific_modes(tmp_path):
    from b12x.tools._collective_ladder_merge import gather_recommendations
    from b12x.tools._collective_ladder_search import RawReport

    paths = []
    for topology in ("switched", "host"):
        path = tmp_path / f"{topology}.jsonl"
        report = RawReport(
            path,
            dict(
                operation="all_gather",
                gather_kernel="compare",
                row_bytes=128,
                sizes=[8192],
                dtype="bfloat16",
                cache_policy="cold",
                hardware=dict(world_size=4),
                timing_protocol=dict(version=3),
            ),
        )
        configurations = [
            (
                "gather_word_exchange",
                "word_exchange",
                4.7 if topology == "switched" else 25.6,
            ),
            ("gather_barrier", "barrier", 5.5 if topology == "switched" else 17.4),
            (
                "gather_selected_word_exchange",
                "word_exchange",
                5.0 if topology == "switched" else 26.0,
            ),
            ("gather_joint", "word_exchange", 5.05 if topology == "switched" else 26.0),
            ("gather_joint", "barrier", 5.8 if topology == "switched" else 17.7),
        ]
        for family, mode, latency in configurations:
            report.append(
                dict(
                    bytes=8192,
                    config=dict(
                        implementation=family,
                        submode=mode,
                        threads=256,
                        blocks=64,
                        active_blocks=32 if mode == "word_exchange" else 4,
                    ),
                    samples_us=[latency] * 6,
                    correctness="passed",
                    phase="final",
                )
            )
        report.append_summary(dict(status="complete"))
        report.close()
        paths.append(path)
    result = gather_recommendations([paths[0]], [paths[1]])
    bucket = result["buckets"][0]
    assert bucket["implementation"] == "gather_joint"
    assert bucket["profiles"] == {"barrier": [4], "word_exchange": [32]}
    assert bucket["switched_regret"] == pytest.approx(5.05 / 4.7 - 1)
    assert bucket["host_regret"] == pytest.approx(17.7 / 17.4 - 1)


def test_joint_catalog_covers_both_protocols_in_one_physical_grid():
    configs = gather_catalog(8192, 128, 4, "runtime")
    joint = [
        c
        for c in configs
        if c["implementation"] == "gather_joint"
        and c["threads"] == 512
        and c["blocks"] == 32
    ]
    assert {c["submode"] for c in joint} == {"barrier", "word_exchange"}
    assert any(
        c["active_blocks"] == 32 and c["submode"] == "word_exchange" for c in joint
    )


def test_proposal_profiles_are_expanded_and_validated():
    from copy import deepcopy
    from b12x.tools._collective_ladder_gather import proposal_catalog

    proposal = dict(
        operation="all_gather",
        world_size=4,
        cache_policy="cold",
        min_bytes=128,
        buckets=[
            dict(
                max_bytes=2048,
                implementation="gather_joint",
                threads=512,
                blocks=8,
                profiles={"word_exchange": [8], "barrier": [1, 2, 4]},
                references=[
                    dict(
                        implementation="gather_word_exchange",
                        threads=32,
                        blocks=64,
                        submode="word_exchange",
                        active_blocks=8,
                    )
                ],
            )
        ],
    )
    configs = proposal_catalog(proposal, 2048, 128, 4)
    assert len(configs) == 5
    assert {c["implementation"] for c in configs} == {
        "gather_joint",
        "gather_word_exchange",
    }
    assert proposal_catalog(proposal, 2048, 512, 4) == configs
    for size, width, world in [(4096, 128, 4), (2048, 768, 4), (2048, 128, 2)]:
        with pytest.raises(ValueError):
            proposal_catalog(proposal, size, width, world)
    broken = deepcopy(proposal)
    broken["buckets"][0]["profiles"]["barrier"] = [9]
    with pytest.raises(ValueError, match="illegal profile"):
        proposal_catalog(broken, 2048, 128, 4)


def test_gather_merge_pools_same_machine_size_subsets(tmp_path):
    from b12x.tools._collective_ladder_merge import gather_recommendations
    from b12x.tools._collective_ladder_search import RawReport

    def write(name, sizes, blocks, latency, *, capacity=16384, status="complete"):
        path = tmp_path / f"{name}.jsonl"
        report = RawReport(
            path,
            dict(
                operation="all_gather",
                gather_kernel="compare",
                row_bytes=128,
                sizes=sizes,
                capacity_bytes=capacity,
                dtype="bfloat16",
                cache_policy="cold",
                hardware=dict(world_size=4),
                timing_protocol=dict(version=3),
            ),
        )
        for size in sizes:
            report.append(
                dict(
                    bytes=size,
                    config=dict(
                        implementation="gather_joint",
                        submode="word_exchange",
                        threads=512,
                        blocks=blocks,
                        active_blocks=blocks,
                    ),
                    samples_us=[latency] * 102,
                    correctness="passed",
                    phase="final",
                )
            )
        report.append_summary(dict(status=status))
        report.close()
        return path

    base = write("base", [8192, 12288, 16384], 64, 6)
    extra = write("extra", [12288], 48, 5)
    result = gather_recommendations([base, extra], [base, extra])
    assert [row["blocks"] for row in result["buckets"]] == [64, 48, 64]
    incompatible = write("capacity", [12288], 48, 5, capacity=12288)
    with pytest.raises(ValueError, match="matching exact payload sizes"):
        gather_recommendations([base, incompatible], [base])
    incomplete = write("incomplete", [12288], 48, 5, status="partial")
    with pytest.raises(ValueError, match="incomplete recommendations"):
        gather_recommendations([base, incomplete], [base])


def test_gather_capacity_flag_is_operation_specific(tmp_path):
    base = [
        "--output",
        str(tmp_path / "report.jsonl"),
        "--gather-capacity-bytes",
        "8MiB",
    ]
    with pytest.raises(SystemExit):
        arguments(base)
    args = arguments(base + ["--operation", "all_gather", "--row-bytes", "128"])
    assert args.gather_capacity_bytes == 8 << 20


def test_gather_recommendation_shares_geometry_across_row_widths(tmp_path):
    from b12x.tools._collective_ladder_merge import gather_recommendations
    from b12x.tools._collective_ladder_search import RawReport

    paths = {"switched": [], "host": []}
    for width in (128, 512):
        for topology in paths:
            path = tmp_path / f"{width}-{topology}.jsonl"
            report = RawReport(
                path,
                dict(
                    operation="all_gather",
                    gather_kernel="compare",
                    row_bytes=width,
                    sizes=[8192],
                    dtype="bfloat16",
                    cache_policy="cold",
                    hardware=dict(world_size=4),
                    timing_protocol=dict(version=3),
                ),
            )
            for threads in (128, 256):
                latency = (
                    (5 if width == 128 or threads == 256 else 8)
                    if topology == "switched"
                    else 10
                )
                report.append(
                    dict(
                        bytes=8192,
                        config=dict(
                            implementation="gather_joint",
                            threads=threads,
                            blocks=32,
                            submode="word_exchange"
                            if topology == "switched"
                            else "barrier",
                            active_blocks=32 if topology == "switched" else 2,
                        ),
                        samples_us=[latency] * 102,
                        correctness="passed",
                        phase="final",
                    )
                )
            report.append_summary(dict(status="complete"))
            report.close()
            paths[topology].append(path)
    result = gather_recommendations(paths["switched"], paths["host"])
    assert set(result) == {
        "operation",
        "world_size",
        "dtype",
        "cache_policy",
        "min_bytes",
        "buckets",
    }
    (bucket,) = result["buckets"]
    assert bucket["threads"] == 256
    assert bucket["profiles"] == {"barrier": [2], "word_exchange": [32]}
    assert bucket["switched_regret"] == bucket["host_regret"] == 0
