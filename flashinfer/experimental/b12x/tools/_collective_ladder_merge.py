"""Combine switched and host gather measurements into shared compiled geometries."""

from collections import defaultdict
from itertools import combinations
import json
import statistics

from ._collective_ladder_search import read_report


def gather_recommendations(switched, host, *, switched_margin=0.02):
    """Minimize host cost within a margin of the fastest device-selected kernel.

    Static comparison kernels do not participate in the runtime geometry choice.
    Reported regret includes them when present, so selector overhead stays visible.
    """
    if not switched or not host or not 0 <= switched_margin <= 0.05:
        raise ValueError(
            "gather recommendations require switched and host reports and a margin in [0,.05]"
        )
    cohorts = {}
    for topology, paths in (("switched", switched), ("host", host)):
        for path in paths:
            metadata, records, summaries = read_report(path)
            if metadata.get("operation") != "all_gather":
                raise ValueError(f"expected an all-gather report: {path}")
            if metadata.get("gather_kernel") not in ("runtime", "compare"):
                raise ValueError(
                    f"recommendations require device-selected gather measurements: {path}"
                )
            if not summaries or summaries[-1]["status"] != "complete":
                raise ValueError(f"report has incomplete recommendations: {path}")
            capacity = metadata.get("capacity_bytes", max(metadata["sizes"]))
            identity = json.dumps(
                dict(
                    topology=topology,
                    hardware=metadata["hardware"],
                    versions=metadata.get("versions"),
                    row_bytes=metadata["row_bytes"],
                    capacity_bytes=capacity,
                    dtype=metadata["dtype"],
                    cache_policy=metadata["cache_policy"],
                    timing_protocol=metadata["timing_protocol"],
                ),
                sort_keys=True,
            )
            if identity not in cohorts:
                cohorts[identity] = (topology, metadata, set(), defaultdict(list), {})
            _, _, sizes, samples, phases = cohorts[identity]
            sizes.update(metadata["sizes"])
            for record in records:
                if record.get("correctness") != "passed" or record.get("error"):
                    raise ValueError(f"report contains a failed candidate: {path}")
                key = record["bytes"], json.dumps(record["config"], sort_keys=True)
                priority = {"final": 3, "replay": 3, "confirmation": 2}.get(
                    record.get("phase"), 1
                )
                if priority > phases.get(key, 0):
                    samples[key] = []
                    phases[key] = priority
                if priority == phases[key]:
                    samples[key].extend(record["samples_us"])
    reports = []
    for topology, metadata, sizes, samples, _ in cohorts.values():
        rows = defaultdict(dict)
        for (size, encoded), values in samples.items():
            config = json.loads(encoded)
            geometry = config["implementation"], config["threads"], config["blocks"]
            value = statistics.median(values)
            previous = rows[size].get(geometry)
            profiles = {} if previous is None else previous[3]
            profiles[config["submode"], config["active_blocks"]] = value
            if previous is None or value < previous[0]:
                rows[size][geometry] = (
                    value,
                    config["submode"],
                    config["active_blocks"],
                    profiles,
                )
        if set(rows) != sizes:
            raise ValueError("report contains unmeasured payload sizes")
        reports.append((topology, dict(metadata, sizes=sorted(sizes)), rows))
    contracts = {
        (
            m["hardware"]["world_size"],
            m["dtype"],
            m["cache_policy"],
            json.dumps(m["timing_protocol"], sort_keys=True),
        )
        for _, m, _ in reports
    }
    if len(contracts) != 1:
        raise ValueError(
            "report timing protocols, TP sizes, cache policies, and dtypes must match"
        )
    for row_bytes in sorted({m["row_bytes"] for _, m, _ in reports}):
        cohort = [
            (topology, metadata, rows)
            for topology, metadata, rows in reports
            if metadata["row_bytes"] == row_bytes
        ]
        if {kind for kind, _, _ in cohort} != {"switched", "host"}:
            raise ValueError(f"row width {row_bytes} needs both topology cohorts")
        sizes = cohort[0][1]["sizes"]
        if any(metadata["sizes"] != sizes for _, metadata, _ in cohort):
            raise ValueError("each row width requires matching exact payload sizes")
    buckets = []
    for size in sorted(
        {size for _, metadata, _ in reports for size in metadata["sizes"]}
    ):
        cohort = [
            (kind, metadata, rows) for kind, metadata, rows in reports if size in rows
        ]
        measurements = [rows[size] for _, _, rows in cohort]
        shared = set.intersection(*(set(row) for row in measurements))
        shared = {
            geometry
            for geometry in shared
            if geometry[0]
            in (
                "gather_joint",
                "gather_selected_barrier",
                "gather_selected_word_exchange",
            )
        }
        if not shared:
            raise ValueError(
                f"no common measured geometry across row widths at {size=}"
            )
        baseline = [min(value[0] for value in row.values()) for row in measurements]
        best = [
            min(
                value[0]
                for geometry, value in row.items()
                if geometry[0]
                in (
                    "gather_joint",
                    "gather_selected_barrier",
                    "gather_selected_word_exchange",
                )
            )
            for row in measurements
        ]

        def costs(geometry, references=best):
            ratios = [
                row[geometry][0] / fastest
                for row, fastest in zip(measurements, references)
            ]
            switched_cost = max(
                ratio
                for ratio, (kind, _, _) in zip(ratios, cohort)
                if kind == "switched"
            )
            host_cost = max(
                ratio for ratio, (kind, _, _) in zip(ratios, cohort) if kind == "host"
            )
            return switched_cost, host_cost

        acceptable = [
            geometry for geometry in shared if costs(geometry)[0] <= 1 + switched_margin
        ]
        geometry = min(
            acceptable or shared,
            key=lambda geometry: (
                (costs(geometry)[1], costs(geometry)[0], geometry)
                if acceptable
                else (*costs(geometry), geometry)
            ),
        )
        candidates = sorted(
            {profile for row in measurements for profile in row[geometry][3]}
        )
        limits = [
            max(row[geometry][0], fastest * (1 + switched_margin))
            if kind == "switched"
            else row[geometry][0] * 1.05
            for (kind, _, _), row, fastest in zip(cohort, measurements, best)
        ]

        def latencies(menu):
            return [
                min(row[geometry][3].get(profile, float("inf")) for profile in menu)
                for row in measurements
            ]

        for count in range(1, len(candidates) + 1):
            menus = [
                menu
                for menu in combinations(candidates, count)
                if all(value <= limit for value, limit in zip(latencies(menu), limits))
            ]
            if menus:
                menu = min(
                    menus,
                    key=lambda menu: (
                        max(
                            value / fastest
                            for value, fastest in zip(latencies(menu), best)
                        ),
                        menu,
                    ),
                )
                break
        profiles = defaultdict(list)
        for mode, active in menu:
            profiles[mode].append(active)
        ratios = [value / fastest for value, fastest in zip(latencies(menu), baseline)]
        switch_cost = max(
            value for value, (kind, _, _) in zip(ratios, cohort) if kind == "switched"
        )
        host_cost = max(
            value for value, (kind, _, _) in zip(ratios, cohort) if kind == "host"
        )
        buckets.append(
            dict(
                max_bytes=size,
                implementation=geometry[0],
                threads=geometry[1],
                blocks=geometry[2],
                profiles={
                    mode: sorted(counts) for mode, counts in sorted(profiles.items())
                },
                switched_regret=switch_cost - 1,
                host_regret=host_cost - 1,
            )
        )
    world, dtype, cache, _ = next(iter(contracts))
    return dict(
        operation="all_gather",
        world_size=world,
        dtype=dtype,
        cache_policy=cache,
        min_bytes=min(row["max_bytes"] for row in buckets),
        buckets=buckets,
    )
