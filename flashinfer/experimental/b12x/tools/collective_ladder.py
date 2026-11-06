"""Measure collective profiles across transfer sizes under torchrun."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time


def byte_size(value):
    text = value.strip().lower()
    for suffix, multiplier in (("mib", 1 << 20), ("kib", 1 << 10)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    else:
        multiplier = 1
    try:
        size = float(text) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be bytes, KiB, or MiB") from error
    if not math.isfinite(size) or size != int(size) or size < 16:
        raise argparse.ArgumentTypeError(
            "sizes must be integer byte counts of at least 16"
        )
    if int(size) % 16:
        raise argparse.ArgumentTypeError("sizes must be 16-byte aligned")
    return int(size)


def parser():
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog="Render an existing report without GPUs: "
        "python -m b12x.tools.collective_ladder plot REPORT.jsonl",
    )
    result.add_argument(
        "--operation", choices=("all_reduce", "all_gather"), default="all_reduce"
    )
    result.add_argument(
        "--row-bytes", type=byte_size, help="local column-gather row width in bytes"
    )
    result.add_argument(
        "--gather-kernel",
        choices=("static", "runtime", "compare"),
        default="runtime",
        help="all-gather kernel candidates (default: runtime)",
    )
    result.add_argument(
        "--gather-proposal",
        type=Path,
        help="measure supplied gather geometries, profiles, and optional reference configurations",
    )
    result.add_argument(
        "--gather-capacity-bytes",
        type=byte_size,
        help="preserve channel capacity when measuring a subset of gather sizes",
    )
    result.add_argument(
        "--output", type=Path, help="incremental JSONL measurement report"
    )
    history = result.add_mutually_exclusive_group()
    history.add_argument(
        "--resume",
        type=Path,
        help="append to this compatible report and reuse measurements",
    )
    history.add_argument(
        "--fresh",
        action="store_true",
        help="start without previous measurements; output must not exist",
    )
    result.add_argument(
        "--sizes",
        nargs="+",
        type=byte_size,
        help="explicit sizes; default: 209 points from 1KiB to 8MiB",
    )
    result.add_argument(
        "--min-bytes", dest="minimum_bytes", type=byte_size, default=1024
    )
    result.add_argument(
        "--max-bytes", dest="maximum_bytes", type=byte_size, default=8 << 20
    )
    result.add_argument("--points-per-doubling", type=int, default=16)
    result.add_argument("--cache-policy", choices=("cold", "warm"), default="cold")
    result.add_argument(
        "--alias",
        action="store_true",
        help="use exact input/output aliases instead of disjoint buffers",
    )
    result.add_argument(
        "--budget-seconds",
        type=float,
        help="charged wall budget, excluding compilation only (default: 600; exhaustive: 3600)",
    )
    result.add_argument(
        "--exhaustive",
        action="store_true",
        help="measure every legal configuration at every requested size",
    )
    result.add_argument(
        "--replay",
        type=Path,
        nargs="+",
        help="independently remeasure the union of reported winners, balanced at each size",
    )
    result.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="minimum rounds in [1,512]; sampling adapts to payload size and measured latency",
    )
    result.add_argument("--compile-workers", type=int, default=None)
    result.add_argument(
        "--plot",
        action="store_true",
        help="write per-report choice and bucket PNGs (requires Matplotlib)",
    )
    return result


def arguments(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.output is None:
        args.output = args.resume
    if args.output is None:
        command.error("--output is required unless --resume names the output")
    if args.resume is not None and args.output.resolve() != args.resume.resolve():
        command.error("--resume and --output must name the same report")
    if args.replay and (args.resume or args.exhaustive):
        command.error("--replay cannot be combined with --resume or --exhaustive")
    if args.replay and args.plot:
        command.error(
            "--plot requires selected recommendations; replay reports have none"
        )
    if not 1 <= args.rounds <= 512:
        command.error("rounds must be in [1,512]")
    if args.compile_workers is not None and args.compile_workers <= 0:
        command.error("compile workers must be positive")
    if args.budget_seconds is None:
        args.budget_seconds = 3600.0 if args.exhaustive else 600.0
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        command.error("budget seconds must be positive and finite")
    if args.minimum_bytes > args.maximum_bytes or args.points_per_doubling <= 0:
        command.error(
            "size bounds must increase and points per doubling must be positive"
        )
    if args.sizes is not None:
        if (args.minimum_bytes, args.maximum_bytes, args.points_per_doubling) != (
            1024,
            8 << 20,
            16,
        ):
            command.error(
                "--sizes cannot be combined with custom range bounds or spacing"
            )
        args.sizes = tuple(sorted(set(args.sizes)))
    if args.operation == "all_gather":
        if args.alias or args.row_bytes is None:
            command.error("all-gather requires --row-bytes and disjoint buffers")
        if args.sizes is not None and any(size % args.row_bytes for size in args.sizes):
            command.error("explicit gather sizes must contain complete rows")
        if args.sizes is None and args.maximum_bytes < args.row_bytes:
            command.error("maximum gather size must contain at least one row")
    elif args.row_bytes is not None:
        command.error("--row-bytes requires --operation all_gather")
    if args.gather_proposal is not None and args.operation != "all_gather":
        command.error("--gather-proposal requires --operation all_gather")
    if args.gather_capacity_bytes is not None and args.operation != "all_gather":
        command.error("--gather-capacity-bytes requires --operation all_gather")
    args.inplace = args.alias
    return args


def range_sizes(first, last, points):
    count = math.ceil(math.log2(last / first) * points)
    values = {first, last}
    for index in range(count):
        size = round(first * 2 ** (index / points) / 16) * 16
        values.add(min(last, max(first, size)))
    return tuple(sorted(values))


def _key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _replay_trials(paths, metadata, read_report, Trial, *, minimum_rounds=1):
    from b12x.tools._collective_ladder_sampling import measurement_rounds

    selections, sources, estimates = {}, {}, {}
    expected = set(metadata["sizes"])
    legal = {
        row["bytes"]: {_key(config) for config in row["configs"]}
        for row in metadata["catalogs"]
    }
    for path in paths:
        previous, records, summaries = read_report(path)
        # Replay imports configurations; every timing uses the active protocol.
        comparable = {k: v for k, v in metadata.items() if k != "timing_protocol"}
        if {k: v for k, v in previous.items() if k != "timing_protocol"} != comparable:
            raise ValueError(
                f"replay report has incompatible hardware or measurement semantics: {path}"
            )
        if not summaries:
            raise ValueError(f"replay report has no recommendation summary: {path}")
        winners = summaries[-1]["winners"]
        measured, failed = set(), set()
        for record in records:
            pair = record["bytes"], _key(record["config"])
            passed = (
                record.get("correctness") in (True, "passed")
                and not record.get("error")
                and record.get("samples_us")
            )
            (measured if passed else failed).add(pair)
        missing = expected - {winner["size_bytes"] for winner in winners}
        if missing:
            raise ValueError(
                f"replay report has no recommendation for sizes {sorted(missing)}: {path}"
            )
        for winner in winners:
            size, config = winner["size_bytes"], winner["config"]
            key = _key(config)
            if size not in legal or key not in legal[size]:
                raise ValueError(
                    f"replay recommendation is outside the legal catalog: {path}, {size}"
                )
            if (size, key) not in measured or (size, key) in failed:
                raise ValueError(
                    f"replay recommendation lacks successful measurements: {path}, {size}"
                )
            estimate = winner["median_us"]
            if (
                type(estimate) not in (int, float)
                or not math.isfinite(estimate)
                or estimate <= 0
            ):
                raise ValueError(
                    f"replay recommendation has an invalid latency estimate: {path}, {size}"
                )
            estimates[size] = min(estimates.get(size, estimate), estimate)
            selections.setdefault(size, {})[key] = config
            sources.setdefault(size, {}).setdefault(key, []).append(str(path))
    trials = [
        Trial(
            size,
            tuple(configs.values()),
            rounds=measurement_rounds(
                size,
                "replay",
                estimate_us=estimates[size],
                minimum_rounds=minimum_rounds,
            ),
            phase="replay",
        )
        for size, configs in sorted(selections.items())
    ]
    return iter(trials), sources, selections


def run(args):
    started = time.monotonic()
    if args.plot:
        from b12x.tools._collective_ladder_plot import require_matplotlib

        require_matplotlib()
    import torch
    import torch.distributed as dist

    from b12x.tools._collective_ladder_gpu import JointCollector
    from b12x.tools._collective_ladder_search import (
        RawReport,
        Search,
        Trial,
        ladder_sizes,
        read_report,
    )

    args.sizes = args.sizes or (
        ladder_sizes()
        if (args.minimum_bytes, args.maximum_bytes, args.points_per_doubling)
        == (1024, 8 << 20, 16)
        else range_sizes(
            args.minimum_bytes, args.maximum_bytes, args.points_per_doubling
        )
    )
    if args.operation == "all_gather":
        args.sizes = tuple(
            sorted(
                {
                    max(args.row_bytes, round(size / args.row_bytes) * args.row_bytes)
                    for size in args.sizes
                }
            )
        )
    args.max_bytes = args.gather_capacity_bytes or max(args.sizes)
    if args.max_bytes < max(args.sizes) or (
        args.operation == "all_gather" and args.max_bytes % args.row_bytes
    ):
        raise ValueError(
            "gather capacity must cover every measured size in complete rows"
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("gloo", timeout=timedelta(hours=12))
    rank = dist.get_rank()
    collector = report = search = replay = None
    failure = None
    final_summary = None
    next_progress = started + 5
    measured_sizes, measured_configs = set(), set()
    excluded = 0.0
    compilation_before = 0.0
    replay_completed = set()
    replay_expected = {}
    replay_sources = {}
    try:
        if args.operation == "all_gather":
            from b12x.tools._collective_ladder_gather import GatherCollector

            collector = GatherCollector(args)
        else:
            collector = JointCollector(args)
        catalogs = {size: collector.catalog(size) for size in args.sizes}
        import flashinfer

        metadata = dict(
            versions=dict(
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                flashinfer=flashinfer.__version__,
                python=sys.version.split()[0],
            ),
            timing_protocol=dict(
                version=3,
                preconditioning_invocations_per_fixture=1,
                samples_per_fixture=2,
                buffer_slots="balanced_per_fixture",
                input_cases=3,
                invocations_per_sample=1,
                samples="gpu_max_over_ranks_us",
                order="rotated_and_reversed",
            ),
            dtype="bfloat16",
            cache_policy=args.cache_policy,
            alias=args.alias,
            sizes=list(args.sizes),
            hardware=collector.metadata,
            catalogs=[
                dict(bytes=size, configs=list(configs))
                for size, configs in catalogs.items()
            ],
        )
        if args.operation == "all_gather":
            metadata.update(operation="all_gather", row_bytes=args.row_bytes)
            if args.gather_capacity_bytes is not None:
                metadata["capacity_bytes"] = args.gather_capacity_bytes
            if args.gather_kernel != "static":
                metadata["gather_kernel"] = args.gather_kernel
        metadata = json.loads(json.dumps(metadata))
        init_error = None
        if rank == 0:
            try:
                report = RawReport(
                    args.output, metadata, resume=args.resume is not None
                )
                if args.replay:
                    replay, replay_sources, replay_expected = _replay_trials(
                        args.replay,
                        metadata,
                        read_report,
                        Trial,
                        minimum_rounds=args.rounds,
                    )
                else:
                    search = Search(
                        catalogs,
                        deadline=started + args.budget_seconds,
                        exhaustive=args.exhaustive,
                        records=report.records,
                        rounds=args.rounds,
                    )
            except Exception as error:
                init_error = f"{type(error).__name__}: {error}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(
            errors, (init_error, float(collector.compilation_seconds))
        )
        excluded = max(item[1] for item in errors)
        compilation_before = float(collector.compilation_seconds)
        if errors[0][0] is not None:
            raise RuntimeError(errors[0][0])
        deadline = started + args.budget_seconds + excluded
        while True:
            command = None
            if rank == 0:
                try:
                    if time.monotonic() >= deadline:
                        command = ("stop", None)
                    else:
                        if search is not None:
                            search.deadline = deadline
                            trial = search.propose()
                        else:
                            trial = next(replay, None)
                        command = (
                            ("stop", None) if trial is None else ("measure", trial)
                        )
                except Exception as error:
                    command = ("error", f"{type(error).__name__}: {error}")
            message = [command]
            dist.broadcast_object_list(message, src=0)
            operation, trial = message[0]
            if operation == "stop":
                break
            if operation == "error":
                raise RuntimeError(trial)
            local_error = None
            records = None
            try:
                records = collector.measure(
                    trial.size_bytes, trial.configs, rounds=trial.rounds
                )
            except Exception as error:
                local_error = f"rank {rank}: {type(error).__name__}: {error}"
            compilation_now = float(collector.compilation_seconds)
            dist.all_gather_object(
                errors, (local_error, max(0.0, compilation_now - compilation_before))
            )
            compilation_before = compilation_now
            excluded += max(item[1] for item in errors)
            deadline = started + args.budget_seconds + excluded
            failures = [item[0] for item in errors if item[0] is not None]
            if failures:
                raise RuntimeError("; ".join(failures))
            write_error = None
            if rank == 0:
                try:
                    for record in records:
                        record = dict(
                            record,
                            phase=trial.phase,
                            batch=trial.batch_id,
                            batch_size=len(trial.configs),
                            rounds=trial.rounds,
                        )
                        if replay is not None:
                            record["replay_sources"] = replay_sources[trial.size_bytes][
                                _key(record["config"])
                            ]
                        report.append(record)
                        if record["correctness"] == "passed":
                            measured_sizes.add(record["bytes"])
                            measured_configs.add(
                                (record["bytes"], _key(record["config"]))
                            )
                        if replay is not None:
                            replay_completed.add(
                                (record["bytes"], _key(record["config"]))
                            )
                    if search is not None:
                        search.deadline = deadline
                        search.observe(trial, records)
                    now = time.monotonic()
                    if now >= next_progress:
                        print(
                            f"{args.operation} ladder: {len(measured_sizes)}/{len(args.sizes)} sizes; "
                            f"{len(measured_configs)} configurations; {trial.phase}; "
                            f"charged {now - started - excluded:.1f}/"
                            f"{args.budget_seconds:.1f} s; compilation {excluded:.1f} s",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_progress = now + 5
                except Exception as error:
                    write_error = f"{type(error).__name__}: {error}"
            message = [write_error]
            dist.broadcast_object_list(message, src=0)
            if message[0] is not None:
                raise RuntimeError(message[0])
        if rank == 0:
            summary = (
                search.summary()
                if search is not None
                else dict(
                    status="complete"
                    if len(replay_completed) == sum(map(len, replay_expected.values()))
                    else "incomplete",
                    winners=[],
                    missing=[
                        dict(
                            size_bytes=size,
                            configs=[
                                config
                                for key, config in configs.items()
                                if (size, key) not in replay_completed
                            ],
                        )
                        for size, configs in replay_expected.items()
                        if any((size, key) not in replay_completed for key in configs)
                    ],
                )
            )
            summary.update(
                budget_seconds=args.budget_seconds,
                compilation_seconds=excluded,
                elapsed_before_summary_seconds=time.monotonic() - started,
                charged_before_summary_seconds=time.monotonic() - started - excluded,
            )
            report.append_summary(summary)
            report.close()
            report = None
            final_summary = summary
    except Exception as error:
        failure = error
        raise
    finally:
        if report is not None:
            report.close()
        if collector is not None:
            collector.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if failure is not None and rank == 0:
            print(f"collective ladder failed: {failure}", file=sys.stderr)

    if final_summary is not None:
        plots = {}
        if args.plot:
            from b12x.tools._collective_ladder_plot import plot_report

            plots = dict(plots=[str(path) for path in plot_report(args.output)])
        final_summary.update(
            elapsed_seconds=time.monotonic() - started,
            charged_seconds=time.monotonic() - started - excluded,
        )
        fields = (
            "status",
            "reason",
            "measured_configurations",
            "measured_sizes",
            "total_sizes",
            "confirmed_sizes",
            "budget_seconds",
            "compilation_seconds",
            "elapsed_seconds",
            "charged_seconds",
        )
        print(
            json.dumps(
                dict(
                    report=str(args.output),
                    **plots,
                    **{
                        key: final_summary[key]
                        for key in fields
                        if key in final_summary
                    },
                ),
                sort_keys=True,
            ),
            flush=True,
        )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "recommend-gather":
        command = argparse.ArgumentParser(
            prog="python -m b12x.tools.collective_ladder recommend-gather"
        )
        command.add_argument("--switched", type=Path, nargs="+", required=True)
        command.add_argument("--host", type=Path, nargs="+", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--switched-margin", type=float, default=0.02)
        args = command.parse_args(argv[1:])
        from b12x.tools._collective_ladder_merge import gather_recommendations

        result = gather_recommendations(
            args.switched, args.host, switched_margin=args.switched_margin
        )
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(args.output)
        return
    if argv and argv[0] == "plot":
        command = argparse.ArgumentParser(
            prog="python -m b12x.tools.collective_ladder plot",
            description="Render one saved collective ladder report without GPUs.",
        )
        command.add_argument("report", type=Path)
        args = command.parse_args(argv[1:])
        from b12x.tools._collective_ladder_plot import plot_report

        paths = plot_report(args.report)
        print(json.dumps(dict(report=str(args.report), plots=list(map(str, paths)))))
        return
    run(arguments(argv))


if __name__ == "__main__":
    main()
