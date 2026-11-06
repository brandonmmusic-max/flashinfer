"""Per-report choice maps and exact-payload tables for collective ladders."""

import math
from pathlib import Path

from b12x.tools._collective_ladder_search import read_report


def require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise RuntimeError(
            "Plotting requires Matplotlib: python -m pip install matplotlib"
        ) from error


def _size_label(size):
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:g} {unit}"
    return f"{size} B"


def _device_label(metadata):
    hardware = metadata.get("hardware", {})
    indices = {
        (topology["host"], gpu["uuid"].removeprefix("GPU-")): gpu["index"]
        for topology in hardware.get("topologies", [])
        for gpu in topology.get("gpu_map", [])
    }
    labels = []
    for device in sorted(hardware.get("devices", []), key=lambda d: d["rank"]):
        host = device.get("host", "")
        index = indices.get((host, device.get("uuid", "").removeprefix("GPU-")))
        location = f"GPU {index}" if index is not None else f"rank {device['rank']}"
        labels.append(f"{host}/{location}" if host else location)
    return ", ".join(labels) or f"{hardware.get('world_size', '?')} ranks"


def _choices(metadata, summary):
    sizes = sorted(metadata["sizes"])
    if not sizes or len(sizes) != len(set(sizes)):
        raise ValueError("plotting requires distinct payload sizes")
    winners = {}
    for winner in summary.get("winners", []):
        size = winner["size_bytes"]
        if size not in sizes or size in winners:
            raise ValueError(
                "recommendations must have distinct declared payload sizes"
            )
        config = winner["config"]
        for field in ("threads", "blocks", "active_blocks"):
            if type(config.get(field)) is not int or config[field] <= 0:
                raise ValueError(f"recommendation requires a positive {field}")
        for field in ("implementation", "submode"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(f"recommendation requires {field}")
        winners[size] = winner
    if not winners:
        raise ValueError("report has no selected recommendations to plot")
    return sizes, [winners.get(size) for size in sizes]


def _heading(path, metadata, summary, rows):
    confirmed = sum(row is not None and row.get("confirmed", False) for row in rows)
    return (
        f"{path.name}\n{_device_label(metadata)}",
        f"{metadata.get('operation', 'all_reduce')} · "
        + (f"row {metadata['row_bytes']} B · " if "row_bytes" in metadata else "")
        + f"{metadata.get('dtype', '?')} · {metadata.get('cache_policy', '?')} cache · "
        f"{'aliased' if metadata.get('alias') else 'disjoint'} buffers · "
        f"{confirmed}/{len(rows)} confirmed · {summary.get('status', '?')}",
    )


def _choice_figure(sizes, rows, heading, details):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    figure = Figure(figsize=(18, 9.5))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        5,
        1,
        height_ratios=[0.55, 0.55, 1, 1, 1],
        left=0.10,
        right=0.985,
        top=0.76,
        bottom=0.13,
        hspace=0.27,
    )
    axes = [figure.add_subplot(grid[index]) for index in range(5)]
    handles = []
    palettes = (
        ("#3366ad", "#e28c27", "#3366ad", "#e28c27", "#49954a", "#4169a1", "#9561b1"),
        ("#4169a1", "#9561b1", "#49954a", "#169a9d", "#cc6a38", "#4169a1", "#9561b1"),
    )
    orders = (
        (
            "joint_latency",
            "joint_bandwidth",
            "gather_barrier",
            "gather_word_exchange",
            "gather_joint",
            "gather_selected_barrier",
            "gather_selected_word_exchange",
        ),
        (
            "full",
            "payload_rsag",
            "bulk_push",
            "bulk_pull",
            "pull",
            "barrier",
            "word_exchange",
        ),
    )
    for axis, field, title, palette, order in zip(
        axes,
        ("implementation", "submode"),
        ("Compiled family", "Protocol / mode"),
        palettes,
        orders,
    ):
        present = {row["config"][field] for row in rows if row is not None}
        names = [name for name in order if name in present] + sorted(
            present - set(order)
        )
        colors = [
            palette[order.index(name)] if name in order else "#777777" for name in names
        ]
        values = [
            [names.index(row["config"][field]) if row else len(names) for row in rows]
        ]
        axis.imshow(
            values,
            cmap=ListedColormap(colors + ["#eeeeee"]),
            vmin=0,
            vmax=len(names),
            aspect="auto",
            extent=(-0.5, len(rows) - 0.5, -0.5, 0.5),
            interpolation="nearest",
        )
        for index, row in enumerate(rows):
            if row is not None and not row.get("confirmed", False):
                axis.add_patch(
                    Rectangle(
                        (index - 0.5, -0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="#555555",
                        linewidth=0,
                    )
                )
        axis.set_yticks([])
        axis.set_ylabel(title, rotation=0, ha="right", va="center")
        handles.extend(
            Patch(color=color, label=name) for name, color in zip(names, colors)
        )
        begin = 0
        while begin < len(rows):
            end = begin + 1
            while end < len(rows) and values[0][end] == values[0][begin]:
                end += 1
            if rows[begin] and end - begin >= max(3, len(rows) // 20):
                axis.text(
                    (begin + end - 1) / 2,
                    0,
                    rows[begin]["config"][field].removeprefix("joint_"),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
            begin = end
    for axis, field, label in zip(
        axes[2:],
        ("threads", "blocks", "active_blocks"),
        ("Threads / CTA", "Physical CTAs", "Active CTAs"),
    ):
        values = [row["config"][field] if row else math.nan for row in rows]
        valid = [value for value in values if math.isfinite(value)]
        axis.step(range(len(rows)), values, where="mid", color="#202b39", linewidth=1)
        axis.scatter(
            range(len(rows)),
            values,
            s=10,
            c=[
                "#202b39" if row and row.get("confirmed", False) else "#c94726"
                for row in rows
            ],
            zorder=3,
        )
        low, high = math.floor(math.log2(min(valid))), math.ceil(math.log2(max(valid)))
        ticks = [2**power for power in range(low, high + 1)]
        axis.set_yscale("log", base=2)
        axis.set_yticks(ticks, [str(tick) for tick in ticks])
        axis.set_ylim(min(valid) / 1.3, max(valid) * 1.3)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    ticks = {0, len(sizes) - 1}
    ticks.update(index for index, size in enumerate(sizes) if size & (size - 1) == 0)
    if len(sizes) <= 14:
        ticks = set(range(len(sizes)))
    ticks = sorted(ticks)
    if len(ticks) > 15:
        ticks = ticks[:: math.ceil(len(ticks) / 14)] + [len(sizes) - 1]
    for axis in axes:
        axis.set_xlim(-0.5, len(rows) - 0.5)
        axis.set_xticks(ticks, [_size_label(sizes[index]) for index in ticks])
        axis.set_xticks(range(len(rows)), minor=True)
        axis.grid(axis="x", which="minor", alpha=0.15, linewidth=0.35)
        axis.grid(axis="x", which="major", alpha=0.3, linewidth=0.5)
        axis.tick_params(axis="x", which="minor", length=0)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False, length=0)
    axes[-1].set_xlabel("Exact measured payload · one column per bucket")
    figure.suptitle(f"PCIe collective choices\n{heading}", fontsize=14, y=0.99)
    figure.text(0.54, 0.87, details, ha="center", fontsize=10)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.85),
        ncols=min(len(handles), 7),
        fontsize=9,
        frameon=False,
    )
    figure.text(
        0.54,
        0.035,
        "Equal-width columns are exact sampled sizes, not dispatch intervals. Gray = missing; hatch / orange dots = unconfirmed.",
        ha="center",
        fontsize=10,
    )
    return figure


def _bucket_figure(sizes, rows, heading, details):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    columns = min(16, len(rows))
    height = math.ceil(len(rows) / columns)
    figure = Figure(figsize=(max(14, columns * 1.4), 2.5 + height * 0.56))
    FigureCanvasAgg(figure)
    axis = figure.add_axes(
        (
            0.025,
            0.6 / figure.get_figheight(),
            0.95,
            (height * 0.56) / figure.get_figheight(),
        )
    )
    axis.set_xlim(0, columns)
    axis.set_ylim(height, 0)
    axis.axis("off")
    for index, (size, row) in enumerate(zip(sizes, rows)):
        y, x = divmod(index, columns)
        face, hatch = "#eeeeee", None
        text = f"#{index:03d}  {size:,} B\nNo recommendation"
        if row:
            config = row["config"]
            family = {
                "joint_latency": "L",
                "joint_bandwidth": "B",
                "gather_joint": "GJ",
                "gather_barrier": "GB",
                "gather_word_exchange": "GW",
                "gather_selected_barrier": "GBD",
                "gather_selected_word_exchange": "GWD",
            }.get(config["implementation"], config["implementation"])
            face = "#e1ebf8" if family == "L" else "#faecd9"
            suffix = "" if row.get("confirmed", False) else " ?"
            hatch = None if not suffix else "///"
            text = f"#{index:03d}  {size:,} B\n{family}/{config['submode']}\nT{config['threads']} G{config['blocks']} A{config['active_blocks']}{suffix}"
        axis.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor=face,
                edgecolor="#aab3be",
                linewidth=0.45,
                hatch=hatch,
            )
        )
        axis.text(
            x + 0.5,
            y + 0.5,
            text,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.3,
        )
    figure.suptitle(
        f"Exact collective buckets\n{heading}",
        fontsize=14,
        y=1 - 0.12 / figure.get_figheight(),
    )
    figure.text(
        0.5, 1 - 1.0 / figure.get_figheight(), details, ha="center", fontsize=10
    )
    figure.text(
        0.5,
        1 - 1.28 / figure.get_figheight(),
        "L = joint_latency · B/ = joint_bandwidth · T = threads/CTA · G = physical CTAs · A = active CTAs · ? = unconfirmed",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.2 / figure.get_figheight(),
        "Zero-based bucket indices; exact byte sizes. Entries do not define dispatch intervals.",
        ha="center",
        fontsize=10,
    )
    return figure


def plot_report(report):
    """Render one report's final summary to adjacent choice and bucket PNGs."""
    require_matplotlib()
    path = Path(report)
    metadata, _, summaries = read_report(path)
    if not summaries:
        raise ValueError("report has no recommendation summary")
    summary = summaries[-1]
    sizes, rows = _choices(metadata, summary)
    heading, details = _heading(path, metadata, summary, rows)
    outputs = []
    for suffix, build in (("choices", _choice_figure), ("buckets", _bucket_figure)):
        output = path.with_suffix(f".{suffix}.png")
        figure = build(sizes, rows, heading, details)
        try:
            figure.savefig(output, dpi=180)
        finally:
            figure.clear()
        outputs.append(output)
    return tuple(outputs)
