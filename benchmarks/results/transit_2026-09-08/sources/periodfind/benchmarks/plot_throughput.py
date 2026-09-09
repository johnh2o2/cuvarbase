#!/usr/bin/env python3
"""Generate log-log throughput plot from benchmark results.

Produces:
  - docs/throughput_points.png  (point-count scaling sweep)
"""

import csv
import os
from collections import defaultdict, OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "throughput_results.csv")
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "docs")

# Colors and markers per algorithm
ALGO_STYLE = {
    "CE":  {"color": "#1f77b4", "marker": "o"},
    "AOV": {"color": "#ff7f0e", "marker": "s"},
    "LS":  {"color": "#2ca02c", "marker": "^"},
    "FPW": {"color": "#d62728", "marker": "D"},
    "BLS": {"color": "#9467bd", "marker": "v"},
}

# Linestyle and width per backend.  Backends are matched by name from the CSV.
# "CPU" is always dashed; anything else is a GPU variant.
BACKEND_STYLE = OrderedDict([
    ("CPU",      {"linestyle": "--",  "linewidth": 2}),
    ("GPU",      {"linestyle": "-",   "linewidth": 2}),
    ("1x P100",  {"linestyle": "-",   "linewidth": 2}),
    ("2x P100",  {"linestyle": "-.",  "linewidth": 2.5}),
])


def load_results(csv_path):
    """Load CSV into nested dict keyed by sweep type.

    Returns: dict[sweep][backend][algo] = {x_vals: [...], throughput: [...]}
    where x_vals is n_points for the "points" sweep and n_curves for "curves".
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"x_vals": [], "throughput": []})
        )
    )
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweep = row["sweep"]
            backend = row["backend"]
            algo = row["algorithm"]
            if sweep == "points":
                x = int(row["n_points"])
            else:
                x = int(row["n_curves"])
            data[sweep][backend][algo]["x_vals"].append(x)
            data[sweep][backend][algo]["throughput"].append(
                float(row["throughput_pts_per_sec"])
            )
    return data


def plot_sweep(sweep_data, xlabel, title, output_path):
    """Plot a single sweep (point-scaling or curve-scaling) and save to PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Discover which backends are present, ordered by BACKEND_STYLE then any
    # unknown backends appended at the end.
    present = list(sweep_data.keys())
    ordered_backends = [b for b in BACKEND_STYLE if b in present]
    for b in present:
        if b not in ordered_backends:
            ordered_backends.append(b)

    for backend in ordered_backends:
        bstyle = BACKEND_STYLE.get(backend,
                                   {"linestyle": "-.", "linewidth": 2.5})
        for algo in ["CE", "AOV", "LS", "FPW", "BLS"]:
            if algo not in sweep_data[backend]:
                continue
            d = sweep_data[backend][algo]
            style = ALGO_STYLE[algo]
            ax.plot(
                d["x_vals"], d["throughput"],
                color=style["color"],
                marker=style["marker"],
                linestyle=bstyle["linestyle"],
                linewidth=bstyle["linewidth"],
                markersize=6,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Throughput (points / sec)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", alpha=0.3)

    # Two-part legend: algorithms (by color) + backends (by linestyle)
    algo_handles = [
        Line2D([0], [0], color=s["color"], marker=s["marker"],
               linestyle="-", linewidth=2, markersize=6)
        for algo, s in ALGO_STYLE.items()
    ]
    algo_labels = list(ALGO_STYLE.keys())

    backend_handles = []
    backend_labels = []
    for b in ordered_backends:
        bstyle = BACKEND_STYLE.get(b, {"linestyle": "-.", "linewidth": 2.5})
        backend_handles.append(
            Line2D([0], [0], color="black",
                   linestyle=bstyle["linestyle"],
                   linewidth=bstyle["linewidth"])
        )
        backend_labels.append(b)

    leg1 = ax.legend(algo_handles, algo_labels, fontsize=10,
                     loc="upper left", title="Algorithm", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(backend_handles, backend_labels, fontsize=10,
              loc="lower right", title="Backend", title_fontsize=10)

    # Custom x-tick labels
    xticks = sorted(set(
        x for b in sweep_data.values()
        for a in b.values()
        for x in a["x_vals"]
    ))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    data = load_results(CSV_PATH)
    os.makedirs(DOCS_DIR, exist_ok=True)

    if "points" in data:
        # Determine range from data for title
        all_pts = sorted(set(
            x for b in data["points"].values()
            for a in b.values()
            for x in a["x_vals"]
        ))
        plot_sweep(
            data["points"],
            xlabel="Points per light curve",
            title=(
                f"Periodfind throughput — {all_pts[0]}–{all_pts[-1]} pts/curve, "
                f"100 curves × 1000 periods"
            ),
            output_path=os.path.join(DOCS_DIR, "throughput_points.png"),
        )



if __name__ == "__main__":
    main()
