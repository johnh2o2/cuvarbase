#!/usr/bin/env python3
"""Render the public timing figure from verified benchmark analysis records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

PROFILES = ['tess_200s', 'tess_gap', 'ztf']
TITLES = {
    'tess_200s': 'TESS: one dense sector',
    'tess_gap': 'TESS: two separated sectors',
    'ztf': 'ZTF: sparse g/r',
}
SUBTITLES = {
    'tess_200s': '200 s cadence · up to 9,736 samples · 26 days',
    'tess_gap': '30 / 10 min cadence · up to 4,295 samples · 735 days',
    'ztf': 'Up to 1,317 samples · 2,744 days',
}
COLORS = {
    'bls_v1': '#008566', 'tls_v1': '#008566', 'bls_pypi': '#2466aa',
    'bls_cpu': '#b55b12', 'bls_gpu': '#8957a5', 'gtls': '#8957a5',
}


def time_label(seconds):
    if seconds < .1:
        return f'{float(f"{seconds * 1000:.2g}"):g} ms'
    return f'{float(f"{seconds:.2g}"):g} s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to the benchmark result directory.')
    args = parser.parse_args()
    recovery = json.loads((args.root / 'recovery_analysis.json').read_text())
    timing = json.loads((args.root / 'timing_analysis.json').read_text())
    for record in (recovery, timing):
        assert record['verification']['complete']
        assert record['verification']['arrays_verified']
    methods = {(r['profile'], r['method']): r for r in recovery['methods']}
    times = {(r['profile'], r['method'], r['mode']): r for r in timing['timings']}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.edgecolor': '#c7cfd5',
        'xtick.color': '#526270', 'ytick.color': '#263c4c',
    })
    fig = plt.figure(figsize=(15, 10.5), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.155, right=.925, top=.815, bottom=.13,
                           hspace=.88, wspace=.79)
    fig.text(.035, .96, 'Faster transit searches across TESS and ZTF cadences',
             fontsize=23, weight='bold', color='#172a3a')
    fig.text(.035, .925, 'Search time per lightcurve · lower is faster',
             fontsize=15, color='#526270')
    fig.legend(handles=[
        Line2D([], [], marker='o', color='#334a5e', markerfacecolor='white',
               linestyle='none', markersize=8, label='One lightcurve'),
        Line2D([], [], marker='o', color='#334a5e', linestyle='none',
               markersize=8, label='Batch of 16: time per lightcurve'),
    ], loc='upper left', bbox_to_anchor=(.028, .904), ncol=2,
        frameon=False, fontsize=12)

    for row, profile in enumerate(PROFILES):
        for col, (family, entries, v1) in enumerate([
            ('BLS', ['bls_v1', 'bls_pypi', 'bls_cpu', 'bls_gpu'], 'bls_v1'),
            ('TLS', ['tls_v1', 'gtls'], 'tls_v1'),
        ]):
            ax = fig.add_subplot(grid[row, col])
            values_on_axis, labels = [], []
            baseline = times[profile, v1, 'batch16']['seconds_per_source']
            for index, method in enumerate(entries):
                values = [times[profile, method, mode]['seconds_per_source']
                          for mode in ('single', 'batch16')]
                values_on_axis.extend(values)
                color = COLORS[method]
                ax.plot(values, [index, index], color=color, lw=2, alpha=.6)
                upper = []
                for mode, value in zip(('single', 'batch16'), values):
                    record = times[profile, method, mode]
                    low = record['min_total_s'] / record['n']
                    high = record['max_total_s'] / record['n']
                    upper.append(high)
                    values_on_axis.extend((low, high))
                    ax.errorbar(value, index,
                                xerr=[[max(0, value-low)], [max(0, high-value)]],
                                fmt='none', ecolor=color, capsize=2, alpha=.6)
                ax.scatter(values[0], index, s=65, edgecolors=color,
                           facecolors='white', linewidths=1.8, zorder=4)
                ax.scatter(values[1], index, s=52, color=color, zorder=5)
                if method == v1:
                    label = 'cuvarbase v1'
                elif method == 'bls_pypi':
                    label = 'cuvarbase 0.2.5'
                elif method == 'bls_cpu':
                    backend = methods[profile, method]['config']['backend']
                    label = 'CPU: ' + ('Astropy' if backend == 'astropy' else 'periodfind')
                elif method == 'bls_gpu':
                    label = 'GPU: periodfind'
                else:
                    label = 'GPU: GTLS'
                labels.append(label)
                annotation = time_label(values[1])
                if method != v1:
                    annotation += f'  ·  {values[1]/baseline:.1f}×'
                ax.annotate(annotation, (max(upper), index), xytext=(8, 0),
                            textcoords='offset points', va='center', fontsize=12,
                            color=color, weight='bold' if method == v1 else 'normal',
                            annotation_clip=False)
            ax.set_xscale('log')
            ax.set_xlim(min(values_on_axis)/1.7, max(values_on_axis)*25)
            ax.set_ylim(len(entries)-.5, -.6)
            ax.set_yticks(range(len(entries)), labels)
            ax.tick_params(axis='y', length=0, pad=10)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: time_label(x)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis='x', alpha=.18)
            ax.set_axisbelow(True)
            ax.set_title(f'{family}  /  {TITLES[profile]}', loc='left',
                         fontsize=14, weight='bold', color='#172a3a', pad=33)
            ax.text(0, 1.10, SUBTITLES[profile], transform=ax.transAxes,
                    fontsize=10.5, color='#526270')
    fig.text(.035, .069,
             'Labels give batch time and the time ratio to v1. Medians of 5 single / 3 batch calls; whiskers span repetitions. Logarithmic axes.',
             fontsize=11, color='#394d5d')
    fig.text(.035, .047,
             'A40 + 7.65 CPU-equivalent allocation. Warm searches from prepared arrays; grid construction and preprocessing excluded.',
             fontsize=11, color='#526270')
    fig.text(.035, .025,
             'Recovery qualifications are in the benchmark report. Equivalent TLS detection sensitivity is not established.',
             fontsize=11, color='#394d5d')
    output = args.output_dir or args.root
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'benchmark_story.{extension}', dpi=160,
                    facecolor='white', metadata={'Creator': 'cuvarbase benchmark tools'})
    plt.close(fig)
    print('Wrote benchmark_story.png / .pdf / .svg')


if __name__ == '__main__':
    main()
