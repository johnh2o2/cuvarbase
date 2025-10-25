#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_algorithms.py

Creates plots and tables showing:
1. CPU vs GPU performance scaling
2. Speedup as function of problem size
3. Strong/weak scaling analysis
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will only generate text report")


def load_results(filename: str):
    """Load benchmark results from JSON."""
    with open(filename) as f:
        return json.load(f)


def plot_scaling(results, output_prefix='benchmark'):
    """Create scaling plots."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    # Group by algorithm
    by_algorithm = {}
    for r in results:
        alg = r['algorithm']
        if alg not in by_algorithm:
            by_algorithm[alg] = []
        by_algorithm[alg].append(r)

    for alg, data in by_algorithm.items():
        # Sort by ndata, nbatch
        data = sorted(data, key=lambda x: (x['ndata'], x['nbatch']))

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{alg} Performance Scaling', fontsize=16)

        # 1. CPU time vs problem size
        ax = axes[0, 0]
        plot_time_scaling(ax, data, 'cpu_time', 'CPU Time vs Problem Size')

        # 2. GPU time vs problem size
        ax = axes[0, 1]
        plot_time_scaling(ax, data, 'gpu_time', 'GPU Time vs Problem Size')

        # 3. Speedup vs ndata
        ax = axes[1, 0]
        plot_speedup_vs_ndata(ax, data)

        # 4. Speedup vs nbatch
        ax = axes[1, 1]
        plot_speedup_vs_nbatch(ax, data)

        plt.tight_layout()
        output_file = f'{output_prefix}_{alg}_scaling.png'
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot: {output_file}")
        plt.close()


def plot_time_scaling(ax, data, time_field, title):
    """Plot runtime vs problem size."""
    # Group by nbatch
    by_nbatch = {}
    for r in data:
        nb = r['nbatch']
        if nb not in by_nbatch:
            by_nbatch[nb] = {'ndata': [], 'time': [], 'extrapolated': []}

        by_nbatch[nb]['ndata'].append(r['ndata'])
        if r[time_field] is not None:
            by_nbatch[nb]['time'].append(r[time_field])
            by_nbatch[nb]['extrapolated'].append(r.get(f'{time_field.split("_")[0]}_extrapolated', False))
        else:
            by_nbatch[nb]['time'].append(np.nan)
            by_nbatch[nb]['extrapolated'].append(False)

    for nb in sorted(by_nbatch.keys()):
        d = by_nbatch[nb]
        ndata = np.array(d['ndata'])
        times = np.array(d['time'])
        extrap = np.array(d['extrapolated'])

        # Plot measured points
        measured = ~extrap & ~np.isnan(times)
        if measured.any():
            ax.plot(ndata[measured], times[measured], 'o-', label=f'nbatch={nb} (measured)',
                   markersize=8)

        # Plot extrapolated points
        if extrap.any():
            ax.plot(ndata[extrap], times[extrap], 's--', label=f'nbatch={nb} (extrap)',
                   markersize=6, alpha=0.6)

    ax.set_xlabel('Number of observations (ndata)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_speedup_vs_ndata(ax, data):
    """Plot speedup vs ndata for different nbatch values."""
    by_nbatch = {}
    for r in data:
        if r['speedup'] is None:
            continue
        nb = r['nbatch']
        if nb not in by_nbatch:
            by_nbatch[nb] = {'ndata': [], 'speedup': []}
        by_nbatch[nb]['ndata'].append(r['ndata'])
        by_nbatch[nb]['speedup'].append(r['speedup'])

    for nb in sorted(by_nbatch.keys()):
        d = by_nbatch[nb]
        ax.plot(d['ndata'], d['speedup'], 'o-', label=f'nbatch={nb}', markersize=8)

    ax.set_xlabel('Number of observations (ndata)')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.set_title('Speedup vs Problem Size')
    ax.set_xscale('log')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='No speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_speedup_vs_nbatch(ax, data):
    """Plot speedup vs nbatch for different ndata values."""
    by_ndata = {}
    for r in data:
        if r['speedup'] is None:
            continue
        nd = r['ndata']
        if nd not in by_ndata:
            by_ndata[nd] = {'nbatch': [], 'speedup': []}
        by_ndata[nd]['nbatch'].append(r['nbatch'])
        by_ndata[nd]['speedup'].append(r['speedup'])

    for nd in sorted(by_ndata.keys()):
        d = by_ndata[nd]
        ax.plot(d['nbatch'], d['speedup'], 'o-', label=f'ndata={nd}', markersize=8)

    ax.set_xlabel('Batch size (nbatch)')
    ax.set_ylabel('Speedup (CPU/GPU)')
    ax.set_title('Speedup vs Batch Size')
    ax.set_xscale('log')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='No speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)


def generate_markdown_report(results, output_file='benchmark_report.md'):
    """Generate markdown report."""
    with open(output_file, 'w') as f:
        f.write("# cuvarbase Algorithm Benchmarks\n\n")

        # Group by algorithm
        by_algorithm = {}
        for r in results:
            alg = r['algorithm']
            if alg not in by_algorithm:
                by_algorithm[alg] = []
            by_algorithm[alg].append(r)

        for alg, data in by_algorithm.items():
            f.write(f"## {alg}\n\n")

            # Create table
            f.write("| ndata | nbatch | CPU Time (s) | GPU Time (s) | Speedup |\n")
            f.write("|-------|--------|--------------|--------------|----------|\n")

            for r in sorted(data, key=lambda x: (x['ndata'], x['nbatch'])):
                ndata = r['ndata']
                nbatch = r['nbatch']

                cpu_str = f"{r['cpu_time']:.2f}" if r['cpu_time'] else "N/A"
                if r.get('cpu_extrapolated', False):
                    cpu_str += "*"

                gpu_str = f"{r['gpu_time']:.2f}" if r['gpu_time'] else "N/A"
                if r.get('gpu_extrapolated', False):
                    gpu_str += "*"

                speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"

                f.write(f"| {ndata} | {nbatch} | {cpu_str} | {gpu_str} | {speedup_str} |\n")

            f.write("\n*\\* = extrapolated value*\n\n")

            # Analysis
            f.write("### Key Findings\n\n")

            # Find maximum speedup
            speedups = [r['speedup'] for r in data if r['speedup'] is not None]
            if speedups:
                max_speedup = max(speedups)
                max_result = [r for r in data if r['speedup'] == max_speedup][0]
                f.write(f"- **Maximum speedup**: {max_speedup:.1f}x at ndata={max_result['ndata']}, nbatch={max_result['nbatch']}\n")

            # Scaling behavior
            f.write(f"- Algorithm complexity: O(N^{ALGORITHM_COMPLEXITY.get(alg, {}).get('ndata', '?')} × Nfreq)\n")

            f.write("\n")

    print(f"Generated report: {output_file}")


# Algorithm complexity reference
ALGORITHM_COMPLEXITY = {
    'sparse_bls': {'ndata': 2, 'nfreq': 1},
    'bls_gpu_fast': {'ndata': 2, 'nfreq': 1},
    'lombscargle': {'ndata': 1, 'nfreq': 1},
}


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('input', type=str, help='Input JSON file from benchmark_algorithms.py')
    parser.add_argument('--output-prefix', type=str, default='benchmark',
                       help='Output file prefix for plots')
    parser.add_argument('--report', type=str, default='benchmark_report.md',
                       help='Output markdown report file')

    args = parser.parse_args()

    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} benchmark results")

    # Generate plots
    plot_scaling(results, args.output_prefix)

    # Generate report
    generate_markdown_report(results, args.report)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
