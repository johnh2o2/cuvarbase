#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_algorithms.py.

Generates:
1. Per-algorithm bar charts (GPU vs CPU baselines)
2. Cost-per-lightcurve comparison across GPU models
3. Markdown report with tables

Usage:
    python scripts/visualize_benchmarks.py benchmark_results.json
    python scripts/visualize_benchmarks.py benchmark_results.json --report results.md
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will only generate text report")


def load_results(filename):
    """Load benchmark results from JSON."""
    with open(filename) as f:
        return json.load(f)


def plot_speedups(data, output_prefix='benchmark'):
    """Bar chart of GPU speedup vs each CPU baseline."""
    if not HAS_MATPLOTLIB:
        return

    results = data['results']
    if not results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    alg_names = []
    speedup_bars = {}  # cpu_name -> list of speedups

    for r in results:
        alg_names.append(r['display_name'])
        for key, val in r.get('speedups', {}).items():
            if key.startswith('gpu_vs_'):
                cpu_name = key[len('gpu_vs_'):]
                if cpu_name not in speedup_bars:
                    speedup_bars[cpu_name] = []
                speedup_bars[cpu_name].append(val)

    if not speedup_bars:
        plt.close()
        return

    x = np.arange(len(alg_names))
    width = 0.8 / max(len(speedup_bars), 1)

    for i, (cpu_name, speedups) in enumerate(speedup_bars.items()):
        # Pad with 0 if some algorithms don't have this baseline
        while len(speedups) < len(alg_names):
            speedups.append(0)
        offset = (i - len(speedup_bars) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f'vs {cpu_name}')
        for bar, val in zip(bars, speedups):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.0f}x', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('GPU Speedup (CPU time / GPU time)')
    ax.set_title('cuvarbase GPU Speedup vs CPU Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(alg_names, rotation=30, ha='right')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outfile = f'{output_prefix}_speedups.png'
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.close()


def plot_time_per_lc(data, output_prefix='benchmark'):
    """Bar chart comparing time per lightcurve across implementations."""
    if not HAS_MATPLOTLIB:
        return

    results = data['results']
    if not results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    alg_names = []
    all_impls = {}  # impl_name -> list of times

    for r in results:
        alg_names.append(r['display_name'])

        # GPU v1
        gpu_entry = r['gpu'].get('cuvarbase_v1', {})
        impl_name = 'cuvarbase GPU'
        if impl_name not in all_impls:
            all_impls[impl_name] = []
        all_impls[impl_name].append(
            gpu_entry.get('time_per_lc', 0))

        # GPU pre-opt
        gpu_old = r['gpu'].get('cuvarbase_preopt', {})
        if 'time_per_lc' in gpu_old:
            impl_name = 'cuvarbase GPU (pre-opt)'
            if impl_name not in all_impls:
                all_impls[impl_name] = [0] * (len(alg_names) - 1)
            all_impls[impl_name].append(gpu_old['time_per_lc'])
        elif 'cuvarbase GPU (pre-opt)' in all_impls:
            all_impls['cuvarbase GPU (pre-opt)'].append(0)

        # CPU baselines
        for cpu_name, cpu_entry in r['cpu'].items():
            impl_name = cpu_entry.get('variant', cpu_name)
            if impl_name not in all_impls:
                all_impls[impl_name] = [0] * (len(alg_names) - 1)
            all_impls[impl_name].append(
                cpu_entry.get('time_per_lc', 0))

    # Pad short lists
    for impl_name in all_impls:
        while len(all_impls[impl_name]) < len(alg_names):
            all_impls[impl_name].append(0)

    x = np.arange(len(alg_names))
    n_impls = len(all_impls)
    width = 0.8 / max(n_impls, 1)

    for i, (impl_name, times) in enumerate(all_impls.items()):
        offset = (i - n_impls / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=impl_name)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Time per lightcurve (seconds)')
    ax.set_title('Time per Lightcurve: GPU vs CPU')
    ax.set_xticks(x)
    ax.set_xticklabels(alg_names, rotation=30, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outfile = f'{output_prefix}_time_per_lc.png'
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.close()


def plot_cost_comparison(data, output_prefix='benchmark'):
    """Bar chart of cost per million lightcurves across GPU models."""
    if not HAS_MATPLOTLIB:
        return

    results = data['results']
    pricing = data.get('runpod_pricing', {})
    if not results or not pricing:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    gpu_models = list(pricing.keys())
    alg_names = [r['display_name'] for r in results]

    x = np.arange(len(gpu_models))
    n_algs = len(results)
    width = 0.8 / max(n_algs, 1)

    for i, r in enumerate(results):
        gpu_entry = r['gpu'].get('cuvarbase_v1', {})
        if 'time_per_lc' not in gpu_entry:
            continue

        costs = []
        for gpu_name in gpu_models:
            price_hr = pricing[gpu_name]['price_hr']
            cost_per_lc = gpu_entry['time_per_lc'] * price_hr / 3600.0
            costs.append(cost_per_lc * 1e6)

        offset = (i - n_algs / 2 + 0.5) * width
        ax.bar(x + offset, costs, width, label=r['display_name'])

    ax.set_xlabel('GPU Model')
    ax.set_ylabel('Cost per million lightcurves ($)')
    ax.set_title('Cost per Million Lightcurves on RunPod (on-demand)')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_models, rotation=30, ha='right')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outfile = f'{output_prefix}_cost.png'
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.close()


def generate_markdown_report(data, output_file='benchmark_report.md'):
    """Generate markdown report from benchmark results."""
    results = data['results']
    system = data.get('system', {})
    pricing = data.get('runpod_pricing', {})

    with open(output_file, 'w') as f:
        f.write("# cuvarbase Benchmark Results\n\n")

        # System info
        f.write("## System\n\n")
        if system:
            f.write(f"- **GPU**: {system.get('gpu_name', 'N/A')}\n")
            f.write(f"- **VRAM**: "
                    f"{system.get('gpu_total_memory_mb', 'N/A')} MB\n")
            f.write(f"- **Platform**: {system.get('platform', 'N/A')}\n")
            f.write(f"- **Python**: {system.get('python_version', 'N/A')}\n")
            f.write(f"- **Timestamp**: {system.get('timestamp', 'N/A')}\n")
        f.write("\n")

        # Parameters
        if results:
            r0 = results[0]
            f.write("## Parameters\n\n")
            f.write(f"- **Observations per lightcurve**: {r0['ndata']}\n")
            f.write(f"- **Batch size**: {r0['nbatch']} lightcurves\n")
            f.write(f"- **Frequency grid**: {r0['nfreq']} points\n")
            f.write(f"- **Baseline**: {r0['baseline']:.0f} days\n\n")

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Algorithm | GPU (s/lc) | Best CPU (s/lc) | "
                "Speedup | $/lc |\n")
        f.write("|-----------|-----------|----------------|"
                "---------|------|\n")

        for r in results:
            alg = r['display_name']

            gpu_entry = r['gpu'].get('cuvarbase_v1', {})
            gpu_str = (f"{gpu_entry['time_per_lc']:.6f}"
                       if 'time_per_lc' in gpu_entry else "N/A")

            cpu_times = {name: e['time_per_lc']
                         for name, e in r['cpu'].items()
                         if 'time_per_lc' in e}
            if cpu_times:
                best_name = min(cpu_times, key=cpu_times.get)
                cpu_str = f"{cpu_times[best_name]:.6f} ({best_name})"
            else:
                cpu_str = "N/A"
                best_name = None

            speedups = r.get('speedups', {})
            if best_name and f'gpu_vs_{best_name}' in speedups:
                sp = speedups[f'gpu_vs_{best_name}']
                sp_str = f"**{sp:.0f}x**"
            else:
                sp_str = "N/A"

            cost = r['cost'].get('cuvarbase_v1', {})
            cost_str = (f"${cost['cost_per_lc']:.8f}"
                        if 'cost_per_lc' in cost else "N/A")

            f.write(f"| {alg} | {gpu_str} | {cpu_str} | "
                    f"{sp_str} | {cost_str} |\n")

        f.write("\n")

        # Per-algorithm details
        f.write("## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r['display_name']}\n\n")
            f.write(f"- Complexity: {r['complexity']}\n")

            for impl, entry in r['gpu'].items():
                if 'time_per_lc' in entry:
                    f.write(f"- GPU ({impl}): "
                            f"{entry['time_per_lc']:.6f} s/lc\n")

            for impl, entry in r['cpu'].items():
                if 'time_per_lc' in entry:
                    f.write(f"- CPU ({entry.get('variant', impl)}): "
                            f"{entry['time_per_lc']:.6f} s/lc\n")

            for key, val in r.get('speedups', {}).items():
                f.write(f"- Speedup ({key}): {val:.1f}x\n")

            f.write("\n")

        # Cost table
        if pricing and any('cuvarbase_v1' in r['cost'] for r in results):
            f.write("## Cost per Million Lightcurves (RunPod on-demand)\n\n")
            header = "| GPU Model | $/hr |"
            sep = "|-----------|------|"
            for r in results:
                header += f" {r['display_name'][:20]} |"
                sep += "------|"
            f.write(header + "\n")
            f.write(sep + "\n")

            for gpu_name, gpu_info in pricing.items():
                row = f"| {gpu_name} | ${gpu_info['price_hr']:.2f} |"
                for r in results:
                    gpu_entry = r['gpu'].get('cuvarbase_v1', {})
                    if 'time_per_lc' in gpu_entry:
                        cost = (gpu_entry['time_per_lc'] *
                                gpu_info['price_hr'] / 3600.0 * 1e6)
                        row += f" ${cost:.2f} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")

            f.write("\n")

    print(f"Generated report: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize benchmark results')
    parser.add_argument('input', type=str,
                        help='Input JSON file from benchmark_algorithms.py')
    parser.add_argument('--output-prefix', type=str, default='benchmark',
                        help='Output file prefix for plots')
    parser.add_argument('--report', type=str, default='benchmark_report.md',
                        help='Output markdown report file')

    args = parser.parse_args()

    data = load_results(args.input)
    n_results = len(data.get('results', []))
    print(f"Loaded {n_results} algorithm benchmark results")

    # Generate plots
    plot_speedups(data, args.output_prefix)
    plot_time_per_lc(data, args.output_prefix)
    plot_cost_comparison(data, args.output_prefix)

    # Generate report
    generate_markdown_report(data, args.report)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
