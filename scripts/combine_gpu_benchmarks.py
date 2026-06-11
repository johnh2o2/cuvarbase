#!/usr/bin/env python3
"""
Combine benchmark results from multiple GPU runs into a unified comparison.

Usage:
    python scripts/combine_gpu_benchmarks.py benchmarks/results/by_gpu/
    python scripts/combine_gpu_benchmarks.py benchmarks/results/by_gpu/ --report results.md
"""

import json
import sys
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


RUNPOD_PRICING = OrderedDict([
    ('RTX_4000_Ada',  0.20),
    ('RTX_4090',      0.34),
    ('V100',          0.19),
    ('L40',           0.69),
    ('A100_SXM',      1.19),
    ('H100_SXM',      2.69),
    ('H200_SXM',      3.59),
])


def load_all_results(results_dir):
    """Load all benchmark JSON files from a directory."""
    results_dir = Path(results_dir)
    all_results = {}

    for f in sorted(results_dir.glob('benchmark_*.json')):
        data = json.loads(f.read_text())
        gpu_name = data['system'].get('gpu_name', f.stem.replace('benchmark_', ''))
        # Extract short name from filename
        short_name = f.stem.replace('benchmark_', '')
        all_results[short_name] = data

    return all_results


def print_comparison(all_results):
    """Print cross-GPU comparison tables."""
    if not all_results:
        print("No results found!")
        return

    gpu_names = list(all_results.keys())

    # Get algorithm list from first result
    first_data = next(iter(all_results.values()))
    algorithms = [r['algorithm'] for r in first_data['results']]

    # --- Table 1: GPU time per lightcurve ---
    print("\n" + "=" * 80)
    print("  GPU TIME PER LIGHTCURVE (seconds)")
    print("=" * 80)

    header = f"{'GPU':<18} "
    for alg in algorithms:
        header += f"{alg:<16} "
    print(header)
    print("-" * len(header))

    for gpu_short, data in all_results.items():
        actual_gpu = data['system'].get('gpu_name', gpu_short)
        row = f"{gpu_short:<18} "
        for alg in algorithms:
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg), None)
            if alg_result:
                gpu_entry = alg_result['gpu'].get('cuvarbase_v1', {})
                if 'time_per_lc' in gpu_entry:
                    row += f"{gpu_entry['time_per_lc']:<16.6f} "
                else:
                    row += f"{'N/A':<16} "
            else:
                row += f"{'N/A':<16} "
        print(row)

    # --- Table 2: Speedup vs fastest CPU baseline ---
    print("\n" + "=" * 80)
    print("  GPU SPEEDUP VS BEST CPU BASELINE")
    print("=" * 80)

    header = f"{'GPU':<18} "
    for alg in algorithms:
        header += f"{alg:<16} "
    print(header)
    print("-" * len(header))

    for gpu_short, data in all_results.items():
        row = f"{gpu_short:<18} "
        for alg in algorithms:
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg), None)
            if alg_result:
                speedups = alg_result.get('speedups', {})
                best_speedup = max(
                    (v for k, v in speedups.items() if k.startswith('gpu_vs_')),
                    default=None)
                if best_speedup is not None:
                    row += f"{best_speedup:<16.1f}x"
                else:
                    row += f"{'N/A':<16} "
            else:
                row += f"{'N/A':<16} "
        print(row)

    # --- Table 3: Cost per million lightcurves ---
    print("\n" + "=" * 80)
    print("  COST PER MILLION LIGHTCURVES ($, RunPod on-demand)")
    print("=" * 80)

    header = f"{'GPU':<18} {'$/hr':<8} "
    for alg in algorithms:
        header += f"{alg:<16} "
    print(header)
    print("-" * len(header))

    for gpu_short, data in all_results.items():
        price_hr = RUNPOD_PRICING.get(gpu_short, 0)
        row = f"{gpu_short:<18} ${price_hr:<7.2f} "
        for alg in algorithms:
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg), None)
            if alg_result:
                gpu_entry = alg_result['gpu'].get('cuvarbase_v1', {})
                if 'time_per_lc' in gpu_entry and price_hr > 0:
                    cost_per_M = gpu_entry['time_per_lc'] * price_hr / 3600 * 1e6
                    row += f"${cost_per_M:<15.2f} "
                else:
                    row += f"{'N/A':<16} "
            else:
                row += f"{'N/A':<16} "
        print(row)

    # --- Find optimal GPU per algorithm ---
    print("\n" + "=" * 80)
    print("  OPTIMAL GPU PER ALGORITHM (lowest $/lc)")
    print("=" * 80)

    for alg in algorithms:
        best_gpu = None
        best_cost = float('inf')
        for gpu_short, data in all_results.items():
            price_hr = RUNPOD_PRICING.get(gpu_short, 0)
            if price_hr == 0:
                continue
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg), None)
            if alg_result:
                gpu_entry = alg_result['gpu'].get('cuvarbase_v1', {})
                if 'time_per_lc' in gpu_entry:
                    cost = gpu_entry['time_per_lc'] * price_hr / 3600
                    if cost < best_cost:
                        best_cost = cost
                        best_gpu = gpu_short
        if best_gpu:
            print(f"  {alg:<20} -> {best_gpu:<18} "
                  f"(${best_cost:.8f}/lc, "
                  f"${best_cost*1e6:.2f}/Mlc)")


def generate_plots(all_results, output_prefix='multi_gpu'):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB or not all_results:
        return

    gpu_names = list(all_results.keys())
    first_data = next(iter(all_results.values()))
    algorithms = [r['display_name'] for r in first_data['results']]
    alg_keys = [r['algorithm'] for r in first_data['results']]

    # --- Plot: Time per LC across GPUs ---
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(gpu_names))
    n_algs = len(algorithms)
    width = 0.8 / max(n_algs, 1)

    for i, (alg_name, alg_key) in enumerate(zip(algorithms, alg_keys)):
        times = []
        for gpu_short in gpu_names:
            data = all_results[gpu_short]
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg_key), None)
            if alg_result:
                gpu_entry = alg_result['gpu'].get('cuvarbase_v1', {})
                times.append(gpu_entry.get('time_per_lc', 0))
            else:
                times.append(0)

        offset = (i - n_algs / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=alg_name)

    ax.set_xlabel('GPU Model')
    ax.set_ylabel('Time per lightcurve (seconds)')
    ax.set_title('cuvarbase Performance Across GPU Models')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_names, rotation=30, ha='right')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_time_comparison.png', dpi=150)
    print(f"Saved: {output_prefix}_time_comparison.png")
    plt.close()

    # --- Plot: Cost per million LCs ---
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (alg_name, alg_key) in enumerate(zip(algorithms, alg_keys)):
        costs = []
        for gpu_short in gpu_names:
            price_hr = RUNPOD_PRICING.get(gpu_short, 0)
            data = all_results[gpu_short]
            alg_result = next((r for r in data['results']
                               if r['algorithm'] == alg_key), None)
            if alg_result and price_hr > 0:
                gpu_entry = alg_result['gpu'].get('cuvarbase_v1', {})
                t = gpu_entry.get('time_per_lc', 0)
                costs.append(t * price_hr / 3600 * 1e6)
            else:
                costs.append(0)

        offset = (i - n_algs / 2 + 0.5) * width
        ax.bar(x + offset, costs, width, label=alg_name)

    ax.set_xlabel('GPU Model')
    ax.set_ylabel('Cost per million lightcurves ($)')
    ax.set_title('cuvarbase Cost Efficiency Across GPU Models (RunPod on-demand)')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_names, rotation=30, ha='right')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cost_comparison.png', dpi=150)
    print(f"Saved: {output_prefix}_cost_comparison.png")
    plt.close()


def generate_markdown(all_results, output_file='multi_gpu_report.md'):
    """Generate markdown comparison report."""
    if not all_results:
        return

    gpu_names = list(all_results.keys())
    first_data = next(iter(all_results.values()))
    algorithms = [(r['algorithm'], r['display_name']) for r in first_data['results']]

    with open(output_file, 'w') as f:
        f.write("# cuvarbase Multi-GPU Benchmark Results\n\n")

        # System info per GPU
        f.write("## Hardware\n\n")
        f.write("| GPU | Full Name | VRAM | Compute |\n")
        f.write("|-----|-----------|------|---------|\n")
        for gpu_short, data in all_results.items():
            sys_info = data.get('system', {})
            f.write(f"| {gpu_short} | {sys_info.get('gpu_name', 'N/A')} | "
                    f"{sys_info.get('gpu_total_memory_mb', 'N/A')} MB | "
                    f"{sys_info.get('gpu_compute_capability', 'N/A')} |\n")
        f.write("\n")

        # Parameters
        r0 = first_data['results'][0]
        f.write("## Parameters\n\n")
        f.write(f"- **Observations**: {r0['ndata']}\n")
        f.write(f"- **Batch**: {r0['nbatch']} lightcurves\n")
        f.write(f"- **Frequencies**: {r0['nfreq']}\n")
        f.write(f"- **Baseline**: {r0['baseline']:.0f} days\n\n")

        # Time per LC table
        f.write("## GPU Time per Lightcurve (seconds)\n\n")
        header = "| GPU |"
        sep = "|-----|"
        for _, disp in algorithms:
            header += f" {disp} |"
            sep += "------|"
        f.write(header + "\n" + sep + "\n")

        for gpu_short, data in all_results.items():
            row = f"| {gpu_short} |"
            for alg_key, _ in algorithms:
                alg_r = next((r for r in data['results']
                              if r['algorithm'] == alg_key), None)
                if alg_r:
                    gpu_e = alg_r['gpu'].get('cuvarbase_v1', {})
                    if 'time_per_lc' in gpu_e:
                        row += f" {gpu_e['time_per_lc']:.6f} |"
                    else:
                        row += " N/A |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        f.write("\n")

        # Cost table
        f.write("## Cost per Million Lightcurves ($ RunPod on-demand)\n\n")
        header = "| GPU | $/hr |"
        sep = "|-----|------|"
        for _, disp in algorithms:
            header += f" {disp} |"
            sep += "------|"
        f.write(header + "\n" + sep + "\n")

        for gpu_short, data in all_results.items():
            price = RUNPOD_PRICING.get(gpu_short, 0)
            row = f"| {gpu_short} | ${price:.2f} |"
            for alg_key, _ in algorithms:
                alg_r = next((r for r in data['results']
                              if r['algorithm'] == alg_key), None)
                if alg_r and price > 0:
                    gpu_e = alg_r['gpu'].get('cuvarbase_v1', {})
                    if 'time_per_lc' in gpu_e:
                        cost = gpu_e['time_per_lc'] * price / 3600 * 1e6
                        row += f" ${cost:.2f} |"
                    else:
                        row += " N/A |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        f.write("\n")

        # Optimal GPU
        f.write("## Optimal GPU per Algorithm (lowest $/lc)\n\n")
        f.write("| Algorithm | Best GPU | $/lc | $/million LC |\n")
        f.write("|-----------|----------|------|-------------|\n")
        for alg_key, disp in algorithms:
            best_gpu = None
            best_cost = float('inf')
            for gpu_short, data in all_results.items():
                price = RUNPOD_PRICING.get(gpu_short, 0)
                if price == 0:
                    continue
                alg_r = next((r for r in data['results']
                              if r['algorithm'] == alg_key), None)
                if alg_r:
                    gpu_e = alg_r['gpu'].get('cuvarbase_v1', {})
                    if 'time_per_lc' in gpu_e:
                        cost = gpu_e['time_per_lc'] * price / 3600
                        if cost < best_cost:
                            best_cost = cost
                            best_gpu = gpu_short
            if best_gpu:
                f.write(f"| {disp} | {best_gpu} | "
                        f"${best_cost:.8f} | ${best_cost*1e6:.2f} |\n")
        f.write("\n")

    print(f"Generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine multi-GPU benchmark results')
    parser.add_argument('results_dir', type=str,
                        help='Directory with benchmark_*.json files')
    parser.add_argument('--output-prefix', type=str,
                        default='multi_gpu',
                        help='Output prefix for plots')
    parser.add_argument('--report', type=str,
                        default='multi_gpu_report.md',
                        help='Output markdown report')

    args = parser.parse_args()

    all_results = load_all_results(args.results_dir)
    print(f"Loaded results from {len(all_results)} GPUs: "
          f"{', '.join(all_results.keys())}")

    print_comparison(all_results)
    generate_plots(all_results, args.output_prefix)
    generate_markdown(all_results, args.report)


if __name__ == '__main__':
    main()
