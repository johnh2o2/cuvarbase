#!/usr/bin/env python3
"""
Benchmark adaptive BLS with dynamic block sizing.

Compares performance across:
1. Standard BLS (fixed block_size=256)
2. Optimized BLS (fixed block_size=256)
3. Adaptive BLS (dynamic block sizing)
"""

import numpy as np
import time
import json
from datetime import datetime

try:
    from cuvarbase import bls
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU not available: {e}")


def generate_test_data(ndata, with_signal=True, period=5.0, depth=0.01):
    """Generate synthetic lightcurve data."""
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 100, ndata)).astype(np.float32)
    y = np.ones(ndata, dtype=np.float32)

    if with_signal:
        # Add transit signal
        phase = (t % period) / period
        in_transit = (phase > 0.4) & (phase < 0.5)
        y[in_transit] -= depth

    # Add noise
    y += np.random.normal(0, 0.01, ndata).astype(np.float32)
    dy = np.ones(ndata, dtype=np.float32) * 0.01

    return t, y, dy


def benchmark_adaptive(ndata_values, nfreq=1000, n_trials=5):
    """
    Benchmark adaptive BLS across different data sizes.

    Parameters
    ----------
    ndata_values : list
        List of ndata values to test
    nfreq : int
        Number of frequency points
    n_trials : int
        Number of trials to average over

    Returns
    -------
    results : dict
        Benchmark results
    """
    print("=" * 80)
    print("ADAPTIVE BLS BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  nfreq: {nfreq}")
    print(f"  trials per config: {n_trials}")
    print(f"  ndata values: {ndata_values}")
    print()

    if not GPU_AVAILABLE:
        print("ERROR: GPU not available, cannot run benchmark")
        return None

    results = {
        'timestamp': datetime.now().isoformat(),
        'nfreq': nfreq,
        'n_trials': n_trials,
        'benchmarks': []
    }

    freqs = np.linspace(0.05, 0.5, nfreq).astype(np.float32)

    for ndata in ndata_values:
        print(f"Testing ndata={ndata}...")

        t, y, dy = generate_test_data(ndata)

        # Determine block size
        block_size = bls._choose_block_size(ndata)
        print(f"  Selected block_size: {block_size}")

        bench = {
            'ndata': int(ndata),
            'block_size': int(block_size)
        }

        # Benchmark 1: Standard (baseline, block_size=256)
        print("  Standard (block_size=256):")
        times_std = []

        # Warm-up
        try:
            _ = bls.eebls_gpu_fast(t, y, dy, freqs)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power_std = bls.eebls_gpu_fast(t, y, dy, freqs)
            elapsed = time.time() - start
            times_std.append(elapsed)

        mean_std = np.mean(times_std)
        std_std = np.std(times_std)

        print(f"    Mean: {mean_std:.4f}s ± {std_std:.4f}s")
        print(f"    Throughput: {ndata * nfreq / mean_std / 1e6:.2f} M eval/s")

        bench['standard'] = {
            'mean_time': float(mean_std),
            'std_time': float(std_std),
            'throughput_Meval_per_sec': float(ndata * nfreq / mean_std / 1e6)
        }

        # Benchmark 2: Optimized (block_size=256)
        print("  Optimized (block_size=256):")
        times_opt = []

        # Warm-up
        try:
            _ = bls.eebls_gpu_fast_optimized(t, y, dy, freqs)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power_opt = bls.eebls_gpu_fast_optimized(t, y, dy, freqs)
            elapsed = time.time() - start
            times_opt.append(elapsed)

        mean_opt = np.mean(times_opt)
        std_opt = np.std(times_opt)

        print(f"    Mean: {mean_opt:.4f}s ± {std_opt:.4f}s")
        print(f"    Throughput: {ndata * nfreq / mean_opt / 1e6:.2f} M eval/s")

        bench['optimized'] = {
            'mean_time': float(mean_opt),
            'std_time': float(std_opt),
            'throughput_Meval_per_sec': float(ndata * nfreq / mean_opt / 1e6)
        }

        # Benchmark 3: Adaptive
        print(f"  Adaptive (block_size={block_size}):")
        times_adapt = []

        # Warm-up
        try:
            _ = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power_adapt = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
            elapsed = time.time() - start
            times_adapt.append(elapsed)

        mean_adapt = np.mean(times_adapt)
        std_adapt = np.std(times_adapt)

        print(f"    Mean: {mean_adapt:.4f}s ± {std_adapt:.4f}s")
        print(f"    Throughput: {ndata * nfreq / mean_adapt / 1e6:.2f} M eval/s")

        bench['adaptive'] = {
            'mean_time': float(mean_adapt),
            'std_time': float(std_adapt),
            'throughput_Meval_per_sec': float(ndata * nfreq / mean_adapt / 1e6)
        }

        # Check correctness
        max_diff_std = np.max(np.abs(power_adapt - power_std))
        max_diff_opt = np.max(np.abs(power_adapt - power_opt))

        print(f"  Correctness:")
        print(f"    Max diff vs standard: {max_diff_std:.2e}")
        print(f"    Max diff vs optimized: {max_diff_opt:.2e}")

        if max_diff_std > 1e-5 or max_diff_opt > 1e-5:
            print(f"    WARNING: Results differ!")

        bench['max_diff_std'] = float(max_diff_std)
        bench['max_diff_opt'] = float(max_diff_opt)

        # Compute speedups
        speedup_vs_std = mean_std / mean_adapt
        speedup_vs_opt = mean_opt / mean_adapt

        print(f"  Speedup:")
        print(f"    vs standard: {speedup_vs_std:.2f}x")
        print(f"    vs optimized: {speedup_vs_opt:.2f}x")
        print()

        bench['speedup_vs_std'] = float(speedup_vs_std)
        bench['speedup_vs_opt'] = float(speedup_vs_opt)

        results['benchmarks'].append(bench)

    return results


def print_summary(results):
    """Print summary table."""
    if results is None:
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'ndata':<8} {'Block':<8} {'Standard':<12} {'Optimized':<12} "
          f"{'Adaptive':<12} {'vs Std':<10} {'vs Opt':<10}")
    print("-" * 80)

    for bench in results['benchmarks']:
        print(f"{bench['ndata']:<8} "
              f"{bench['block_size']:<8} "
              f"{bench['standard']['mean_time']:<12.4f} "
              f"{bench['optimized']['mean_time']:<12.4f} "
              f"{bench['adaptive']['mean_time']:<12.4f} "
              f"{bench['speedup_vs_std']:<10.2f}x "
              f"{bench['speedup_vs_opt']:<10.2f}x")


def save_results(results, filename):
    """Save results to JSON file."""
    if results is None:
        return

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    """Run benchmark suite."""
    # Extended test range focusing on small ndata where adaptive helps most
    ndata_values = [10, 20, 30, 50, 64, 100, 128, 200, 500, 1000, 5000, 10000]
    nfreq = 1000
    n_trials = 5

    results = benchmark_adaptive(ndata_values, nfreq=nfreq, n_trials=n_trials)
    print_summary(results)
    save_results(results, 'bls_adaptive_benchmark.json')

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
