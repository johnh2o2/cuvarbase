#!/usr/bin/env python3
"""
Benchmark script for BLS kernel optimization.

Tests BLS performance on various lightcurve sizes to establish baseline
and measure improvements from kernel optimizations.
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


def benchmark_bls(ndata_values, nfreq=1000, n_trials=5):
    """
    Benchmark BLS for different data sizes.

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
    print("BLS KERNEL OPTIMIZATION BASELINE BENCHMARK")
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

        times = []

        # Warm-up run
        try:
            _ = bls.eebls_gpu_fast(t, y, dy, freqs)
        except Exception as e:
            print(f"  ERROR on warm-up: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power = bls.eebls_gpu_fast(t, y, dy, freqs)
            elapsed = time.time() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)

        print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"  Min:  {min_time:.4f}s")
        print(f"  Throughput: {ndata * nfreq / mean_time / 1e6:.2f} M eval/s")

        results['benchmarks'].append({
            'ndata': int(ndata),
            'mean_time': float(mean_time),
            'std_time': float(std_time),
            'min_time': float(min_time),
            'times': [float(t) for t in times],
            'throughput_Meval_per_sec': float(ndata * nfreq / mean_time / 1e6)
        })

    return results


def print_summary(results):
    """Print summary table."""
    if results is None:
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'ndata':<10} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Throughput (M/s)'}")
    print("-" * 80)

    for bench in results['benchmarks']:
        print(f"{bench['ndata']:<10} {bench['mean_time']:<15.4f} "
              f"{bench['std_time']:<15.4f} {bench['throughput_Meval_per_sec']:<15.2f}")


def save_results(results, filename):
    """Save results to JSON file."""
    if results is None:
        return

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    """Run benchmark suite."""
    # Test sizes: 10, 100, 1000, 10000 as requested
    ndata_values = [10, 100, 1000, 10000]
    nfreq = 1000
    n_trials = 5

    results = benchmark_bls(ndata_values, nfreq=nfreq, n_trials=n_trials)
    print_summary(results)
    save_results(results, 'bls_baseline_benchmark.json')

    print("\n" + "=" * 80)
    print("BASELINE ESTABLISHED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Analyze kernel for optimization opportunities")
    print("2. Implement optimizations")
    print("3. Re-run this benchmark to measure improvements")
    print("4. Compare results: python scripts/compare_bls_benchmarks.py")


if __name__ == '__main__':
    main()
