#!/usr/bin/env python3
"""
Compare baseline vs optimized BLS kernel performance.

This script benchmarks both the standard and optimized BLS kernels
to measure the speedup from our optimizations.
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


def benchmark_comparison(ndata_values, nfreq=1000, n_trials=5):
    """
    Compare standard vs optimized BLS kernels.

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
    print("BLS KERNEL OPTIMIZATION COMPARISON")
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

        # Benchmark standard kernel
        print("  Standard kernel:")
        times_standard = []

        # Warm-up
        try:
            _ = bls.eebls_gpu_fast(t, y, dy, freqs)
        except Exception as e:
            print(f"    ERROR on warm-up: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power_std = bls.eebls_gpu_fast(t, y, dy, freqs)
            elapsed = time.time() - start
            times_standard.append(elapsed)

        mean_std = np.mean(times_standard)
        std_std = np.std(times_standard)

        print(f"    Mean: {mean_std:.4f}s ± {std_std:.4f}s")
        print(f"    Throughput: {ndata * nfreq / mean_std / 1e6:.2f} M eval/s")

        # Benchmark optimized kernel
        print("  Optimized kernel:")
        times_optimized = []

        # Warm-up
        try:
            _ = bls.eebls_gpu_fast_optimized(t, y, dy, freqs)
        except Exception as e:
            print(f"    ERROR on warm-up: {e}")
            continue

        # Timed runs
        for trial in range(n_trials):
            start = time.time()
            power_opt = bls.eebls_gpu_fast_optimized(t, y, dy, freqs)
            elapsed = time.time() - start
            times_optimized.append(elapsed)

        mean_opt = np.mean(times_optimized)
        std_opt = np.std(times_optimized)

        print(f"    Mean: {mean_opt:.4f}s ± {std_opt:.4f}s")
        print(f"    Throughput: {ndata * nfreq / mean_opt / 1e6:.2f} M eval/s")

        # Check correctness
        max_diff = np.max(np.abs(power_std - power_opt))
        print(f"  Max difference: {max_diff:.2e}")

        if max_diff > 1e-5:
            print(f"  WARNING: Results differ by more than 1e-5!")

        # Compute speedup
        speedup = mean_std / mean_opt
        print(f"  Speedup: {speedup:.2f}x")
        print()

        results['benchmarks'].append({
            'ndata': int(ndata),
            'standard': {
                'mean_time': float(mean_std),
                'std_time': float(std_std),
                'times': [float(t) for t in times_standard],
                'throughput_Meval_per_sec': float(ndata * nfreq / mean_std / 1e6)
            },
            'optimized': {
                'mean_time': float(mean_opt),
                'std_time': float(std_opt),
                'times': [float(t) for t in times_optimized],
                'throughput_Meval_per_sec': float(ndata * nfreq / mean_opt / 1e6)
            },
            'speedup': float(speedup),
            'max_diff': float(max_diff)
        })

    return results


def print_summary(results):
    """Print summary table."""
    if results is None:
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'ndata':<10} {'Standard (s)':<15} {'Optimized (s)':<15} {'Speedup':<10} {'Max Diff'}")
    print("-" * 80)

    for bench in results['benchmarks']:
        print(f"{bench['ndata']:<10} "
              f"{bench['standard']['mean_time']:<15.4f} "
              f"{bench['optimized']['mean_time']:<15.4f} "
              f"{bench['speedup']:<10.2f}x "
              f"{bench['max_diff']:.2e}")


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

    results = benchmark_comparison(ndata_values, nfreq=nfreq, n_trials=n_trials)
    print_summary(results)
    save_results(results, 'bls_optimization_comparison.json')

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
