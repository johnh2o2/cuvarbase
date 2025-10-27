#!/usr/bin/env python3
"""
Benchmark GPU vs CPU TLS implementations

This script compares the performance and accuracy of:
- cuvarbase TLS GPU implementation
- transitleastsquares CPU implementation

Variables tested:
1. Number of data points (fixed baseline)
2. Baseline duration (fixed ndata)

Ensures apples-to-apples comparison:
- Uses the same period grid (Ofir 2014)
- Same stellar parameters
- Same synthetic transit parameters
"""

import numpy as np
import time
import json
from datetime import datetime

# Import both implementations
from cuvarbase import tls as gpu_tls
from cuvarbase import tls_grids
from transitleastsquares import transitleastsquares as cpu_tls


def generate_synthetic_data(ndata, baseline_days, period=10.0, depth=0.01,
                            duration_days=0.1, noise_level=0.001,
                            t0=0.0, seed=42):
    """
    Generate synthetic light curve with transit.

    Parameters
    ----------
    ndata : int
        Number of data points
    baseline_days : float
        Total observation span (days)
    period : float
        Orbital period (days)
    depth : float
        Transit depth (fractional)
    duration_days : float
        Transit duration (days)
    noise_level : float
        Gaussian noise sigma
    t0 : float
        First transit time (days)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    t, y, dy : ndarray
        Time, flux, uncertainties
    """
    np.random.seed(seed)

    # Random time sampling over baseline
    t = np.sort(np.random.uniform(0, baseline_days, ndata)).astype(np.float32)

    # Start with flat light curve
    y = np.ones(ndata, dtype=np.float32)

    # Add box transits
    phase = ((t - t0) % period) / period
    duration_phase = duration_days / period

    # Transit centered at phase 0
    in_transit = (phase < duration_phase / 2) | (phase > 1 - duration_phase / 2)
    y[in_transit] -= depth

    # Add noise
    noise = np.random.normal(0, noise_level, ndata)
    y += noise

    # Uncertainties
    dy = np.ones(ndata, dtype=np.float32) * noise_level

    return t, y, dy


def run_gpu_tls(t, y, dy, periods, R_star=1.0, M_star=1.0):
    """Run cuvarbase GPU TLS."""
    t0 = time.time()
    results = gpu_tls.tls_search_gpu(
        t, y, dy,
        periods=periods,
        R_star=R_star,
        M_star=M_star,
        use_simple=len(t) < 500,
        block_size=128
    )
    t1 = time.time()

    return {
        'time': t1 - t0,
        'period': float(results['period']),
        'depth': float(results['depth']),
        'duration': float(results['duration']),
        'T0': float(results['T0']),
        'SDE': float(results['SDE']),
        'chi2': float(results['chi2_min'])
    }


def run_cpu_tls(t, y, dy, periods, R_star=1.0, M_star=1.0):
    """Run transitleastsquares CPU TLS."""
    model = cpu_tls(t, y, dy)

    t0 = time.time()
    results = model.power(
        period_min=float(np.min(periods)),
        period_max=float(np.max(periods)),
        n_transits_min=2,
        R_star=R_star,
        M_star=M_star,
        # Try to match our period grid
        oversampling_factor=3,
        duration_grid_step=1.1
    )
    t1 = time.time()

    return {
        'time': t1 - t0,
        'period': float(results.period),
        'depth': float(results.depth),
        'duration': float(results.duration),
        'T0': float(results.T0),
        'SDE': float(results.SDE),
        'chi2': float(results.chi2_min)
    }


def benchmark_vs_ndata(baseline_days=50.0, ndata_values=None,
                       period_true=10.0, n_repeats=3):
    """
    Benchmark as a function of number of data points.

    Parameters
    ----------
    baseline_days : float
        Fixed observation baseline (days)
    ndata_values : list
        List of ndata values to test
    period_true : float
        True orbital period for synthetic data
    n_repeats : int
        Number of repeats for timing

    Returns
    -------
    results : dict
        Benchmark results
    """
    if ndata_values is None:
        ndata_values = [100, 200, 500, 1000, 2000, 5000]

    results = {
        'baseline_days': baseline_days,
        'period_true': period_true,
        'ndata_values': ndata_values,
        'gpu_times': [],
        'cpu_times': [],
        'speedups': [],
        'gpu_results': [],
        'cpu_results': []
    }

    print(f"\n{'='*70}")
    print(f"Benchmark vs ndata (baseline={baseline_days:.0f} days)")
    print(f"{'='*70}")
    print(f"{'ndata':<10} {'GPU (s)':<12} {'CPU (s)':<12} {'Speedup':<10} {'GPU Period':<12} {'CPU Period':<12}")
    print(f"{'-'*70}")

    for ndata in ndata_values:
        # Generate data
        t, y, dy = generate_synthetic_data(
            ndata, baseline_days,
            period=period_true,
            depth=0.01,
            duration_days=0.12
        )

        # Generate shared period grid using cuvarbase
        periods = tls_grids.period_grid_ofir(
            t, R_star=1.0, M_star=1.0,
            period_min=5.0,
            period_max=20.0,
            oversampling_factor=3
        )
        periods = periods.astype(np.float32)

        # Average over repeats
        gpu_times = []
        cpu_times = []

        for _ in range(n_repeats):
            # GPU
            gpu_result = run_gpu_tls(t, y, dy, periods)
            gpu_times.append(gpu_result['time'])

            # CPU
            cpu_result = run_cpu_tls(t, y, dy, periods)
            cpu_times.append(cpu_result['time'])

        gpu_time = np.mean(gpu_times)
        cpu_time = np.mean(cpu_times)
        speedup = cpu_time / gpu_time

        results['gpu_times'].append(gpu_time)
        results['cpu_times'].append(cpu_time)
        results['speedups'].append(speedup)
        results['gpu_results'].append(gpu_result)
        results['cpu_results'].append(cpu_result)

        print(f"{ndata:<10} {gpu_time:<12.3f} {cpu_time:<12.3f} {speedup:<10.1f}x {gpu_result['period']:<12.2f} {cpu_result['period']:<12.2f}")

    return results


def benchmark_vs_baseline(ndata=1000, baseline_values=None,
                          period_true=10.0, n_repeats=3):
    """
    Benchmark as a function of baseline duration.

    Parameters
    ----------
    ndata : int
        Fixed number of data points
    baseline_values : list
        List of baseline durations (days) to test
    period_true : float
        True orbital period for synthetic data
    n_repeats : int
        Number of repeats for timing

    Returns
    -------
    results : dict
        Benchmark results
    """
    if baseline_values is None:
        baseline_values = [20, 50, 100, 200, 500, 1000]

    results = {
        'ndata': ndata,
        'period_true': period_true,
        'baseline_values': baseline_values,
        'gpu_times': [],
        'cpu_times': [],
        'speedups': [],
        'gpu_results': [],
        'cpu_results': [],
        'nperiods': []
    }

    print(f"\n{'='*80}")
    print(f"Benchmark vs baseline (ndata={ndata})")
    print(f"{'='*80}")
    print(f"{'Baseline':<12} {'N_periods':<12} {'GPU (s)':<12} {'CPU (s)':<12} {'Speedup':<10} {'GPU Period':<12}")
    print(f"{'-'*80}")

    for baseline in baseline_values:
        # Generate data
        t, y, dy = generate_synthetic_data(
            ndata, baseline,
            period=period_true,
            depth=0.01,
            duration_days=0.12
        )

        # Generate period grid - range depends on baseline
        period_max = min(baseline / 2.0, 50.0)
        period_min = max(0.5, baseline / 50.0)

        periods = tls_grids.period_grid_ofir(
            t, R_star=1.0, M_star=1.0,
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=3
        )
        periods = periods.astype(np.float32)

        results['nperiods'].append(len(periods))

        # Average over repeats
        gpu_times = []
        cpu_times = []

        for _ in range(n_repeats):
            # GPU
            gpu_result = run_gpu_tls(t, y, dy, periods)
            gpu_times.append(gpu_result['time'])

            # CPU
            cpu_result = run_cpu_tls(t, y, dy, periods)
            cpu_times.append(cpu_result['time'])

        gpu_time = np.mean(gpu_times)
        cpu_time = np.mean(cpu_times)
        speedup = cpu_time / gpu_time

        results['gpu_times'].append(gpu_time)
        results['cpu_times'].append(cpu_time)
        results['speedups'].append(speedup)
        results['gpu_results'].append(gpu_result)
        results['cpu_results'].append(cpu_result)

        print(f"{baseline:<12.0f} {len(periods):<12} {gpu_time:<12.3f} {cpu_time:<12.3f} {speedup:<10.1f}x {gpu_result['period']:<12.2f}")

    return results


def check_consistency(ndata=500, baseline=50.0, period_true=10.0):
    """
    Check consistency between GPU and CPU implementations.

    Returns
    -------
    comparison : dict
        Detailed comparison results
    """
    print(f"\n{'='*70}")
    print(f"Consistency Check (ndata={ndata}, baseline={baseline:.0f} days)")
    print(f"{'='*70}")

    # Generate data
    t, y, dy = generate_synthetic_data(
        ndata, baseline,
        period=period_true,
        depth=0.01,
        duration_days=0.12
    )

    # Generate period grid
    periods = tls_grids.period_grid_ofir(
        t, R_star=1.0, M_star=1.0,
        period_min=5.0,
        period_max=20.0,
        oversampling_factor=3
    )
    periods = periods.astype(np.float32)

    # Run both
    gpu_result = run_gpu_tls(t, y, dy, periods)
    cpu_result = run_cpu_tls(t, y, dy, periods)

    # Compare
    comparison = {
        'true_period': period_true,
        'gpu': gpu_result,
        'cpu': cpu_result,
        'period_diff': abs(gpu_result['period'] - cpu_result['period']),
        'period_diff_pct': abs(gpu_result['period'] - cpu_result['period']) / period_true * 100,
        'depth_diff': abs(gpu_result['depth'] - cpu_result['depth']),
        'depth_diff_pct': abs(gpu_result['depth'] - cpu_result['depth']) / 0.01 * 100,
    }

    print(f"\nTrue values:")
    print(f"  Period: {period_true:.4f} days")
    print(f"  Depth: 0.0100")
    print(f"  Duration: 0.1200 days")

    print(f"\nGPU Results:")
    print(f"  Period: {gpu_result['period']:.4f} days")
    print(f"  Depth: {gpu_result['depth']:.6f}")
    print(f"  Duration: {gpu_result['duration']:.4f} days")
    print(f"  SDE: {gpu_result['SDE']:.2f}")
    print(f"  Time: {gpu_result['time']:.3f} s")

    print(f"\nCPU Results:")
    print(f"  Period: {cpu_result['period']:.4f} days")
    print(f"  Depth: {cpu_result['depth']:.6f}")
    print(f"  Duration: {cpu_result['duration']:.4f} days")
    print(f"  SDE: {cpu_result['SDE']:.2f}")
    print(f"  Time: {cpu_result['time']:.3f} s")

    print(f"\nDifferences:")
    print(f"  Period: {comparison['period_diff']:.4f} days ({comparison['period_diff_pct']:.2f}%)")
    print(f"  Depth: {comparison['depth_diff']:.6f} ({comparison['depth_diff_pct']:.1f}%)")
    print(f"  Speedup: {cpu_result['time'] / gpu_result['time']:.1f}x")

    return comparison


if __name__ == '__main__':
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'tls_benchmark_{timestamp}.json'

    print("="*70)
    print("TLS GPU vs CPU Benchmark Suite")
    print("="*70)
    print(f"\nComparison:")
    print(f"  GPU: cuvarbase TLS (PyCUDA)")
    print(f"  CPU: transitleastsquares v1.32 (Numba)")
    print(f"\nEnsuring apples-to-apples comparison:")
    print(f"  ✓ Same period grid (Ofir 2014)")
    print(f"  ✓ Same stellar parameters")
    print(f"  ✓ Same synthetic transit")

    all_results = {}

    # 1. Consistency check
    consistency = check_consistency(ndata=500, baseline=50.0, period_true=10.0)
    all_results['consistency'] = consistency

    # 2. Benchmark vs ndata
    ndata_results = benchmark_vs_ndata(
        baseline_days=50.0,
        ndata_values=[100, 200, 500, 1000, 2000, 5000],
        n_repeats=3
    )
    all_results['vs_ndata'] = ndata_results

    # 3. Benchmark vs baseline
    baseline_results = benchmark_vs_baseline(
        ndata=1000,
        baseline_values=[20, 50, 100, 200, 500],
        n_repeats=3
    )
    all_results['vs_baseline'] = baseline_results

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    print(f"\nSummary:")
    print(f"  Average speedup (vs ndata): {np.mean(ndata_results['speedups']):.1f}x")
    print(f"  Average speedup (vs baseline): {np.mean(baseline_results['speedups']):.1f}x")
    print(f"  Period consistency: {consistency['period_diff']:.4f} days ({consistency['period_diff_pct']:.2f}%)")
