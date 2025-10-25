#!/usr/bin/env python3
"""
Benchmark BLS with realistic parameters for batch lightcurve processing.

Uses:
- 10-year time baseline
- Keplerian frequency/q grids
- Typical TESS/ground-based survey ndata values
- Batch processing of multiple lightcurves
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


def generate_realistic_lightcurve(ndata, time_baseline_years=10, period=None,
                                   depth=0.01, rho_star=1.0, seed=None):
    """
    Generate realistic lightcurve for survey data.

    Parameters
    ----------
    ndata : int
        Number of observations
    time_baseline_years : float
        Total time baseline in years
    period : float, optional
        Transit period in days. If None, generates noise only.
    depth : float
        Transit depth
    rho_star : float
        Stellar density in solar units (for Keplerian q)
    seed : int, optional
        Random seed

    Returns
    -------
    t, y, dy : arrays
        Time, magnitude, and uncertainties
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate realistic time sampling (gaps, clusters)
    time_baseline_days = time_baseline_years * 365.25

    # Simulate survey observing pattern: clusters of observations with gaps
    n_seasons = int(time_baseline_years)
    points_per_season = ndata // n_seasons

    t_list = []
    for season in range(n_seasons):
        season_start = season * 365.25
        season_end = season_start + 200  # 200-day observing season

        # Random observations within season
        t_season = np.random.uniform(season_start, season_end, points_per_season)
        t_list.append(t_season)

    # Add remaining points
    remaining = ndata - len(np.concatenate(t_list))
    if remaining > 0:
        t_extra = np.random.uniform(0, time_baseline_days, remaining)
        t_list.append(t_extra)

    t = np.sort(np.concatenate(t_list)).astype(np.float32)
    t = t[:ndata]  # Ensure exact ndata

    y = np.ones(ndata, dtype=np.float32)

    if period is not None:
        # Add realistic transit signal with Keplerian duration
        phase = (t % period) / period

        # Transit duration from Keplerian assumption
        q = bls.q_transit(1.0/period, rho=rho_star)

        in_transit = phase < q
        y[in_transit] -= depth

    # Add realistic noise
    scatter = 0.01  # 1% photometric precision
    y += np.random.normal(0, scatter, ndata).astype(np.float32)
    dy = np.ones(ndata, dtype=np.float32) * scatter

    return t, y, dy


def get_keplerian_grid(t, fmin_frac=1.0, fmax_frac=1.0, samples_per_peak=2,
                       qmin_fac=0.5, qmax_fac=2.0, rho=1.0):
    """
    Generate Keplerian frequency grid for realistic BLS search.

    Parameters
    ----------
    t : array
        Observation times
    fmin_frac, fmax_frac : float
        Fraction of auto-determined limits
    samples_per_peak : float
        Oversampling factor
    qmin_fac, qmax_fac : float
        Fraction of Keplerian q to search
    rho : float
        Stellar density in solar units

    Returns
    -------
    freqs : array
        Frequency grid
    qmins, qmaxes : arrays
        Min and max q values for each frequency
    """
    fmin = bls.fmin_transit(t, rho=rho) * fmin_frac
    fmax = bls.fmax_transit(rho=rho, qmax=0.5/qmax_fac) * fmax_frac

    freqs, q0vals = bls.transit_autofreq(t, fmin=fmin, fmax=fmax,
                                         samples_per_peak=samples_per_peak,
                                         qmin_fac=qmin_fac, qmax_fac=qmax_fac,
                                         rho=rho)

    qmins = q0vals * qmin_fac
    qmaxes = q0vals * qmax_fac

    return freqs, qmins, qmaxes


def benchmark_single_vs_batch(ndata, n_lightcurves, time_baseline=10, n_trials=3):
    """
    Benchmark single lightcurve vs batch processing.

    Parameters
    ----------
    ndata : int
        Number of observations per lightcurve
    n_lightcurves : int
        Number of lightcurves to process
    time_baseline : float
        Time baseline in years
    n_trials : int
        Number of trials

    Returns
    -------
    results : dict
        Benchmark results
    """
    print(f"\nBenchmarking ndata={ndata}, n_lightcurves={n_lightcurves}...")

    # Generate realistic lightcurves
    lightcurves = []
    for i in range(n_lightcurves):
        t, y, dy = generate_realistic_lightcurve(ndata, time_baseline_years=time_baseline,
                                                 period=5.0 if i % 3 == 0 else None,
                                                 seed=42+i)
        lightcurves.append((t, y, dy))

    # Generate Keplerian frequency grid (same for all)
    t0, _, _ = lightcurves[0]
    freqs, qmins, qmaxes = get_keplerian_grid(t0)

    nfreq = len(freqs)
    print(f"  Keplerian grid: {nfreq} frequencies")
    print(f"  Period range: {1/freqs[-1]:.2f} - {1/freqs[0]:.2f} days")

    results = {
        'ndata': int(ndata),
        'n_lightcurves': int(n_lightcurves),
        'nfreq': int(nfreq),
        'time_baseline_years': float(time_baseline)
    }

    # Benchmark 1: Sequential processing with standard kernel
    print("  Sequential (standard)...")
    times_seq_std = []

    for trial in range(n_trials):
        start = time.time()
        for t, y, dy in lightcurves:
            _ = bls.eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)
        elapsed = time.time() - start
        times_seq_std.append(elapsed)

    mean_seq_std = np.mean(times_seq_std)
    print(f"    Mean: {mean_seq_std:.3f}s")
    print(f"    Per LC: {mean_seq_std/n_lightcurves:.3f}s")

    results['sequential_standard'] = {
        'total_time': float(mean_seq_std),
        'per_lc_time': float(mean_seq_std / n_lightcurves),
        'throughput_lc_per_sec': float(n_lightcurves / mean_seq_std)
    }

    # Benchmark 2: Sequential with adaptive kernel
    print("  Sequential (adaptive)...")
    times_seq_adapt = []

    for trial in range(n_trials):
        start = time.time()
        for t, y, dy in lightcurves:
            _ = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)
        elapsed = time.time() - start
        times_seq_adapt.append(elapsed)

    mean_seq_adapt = np.mean(times_seq_adapt)
    print(f"    Mean: {mean_seq_adapt:.3f}s")
    print(f"    Per LC: {mean_seq_adapt/n_lightcurves:.3f}s")

    results['sequential_adaptive'] = {
        'total_time': float(mean_seq_adapt),
        'per_lc_time': float(mean_seq_adapt / n_lightcurves),
        'throughput_lc_per_sec': float(n_lightcurves / mean_seq_adapt)
    }

    # Compute speedups
    speedup = mean_seq_std / mean_seq_adapt
    print(f"  Speedup (adaptive vs standard): {speedup:.2f}x")

    results['speedup_adaptive_vs_standard'] = float(speedup)

    # Estimate cost savings
    cost_per_hour = 0.34  # RunPod RTX 4000 Ada spot price
    hours_std = (mean_seq_std / 3600) * (5e6 / n_lightcurves)  # Scale to 5M LCs
    hours_adapt = (mean_seq_adapt / 3600) * (5e6 / n_lightcurves)

    cost_std = hours_std * cost_per_hour
    cost_adapt = hours_adapt * cost_per_hour
    cost_savings = cost_std - cost_adapt

    print(f"\n  Estimated cost for 5M lightcurves:")
    print(f"    Standard: ${cost_std:.2f} ({hours_std:.1f} hours)")
    print(f"    Adaptive: ${cost_adapt:.2f} ({hours_adapt:.1f} hours)")
    print(f"    Savings: ${cost_savings:.2f} ({100*(1-cost_adapt/cost_std):.1f}%)")

    results['cost_estimate_5M_lcs'] = {
        'standard_usd': float(cost_std),
        'adaptive_usd': float(cost_adapt),
        'savings_usd': float(cost_savings),
        'savings_percent': float(100*(1-cost_adapt/cost_std))
    }

    return results


def main():
    """Run realistic batch benchmark."""
    print("=" * 80)
    print("BATCH KEPLERIAN BLS BENCHMARK")
    print("=" * 80)
    print("\nRealistic parameters:")
    print("  - 10-year time baseline")
    print("  - Keplerian frequency/q grids")
    print("  - Survey-like time sampling (seasonal gaps)")
    print()

    if not GPU_AVAILABLE:
        print("ERROR: GPU not available")
        return

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': []
    }

    # Test configurations representing different survey types
    configs = [
        # (ndata, n_lcs, description)
        (100, 10, "Sparse ground-based (e.g., MEarth, HATNet)"),
        (500, 10, "Dense ground-based (e.g., NGTS, HATPI)"),
        (20000, 5, "Space-based (e.g., TESS, Kepler)"),
    ]

    for ndata, n_lcs, desc in configs:
        print(f"\n{desc}")
        print("-" * 80)

        results = benchmark_single_vs_batch(ndata, n_lcs, time_baseline=10, n_trials=3)
        results['description'] = desc
        all_results['benchmarks'].append(results)

    # Save results
    filename = 'bls_batch_keplerian_benchmark.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {filename}")
    print("=" * 80)


if __name__ == '__main__':
    main()
