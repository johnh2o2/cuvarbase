#!/usr/bin/env python3
"""
Correctness tests and benchmarks for BLS batch + cuFINUFFT LS features.

Tests:
  A) BLS batch correctness: batch vs single-LC loop at multiple ndata
  B) cuFINUFFT LS correctness: cufinufft vs custom NFFT backend
  C) Keplerian frequency grid validation

Benchmarks:
  D) BLS batch throughput across survey profiles (ZTF, HAT-Net, TESS, Kepler)
  E) cuFINUFFT LS performance across ndata x nfreq grid
  F) Keplerian grid impact (frequency reduction + BLS time savings)

Usage:
    python scripts/benchmark_new_features.py                # all tests + benchmarks
    python scripts/benchmark_new_features.py --tests-only   # correctness only
    python scripts/benchmark_new_features.py --bench-only   # benchmarks only
    python scripts/benchmark_new_features.py --skip-cufinufft  # skip cufinufft tests

Output: JSON results in benchmarks/results/benchmark_results_new_features.json
"""

import numpy as np
import time
import json
import sys
import traceback
import argparse
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# GPU imports
# ---------------------------------------------------------------------------
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("ERROR: pycuda not available. GPU required for these benchmarks.")
    sys.exit(1)

import cuvarbase.bls as cvb_bls
import cuvarbase.lombscargle as cvb_ls
from cuvarbase.bls_frequencies import (
    keplerian_freq_grid, uniform_freq_grid, freq_grid_stats
)

HAS_CUFINUFFT = False
try:
    from cuvarbase.cufinufft_backend import HAS_CUFINUFFT
except ImportError:
    pass

HAS_NIFTY_LS = False
try:
    import nifty_ls
    HAS_NIFTY_LS = True
except ImportError:
    pass

HAS_ASTROPY = False
try:
    from astropy.timeseries import BoxLeastSquares, LombScargle
    HAS_ASTROPY = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_function(func, n_iter=3, warmup=1):
    """Time a zero-argument callable, return (median_seconds, all_times)."""
    for _ in range(warmup):
        func()
    cuda.Context.synchronize()

    times = []
    for _ in range(n_iter):
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        func()
        cuda.Context.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), times


def time_function_cpu(func, n_iter=3, warmup=1, timeout=60.0):
    """Time a CPU function with timeout. Returns None if exceeds timeout."""
    for _ in range(warmup):
        t0 = time.perf_counter()
        func()
        if time.perf_counter() - t0 > timeout:
            return None, []

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if t1 - t0 > timeout:
            break

    return float(np.median(times)), times


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_transit_lc(ndata, baseline, period, depth=0.01, duration_frac=0.02,
                        noise=0.002, seed=None):
    """Generate a lightcurve with an injected box transit."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float32)
    phase = (t % period) / period
    y = np.ones(ndata, dtype=np.float32)
    in_transit = phase < duration_frac
    y[in_transit] -= depth
    y += rng.randn(ndata).astype(np.float32) * noise
    dy = np.full(ndata, noise, dtype=np.float32)
    return t, y, dy


def generate_sinusoidal_lc(ndata, baseline, period, amplitude=0.01,
                           noise=0.002, seed=None):
    """Generate a lightcurve with an injected sinusoidal signal."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float32)
    y = amplitude * np.cos(2 * np.pi * t / period).astype(np.float32)
    y += rng.randn(ndata).astype(np.float32) * noise
    dy = np.full(ndata, noise, dtype=np.float32)
    return t, y, dy


# ---------------------------------------------------------------------------
# Survey profiles
# ---------------------------------------------------------------------------

SURVEY_PROFILES = OrderedDict([
    ('ZTF-like', {
        'ndata': 150,
        'baseline': 730.0,
        'period_min': 0.5,
        'period_max': 100.0,
        'nlcs_bench': 500,
        'qmin': 0.01,
        'qmax': 0.15,
        'inject_period': 3.0,
    }),
    ('HAT-Net', {
        'ndata': 6000,
        'baseline': 3650.0,
        'period_min': 0.5,
        'period_max': 100.0,
        'nlcs_bench': 200,
        'qmin': 0.01,
        'qmax': 0.1,
        'inject_period': 2.5,
    }),
    ('TESS-1sector', {
        'ndata': 20000,
        'baseline': 27.0,
        'period_min': 0.5,
        'period_max': 13.5,
        'nlcs_bench': 50,
        'qmin': 0.005,
        'qmax': 0.1,
        'inject_period': 5.0,
    }),
    ('Kepler', {
        'ndata': 65000,
        'baseline': 1460.0,
        'period_min': 0.5,
        'period_max': 500.0,
        'nlcs_bench': 10,
        'qmin': 0.005,
        'qmax': 0.1,
        'inject_period': 10.0,
    }),
])


# ============================================================================
# A) BLS Batch Correctness
# ============================================================================

def test_bls_batch_correctness():
    """Compare batch BLS vs single-LC loop across ndata values."""
    print("\n" + "=" * 70)
    print("A) BLS Batch Correctness Tests")
    print("=" * 70)

    results = {}
    test_configs = [
        (200,  730.0, 3.0),
        (2000, 180.0, 2.5),
        (20000, 27.0, 5.0),
    ]

    nfreq = 2000
    qmin, qmax = 0.01, 0.15
    n_lcs = 10

    all_pass = True

    for ndata, baseline, inject_period in test_configs:
        print(f"\n  ndata={ndata}, baseline={baseline}d, "
              f"inject_P={inject_period}d, nlcs={n_lcs}")

        # Generate lightcurves
        lightcurves = []
        for i in range(n_lcs):
            t, y, dy = generate_transit_lc(
                ndata, baseline, inject_period,
                depth=0.01, noise=0.003, seed=42 + i
            )
            lightcurves.append((t, y, dy))

        # Frequency grid
        fmin = 1.0 / min(inject_period * 2, baseline / 2)
        fmax = 1.0 / max(0.3, inject_period / 3)
        freqs = np.linspace(fmin, fmax, nfreq).astype(np.float32)

        # Single-LC loop
        single_results = []
        for t, y, dy in lightcurves:
            bls = cvb_bls.eebls_gpu_fast_adaptive(
                t, y, dy, freqs, qmin=qmin, qmax=qmax
            )
            single_results.append(np.array(bls))
        cuda.Context.synchronize()

        # Batch
        batch_results = cvb_bls.eebls_gpu_batch(
            lightcurves, freqs, qmin=qmin, qmax=qmax
        )
        cuda.Context.synchronize()

        # Compare: peaks must match; absolute values may differ due to
        # float32 accumulation precision (batch preprocesses in float64,
        # single-LC may use float32 weights depending on input dtype).
        max_rdiff = 0.0
        peaks_match = 0
        min_corr = 1.0
        all_close = True
        for i in range(n_lcs):
            s = np.asarray(single_results[i], dtype=np.float64)
            b = np.asarray(batch_results[i], dtype=np.float64)

            if s.shape != b.shape:
                print(f"    LC {i}: SHAPE MISMATCH {s.shape} vs {b.shape}")
                all_close = False
                continue

            # Primary check: peak frequency matches
            peak_s = freqs[np.argmax(s)]
            peak_b = freqs[np.argmax(b)]
            df = freqs[1] - freqs[0]
            if abs(peak_s - peak_b) < df * 2:
                peaks_match += 1

            # Correlation check: periodogram shapes must be correlated
            corr = np.corrcoef(s, b)[0, 1]
            min_corr = min(min_corr, corr)

            rdiff = np.max(np.abs(s - b) / (np.abs(s) + 1e-10))
            max_rdiff = max(max_rdiff, rdiff)

        # Pass if: all peaks match AND correlation > 0.99
        peaks_ok = peaks_match == n_lcs
        corr_ok = min_corr > 0.99
        config_pass = peaks_ok and corr_ok
        if not config_pass:
            all_pass = False

        status = "PASS" if config_pass else "FAIL"
        print(f"    {status}: peak_match={peaks_match}/{n_lcs}, "
              f"corr={min_corr:.6f}, max_rdiff={max_rdiff:.2e}")

        results[f"ndata_{ndata}"] = {
            'ndata': ndata,
            'baseline': baseline,
            'n_lcs': n_lcs,
            'nfreq': nfreq,
            'max_rdiff': float(max_rdiff),
            'min_correlation': float(min_corr),
            'peaks_match': peaks_match,
            'pass': config_pass,
        }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass, results


# ============================================================================
# B) cuFINUFFT LS Correctness
# ============================================================================

def test_cufinufft_ls_correctness():
    """Compare cuFINUFFT vs custom NFFT LS backend."""
    print("\n" + "=" * 70)
    print("B) cuFINUFFT LS Correctness Tests")
    print("=" * 70)

    if not HAS_CUFINUFFT:
        print("  SKIPPED: cufinufft not installed")
        return True, {'skipped': True}

    results = {}
    test_configs = [
        (1000,  5000,  365.0, 5.0),
        (5000,  10000, 365.0, 3.0),
        (10000, 20000, 365.0, 7.0),
    ]
    n_lcs = 5
    all_pass = True

    for ndata, nfreq, baseline, inject_period in test_configs:
        print(f"\n  ndata={ndata}, nfreq={nfreq}, baseline={baseline}d, "
              f"inject_P={inject_period}d")

        max_adiff = 0.0
        peak_matches = 0
        min_corr = 1.0
        config_pass = True

        for i in range(n_lcs):
            t, y, dy = generate_sinusoidal_lc(
                ndata, baseline, inject_period,
                amplitude=0.01, noise=0.002, seed=100 + i
            )

            # Frequency grid (NFFT-compatible: freqs = k * df)
            fmax = 2.0
            df = fmax / nfreq
            freqs = (np.arange(1, nfreq + 1) * df).astype(np.float32)

            # Custom NFFT backend
            proc_custom = cvb_ls.LombScargleAsyncProcess(use_cufinufft=False)
            res_custom = proc_custom.run([(t, y, dy)], freqs=[freqs])
            proc_custom.finish()
            _, pow_custom = res_custom[0]

            # cuFINUFFT backend
            proc_cufinufft = cvb_ls.LombScargleAsyncProcess(use_cufinufft=True)
            res_cufinufft = proc_cufinufft.run([(t, y, dy)], freqs=[freqs])
            proc_cufinufft.finish()
            _, pow_cufinufft = res_cufinufft[0]

            pow_c = np.asarray(pow_custom, dtype=np.float64)
            pow_f = np.asarray(pow_cufinufft, dtype=np.float64)

            # Max abs diff (more meaningful than relative for small values)
            adiff = np.max(np.abs(pow_c - pow_f))
            max_adiff = max(max_adiff, adiff)

            # Correlation
            corr = np.corrcoef(pow_c, pow_f)[0, 1]
            min_corr = min(min_corr, corr)

            peak_c = freqs[np.argmax(pow_c)]
            peak_f = freqs[np.argmax(pow_f)]
            if abs(peak_c - peak_f) < df * 2:
                peak_matches += 1

        max_rdiff = max_adiff

        # Pass if: peaks match AND correlation > 0.9999 AND max abs diff < 0.01
        peaks_ok = peak_matches == n_lcs
        corr_ok = min_corr > 0.9999
        adiff_ok = max_rdiff < 0.01
        config_pass = peaks_ok and corr_ok and adiff_ok
        if not config_pass:
            all_pass = False

        status = "PASS" if config_pass else "FAIL"
        print(f"    {status}: max_abs_diff={max_rdiff:.2e}, "
              f"corr={min_corr:.8f}, peak_match={peak_matches}/{n_lcs}")

        results[f"ndata_{ndata}_nfreq_{nfreq}"] = {
            'ndata': ndata,
            'nfreq': nfreq,
            'n_lcs': n_lcs,
            'max_abs_diff': float(max_rdiff),
            'min_correlation': float(min_corr),
            'peak_matches': peak_matches,
            'pass': config_pass,
        }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass, results


# ============================================================================
# C) Keplerian Grid Validation
# ============================================================================

def test_keplerian_grid():
    """Validate Keplerian frequency grids for each survey profile."""
    print("\n" + "=" * 70)
    print("C) Keplerian Frequency Grid Validation")
    print("=" * 70)

    results = {}
    all_pass = True

    for name, profile in SURVEY_PROFILES.items():
        kep_freqs = keplerian_freq_grid(
            profile['period_min'], profile['period_max'], profile['baseline']
        )
        uni_freqs = uniform_freq_grid(
            profile['period_min'], profile['period_max'], profile['baseline']
        )

        stats_kep = freq_grid_stats(kep_freqs, profile['baseline'])
        stats_uni = freq_grid_stats(uni_freqs, profile['baseline'])

        reduction = stats_uni['nfreq'] / max(stats_kep['nfreq'], 1)

        # Validate: Keplerian grid should be strictly smaller
        grid_ok = stats_kep['nfreq'] < stats_uni['nfreq']
        # Validate: freq range covers expected range
        range_ok = (kep_freqs[0] <= 1.0 / profile['period_max'] * 1.01 and
                    kep_freqs[-1] >= 1.0 / profile['period_min'] * 0.99)

        config_pass = grid_ok and range_ok
        if not config_pass:
            all_pass = False

        status = "PASS" if config_pass else "FAIL"
        print(f"\n  {name}: {status}")
        print(f"    Keplerian: {stats_kep['nfreq']:,} freqs")
        print(f"    Uniform:   {stats_uni['nfreq']:,} freqs")
        print(f"    Reduction: {reduction:.1f}x")
        print(f"    Period range: [{stats_kep['period_min']:.2f}, "
              f"{stats_kep['period_max']:.2f}]d")

        results[name] = {
            'keplerian_nfreq': stats_kep['nfreq'],
            'uniform_nfreq': stats_uni['nfreq'],
            'reduction_factor': float(reduction),
            'kep_stats': stats_kep,
            'pass': config_pass,
        }

    # Transit detection check: verify known period is found with both grids
    print("\n  Transit detection with Keplerian grid:")
    t, y, dy = generate_transit_lc(5000, 180.0, 2.5, depth=0.015, seed=99)
    kep_freqs = keplerian_freq_grid(0.5, 10.0, 180.0)
    bls_kep = cvb_bls.eebls_gpu_fast_adaptive(t, y, dy, kep_freqs, qmin=0.01, qmax=0.1)
    detected_period_kep = 1.0 / kep_freqs[np.argmax(bls_kep)]
    detect_ok = abs(detected_period_kep - 2.5) / 2.5 < 0.05
    print(f"    Injected P=2.5d, detected P={detected_period_kep:.3f}d "
          f"({'PASS' if detect_ok else 'FAIL'})")
    if not detect_ok:
        all_pass = False
    results['transit_detection'] = {
        'injected_period': 2.5,
        'detected_period': float(detected_period_kep),
        'pass': detect_ok,
    }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass, results


# ============================================================================
# D) BLS Batch Throughput Benchmark
# ============================================================================

def bench_bls_batch_throughput():
    """Benchmark BLS batch vs single-LC loop across survey profiles."""
    print("\n" + "=" * 70)
    print("D) BLS Batch Throughput Benchmark")
    print("=" * 70)

    results = {}

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        nlcs = profile['nlcs_bench']
        qmin = profile['qmin']
        qmax = profile['qmax']
        inject_period = profile['inject_period']

        print(f"\n  {name}: ndata={ndata}, nlcs={nlcs}, "
              f"baseline={baseline}d")

        # Generate lightcurves
        lightcurves = []
        for i in range(nlcs):
            t, y, dy = generate_transit_lc(
                ndata, baseline, inject_period,
                depth=0.01, noise=0.003, seed=200 + i
            )
            lightcurves.append((t, y, dy))

        # Keplerian frequency grid for this survey
        kep_freqs = keplerian_freq_grid(
            profile['period_min'], profile['period_max'], baseline
        )
        nfreq = len(kep_freqs)
        print(f"    Keplerian freqs: {nfreq}")

        # -- Single-LC loop --
        def run_single():
            for t, y, dy in lightcurves:
                cvb_bls.eebls_gpu_fast_adaptive(
                    t, y, dy, kep_freqs, qmin=qmin, qmax=qmax
                )

        print(f"    Timing single-LC loop ({nlcs} LCs)...", end='', flush=True)
        t_single, times_single = time_function(run_single, n_iter=3, warmup=1)
        lc_per_sec_single = nlcs / t_single
        print(f" {t_single:.3f}s ({lc_per_sec_single:.0f} LC/s)")

        # -- Batch --
        def run_batch():
            cvb_bls.eebls_gpu_batch(
                lightcurves, kep_freqs, qmin=qmin, qmax=qmax
            )

        print(f"    Timing batch ({nlcs} LCs)...", end='', flush=True)
        t_batch, times_batch = time_function(run_batch, n_iter=3, warmup=1)
        lc_per_sec_batch = nlcs / t_batch
        print(f" {t_batch:.3f}s ({lc_per_sec_batch:.0f} LC/s)")

        speedup = t_single / t_batch if t_batch > 0 else float('inf')
        print(f"    Batch speedup: {speedup:.2f}x")

        results[name] = {
            'ndata': ndata,
            'nlcs': nlcs,
            'nfreq_keplerian': nfreq,
            'baseline': baseline,
            'time_single_s': float(t_single),
            'time_batch_s': float(t_batch),
            'times_single': [float(x) for x in times_single],
            'times_batch': [float(x) for x in times_batch],
            'lc_per_sec_single': float(lc_per_sec_single),
            'lc_per_sec_batch': float(lc_per_sec_batch),
            'batch_speedup': float(speedup),
        }

    # Summary table
    print("\n  " + "-" * 70)
    print(f"  {'Survey':<15} {'ndata':>6} {'nfreq':>7} {'Single':>10} "
          f"{'Batch':>10} {'Speedup':>8} {'LC/s':>10}")
    print("  " + "-" * 70)
    for name, r in results.items():
        print(f"  {name:<15} {r['ndata']:>6} {r['nfreq_keplerian']:>7} "
              f"{r['time_single_s']:>9.3f}s {r['time_batch_s']:>9.3f}s "
              f"{r['batch_speedup']:>7.2f}x "
              f"{r['lc_per_sec_batch']:>9.0f}")

    return results


# ============================================================================
# E) cuFINUFFT LS Performance Benchmark
# ============================================================================

def bench_cufinufft_ls():
    """Benchmark cuFINUFFT vs custom NFFT vs nifty-ls vs astropy.

    IMPORTANT: GPU processes are created once and reused across iterations
    to measure steady-state compute throughput, not compilation overhead.
    Compilation (~150ms) happens once per process lifetime and is amortized
    across millions of LCs in survey-scale use.
    """
    print("\n" + "=" * 70)
    print("E) cuFINUFFT LS Performance Benchmark (single-LC, steady-state)")
    print("=" * 70)

    if not HAS_CUFINUFFT:
        print("  SKIPPED: cufinufft not installed")
        return {'skipped': True}

    results = {}
    ndata_values = [1000, 5000, 10000, 50000]
    nfreq_values = [5000, 50000]
    baseline = 365.0

    # Create GPU processes ONCE (compilation happens here)
    print("  Pre-compiling GPU kernels...", end='', flush=True)
    proc_custom = cvb_ls.LombScargleAsyncProcess(use_cufinufft=False)
    proc_cufinufft = cvb_ls.LombScargleAsyncProcess(use_cufinufft=True)

    # Trigger compilation with a small dummy run
    dummy_t, dummy_y, dummy_dy = generate_sinusoidal_lc(100, 10.0, 2.0, seed=0)
    dummy_freqs = np.linspace(0.1, 1.0, 100).astype(np.float32)
    proc_custom.run([(dummy_t, dummy_y, dummy_dy)], freqs=[dummy_freqs])
    proc_custom.finish()
    proc_cufinufft.run([(dummy_t, dummy_y, dummy_dy)], freqs=[dummy_freqs])
    proc_cufinufft.finish()
    print(" done")

    for ndata in ndata_values:
        for nfreq in nfreq_values:
            key = f"ndata_{ndata}_nfreq_{nfreq}"
            print(f"\n  ndata={ndata}, nfreq={nfreq}")

            t, y, dy = generate_sinusoidal_lc(
                ndata, baseline, 5.0, amplitude=0.01, seed=300
            )

            # NFFT-compatible frequency grid
            fmax = 2.0
            df = fmax / nfreq
            freqs = (np.arange(1, nfreq + 1) * df).astype(np.float32)

            entry = {
                'ndata': ndata,
                'nfreq': nfreq,
            }

            # Custom NFFT GPU (reuse pre-compiled process)
            def run_custom():
                proc_custom.run([(t, y, dy)], freqs=[freqs])
                proc_custom.finish()

            print(f"    Custom NFFT GPU...", end='', flush=True)
            t_custom, _ = time_function(run_custom, n_iter=5, warmup=2)
            print(f" {t_custom*1000:.1f}ms")
            entry['time_custom_gpu_ms'] = float(t_custom * 1000)

            # cuFINUFFT GPU (reuse pre-compiled process)
            def run_cufinufft_fn():
                proc_cufinufft.run([(t, y, dy)], freqs=[freqs])
                proc_cufinufft.finish()

            print(f"    cuFINUFFT GPU...", end='', flush=True)
            t_cufinufft, _ = time_function(run_cufinufft_fn, n_iter=5, warmup=2)
            print(f" {t_cufinufft*1000:.1f}ms")
            entry['time_cufinufft_gpu_ms'] = float(t_cufinufft * 1000)

            entry['cufinufft_vs_custom'] = float(t_custom / t_cufinufft) \
                if t_cufinufft > 0 else None

            # nifty-ls CPU
            if HAS_NIFTY_LS:
                def run_nifty():
                    nifty_ls.lombscargle(
                        t.astype(np.float64),
                        y.astype(np.float64),
                        dy.astype(np.float64),
                        fmin=float(freqs[0]),
                        fmax=float(freqs[-1]),
                        Nf=nfreq,
                    )

                print(f"    nifty-ls CPU...", end='', flush=True)
                t_nifty, _ = time_function_cpu(run_nifty, n_iter=5, warmup=2)
                if t_nifty is not None:
                    print(f" {t_nifty*1000:.1f}ms")
                    entry['time_nifty_cpu_ms'] = float(t_nifty * 1000)
                    entry['cufinufft_vs_nifty'] = float(t_nifty / t_cufinufft) \
                        if t_cufinufft > 0 else None
                else:
                    print(f" TIMEOUT")
                    entry['time_nifty_cpu_ms'] = None

            # astropy CPU
            if HAS_ASTROPY:
                def run_astropy():
                    ls = LombScargle(t.astype(np.float64),
                                     y.astype(np.float64),
                                     dy.astype(np.float64))
                    ls.power(freqs.astype(np.float64))

                print(f"    astropy CPU...", end='', flush=True)
                t_astropy, _ = time_function_cpu(
                    run_astropy, n_iter=3, warmup=1, timeout=60.0
                )
                if t_astropy is not None:
                    print(f" {t_astropy*1000:.1f}ms")
                    entry['time_astropy_cpu_ms'] = float(t_astropy * 1000)
                else:
                    print(f" TIMEOUT (>60s)")
                    entry['time_astropy_cpu_ms'] = None

            results[key] = entry

    # Summary table
    print("\n  " + "-" * 80)
    print(f"  {'ndata':>6} {'nfreq':>6} {'Custom':>10} {'cuFINUFFT':>10} "
          f"{'Speedup':>8} {'nifty':>10} {'astropy':>10}")
    print("  " + "-" * 80)
    for key, r in results.items():
        custom_str = f"{r['time_custom_gpu_ms']:.1f}ms"
        cufinufft_str = f"{r['time_cufinufft_gpu_ms']:.1f}ms"
        speedup_str = f"{r.get('cufinufft_vs_custom', 0):.2f}x" \
            if r.get('cufinufft_vs_custom') else "N/A"
        nifty_str = f"{r['time_nifty_cpu_ms']:.1f}ms" \
            if r.get('time_nifty_cpu_ms') else "N/A"
        astropy_str = f"{r['time_astropy_cpu_ms']:.1f}ms" \
            if r.get('time_astropy_cpu_ms') else "N/A"
        print(f"  {r['ndata']:>6} {r['nfreq']:>6} {custom_str:>10} "
              f"{cufinufft_str:>10} {speedup_str:>8} "
              f"{nifty_str:>10} {astropy_str:>10}")

    return results


# ============================================================================
# F) Keplerian Grid Impact
# ============================================================================

def bench_keplerian_grid_impact():
    """Measure BLS time savings from Keplerian vs uniform grids."""
    print("\n" + "=" * 70)
    print("F) Keplerian Grid Impact on BLS Performance")
    print("=" * 70)

    results = {}

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        qmin = profile['qmin']
        qmax = profile['qmax']
        inject_period = profile['inject_period']

        print(f"\n  {name}: ndata={ndata}, baseline={baseline}d")

        t, y, dy = generate_transit_lc(
            ndata, baseline, inject_period, depth=0.01, seed=400
        )

        # Generate both grids
        kep_freqs = keplerian_freq_grid(
            profile['period_min'], profile['period_max'], baseline
        )
        uni_freqs = uniform_freq_grid(
            profile['period_min'], profile['period_max'], baseline
        )

        print(f"    Uniform:   {len(uni_freqs):>7,} freqs")
        print(f"    Keplerian: {len(kep_freqs):>7,} freqs "
              f"({len(uni_freqs)/len(kep_freqs):.1f}x reduction)")

        # Time with uniform grid
        def run_uniform():
            cvb_bls.eebls_gpu_fast_adaptive(
                t, y, dy, uni_freqs, qmin=qmin, qmax=qmax
            )

        print(f"    Timing uniform...", end='', flush=True)
        t_uniform, _ = time_function(run_uniform, n_iter=5, warmup=2)
        print(f" {t_uniform*1000:.2f}ms")

        # Time with Keplerian grid
        def run_keplerian():
            cvb_bls.eebls_gpu_fast_adaptive(
                t, y, dy, kep_freqs, qmin=qmin, qmax=qmax
            )

        print(f"    Timing Keplerian...", end='', flush=True)
        t_keplerian, _ = time_function(run_keplerian, n_iter=5, warmup=2)
        print(f" {t_keplerian*1000:.2f}ms")

        speedup = t_uniform / t_keplerian if t_keplerian > 0 else float('inf')
        print(f"    Time speedup: {speedup:.2f}x")

        results[name] = {
            'ndata': ndata,
            'baseline': baseline,
            'nfreq_uniform': len(uni_freqs),
            'nfreq_keplerian': len(kep_freqs),
            'freq_reduction': float(len(uni_freqs) / len(kep_freqs)),
            'time_uniform_ms': float(t_uniform * 1000),
            'time_keplerian_ms': float(t_keplerian * 1000),
            'time_speedup': float(speedup),
        }

    # Summary table
    print("\n  " + "-" * 75)
    print(f"  {'Survey':<15} {'Uni freqs':>10} {'Kep freqs':>10} "
          f"{'Reduction':>10} {'T_uni':>10} {'T_kep':>10} {'Speedup':>8}")
    print("  " + "-" * 75)
    for name, r in results.items():
        print(f"  {name:<15} {r['nfreq_uniform']:>10,} "
              f"{r['nfreq_keplerian']:>10,} "
              f"{r['freq_reduction']:>9.1f}x "
              f"{r['time_uniform_ms']:>9.2f}ms "
              f"{r['time_keplerian_ms']:>9.2f}ms "
              f"{r['time_speedup']:>7.2f}x")

    return results


# ============================================================================
# G) LS Survey-Scale Throughput Benchmark
# ============================================================================

def _ls_nfreq(baseline, period_min, period_max, oversampling=5):
    """Standard LS frequency count per VanderPlas (2018).

    df = 1 / (oversampling * baseline)
    nfreq = (fmax - fmin) / df

    For irregularly sampled data there is no Nyquist frequency — the LS
    periodogram can probe arbitrarily high frequencies (VanderPlas 2018).
    period_min and period_max are science-motivated.
    """
    fmin = 1.0 / period_max
    fmax = 1.0 / period_min
    return int(np.ceil((fmax - fmin) * oversampling * baseline))


# LS searches for all variability types (binaries, RR Lyrae, delta Scuti,
# Cepheids, etc.), so the period range is much broader than BLS transit
# searches. period_min ~ 0.01d (short-period delta Scuti), period_max ~
# baseline/2 (need ~2 cycles for reliable detection).
LS_PERIOD_MIN = 0.01  # days — captures delta Scuti, short-period binaries
LS_SURVEY_CONFIGS = OrderedDict()
for _name, _prof in SURVEY_PROFILES.items():
    _baseline = _prof['baseline']
    _period_max = _baseline
    _nfreq = _ls_nfreq(_baseline, LS_PERIOD_MIN, _period_max)
    LS_SURVEY_CONFIGS[_name] = {
        'ndata': _prof['ndata'],
        'baseline': _baseline,
        'period_min': LS_PERIOD_MIN,
        'period_max': _period_max,
        'nfreq': _nfreq,
        'nlcs': _prof['nlcs_bench'] * 2,
        'batch_size': 1,  # batch_size=1 is fastest (avoids multi-stream overhead)
        'inject_period': _prof['inject_period'],
    }


def bench_ls_survey_throughput():
    """Benchmark LS throughput for processing many LCs (survey-scale).

    Uses batched_run_const_nfreq() which pre-allocates GPU memory once
    and reuses it across all lightcurves, measuring true amortized throughput.
    Compares GPU (custom NFFT) vs nifty-ls (CPU NFFT).
    """
    print("\n" + "=" * 70)
    print("G) LS Survey-Scale Throughput (batched, amortized)")
    print("=" * 70)

    results = {}

    for name, config in LS_SURVEY_CONFIGS.items():
        ndata = config['ndata']
        baseline = config['baseline']
        nfreq = config['nfreq']
        nlcs = config['nlcs']
        batch_size = config['batch_size']
        inject_period = config['inject_period']

        print(f"\n  {name}: ndata={ndata}, nfreq={nfreq}, nlcs={nlcs}, "
              f"batch_size={batch_size}, "
              f"P=[{config['period_min']},{config['period_max']}]d")

        # Generate lightcurves
        lightcurves = []
        for i in range(nlcs):
            t, y, dy = generate_sinusoidal_lc(
                ndata, baseline, inject_period,
                amplitude=0.01, noise=0.003, seed=500 + i
            )
            lightcurves.append((t, y, dy))

        # NFFT-compatible frequency grid: freqs = (k0 + i) * df
        fmin = 1.0 / config['period_max']
        fmax = 1.0 / config['period_min']
        df = (fmax - fmin) / nfreq
        k0 = max(1, int(round(fmin / df)))
        freqs = (df * (k0 + np.arange(nfreq))).astype(np.float32)

        entry = {
            'ndata': ndata,
            'nfreq': nfreq,
            'nlcs': nlcs,
            'batch_size': batch_size,
        }

        # GPU batched (custom NFFT) - uses batched_run_const_nfreq
        print(f"    GPU batched (custom NFFT)...", end='', flush=True)
        try:
            proc_gpu = cvb_ls.LombScargleAsyncProcess(use_cufinufft=False)

            def run_gpu_batched():
                proc_gpu.batched_run_const_nfreq(
                    lightcurves, batch_size=batch_size,
                    freqs=freqs, only_return_best_freqs=False
                )
                proc_gpu.finish()

            t_gpu, _ = time_function(run_gpu_batched, n_iter=3, warmup=1)
            lc_per_sec_gpu = nlcs / t_gpu
            print(f" {t_gpu:.3f}s ({lc_per_sec_gpu:.0f} LC/s, "
                  f"{t_gpu/nlcs*1000:.2f} ms/LC)")
            entry['time_gpu_batched_s'] = float(t_gpu)
            entry['lc_per_sec_gpu'] = float(lc_per_sec_gpu)
            entry['ms_per_lc_gpu'] = float(t_gpu / nlcs * 1000)
        except Exception as e:
            print(f" ERROR: {e}")
            traceback.print_exc()
            entry['time_gpu_batched_s'] = None
            entry['lc_per_sec_gpu'] = None

        # GPU batched (cuFINUFFT) - if available
        if HAS_CUFINUFFT:
            print(f"    GPU batched (cuFINUFFT)...", end='', flush=True)
            try:
                proc_cufi = cvb_ls.LombScargleAsyncProcess(use_cufinufft=True)

                def run_cufi_batched():
                    proc_cufi.batched_run_const_nfreq(
                        lightcurves, batch_size=batch_size,
                        freqs=freqs, only_return_best_freqs=False
                    )
                    proc_cufi.finish()

                t_cufi, _ = time_function(run_cufi_batched, n_iter=3, warmup=1)
                lc_per_sec_cufi = nlcs / t_cufi
                print(f" {t_cufi:.3f}s ({lc_per_sec_cufi:.0f} LC/s, "
                      f"{t_cufi/nlcs*1000:.2f} ms/LC)")
                entry['time_cufinufft_batched_s'] = float(t_cufi)
                entry['lc_per_sec_cufinufft'] = float(lc_per_sec_cufi)
                entry['ms_per_lc_cufinufft'] = float(t_cufi / nlcs * 1000)
            except Exception as e:
                print(f" ERROR: {e}")
                traceback.print_exc()
                entry['time_cufinufft_batched_s'] = None
                entry['lc_per_sec_cufinufft'] = None

        # nifty-ls CPU sequential
        if HAS_NIFTY_LS:
            print(f"    nifty-ls CPU sequential...", end='', flush=True)
            try:
                def run_nifty_seq():
                    for t, y, dy in lightcurves:
                        nifty_ls.lombscargle(
                            t.astype(np.float64),
                            y.astype(np.float64),
                            dy.astype(np.float64),
                            fmin=float(freqs[0]),
                            fmax=float(freqs[-1]),
                            Nf=nfreq,
                        )

                t_nifty, _ = time_function_cpu(
                    run_nifty_seq, n_iter=3, warmup=1, timeout=120.0
                )
                if t_nifty is not None:
                    lc_per_sec_nifty = nlcs / t_nifty
                    print(f" {t_nifty:.3f}s ({lc_per_sec_nifty:.0f} LC/s, "
                          f"{t_nifty/nlcs*1000:.2f} ms/LC)")
                    entry['time_nifty_seq_s'] = float(t_nifty)
                    entry['lc_per_sec_nifty'] = float(lc_per_sec_nifty)
                    entry['ms_per_lc_nifty'] = float(t_nifty / nlcs * 1000)
                    # GPU vs nifty-ls speedup
                    if entry.get('time_gpu_batched_s'):
                        entry['gpu_vs_nifty_speedup'] = float(
                            t_nifty / entry['time_gpu_batched_s']
                        )
                else:
                    print(f" TIMEOUT (>120s)")
                    entry['time_nifty_seq_s'] = None
            except Exception as e:
                print(f" ERROR: {e}")
                entry['time_nifty_seq_s'] = None

        results[name] = entry

    # Summary table
    print("\n  " + "-" * 85)
    print(f"  {'Survey':<15} {'ndata':>6} {'nfreq':>6} "
          f"{'GPU ms/LC':>10} {'cuFI ms/LC':>11} {'nifty ms/LC':>12} "
          f"{'GPU/nifty':>10}")
    print("  " + "-" * 85)
    for name, r in results.items():
        gpu_str = f"{r['ms_per_lc_gpu']:.2f}" if r.get('ms_per_lc_gpu') else "ERR"
        cufi_str = f"{r['ms_per_lc_cufinufft']:.2f}" \
            if r.get('ms_per_lc_cufinufft') else "N/A"
        nifty_str = f"{r['ms_per_lc_nifty']:.2f}" \
            if r.get('ms_per_lc_nifty') else "N/A"
        speedup_str = f"{r['gpu_vs_nifty_speedup']:.2f}x" \
            if r.get('gpu_vs_nifty_speedup') else "N/A"
        print(f"  {name:<15} {r['ndata']:>6} {r['nfreq']:>6} "
              f"{gpu_str:>10} {cufi_str:>11} {nifty_str:>12} "
              f"{speedup_str:>10}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test and benchmark BLS batch + cuFINUFFT LS features'
    )
    parser.add_argument('--tests-only', action='store_true',
                        help='Run only correctness tests')
    parser.add_argument('--bench-only', action='store_true',
                        help='Run only benchmarks (skip correctness)')
    parser.add_argument('--skip-cufinufft', action='store_true',
                        help='Skip cuFINUFFT-related tests and benchmarks')
    parser.add_argument('--output', type=str,
                        default='benchmarks/results/benchmark_results_new_features.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # GPU info
    dev = pycuda.autoinit.device
    gpu_name = dev.name()
    gpu_mem = dev.total_memory() // (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_mem} MB)")
    print(f"cuFINUFFT available: {HAS_CUFINUFFT}")
    print(f"nifty-ls available: {HAS_NIFTY_LS}")
    print(f"astropy available: {HAS_ASTROPY}")

    all_results = {
        'meta': {
            'gpu': gpu_name,
            'gpu_memory_mb': gpu_mem,
            'timestamp': datetime.now().isoformat(),
            'has_cufinufft': HAS_CUFINUFFT,
            'has_nifty_ls': HAS_NIFTY_LS,
            'has_astropy': HAS_ASTROPY,
        },
    }

    run_tests = not args.bench_only
    run_bench = not args.tests_only
    skip_cufinufft = args.skip_cufinufft

    tests_passed = True

    # ---- Correctness Tests ----
    if run_tests:
        try:
            ok, res = test_bls_batch_correctness()
            all_results['test_bls_batch'] = res
            if not ok:
                tests_passed = False
        except Exception as e:
            print(f"\n  ERROR in BLS batch test: {e}")
            traceback.print_exc()
            all_results['test_bls_batch'] = {'error': str(e)}
            tests_passed = False

        if not skip_cufinufft:
            try:
                ok, res = test_cufinufft_ls_correctness()
                all_results['test_cufinufft_ls'] = res
                if not ok:
                    tests_passed = False
            except Exception as e:
                print(f"\n  ERROR in cuFINUFFT LS test: {e}")
                traceback.print_exc()
                all_results['test_cufinufft_ls'] = {'error': str(e)}
                tests_passed = False

        try:
            ok, res = test_keplerian_grid()
            all_results['test_keplerian_grid'] = res
            if not ok:
                tests_passed = False
        except Exception as e:
            print(f"\n  ERROR in Keplerian grid test: {e}")
            traceback.print_exc()
            all_results['test_keplerian_grid'] = {'error': str(e)}
            tests_passed = False

    if run_tests and not tests_passed:
        print("\n" + "!" * 70)
        print("WARNING: Some correctness tests FAILED. Benchmark results "
              "may not be meaningful.")
        print("!" * 70)

    # ---- Benchmarks ----
    if run_bench:
        try:
            all_results['bench_bls_batch'] = bench_bls_batch_throughput()
        except Exception as e:
            print(f"\n  ERROR in BLS batch benchmark: {e}")
            traceback.print_exc()
            all_results['bench_bls_batch'] = {'error': str(e)}

        if not skip_cufinufft:
            try:
                all_results['bench_cufinufft_ls'] = bench_cufinufft_ls()
            except Exception as e:
                print(f"\n  ERROR in cuFINUFFT LS benchmark: {e}")
                traceback.print_exc()
                all_results['bench_cufinufft_ls'] = {'error': str(e)}

        try:
            all_results['bench_keplerian_grid'] = bench_keplerian_grid_impact()
        except Exception as e:
            print(f"\n  ERROR in Keplerian grid benchmark: {e}")
            traceback.print_exc()
            all_results['bench_keplerian_grid'] = {'error': str(e)}

        try:
            all_results['bench_ls_survey'] = bench_ls_survey_throughput()
        except Exception as e:
            print(f"\n  ERROR in LS survey throughput benchmark: {e}")
            traceback.print_exc()
            all_results['bench_ls_survey'] = {'error': str(e)}

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    if run_tests:
        print(f"\nTests: {'ALL PASSED' if tests_passed else 'SOME FAILED'}")

    return 0 if tests_passed else 1


if __name__ == '__main__':
    sys.exit(main())
