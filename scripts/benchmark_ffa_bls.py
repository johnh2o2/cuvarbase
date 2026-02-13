#!/usr/bin/env python3
"""
Benchmark FFA-BLS against standard GPU BLS across survey profiles.

Tests:
  A) FFA-BLS correctness: GPU vs CPU at multiple configurations
  B) FFA-BLS transit injection-recovery across surveys

Benchmarks:
  C) FFA-BLS vs standard BLS (Keplerian grid) across survey profiles
  D) FFA-BLS scaling: time vs N_p, N_obs, m

Usage:
    python scripts/benchmark_ffa_bls.py               # all tests + benchmarks
    python scripts/benchmark_ffa_bls.py --tests-only   # correctness only
    python scripts/benchmark_ffa_bls.py --bench-only   # benchmarks only

Output: JSON results in benchmark_results_ffa_bls.json
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
# numpy 2.x compatibility for scikit-cuda
# ---------------------------------------------------------------------------
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

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
from cuvarbase.ffa_bls import eebls_ffa_gpu, eebls_ffa_cpu
from cuvarbase.bls_frequencies import keplerian_freq_grid

HAS_ASTROPY = False
try:
    from astropy.timeseries import BoxLeastSquares
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


def time_function_cpu(func, n_iter=3, warmup=1, timeout=120.0):
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
    t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float64)
    phase = (t % period) / period
    y = np.ones(ndata, dtype=np.float64)
    in_transit = phase < duration_frac
    y[in_transit] -= depth
    y += rng.randn(ndata) * noise
    dy = np.full(ndata, noise, dtype=np.float64)
    return t, y, dy


# ---------------------------------------------------------------------------
# Survey profiles (from benchmark_new_features.py)
# ---------------------------------------------------------------------------

# FFA period ranges are narrower than standard BLS because the Phase 1
# implementation iterates octaves sequentially. Each octave spans one
# period bin (dt = P_min / m_bins), so P=[0.5, 100] with m_bins=100
# means 19,900 octaves -- too many for a sequential Python loop.
# Phase 2 (octave batching) will fix this. For now, benchmark with
# realistic narrow ranges that show FFA's algorithmic advantage.
SURVEY_PROFILES = OrderedDict([
    ('ZTF-like', {
        'ndata': 150,
        'baseline': 730.0,
        'period_min': 1.0,
        'period_max': 10.0,
        'nlcs_bench': 50,
        'qmin': 0.01,
        'qmax': 0.15,
        'inject_period': 3.0,
    }),
    ('HAT-Net', {
        'ndata': 6000,
        'baseline': 3650.0,
        'period_min': 1.0,
        'period_max': 10.0,
        'nlcs_bench': 20,
        'qmin': 0.01,
        'qmax': 0.1,
        'inject_period': 2.5,
    }),
    ('TESS-1sector', {
        'ndata': 20000,
        'baseline': 27.0,
        'period_min': 1.0,
        'period_max': 13.5,
        'nlcs_bench': 10,
        'qmin': 0.005,
        'qmax': 0.1,
        'inject_period': 5.0,
    }),
    ('Kepler', {
        'ndata': 65000,
        'baseline': 1460.0,
        'period_min': 1.0,
        'period_max': 20.0,
        'nlcs_bench': 3,
        'qmin': 0.005,
        'qmax': 0.1,
        'inject_period': 10.0,
    }),
])


# ============================================================================
# A) FFA-BLS GPU vs CPU Correctness
# ============================================================================

def test_ffa_gpu_vs_cpu():
    """Compare GPU and CPU FFA-BLS implementations."""
    print("\n" + "=" * 70)
    print("A) FFA-BLS GPU vs CPU Correctness")
    print("=" * 70)

    results = {}
    all_pass = True

    test_configs = [
        (300, 100.0, 0.5, 5.0, 50),
        (500, 365.0, 0.5, 10.0, 100),
        (1000, 730.0, 1.0, 20.0, 100),
    ]

    for ndata, baseline, pmin, pmax, m_bins in test_configs:
        key = f"ndata_{ndata}_P_{pmin}_{pmax}"
        print(f"\n  ndata={ndata}, baseline={baseline}d, "
              f"P=[{pmin},{pmax}]d, m_bins={m_bins}")

        t, y, dy = generate_transit_lc(
            ndata, baseline, 2.5, depth=0.01, noise=0.003, seed=42)

        kwargs = dict(period_min=pmin, period_max=pmax,
                      m_bins=m_bins, qmin=0.01, qmax=0.15, dlogq=0.2)

        periods_cpu, power_cpu = eebls_ffa_cpu(t, y, dy, **kwargs)
        periods_gpu, power_gpu = eebls_ffa_gpu(t, y, dy, **kwargs)

        if len(periods_cpu) == 0 or len(periods_gpu) == 0:
            print(f"    SKIP: no periods (cpu={len(periods_cpu)}, gpu={len(periods_gpu)})")
            continue

        # Check period grids match
        period_match = np.allclose(periods_cpu, periods_gpu, rtol=1e-5)

        # Check power correlation
        corr = np.corrcoef(power_cpu, power_gpu)[0, 1]

        # Check peak match
        peak_cpu = periods_cpu[np.argmax(power_cpu)]
        peak_gpu = periods_gpu[np.argmax(power_gpu)]
        dp = abs(peak_cpu - peak_gpu)
        # Allow peak match within 1 period step
        if len(periods_cpu) > 1:
            period_step = np.median(np.diff(periods_cpu))
            peak_match = dp < 2 * period_step
        else:
            peak_match = dp < 0.01

        # Max absolute difference
        max_adiff = float(np.max(np.abs(power_cpu - power_gpu)))

        config_pass = period_match and corr > 0.999 and peak_match
        if not config_pass:
            all_pass = False

        status = "PASS" if config_pass else "FAIL"
        print(f"    {status}: corr={corr:.6f}, max_adiff={max_adiff:.2e}, "
              f"peak_match={peak_match}, period_match={period_match}")
        print(f"    n_periods={len(periods_cpu)}, "
              f"peak_cpu={peak_cpu:.4f}d, peak_gpu={peak_gpu:.4f}d")

        results[key] = {
            'ndata': ndata,
            'n_periods': len(periods_cpu),
            'correlation': float(corr),
            'max_abs_diff': max_adiff,
            'peak_match': peak_match,
            'period_grid_match': period_match,
            'pass': config_pass,
        }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass, results


# ============================================================================
# B) Transit Injection-Recovery
# ============================================================================

def test_ffa_transit_recovery():
    """Verify FFA-BLS recovers injected transits across survey profiles."""
    print("\n" + "=" * 70)
    print("B) FFA-BLS Transit Injection-Recovery")
    print("=" * 70)

    results = {}
    all_pass = True

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        pmin = profile['period_min']
        pmax = profile['period_max']
        inject_period = profile['inject_period']
        qmin = profile['qmin']
        qmax = profile['qmax']

        print(f"\n  {name}: ndata={ndata}, baseline={baseline}d, "
              f"inject_P={inject_period}d")

        n_trials = 5
        n_recovered = 0

        for i in range(n_trials):
            t, y, dy = generate_transit_lc(
                ndata, baseline, inject_period,
                depth=0.01, noise=0.003, seed=42 + i)

            periods, power = eebls_ffa_gpu(
                t, y, dy, pmin, pmax,
                qmin=qmin, qmax=qmax, dlogq=0.2)

            if len(periods) == 0:
                continue

            best_period = periods[np.argmax(power)]
            rel_err = abs(best_period - inject_period) / inject_period
            # Accept also period aliases (half, double)
            alias_ok = any(
                abs(best_period - inject_period * f) / (inject_period * f) < 0.05
                for f in [0.5, 1.0, 2.0]
            )
            if alias_ok:
                n_recovered += 1

        config_pass = n_recovered >= 3  # at least 3/5 recovered
        if not config_pass:
            all_pass = False

        status = "PASS" if config_pass else "FAIL"
        print(f"    {status}: recovered {n_recovered}/{n_trials} "
              f"(best P={best_period:.4f}d vs true {inject_period:.1f}d)")

        results[name] = {
            'ndata': ndata,
            'inject_period': inject_period,
            'n_recovered': n_recovered,
            'n_trials': n_trials,
            'pass': config_pass,
        }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass, results


# ============================================================================
# C0) Phase 1 vs Phase 2 (Sequential vs Batch) Comparison
# ============================================================================

def bench_phase1_vs_phase2():
    """Compare Phase 1 (sequential) vs Phase 2 (batched) FFA-BLS."""
    print("\n" + "=" * 70)
    print("C0) FFA-BLS Phase 1 (Sequential) vs Phase 2 (Batched)")
    print("=" * 70)

    results = {}

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        pmin = profile['period_min']
        pmax = profile['period_max']
        qmin = profile['qmin']
        qmax = profile['qmax']
        inject_period = profile['inject_period']

        print(f"\n  {name}: ndata={ndata}, baseline={baseline}d, "
              f"P=[{pmin},{pmax}]d")

        t, y, dy = generate_transit_lc(
            ndata, baseline, inject_period, depth=0.01, seed=42)

        ffa_kwargs = dict(period_min=pmin, period_max=pmax,
                          qmin=qmin, qmax=qmax, dlogq=0.2)

        # Phase 1 (sequential)
        def run_phase1():
            return eebls_ffa_gpu(t, y, dy, use_batch=False, **ffa_kwargs)

        print(f"    Phase 1 (sequential)...", end='', flush=True)
        t_p1, times_p1 = time_function(run_phase1, n_iter=5, warmup=2)
        periods_p1, power_p1 = run_phase1()
        peak_p1 = periods_p1[np.argmax(power_p1)] if len(power_p1) > 0 else 0
        print(f" {t_p1*1000:.2f}ms")

        # Phase 2 (batched)
        def run_phase2():
            return eebls_ffa_gpu(t, y, dy, use_batch=True, **ffa_kwargs)

        print(f"    Phase 2 (batched)...", end='', flush=True)
        t_p2, times_p2 = time_function(run_phase2, n_iter=5, warmup=2)
        periods_p2, power_p2 = run_phase2()
        peak_p2 = periods_p2[np.argmax(power_p2)] if len(power_p2) > 0 else 0
        print(f" {t_p2*1000:.2f}ms")

        speedup = t_p1 / t_p2 if t_p2 > 0 else float('inf')
        print(f"    Phase 2 / Phase 1 speedup: {speedup:.1f}x")

        # Correctness: verify both paths agree
        if len(power_p1) > 0 and len(power_p2) > 0:
            corr = np.corrcoef(power_p1, power_p2)[0, 1]
            print(f"    Correlation: {corr:.6f}")
        else:
            corr = None

        results[name] = {
            'ndata': ndata,
            'baseline': baseline,
            'period_range': [pmin, pmax],
            'n_periods': len(periods_p1),
            'time_phase1_ms': float(t_p1 * 1000),
            'time_phase2_ms': float(t_p2 * 1000),
            'times_phase1': [float(x) for x in times_p1],
            'times_phase2': [float(x) for x in times_p2],
            'phase2_speedup': float(speedup),
            'correlation': float(corr) if corr is not None else None,
            'peak_phase1_d': float(peak_p1),
            'peak_phase2_d': float(peak_p2),
        }

    # Summary table
    print("\n  " + "-" * 75)
    print(f"  {'Survey':<15} {'ndata':>6} {'n_periods':>10} "
          f"{'Phase1':>10} {'Phase2':>10} {'Speedup':>8} {'Corr':>8}")
    print("  " + "-" * 75)
    for name, r in results.items():
        corr_str = f"{r['correlation']:.4f}" if r['correlation'] else "N/A"
        print(f"  {name:<15} {r['ndata']:>6} {r['n_periods']:>10,} "
              f"{r['time_phase1_ms']:>9.2f}ms "
              f"{r['time_phase2_ms']:>9.2f}ms "
              f"{r['phase2_speedup']:>7.1f}x "
              f"{corr_str:>8}")

    return results


# ============================================================================
# C) FFA-BLS vs Standard BLS Benchmark
# ============================================================================

def bench_ffa_vs_standard_bls():
    """Benchmark FFA-BLS vs standard GPU BLS (Keplerian grid)."""
    print("\n" + "=" * 70)
    print("C) FFA-BLS vs Standard BLS (Keplerian Grid) Benchmark")
    print("=" * 70)

    results = {}

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        pmin = profile['period_min']
        pmax = profile['period_max']
        qmin = profile['qmin']
        qmax = profile['qmax']
        inject_period = profile['inject_period']

        print(f"\n  {name}: ndata={ndata}, baseline={baseline}d, "
              f"P=[{pmin},{pmax}]d")

        t, y, dy = generate_transit_lc(
            ndata, baseline, inject_period, depth=0.01, seed=42)

        # Standard BLS with Keplerian grid
        kep_freqs = keplerian_freq_grid(pmin, pmax, baseline)
        nfreq_kep = len(kep_freqs)
        print(f"    Standard BLS: {nfreq_kep:,} Keplerian freqs")

        def run_standard():
            return cvb_bls.eebls_gpu_fast_adaptive(
                t, y, dy, kep_freqs, qmin=qmin, qmax=qmax)

        print(f"    Timing standard BLS...", end='', flush=True)
        t_standard, times_standard = time_function(run_standard, n_iter=5, warmup=2)
        power_standard = run_standard()
        peak_std = 1.0 / kep_freqs[np.argmax(power_standard)]
        print(f" {t_standard*1000:.2f}ms (peak P={peak_std:.4f}d)")

        # FFA-BLS
        # First run to see how many periods it generates
        periods_ffa, power_ffa = eebls_ffa_gpu(
            t, y, dy, pmin, pmax, qmin=qmin, qmax=qmax, dlogq=0.2)
        nperiods_ffa = len(periods_ffa)
        peak_ffa = periods_ffa[np.argmax(power_ffa)] if len(power_ffa) > 0 else 0
        print(f"    FFA-BLS:      {nperiods_ffa:,} periods (native grid)")

        def run_ffa():
            return eebls_ffa_gpu(
                t, y, dy, pmin, pmax, qmin=qmin, qmax=qmax, dlogq=0.2)

        print(f"    Timing FFA-BLS...", end='', flush=True)
        t_ffa, times_ffa = time_function(run_ffa, n_iter=5, warmup=2)
        print(f" {t_ffa*1000:.2f}ms (peak P={peak_ffa:.4f}d)")

        speedup = t_standard / t_ffa if t_ffa > 0 else float('inf')
        print(f"    FFA/Standard speedup: {speedup:.2f}x")

        # Correctness check: both should find similar peak period
        if len(power_ffa) > 0 and np.max(power_standard) > 0.001:
            period_agree = any(
                abs(peak_ffa - inject_period * f) / (inject_period * f) < 0.1
                for f in [0.5, 1.0, 2.0]
            )
            print(f"    Peak agreement: {'YES' if period_agree else 'NO'} "
                  f"(std={peak_std:.4f}d, ffa={peak_ffa:.4f}d, true={inject_period}d)")

        # Astropy CPU comparison
        t_astropy = None
        if HAS_ASTROPY:
            print(f"    Timing astropy BLS...", end='', flush=True)

            def run_astropy():
                bls = BoxLeastSquares(t, y, dy)
                periods = np.linspace(pmin, pmax, nfreq_kep)
                bls.power(periods, 0.02, oversample=5)

            t_astropy, _ = time_function_cpu(run_astropy, n_iter=3, warmup=1, timeout=60.0)
            if t_astropy is not None:
                print(f" {t_astropy*1000:.1f}ms")
            else:
                print(f" TIMEOUT")

        results[name] = {
            'ndata': ndata,
            'baseline': baseline,
            'period_range': [pmin, pmax],
            'nfreq_keplerian': nfreq_kep,
            'nperiods_ffa': nperiods_ffa,
            'time_standard_ms': float(t_standard * 1000),
            'time_ffa_ms': float(t_ffa * 1000),
            'times_standard': [float(x) for x in times_standard],
            'times_ffa': [float(x) for x in times_ffa],
            'ffa_speedup': float(speedup),
            'peak_standard_d': float(peak_std),
            'peak_ffa_d': float(peak_ffa),
            'inject_period_d': inject_period,
        }
        if t_astropy is not None:
            results[name]['time_astropy_ms'] = float(t_astropy * 1000)
            results[name]['ffa_vs_astropy_speedup'] = float(t_astropy / t_ffa)

    # Summary table
    print("\n  " + "-" * 90)
    print(f"  {'Survey':<15} {'ndata':>6} {'Kep freqs':>10} {'FFA pers':>10} "
          f"{'Std BLS':>10} {'FFA-BLS':>10} {'Speedup':>8} {'astropy':>10}")
    print("  " + "-" * 90)
    for name, r in results.items():
        astropy_str = f"{r['time_astropy_ms']:.1f}ms" \
            if r.get('time_astropy_ms') else "N/A"
        print(f"  {name:<15} {r['ndata']:>6} {r['nfreq_keplerian']:>10,} "
              f"{r['nperiods_ffa']:>10,} "
              f"{r['time_standard_ms']:>9.2f}ms "
              f"{r['time_ffa_ms']:>9.2f}ms "
              f"{r['ffa_speedup']:>7.2f}x "
              f"{astropy_str:>10}")

    return results


# ============================================================================
# D) FFA-BLS Throughput (Survey-Scale)
# ============================================================================

def bench_ffa_throughput():
    """Benchmark FFA-BLS throughput for processing many lightcurves."""
    print("\n" + "=" * 70)
    print("D) FFA-BLS Survey-Scale Throughput")
    print("=" * 70)

    results = {}

    for name, profile in SURVEY_PROFILES.items():
        ndata = profile['ndata']
        baseline = profile['baseline']
        pmin = profile['period_min']
        pmax = profile['period_max']
        qmin = profile['qmin']
        qmax = profile['qmax']
        nlcs = profile['nlcs_bench']
        inject_period = profile['inject_period']

        print(f"\n  {name}: ndata={ndata}, nlcs={nlcs}, "
              f"P=[{pmin},{pmax}]d")

        # Generate lightcurves
        lightcurves = []
        for i in range(nlcs):
            t, y, dy = generate_transit_lc(
                ndata, baseline, inject_period,
                depth=0.01, noise=0.003, seed=200 + i)
            lightcurves.append((t, y, dy))

        # FFA-BLS: process all LCs
        def run_ffa_all():
            for t, y, dy in lightcurves:
                eebls_ffa_gpu(t, y, dy, pmin, pmax,
                              qmin=qmin, qmax=qmax, dlogq=0.2)

        print(f"    Timing FFA-BLS ({nlcs} LCs)...", end='', flush=True)
        t_ffa, times_ffa = time_function(run_ffa_all, n_iter=3, warmup=1)
        lc_per_sec_ffa = nlcs / t_ffa
        ms_per_lc_ffa = t_ffa / nlcs * 1000
        print(f" {t_ffa:.3f}s ({lc_per_sec_ffa:.0f} LC/s, "
              f"{ms_per_lc_ffa:.2f} ms/LC)")

        # Standard BLS with Keplerian grid
        kep_freqs = keplerian_freq_grid(pmin, pmax, baseline)

        def run_std_all():
            for t, y, dy in lightcurves:
                cvb_bls.eebls_gpu_fast_adaptive(
                    t, y, dy, kep_freqs, qmin=qmin, qmax=qmax)

        print(f"    Timing standard BLS ({nlcs} LCs)...", end='', flush=True)
        t_std, times_std = time_function(run_std_all, n_iter=3, warmup=1)
        lc_per_sec_std = nlcs / t_std
        ms_per_lc_std = t_std / nlcs * 1000
        print(f" {t_std:.3f}s ({lc_per_sec_std:.0f} LC/s, "
              f"{ms_per_lc_std:.2f} ms/LC)")

        speedup = t_std / t_ffa if t_ffa > 0 else float('inf')
        print(f"    FFA/Standard speedup: {speedup:.2f}x")

        results[name] = {
            'ndata': ndata,
            'nlcs': nlcs,
            'nfreq_keplerian': len(kep_freqs),
            'time_ffa_s': float(t_ffa),
            'time_standard_s': float(t_std),
            'ms_per_lc_ffa': float(ms_per_lc_ffa),
            'ms_per_lc_standard': float(ms_per_lc_std),
            'lc_per_sec_ffa': float(lc_per_sec_ffa),
            'lc_per_sec_standard': float(lc_per_sec_std),
            'ffa_speedup': float(speedup),
        }

    # Summary table
    print("\n  " + "-" * 80)
    print(f"  {'Survey':<15} {'ndata':>6} {'nlcs':>5} "
          f"{'Std ms/LC':>10} {'FFA ms/LC':>10} {'Speedup':>8} "
          f"{'FFA LC/s':>10}")
    print("  " + "-" * 80)
    for name, r in results.items():
        print(f"  {name:<15} {r['ndata']:>6} {r['nlcs']:>5} "
              f"{r['ms_per_lc_standard']:>9.2f}ms "
              f"{r['ms_per_lc_ffa']:>9.2f}ms "
              f"{r['ffa_speedup']:>7.2f}x "
              f"{r['lc_per_sec_ffa']:>9.0f}")

    return results


# ============================================================================
# E) FFA-BLS Scaling
# ============================================================================

def bench_ffa_scaling():
    """Measure FFA-BLS time scaling with N_obs, period range, m_bins."""
    print("\n" + "=" * 70)
    print("E) FFA-BLS Scaling Behavior")
    print("=" * 70)

    results = {}

    # E1: Scale with N_obs (fixed narrow period range)
    print("\n  E1: Scaling with N_obs (P=[1.0, 10.0]d, baseline=365d)")
    ndata_values = [100, 500, 1000, 5000, 10000, 50000]
    scaling_nobs = {}
    for ndata in ndata_values:
        t, y, dy = generate_transit_lc(ndata, 365.0, 3.0, seed=42)
        def run():
            eebls_ffa_gpu(t, y, dy, 1.0, 10.0, qmin=0.01, qmax=0.15)
        t_ms, _ = time_function(run, n_iter=5, warmup=2)
        t_ms *= 1000
        print(f"    N_obs={ndata:>6}: {t_ms:.2f}ms")
        scaling_nobs[ndata] = float(t_ms)

    results['scaling_nobs'] = scaling_nobs

    # E2: Scale with period range (fixed ndata)
    print("\n  E2: Scaling with period range (ndata=5000, baseline=365d)")
    pmax_values = [2.0, 5.0, 10.0, 20.0]
    scaling_prange = {}
    t, y, dy = generate_transit_lc(5000, 365.0, 3.0, seed=42)
    for pmax in pmax_values:
        def run():
            eebls_ffa_gpu(t, y, dy, 1.0, pmax, qmin=0.01, qmax=0.15)
        t_ms, _ = time_function(run, n_iter=5, warmup=2)
        t_ms *= 1000
        periods, _ = eebls_ffa_gpu(t, y, dy, 1.0, pmax, qmin=0.01, qmax=0.15)
        print(f"    P_max={pmax:>6.1f}d: {t_ms:.2f}ms ({len(periods):,} periods)")
        scaling_prange[pmax] = {'time_ms': float(t_ms), 'n_periods': len(periods)}

    results['scaling_prange'] = scaling_prange

    # E3: Scale with m_bins (fixed ndata, period range)
    print("\n  E3: Scaling with m_bins (ndata=5000, P=[1.0, 10.0]d, baseline=365d)")
    mbins_values = [25, 50, 100, 200]
    scaling_mbins = {}
    t, y, dy = generate_transit_lc(5000, 365.0, 3.0, seed=42)
    for m_bins in mbins_values:
        def run():
            eebls_ffa_gpu(t, y, dy, 1.0, 10.0, m_bins=m_bins,
                          qmin=0.01, qmax=0.15)
        t_ms, _ = time_function(run, n_iter=5, warmup=2)
        t_ms *= 1000
        periods, _ = eebls_ffa_gpu(t, y, dy, 1.0, 10.0, m_bins=m_bins,
                                    qmin=0.01, qmax=0.15)
        print(f"    m_bins={m_bins:>4}: {t_ms:.2f}ms ({len(periods):,} periods)")
        scaling_mbins[m_bins] = {'time_ms': float(t_ms), 'n_periods': len(periods)}

    results['scaling_mbins'] = scaling_mbins

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FFA-BLS against standard GPU BLS')
    parser.add_argument('--tests-only', action='store_true',
                        help='Run only correctness tests')
    parser.add_argument('--bench-only', action='store_true',
                        help='Run only benchmarks (skip correctness)')
    parser.add_argument('--output', type=str,
                        default='benchmark_results_ffa_bls.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # GPU info
    dev = pycuda.autoinit.device
    gpu_name = dev.name()
    gpu_mem = dev.total_memory() // (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_mem} MB)")
    print(f"astropy available: {HAS_ASTROPY}")

    all_results = {
        'meta': {
            'gpu': gpu_name,
            'gpu_memory_mb': gpu_mem,
            'timestamp': datetime.now().isoformat(),
            'has_astropy': HAS_ASTROPY,
        },
    }

    run_tests = not args.bench_only
    run_bench = not args.tests_only
    tests_passed = True

    # ---- Correctness Tests ----
    if run_tests:
        try:
            ok, res = test_ffa_gpu_vs_cpu()
            all_results['test_gpu_vs_cpu'] = res
            if not ok:
                tests_passed = False
        except Exception as e:
            print(f"\n  ERROR in GPU vs CPU test: {e}")
            traceback.print_exc()
            all_results['test_gpu_vs_cpu'] = {'error': str(e)}
            tests_passed = False

        try:
            ok, res = test_ffa_transit_recovery()
            all_results['test_transit_recovery'] = res
            if not ok:
                tests_passed = False
        except Exception as e:
            print(f"\n  ERROR in transit recovery test: {e}")
            traceback.print_exc()
            all_results['test_transit_recovery'] = {'error': str(e)}
            tests_passed = False

    if run_tests and not tests_passed:
        print("\n" + "!" * 70)
        print("WARNING: Some correctness tests FAILED.")
        print("!" * 70)

    # ---- Benchmarks ----
    if run_bench:
        try:
            all_results['bench_phase1_vs_phase2'] = bench_phase1_vs_phase2()
        except Exception as e:
            print(f"\n  ERROR in Phase 1 vs Phase 2 benchmark: {e}")
            traceback.print_exc()
            all_results['bench_phase1_vs_phase2'] = {'error': str(e)}

        try:
            all_results['bench_ffa_vs_standard'] = bench_ffa_vs_standard_bls()
        except Exception as e:
            print(f"\n  ERROR in FFA vs standard benchmark: {e}")
            traceback.print_exc()
            all_results['bench_ffa_vs_standard'] = {'error': str(e)}

        try:
            all_results['bench_ffa_throughput'] = bench_ffa_throughput()
        except Exception as e:
            print(f"\n  ERROR in FFA throughput benchmark: {e}")
            traceback.print_exc()
            all_results['bench_ffa_throughput'] = {'error': str(e)}

        try:
            all_results['bench_ffa_scaling'] = bench_ffa_scaling()
        except Exception as e:
            print(f"\n  ERROR in FFA scaling benchmark: {e}")
            traceback.print_exc()
            all_results['bench_ffa_scaling'] = {'error': str(e)}

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
