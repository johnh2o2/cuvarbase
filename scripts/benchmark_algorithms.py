#!/usr/bin/env python3
"""
Comprehensive benchmark suite for cuvarbase algorithms.

Measures GPU vs CPU performance for all cuvarbase algorithms using CUDA event
timing (GPU) and perf_counter (CPU). Computes cost-per-lightcurve estimates
based on RunPod on-demand pricing.

Usage:
    # Run all benchmarks at default parameters (10k obs, 10yr baseline)
    python scripts/benchmark_algorithms.py

    # Specific algorithms
    python scripts/benchmark_algorithms.py --algorithms bls_standard bls_sparse ls

    # Custom parameters
    python scripts/benchmark_algorithms.py --ndata 10000 --baseline 3652.5

    # Tag with GPU model for cost calculations
    python scripts/benchmark_algorithms.py --gpu-model H100

See docs/BENCHMARKING.md for full instructions.
"""

import numpy as np
import time
import json
import sys
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# GPU imports (deferred so CPU baselines can run without pycuda)
# ---------------------------------------------------------------------------
HAS_GPU = False
HAS_CUDA_EVENTS = False
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_GPU = True
    HAS_CUDA_EVENTS = True
except ImportError:
    pass

try:
    import cuvarbase.bls as cvb_bls
    import cuvarbase.lombscargle as cvb_ls
    import cuvarbase.pdm as cvb_pdm
    import cuvarbase.ce as cvb_ce
    import cuvarbase.tls as cvb_tls
    HAS_CUVARBASE = True
except ImportError as e:
    HAS_CUVARBASE = False
    print(f"Warning: Could not import cuvarbase: {e}")

try:
    from cuvarbase.bls_frequencies import keplerian_freq_grid
    HAS_BLS_FREQ = True
except ImportError:
    HAS_BLS_FREQ = False

# ---------------------------------------------------------------------------
# CPU baseline imports
# ---------------------------------------------------------------------------
HAS_ASTROPY = False
try:
    from astropy.timeseries import BoxLeastSquares, LombScargle
    HAS_ASTROPY = True
except ImportError:
    pass

HAS_NIFTY_LS = False
try:
    import nifty_ls
    HAS_NIFTY_LS = True
except ImportError:
    pass

HAS_TLS_CPU = False
try:
    from transitleastsquares import transitleastsquares
    HAS_TLS_CPU = True
except ImportError:
    pass

HAS_PYASTRONOMY = False
try:
    from PyAstronomy.pyTiming import pyPDM
    HAS_PYASTRONOMY = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# RunPod on-demand pricing ($/hr, community cloud, as of 2025-Q4)
# ---------------------------------------------------------------------------
RUNPOD_PRICING = OrderedDict([
    ('RTX_4000_Ada',  {'price_hr': 0.20, 'vram_gb': 20,  'arch': 'Ada Lovelace', 'year': 2023}),
    ('RTX_4090',      {'price_hr': 0.34, 'vram_gb': 24,  'arch': 'Ada Lovelace', 'year': 2022}),
    ('V100',          {'price_hr': 0.19, 'vram_gb': 16,  'arch': 'Volta',        'year': 2017}),
    ('L40',           {'price_hr': 0.69, 'vram_gb': 48,  'arch': 'Ada Lovelace', 'year': 2023}),
    ('A100_PCIe',     {'price_hr': 0.79, 'vram_gb': 80,  'arch': 'Ampere',       'year': 2020}),
    ('A100_SXM',      {'price_hr': 1.19, 'vram_gb': 80,  'arch': 'Ampere',       'year': 2020}),
    ('H100_PCIe',     {'price_hr': 1.99, 'vram_gb': 80,  'arch': 'Hopper',       'year': 2022}),
    ('H100_SXM',      {'price_hr': 2.69, 'vram_gb': 80,  'arch': 'Hopper',       'year': 2022}),
    ('H200_SXM',      {'price_hr': 3.59, 'vram_gb': 141, 'arch': 'Hopper',       'year': 2024}),
])


# ---------------------------------------------------------------------------
# Algorithm complexity (for extrapolation when CPU would be too slow)
# ---------------------------------------------------------------------------
ALGORITHM_COMPLEXITY = {
    # Standard (binned) BLS: O(N * Nfreq)
    'bls_standard': {'ndata': 1, 'nfreq': 1},
    # Sparse BLS: O(N^2 * Nfreq)
    'bls_sparse':   {'ndata': 2, 'nfreq': 1},
    # Lomb-Scargle: O(N * Nfreq) [direct] or O(N + Nfreq*log(Nfreq)) [NFFT]
    'ls':           {'ndata': 1, 'nfreq': 1},
    # PDM: O(N * Nfreq)
    'pdm':          {'ndata': 1, 'nfreq': 1},
    # Conditional Entropy: O(N * Nfreq)
    'ce':           {'ndata': 1, 'nfreq': 1},
    # TLS: O(N * Nperiod * Nduration)
    'tls':          {'ndata': 1, 'nfreq': 1},
}


# ============================================================================
# Timing utilities
# ============================================================================

class Timer:
    """Context manager for timing with optional CUDA events."""

    def __init__(self, use_cuda_events=False):
        self.use_cuda_events = use_cuda_events and HAS_CUDA_EVENTS
        self.elapsed = None

    def __enter__(self):
        if self.use_cuda_events:
            self.start_event = cuda.Event()
            self.end_event = cuda.Event()
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda_events:
            self.end_event.record()
            self.end_event.synchronize()
            self.elapsed = self.start_event.time_till(self.end_event) / 1000.0
        else:
            self.elapsed = time.perf_counter() - self.start_time


def time_function(func, n_iter=3, warmup=1, use_cuda=False):
    """
    Time a function over multiple iterations, returning median time.

    Parameters
    ----------
    func : callable
        Zero-argument callable to time.
    n_iter : int
        Number of timed iterations.
    warmup : int
        Number of warmup iterations (not timed).
    use_cuda : bool
        Use CUDA event timing.

    Returns
    -------
    median_time : float
        Median elapsed time in seconds.
    all_times : list of float
        All individual timings.
    """
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_iter):
        with Timer(use_cuda_events=use_cuda) as t:
            func()
        times.append(t.elapsed)

    return np.median(times), times


# ============================================================================
# Data generation
# ============================================================================

def generate_lightcurve(ndata, baseline=3652.5, seed=None):
    """
    Generate a synthetic lightcurve.

    Parameters
    ----------
    ndata : int
        Number of observations.
    baseline : float
        Observation baseline in days (default: 10 years).
    seed : int, optional
        Random seed.

    Returns
    -------
    t, y, dy : ndarray (float32)
    """
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float32)

    # Inject a transit-like signal at P=5 days, depth=0.01, duration=0.1 days
    phase = (t % 5.0) / 5.0
    y = np.ones(ndata, dtype=np.float32)
    in_transit = (phase < 0.02) | (phase > 0.98)
    y[in_transit] -= 0.01
    y += rng.randn(ndata).astype(np.float32) * 0.002

    dy = np.full(ndata, 0.002, dtype=np.float32)
    return t, y, dy


def generate_batch(ndata, nbatch, baseline=3652.5, seed=42):
    """Generate a batch of lightcurves."""
    return [generate_lightcurve(ndata, baseline, seed=seed + i)
            for i in range(nbatch)]


# ============================================================================
# Frequency / period grids
# ============================================================================

def make_freq_grid(nfreq, fmin=None, fmax=2.0):
    """
    Linearly-spaced frequency grid compatible with NFFT-based algorithms.

    Constructs freqs = k * df for k = 1, 2, ..., nfreq where df = fmax/nfreq.
    This ensures fmin/df is an integer (required by cuvarbase LS and nifty-ls).

    If fmin is specified, constructs freqs = linspace(fmin, fmax, nfreq) instead
    (may not be NFFT-compatible).
    """
    if fmin is not None:
        return np.linspace(fmin, fmax, nfreq).astype(np.float32)
    df = fmax / nfreq
    return (np.arange(1, nfreq + 1) * df).astype(np.float32)


def make_period_grid(nperiods, pmin=0.5, pmax=50.0):
    """Period grid for BLS/TLS benchmarks."""
    return np.linspace(pmin, pmax, nperiods).astype(np.float64)


# ============================================================================
# Individual benchmark functions
#
# Each returns (median_time_seconds, metadata_dict).
# ============================================================================

# --- BLS: Standard (binned) GPU -------------------------------------------

def bench_bls_standard_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase eebls_gpu_fast_adaptive (best standard BLS)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        for t, y, dy in batch:
            cvb_bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'eebls_gpu_fast_adaptive', 'times': times}


def bench_bls_standard_gpu_old(ndata, nbatch, nfreq, baseline):
    """cuvarbase eebls_gpu_fast (pre-optimization baseline)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        for t, y, dy in batch:
            cvb_bls.eebls_gpu_fast(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'eebls_gpu_fast (v0.4 baseline)', 'times': times}


def bench_bls_standard_cpu(ndata, nbatch, nfreq, baseline):
    """astropy BoxLeastSquares (CPU baseline)."""
    if not HAS_ASTROPY:
        return None, {'error': 'astropy not installed'}

    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)
    periods = 1.0 / freqs[::-1].astype(np.float64)
    durations = np.array([0.01, 0.02, 0.05, 0.1, 0.2])  # days

    def run():
        for t, y, dy in batch:
            model = BoxLeastSquares(t.astype(np.float64), y.astype(np.float64),
                                   dy=dy.astype(np.float64))
            model.power(periods, durations)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'astropy BoxLeastSquares', 'times': times}


# --- BLS: Sparse ----------------------------------------------------------

def bench_bls_sparse_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase sparse_bls_gpu."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq, fmin=0.01, fmax=0.5)

    def run():
        for t, y, dy in batch:
            cvb_bls.sparse_bls_gpu(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'sparse_bls_gpu', 'times': times}


def bench_bls_sparse_cpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase sparse_bls_cpu."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq, fmin=0.01, fmax=0.5)

    def run():
        for t, y, dy in batch:
            cvb_bls.sparse_bls_cpu(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'sparse_bls_cpu', 'times': times}


# --- Lomb-Scargle ---------------------------------------------------------

def bench_ls_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase LombScargleAsyncProcess (GPU, NFFT)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)
    # LombScargleAsyncProcess.run() expects freqs as a list of arrays (one per LC)
    freq_list = [freqs] * len(batch)

    def run():
        proc = cvb_ls.LombScargleAsyncProcess()
        results = proc.run([(t, y, dy) for t, y, dy in batch], freqs=freq_list)
        proc.finish()

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'cuvarbase LombScargleAsyncProcess', 'times': times}


def bench_ls_cpu_astropy(ndata, nbatch, nfreq, baseline):
    """astropy LombScargle (CPU baseline)."""
    if not HAS_ASTROPY:
        return None, {'error': 'astropy not installed'}

    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq).astype(np.float64)

    def run():
        for t, y, dy in batch:
            ls = LombScargle(t.astype(np.float64), y.astype(np.float64),
                             dy=dy.astype(np.float64))
            ls.power(freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'astropy LombScargle', 'times': times}


def bench_ls_cpu_nifty(ndata, nbatch, nfreq, baseline):
    """nifty-ls (CPU NUFFT, Flatiron)."""
    if not HAS_NIFTY_LS:
        return None, {'error': 'nifty-ls not installed'}

    batch = generate_batch(ndata, nbatch, baseline)
    # Build grid directly in float64 to preserve exact regularity
    df64 = 2.0 / nfreq
    freqs = df64 * np.arange(1, nfreq + 1)  # float64

    def run():
        for t, y, dy in batch:
            ls = LombScargle(t.astype(np.float64), y.astype(np.float64),
                             dy=dy.astype(np.float64))
            ls.power(freqs, method='fastnifty')

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'nifty-ls (CPU, fastnifty)', 'times': times}


# --- PDM ------------------------------------------------------------------

def bench_pdm_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase PDMAsyncProcess (GPU)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    proc = cvb_pdm.PDMAsyncProcess()

    def run():
        w = np.ones(ndata, dtype=np.float32) / ndata
        proc.run([(t, y, w, freqs) for t, y, dy in batch],
                 kind='binned_linterp', nbins=10)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'cuvarbase PDMAsyncProcess', 'times': times}


def bench_pdm_cpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase pdm2_cpu (CPU fallback)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        for t, y, dy in batch:
            w = np.ones(len(t), dtype=np.float32) / len(t)
            cvb_pdm.pdm2_cpu(t, y, w, freqs, nbins=10)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'cuvarbase pdm2_cpu', 'times': times}


def bench_pdm_cpu_pyastronomy(ndata, nbatch, nfreq, baseline):
    """PyAstronomy PDM (CPU baseline)."""
    if not HAS_PYASTRONOMY:
        return None, {'error': 'PyAstronomy not installed'}

    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    df = float(freqs[1] - freqs[0])

    def run():
        for t, y, dy in batch:
            P = pyPDM.PyPDM(t.astype(np.float64), y.astype(np.float64))
            scanner = pyPDM.Scanner(minVal=fmin, maxVal=fmax, dVal=df,
                                    mode="frequency")
            P.pdmEquiBinCover(10, 3, scanner)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'PyAstronomy PDM', 'times': times}


# --- Conditional Entropy --------------------------------------------------

def bench_ce_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase ConditionalEntropyAsyncProcess (GPU)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    proc = cvb_ce.ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)

    def run():
        proc.run([(t, y, dy) for t, y, dy in batch], freqs=freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'cuvarbase ConditionalEntropyAsyncProcess',
                 'times': times}


def bench_ce_cpu(ndata, nbatch, nfreq, baseline):
    """Pure-numpy conditional entropy (CPU baseline)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def ce_single(t, y, freqs, nphase_bins=10, nmag_bins=5):
        """Minimal CE implementation for benchmarking."""
        results = np.empty(len(freqs))
        mag_edges = np.linspace(y.min(), y.max() + 1e-10, nmag_bins + 1)
        for i, f in enumerate(freqs):
            phase = (t * f) % 1.0
            H, _, _ = np.histogram2d(phase, y,
                                     bins=[nphase_bins, mag_edges])
            H = H / H.sum()
            p_phase = H.sum(axis=1)
            mask = H > 0
            Hc = np.sum(H[mask] * np.log(
                np.broadcast_to(p_phase[:, None], H.shape)[mask] / H[mask]))
            results[i] = Hc
        return results

    def run():
        for t, y, dy in batch:
            ce_single(t, y, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=False)
    return med, {'variant': 'numpy CE (CPU)', 'times': times}


# --- TLS ------------------------------------------------------------------

def bench_tls_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase tls_transit (GPU, Keplerian)."""
    batch = generate_batch(ndata, nbatch, baseline)

    def run():
        for t, y, dy in batch:
            cvb_tls.tls_transit(t, y, dy,
                                R_star=1.0, M_star=1.0,
                                period_min=0.5, period_max=min(50.0, baseline / 2))

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'cuvarbase tls_transit', 'times': times}


def bench_tls_cpu(ndata, nbatch, nfreq, baseline):
    """transitleastsquares (CPU baseline)."""
    if not HAS_TLS_CPU:
        return None, {'error': 'transitleastsquares not installed'}

    batch = generate_batch(ndata, nbatch, baseline)

    def run():
        for t, y, dy in batch:
            model = transitleastsquares(t.astype(np.float64),
                                        y.astype(np.float64),
                                        dy.astype(np.float64))
            model.power(period_min=0.5,
                        period_max=min(50.0, baseline / 2),
                        show_progress_bar=False)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'transitleastsquares (CPU)', 'times': times}


# --- BLS Batch (multi-LC) -------------------------------------------------

# Realistic survey profiles for batch BLS benchmarks
SURVEY_PROFILES = OrderedDict([
    ('tess_1sector', {
        'display_name': 'TESS 1-sector',
        'ndata': 20000, 'baseline': 27, 'period_min': 0.5, 'period_max': 13.5,
        'qmin': 0.005, 'qmax': 0.1, 'n_lcs': 1000,
    }),
    ('tess_extended', {
        'display_name': 'TESS extended',
        'ndata': 50000, 'baseline': 365, 'period_min': 0.5, 'period_max': 180,
        'qmin': 0.005, 'qmax': 0.1, 'n_lcs': 1000,
    }),
    ('kepler', {
        'display_name': 'Kepler',
        'ndata': 65000, 'baseline': 1460, 'period_min': 0.5, 'period_max': 500,
        'qmin': 0.005, 'qmax': 0.1, 'n_lcs': 500,
    }),
    ('hatnet', {
        'display_name': 'HAT-Net',
        'ndata': 6000, 'baseline': 180, 'period_min': 0.5, 'period_max': 10,
        'qmin': 0.01, 'qmax': 0.1, 'n_lcs': 2000,
    }),
    ('ztf', {
        'display_name': 'ZTF',
        'ndata': 150, 'baseline': 730, 'period_min': 0.5, 'period_max': 100,
        'qmin': 0.01, 'qmax': 0.15, 'n_lcs': 5000,
    }),
])


def bench_bls_batch_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase eebls_gpu_batch (multi-LC kernel)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        cvb_bls.eebls_gpu_batch(batch, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'eebls_gpu_batch', 'times': times}


def bench_bls_batch_single_gpu(ndata, nbatch, nfreq, baseline):
    """cuvarbase eebls_gpu_fast_adaptive in a Python loop (baseline)."""
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        for t, y, dy in batch:
            cvb_bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'eebls_gpu_fast_adaptive (loop)', 'times': times}


def bench_bls_batch_survey(survey_name):
    """Benchmark batch BLS for a specific survey profile."""
    if not HAS_BLS_FREQ:
        return None, {'error': 'bls_frequencies not available'}

    profile = SURVEY_PROFILES[survey_name]
    ndata = profile['ndata']
    n_lcs = profile['n_lcs']
    baseline = profile['baseline']

    freqs = keplerian_freq_grid(
        profile['period_min'], profile['period_max'], baseline
    )
    batch = generate_batch(ndata, n_lcs, baseline)

    def run():
        cvb_bls.eebls_gpu_batch(
            batch, freqs,
            qmin=profile['qmin'], qmax=profile['qmax']
        )

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {
        'variant': f'eebls_gpu_batch ({profile["display_name"]})',
        'survey': survey_name,
        'nfreq_keplerian': len(freqs),
        'times': times,
    }


# ============================================================================
# Algorithm registry
# ============================================================================

ALGORITHMS = OrderedDict([
    ('bls_standard', {
        'display_name': 'Standard BLS (binned)',
        'complexity': 'O(N * Nfreq)',
        'gpu_func': bench_bls_standard_gpu,
        'cpu_funcs': OrderedDict([
            ('astropy', bench_bls_standard_cpu),
        ]),
        'gpu_old_func': bench_bls_standard_gpu_old,
    }),
    ('bls_sparse', {
        'display_name': 'Sparse BLS',
        'complexity': 'O(N^2 * Nfreq)',
        'gpu_func': bench_bls_sparse_gpu,
        'cpu_funcs': OrderedDict([
            ('cuvarbase_cpu', bench_bls_sparse_cpu),
        ]),
        'gpu_old_func': None,
    }),
    ('ls', {
        'display_name': 'Lomb-Scargle',
        'complexity': 'O(N + Nfreq*log(Nfreq))',
        'gpu_func': bench_ls_gpu,
        'cpu_funcs': OrderedDict([
            ('astropy', bench_ls_cpu_astropy),
            ('nifty_ls', bench_ls_cpu_nifty),
        ]),
        'gpu_old_func': None,
    }),
    ('pdm', {
        'display_name': 'Phase Dispersion Minimization',
        'complexity': 'O(N * Nfreq)',
        'gpu_func': bench_pdm_gpu,
        'cpu_funcs': OrderedDict([
            ('cuvarbase_cpu', bench_pdm_cpu),
            ('pyastronomy', bench_pdm_cpu_pyastronomy),
        ]),
        'gpu_old_func': None,
    }),
    ('ce', {
        'display_name': 'Conditional Entropy',
        'complexity': 'O(N * Nfreq)',
        'gpu_func': bench_ce_gpu,
        'cpu_funcs': OrderedDict([
            ('numpy', bench_ce_cpu),
        ]),
        'gpu_old_func': None,
    }),
    ('tls', {
        'display_name': 'Transit Least Squares',
        'complexity': 'O(N * Nperiod * Nduration)',
        'gpu_func': bench_tls_gpu,
        'cpu_funcs': OrderedDict([
            ('transitleastsquares', bench_tls_cpu),
        ]),
        'gpu_old_func': None,
    }),
    ('bls_batch', {
        'display_name': 'BLS Batch (multi-LC)',
        'complexity': 'O(N * Nfreq * N_lc)',
        'gpu_func': bench_bls_batch_gpu,
        'cpu_funcs': OrderedDict([
            ('astropy', bench_bls_standard_cpu),
        ]),
        'gpu_old_func': bench_bls_batch_single_gpu,
    }),
])


# ============================================================================
# System info
# ============================================================================

def get_system_info():
    """Collect system information for the benchmark report."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat(),
    }

    if HAS_GPU:
        dev = cuda.Device(0)
        info['gpu_name'] = dev.name()
        info['gpu_compute_capability'] = '%d.%d' % dev.compute_capability()
        info['gpu_total_memory_mb'] = dev.total_memory() // (1024 * 1024)
        try:
            info['cuda_driver_version'] = '%d.%d' % (
                cuda.get_driver_version() // 1000,
                (cuda.get_driver_version() % 1000) // 10)
        except Exception:
            pass

    if HAS_ASTROPY:
        import astropy
        info['astropy_version'] = astropy.__version__

    if HAS_NIFTY_LS:
        info['nifty_ls_version'] = nifty_ls.__version__

    return info


# ============================================================================
# Cost calculations
# ============================================================================

def compute_cost_per_lc(gpu_time_per_lc, gpu_model):
    """
    Compute cost per lightcurve on RunPod.

    Parameters
    ----------
    gpu_time_per_lc : float
        GPU seconds per lightcurve.
    gpu_model : str
        Key into RUNPOD_PRICING.

    Returns
    -------
    dict with cost info, or None if gpu_model not in pricing table.
    """
    if gpu_model not in RUNPOD_PRICING:
        return None

    price = RUNPOD_PRICING[gpu_model]
    cost_per_sec = price['price_hr'] / 3600.0
    cost_per_lc = gpu_time_per_lc * cost_per_sec
    lc_per_dollar = 1.0 / cost_per_lc if cost_per_lc > 0 else float('inf')

    return {
        'gpu_model': gpu_model,
        'price_per_hr': price['price_hr'],
        'gpu_sec_per_lc': gpu_time_per_lc,
        'cost_per_lc': cost_per_lc,
        'lc_per_dollar': lc_per_dollar,
        'cost_per_million_lc': cost_per_lc * 1e6,
    }


# ============================================================================
# Main benchmark runner
# ============================================================================

def run_benchmarks(algorithms, ndata, nbatch, nfreq, baseline, gpu_model,
                   max_cpu_time=300.0):
    """
    Run the full benchmark suite.

    Parameters
    ----------
    algorithms : list of str
        Algorithm keys to benchmark.
    ndata : int
        Observations per lightcurve.
    nbatch : int
        Number of lightcurves in batch.
    nfreq : int
        Frequency grid size.
    baseline : float
        Observation baseline in days.
    gpu_model : str
        GPU model name for cost calculations.
    max_cpu_time : float
        Maximum CPU time before skipping (seconds).

    Returns
    -------
    results : list of dict
        Benchmark results.
    """
    results = []

    for alg_key in algorithms:
        if alg_key not in ALGORITHMS:
            print(f"Unknown algorithm: {alg_key}, skipping")
            continue

        alg = ALGORITHMS[alg_key]
        print(f"\n{'='*70}")
        print(f"  {alg['display_name']}  ({alg['complexity']})")
        print(f"  ndata={ndata}  nbatch={nbatch}  nfreq={nfreq}  "
              f"baseline={baseline:.0f}d")
        print(f"{'='*70}")

        entry = {
            'algorithm': alg_key,
            'display_name': alg['display_name'],
            'complexity': alg['complexity'],
            'ndata': ndata,
            'nbatch': nbatch,
            'nfreq': nfreq,
            'baseline': baseline,
            'gpu': {},
            'cpu': {},
            'speedups': {},
            'cost': {},
        }

        # --- GPU benchmark ---
        if HAS_CUVARBASE and HAS_GPU:
            print(f"\n  GPU (cuvarbase v1.0)...", end=" ", flush=True)
            try:
                gpu_time, gpu_meta = alg['gpu_func'](ndata, nbatch, nfreq,
                                                      baseline)
                gpu_per_lc = gpu_time / nbatch
                entry['gpu']['cuvarbase_v1'] = {
                    'total_time': gpu_time,
                    'time_per_lc': gpu_per_lc,
                    **gpu_meta,
                }
                print(f"{gpu_time:.4f}s total, {gpu_per_lc:.6f}s/lc")

                # Cost calculation
                cost = compute_cost_per_lc(gpu_per_lc, gpu_model)
                if cost:
                    entry['cost']['cuvarbase_v1'] = cost
                    print(f"    Cost: ${cost['cost_per_lc']:.8f}/lc  "
                          f"({cost['lc_per_dollar']:.0f} lc/$)")

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                entry['gpu']['cuvarbase_v1'] = {'error': str(e)}

            # --- GPU old version (for version comparison) ---
            if alg.get('gpu_old_func'):
                print(f"  GPU (cuvarbase pre-opt)...", end=" ", flush=True)
                try:
                    old_time, old_meta = alg['gpu_old_func'](
                        ndata, nbatch, nfreq, baseline)
                    old_per_lc = old_time / nbatch
                    entry['gpu']['cuvarbase_preopt'] = {
                        'total_time': old_time,
                        'time_per_lc': old_per_lc,
                        **old_meta,
                    }
                    print(f"{old_time:.4f}s total, {old_per_lc:.6f}s/lc")

                    # Speedup vs old version
                    if 'cuvarbase_v1' in entry['gpu']:
                        v1_time = entry['gpu']['cuvarbase_v1']['total_time']
                        if v1_time > 0:
                            improvement = old_time / v1_time
                            entry['speedups']['v1_vs_preopt'] = improvement
                            print(f"    v1.0 is {improvement:.1f}x faster "
                                  f"than pre-optimization")

                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    entry['gpu']['cuvarbase_preopt'] = {'error': str(e)}

        # --- CPU baselines ---
        for cpu_name, cpu_func in alg['cpu_funcs'].items():
            print(f"  CPU ({cpu_name})...", end=" ", flush=True)
            try:
                cpu_time, cpu_meta = cpu_func(ndata, nbatch, nfreq, baseline)
                if cpu_time is None:
                    print(f"SKIPPED: {cpu_meta.get('error', 'unknown')}")
                    entry['cpu'][cpu_name] = cpu_meta
                    continue

                cpu_per_lc = cpu_time / nbatch
                entry['cpu'][cpu_name] = {
                    'total_time': cpu_time,
                    'time_per_lc': cpu_per_lc,
                    **cpu_meta,
                }
                print(f"{cpu_time:.4f}s total, {cpu_per_lc:.6f}s/lc")

                # Speedup: CPU / GPU
                if ('cuvarbase_v1' in entry['gpu'] and
                        'total_time' in entry['gpu']['cuvarbase_v1']):
                    gpu_t = entry['gpu']['cuvarbase_v1']['total_time']
                    if gpu_t > 0:
                        speedup = cpu_time / gpu_t
                        entry['speedups'][f'gpu_vs_{cpu_name}'] = speedup
                        print(f"    GPU is {speedup:.1f}x faster than "
                              f"{cpu_name}")

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                entry['cpu'][cpu_name] = {'error': str(e)}

        results.append(entry)

    return results


# ============================================================================
# Report generation
# ============================================================================

def print_summary(results, gpu_model):
    """Print a summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUMMARY")
    if gpu_model in RUNPOD_PRICING:
        print(f"  GPU: {gpu_model}  "
              f"(${RUNPOD_PRICING[gpu_model]['price_hr']:.2f}/hr RunPod)")
    print(f"{'='*80}\n")

    header = (f"{'Algorithm':<25} {'GPU (s/lc)':<14} {'CPU (s/lc)':<14} "
              f"{'Speedup':<10} {'$/lc':<12}")
    print(header)
    print("-" * len(header))

    for r in results:
        alg_name = r['display_name'][:24]

        # GPU time
        gpu_entry = r['gpu'].get('cuvarbase_v1', {})
        gpu_str = (f"{gpu_entry['time_per_lc']:.6f}"
                   if 'time_per_lc' in gpu_entry else "N/A")

        # Best CPU time (fastest baseline)
        cpu_times = {}
        for name, entry in r['cpu'].items():
            if 'time_per_lc' in entry:
                cpu_times[name] = entry['time_per_lc']

        if cpu_times:
            best_cpu_name = min(cpu_times, key=cpu_times.get)
            best_cpu_time = cpu_times[best_cpu_name]
            cpu_str = f"{best_cpu_time:.6f}"
        else:
            cpu_str = "N/A"
            best_cpu_time = None

        # Speedup
        if ('time_per_lc' in gpu_entry and best_cpu_time is not None and
                gpu_entry['time_per_lc'] > 0):
            speedup = best_cpu_time / gpu_entry['time_per_lc']
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"

        # Cost
        cost_entry = r['cost'].get('cuvarbase_v1', {})
        cost_str = (f"${cost_entry['cost_per_lc']:.8f}"
                    if 'cost_per_lc' in cost_entry else "N/A")

        print(f"{alg_name:<25} {gpu_str:<14} {cpu_str:<14} "
              f"{speedup_str:<10} {cost_str:<12}")

    print()


def save_results(results, system_info, output_file):
    """Save results to JSON."""
    output = {
        'system': system_info,
        'results': results,
        'runpod_pricing': dict(RUNPOD_PRICING),
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark cuvarbase algorithms (GPU vs CPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all algorithms with defaults (10k obs, 10yr baseline)
  python scripts/benchmark_algorithms.py

  # Just BLS and LS
  python scripts/benchmark_algorithms.py --algorithms bls_standard ls

  # TESS-like parameters
  python scripts/benchmark_algorithms.py --ndata 20000 --baseline 730

  # Tag results with GPU model for cost calculation
  python scripts/benchmark_algorithms.py --gpu-model H100_SXM

Available algorithms: """ + ', '.join(ALGORITHMS.keys())
    )

    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=list(ALGORITHMS.keys()),
                        help='Algorithms to benchmark (default: all)')
    parser.add_argument('--ndata', type=int, default=10000,
                        help='Observations per lightcurve (default: 10000)')
    parser.add_argument('--nbatch', type=int, default=100,
                        help='Number of lightcurves in batch (default: 100)')
    parser.add_argument('--nfreq', type=int, default=10000,
                        help='Frequency grid size (default: 10000)')
    parser.add_argument('--baseline', type=float, default=3652.5,
                        help='Observation baseline in days (default: 3652.5 = 10yr)')
    parser.add_argument('--gpu-model', type=str, default='H100_SXM',
                        choices=list(RUNPOD_PRICING.keys()),
                        help='GPU model for cost calculations (default: H100_SXM)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file (default: benchmark_results.json)')
    parser.add_argument('--max-cpu-time', type=float, default=300.0,
                        help='Max CPU time before skipping (default: 300s)')

    args = parser.parse_args()

    print("cuvarbase Benchmark Suite")
    print("=" * 40)
    print(f"Parameters: ndata={args.ndata}, nbatch={args.nbatch}, "
          f"nfreq={args.nfreq}, baseline={args.baseline:.0f}d")
    print(f"GPU available: {HAS_GPU}")
    print(f"cuvarbase available: {HAS_CUVARBASE}")
    print(f"CPU baselines: astropy={HAS_ASTROPY}, nifty-ls={HAS_NIFTY_LS}, "
          f"TLS={HAS_TLS_CPU}, PyAstronomy={HAS_PYASTRONOMY}")

    system_info = get_system_info()
    for k, v in system_info.items():
        print(f"  {k}: {v}")

    results = run_benchmarks(
        algorithms=args.algorithms,
        ndata=args.ndata,
        nbatch=args.nbatch,
        nfreq=args.nfreq,
        baseline=args.baseline,
        gpu_model=args.gpu_model,
        max_cpu_time=args.max_cpu_time,
    )

    print_summary(results, args.gpu_model)
    save_results(results, system_info, args.output)

    # Print cost comparison across GPU models
    print(f"\n{'='*80}")
    print("  COST PER LIGHTCURVE ACROSS GPU MODELS")
    print(f"{'='*80}\n")

    header = f"{'GPU Model':<18} {'$/hr':<8} "
    for r in results:
        header += f"{r['algorithm']:<16} "
    print(header)
    print("-" * len(header))

    for gpu_name, gpu_info in RUNPOD_PRICING.items():
        row = f"{gpu_name:<18} ${gpu_info['price_hr']:<7.2f} "
        for r in results:
            gpu_entry = r['gpu'].get('cuvarbase_v1', {})
            if 'time_per_lc' in gpu_entry:
                cost = compute_cost_per_lc(gpu_entry['time_per_lc'], gpu_name)
                if cost:
                    row += f"${cost['cost_per_lc']:<15.8f} "
                else:
                    row += f"{'N/A':<16} "
            else:
                row += f"{'N/A':<16} "
        print(row)

    print("\nNote: Cost projections for GPUs other than the one used for "
          "benchmarking are estimates based on the measured GPU time. Actual "
          "performance varies by architecture. Run benchmarks on each GPU "
          "for accurate numbers.")


if __name__ == '__main__':
    main()
