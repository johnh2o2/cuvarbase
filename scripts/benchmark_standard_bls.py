#!/usr/bin/env python3
"""
Benchmark standard (non-sparse) BLS with Keplerian assumption.

Compares:
- Astropy BoxLeastSquares (CPU baseline)
- cuvarbase eebls_gpu_fast (GPU)

For TESS-realistic parameters: ndata=20000, nfreq=1000
"""

import numpy as np
import time
import json
import argparse
from astropy.timeseries import BoxLeastSquares

try:
    from cuvarbase import bls
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("WARNING: cuvarbase not available, GPU benchmarks will be skipped")


def benchmark_astropy_bls(ndata, nfreq, nbatch=1):
    """Benchmark astropy BoxLeastSquares (CPU)."""
    np.random.seed(42)

    total_time = 0
    for _ in range(nbatch):
        t = np.sort(np.random.uniform(0, 27, ndata))
        y = np.random.randn(ndata) * 0.01
        dy = np.ones(ndata) * 0.01

        freqs = np.linspace(1.0/13.5, 1.0/0.5, nfreq)
        periods = 1.0 / freqs
        durations = 0.05 * (periods / 10) ** (1/3)  # Keplerian

        model = BoxLeastSquares(t, y, dy)
        start = time.time()
        results = model.power(periods, duration=durations)
        total_time += time.time() - start

    return total_time


def benchmark_cuvarbase_gpu(ndata, nfreq, nbatch=1):
    """Benchmark cuvarbase eebls_gpu_fast."""
    if not GPU_AVAILABLE:
        return None

    np.random.seed(42)

    # Warm up GPU
    t_warmup = np.sort(np.random.uniform(0, 27, 100)).astype(np.float32)
    y_warmup = np.random.randn(100).astype(np.float32) * 0.01
    dy_warmup = np.ones(100, dtype=np.float32) * 0.01
    freqs_warmup = np.linspace(1.0/13.5, 1.0/0.5, 10).astype(np.float32)
    _ = bls.eebls_gpu_fast(t_warmup, y_warmup, dy_warmup, freqs_warmup)

    total_time = 0
    for _ in range(nbatch):
        t = np.sort(np.random.uniform(0, 27, ndata)).astype(np.float32)
        y = np.random.randn(ndata).astype(np.float32) * 0.01
        dy = np.ones(ndata, dtype=np.float32) * 0.01

        freqs = np.linspace(1.0/13.5, 1.0/0.5, nfreq).astype(np.float32)

        start = time.time()
        results = bls.eebls_gpu_fast(t, y, dy, freqs)
        total_time += time.time() - start

    return total_time


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("STANDARD BLS BENCHMARK (Non-sparse, Keplerian assumption)")
    print("=" * 80)

    # Test configurations
    configs = [
        {'ndata': 1000, 'nfreq': 100, 'nbatch': 1},
        {'ndata': 1000, 'nfreq': 100, 'nbatch': 10},
        {'ndata': 10000, 'nfreq': 1000, 'nbatch': 1},
        {'ndata': 20000, 'nfreq': 1000, 'nbatch': 1},
        {'ndata': 20000, 'nfreq': 1000, 'nbatch': 10},
    ]

    results = []

    for config in configs:
        ndata = config['ndata']
        nfreq = config['nfreq']
        nbatch = config['nbatch']

        print(f"\nConfig: ndata={ndata}, nfreq={nfreq}, nbatch={nbatch}")

        # CPU benchmark
        print("  Running Astropy CPU benchmark...", end=' ', flush=True)
        time_cpu = benchmark_astropy_bls(ndata, nfreq, nbatch)
        print(f"{time_cpu:.2f}s")

        # GPU benchmark
        if GPU_AVAILABLE:
            print("  Running cuvarbase GPU benchmark...", end=' ', flush=True)
            time_gpu = benchmark_cuvarbase_gpu(ndata, nfreq, nbatch)
            print(f"{time_gpu:.2f}s")
            speedup = time_cpu / time_gpu if time_gpu else None
            if speedup:
                print(f"  Speedup: {speedup:.1f}x")
        else:
            time_gpu = None
            speedup = None

        results.append({
            'ndata': ndata,
            'nfreq': nfreq,
            'nbatch': nbatch,
            'time_cpu': time_cpu,
            'time_gpu': time_gpu,
            'speedup': speedup,
        })

    # Save results
    with open('standard_bls_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"{'ndata':<8} {'nfreq':<8} {'nbatch':<8} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup'}")
    print("-" * 80)

    for r in results:
        gpu_str = f"{r['time_gpu']:.2f}" if r['time_gpu'] else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
        print(f"{r['ndata']:<8} {r['nfreq']:<8} {r['nbatch']:<8} {r['time_cpu']:<12.2f} {gpu_str:<12} {speedup_str}")

    # TESS-scale analysis
    if any(r['ndata'] == 20000 and r['nbatch'] == 1 for r in results):
        tess_result = [r for r in results if r['ndata'] == 20000 and r['nbatch'] == 1][0]

        print("\n" + "=" * 80)
        print("TESS CATALOG PROJECTION (5M lightcurves, 20k obs each):")
        print("=" * 80)

        # CPU projections
        time_per_lc_cpu = tess_result['time_cpu']

        cpu_options = [
            {'name': 'Hetzner CCX63 (48 vCPU)', 'cores': 48, 'eff': 0.85, 'cost_hr': 0.82},
            {'name': 'AWS c7i.24xlarge (96 vCPU, spot)', 'cores': 96, 'eff': 0.80, 'cost_hr': 4.08 * 0.70},
            {'name': 'AWS c7i.48xlarge (192 vCPU, spot)', 'cores': 192, 'eff': 0.75, 'cost_hr': 8.16 * 0.70},
        ]

        print("\nCPU Options (Astropy BLS):")
        for opt in cpu_options:
            speedup = opt['cores'] * opt['eff']
            time_per_lc = time_per_lc_cpu / speedup
            total_hours = time_per_lc * 5_000_000 / 3600
            total_days = total_hours / 24
            total_cost = total_hours * opt['cost_hr']

            print(f"  {opt['name']:45s}: {total_days:6.1f} days, ${total_cost:10,.0f}")

        # GPU projections
        if tess_result['time_gpu']:
            time_per_lc_gpu = tess_result['time_gpu']

            # Check if we have batch=10 data
            tess_batch = [r for r in results if r['ndata'] == 20000 and r['nbatch'] == 10]
            if tess_batch:
                time_per_lc_gpu_batched = tess_batch[0]['time_gpu'] / 10
                batch_efficiency = time_per_lc_gpu / time_per_lc_gpu_batched
                print(f"\n  GPU batch efficiency: {batch_efficiency:.2f}x at nbatch=10")
                time_per_lc_gpu = time_per_lc_gpu_batched

            gpu_options = [
                {'name': 'RunPod RTX 4000 Ada (spot)', 'speedup': 1.0, 'cost_hr': 0.29 * 0.80},
                {'name': 'RunPod L40 (spot)', 'speedup': 1.5, 'cost_hr': 0.49 * 0.80},
                {'name': 'RunPod A100 40GB (spot)', 'speedup': 2.0, 'cost_hr': 0.89 * 0.85},
                {'name': 'RunPod H100 (spot)', 'speedup': 3.5, 'cost_hr': 1.99 * 0.85},
            ]

            print("\nGPU Options (cuvarbase eebls_gpu_fast, single GPU):")
            for opt in gpu_options:
                time_per_lc = time_per_lc_gpu / opt['speedup']
                total_hours = time_per_lc * 5_000_000 / 3600
                total_days = total_hours / 24
                total_cost = total_hours * opt['cost_hr']

                print(f"  {opt['name']:45s}: {total_days:6.1f} days, ${total_cost:10,.0f}")

    print("\nResults saved to: standard_bls_benchmark.json")


if __name__ == '__main__':
    run_benchmarks()
