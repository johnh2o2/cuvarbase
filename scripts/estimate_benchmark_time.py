#!/usr/bin/env python3
"""
Estimate benchmark runtime based on algorithm complexity and configuration.

Provides rough estimates to help plan benchmarking runs.
"""

import argparse
from typing import Dict, Tuple

# Algorithm complexities (exponents for ndata, nfreq scaling)
COMPLEXITY = {
    'sparse_bls': {'ndata': 2, 'nfreq': 1, 'base_time_cpu': 0.5, 'base_time_gpu': 0.002},
    'bls_gpu_fast': {'ndata': 2, 'nfreq': 1, 'base_time_cpu': None, 'base_time_gpu': 0.002},
}

# Base measurements (seconds) for ndata=100, nfreq=100, nbatch=1
# These are rough estimates based on RTX A5000
BASE_CONFIG = {'ndata': 100, 'nfreq': 100, 'nbatch': 1}


def estimate_runtime(algorithm: str, ndata: int, nfreq: int, nbatch: int,
                    backend: str = 'gpu') -> float:
    """
    Estimate runtime for a single configuration.

    Parameters
    ----------
    algorithm : str
        Algorithm name
    ndata : int
        Number of observations per lightcurve
    nfreq : int
        Number of frequencies
    nbatch : int
        Number of lightcurves
    backend : str
        'cpu' or 'gpu'

    Returns
    -------
    time : float
        Estimated time in seconds
    """
    if algorithm not in COMPLEXITY:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    comp = COMPLEXITY[algorithm]
    base_key = f'base_time_{backend}'

    if comp[base_key] is None:
        return float('inf')  # No CPU version

    base_time = comp[base_key]

    # Scale from base configuration
    scale_ndata = (ndata / BASE_CONFIG['ndata']) ** comp['ndata']
    scale_nfreq = (nfreq / BASE_CONFIG['nfreq']) ** comp['nfreq']
    scale_nbatch = nbatch / BASE_CONFIG['nbatch']

    return base_time * scale_ndata * scale_nfreq * scale_nbatch


def estimate_full_suite(algorithm: str,
                       ndata_values: list,
                       nbatch_values: list,
                       nfreq: int,
                       max_cpu_time: float,
                       max_gpu_time: float) -> Dict:
    """
    Estimate full benchmark suite runtime.

    Returns
    -------
    summary : dict
        Contains total times, number of experiments, etc.
    """
    cpu_measured = []
    cpu_extrapolated = []
    gpu_measured = []
    gpu_extrapolated = []

    for ndata in ndata_values:
        for nbatch in nbatch_values:
            # Estimate CPU time
            cpu_time = estimate_runtime(algorithm, ndata, nfreq, nbatch, 'cpu')
            if cpu_time == float('inf'):
                pass  # No CPU version
            elif cpu_time <= max_cpu_time:
                cpu_measured.append(cpu_time)
            else:
                cpu_extrapolated.append((ndata, nbatch))

            # Estimate GPU time
            gpu_time = estimate_runtime(algorithm, ndata, nfreq, nbatch, 'gpu')
            if gpu_time <= max_gpu_time:
                gpu_measured.append(gpu_time)
            else:
                gpu_extrapolated.append((ndata, nbatch))

    total_cpu = sum(cpu_measured)
    total_gpu = sum(gpu_measured)
    total_time = total_cpu + total_gpu

    return {
        'algorithm': algorithm,
        'total_experiments': len(ndata_values) * len(nbatch_values),
        'cpu_measured': len(cpu_measured),
        'cpu_extrapolated': len(cpu_extrapolated),
        'gpu_measured': len(gpu_measured),
        'gpu_extrapolated': len(gpu_extrapolated),
        'total_cpu_time': total_cpu,
        'total_gpu_time': total_gpu,
        'total_time': total_time,
        'cpu_extrap_configs': cpu_extrapolated,
        'gpu_extrap_configs': gpu_extrapolated,
    }


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description='Estimate benchmark runtime')
    parser.add_argument('--algorithms', nargs='+', default=['sparse_bls'],
                       help='Algorithms to estimate')
    parser.add_argument('--max-cpu-time', type=float, default=300,
                       help='Max CPU time before extrapolation (seconds)')
    parser.add_argument('--max-gpu-time', type=float, default=120,
                       help='Max GPU time before extrapolation (seconds)')

    args = parser.parse_args()

    # Benchmark grid
    ndata_values = [10, 100, 1000]
    nbatch_values = [1, 10, 100, 1000]
    nfreq = 100

    print("=" * 70)
    print("BENCHMARK RUNTIME ESTIMATES")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  ndata values: {ndata_values}")
    print(f"  nbatch values: {nbatch_values}")
    print(f"  nfreq: {nfreq}")
    print(f"  CPU timeout: {format_time(args.max_cpu_time)}")
    print(f"  GPU timeout: {format_time(args.max_gpu_time)}")
    print()

    total_estimate = 0

    for algorithm in args.algorithms:
        if algorithm not in COMPLEXITY:
            print(f"Warning: Unknown algorithm '{algorithm}', skipping")
            continue

        print("-" * 70)
        print(f"Algorithm: {algorithm}")
        print("-" * 70)

        summary = estimate_full_suite(
            algorithm, ndata_values, nbatch_values, nfreq,
            args.max_cpu_time, args.max_gpu_time
        )

        print(f"Total experiments: {summary['total_experiments']}")
        print()
        print(f"CPU benchmarks:")
        print(f"  Measured: {summary['cpu_measured']} experiments")
        print(f"  Extrapolated: {summary['cpu_extrapolated']} experiments")
        print(f"  Total CPU time: {format_time(summary['total_cpu_time'])}")
        print()
        print(f"GPU benchmarks:")
        print(f"  Measured: {summary['gpu_measured']} experiments")
        print(f"  Extrapolated: {summary['gpu_extrapolated']} experiments")
        print(f"  Total GPU time: {format_time(summary['total_gpu_time'])}")
        print()
        print(f"Total runtime estimate: {format_time(summary['total_time'])}")

        if summary['cpu_extrap_configs']:
            print()
            print(f"CPU extrapolated configs (too slow):")
            for ndata, nbatch in summary['cpu_extrap_configs']:
                est_time = estimate_runtime(algorithm, ndata, nfreq, nbatch, 'cpu')
                print(f"  ndata={ndata}, nbatch={nbatch}: ~{format_time(est_time)}")

        if summary['gpu_extrap_configs']:
            print()
            print(f"GPU extrapolated configs:")
            for ndata, nbatch in summary['gpu_extrap_configs']:
                est_time = estimate_runtime(algorithm, ndata, nfreq, nbatch, 'gpu')
                print(f"  ndata={ndata}, nbatch={nbatch}: ~{format_time(est_time)}")

        print()
        total_estimate += summary['total_time']

    print("=" * 70)
    print(f"TOTAL ESTIMATED TIME: {format_time(total_estimate)}")
    print("=" * 70)
    print()
    print("Notes:")
    print("  - These are rough estimates based on RTX A5000 performance")
    print("  - Actual times may vary by ±50% depending on GPU model and system load")
    print("  - Extrapolated experiments add negligible runtime (~1s each)")
    print("  - First run may be slower due to CUDA compilation")
    print()


if __name__ == '__main__':
    main()
