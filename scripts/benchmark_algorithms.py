#!/usr/bin/env python3
"""
Comprehensive benchmark suite for cuvarbase algorithms.

Benchmarks CPU vs GPU performance across different algorithms as a function of:
1. Number of observations per lightcurve (ndata)
2. Number of lightcurves in batch (nbatch)

For experiments that would take too long on CPU, extrapolates using
algorithm-specific scaling laws.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import argparse

# Add cuvarbase to path if running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cuvarbase.bls as bls
    import cuvarbase.lombscargle as ls
    import cuvarbase.pdm as pdm
    HAS_GPU = True
except ImportError as e:
    print(f"Warning: Could not import cuvarbase GPU modules: {e}")
    HAS_GPU = False


# ============================================================================
# Data Generation
# ============================================================================

def generate_lightcurve(ndata: int, baseline: float = 5*365.25,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic lightcurve with random sampling.

    Parameters
    ----------
    ndata : int
        Number of observations
    baseline : float
        Observation baseline in days (default: 5 years)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    t : array
        Observation times
    y : array
        Flux measurements
    dy : array
        Measurement uncertainties
    """
    if seed is not None:
        np.random.seed(seed)

    # Random sampling over baseline
    t = np.sort(np.random.uniform(0, baseline, ndata))

    # Simple sinusoidal signal + noise
    freq = 1.0 / 100.0  # 100-day period
    amp = 0.1
    y = amp * np.sin(2 * np.pi * freq * t) + np.random.randn(ndata) * 0.05
    dy = np.ones(ndata) * 0.05

    return t.astype(np.float32), y.astype(np.float32), dy.astype(np.float32)


def generate_batch(ndata: int, nbatch: int, baseline: float = 5*365.25,
                  seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate a batch of lightcurves."""
    if seed is not None:
        np.random.seed(seed)

    lightcurves = []
    for i in range(nbatch):
        lc_seed = None if seed is None else seed + i
        lightcurves.append(generate_lightcurve(ndata, baseline, lc_seed))
    return lightcurves


# ============================================================================
# Algorithm Complexity and Scaling Laws
# ============================================================================

ALGORITHM_COMPLEXITY = {
    # BLS algorithms - O(N² * Nfreq) for binned, O(N² * Nfreq) for sparse
    'bls_gpu_fast': {'ndata': 2, 'nfreq': 1, 'nbatch': 1},
    'bls_gpu_custom': {'ndata': 2, 'nfreq': 1, 'nbatch': 1},
    'sparse_bls_gpu': {'ndata': 2, 'nfreq': 1, 'nbatch': 1},

    # Lomb-Scargle - O(N * Nfreq)
    'lombscargle_gpu': {'ndata': 1, 'nfreq': 1, 'nbatch': 1},

    # PDM - O(N * Nfreq)
    'pdm_gpu': {'ndata': 1, 'nfreq': 1, 'nbatch': 1},
}


def estimate_runtime(algorithm: str, ndata: int, nfreq: int, nbatch: int,
                    reference_time: float, ref_ndata: int, ref_nfreq: int,
                    ref_nbatch: int) -> float:
    """
    Estimate runtime using scaling law.

    Parameters
    ----------
    algorithm : str
        Algorithm name
    ndata, nfreq, nbatch : int
        Target problem size
    reference_time : float
        Measured time for reference problem
    ref_ndata, ref_nfreq, ref_nbatch : int
        Reference problem size

    Returns
    -------
    estimated_time : float
        Estimated runtime in seconds
    """
    complexity = ALGORITHM_COMPLEXITY.get(algorithm, {'ndata': 1, 'nfreq': 1, 'nbatch': 1})

    scale_ndata = (ndata / ref_ndata) ** complexity['ndata']
    scale_nfreq = (nfreq / ref_nfreq) ** complexity['nfreq']
    scale_nbatch = (nbatch / ref_nbatch) ** complexity['nbatch']

    return reference_time * scale_ndata * scale_nfreq * scale_nbatch


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, algorithm: str, ndata: int, nbatch: int, nfreq: int):
        self.algorithm = algorithm
        self.ndata = ndata
        self.nbatch = nbatch
        self.nfreq = nfreq
        self.cpu_time = None
        self.gpu_time = None
        self.cpu_extrapolated = False
        self.gpu_extrapolated = False
        self.error = None

    def set_cpu_time(self, time_seconds: float, extrapolated: bool = False):
        self.cpu_time = time_seconds
        self.cpu_extrapolated = extrapolated

    def set_gpu_time(self, time_seconds: float, extrapolated: bool = False):
        self.gpu_time = time_seconds
        self.gpu_extrapolated = extrapolated

    def speedup(self) -> Optional[float]:
        if self.cpu_time and self.gpu_time:
            return self.cpu_time / self.gpu_time
        return None

    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'ndata': self.ndata,
            'nbatch': self.nbatch,
            'nfreq': self.nfreq,
            'cpu_time': self.cpu_time,
            'gpu_time': self.gpu_time,
            'cpu_extrapolated': self.cpu_extrapolated,
            'gpu_extrapolated': self.gpu_extrapolated,
            'speedup': self.speedup(),
            'error': self.error
        }


class BenchmarkRunner:
    """Runs benchmarks with timeout and extrapolation support."""

    def __init__(self, max_cpu_time: float = 300.0, max_gpu_time: float = 60.0):
        """
        Parameters
        ----------
        max_cpu_time : float
            Maximum CPU runtime before switching to extrapolation (seconds)
        max_gpu_time : float
            Maximum GPU runtime before switching to extrapolation (seconds)
        """
        self.max_cpu_time = max_cpu_time
        self.max_gpu_time = max_gpu_time
        self.results: List[BenchmarkResult] = []

    def run_with_timeout(self, func: Callable, timeout: float,
                        *args, **kwargs) -> Tuple[Optional[float], bool]:
        """
        Run function with timeout check.

        Returns
        -------
        runtime : float or None
            Runtime in seconds, or None if skipped
        success : bool
            True if actually run, False if extrapolated/skipped
        """
        # Simple timeout: if estimated time > timeout, skip
        start = time.time()
        try:
            func(*args, **kwargs)
            return time.time() - start, True
        except Exception as e:
            print(f"Error in benchmark: {e}")
            return None, False

    def benchmark_algorithm(self, algorithm_name: str,
                          benchmark_func: Callable,
                          ndata_values: List[int],
                          nbatch_values: List[int],
                          nfreq: int = 100):
        """
        Benchmark an algorithm across parameter grid.

        Parameters
        ----------
        algorithm_name : str
            Name of algorithm
        benchmark_func : callable
            Function with signature (ndata, nbatch, nfreq, backend='cpu'|'gpu')
            that runs the benchmark and returns runtime in seconds
        ndata_values : list of int
            Observation counts to test
        nbatch_values : list of int
            Batch sizes to test
        nfreq : int
            Number of frequencies to test
        """
        print(f"\n{'='*70}")
        print(f"Benchmarking: {algorithm_name}")
        print(f"{'='*70}")

        # Track reference measurements for extrapolation
        cpu_reference = {}  # (ndata, nbatch) -> time
        gpu_reference = {}

        for ndata in ndata_values:
            for nbatch in nbatch_values:
                result = BenchmarkResult(algorithm_name, ndata, nbatch, nfreq)

                print(f"\nConfiguration: ndata={ndata}, nbatch={nbatch}, nfreq={nfreq}")

                # CPU Benchmark
                print("  CPU: ", end="", flush=True)

                # Check if we should extrapolate
                should_extrapolate_cpu = False
                if cpu_reference:
                    # Estimate based on closest smaller reference
                    ref_key = self._find_closest_reference(cpu_reference, ndata, nbatch)
                    if ref_key:
                        ref_ndata, ref_nbatch = ref_key
                        estimated_time = estimate_runtime(
                            algorithm_name, ndata, nfreq, nbatch,
                            cpu_reference[ref_key], ref_ndata, nfreq, ref_nbatch
                        )
                        if estimated_time > self.max_cpu_time:
                            should_extrapolate_cpu = True
                            result.set_cpu_time(estimated_time, extrapolated=True)
                            print(f"Extrapolated: {estimated_time:.2f}s (est.)")

                if not should_extrapolate_cpu:
                    try:
                        cpu_time = benchmark_func(ndata, nbatch, nfreq, backend='cpu')
                        result.set_cpu_time(cpu_time, extrapolated=False)
                        cpu_reference[(ndata, nbatch)] = cpu_time
                        print(f"Measured: {cpu_time:.2f}s")
                    except Exception as e:
                        print(f"Error: {e}")
                        result.error = str(e)

                # GPU Benchmark
                if HAS_GPU:
                    print("  GPU: ", end="", flush=True)

                    should_extrapolate_gpu = False
                    if gpu_reference:
                        ref_key = self._find_closest_reference(gpu_reference, ndata, nbatch)
                        if ref_key:
                            ref_ndata, ref_nbatch = ref_key
                            estimated_time = estimate_runtime(
                                algorithm_name, ndata, nfreq, nbatch,
                                gpu_reference[ref_key], ref_ndata, nfreq, ref_nbatch
                            )
                            if estimated_time > self.max_gpu_time:
                                should_extrapolate_gpu = True
                                result.set_gpu_time(estimated_time, extrapolated=True)
                                print(f"Extrapolated: {estimated_time:.2f}s (est.)")

                    if not should_extrapolate_gpu:
                        try:
                            gpu_time = benchmark_func(ndata, nbatch, nfreq, backend='gpu')
                            result.set_gpu_time(gpu_time, extrapolated=False)
                            gpu_reference[(ndata, nbatch)] = gpu_time
                            print(f"Measured: {gpu_time:.2f}s")
                        except Exception as e:
                            print(f"Error: {e}")
                            if result.error is None:
                                result.error = str(e)

                # Report speedup
                if result.speedup():
                    marker = "*" if (result.cpu_extrapolated or result.gpu_extrapolated) else ""
                    print(f"  Speedup: {result.speedup():.1f}x{marker}")

                self.results.append(result)

    def _find_closest_reference(self, references: Dict, ndata: int,
                               nbatch: int) -> Optional[Tuple[int, int]]:
        """Find closest smaller reference measurement."""
        candidates = [(nd, nb) for nd, nb in references.keys()
                     if nd <= ndata and nb <= nbatch]
        if not candidates:
            return None
        # Return largest reference that's still smaller
        return max(candidates, key=lambda x: x[0] * x[1])

    def save_results(self, filename: str):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nResults saved to: {filename}")

    def print_summary(self):
        """Print summary table."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")

        # Group by algorithm
        by_algorithm = {}
        for r in self.results:
            if r.algorithm not in by_algorithm:
                by_algorithm[r.algorithm] = []
            by_algorithm[r.algorithm].append(r)

        for alg, results in by_algorithm.items():
            print(f"\n{alg}:")
            print(f"{'ndata':<10} {'nbatch':<10} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<10}")
            print("-" * 70)

            for r in results:
                cpu_str = f"{r.cpu_time:.2f}" if r.cpu_time else "N/A"
                if r.cpu_extrapolated:
                    cpu_str += "*"

                gpu_str = f"{r.gpu_time:.2f}" if r.gpu_time else "N/A"
                if r.gpu_extrapolated:
                    gpu_str += "*"

                speedup_str = f"{r.speedup():.1f}x" if r.speedup() else "N/A"

                print(f"{r.ndata:<10} {r.nbatch:<10} {cpu_str:<15} {gpu_str:<15} {speedup_str:<10}")

        print("\n* = extrapolated value")


# ============================================================================
# Algorithm-Specific Benchmark Functions
# ============================================================================

def benchmark_sparse_bls(ndata: int, nbatch: int, nfreq: int, backend: str = 'gpu') -> float:
    """Benchmark sparse BLS algorithm."""
    lightcurves = generate_batch(ndata, nbatch)
    freqs = np.linspace(0.005, 0.02, nfreq).astype(np.float32)

    start = time.time()

    for t, y, dy in lightcurves:
        if backend == 'gpu':
            _ = bls.sparse_bls_gpu(t, y, dy, freqs)
        else:
            _ = bls.sparse_bls_cpu(t, y, dy, freqs)

    return time.time() - start


def benchmark_bls_gpu_fast(ndata: int, nbatch: int, nfreq: int, backend: str = 'gpu') -> float:
    """Benchmark fast BLS algorithm."""
    if backend == 'cpu':
        # No CPU equivalent for fast BLS
        raise NotImplementedError("Fast BLS is GPU-only")

    lightcurves = generate_batch(ndata, nbatch)
    freqs = np.linspace(0.005, 0.02, nfreq).astype(np.float32)

    start = time.time()

    for t, y, dy in lightcurves:
        _ = bls.eebls_gpu_fast(t, y, dy, freqs)

    return time.time() - start


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark cuvarbase algorithms')
    parser.add_argument('--max-cpu-time', type=float, default=300.0,
                       help='Max CPU time before extrapolation (seconds)')
    parser.add_argument('--max-gpu-time', type=float, default=60.0,
                       help='Max GPU time before extrapolation (seconds)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['sparse_bls'],
                       help='Algorithms to benchmark')

    args = parser.parse_args()

    # Benchmark grid: 10, 100, 1000 ndata x 1, 10, 100, 1000 nbatch
    ndata_values = [10, 100, 1000]
    nbatch_values = [1, 10, 100, 1000]
    nfreq = 100

    runner = BenchmarkRunner(max_cpu_time=args.max_cpu_time,
                            max_gpu_time=args.max_gpu_time)

    # Run benchmarks
    if 'sparse_bls' in args.algorithms:
        runner.benchmark_algorithm('sparse_bls', benchmark_sparse_bls,
                                  ndata_values, nbatch_values, nfreq)

    if 'bls_gpu_fast' in args.algorithms and HAS_GPU:
        runner.benchmark_algorithm('bls_gpu_fast', benchmark_bls_gpu_fast,
                                  ndata_values, nbatch_values, nfreq)

    # Print and save results
    runner.print_summary()
    runner.save_results(args.output)

    print(f"\n{'='*80}")
    print("GPU Architecture Notes:")
    print(f"{'='*80}")
    print("""
GPU generation differences (for these algorithms):

RTX A5000 (Ampere, 2021):
  - Good baseline performance
  - 24GB VRAM, 8192 CUDA cores
  - PCIe Gen 4
  - Expected: 1x baseline

L40 (Ada Lovelace, 2023):
  - ~1.5-2x faster than A5000 for FP32
  - 48GB VRAM, improved memory bandwidth
  - Better for large batches

A100 (Ampere, 2020):
  - Professional compute card
  - ~1.5-2x faster than A5000 for these workloads
  - 40/80GB VRAM options
  - Higher memory bandwidth (1.5-2 TB/s)
  - Best for mixed precision if utilized

H100 (Hopper, 2022):
  - ~2-3x faster than A100 for FP32
  - 80GB VRAM, ~3 TB/s bandwidth
  - Transformer engine (not used here)
  - Expected: 3-4x faster than A5000

H200 (Hopper refresh, 2024):
  - ~5-10% faster than H100
  - 141GB HBM3e, ~4.8 TB/s bandwidth
  - Best for memory-bound workloads
  - Expected: 3.5-4.5x faster than A5000

B200 (Blackwell, 2025):
  - ~2-3x faster than H100 for compute
  - 192GB HBM3e
  - Most benefit from FP4/FP6 (not applicable here)
  - For FP32: ~5-6x faster than A5000
  - Memory bandwidth improvements help large batches

Key factors for these algorithms:
1. Memory bandwidth > compute (BLS is memory-bound)
2. Batch processing benefits from higher VRAM
3. FP32 performance matters (we use float32)
4. Newer architectures have better occupancy/scheduling

Rough speedup estimates vs A5000:
  A5000: 1.0x
  L40:   1.5-2.0x
  A100:  1.5-2.5x
  H100:  3.0-4.0x
  H200:  3.5-4.5x
  B200:  5.0-7.0x (mostly from bandwidth for our workloads)
""")


if __name__ == '__main__':
    main()
