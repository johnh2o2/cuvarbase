#!/usr/bin/env python
"""Throughput benchmark: CPU (Rust) vs CUDA backends.

Not a pytest test — run as a standalone script:
    python tests/bench_cpu_vs_cuda.py

Requires GPU for CUDA timings; CPU-only timings are always printed.
"""

import subprocess
import time

import numpy as np

# Check GPU availability
HAS_GPU = False
try:
    from periodfind.gpu import AOV as CudaAOV
    from periodfind.gpu import FPW as CudaFPW
    from periodfind.gpu import BoxLeastSquares as CudaBLS
    from periodfind.gpu import ConditionalEntropy as CudaCE
    from periodfind.gpu import LombScargle as CudaLS

    ret = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    if ret.returncode == 0:
        HAS_GPU = True
except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
    pass

from periodfind.cpu import AOV as CpuAOV
from periodfind.cpu import BoxLeastSquares as CpuBLS
from periodfind.cpu import ConditionalEntropy as CpuCE
from periodfind.cpu import FPW as CpuFPW
from periodfind.cpu import LombScargle as CpuLS


def make_lightcurve(n_points=500, period=5.0, seed=42):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 200, n_points)).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    mags = (np.sin(phase) + 1.0).astype(np.float32)
    mags += rng.normal(0, 0.05, n_points).astype(np.float32)
    errs = np.full(n_points, 0.05, dtype=np.float32)
    return times, mags, errs


def bench(fn, warmup=1, repeats=3):
    """Time a function, returning median wall-clock time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)


WORKLOADS = [
    ("Small   (1×200×1)",   1,   200, 1),
    ("Medium  (1×2000×3)",  1,  2000, 3),
    ("Large   (10×5000×1)", 10, 5000, 1),
    ("XL      (50×2000×1)", 50, 2000, 1),
    ("XXL    (100×2000×1)", 100, 2000, 1),
    ("Batch  (500×1000×1)", 500, 1000, 1),
]

ALGORITHMS = [
    ("CE",  CpuCE,  (CudaCE  if HAS_GPU else None), {"n_phase": 10, "n_mag": 10}, False),
    ("AOV", CpuAOV, (CudaAOV if HAS_GPU else None), {"n_phase": 10}, False),
    ("LS",  CpuLS,  (CudaLS  if HAS_GPU else None), {}, False),
    ("FPW", CpuFPW, (CudaFPW if HAS_GPU else None), {"n_bins": 10}, True),
    ("BLS", CpuBLS, (CudaBLS if HAS_GPU else None), {"n_bins": 50, "qmin": 0.01, "qmax": 0.5}, True),
]


def main():
    header = f"{'Algorithm':<6} | {'Workload':<22} | {'CUDA (ms)':>10} | {'CPU (ms)':>10} | {'Speedup':>8}"
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for algo_name, CpuCls, CudaCls, kwargs, needs_errs in ALGORITHMS:
        cpu_algo = CpuCls(**kwargs)
        cuda_algo = CudaCls(**kwargs) if CudaCls else None

        for wl_name, n_curves, n_periods, n_pdts in WORKLOADS:
            # Generate data
            times_list = []
            mags_list = []
            errs_list = []
            for i in range(n_curves):
                t, m, e = make_lightcurve(n_points=500, period=5.0, seed=i)
                times_list.append(t)
                mags_list.append(m)
                errs_list.append(e)

            periods = np.linspace(1.0, 10.0, n_periods, dtype=np.float32)
            period_dts = np.linspace(-0.001, 0.001, n_pdts, dtype=np.float32)

            extra = {"errs": errs_list} if needs_errs else {}

            # CPU benchmark
            cpu_ms = bench(
                lambda: cpu_algo.calc(
                    times_list, mags_list, periods, period_dts, output="stats", **extra
                )
            )

            # CUDA benchmark
            if cuda_algo:
                cuda_ms = bench(
                    lambda: cuda_algo.calc(
                        times_list, mags_list, periods, period_dts, output="stats", **extra
                    )
                )
                speedup = cpu_ms / cuda_ms
                cuda_str = f"{cuda_ms:10.1f}"
                speedup_str = f"{speedup:7.1f}x"
            else:
                cuda_str = "       N/A"
                speedup_str = "     N/A"

            print(f"{algo_name:<6} | {wl_name:<22} | {cuda_str} | {cpu_ms:10.1f} | {speedup_str}")

    print(sep)
    print()


if __name__ == "__main__":
    main()
