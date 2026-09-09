#!/usr/bin/env python3
"""Throughput benchmark for periodfind CPU and GPU backends.

Measures wall-clock time for each algorithm across a range of light curve sizes.
Outputs a CSV table and a log-scale plot.
"""

import argparse
import time
import sys
import os
import csv

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N_POINTS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_CURVES = 100  # batch size for point-scaling sweep

# Curve-count scaling parameters
CURVE_COUNTS = [100]
FIXED_N_POINTS = 1024  # fixed point count for curve-scaling sweep

N_PERIODS = 1000
PERIOD_MIN, PERIOD_MAX = 0.5, 10.0
N_WARMUP = 1
N_REPEAT = 3

# Algorithm configs
ALGO_CONFIGS = {
    "CE":  {"n_phase": 10, "n_mag": 10},
    "AOV": {"n_phase": 15},
    "LS":  {},
    "FPW": {"n_bins": 10},
    "BLS": {"n_bins": 50, "qmin": 0.01, "qmax": 0.5},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lightcurves(n_curves, n_points, seed=42):
    """Generate synthetic sinusoidal light curves."""
    rng = np.random.RandomState(seed)
    period = 2.5
    times_list, mags_list, errs_list = [], [], []
    for _ in range(n_curves):
        t = np.sort(rng.uniform(0, 100, n_points)).astype(np.float32)
        m = (np.sin(2 * np.pi * t / period) + rng.normal(0, 0.1, n_points)).astype(np.float32)
        e = np.full(n_points, 0.1, dtype=np.float32)
        times_list.append(t)
        mags_list.append(m)
        errs_list.append(e)
    return times_list, mags_list, errs_list


def make_periods():
    periods = np.linspace(PERIOD_MIN, PERIOD_MAX, N_PERIODS, dtype=np.float32)
    period_dts = np.array([0.0], dtype=np.float32)
    return periods, period_dts


def create_algo(name, backend_mod):
    """Instantiate an algorithm from the given backend module."""
    cfg = ALGO_CONFIGS[name].copy()
    cls = getattr(backend_mod, {
        "CE": "ConditionalEntropy",
        "AOV": "AOV",
        "LS": "LombScargle",
        "FPW": "FPW",
        "BLS": "BoxLeastSquares",
    }[name])
    return cls(**cfg)


def bench_algo(algo, times, mags, errs, periods, period_dts, name):
    """Run the algorithm and return median wall-clock seconds."""
    # All algorithms use .calc() but with slightly different signatures
    # CE and AOV don't take errs; FPW and BLS do
    needs_errs = name in ("FPW", "BLS")

    # Warmup
    for _ in range(N_WARMUP):
        if needs_errs:
            algo.calc(times, mags, periods, period_dts, errs=errs, output="stats")
        else:
            algo.calc(times, mags, periods, period_dts, output="stats")

    # Timed runs
    elapsed = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        if needs_errs:
            algo.calc(times, mags, periods, period_dts, errs=errs, output="stats")
        else:
            algo.calc(times, mags, periods, period_dts, output="stats")
        elapsed.append(time.perf_counter() - t0)

    return np.median(elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_point_scaling(backend_name, backend_mod):
    """Sweep over point counts at fixed curve count. Return list of result dicts."""
    periods, period_dts = make_periods()
    results = []

    for algo_name in ALGO_CONFIGS:
        try:
            algo = create_algo(algo_name, backend_mod)
        except (ImportError, AttributeError) as e:
            print(f"  [SKIP] {algo_name} on {backend_name}: {e}", file=sys.stderr)
            continue

        for n_pts in N_POINTS:
            times, mags, errs = make_lightcurves(N_CURVES, n_pts)
            try:
                secs = bench_algo(algo, times, mags, errs, periods, period_dts, algo_name)
            except Exception as e:
                print(f"  [ERR] {algo_name}/{backend_name} n={n_pts}: {e}", file=sys.stderr)
                continue

            total_points = N_CURVES * n_pts
            throughput = total_points / secs  # points/sec
            row = {
                "sweep": "points",
                "backend": backend_name,
                "algorithm": algo_name,
                "n_points": n_pts,
                "n_curves": N_CURVES,
                "n_periods": N_PERIODS,
                "wall_sec": round(secs, 4),
                "throughput_pts_per_sec": round(throughput),
            }
            results.append(row)
            print(f"  {algo_name:4s} | {backend_name:4s} | n={n_pts:6d} | "
                  f"{secs:8.4f}s | {throughput:12,.0f} pts/s")

    return results


def run_curve_scaling(backend_name, backend_mod):
    """Sweep over curve counts at fixed point count. Return list of result dicts."""
    periods, period_dts = make_periods()
    results = []

    for algo_name in ALGO_CONFIGS:
        try:
            algo = create_algo(algo_name, backend_mod)
        except (ImportError, AttributeError) as e:
            print(f"  [SKIP] {algo_name} on {backend_name}: {e}", file=sys.stderr)
            continue

        for n_curves in CURVE_COUNTS:
            times, mags, errs = make_lightcurves(n_curves, FIXED_N_POINTS)
            try:
                secs = bench_algo(algo, times, mags, errs, periods, period_dts, algo_name)
            except Exception as e:
                print(f"  [ERR] {algo_name}/{backend_name} curves={n_curves}: {e}",
                      file=sys.stderr)
                continue

            total_points = n_curves * FIXED_N_POINTS
            throughput = total_points / secs  # points/sec
            row = {
                "sweep": "curves",
                "backend": backend_name,
                "algorithm": algo_name,
                "n_points": FIXED_N_POINTS,
                "n_curves": n_curves,
                "n_periods": N_PERIODS,
                "wall_sec": round(secs, 4),
                "throughput_pts_per_sec": round(throughput),
            }
            results.append(row)
            print(f"  {algo_name:4s} | {backend_name:4s} | curves={n_curves:4d} | "
                  f"{secs:8.4f}s | {throughput:12,.0f} pts/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "throughput_results.csv"),
        help="Output CSV path (default: benchmarks/throughput_results.csv)",
    )
    parser.add_argument(
        "--gpu-label", default="GPU",
        help="Label for the GPU backend in the CSV (e.g. '1x P100', '2x P100')",
    )
    parser.add_argument(
        "--gpu-only", action="store_true",
        help="Skip the CPU backend, run GPU benchmarks only",
    )
    args = parser.parse_args()
    csv_path = args.output

    all_results = []

    backends = []
    if not args.gpu_only:
        try:
            from periodfind import cpu as cpu_mod
            backends.append(("CPU", cpu_mod))
        except ImportError as e:
            print(f"CPU backend not available: {e}", file=sys.stderr)
    try:
        from periodfind import gpu as gpu_mod
        backends.append((args.gpu_label, gpu_mod))
    except ImportError as e:
        print(f"GPU backend not available: {e}", file=sys.stderr)

    # Sweep 1: vary point count at fixed curve count
    print("=" * 60)
    print(f"Point-count scaling  (n_curves={N_CURVES})")
    print("=" * 60)
    for name, mod in backends:
        print(f"\n--- {name} backend ---")
        all_results.extend(run_point_scaling(name, mod))

    # Sweep 2: vary curve count at fixed point count
    print()
    print("=" * 60)
    print(f"Curve-count scaling  (n_points={FIXED_N_POINTS})")
    print("=" * 60)
    for name, mod in backends:
        print(f"\n--- {name} backend ---")
        all_results.extend(run_curve_scaling(name, mod))

    # Write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nResults written to {csv_path}")

    return all_results


if __name__ == "__main__":
    main()
