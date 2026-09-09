#!/usr/bin/env python
"""CUDA optimization benchmark for periodfind.

Measures baseline CUDA performance across three workload sizes for each
algorithm (Conditional Entropy, Analysis of Variance, Lomb-Scargle).
Run before and after optimization passes to quantify improvement.

Usage:
    python tests/bench_cuda_opts.py

Requires a CUDA-capable GPU and the periodfind CUDA extensions.
"""

import json
import statistics
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------


def check_gpu():
    """Return True if CUDA extensions load and nvidia-smi succeeds."""
    try:
        from periodfind.gpu import (
            AOV,  # noqa: F401
            ConditionalEntropy,  # noqa: F401
            LombScargle,  # noqa: F401
        )
    except ImportError as exc:
        print(f"ERROR: Could not import CUDA extensions: {exc}")
        print("Build the package with CUDA support first.")
        return False

    try:
        ret = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        if ret.returncode != 0:
            print("ERROR: nvidia-smi returned non-zero exit code.")
            print("No CUDA GPU detected.")
            return False
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. No CUDA GPU detected.")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out. GPU may be in a bad state.")
        return False

    return True


# ---------------------------------------------------------------------------
# Synthetic light-curve generator
# ---------------------------------------------------------------------------


def make_sinusoidal_lightcurve(
    period=5.0, n_points=500, amplitude=1.0, noise_std=0.05, t_span=200.0, seed=42
):
    """Generate a synthetic sinusoidal light curve with a known period.

    Returns (times, mags) as float32 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    mags = (amplitude * np.sin(phase) + amplitude).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def bench(fn, n_warmup=2, n_timed=5):
    """Time *fn* and return (median_ms, min_ms, max_ms).

    Performs *n_warmup* untimed calls, then *n_timed* calls measured with
    time.perf_counter.  Returns the median, min, and max of the timed runs
    in milliseconds.
    """
    for _ in range(n_warmup):
        fn()

    elapsed = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        elapsed.append((t1 - t0) * 1000.0)

    return statistics.median(elapsed), min(elapsed), max(elapsed)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def verify_detected_period(
    algo, algo_name, times_list, mags_list, periods, period_dts, true_period, use_max
):
    """Run the algorithm once and check the detected period is near *true_period*.

    For CE (use_max=False) the best statistic is the minimum; for AOV and LS
    (use_max=True) it is the maximum.  Returns (detected_period, passed).
    """
    result = algo.calc(times_list, mags_list, periods, period_dts, output="periodogram")
    pgram_data = result[0].data  # shape (n_periods, n_pdts)

    # Collapse over period-derivative axis
    if use_max:
        collapsed = pgram_data.max(axis=1)
        best_idx = int(np.argmax(collapsed))
    else:
        collapsed = pgram_data.min(axis=1)
        best_idx = int(np.argmin(collapsed))

    detected = float(periods[best_idx])

    # Allow true period or half/double harmonic, 5 % tolerance
    candidates = [true_period, true_period / 2.0, true_period * 2.0]
    passed = any(abs(detected - c) / c < 0.05 for c in candidates)
    return detected, passed


# ---------------------------------------------------------------------------
# Workload / algorithm definitions
# ---------------------------------------------------------------------------

WORKLOADS = [
    # (label, n_curves, n_points, n_periods, n_pdts)
    ("Small  (1x500x200x1)", 1, 500, 200, 1),
    ("Medium (5x500x1000x3)", 5, 500, 1000, 3),
    ("Large  (20x500x5000x1)", 20, 500, 5000, 1),
]

TRUE_PERIOD = 5.0
T_SPAN = 200.0
N_POINTS = 500


def build_algorithms():
    """Import and instantiate each CUDA algorithm.

    Returns a list of (name, algo_instance, use_max) tuples.
    """
    from periodfind.gpu import AOV, ConditionalEntropy, LombScargle

    return [
        ("CE", ConditionalEntropy(n_phase=10, n_mag=10), False),
        ("AOV", AOV(n_phase=10), True),
        ("LS", LombScargle(), True),
    ]


def build_workload_data(n_curves, n_points, n_periods, n_pdts):
    """Generate light-curve lists and trial-parameter arrays for one workload."""
    times_list = []
    mags_list = []
    for i in range(n_curves):
        t, m = make_sinusoidal_lightcurve(
            period=TRUE_PERIOD,
            n_points=n_points,
            t_span=T_SPAN,
            seed=i,
        )
        times_list.append(t)
        mags_list.append(m)

    periods = np.linspace(1.0, 10.0, n_periods, dtype=np.float32)

    if n_pdts == 1:
        period_dts = np.array([0.0], dtype=np.float32)
    else:
        period_dts = np.linspace(-0.001, 0.001, n_pdts, dtype=np.float32)

    return times_list, mags_list, periods, period_dts


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def run_benchmarks():
    """Run all benchmarks and return a results dict."""
    algorithms = build_algorithms()

    # Table header
    hdr = (
        f"{'Algorithm':<10} | {'Workload':<26} | "
        f"{'Median (ms)':>12} | {'Min (ms)':>10} | {'Max (ms)':>10} | "
        f"{'Period':>8} | {'OK':>4}"
    )
    sep = "-" * len(hdr)
    print()
    print(hdr)
    print(sep)

    results = {}

    for algo_name, algo, use_max in algorithms:
        for wl_label, n_curves, n_points, n_periods, n_pdts in WORKLOADS:
            times_list, mags_list, periods, period_dts = build_workload_data(
                n_curves,
                n_points,
                n_periods,
                n_pdts,
            )

            # -- correctness check ----------------------------------------
            detected, passed = verify_detected_period(
                algo,
                algo_name,
                times_list,
                mags_list,
                periods,
                period_dts,
                TRUE_PERIOD,
                use_max,
            )

            # -- timing ---------------------------------------------------
            def run():
                algo.calc(
                    times_list,
                    mags_list,
                    periods,
                    period_dts,
                    output="stats",
                )

            med_ms, min_ms, max_ms = bench(run, n_warmup=2, n_timed=5)

            # -- record ---------------------------------------------------
            key = f"{algo_name}_{wl_label.strip()}"
            results[key] = {
                "algorithm": algo_name,
                "workload": wl_label.strip(),
                "n_curves": n_curves,
                "n_points": n_points,
                "n_periods": n_periods,
                "n_pdts": n_pdts,
                "median_ms": round(med_ms, 3),
                "min_ms": round(min_ms, 3),
                "max_ms": round(max_ms, 3),
                "detected_period": round(detected, 4),
                "correct": passed,
            }

            ok_str = "PASS" if passed else "FAIL"
            print(
                f"{algo_name:<10} | {wl_label:<26} | "
                f"{med_ms:12.3f} | {min_ms:10.3f} | {max_ms:10.3f} | "
                f"{detected:8.4f} | {ok_str:>4}"
            )

        # Visual separator between algorithms
        print(sep)

    print()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    if not check_gpu():
        sys.exit(1)

    print("=" * 60)
    print("periodfind CUDA baseline benchmark")
    print("=" * 60)
    print(f"True period       : {TRUE_PERIOD}")
    print(f"Points per curve  : {N_POINTS}")
    print(f"Time span         : {T_SPAN}")
    print("Warmup runs       : 2")
    print("Timed runs        : 5")
    print("Timing method     : time.perf_counter, reporting median")

    results = run_benchmarks()

    # Dump JSON for programmatic comparison
    print("Raw results (JSON):")
    print(json.dumps(results, indent=2))

    # Summary: flag any correctness failures
    failures = [k for k, v in results.items() if not v["correct"]]
    if failures:
        print(f"\nWARNING: {len(failures)} workload(s) failed the correctness check:")
        for f in failures:
            r = results[f]
            print(
                f"  {r['algorithm']} / {r['workload']} "
                f"-- detected {r['detected_period']}, "
                f"expected ~{TRUE_PERIOD}"
            )
    else:
        print("\nAll correctness checks passed.")

    return results


if __name__ == "__main__":
    main()
