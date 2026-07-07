"""Cold single-shot timing: one light curve, one fresh process, NO warmup, so
the kernel JIT compile is INCLUDED for both cuvarbase and GTLS. The CUDA context
is initialized before the clock starts (a fixed driver cost both pay), so the
measured number is compile + search — the true one-star cold cost. The launching
shell also times the whole process (python import + context init + this).

Standalone Ofir grid + LC (numpy/batman only) so the GTLS process never imports
cuvarbase (fair: each loads only its own stack).

Usage: python cold_shot.py --method {gtls_skip8|cuv_tls_matched|cuv_tls_default} --baseline 1500
"""
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np

G = 6.67430e-11; RSUN = 6.957e8; MSUN = 1.9884e30; SPD = 86400.0; RJUP = 6.9911e7


def ofir_grid(t, os=3, pmin=0.6, n_transits_min=2):
    T = (t.max() - t.min()) * SPD
    fmin = n_transits_min / T
    fmax = 1 / (2 * np.pi) * np.sqrt(G * MSUN / (3 * RSUN) ** 3)
    A = (2 * np.pi) ** (2 / 3) / np.pi * RSUN / (G * MSUN) ** (1 / 3) / (T * os)
    C = fmin ** (1 / 3) - A / 3
    n = int(np.ceil((fmax ** (1 / 3) - fmin ** (1 / 3) + A / 3) * 3 / A))
    x = np.arange(n) + 1
    per = 1 / ((A / 3 * x + C) ** 3) / SPD
    return np.sort(per[per > pmin])


def make_lc(baseline, cad=30 / 60 / 24, P=8.13, depth=4e-3, noise=4e-3, seed=1):
    import batman
    rng = np.random.RandomState(seed)
    n = int(round(baseline / cad)); t = np.arange(n) * cad
    y = 1 + rng.randn(n) * noise; dy = np.full(n, noise)
    a = (G * MSUN * (P * SPD) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / RSUN
    pm = batman.TransitParams()
    pm.t0 = 0.35 * P; pm.per = P; pm.rp = float(np.sqrt(depth)); pm.a = float(a)
    pm.inc = 90; pm.ecc = 0; pm.w = 90; pm.u = [0.4804, 0.1867]
    pm.limb_dark = "quadratic"
    y = y + (batman.TransitModel(pm, t).light_curve(pm) - 1)
    return t, y, dy


def gtls_qwin(P):
    Ps = P * SPD
    pfmin = 4 * Ps / (20848 * 1e15); pfmax = 4 * Ps / (416970 * 1e15)
    dmin = np.minimum((RSUN * 0.05) * pfmin ** (1 / 3) / Ps, 0.15)
    dmax = np.minimum((RSUN * 4.0 + 2 * RJUP) * pfmax ** (1 / 3) / Ps, 0.15)
    dmin = np.clip(dmin, 1e-5, 0.15 * 0.999); dmax = np.clip(dmax, dmin * 1.0001, 0.999)
    return dmin, dmax


ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True)
ap.add_argument("--baseline", type=int, required=True)
args = ap.parse_args()

t, y, dy = make_lc(args.baseline)
periods = ofir_grid(t)

if args.method.startswith("gtls"):
    import cupy as cp
    cp.arange(1).sum(); cp.cuda.Stream.null.synchronize()   # init context (untimed)
    from gputls import gtls
    t0fit = 0.0 if args.method == "gtls_full" else 0.125
    c0 = time.perf_counter()
    res = gtls(t=t, y=y, dy=dy, verbose=False).power(
        periods=periods, R_star=1, M_star=1, oversampling_factor=3,
        T0_fit_margin=t0fit, verbose=False, show_progress_bar=False)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - c0
    print("RESULT %s %d nper=%d search_compile_s=%.3f P=%.4f SDE=%.2f"
          % (args.method, args.baseline, len(periods), dt, res.period, res.SDE))
else:
    import pycuda.autoprimaryctx            # init context at import (untimed)
    import pycuda.driver as drv
    drv.Context.synchronize()
    from cuvarbase.tls import tls_search_batch
    if args.method == "cuv_tls_matched":
        qmn, qmx = gtls_qwin(periods)
        kw = dict(qmin=qmn, qmax=qmx, n_durations=38, t0_oversample=8.0)
    else:
        kw = dict(n_durations=15, t0_oversample=3.0)
    c0 = time.perf_counter()
    res = tls_search_batch([(t, y, dy)], R_star=1, M_star=1, periods=periods,
                           oversampling_factor=3, refine_top_k=50,
                           u=[0.4804, 0.1867], limb_dark="quadratic",
                           return_arrays=False, **kw)[0]
    drv.Context.synchronize()
    dt = time.perf_counter() - c0
    print("RESULT %s %d nper=%d search_compile_s=%.3f P=%.4f SDE=%.2f"
          % (args.method, args.baseline, len(periods), dt,
             res.get("period", -1), res.get("SDE", -1)))
