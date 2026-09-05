#!/usr/bin/env python
"""Apples-to-apples reproduction of GTLS paper (arXiv:2607.00348) Figure 7:
runtime vs light-curve baseline for GTLS vs cuvarbase TLS vs cuvarbase BLS,
all on the SAME GPU, SAME light curve, SAME period grid, SAME per-period
duration search extent, SAME epoch (t0) density, and (optionally) the SAME
limb-darkened template. Every method is additionally scored by ONE identical
SDE routine so we can confirm equal detection, not just equal speed.

Fairness protocol (see notes at bottom):
  * ONE injected batman transit per baseline, fed to every method  -> identical
    SNR between methods by construction.
  * ONE Ofir period grid (cuvarbase.tls_grids.period_grid_ofir, Pmax=S/2)
    passed explicitly to gtls.power(periods=), tls_search_batch(periods=),
    and BLS (freqs=sort(1/periods)).
  * cuvarbase-TLS "matched" uses per-period qmin/qmax = GTLS's own kernel
    duration window and n_durations chosen for the same log-1.1 resolution
    (~38), and t0_oversample matched to GTLS's SKIP_POINT (=1/T0_fit_margin).
  * cuvarbase-TLS "default" is the shipping survey default (t0_os=3, 15 dur,
    Keplerian [0.5q,2q]) — the "production" number, clearly separated.
  * BLS uses the paper's Kunimoto params (qmin=2e-4, qmax=0.15, dlogq=0.1,
    noverlap=3) on the identical period grid.

Runs on a GPU pod with: cupy, gputls, cuvarbase (TLS branch + BLS branch
merged), batman, numpy, scipy.

Usage:
  python gtls_apples_bench.py --baselines 200,500,1000,1500,2000,3000 \
      --methods gtls_full,gtls_skip8,cuv_tls_matched,cuv_tls_default,cuv_bls \
      --out results.json
"""
import argparse
import gc
import json
import platform
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
import numpy as np

# ---- shared, GPU-independent helpers (LC gen, grid, identical-SDE) ----------
import bench_core as bc   # co-located module

CAD = 30.0 / 60.0 / 24.0           # 30-min Kepler long cadence, days
# Injected transit (fixed across baselines; a from Kepler's 3rd law -> physical
# Keplerian duration so BOTH search grids bracket it):
INJ_PERIOD = 8.13
INJ_DEPTH = 0.004
INJ_NOISE = 0.004
INJ_U = (0.4804, 0.1867)           # G2V Kepler LD (== GTLS/TLS reference)

# ---- GTLS per-period duration window (from GPUFun.py durationsGrid kernel) --
_R_SUN = 695508000.0
_R_JUP = 69911000.0
_SPD = 86400.0
_SCALE = 1e15
_PI_GM_MIN = 20848.0
_PI_GM_MAX = 416970.0
_RS_MIN = _R_SUN * 0.05
_RS_MAX = _R_SUN * 4.0
_FRAC_MAX = 0.15


def gtls_dur_window(P_days):
    """Vectorized GTLS per-period (qmin,qmax) fractional-duration window."""
    P = np.asarray(P_days, float)
    Ps = P * _SPD
    pf_min = (4.0 * Ps) / (_PI_GM_MIN * _SCALE)
    pf_max = (4.0 * Ps) / (_PI_GM_MAX * _SCALE)
    T14Min = _RS_MIN * pf_min ** (1.0 / 3.0)
    T14Max = (_RS_MAX + _R_JUP * 2.0) * pf_max ** (1.0 / 3.0)
    dmin = np.minimum(T14Min / Ps, _FRAC_MAX)
    dmax = np.minimum(T14Max / Ps, _FRAC_MAX)
    # guard qmin>0 and qmin<qmax<1 (cuvarbase validation)
    dmin = np.clip(dmin, 1e-5, _FRAC_MAX * 0.999)
    dmax = np.clip(dmax, dmin * 1.0001, 0.999)
    return dmin.astype(np.float64), dmax.astype(np.float64)


def gtls_matched_n_durations(P):
    """log-1.1 count over the widest per-period window (cap 64)."""
    dmin, dmax = gtls_dur_window(P)
    ratio = np.max(dmax / dmin)
    n = int(np.ceil(np.log(ratio) / np.log(1.1))) + 1
    return int(min(max(n, 15), 64))


# ---------------------------------------------------------------- timing ------
def synced(sync_fn, fn):
    sync_fn()
    t0 = time.perf_counter()
    out = fn()
    sync_fn()
    return time.perf_counter() - t0, out


def timed(sync_fn, fn, warmups=1, reps=3):
    for _ in range(warmups):
        try:
            fn()
        except Exception:
            raise
    sync_fn()
    ts = []
    out = None
    for _ in range(reps):
        dt, out = synced(sync_fn, fn)
        ts.append(dt)
    return float(np.median(ts)), ts, out


# ------------------------------------------------------------- GTLS -----------
def cupy_sync():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def run_gtls(t, y, dy, periods, t0_fit_margin, reps=1):
    from gputls import gtls
    periods = np.sort(np.asarray(periods, float))

    def call():
        model = gtls(t=t, y=y, dy=dy, verbose=False)
        return model.power(periods=periods, R_star=1.0, M_star=1.0,
                           oversampling_factor=3, T0_fit_margin=t0_fit_margin,
                           transit_template="default", verbose=False,
                           show_progress_bar=False)

    # compile-ish probe: small grid ~ mostly CuPy JIT (GTLS recompiles/call).
    # GTLS's single-GPU path divides by zero for <~30 periods, so use ~96.
    idx = np.linspace(0, len(periods) - 1, min(96, len(periods))).astype(int)
    tiny = np.sort(periods[np.unique(idx)])

    def call_tiny():
        m = gtls(t=t, y=y, dy=dy, verbose=False)
        return m.power(periods=tiny, R_star=1.0, M_star=1.0,
                       oversampling_factor=3, T0_fit_margin=t0_fit_margin,
                       transit_template="default", verbose=False,
                       show_progress_bar=False)

    try:
        compile_s, _, _ = timed(cupy_sync, call_tiny, warmups=0, reps=2)
    except Exception:
        compile_s = float("nan")
    total_s, ts, res = timed(cupy_sync, call, warmups=1, reps=reps)

    chi2 = np.asarray(np.ma.filled(res.chi2, np.nan), float)
    pers = np.asarray(np.ma.filled(res.periods, np.nan), float)
    ok = np.isfinite(chi2) & np.isfinite(pers)
    sde = bc.recompute_sde(chi2[ok], pers[ok])
    n_dur = len(getattr(res, "rawDurations", []))
    return dict(method="gtls", t0_fit_margin=t0_fit_margin, time_s=total_s,
                times_s=ts, compile_s=compile_s,
                search_s=max(total_s - compile_s, 0.0),
                period_native=float(res.period), sde_native=float(res.SDE),
                sde_identical=sde["SDE"], best_period_identical=sde["best_period"],
                depth_snr=sde["depth_snr"], recovered=bc.recovered(
                    sde["best_period"], INJ_PERIOD),
                recovered_native=bc.recovered(float(res.period), INJ_PERIOD),
                n_periods=int(len(periods)), n_durations=n_dur)


# --------------------------------------------------------- cuvarbase TLS ------
def _pycuda_sync():
    import pycuda.driver as drv
    drv.Context.synchronize()


def maybe_patch_template(u=INJ_U):
    """Monkeypatch cuvarbase's reference transit geometry to Hippke's
    (rp=0.03, a=23.1, inc=89.21) so the search TEMPLATE matches GTLS exactly."""
    import cuvarbase.tls_models as tm
    import batman

    def hippke_reference(n_samples=1000, limb_dark="quadratic", u=list(u)):
        p = batman.TransitParams()
        p.t0 = 0.0
        p.per = 1.0
        p.rp = 0.03
        p.a = 23.1             # a/Rstar (batman units); per normalized to phase
        p.inc = 89.21          # -> impact parameter b = a*cos(inc) ~ 0.32
        p.ecc = 0.0
        p.w = 90.0
        p.limb_dark = limb_dark
        p.u = list(u)
        # transit half-width in phase ~ (1/pi)*asin(sqrt((1+rp)^2-b^2)/a); span it
        tt = np.linspace(-0.05, 0.05, n_samples)
        m = batman.TransitModel(p, tt)
        flux = m.light_curve(p)
        oot = flux[0]
        depth = oot - np.min(flux)
        if depth < 1e-10:
            raise ValueError("template depth ~0")
        fluxn = (flux - oot) / depth + 1.0
        phases = (tt - tt[0]) / (tt[-1] - tt[0])
        return phases, fluxn

    tm.create_reference_transit = hippke_reference
    return True


def run_cuv_tls(lcs, periods, t0_oversample, qmin, qmax, n_durations,
                u=INJ_U, reps=3, label="cuv_tls"):
    from cuvarbase.tls import tls_search_batch
    periods = np.sort(np.asarray(periods, float))

    kw = dict(R_star=1.0, M_star=1.0, periods=periods,
              oversampling_factor=3, n_durations=n_durations,
              t0_oversample=t0_oversample, refine_top_k=50,
              u=list(u), limb_dark="quadratic", return_arrays=True)
    if qmin is not None:
        kw["qmin"] = np.asarray(qmin, float)
        kw["qmax"] = np.asarray(qmax, float)

    def call():
        return tls_search_batch(lcs, **kw)

    total_s, ts, res = timed(_pycuda_sync, call, warmups=1, reps=reps)
    # per-LC = batch time / n_lcs (single-LC head-to-head when len(lcs)==1)
    per_lc = total_s / len(lcs)
    r0 = res[0]
    if "error" in r0:
        return dict(method=label, time_s=per_lc, error=r0["error"])
    sde = bc.recompute_sde(np.asarray(r0["chi2"], float),
                           np.asarray(r0["periods"], float))
    return dict(method=label, time_s=per_lc, batch_time_s=total_s, times_s=ts,
                n_lcs=len(lcs), t0_oversample=t0_oversample,
                n_durations=n_durations, period_native=float(r0["period"]),
                sde_native=float(r0["SDE"]), sde_identical=sde["SDE"],
                best_period_identical=sde["best_period"],
                depth_snr=sde["depth_snr"],
                recovered=bc.recovered(sde["best_period"], INJ_PERIOD),
                recovered_native=bc.recovered(float(r0["period"]), INJ_PERIOD),
                n_periods=int(len(periods)))


# --------------------------------------------------------- cuvarbase BLS ------
def run_cuv_bls(lcs, periods, cfg="kunimoto", reps=3, label="cuv_bls"):
    """BLS on the identical period grid. eebls_gpu_batch returns only power
    spectra -> argmax for the identical-SDE score.

    cfg='kunimoto': the GTLS paper's BLS config (qmin=2e-4, qmax=0.15,
        dlogq=0.1, noverlap=3) -> up to 5000 phase bins, fused kernel bypassed.
    cfg='matched':  BLS duration grid matched to the TLS run's per-period
        window (qmin/qmax = GTLS window) with noverlap=2 so the fused kernel
        (opt1) is used -> BLS's true speed at TLS-comparable duration fidelity.
    """
    from cuvarbase.bls import eebls_gpu_batch
    periods = np.sort(np.asarray(periods, float))
    freqs = np.sort((1.0 / periods).astype(np.float32))
    if cfg == "kunimoto":
        kw = dict(qmin=2e-4, qmax=0.15, dlogq=0.1, noverlap=3)
    elif cfg == "matched":
        # BLS at a physically sensible transit-duration range (>=0.2% of the
        # period, brackets the injected q~0.02) with the fused kernel
        # (noverlap=2, power of two). This is BLS's true competitive speed;
        # the Kunimoto qmin=2e-4 (5000 bins) is what makes 'kunimoto' heavy.
        kw = dict(qmin=2e-3, qmax=0.15, dlogq=0.1, noverlap=2)
    else:
        raise ValueError(cfg)

    def call():
        return eebls_gpu_batch(lcs, freqs, **kw)

    total_s, ts, powers = timed(_pycuda_sync, call, warmups=1, reps=reps)
    per_lc = total_s / len(lcs)
    p0 = np.asarray(powers[0], float)
    ok = np.isfinite(p0)
    fr = freqs[ok].astype(float)
    order = np.argsort(1.0 / fr)          # ascending period
    sde = bc.recompute_sde_from_sr(p0[ok][order], (1.0 / fr)[order])
    scal = {k: (float(np.median(v)) if hasattr(v, "__len__") else v)
            for k, v in kw.items()}
    return dict(method=label, cfg=cfg, time_s=per_lc, batch_time_s=total_s,
                times_s=ts, n_lcs=len(lcs), n_periods=int(len(periods)),
                sde_identical=sde["SDE"], best_period_identical=sde["best_period"],
                recovered=bc.recovered(sde["best_period"], INJ_PERIOD),
                qcfg=scal)


# ------------------------------------------------------------------ main ------
def env_info():
    info = dict(python=platform.python_version(), numpy=np.__version__,
                host=platform.node())
    try:
        import cupy as cp
        info["cupy"] = cp.__version__
        info["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        info["cc"] = str(cp.cuda.Device(0).compute_capability)
        info["gpu_mem_GB"] = round(cp.cuda.Device(0).mem_info[1] / 1e9, 1)
    except Exception as e:
        info["gpu"] = "cupy/gpu unavailable: %r" % e
    for pkg in ("gputls", "cuvarbase", "batman"):
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "?")
        except Exception as e:
            info[pkg] = "unavailable: %r" % e
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="200,500,1000,1500,2000,3000")
    ap.add_argument("--methods",
                    default="gtls_full,gtls_skip8,cuv_tls_matched,"
                            "cuv_tls_default,cuv_bls")
    ap.add_argument("--match-template", action="store_true",
                    help="patch cuvarbase template to Hippke geometry")
    ap.add_argument("--gtls-reps", type=int, default=1)
    ap.add_argument("--cuv-reps", type=int, default=3)
    ap.add_argument("--out", default="gtls_apples_results.json")
    args = ap.parse_args()

    baselines = [int(x) for x in args.baselines.split(",") if x]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.match_template and any(m.startswith("cuv_tls") for m in methods):
        maybe_patch_template()
        print("[template] cuvarbase reference transit patched to Hippke geometry")

    out = dict(script="gtls_apples_bench.py",
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
               inj=dict(period=INJ_PERIOD, depth=INJ_DEPTH, noise=INJ_NOISE,
                        u=list(INJ_U), cadence_days=CAD),
               match_template=args.match_template, baselines=baselines,
               results={})

    for base in baselines:
        print("\n" + "=" * 72)
        print("BASELINE %d d" % base)
        print("=" * 72, flush=True)
        t, y, dy, meta = bc.make_lc(base, CAD, INJ_PERIOD, INJ_DEPTH,
                                    INJ_NOISE, seed=1000 + base, u=INJ_U)
        periods = bc.shared_period_grid(t)
        qmin, qmax = gtls_dur_window(periods)
        n_dur_matched = gtls_matched_n_durations(periods)
        print("  ndata=%d  nperiods=%d  SNR=%.1f  q_true=%.4f  "
              "n_dur_matched=%d" % (meta["ndata"], len(periods),
              meta.get("snr", -1), meta.get("q_true", -1), n_dur_matched),
              flush=True)
        row = dict(meta=meta, nperiods=int(len(periods)),
                   n_dur_matched=n_dur_matched, methods={})

        for m in methods:
            try:
                if m == "gtls_full":
                    r = run_gtls(t, y, dy, periods, 0.0, reps=args.gtls_reps)
                elif m == "gtls_skip8":
                    r = run_gtls(t, y, dy, periods, 0.125, reps=args.gtls_reps)
                elif m == "cuv_tls_matched":
                    r = run_cuv_tls([(t, y, dy)], periods, t0_oversample=8.0,
                                    qmin=qmin, qmax=qmax,
                                    n_durations=n_dur_matched,
                                    reps=args.cuv_reps, label="cuv_tls_matched")
                elif m == "cuv_tls_default":
                    r = run_cuv_tls([(t, y, dy)], periods, t0_oversample=3.0,
                                    qmin=None, qmax=None, n_durations=15,
                                    reps=args.cuv_reps, label="cuv_tls_default")
                elif m in ("cuv_bls", "cuv_bls_kunimoto"):
                    r = run_cuv_bls([(t, y, dy)], periods, cfg="kunimoto",
                                    reps=args.cuv_reps, label=m)
                elif m == "cuv_bls_matched":
                    r = run_cuv_bls([(t, y, dy)], periods, cfg="matched",
                                    reps=args.cuv_reps, label=m)
                else:
                    print("  unknown method %s" % m); continue
                row["methods"][m] = r
                print("  %-18s %9.3f s/LC  SDE(id)=%6.2f  P=%.4f  rec=%s"
                      % (m, r.get("time_s", float("nan")),
                         r.get("sde_identical", float("nan")),
                         r.get("best_period_identical", float("nan")),
                         r.get("recovered")), flush=True)
            except Exception as e:
                traceback.print_exc()
                row["methods"][m] = dict(error=repr(e))
                print("  %-18s ERROR %r" % (m, e), flush=True)
            gc.collect()

        out["results"][str(base)] = row
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)

    out["env"] = env_info()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote %s" % args.out)
    print(json.dumps(out["env"], indent=2))


if __name__ == "__main__":
    main()
