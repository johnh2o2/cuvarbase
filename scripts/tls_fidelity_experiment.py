"""Apples-to-apples fidelity + timing: cuvarbase fast TLS vs reference
transitleastsquares on the SAME light curves and SAME period grid.

The question is the detection statistic, not just recovery. cuvarbase's
fast path evaluates a COARSE epoch (t0) grid (t0_oversample=3 by
default) plus an exact refinement of the top candidate periods; the
reference steps t0 ~100x finer everywhere. Does coarsening cost SNR?

To compare cleanly we hold the *statistic* fixed: cuvarbase and the
reference define the SR->SDE transform differently, so we recompute SDE
with cuvarbase.tls_stats on BOTH methods' chi2(period) spectra. The
only thing that then varies is the fidelity of the chi2 spectrum. We
also report a definition-free signal strength, the depth SNR at the
recovered period, sqrt(chi2_null - chi2_min).

Runs, on identical injected light curves + one shared Ofir grid:
  - cuvarbase fast, t0_oversample=3   (default)
  - cuvarbase fast, t0_oversample=33  (reference-matched epoch grid)
  - reference transitleastsquares

Usage (GPU pod, batman + transitleastsquares installed):
    python scripts/tls_fidelity_experiment.py [--regime tess-ffi] [--nlc 12]
"""
import argparse
import contextlib
import os
import sys
import time
import warnings
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

import numpy as np

try:  # keep our report lines from being clobbered by the C-ext stdout
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REGIMES = {
    'tess-ffi': dict(ndata=1310, cadence=30. / 60 / 24, noise=1e-3,
                     pinj=7.7, depth=0.005, pmin=0.6, pmax=13.7),
    'k2':       dict(ndata=4320, cadence=30. / 60 / 24, noise=8e-4,
                     pinj=12.4, depth=0.004, pmin=0.6, pmax=45.),
}


def make_lc(c, seed, inject=True):
    rng = np.random.RandomState(seed)
    t = np.arange(c['ndata']) * c['cadence']
    y = 1.0 + rng.randn(c['ndata']) * c['noise']
    if inject:
        q = 0.0763 * c['pinj'] ** (-2.0 / 3.0)
        t0 = 0.3 * c['pinj']
        rel = np.abs(((t - t0 + 0.5 * c['pinj']) % c['pinj'])
                     - 0.5 * c['pinj'])
        y[rel < 0.5 * q * c['pinj']] -= c['depth']
    return t, y, np.full(c['ndata'], c['noise'])


def recovered(p_found, p_inj, tol=0.01):
    for k in (1.0, 2.0, 0.5, 3.0, 1 / 3.0):
        if abs(p_found - k * p_inj) / (k * p_inj) < tol:
            return True
    return False


def sde_identical(chi2, periods):
    """cuvarbase's SDE, applied to any chi2(period) spectrum, so both
    methods are scored by the identical statistic."""
    from cuvarbase import tls_stats
    chi2 = np.asarray(chi2, dtype=float)
    ok = np.isfinite(chi2) & (chi2 < 1e29)
    c = chi2[ok]
    best = int(np.argmin(c))
    stats = tls_stats.compute_all_statistics(
        c, np.asarray(periods)[ok], best, 0.01, 0.1, 10)
    return float(stats['SDE'])


def run_cuvarbase(lcs, periods, t0_oversample):
    import pycuda.driver as cuda
    from cuvarbase.tls import tls_search_batch
    from cuvarbase.base import ensure_context
    ensure_context()
    cuda.Context.synchronize()
    t0 = time.perf_counter()
    res = tls_search_batch(
        lcs, R_star=1.0, M_star=1.0, periods=periods,
        t0_oversample=t0_oversample, refine_top_k=50,
        return_arrays=True)
    cuda.Context.synchronize()
    ms = (time.perf_counter() - t0) / len(lcs) * 1000
    rows = []
    for r in res:
        if 'error' in r:
            rows.append(None); continue
        sde_id = sde_identical(r['chi2'], r['periods'])
        # depth SNR = sqrt(chi2_null - chi2_min); chi2_null ~ max over grid
        cfin = np.asarray(r['chi2'])[np.isfinite(r['chi2'])]
        dsnr = float(np.sqrt(max(cfin.max() - r['chi2_min'], 0.0)))
        rows.append(dict(period=r['period'], sde_native=r['SDE'],
                         sde_id=sde_id, dsnr=dsnr))
    return rows, ms


def run_reference(lcs, periods):
    from transitleastsquares import transitleastsquares
    pmin, pmax = float(periods.min()), float(periods.max())
    rows = []
    t_tot = 0.0
    for (t, y, dy) in lcs:
        model = transitleastsquares(t, y, dy)
        t0 = time.perf_counter()
        with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
            r = model.power(R_star=1.0, M_star=1.0,
                            period_min=pmin, period_max=pmax,
                            oversampling_factor=3, use_threads=cpu_count(),
                            show_progress_bar=False)
        t_tot += time.perf_counter() - t0
        chi2 = np.asarray(getattr(r, 'chi2'))
        pers = np.asarray(getattr(r, 'periods'))
        sde_id = sde_identical(chi2, pers)
        cmin = float(np.nanmin(chi2))
        dsnr = float(np.sqrt(max(np.nanmax(chi2) - cmin, 0.0)))
        rows.append(dict(period=float(r.period), sde_native=float(r.SDE),
                         sde_id=sde_id, dsnr=dsnr))
    return rows, t_tot / len(lcs) * 1000


def report(tag, rows, ms, p_inj, n_inj):
    inj = [r for r in rows[:n_inj] if r]
    rec = sum(recovered(r['period'], p_inj) for r in inj)
    sid = np.median([r['sde_id'] for r in inj])
    snat = np.median([r['sde_native'] for r in inj])
    dsnr = np.median([r['dsnr'] for r in inj])
    print("  %-38s SDE(identical)=%6.2f  SDE(native)=%6.2f  "
          "depthSNR=%5.2f  recov=%d/%d  %8.1f ms/LC"
          % (tag, sid, snat, dsnr, rec, len(inj), ms))
    return dict(tag=tag, sde_id=sid, sde_native=snat, dsnr=dsnr,
                recovered=rec, n=len(inj), ms=ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', default='tess-ffi', choices=list(REGIMES))
    ap.add_argument('--nlc', type=int, default=12)
    ap.add_argument('--depth', type=float, default=None,
                    help='override injection depth (test marginal signals)')
    ap.add_argument('--skip-reference', action='store_true')
    args = ap.parse_args()

    from cuvarbase import tls_grids
    cfg = dict(REGIMES[args.regime])
    if args.depth is not None:
        cfg['depth'] = args.depth
    n_inj = args.nlc
    lcs = [make_lc(cfg, 5000 + i, inject=True) for i in range(n_inj)]
    lcs += [make_lc(cfg, 9000 + i, inject=False) for i in range(3)]

    periods = tls_grids.period_grid_ofir(
        lcs[0][0], R_star=1.0, M_star=1.0, oversampling_factor=3,
        period_min=cfg['pmin'], period_max=cfg['pmax'])
    print("\n=== %s: ndata=%d nperiods=%d P_inj=%.2fd depth=%.4f "
          "(%d inj LCs) ===" % (args.regime, cfg['ndata'], len(periods),
                                cfg['pinj'], cfg['depth'], n_inj))
    print("  SDE(identical) = cuvarbase SDE recomputed on each method's "
          "chi2 spectrum;\n  depthSNR = sqrt(chi2_null - chi2_min) at "
          "the recovered period.\n")

    _ = run_cuvarbase(lcs[:2], periods, 3.0)
    _ = run_cuvarbase(lcs[:2], periods, 33.0)

    out = []
    r, ms = run_cuvarbase(lcs, periods, 3.0)
    out.append(report("cuvarbase t0os=3 (default)", r, ms, cfg['pinj'], n_inj))
    r, ms = run_cuvarbase(lcs, periods, 33.0)
    out.append(report("cuvarbase t0os=33 (matched)", r, ms, cfg['pinj'], n_inj))
    if not args.skip_reference:
        r, ms = run_reference(lcs, periods)
        out.append(report("reference transitleastsquares", r, ms, cfg['pinj'], n_inj))

    ref = next((o for o in out if 'reference' in o['tag']), None)
    if ref:
        print("\n  --- vs reference (identical-SDE basis) ---")
        for o in out:
            if 'cuvarbase' in o['tag']:
                print("  %-32s SDE ratio=%.2f  depthSNR ratio=%.2f  "
                      "speedup=%.0fx"
                      % (o['tag'], o['sde_id'] / ref['sde_id'],
                         o['dsnr'] / ref['dsnr'], ref['ms'] / o['ms']))


if __name__ == '__main__':
    main()
