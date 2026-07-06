"""Smoke + parity test for the fast TLS path (run on a GPU pod).

Checks, in order:
1. The fast kernels compile.
2. Fast path vs legacy kernel on the same explicit trial grid:
   chi2 spectra strongly correlated, same best period, similar SDE.
3. Fast path recovers an injected transit (period + SDE), single LC.
4. Batch of mixed lightcurves: per-LC results match single-LC calls.
5. Large-ndata lightcurve (beyond the legacy 3,500-point cap) works.
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*EXPERIMENTAL.*')

from cuvarbase import tls
from cuvarbase import tls_grids


def make_lc(ndata, baseline, period, depth, noise, seed, t0_frac=0.3):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata))
    y = 1.0 + rng.randn(ndata) * noise
    q = 0.0763 * period ** (-2.0 / 3.0)
    t0 = t0_frac * period
    rel = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
    y[rel < 0.5 * q * period] -= depth
    dy = np.full(ndata, noise)
    return t, y, dy


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[%s] %s %s" % (status, name, detail))
    if not cond:
        check.failures += 1


check.failures = 0


def main():
    # ---------------- 1. compile ----------------
    t_start = time.time()
    kernels = tls.compile_tls_fast(block_size=128, nbins=1024)
    check("compile", set(kernels) == {'search', 'refine'},
          "(%.1fs)" % (time.time() - t_start))

    # ---------------- 2. parity vs legacy ----------------
    ndata, baseline = 1200, 27.0
    P_inj, depth = 5.123, 0.01
    t, y, dy = make_lc(ndata, baseline, P_inj, depth, 2e-3, seed=42)

    periods = tls_grids.period_grid_ofir(
        t, R_star=1.0, M_star=1.0, oversampling_factor=3,
        period_min=1.0, period_max=12.0).astype(np.float64)
    _, _, qv = tls_grids.duration_grid_keplerian(
        periods, R_star=1.0, M_star=1.0, R_planet=1.0,
        qmin_fac=0.5, qmax_fac=2.0, n_durations=15)
    qmin, qmax = qv * 0.5, qv * 2.0

    t0 = time.time()
    r_old = tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=qmin, qmax=qmax, n_durations=15,
                               use_fast=False)
    t_old = time.time() - t0

    t0 = time.time()
    r_new = tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=qmin, qmax=qmax, n_durations=15,
                               use_fast=True)
    t_new = time.time() - t0

    c_old = r_old['chi2']
    c_new = r_new['chi2']
    both = np.isfinite(c_old) & np.isfinite(c_new)
    corr = np.corrcoef(c_old[both], c_new[both])[0, 1]
    check("parity/chi2-corr", corr > 0.99, "corr=%.5f" % corr)
    check("parity/best-period",
          abs(r_new['period'] - r_old['period']) / r_old['period'] < 0.01,
          "old=%.4f new=%.4f" % (r_old['period'], r_new['period']))
    check("parity/period-hit",
          abs(r_new['period'] - P_inj) / P_inj < 0.01,
          "P=%.4f (inj %.4f)" % (r_new['period'], P_inj))
    check("parity/SDE", r_new['SDE'] > 0.8 * r_old['SDE'],
          "old=%.2f new=%.2f" % (r_old['SDE'], r_new['SDE']))
    check("parity/depth",
          abs(r_new['depth'] - depth) / depth < 0.5,
          "depth=%.4f" % r_new['depth'])
    med_old = np.median(c_old[both])
    med_new = np.median(c_new[both])
    check("parity/chi2-scale", abs(med_new / med_old - 1) < 0.05,
          "median old=%.1f new=%.1f" % (med_old, med_new))
    print("       timing: legacy %.3fs, fast %.3fs (%.1fx)"
          % (t_old, t_new, t_old / max(t_new, 1e-9)))

    # ---------------- 3. auto-grid recovery ----------------
    res = tls.tls_transit(t, y, dy, R_star=1.0, M_star=1.0,
                          period_min=1.0, period_max=12.0)
    check("auto/period", abs(res['period'] - P_inj) / P_inj < 0.01,
          "P=%.4f SDE=%.2f" % (res['period'], res['SDE']))
    check("auto/SDE", res['SDE'] > 5.0, "SDE=%.2f" % res['SDE'])

    # ---------------- 4. batch consistency ----------------
    lcs = []
    P_injs = [3.3, 7.7, 0.0]  # third LC = pure noise
    for i, P in enumerate(P_injs):
        if P > 0:
            lcs.append(make_lc(1500 + 400 * i, 27.0, P, 0.012, 2e-3,
                               seed=100 + i))
        else:
            rng = np.random.RandomState(100 + i)
            tt = np.sort(rng.uniform(0, 27.0, 1500 + 400 * i))
            lcs.append((tt, 1.0 + rng.randn(len(tt)) * 2e-3,
                        np.full(len(tt), 2e-3)))

    # shared explicit grid so batch and single calls are comparable
    # (auto grids depend on each lightcurve's exact baseline)
    tspan_max = max(lc[0].max() - lc[0].min() for lc in lcs)
    t_ref = [lc for lc in lcs
             if lc[0].max() - lc[0].min() == tspan_max][0][0]
    shared_periods = tls_grids.period_grid_ofir(
        t_ref, R_star=1.0, M_star=1.0, oversampling_factor=3,
        period_min=1.0, period_max=12.0)

    batch = tls.tls_search_batch(lcs, R_star=1.0, M_star=1.0,
                                 periods=shared_periods)
    singles = [tls.tls_search_batch([lc], R_star=1.0, M_star=1.0,
                                    periods=shared_periods)[0]
               for lc in lcs]
    for i, (b, s) in enumerate(zip(batch, singles)):
        if P_injs[i] > 0:
            # non-deterministic atomics can flip near-tied neighboring
            # grid points; allow a few grid steps of slack
            check("batch/lc%d-period-match" % i,
                  abs(b['period'] - s['period']) / s['period'] < 5e-3,
                  "batch=%.5f single=%.5f" % (b['period'], s['period']))
            check("batch/lc%d-recovered" % i,
                  abs(b['period'] - P_injs[i]) / P_injs[i] < 0.01,
                  "P=%.4f SDE=%.2f" % (b['period'], b['SDE']))
        else:
            check("batch/lc%d-noise-SDE-consistent" % i,
                  abs(b['SDE'] - s['SDE']) < 1.5,
                  "batch=%.2f single=%.2f" % (b['SDE'], s['SDE']))
    sde_noise = batch[2]['SDE']
    sde_sig = batch[0]['SDE']
    check("batch/noise-SDE-lower", sde_noise < sde_sig,
          "sig=%.2f noise=%.2f" % (sde_sig, sde_noise))

    # ---------------- 5. large ndata (legacy cap exceeded) ----------------
    t5, y5, dy5 = make_lc(20000, 27.0, 4.56, 0.008, 2e-3, seed=7)
    t0 = time.time()
    r5 = tls.tls_search_batch([(t5, y5, dy5)], R_star=1.0, M_star=1.0,
                              period_min=1.0, period_max=12.0)[0]
    dt5 = time.time() - t0
    check("large/period", abs(r5['period'] - 4.56) / 4.56 < 0.01,
          "P=%.4f SDE=%.2f (%.2fs)" % (r5['period'], r5['SDE'], dt5))

    # BJD-scale time offsets
    r6 = tls.tls_search_batch([(t5 + 2457000.0, y5, dy5)],
                              R_star=1.0, M_star=1.0,
                              period_min=1.0, period_max=12.0)[0]
    check("large/bjd-offset", abs(r6['period'] - 4.56) / 4.56 < 0.01,
          "P=%.4f SDE=%.2f" % (r6['period'], r6['SDE']))

    print()
    if check.failures:
        print("%d FAILURES" % check.failures)
        sys.exit(1)
    print("ALL SMOKE CHECKS PASSED")


if __name__ == '__main__':
    main()
