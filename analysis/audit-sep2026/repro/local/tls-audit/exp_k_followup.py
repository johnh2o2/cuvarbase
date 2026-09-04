"""Experiment K: follow-ups. (a) baseline-offset sensitivity in the reference; (b) short-duration/M-dwarf case with finer t0;
(c) SR definition effect on strong signals; (d) descending period grid; (e) odd_even ddof; (f) SDE vs period-range."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu, tls_search_batch
from cuvarbase import tls_grids, tls_stats
from transitleastsquares.stats import spectra
warnings.simplefilter('ignore')

print("--- (a) baseline offset: reference vs cuvarbase (P=7.3 rp=0.05 sigma=1e-3, 60 d) ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
for off in [0.0, -5e-4, +5e-4]:
    ref, dt = run_ref(t, y * (1 + off), dy, threads=24)
    periods = np.sort(ref.periods)
    r = tls_search_gpu(t, y * (1 + off), dy, periods=periods)
    print("offset %+.0e: ref SDE=%.2f P=%.4f depth=%.4f | cuv SDE=%.2f P=%.4f depth=%.4f n_failed=%d [%ds]" % (off, ref.SDE, ref.period, 1 - ref.depth, r['SDE'], r['period'], r['depth'], r['n_failed_periods'], dt))

print("--- (b) M-dwarf short-duration case (q=0.0097, ~3.4 pts/transit): t0_oversample and refine ---")
t, y, dy, info = make_lc(7.3, 0.08, 2.0, baseline=90.0, sigma=1e-3, seed=7, R_star=0.3, M_star=0.3)
periods = tls_grids.period_grid_ofir(np.arange(0, 90.0, 1/48))
for kw in [dict(), dict(t0_oversample=10.0), dict(t0_oversample=33.0), dict(t0_oversample=33.0, nbins=8192)]:
    r = tls_search_gpu(t, y, dy, periods=periods, **kw)
    print("   std window %-40s SDE=%.2f SDE_raw=%.2f depth=%.4f (true %.4f) dur=%.3f (true %.3f)" % (kw, r['SDE'], r['SDE_raw'], r['depth'], info['true_depth'], r['duration'], info['t14']))
q = tls_grids.q_transit(periods, R_star=0.3, M_star=0.3)
r = tls_search_gpu(t, y, dy, periods=periods, qmin=0.5*q, qmax=2*q, R_star=0.3, M_star=0.3)
print("   correct-star Keplerian window: SDE=%.2f depth=%.4f dur=%.3f" % (r['SDE'], r['depth'], r['duration']))
r = tls_search_gpu(t, y, dy, periods=periods, qmin=0.5*q, qmax=2*q, R_star=0.3, M_star=0.3, t0_oversample=33.0)
print("   correct-star Keplerian window, t0_os=33: SDE=%.2f depth=%.4f dur=%.3f" % (r['SDE'], r['depth'], r['duration']))

print("--- (c) SR definition on strong signal (P=3 rp=0.10): cuvarbase SDE vs reference spectra() on the same chi2 ---")
t, y, dy, info = make_lc(3.0, 0.10, 1.11, baseline=60.0, sigma=1e-3, seed=hash('P3_rp0.10_b0') % 1000)
periods = tls_grids.period_grid_ofir(t)
r = tls_search_gpu(t, y, dy, periods=periods, return_arrays=True)
chi2 = r['chi2']; m = np.isfinite(chi2)
SR, praw, pw, SDEr, SDE = spectra(chi2[m], 3)
SDEc, SDEc_raw, powc = tls_stats.signal_detection_efficiency(chi2[m])
print("   cuvarbase SDE=%.2f (raw %.2f) | reference spectra() on cuvarbase chi2: SDE=%.2f (raw %.2f) | chi2_min=%.0f chi2_max=%.0f score_best/chi2_0=%.2f" % (SDEc, SDEc_raw, SDE, SDEr, np.nanmin(chi2), np.nanmax(chi2), (np.nanmax(chi2)-np.nanmin(chi2))/np.nanmax(chi2)))
# medfilt zero-padding vs running_median edge extension: power at the grid edges under the two conventions
from scipy import signal
SRc = tls_stats.signal_residue(chi2[m])
trend_scipy = signal.medfilt(SRc, 91)
from transitleastsquares.helpers import running_median
trend_ref = running_median(SRc, 91)
print("   SR trend at edges: scipy medfilt(zero-pad) first5=%s last5=%s ; running_median first/last=%.3e/%.3e ; typical SR=%.3e" % (np.round(trend_scipy[:5], 5), np.round(trend_scipy[-5:], 5), trend_ref[0], trend_ref[-1], np.median(SRc)))

print("--- (d) descending / unsorted period grid ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
periods = tls_grids.period_grid_ofir(t)
ra = tls_search_gpu(t, y, dy, periods=periods)
rd = tls_search_gpu(t, y, dy, periods=periods[::-1].copy())
rng = np.random.RandomState(0); sh = periods.copy(); rng.shuffle(sh)
rs = tls_search_gpu(t, y, dy, periods=sh)
print("   ascending: P=%.4f SDE=%.2f unc=%.5f | descending: P=%.4f SDE=%.2f unc=%.5f | shuffled: P=%.4f SDE=%.2f unc=%.5f" % (ra['period'], ra['SDE'], ra['period_uncertainty'], rd['period'], rd['SDE'], rd['period_uncertainty'], rs['period'], rs['SDE'], rs['period_uncertainty']))

print("--- (e) odd_even_mismatch with 2+2 transits ---")
print("   ", tls_stats.odd_even_mismatch([0.010, 0.012], [0.010, 0.012]), tls_stats.odd_even_mismatch([0.010, 0.010], [0.012, 0.012]))

print("--- (f) SDE of the same signal vs period-range / grid size (fast path, P=7.3 rp=0.05) ---")
for pmin, pmax in [(6.5, 8.0), (3.0, 15.0), (0.6, 30.0), (1.0, 30.0)]:
    per = tls_grids.period_grid_ofir(t, period_min=pmin, period_max=pmax)
    r = tls_search_gpu(t, y, dy, periods=per)
    print("   range [%.1f, %.1f] n=%d: SDE=%.2f SDE_raw=%.2f FAP=%.2g SNR=%.1f" % (pmin, pmax, len(per), r['SDE'], r['SDE_raw'], r['FAP'], r['SNR']))
