"""Experiment D: T0 return semantics; flat/offset/gaps edge cases; period uncertainty; pink noise."""
import numpy as np, sys, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu, tls_search_batch, tls_transit
from cuvarbase import tls_grids, tls_stats

P, rp = 3.0, 0.08
print("--- T0 semantics ---")
for t_start in [0.0, 100.3, 2457000.3]:
    t0_true = t_start + 0.37 * P
    t, y, dy, info = make_lc(P, rp, t0_true, baseline=40.0, sigma=1e-3, seed=3, t_start=t_start)
    periods = np.linspace(2.9, 3.1, 300)
    rf = tls_search_gpu(t, y, dy, periods=periods)
    rb = tls_search_batch([(t, y, dy)], periods=periods)[0]
    line = "t_start=%.1f true_t0=%.4f | fast T0=%.4f  batch t0_phase=%.4f batch T0=%.4f" % (t_start, t0_true, rf['T0'], rb['t0_phase'], rb['T0'])
    if t_start < 1e4:
        rl = tls_search_gpu(t, y, dy, periods=periods, use_fast=False)
        line += " | legacy T0=%.4f (phase of t/P: true=%.4f)" % (rl['T0'], (t0_true / P) % 1)
    epoch = np.floor(t.min())
    line += " | phase rel. to epoch floor(min t)=%.1f: true=%.4f" % (epoch, ((t0_true - epoch) / P) % 1)
    print(line)
    # is the returned batch T0 a real transit time?
    print("   batch T0 - nearest true transit: %.5f d (duration %.3f d)" % (np.min(np.abs(rb['T0'] - (t0_true + P * np.arange(-2, 20)))), info['t14']))

print("--- flat noiseless input ---")
t = np.arange(2000) / 48.0; y = np.ones_like(t); dy = 1e-3 * np.ones_like(t)
try:
    r = tls_search_gpu(t, y, dy, periods=np.linspace(1, 5, 200)); print("fast flat: SDE=%s n_failed=%s" % (r['SDE'], r['n_failed_periods']))
except Exception as e: print("fast flat ->", type(e).__name__, e)
r = tls_search_batch([(t, y, dy)], periods=np.linspace(1, 5, 200)); print("batch flat ->", r[0])
try:
    r = tls_search_gpu(t, y, dy, periods=np.linspace(1, 5, 200), use_fast=False); print("legacy flat: SDE=%s" % r['SDE'])
except Exception as e: print("legacy flat ->", type(e).__name__, e)

print("--- baseline offset sensitivity (sigma=1e-3, rp=0.05, P=7.3, 60 d) ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
periods = tls_grids.period_grid_ofir(t)
for off in [0.0, 2e-4, 5e-4, 1e-3, 2e-3, -5e-4, -1e-3]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = tls_search_gpu(t, y * (1 + off), dy, periods=periods)
    print("offset %+.1e: SDE=%.2f depth=%.4f (true %.4f) period=%.4f n_failed=%d/%d" % (off, r['SDE'], r['depth'], info['true_depth'], r['period'], r['n_failed_periods'], len(periods)))
print("--- data gap (30%% missing in the middle) ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
m = (t < 20) | (t > 38)
r = tls_search_gpu(t[m], y[m], dy[m], periods=periods)
print("gap: SDE=%.2f period=%.4f depth=%.4f n_failed=%d" % (r['SDE'], r['period'], r['depth'], r['n_failed_periods']))
print("--- period uncertainty ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
r = tls_search_gpu(t, y, dy, periods=periods, return_arrays=True)
i = int(np.nanargmin(r['chi2'])); dP = np.diff(periods)[i]
print("period=%.5f unc=%.5f grid spacing there=%.5f chi2 around min: %s" % (r['period'], r['period_uncertainty'], dP, np.round(r['chi2'][i-3:i+4] - np.nanmin(r['chi2']), 1)))
print("--- pink_noise_correction ignores n_transits:", tls_stats.pink_noise_correction(10.0, 1), tls_stats.pink_noise_correction(10.0, 100), tls_stats.pink_noise_correction(10.0, 100, correlation_length=4))
print("--- n_transits reported vs actual ---")
intr = np.abs((((t - 2.0)/7.3 + 0.5) % 1) - 0.5) * 7.3 < info['t14']/2
print("n_transits=%d (span/P=%.2f; actual distinct transits with data: %d)" % (r['n_transits'], (t.max()-t.min())/7.3, len(np.unique(np.round((t[intr] - 2.0)/7.3)))))
print("--- SNR definition: chi2_null=max(chi2 over grid) vs chi2_0 ---")
chi2 = r['chi2']; print("SNR=%.2f sqrt(max(chi2)-chi2_best)=%.2f ; sqrt(chi2_0 - chi2_best) where chi2_0=sum((1-y)^2/dy^2)=%.2f" % (r['SNR'], np.sqrt(np.nanmax(chi2) - chi2[i]), np.sqrt(np.sum((1-y)**2/dy**2) - chi2[i])))
print("--- tls_transit (Keplerian) on the same LC ---")
rt = tls_transit(t, y, dy, period_min=1.0, period_max=30.0)
print("tls_transit: period=%.4f SDE=%.2f depth=%.4f dur=%.3f (true t14=%.3f) T0=%.4f (true t0=2.0)" % (rt['period'], rt['SDE'], rt['depth'], rt['duration'], info['t14'], rt['T0']))
