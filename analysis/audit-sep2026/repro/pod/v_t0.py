"""Verifier reproduction for tls-T0: what does 'T0' mean on each TLS path?"""
import numpy as np, sys, warnings
sys.path.insert(0, '/workspace/scratch')
warnings.simplefilter('ignore')
from audit_common import make_lc
from cuvarbase.tls import tls_search_gpu, tls_search_batch, tls_transit

P, rp = 3.0, 0.08
periods = np.linspace(2.9, 3.1, 300)

def row(t_start, frac_phase):
    t0_true = t_start + frac_phase * P
    t, y, dy, info = make_lc(P, rp, t0_true, baseline=40.0, sigma=1e-3, seed=3, t_start=t_start)
    epoch = np.floor(t.min())
    rf = tls_search_gpu(t, y, dy, periods=periods)                      # default (fast)
    rb = tls_search_batch([(t, y, dy)], periods=periods)[0]
    out = dict(t_start=t_start, min_t=t.min(), t0_true=t0_true, t14=info['t14'],
               fast_T0=rf['T0'], batch_t0_phase=rb['t0_phase'], batch_T0=rb['T0'],
               phase_rel_epoch=((t0_true - epoch) / P) % 1, phase_rel_zero=(t0_true / P) % 1,
               batch_T0_minus_min_t=rb['T0'] - t.min(),
               batch_T0_nearest_true_transit=np.min(np.abs(rb['T0'] - (t0_true + P * np.arange(-3, 20)))))
    if t_start < 1e4:
        rl = tls_search_gpu(t, y, dy, periods=periods, use_fast=False)  # legacy
        out['legacy_T0'] = rl['T0']
    # examples/tls_example.py fold: phases=(t%P)/P, model window at T0 -> does it cover the real transit?
    ph = (t % rf['period']) / rf['period']
    dur_ph = rf['duration'] / rf['period']
    model_in = np.abs((ph - rf['T0'] + 0.5) % 1.0 - 0.5) < dur_ph / 2
    true_in = np.abs(((t - t0_true + 0.5 * P) % P) - 0.5 * P) < info['t14'] / 2
    out['example_fold_overlap_fast'] = (model_in & true_in).sum() / max(true_in.sum(), 1)
    # correct fold for fast path: phase relative to floor(min t)
    ph2 = ((t - epoch) % rf['period']) / rf['period']
    model_in2 = np.abs((ph2 - rf['T0'] + 0.5) % 1.0 - 0.5) < dur_ph / 2
    out['epoch_fold_overlap_fast'] = (model_in2 & true_in).sum() / max(true_in.sum(), 1)
    return out

for ts, fp in [(0.0, 0.37), (100.3, 0.37), (2457000.3, 0.37), (100.9, 0.8)]:
    r = row(ts, fp)
    print("t_start=%.1f min_t=%.3f true_t0=%.4f (t14=%.3f d)" % (r['t_start'], r['min_t'], r['t0_true'], r['t14']))
    print("   fast   T0=%.4f   [phase rel floor(min t)=%.4f ; phase rel t=0 =%.4f]" % (r['fast_T0'], r['phase_rel_epoch'], r['phase_rel_zero']))
    if 'legacy_T0' in r:
        print("   legacy T0=%.4f" % r['legacy_T0'])
    print("   batch  t0_phase=%.4f T0=%.4f   T0-min(t)=%+.4f d ; |T0 - nearest true transit|=%.5f d" % (r['batch_t0_phase'], r['batch_T0'], r['batch_T0_minus_min_t'], r['batch_T0_nearest_true_transit']))
    print("   examples/tls_example.py fold ((t%%P)/P) with fast T0: overlap with true in-transit = %.2f ; fold rel. epoch: %.2f" % (r['example_fold_overlap_fast'], r['epoch_fold_overlap_fast']))

print("--- tls_transit ---")
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
rt = tls_transit(t, y, dy, period_min=1.0, period_max=30.0)
print("tls_transit: period=%.4f T0=%.4f (true t0=2.0 -> phase 2.0/P=%.4f)" % (rt['period'], rt['T0'], (2.0 / rt['period']) % 1))
t2 = t + 2457000.3
rt2 = tls_transit(t2, y, dy, period_min=1.0, period_max=30.0)
print("tls_transit(t+2457000.3): period=%.4f T0=%.4f (true t0=%.1f -> phase rel floor(min t)=%.4f)" % (rt2['period'], rt2['T0'], 2457002.3, ((2457002.3 - 2457000.0) / rt2['period']) % 1))
print("--- reference TLS T0 convention on the same LC (P=3, t_start=100.9, transit at 103.3) ---")
try:
    from transitleastsquares import transitleastsquares as TLS
    t, y, dy, info = make_lc(P, rp, 100.9 + 0.8 * P, baseline=40.0, sigma=1e-3, seed=3, t_start=100.9)
    m = TLS(t, y, dy)
    res = m.power(period_min=2.9, period_max=3.1, oversampling_factor=3, use_threads=4, show_progress_bar=False)
    print("reference TLS: period=%.4f T0=%.4f  min(t)=%.3f  transit_times[:3]=%s" % (res.period, res.T0, t.min(), np.round(res.transit_times[:3], 4)))
except Exception as e:
    print("reference TLS failed:", type(e).__name__, e)
