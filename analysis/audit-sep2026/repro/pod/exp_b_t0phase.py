"""Experiment B: SDE sensitivity to the injected epoch relative to the coarse t0 grid."""
import numpy as np, sys, json, time
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu
from cuvarbase import tls_grids

P, rp, base, sig = 7.3, 0.05, 60.0, 1e-3
# fixed period grid from tls_grids (same as reference formula); ~ what a user gets by default
t, y, dy, info = make_lc(P, rp, 0.3 * P, baseline=base, sigma=sig, seed=1)
periods = tls_grids.period_grid_ofir(t)
q_true = info['t14'] / P
print("q_true=%.4f  ndata=%d nperiods=%d  coarse t0 stride at q_true = %.5f phase (%.3f d)" % (q_true, len(t), len(periods), q_true/3, q_true/3*P))
# sweep t0 across ~1.5 coarse strides in 12 steps, at fixed noise realisation
stride = q_true / 3 * P
t0s = 0.3 * P + np.linspace(0, 1.5 * stride, 13)
rows = []
for t0 in t0s:
    t, y, dy, info = make_lc(P, rp, t0, baseline=base, sigma=sig, seed=1)
    r3 = tls_search_gpu(t, y, dy, periods=periods)
    r33 = tls_search_gpu(t, y, dy, periods=periods, t0_oversample=33.0)
    rl3 = tls_search_gpu(t, y, dy, periods=periods, use_fast=False)
    rl33 = tls_search_gpu(t, y, dy, periods=periods, use_fast=False, t0_oversample=33.0)
    rows.append((t0, r3['SDE'], r33['SDE'], rl3['SDE'], rl33['SDE'], r3['SNR'], r33['SNR'], r3['depth'], r33['depth'], r3['period'], r33['period']))
    print("t0=%.4f (%.2f strides): SDE fast3=%.2f fast33=%.2f legacy3=%.2f legacy33=%.2f | SNR3=%.1f SNR33=%.1f | depth3=%.4f depth33=%.4f | P3=%.4f P33=%.4f" % ((t0, (t0-0.3*P)/stride) + tuple(rows[-1][1:])))
    sys.stdout.flush()
a = np.array(rows)
print("SDE range fast3: %.2f..%.2f (rel spread %.1f%%), fast33: %.2f..%.2f (%.1f%%), legacy3: %.2f..%.2f, legacy33: %.2f..%.2f" % (
    a[:,1].min(), a[:,1].max(), 100*(a[:,1].max()-a[:,1].min())/a[:,1].mean(),
    a[:,2].min(), a[:,2].max(), 100*(a[:,2].max()-a[:,2].min())/a[:,2].mean(),
    a[:,3].min(), a[:,3].max(), a[:,4].min(), a[:,4].max()))
# reference at 3 phases (worst / best of fast3)
import transitleastsquares as tlsref
for idx in [int(np.argmin(a[:,1])), int(np.argmax(a[:,1]))]:
    t0 = t0s[idx]
    t, y, dy, info = make_lc(P, rp, t0, baseline=base, sigma=sig, seed=1)
    ref, dt = run_ref(t, y, dy)
    print("reference at t0=%.4f: SDE=%.2f (fast3 %.2f, fast33 %.2f) [%.1fs, %d periods]" % (t0, ref.SDE, a[idx,1], a[idx,2], dt, len(ref.periods)))
