"""Experiment E: transits outside the default Keplerian duration window [0.5,2]*q_kep(R_planet=1 R_earth)."""
import numpy as np, sys, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu, tls_search_batch
from cuvarbase import tls_grids
P, base, sig = 7.3, 90.0, 1e-3
periods = tls_grids.period_grid_ofir(np.arange(0, base, 1/48))
qk = tls_grids.q_transit(P)
print("q_kep(P=%.1f, R_planet=1Re)=%.5f -> window [%.5f, %.5f]; nperiods=%d" % (P, qk, 0.5*qk, 2*qk, len(periods)))
cases = [('b=0 rp=0.05', dict(b=0.0, rp=0.05, R_star=1.0, M_star=1.0)),
         ('b=0.92 rp=0.05 (grazing)', dict(b=0.92, rp=0.05, R_star=1.0, M_star=1.0)),
         ('b=0.96 rp=0.08', dict(b=0.96, rp=0.08, R_star=1.0, M_star=1.0)),
         ('subgiant R=2.2 M=1.1 (user assumes Sun)', dict(b=0.0, rp=0.05, R_star=2.2, M_star=1.1)),
         ('M-dwarf R=0.3 M=0.3 (user assumes Sun)', dict(b=0.0, rp=0.08, R_star=0.3, M_star=0.3)),
         ]
import transitleastsquares as tlsref
for label, kw in cases:
    t, y, dy, info = make_lc(P, kw['rp'], 2.0, baseline=base, sigma=sig, seed=7, R_star=kw['R_star'], M_star=kw['M_star'], b=kw['b'])
    q_true = info['t14'] / P
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rb = tls_search_batch([(t, y, dy)], periods=periods)[0]             # default Keplerian window, R=M=1
        rs = tls_search_gpu(t, y, dy, periods=periods)                       # standard fixed window [0.005,0.15]
    ref, dt = run_ref(t, y, dy, threads=24)
    print("%-42s q_true=%.5f (%.2f x q_kep) depth=%.4f | batch_kep: P=%.4f SDE=%.2f dep=%.4f dur=%.3f | std: P=%.4f SDE=%.2f dep=%.4f dur=%.3f | ref: P=%.4f SDE=%.2f dep=%.4f dur=%.3f [%.0fs]" % (
        label, q_true, q_true / qk, info['true_depth'], rb['period'], rb['SDE'], rb['depth'], rb['duration'], rs['period'], rs['SDE'], rs['depth'], rs['duration'], ref.period, ref.SDE, 1 - ref.depth, ref.duration, dt))
    sys.stdout.flush()
