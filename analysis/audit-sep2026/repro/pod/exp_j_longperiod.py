"""Experiment J: default tls_search_gpu (fixed q window [0.005,0.15]) vs Keplerian window at long periods."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu, tls_search_batch, tls_transit
from cuvarbase import tls_grids
base = 1400.0
for P, rp, sig in [(120.0, 0.03, 3e-4), (365.0, 0.04, 3e-4)]:
    t, y, dy, info = make_lc(P, rp, 0.41 * P, baseline=base, sigma=sig, seed=11)
    q_true = info['t14'] / P
    periods = tls_grids.period_grid_ofir(t, period_min=0.5 * P, period_max=1.5 * P)
    print("P=%.0f rp=%.2f ndata=%d nperiods=%d q_true=%.5f (std window qmin=0.005 = %.1fx q_true)" % (P, rp, len(t), len(periods), q_true, 0.005 / q_true))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t1 = time.time(); rs = tls_search_gpu(t, y, dy, periods=periods); dts = time.time() - t1
        t1 = time.time(); rb = tls_search_batch([(t, y, dy)], periods=periods)[0]; dtb = time.time() - t1
    print("   default tls_search_gpu (std window): P=%.3f SDE=%.2f depth=%.5f (true %.5f) dur=%.3f (true %.3f) SNR=%.1f [%.1fs]" % (rs['period'], rs['SDE'], rs['depth'], info['true_depth'], rs['duration'], info['t14'], rs['SNR'], dts))
    print("   tls_search_batch (Keplerian window):  P=%.3f SDE=%.2f depth=%.5f dur=%.3f SNR=%.1f [%.1fs]" % (rb['period'], rb['SDE'], rb['depth'], rb['duration'], rb['SNR'], dtb))
