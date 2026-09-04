"""Verifier: does the default tls_search_gpu path (no periods, no qmin/qmax) mis-recover long-period transits
because of the fixed q window [0.005,0.15]?  True default path = automatic Ofir grid to span/2."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import make_lc
from cuvarbase.tls import tls_search_gpu, tls_search_batch, tls_transit
from cuvarbase import tls_grids

def run(label, fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t1 = time.time(); r = fn(*a, **k); dt = time.time() - t1
    if isinstance(r, list): r = r[0]
    print("   %-38s P=%8.3f SDE=%6.2f depth=%.5f dur=%.3f SNR=%5.1f nper=%d [%.1fs]" % (
        label, r['period'], r['SDE'], r['depth'], r['duration'], r['SNR'], len(r.get('periods', [])), dt))
    return r

cases = [
    # (label, P, rp, sigma, baseline, R, M)
    ("Sun P=45 (q_kep>0.005)",  45.0, 0.03, 3e-4, 1400.0, 1.0, 1.0),
    ("Sun P=90",                 90.0, 0.03, 3e-4, 1400.0, 1.0, 1.0),
    ("Sun P=365",               365.0, 0.04, 3e-4, 1400.0, 1.0, 1.0),
    ("Mdwarf P=30 (200d base)",  30.0, 0.05, 5e-4,  200.0, 0.3, 0.3),
]
for label, P, rp, sig, base, R, M in cases:
    t, y, dy, info = make_lc(P, rp, 0.41 * P, baseline=base, sigma=sig, seed=11, R_star=R, M_star=M)
    q_true = info['t14'] / P
    print("%s: rp=%.2f ndata=%d true depth=%.5f t14=%.3f d q_true=%.5f (0.005/q_true=%.2f); q_kep(R,M,1Re)=%.5f" % (
        label, rp, len(t), info['true_depth'], info['t14'], q_true, 0.005 / q_true, tls_grids.q_transit(P, R, M)))
    # TRUE default path: no periods, no qmin/qmax  -> Ofir grid to span/2, fixed window
    run("tls_search_gpu DEFAULT (fixed window)", tls_search_gpu, t, y, dy, R_star=R, M_star=M)
    # same call but Keplerian window from the same R,M (what tls_transit / tls_search_batch do by default)
    run("tls_transit (Keplerian window)", tls_transit, t, y, dy, R_star=R, M_star=M)
    run("tls_search_batch default", tls_search_batch, [(t, y, dy)], R_star=R, M_star=M)
    # legacy per-point kernel would also use the fixed window (hard-coded in tls.cu) but caps ndata ~3500: skip
