"""TLS coarse kernel: shared-memory atomic contention from time-ordered input (sorted vs conflict_scatter_perm vs random)."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase import tls, tls_grids
from cuvarbase.utils import conflict_scatter_perm
exec(open('tls_profile.py').read().split("T = collections.defaultdict")[0].split("from cuvarbase import tls")[1].replace(", tls_models, tls_grids, tls_stats", ""))
def bench(fn, reps=5):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(xs))
rng = np.random.RandomState(0)
for reg in ('tess-ffi', 'tess-yr', 'kepler-4yr'):
    c = cfgs[reg]; t, y, dy = make(c, 1)[0]
    per = tls_grids.period_grid_ofir(t, period_min=c['pmin'], period_max=c['pmax']); qv = tls_grids.q_transit(per); qmin, qmax = qv * 0.5, qv * 2
    pg = conflict_scatter_perm(len(t)); pr = rng.permutation(len(t))
    orders = [("time-sorted (as shipped)", np.arange(len(t))), ("conflict_scatter_perm", pg), ("random perm", pr)]
    res = {}
    for label, p in orders:
        lc = (t[p], y[p], dy[p])
        f = lambda: tls.tls_search_batch([lc], periods=per, qmin=qmin, qmax=qmax, refine_top_k=0, return_arrays=True)
        r = f(); res[label] = r[0]['chi2']
        f1 = lambda: tls.tls_search_batch([lc], periods=per, qmin=qmin, qmax=qmax, return_arrays=False)
        print("   %-10s %-26s coarse-only %.1f ms | default (refine on) %.1f ms" % (reg, label, bench(f, 3), bench(f1, 3)))
    print("      max|chi2 sorted - chi2 perm| = %.2e (chi2 ~ %.0f)" % (np.nanmax(np.abs(res[orders[0][0]] - res[orders[1][0]])), np.nanmedian(res[orders[0][0]])))
