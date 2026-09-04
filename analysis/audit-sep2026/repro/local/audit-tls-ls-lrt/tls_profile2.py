"""TLS follow-ups: duration_grid_keplerian cost, thread-pool effect, fold-vs-scan split of the coarse kernel, band structure."""
import warnings, time, os; warnings.filterwarnings('ignore')
import numpy as np
import pycuda.driver as cuda
from cuvarbase import tls, tls_grids
exec(open('tls_profile.py').read().split("T = collections.defaultdict")[0].split("from cuvarbase import tls")[1].replace(", tls_models, tls_grids, tls_stats", ""))  # cfgs, make
med = lambda xs: 1e3 * float(np.median(xs))
def bench(fn, reps=5):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return med(xs)

print("(a) duration_grid_keplerian (called by tls_transit, table unused by the fast path) vs q_transit:")
for reg in ('tess-ffi', 'tess-yr', 'kepler-4yr'):
    c = cfgs[reg]; lc = make(c, 1)[0]
    per = tls_grids.period_grid_ofir(lc[0], period_min=c['pmin'], period_max=c['pmax'])
    print("   %-10s nperiods=%6d: duration_grid_keplerian %.2f ms ; q_transit %.2f ms ; period_grid_ofir %.2f ms"
          % (reg, len(per), bench(lambda: tls_grids.duration_grid_keplerian(per), 3), bench(lambda: tls_grids.q_transit(per)), bench(lambda: tls_grids.period_grid_ofir(lc[0], period_min=c['pmin'], period_max=c['pmax']))))

print("\n(b) batch wall time: default thread pool (min(8,cpu)) vs 1 worker for the per-LC statistics:")
orig_cpu = tls.os.cpu_count
for reg, nlc in (('tess-ffi', 64), ('tess-yr', 16)):
    c = cfgs[reg]; lcs = make(c, nlc)
    tls.tls_search_batch(lcs[:2], period_min=c['pmin'], period_max=c['pmax'])
    t8 = bench(lambda: tls.tls_search_batch(lcs, period_min=c['pmin'], period_max=c['pmax']), 3)
    tls.os.cpu_count = lambda: 1
    t1 = bench(lambda: tls.tls_search_batch(lcs, period_min=c['pmin'], period_max=c['pmax']), 3)
    tls.os.cpu_count = orig_cpu
    t0k = bench(lambda: tls.tls_search_batch(lcs, period_min=c['pmin'], period_max=c['pmax'], refine_top_k=0), 3)
    print("   %-10s %d LCs: 8 workers %.1f ms (%.2f ms/LC) | 1 worker %.1f ms (%.2f ms/LC) | 8 workers, refine_top_k=0: %.1f ms" % (reg, nlc, t8, t8 / nlc, t1, t1 / nlc, t0k))

print("\n(c) coarse kernel: time vs n_durations (single LC, refine off) -> intercept = fold/bin + fixed cost, slope = trial scan:")
for reg in ('tess-yr', 'kepler-4yr'):
    c = cfgs[reg]; lc = make(c, 1)[0]
    per = tls_grids.period_grid_ofir(lc[0], period_min=c['pmin'], period_max=c['pmax'])
    qv = tls_grids.q_transit(per); qmin, qmax = qv * 0.5, qv * 2
    # band structure
    need = 3.0 / qmin; nb = np.minimum(np.power(2, np.ceil(np.log2(np.clip(need, 256, None)))), 8192)
    u, cnt = np.unique(nb, return_counts=True)
    print("   %-10s nperiods=%d ndata=%d bands (nbins:count): %s" % (reg, len(per), c['ndata'], dict(zip(u.astype(int), cnt))))
    rows = []
    for nd in (2, 5, 15, 30):
        f = lambda: tls.tls_search_batch([lc], periods=per, qmin=qmin, qmax=qmax, n_durations=nd, refine_top_k=0)
        f(); rows.append((nd, bench(f, 3)))
    nds = np.array([r[0] for r in rows]); ts = np.array([r[1] for r in rows]); A = np.vstack([np.ones_like(nds), nds]).T
    b, a = np.linalg.lstsq(A, ts, rcond=None)[0]
    print("      " + "  ".join("nd=%d: %.1f ms" % r for r in rows) + "  -> fit: %.1f ms + %.2f ms/duration (fold+fixed = %.0f%% at nd=15)" % (b, a, 100 * b / (b + 15 * a)))
    # same LC repeated 4x in batch to see per-LC scaling (GPU-limited?)
    f4 = lambda: tls.tls_search_batch([lc] * 4, periods=per, qmin=qmin, qmax=qmax, n_durations=15, refine_top_k=0)
    f4(); print("      4 LCs batch, refine off: %.1f ms/LC" % (bench(f4, 3) / 4))
