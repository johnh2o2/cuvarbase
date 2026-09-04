"""Experiment I: where does single-LC tls_search_gpu time go? (relative, noisy GPU)"""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase import tls_models, tls_grids
import cuvarbase.tls as T
from cuvarbase.tls import tls_search_gpu
t, y, dy, info = make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=5)
periods = tls_grids.period_grid_ofir(t)
tls_search_gpu(t, y, dy, periods=periods)  # warm (compile)
def timeit(f, n=5):
    ts = []
    for _ in range(n):
        t1 = time.perf_counter(); f(); ts.append(time.perf_counter() - t1)
    return np.median(ts), np.min(ts)
print("ndata=%d nperiods=%d" % (len(t), len(periods)))
print("generate_template_tables (batman):     median %.1f ms" % (1e3 * timeit(lambda: tls_models.generate_template_tables(n_table=1024))[0]))
print("generate_transit_template n=8193:      median %.1f ms" % (1e3 * timeit(lambda: tls_models.generate_transit_template(n_template=8193))[0]))
print("_preprocess_batch:                     median %.1f ms" % (1e3 * timeit(lambda: T._preprocess_batch([(t, y, dy)]))[0]))
print("tls_search_gpu full call:              median %.1f ms" % (1e3 * timeit(lambda: tls_search_gpu(t, y, dy, periods=periods))[0]))
print("tls_search_gpu refine_top_k=0:         median %.1f ms" % (1e3 * timeit(lambda: tls_search_gpu(t, y, dy, periods=periods, refine_top_k=0))[0]))
# monkeypatch template tables to a cache to see the saving
tabs = tls_models.generate_template_tables(n_table=1024)
orig = tls_models.generate_template_tables
tls_models.generate_template_tables = lambda **k: tabs
print("tls_search_gpu with cached tables:     median %.1f ms" % (1e3 * timeit(lambda: tls_search_gpu(t, y, dy, periods=periods))[0]))
tls_models.generate_template_tables = orig
# stats cost
from cuvarbase import tls_stats
r = tls_search_gpu(t, y, dy, periods=periods, return_arrays=True)
chi2 = r['chi2'][np.isfinite(r['chi2'])]
print("compute_all_statistics (n=%d):        median %.1f ms" % (len(chi2), 1e3 * timeit(lambda: tls_stats.compute_all_statistics(chi2, periods[:len(chi2)], int(np.argmin(chi2)), 0.01, 0.1, 5))[0]))
# batch of 64 LCs same grid: per-LC time
lcs = [make_lc(7.3, 0.05, 2.0, baseline=60.0, sigma=1e-3, seed=i)[:3] for i in range(64)]
from cuvarbase.tls import tls_search_batch
tls_search_batch(lcs[:2], periods=periods)
m, _ = timeit(lambda: tls_search_batch(lcs, periods=periods), n=3)
print("tls_search_batch 64 LCs:               median %.1f ms total = %.2f ms/LC" % (1e3 * m, 1e3 * m / 64))
m, _ = timeit(lambda: tls_search_batch(lcs, periods=periods, refine_top_k=0), n=3)
print("tls_search_batch 64 LCs no refine:     median %.1f ms total = %.2f ms/LC" % (1e3 * m, 1e3 * m / 64))
