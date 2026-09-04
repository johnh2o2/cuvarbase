"""TLS fast-path stage profile: single-LC tls_search_gpu and batch tls_search_batch (GPU shared -> medians)."""
import warnings, time, functools, collections; warnings.filterwarnings('ignore')
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from scipy import signal, ndimage
from cuvarbase import tls, tls_models, tls_grids, tls_stats

cfgs = {
    'tess-ffi':   dict(ndata=1310, cadence=30. / 60 / 24, noise=1e-3, pinj=7.7, depth=0.005, pmin=0.6, pmax=13.7),
    'tess-yr':    dict(ndata=16850, cadence=30. / 60 / 24, noise=1e-3, pinj=21.7, depth=0.004, pmin=0.6, pmax=175.),
    'kepler-4yr': dict(ndata=65440, cadence=30. / 60 / 24, noise=6e-4, pinj=41.3, depth=0.003, pmin=0.6, pmax=500.),
}
def make(c, nlc, seed=0):
    out = []
    for i in range(nlc):
        rng = np.random.RandomState(seed + i)
        t = np.arange(c['ndata']) * c['cadence']; y = 1 + rng.randn(c['ndata']) * c['noise']
        q = 0.0763 * c['pinj'] ** (-2. / 3); rel = np.abs(((t - 0.3 * c['pinj'] + 0.5 * c['pinj']) % c['pinj']) - 0.5 * c['pinj'])
        y[rel < 0.5 * q * c['pinj']] -= c['depth']; out.append((t, y, np.full(c['ndata'], c['noise'])))
    return out

T = collections.defaultdict(float); C = collections.defaultdict(int)
def timed(name, fn, sync=False):
    @functools.wraps(fn)
    def w(*a, **k):
        if sync: cuda.Context.synchronize()
        t0 = time.perf_counter(); r = fn(*a, **k)
        if sync: cuda.Context.synchronize()
        T[name] += time.perf_counter() - t0; C[name] += 1; return r
    return w
# host-side stages
tls.tls_models.generate_template_tables = timed('template_tables', tls_models.generate_template_tables)
tls._preprocess_batch = timed('preprocess_cpu', tls._preprocess_batch)
tls.tls_grids.period_grid_ofir = timed('period_grid', tls_grids.period_grid_ofir)
tls.tls_grids.q_transit = timed('q_transit', tls_grids.q_transit)
tls.tls_stats.compute_all_statistics = timed('stats(medfilt etc)', tls_stats.compute_all_statistics)
tls.tls_stats.compute_period_uncertainty = timed('period_unc', tls_stats.compute_period_uncertainty)
tls.gpuarray.to_gpu = timed('to_gpu(alloc+h2d)', gpuarray.to_gpu, sync=True)
tls.gpuarray.empty = timed('gpuarray.empty', gpuarray.empty, sync=True)
gpuarray.GPUArray.get = timed('d2h .get()', gpuarray.GPUArray.get, sync=True)
gpuarray.GPUArray.set = timed('h2d .set()', gpuarray.GPUArray.set, sync=True)
_orig_get = tls._get_cached_fast_kernels
def patched(*a, **k):
    ks = _orig_get(*a, **k)
    return {'search': timed('KERNEL search', ks['search'], sync=True), 'refine': timed('KERNEL refine', ks['refine'], sync=True)}
tls._get_cached_fast_kernels = patched
_orig_ptr = tls.ThreadPoolExecutor

def profile(label, fn, reps=5, per=1):
    fn()  # warm
    T.clear(); C.clear()
    tot = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); tot.append(time.perf_counter() - t0)
    tot_med = float(np.median(tot)); s = sum(T.values()) / reps
    print("\n%s: total median %.2f ms/call = %.2f ms/LC (min %.2f)" % (label, 1e3 * tot_med, 1e3 * tot_med / per, 1e3 * min(tot) / per))
    for k in sorted(T, key=lambda k: -T[k]):
        print("   %-22s %8.3f ms/call (%4.1f%%) x%d calls" % (k, 1e3 * T[k] / reps, 100 * T[k] / reps / tot_med, C[k] // reps))
    print("   %-22s %8.3f ms/call" % ('unaccounted', 1e3 * (tot_med - s)))

for reg in ('tess-ffi', 'tess-yr', 'kepler-4yr'):
    c = cfgs[reg]; lc = make(c, 1)[0]
    profile("%s single tls_transit" % reg, lambda: tls.tls_transit(lc[0], lc[1], lc[2], period_min=c['pmin'], period_max=c['pmax']), reps=5 if reg != 'kepler-4yr' else 3)
    if reg == 'tess-ffi':
        profile("%s single tls_transit refine_top_k=0" % reg, lambda: tls.tls_transit(lc[0], lc[1], lc[2], period_min=c['pmin'], period_max=c['pmax'], refine_top_k=0))
        lcs = make(c, 64)
        profile("%s batch 64 LCs" % reg, lambda: tls.tls_search_batch(lcs, period_min=c['pmin'], period_max=c['pmax']), reps=3, per=64)
    if reg == 'tess-yr':
        lcs = make(c, 16)
        profile("%s batch 16 LCs" % reg, lambda: tls.tls_search_batch(lcs, period_min=c['pmin'], period_max=c['pmax']), reps=3, per=16)

print("\nmedfilt vs ndimage.median_filter (kernel 91):")
for n in (2486, 25000, 172000):
    x = np.random.rand(n)
    a = min(time.perf_counter() - t0 for t0 in [time.perf_counter()] for _ in [signal.medfilt(x, 91)])
    b = min(time.perf_counter() - t0 for t0 in [time.perf_counter()] for _ in [ndimage.median_filter(x, 91, mode='constant')])
    eq = np.allclose(signal.medfilt(x, 91), ndimage.median_filter(x, 91, mode='constant'))
    print("   n=%6d: scipy.signal.medfilt %.2f ms ; ndimage.median_filter(mode=constant) %.2f ms ; identical=%s" % (n, 1e3 * a, 1e3 * b, eq))
print("\ntemplate tables generation alone: %.2f ms" % (1e3 * min(time.perf_counter() - t0 for t0 in [time.perf_counter()] for _ in [tls_models.generate_template_tables()])))
