import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, pycuda.gpuarray as gpuarray
from cuvarbase import tls
exec(open('tls_profile.py').read().split("T = collections.defaultdict")[0].split("from cuvarbase import tls")[1].replace(", tls_models, tls_grids, tls_stats", ""))
def bench(fn, reps=7):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(xs)), 1e3 * min(xs)
c = cfgs['tess-ffi']; t, y, dy = make(c, 1)[0]
f = lambda: tls.tls_transit(t, y, dy, period_min=c['pmin'], period_max=c['pmax']); f()
print("tess-ffi single tls_transit default: median %.2f ms (min %.2f)" % bench(f))
f2 = lambda: tls.tls_search_gpu(t, y, dy, period_min=c['pmin'], period_max=c['pmax']); f2()
print("tess-ffi single tls_search_gpu (standard grid) default: median %.2f ms (min %.2f)" % bench(f2))
n = [0]; oe = gpuarray.empty; og = gpuarray.to_gpu
tls.gpuarray.empty = lambda *a, **k: (n.__setitem__(0, n[0] + 1), oe(*a, **k))[1]; tls.gpuarray.to_gpu = lambda *a, **k: (n.__setitem__(0, n[0] + 1), og(*a, **k))[1]
f(); print("device allocations per tls_transit call: %d" % n[0])
