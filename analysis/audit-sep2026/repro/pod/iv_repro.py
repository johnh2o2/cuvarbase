import sys, warnings, numpy as np
warnings.simplefilter('ignore')
from cuvarbase import bls as B
case = sys.argv[1]
rng = np.random.RandomState(0)
N = 600
t = np.sort(rng.uniform(0, 100, N)); y = 1 + 0.01*rng.randn(N); dy = 0.01*np.ones(N)
F = np.linspace(0.1, 3.0, 300)
def run(fn, *a, **k):
    try:
        r = fn(*a, **k); 
        p = r[0] if isinstance(r, tuple) else r
        return 'ok n=%d nonfinite=%d' % (len(p), int(np.sum(~np.isfinite(p))))
    except BaseException as e:
        return 'RAISE %s: %s' % (type(e).__name__, str(e).splitlines()[0][:90])
if case == 'fast_nanq':
    print('fast qmin/qmax NaN     :', run(B.eebls_gpu_fast, t, y, dy, F[:2], qmin=np.array([np.nan, 0.01]), qmax=np.array([np.nan, 0.2])))
elif case == 'std_nanq':
    print('std  qmin/qmax NaN     :', run(B.eebls_gpu, t, y, dy, F[:2], qmin=np.array([np.nan, 0.01]), qmax=np.array([np.nan, 0.2])))
elif case == 'std_nant':
    tt = t.copy(); tt[137] = np.nan
    print('std  NaN in t          :', run(B.eebls_gpu, tt, y, dy, F, qmin=0.01, qmax=0.2))
elif case == 'transit_default_N4':
    t4, y4, d4 = t[:4], y[:4], dy[:4]
    print('eebls_transit_gpu N=4 default(use_fast=False):', run(B.eebls_transit_gpu, t4, y4, d4))
elif case == 'transit_top_N4':
    t4, y4, d4 = t[:4], y[:4], dy[:4]
    print('eebls_transit N=4 default:', run(B.eebls_transit, t4, y4, d4))
elif case == 'transit_fast_N4':
    t4, y4, d4 = t[:4], y[:4], dy[:4]
    print('eebls_transit_gpu N=4 use_fast=True:', run(B.eebls_transit_gpu, t4, y4, d4, use_fast=True))
elif case == 'transit_default_nant':
    tt = t.copy(); tt[137] = np.nan
    print('eebls_transit_gpu NaN t default:', run(B.eebls_transit_gpu, tt, y, dy))
elif case == 'transit_top_nant':
    tt = t.copy(); tt[137] = np.nan
    print('eebls_transit NaN t default:', run(B.eebls_transit, tt, y, dy))
print('   subsequent valid fast call:', run(B.eebls_gpu_fast, t, y, dy, F, qmin=0.01, qmax=0.2))
