import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase.utils import autofrequency
from astropy.timeseries import LombScargle
rng = np.random.RandomState(1)
N, T = 600, 100.0
t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N)
dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
ls = LombScargle(t, y, dy); proc = LombScargleAsyncProcess()
fu = np.concatenate([np.arange(0.1, 1.0, 0.002), np.arange(1.0, 5.0, 0.01)])
df = fu[1]-fu[0]; fi = df*(get_k0(fu)+np.arange(len(fu)))
out = proc.batched_run_const_nfreq([(t, y, dy)], freqs=fu)
print('batched return type/len:', type(out), len(out))
fr, p = out[0]
print('returned freqs is user grid:', np.array_equal(np.asarray(fr).ravel()[:len(fu)], fu) if np.size(fr)>=len(fu) else fr)
p = np.asarray(p[0]) if isinstance(p, list) else np.asarray(p)
def met(a, b): return 'maxabs=%.2e corr=%.4f' % (np.abs(a-b).max(), np.corrcoef(a, b)[0, 1])
print('batched: vs astropy@USER: %s | vs astropy@IMPLIED: %s' % (met(ls.power(fu), p), met(ls.power(fi), p)))
# proposed check: does it falsely reject autofrequency grids / linspace grids (float64 and float32)?
for name, f in [('autofrequency', autofrequency(t)), ('linspace 0.1..10 nf=50001', np.linspace(0.1, 10, 50001)),
                ('arange float32', (0.002*(50+np.arange(2_000_000))).astype(np.float32)),
                ('df*(k0+arange) 2M', 0.002*(50+np.arange(2_000_000)))]:
    d = np.diff(f); df = f[1]-f[0]
    ok = np.allclose(d, df, rtol=1e-6, atol=0); print('%-28s uniform-check(rtol=1e-6) passes=%s  max|diff/df-1|=%.2e' % (name, ok, np.abs(d/df-1).max()))
