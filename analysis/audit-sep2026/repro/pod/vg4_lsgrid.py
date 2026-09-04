import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_simple
from astropy.timeseries import LombScargle
rng = np.random.RandomState(1); N, T = 600, 100.0
t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N); dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
df = 1.0/(5*(t.max()-t.min()))
f = df*(997+np.arange(498))   # fmin=2, fmax=3
a = LombScargle(t, y, dy).power(f)
p = LombScargleAsyncProcess()
bf, bp = p.batched_run_const_nfreq([(t, y, dy)], freqs=f, only_return_best_freqs=True, use_fap=False) if 'use_fap' in p.batched_run_const_nfreq.__code__.co_varnames else p.batched_run_const_nfreq([(t, y, dy)], freqs=f, only_return_best_freqs=True)
print('batched_run_const_nfreq fmin=2 fmax=3: best_freq=%s best_pow=%s ; astropy best=%.4f pow=%.4f' % (bf, bp, f[np.argmax(a)], a.max()))
res = lomb_scargle_simple(t, y, dy, freqs=f); g = np.asarray(res[1] if isinstance(res, (tuple, list)) else res)[:len(f)]
print('lomb_scargle_simple fmin=2 fmax=3: max=%.3g argmax f=%.4f (astropy %.4f) maxabs=%.2e' % (g.max(), f[np.argmax(g)], f[np.argmax(a)], np.abs(g-a).max()))
# small grids (auditor h5): nf=8, k0=50
f8 = df*(50+np.arange(8)); a8 = LombScargle(t, y, dy).power(f8); r = p.run([(t, y, dy)], freqs=[f8]); p.finish(); g8 = np.copy(r[0][1])
print('nf=8 k0=50: gpu=%s astropy max=%.3g' % (np.array2string(g8, precision=3), a8.max()))
# batched_run_const_nfreq on default grid: floor from psi sharing present?
f0 = df*(1+np.arange(1495)); a0 = LombScargle(t, y, dy).power(f0)
fr, pw = p.batched_run_const_nfreq([(t, y, dy)], freqs=f0)
print('batched default grid k0=1: maxabs vs astropy=%.2e' % np.abs(np.asarray(pw[0])-a0).max())
