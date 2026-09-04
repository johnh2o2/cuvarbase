import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.cunfft import NFFTAsyncProcess
from astropy.timeseries import LombScargle

def lc(T, N=600, seed=1):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N); dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
    return t, y, dy

print('=== A. LS use_fft=True: float32 vs use_double=True (are they different at all?) ===')
t, y, dy = lc(100.0); df = 1.0/(5*(t.max()-t.min())); f = df*(1+np.arange(1495)); a = LombScargle(t, y, dy).power(f)
p32 = LombScargleAsyncProcess(); r = p32.run([(t, y, dy)], freqs=[f]); p32.finish(); g32 = np.copy(r[0][1]); print('  float32 dtype', g32.dtype)
p64 = LombScargleAsyncProcess(use_double=True); r = p64.run([(t, y, dy)], freqs=[f]); p64.finish(); g64 = np.copy(r[0][1]); print('  float64 dtype', g64.dtype)
print('  max|g32-g64|=%.2e  max|g32-astropy|=%.2e max|g64-astropy|=%.2e' % (np.abs(g32-g64).max(), np.abs(g32-a).max(), np.abs(g64-a).max()))

print('=== B. NFFT adjoint per-mode error vs float64 direct sums, LS layout: modes k0..k0+nf-1 on grid n=sigma*nf ===')
w = 1.0/dy**2; w /= w.sum(); yw = w*(y - np.sum(w*y))
tc = t - t.mean()   # LS mean-centers t (normalize_light_curves)
nf = 1000; df = 1.0/(5*(tc.max()-tc.min())); spp = 1.0/((tc.max()-tc.min())*df)
for sigma in (4, 8):
  for use_double in (False, True):
    proc = NFFTAsyncProcess(sigma=sigma, use_double=use_double)
    for k0 in (0, nf//2, nf, 2*nf):
        f0 = k0*df
        g = proc.run([(tc, yw, nf)], minimum_frequency=f0, samples_per_peak=spp)[0]
        proc.finish() if hasattr(proc, 'finish') else None
        import pycuda.driver as cuda; cuda.Context.synchronize()
        g = np.copy(g)
        fk = df*(k0+np.arange(nf))
        ex = np.array([np.sum(yw*np.exp(2j*np.pi*fq*tc)) for fq in fk])
        d = np.abs(g-ex); sc = np.abs(ex).max(); q = [d[i*nf//4:(i+1)*nf//4].max()/sc for i in range(4)]
        print('  sigma=%d double=%-5s k0/nf=%.1f (k0+nf)/n=%.3f  maxrel=%.1e  per-quarter=%s' % (sigma, use_double, k0/nf, (k0+nf)/(sigma*nf), d.max()/sc, ' '.join('%.1e' % x for x in q)))

print('=== C. LS floor vs baseline T (k0=1 grids, fmax=3, spp=5) float32 and float64 ===')
for T in (10.0, 100.0, 1000.0):
    t, y, dy = lc(T); df = 1.0/(5*(t.max()-t.min())); nf = int(round((3.0-df)/df)); f = df*(1+np.arange(nf)); a = LombScargle(t, y, dy).power(f)
    for kw in (dict(), dict(use_double=True)):
        p = LombScargleAsyncProcess(**kw); g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
        gd = np.copy(p.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); p.finish()
        print('  T=%-6g nf=%-6d %-22s NFFT-vs-astropy maxabs=%.2e  direct-vs-astropy=%.2e  NFFT-vs-direct=%.2e  peak: gpu=%.4f astropy=%.4f' % (T, nf, kw, np.abs(g-a).max(), np.abs(gd-a).max(), np.abs(g-gd).max(), g.max(), a.max()))
