import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from astropy.timeseries import LombScargle

rng = np.random.RandomState(1)
N, T = 600, 100.0
t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N); dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
df = 1.0/(5*(t.max()-t.min()))

def grid(fmin, fmax):
    k0 = int(round(fmin/df)); nf = int(round((fmax-fmin)/df)); return df*(k0+np.arange(nf)), k0, nf
def met(a, g):
    d = np.abs(a-g); return d.max(), d.max()/a.max(), np.corrcoef(a, g)[0,1], np.argmax(a)==np.argmax(g), int(np.argmax(d))

print('=== 1. user path with NO explicit freqs: run(minimum_frequency=, maximum_frequency=) ===')
p = LombScargleAsyncProcess()
for fmin, fmax in ((0.002, 3.0), (1.0, 3.0), (1.5, 3.0), (2.0, 3.0), (0.5, 1.0)):
    r = p.run([(t, y, dy)], minimum_frequency=fmin, maximum_frequency=fmax); p.finish()
    f, g = r[0][0], np.copy(r[0][1]); k0 = get_k0(f); nf = len(f)
    a = LombScargle(t, y, dy).power(f)
    mx, rel, c, same, im = met(a, g)
    print('fmin=%-5g fmax=%-4g k0=%-5d nf=%-5d (k0+nf)/n=%.2f  maxabs=%.2e rel=%.2e corr=%.4f argmax_same=%s  gpu[max]=%.3g astropy[max]=%.3g' % (fmin, fmax, k0, nf, (k0+nf)/(4.0*nf), mx, rel, c, same, g.max(), a.max()))

print('=== 2. floor at k0=1: float32 vs float64, sigma 4/8, error location in band ===')
f, k0, nf = grid(0.002, 3.0)
a = LombScargle(t, y, dy).power(f)
for kw in (dict(), dict(sigma=8), dict(use_double=True), dict(use_double=True, sigma=8), dict(use_double=True, m=12)):
    p = LombScargleAsyncProcess(**kw); g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
    d = np.abs(a-g); q = [d[i*nf//4:(i+1)*nf//4].max() for i in range(4)]
    print('%-40s maxabs=%.2e rel=%.2e  per-quarter maxabs=%s  at k/nf=%.2f' % (kw, d.max(), d.max()/a.max(), ' '.join('%.1e' % x for x in q), np.argmax(d)/nf))
p = LombScargleAsyncProcess(); g = np.copy(p.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); p.finish()
d = np.abs(a-g); print('%-40s maxabs=%.2e rel=%.2e' % ('float32 direct sums', d.max(), d.max()/a.max()))

print('=== 3. candidate fix: sigma_eff = ceil(sigma*(k0+nf)/nf) (the autoadjust_sigma formula, lombscargle.py:596) ===')
for fmin, fmax in ((0.1, 3.0), (1.0, 3.0), (1.5, 3.0), (2.0, 3.0), (2.5, 3.0), (5.0, 10.0), (0.5, 1.0)):
    f, k0, nf = grid(fmin, fmax); a = LombScargle(t, y, dy).power(f)
    row = []
    for label, sig in (('default', 4), ('sigma_eff', int(np.ceil(4.0*(k0+nf)/nf)))):
        p = LombScargleAsyncProcess(sigma=sig); g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
        mx, rel, c, same, im = met(a, g); row.append('%s(sigma=%d): rel=%.1e corr=%.4f' % (label, sig, rel, c))
    p = LombScargleAsyncProcess(sigma=int(np.ceil(4.0*(k0+nf)/nf)), use_double=True); g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
    mx, rel, c, same, im = met(a, g); row.append('sigma_eff+double: rel=%.1e' % rel)
    print('fmin=%-4g fmax=%-3g k0/nf=%.2f | %s' % (fmin, fmax, k0/nf, ' | '.join(row)))

print('=== 4. nharmonics=2 on fmin>=fmax/2 grid (default sigma) vs direct sums ===')
f, k0, nf = grid(2.0, 3.0)
p = LombScargleAsyncProcess(nharmonics=2)
g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
gd = np.copy(p.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); p.finish()
print('nharm=2 fmin=2 fmax=3: NFFT max=%.3g direct max=%.3g maxabs diff=%.2e' % (g.max(), gd.max(), np.abs(g-gd).max()))
