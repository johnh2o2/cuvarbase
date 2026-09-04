import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx, pycuda.driver as cuda
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from astropy.timeseries import LombScargle

rng = np.random.RandomState(1); N, T = 600, 100.0
t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N); dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
df = 1.0/(5*(t.max()-t.min()))
def grid(fmin, fmax):
    k0 = int(round(fmin/df)); nf = int(round((fmax-fmin)/df)); return df*(k0+np.arange(nf)), k0, nf
w = 1.0/dy**2; w /= w.sum(); tc = t - t.mean()

print('=== 1. spectra inside the LS memory (double precision), k0=1 grid: w-spectrum vs yw-spectrum accuracy ===')
f, k0, nf = grid(0.002, 3.0); a = LombScargle(t, y, dy).power(f)
p = LombScargleAsyncProcess(use_double=True)
M = p.allocate([(t, y, dy)], nfreqs=[nf], k0s=[k0]); r = p.run([(t, y, dy)], freqs=[f], memory=M); p.finish(); g0 = np.copy(r[0][1]); mem = M[0]
print('  n_yw=%d n_w=%d  (w grid has its own ng but reuses yw psi tables: nfft_mem_w.precomp_psi=%s, q1 shared=%s)' % (mem.nfft_mem_yw.n, mem.nfft_mem_w.n, mem.nfft_mem_w.precomp_psi, mem.nfft_mem_w.q1 is mem.nfft_mem_yw.q1))
sw = mem.nfft_mem_w.ghat_g.get()[:mem.nfft_mem_w.nf]; syw = mem.nfft_mem_yw.ghat_g.get()[:nf]
fw = df*(k0+np.arange(mem.nfft_mem_w.nf))
ex_w = np.array([np.sum(w*np.exp(2j*np.pi*fq*tc)) for fq in fw])
ybar = np.sum(w*y); yw = w*(y-ybar)
ex_yw = np.array([np.sum(yw*np.exp(2j*np.pi*fq*tc)) for fq in f])
print('  w-spectrum  max|gpu-exact|/max|exact| = %.2e' % (np.abs(sw-ex_w).max()/np.abs(ex_w).max()))
print('  yw-spectrum max|gpu-exact|/max|exact| = %.2e' % (np.abs(syw-ex_yw).max()/np.abs(ex_yw).max()))
print('  LS power vs astropy maxabs = %.2e' % np.abs(g0-a).max())

print('=== 2. give the w grid its own psi tables (monkeypatch memory only) ===')
mem.nfft_mem_w.precomp_psi = True; mem.nfft_mem_w.allocate_precomp_psi(n0=mem.n0)
r = p.run([(t, y, dy)], freqs=[f], memory=M); p.finish(); g1 = np.copy(r[0][1])
sw = mem.nfft_mem_w.ghat_g.get()[:mem.nfft_mem_w.nf]
print('  w-spectrum  max|gpu-exact|/max|exact| = %.2e' % (np.abs(sw-ex_w).max()/np.abs(ex_w).max()))
print('  LS power vs astropy maxabs = %.2e' % np.abs(g1-a).max())

print('=== 3. workaround via public API: run(..., fast_grid=False) (slow_gaussian_grid does not use psi) ===')
for kw in (dict(), dict(use_double=True)):
    p2 = LombScargleAsyncProcess(**kw); r = p2.run([(t, y, dy)], freqs=[f], fast_grid=False); p2.finish(); g2 = np.copy(r[0][1])
    print('  %-22s LS power vs astropy maxabs = %.2e' % (kw, np.abs(g2-a).max()))

print('=== 4. own-psi float32 + sigma_eff on the failing grids ===')
for fmin, fmax in ((0.1, 3.0), (1.0, 3.0), (1.5, 3.0), (2.0, 3.0), (0.5, 1.0)):
    f, k0, nf = grid(fmin, fmax); a = LombScargle(t, y, dy).power(f)
    row = []
    for sig in (4, int(np.ceil(4.0*(k0+nf)/nf))):
        p3 = LombScargleAsyncProcess(sigma=sig); M3 = p3.allocate([(t, y, dy)], nfreqs=[nf], k0s=[k0]); p3.memory = M3
        m3 = M3[0]; m3.nfft_mem_w.precomp_psi = True; m3.nfft_mem_w.allocate_precomp_psi(n0=m3.n0)
        r = p3.run([(t, y, dy)], freqs=[f], memory=p3.memory); p3.finish(); g3 = np.copy(r[0][1])
        d = np.abs(g3-a); row.append('sigma=%d: maxabs=%.1e rel=%.1e corr=%.5f' % (sig, d.max(), d.max()/a.max(), np.corrcoef(a, g3)[0, 1]))
    print('  fmin=%-4g fmax=%-3g k0/nf=%.2f | %s' % (fmin, fmax, k0/nf, ' | '.join(row)))
