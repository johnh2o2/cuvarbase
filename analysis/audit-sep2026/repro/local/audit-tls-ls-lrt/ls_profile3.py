"""LS: setdata CPU breakdown at large N; top-of-band float32 phase error vs epoch choice (mean vs min subtraction)."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
import cuvarbase.lombscargle as lsmod
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.memory import LombScargleMemory
from cuvarbase.memory.lombscargle_memory import weights
rng = np.random.RandomState(3); T = 1095.0
def bench(fn, reps=7):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(xs))
for N in (20000, 50000, 200000):
    t = np.sort(rng.rand(N) * T); dy = 0.02 * (1 + rng.rand(N)); y = 15 + dy * rng.randn(N)
    print("N=%d: builtin min(t) %.2f ms, max(t) %.2f ms, weights(dy) [uses builtin sum] %.2f ms, np.min %.3f ms, np.sum %.3f ms" % (N, bench(lambda: min(t)), bench(lambda: max(t)), bench(lambda: weights(dy)), bench(lambda: np.min(t)), bench(lambda: np.sum(dy))))
    mem = LombScargleMemory(4, None, 8, k0=1, buffered_transfer=True, n0_buffer=N, use_fft=False); mem.allocate(nf=10)
    print("   LombScargleMemory.setdata total %.2f ms" % bench(lambda: mem.setdata(t=t, y=y, dy=dy)))

print("\ntop-of-band accuracy vs epoch convention (N=1000, nf=365000, sigma=4, m=8, psi-fixed so that only the phase-factor error remains):")
N = 1000; t = np.sort(rng.rand(N) * T); dy = 0.02 * (1 + rng.rand(N)); y = 15 + 0.1 * np.sin(2 * np.pi * t / 0.37) + dy * rng.randn(N)
nf = 365000; df = 100.0 / nf; freqs = df * (1 + np.arange(nf))
orig_ag = LombScargleMemory.allocate_grids
def allocate_grids(self, **kws):
    r = orig_ag(self, **kws)
    if self.use_fft:
        self.nfft_mem_w.precomp_psi = True; self.nfft_mem_w.allocate_precomp_psi(n0=self.n0_buffer if self.buffered_transfer else kws.get('n0', self.n0))
    return r
LombScargleMemory.allocate_grids = allocate_grids
orig_norm = lsmod.normalize_light_curves
def norm_min(data):
    return [(np.asarray(tt) - np.min(tt), np.asarray(yy) - np.mean(yy), dd) for tt, yy, dd in data]
sl_top = slice(nf - 4000, nf); sl_mid = slice(nf // 2 - 2000, nf // 2 + 2000)
ref_top = LombScargle(t, y, dy).power(freqs[sl_top], method='cython'); ref_mid = LombScargle(t, y, dy).power(freqs[sl_mid], method='cython')
for label, nrm, kw in [("mean-subtracted t (shipped)", orig_norm, {}), ("min-subtracted t", norm_min, {}), ("mean-subtracted, use_double", orig_norm, dict(use_double=True)), ("min-subtracted, use_double", norm_min, dict(use_double=True))]:
    lsmod.normalize_light_curves = nrm
    pr = LombScargleAsyncProcess(**kw); r = pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs); p = np.array(r[0][1], np.float64)
    lsmod.normalize_light_curves = orig_norm
    print("   %-30s mid-band max|d|=%.2e (rel %.1e)  top-band max|d|=%.2e (rel %.1e)" % (label, np.abs(p[sl_mid] - ref_mid).max(), np.abs(p[sl_mid] - ref_mid).max() / ref_mid.max(), np.abs(p[sl_top] - ref_top).max(), np.abs(p[sl_top] - ref_top).max() / ref_top.max()))
