"""Isolate fix F (floorf->floor) only: NFFT vs exact (k<nf/2), and LS vs astropy, unpatched vs F-patched,
float32 (default) and double. Float32 run-to-run atomicAdd nondeterminism given as baseline."""
import sys; sys.path.insert(0, '/workspace/scratch')
import numpy as np, warnings; warnings.simplefilter('ignore')
import pycuda.driver as cuda
import patches
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.lombscargle import LombScargleAsyncProcess
from astropy.timeseries import LombScargle

def nfft(t, y, nf, dbl):
    p = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8, autoset_m=False)
    g = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); g = np.asarray(g).copy(); del p; return g
print("== NFFT adjoint vs exact DFT (k < nf/2), errors relative to ||y||_1 ==")
for seed, n0, nf, T in [(4, 2000, 4000, 100.), (4, 6000, 200000, 3650.), (4, 6000, 894250, 3650.), (7, 300, 912499, 3650.)]:
    rng = np.random.RandomState(seed); t = np.sort(rng.rand(n0) * T); y = rng.randn(n0); l1 = np.sum(np.abs(y))
    ks = np.arange(0, nf // 2, max(1, nf // 60)); ex = np.array([np.sum(y * np.exp(2j * np.pi * k * t / (t.max() - t.min()))) for k in ks])
    patches.apply(F=False, P=False, B=False); f32a = nfft(t, y, nf, False); f32b = nfft(t, y, nf, False); dbl0 = nfft(t, y, nf, True)
    patches.apply(F=True, P=False, B=False);  f32F = nfft(t, y, nf, False); dblF = nfft(t, y, nf, True)
    e = lambda g: np.max(np.abs(g[ks] - ex)) / l1
    print("n0=%d nf=%d ng=%d | f32 unpatched %.2e (run2 %.2e, run-to-run %.1e) f32 F-patched %.2e (vs unpatched %.1e) | double unpatched %.2e -> F-patched %.2e"
          % (n0, nf, 2 * nf, e(f32a), e(f32b), np.max(np.abs(f32a - f32b)) / l1, e(f32F), np.max(np.abs(f32a - f32F)) / l1, e(dbl0), e(dblF)))

print("== LS vs astropy (cython), F only ==")
rng = np.random.RandomState(11)
def ls_case(n, T, fsig, fmin, fmax, stride):
    t = np.sort(rng.rand(n) * T); y = 10 + 0.5 * np.sin(2 * np.pi * fsig * t + 0.3) + 0.3 * rng.randn(n); dy = 0.3 * (0.5 + rng.rand(n))
    df = 1. / (5 * T); k0 = max(1, int(round(fmin / df))); nfr = int(fmax / df) - k0; freqs = df * (k0 + np.arange(nfr))
    idx = np.unique(np.concatenate([np.arange(0, nfr, stride), np.argmin(np.abs(freqs - fsig)) + np.arange(-3, 4)]))
    pa = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freqs[idx], method='cython'); ipk = np.argmax(pa)
    for dbl in (False, True):
        row = []
        for F in (False, True):
            patches.apply(F=F, P=False, B=False)
            proc = LombScargleAsyncProcess(use_double=dbl); r = proc.run([(t, y, dy)], freqs=freqs); proc.finish()
            p = np.array(r[0][1], dtype=np.float64); d = np.abs(p[idx] - pa); del proc
            row.append("F=%d max|d|=%.2e peakrel=%.2e (gpu %.4f)" % (F, d.max(), abs(p[idx][ipk] - pa[ipk]) / pa[ipk], p[idx][ipk]))
        print("LS n=%d T=%g nf=%d k0=%d dbl=%d astropy-peak %.4f | %s" % (n, T, nfr, k0, dbl, pa[ipk], " | ".join(row)))
ls_case(1000, 365., 18.0, 0.1, 20., 7)
ls_case(6000, 3650., 40.0, 1.0, 50., 61)
patches.apply(F=False, P=False, B=False)
