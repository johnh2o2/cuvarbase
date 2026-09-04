"""NFFT accuracy vs exact adjoint DFT at large grid sizes: float32 vs use_double.
Hypothesis: fast_gaussian_grid uses floorf(ng*xval - m) even under DOUBLE_PRECISION, so for
ng*xval > 2^24 the integer grid index disagrees with precompute_psi's double modflt()."""
import numpy as np, json, warnings
warnings.simplefilter('ignore')
from cuvarbase.cunfft import NFFTAsyncProcess
rng = np.random.RandomState(4)
res = {}
def exact(t, y, ks, T):
    # kernel convention: ghat[k] = sum_j y_j exp(2 pi i k t_j / T) with absolute-t phases (T = tmax - tmin)
    return np.array([np.sum(y * np.exp(2j*np.pi*k*t/T)) for k in ks])
for n0, nf, T in [(2000, 4000, 100.), (6000, 200000, 3650.), (6000, 894250, 3650.), (6000, 894250*2, 3650.)]:
    t = np.sort(rng.rand(n0) * T); y = rng.randn(n0)
    ks = np.unique(np.concatenate([np.arange(0, nf, max(1, nf//7)), [nf//4, nf//3, nf//2 - 1]]))
    ex = exact(t, y, ks, T)
    for dbl in (False, True):
        proc = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8)
        g = proc.run([(t, y, nf)])[0]; proc.finish(); g = np.asarray(g).copy()
        err = np.abs(g[ks] - ex) / np.sum(np.abs(y))
        ng = int(2 * nf)
        print("n0=%d nf=%d T=%g ng=%d use_double=%s: max rel err (|dG|/||y||_1) over sampled k = %.2e ; at k=nf/4: %.2e ; k=nf/2-1: %.2e ; ng*x max=%.2e (2^24=%.2e)"
              % (n0, nf, T, ng, dbl, err.max(), err[ks == nf//4][0], err[ks == nf//2 - 1][0], ng, 2**24))
        res['n%d_nf%d_dbl%d' % (n0, nf, dbl)] = dict(maxerr=float(err.max()), err_quarter=float(err[ks == nf//4][0]), ng=ng)
json.dump(res, open('/workspace/scratch/af_nfft2.json', 'w'), indent=1)
