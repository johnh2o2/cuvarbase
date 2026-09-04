"""Root cause of the double-precision NFFT error: fast_gaussian_grid computes
u = (int) floorf(ng*xval - m) -- floorf rounds the DOUBLE argument to float32 first, so points
whose fractional grid position is within one float32 ulp of the next integer are gridded one cell
too far (with weights computed from the exact fractional part). Verify by replicating the misplaced
u in a float64 numpy replica and matching the GPU output to ~1e-9."""
import numpy as np, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.cunfft import NFFTAsyncProcess
def replica(t, y, nf, sigma, m, floorf_bug):
    ng = int(sigma*nf); b = 2*sigma*m/((2*sigma-1)*np.pi)
    x0, xf = t.min(), t.max(); xval = (t - x0)/(xf - x0)
    xg = m + (ng*xval - np.floor(ng*xval))
    q1 = np.exp(-xg*xg/b)/np.sqrt(b*np.pi); q2 = np.exp(2*xg/b); q3 = np.exp(-np.arange(2*m+1)**2/b)
    arg = ng*xval - m
    u = np.floor(arg.astype(np.float32)).astype(np.int64) if floorf_bug else np.floor(arg).astype(np.int64)
    nbad = int(np.sum(u != np.floor(arg).astype(np.int64)))
    grid = np.zeros(ng); Q = q1.copy()
    for k in range(2*m+1):
        np.add.at(grid, (u + k) % ng, Q*q3[k]*y); Q = Q*q2
    G = np.fft.ifft(grid)*ng; k = np.arange(nf); n0_ = (x0/(xf-x0))*ng
    return G[:nf]*np.exp(1j*2*np.pi*n0_*k/ng)*np.exp(b*(np.pi*k/ng)**2), nbad
for seed, n0, nf, T in [(4, 2000, 4000, 100.), (5, 2000, 4000, 100.), (4, 6000, 200000, 3650.)]:
    rng = np.random.RandomState(seed); t = np.sort(rng.rand(n0)*T); y = rng.randn(n0); l1 = np.sum(np.abs(y))
    p = NFFTAsyncProcess(use_double=True, sigma=2, m=8)
    g = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); g = np.asarray(g).copy()
    ks = np.arange(0, nf//2, max(1, nf//40))
    ex = np.array([np.sum(y*np.exp(2j*np.pi*k*t/(t.max()-t.min()))) for k in ks])
    rb, nbad = replica(t, y, nf, 2, 8, True); rg, _ = replica(t, y, nf, 2, 8, False)
    print("seed=%d n0=%d nf=%d ng=%d: points with floorf-misplaced u: %d/%d | GPU(double) vs exact %.2e | replica WITH floorf bug vs GPU %.2e | replica without bug vs GPU %.2e | replica without bug vs exact %.2e"
          % (seed, n0, nf, 2*nf, nbad, n0, np.max(np.abs(g[ks]-ex))/l1, np.max(np.abs(g[ks]-rb[ks]))/l1, np.max(np.abs(g[ks]-rg[ks]))/l1, np.max(np.abs(rg[ks]-ex))/l1))
