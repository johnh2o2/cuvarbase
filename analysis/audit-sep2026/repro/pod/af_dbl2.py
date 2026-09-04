"""Why does the double NFFT give 1e-9 in one harness and 3e-5 in another? Same data, three call styles."""
import numpy as np, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.cunfft import NFFTAsyncProcess
rng = np.random.RandomState(4)
n0, nf, T = 2000, 4000, 100.
t = np.sort(rng.rand(n0)*T); y = rng.randn(n0); l1 = np.sum(np.abs(y))
ks = np.arange(0, nf//2, 25); ex = np.array([np.sum(y*np.exp(2j*np.pi*k*t/(t.max()-t.min()))) for k in ks])
for dbl in (False, True):
    p = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8)
    g1 = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); g1 = np.asarray(g1).copy()
    g1b = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); g1b = np.asarray(g1b).copy()
    m_ = p.allocate([(t, y, nf)])[0]
    p.run([(t, y, nf)], memory=[m_]); cuda.Context.synchronize(); g2 = m_.ghat_c.copy()
    m_.ghat_g.fill(0); cuda.Context.synchronize()
    p.run([(t, y, nf)], memory=[m_]); cuda.Context.synchronize(); g3 = m_.ghat_c.copy()
    e = lambda g: np.max(np.abs(g[ks]-ex))/l1
    print("use_double=%-5s: fresh#1 %.2e | fresh#2 %.2e | alloc+run(memory) %.2e | +fill(0)+run again %.2e ; fresh#1 vs fresh#2 max|d|=%.2e ; fresh vs alloc: %.2e ; m=%d b=%.3f n=%d dtype t=%s ghat=%s"
          % (dbl, e(g1), e(g1b), e(g2), e(g3), np.max(np.abs(g1-g1b)), np.max(np.abs(g1-g2)), m_.m, m_.b, m_.n, m_.t.dtype, m_.ghat_c.dtype))
    # is the error concentrated at particular k?
    err = np.abs(g1[ks]-ex)/l1
    print("   fresh#1 error vs k: k=0: %.1e, k=25: %.1e, k=1000: %.1e, k=1975: %.1e ; median %.1e" % (err[0], err[1], err[40], err[-1], np.median(err)))
