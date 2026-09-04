"""Candidate fix check: sigma=4 (or 2nf modes truncated) makes the full returned band accurate; jitter and S values."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram
def adjoint_dft_chunked(t, y, nf, chunk=2000):
    t = np.asarray(t, np.float64); y = np.asarray(y, np.float64)
    x = t / (t.max() - t.min()); out = np.empty(nf, np.complex128)
    for a in range(0, nf, chunk):
        k = np.arange(a, min(nf, a+chunk)); out[a:a+len(k)] = np.exp(2j*np.pi*np.outer(k, x)) @ y
    return out
rng = np.random.RandomState(0)
t = make_times(rng); n = len(t); nf = 2*n
y = 3e-3*rng.randn(n); y -= y.mean(); E = adjoint_dft_chunked(t, y, nf); rms = np.sqrt(np.mean(np.abs(E)**2))
for lab, kw, mult in (('sigma=2 (default)', {}, 1), ('sigma=4', {'sigma': 4.0}, 1), ('sigma=2, 2nf modes truncated', {}, 2)):
    for dbl in (False, True):
        p = NUFFTLRTAsyncProcess(use_double=dbl, **kw)
        G = p.compute_nufft(t, y, mult*nf).astype(np.complex128)[:nf]
        G2 = p.compute_nufft(t, y, mult*nf).astype(np.complex128)[:nf]
        lo, hi = slice(1, nf//2), slice(nf//2, nf)
        print('%-30s double=%-5s m=%d: max|G-E|/rms lower=%.1e upper=%.1e  repeat upper=%.1e' % (lab, dbl, p.nufft_proc.get_m(mult*nf, y=y.astype(p.real_type)),
              np.abs(G[lo]-E[lo]).max()/rms, np.abs(G[hi]-E[hi]).max()/rms, np.abs(G[hi]-G2[hi]).max()/rms))
# TESS-like default path: S_true and jitter with sigma=4
tt = np.arange(0, 27, 2/1440.); tt = tt[(tt < 13) | (tt > 14)]; n = len(tt); P, dur, depth, sig = 5.3, 0.12, 0.002, 1e-3
r3 = np.random.RandomState(7); yy = 1 + sig*r3.randn(n) + box(tt, P, 1.3, dur, depth)
for lab, kw in (('sigma=2 f32 (default)', {}), ('sigma=4 f32', {'sigma': 4.0}), ('sigma=4 f64', {'sigma': 4.0, 'use_double': True}), ('sigma=2 f64', {'use_double': True})):
    p = NUFFTLRTAsyncProcess(**kw)
    s = [p.run(tt, yy, np.array([P]), durations=np.array([dur]), epochs=np.array([1.3]))[0,0,0] for _ in range(3)]
    print('%-24s S_true(run) = %s' % (lab, ' '.join('%.4f' % v for v in s)))
