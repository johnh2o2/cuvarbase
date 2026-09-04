"""Minimal deterministic reproduction of the floorf-on-double gridding bug in
fast_gaussian_grid (cunfft.cu:151). Installed tree untouched; the patched kernel is
compiled from a scratch copy via a find_kernel monkeypatch."""
import numpy as np, warnings; warnings.simplefilter('ignore')
import pycuda.driver as cuda
import cuvarbase.cunfft as cm
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.utils import find_kernel as _fk

def patched_find_kernel(name):
    if name != 'cunfft': return _fk(name)
    src = open(_fk('cunfft')).read(); assert src.count('floorf(') == 1
    p = '/workspace/scratch/cunfft_floor_patched.cu'; open(p, 'w').write(src.replace('floorf(', 'floor('))
    return p

def ref_grid(t, y, ng, m, b):
    """float64 numpy reference of the Gaussian gridding, floor() in float64."""
    x0, xf = t.min(), t.max(); xval = (t - x0) / (xf - x0)
    g = np.zeros(ng)
    for ti, yi in zip(xval, y):
        u = int(np.floor(ng * ti - m))
        for k in range(2 * m + 1):
            g[(u + k) % ng] += yi * np.exp(-((ng * ti - (u + k)) ** 2) / b) / np.sqrt(np.pi * b)
    return g

nf, sigma, m = 32, 2, 4; ng = sigma * nf
K = 20                                   # ng*t1 = K - 2^-30: floor is K-1 in double, K after float32 rounding
t1 = (K - 2.0 ** -30) / ng               # exact in binary (ng is a power of two)
t = np.array([0.0, t1, 1.0]); y = np.array([0.0, 1.0, 0.0])
arg = ng * t1 - m
print("ng*xval-m = %.17g ; floor(double)=%d ; floorf(float32)=%d" % (arg, np.floor(arg), np.floor(np.float32(arg))))

def grids(use_double, patched):
    cm.find_kernel = patched_find_kernel if patched else _fk
    p = NFFTAsyncProcess(use_double=use_double, sigma=sigma, m=m, autoset_m=False)
    gf = p.run([(t, y, nf)], just_return_gridded_data=True, fast_grid=True)[0].copy()
    p2 = NFFTAsyncProcess(use_double=use_double, sigma=sigma, m=m, autoset_m=False)
    gs = p2.run([(t, y, nf)], just_return_gridded_data=True, fast_grid=False)[0].copy()
    cuda.Context.synchronize(); return np.asarray(gf, float), np.asarray(gs, float)

b = 2 * sigma * m / ((2 * sigma - 1) * np.pi)
gr = ref_grid(t, y, ng, m, b)
for use_double in (False, True):
    for patched in (False, True):
        gf, gs = grids(use_double, patched)
        nz = np.nonzero(gf)[0]
        print("use_double=%d patched=%d: fast-grid nonzero cells %s | max|fast-slow|=%.2e | max|fast-ref|=%.2e | max|slow-ref|=%.2e"
              % (use_double, patched, list(nz), np.max(np.abs(gf - gs)), np.max(np.abs(gf - gr)), np.max(np.abs(gs - gr))))
cm.find_kernel = _fk
