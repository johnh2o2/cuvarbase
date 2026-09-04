"""Does floorf->floor change the float32 (default) path? Compare full NFFT output bitwise."""
import numpy as np, warnings; warnings.simplefilter('ignore')
import pycuda.driver as cuda
import cuvarbase.cunfft as cm
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.utils import find_kernel as _fk
def pfk(name):
    if name != 'cunfft': return _fk(name)
    src = open(_fk('cunfft')).read(); p = '/workspace/scratch/cunfft_floor_patched.cu'; open(p, 'w').write(src.replace('floorf(', 'floor(')); return p
for seed, n0, nf, T in [(4, 2000, 4000, 100.), (4, 6000, 200000, 3650.), (7, 300, 912499, 3650.)]:
    rng = np.random.RandomState(seed); t = np.sort(rng.rand(n0) * T); y = rng.randn(n0)
    out = {}
    for dbl in (False, True):
        for patched in (False, True):
            cm.find_kernel = pfk if patched else _fk
            p = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8, autoset_m=False)
            g = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); out[(dbl, patched)] = np.asarray(g).copy(); del p
    l1 = np.sum(np.abs(y))
    ks = np.arange(0, nf, max(1, nf // 200)); ex = np.array([np.sum(y * np.exp(2j * np.pi * k * t / (t.max() - t.min()))) for k in ks])
    print("seed=%d n0=%d nf=%d ng=%d: f32 patched-vs-unpatched bitwise identical: %s (max|d|=%.1e) | double unpatched vs exact %.2e | double patched vs exact %.2e | f32 vs exact %.2e"
          % (seed, n0, nf, 2 * nf, np.array_equal(out[(False, False)], out[(False, True)]), np.max(np.abs(out[(False, False)] - out[(False, True)])),
             np.max(np.abs(out[(True, False)][ks] - ex)) / l1, np.max(np.abs(out[(True, True)][ks] - ex)) / l1, np.max(np.abs(out[(False, False)][ks] - ex)) / l1))
cm.find_kernel = _fk
