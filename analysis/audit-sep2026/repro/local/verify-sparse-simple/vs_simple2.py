import numpy as np
import pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls
from cuvarbase.utils import subtract_epoch
kern_s = compile_sparse_bls(block_size=64, use_simple=True)
print("seed  nn  simple_max  f  q_best  phi_best  W(box,f64)  1-W  nobs_in_box/N  cpu_at_f")
for seed in range(6):
    r = np.random.RandomState(seed); nn = 20 + seed % 30
    t = np.concatenate([n + 0.3 * np.sort(r.rand(6)) for n in range(nn)])
    y = 12.0 + 0.01 * r.randn(len(t)); dy = 0.01 * np.ones_like(y)
    fr = np.linspace(0.995, 1.005, 101)
    ps, ss = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
    pc, _ = sparse_bls_cpu(t, y, dy, fr)
    k = int(np.argmax(ps)); q, phi0 = ss[k]
    t64, _ = subtract_epoch(t)
    ph = (t64 * fr[k]) % 1.0
    w = (1 / dy**2) / np.sum(1 / dy**2)
    inbox = ((ph - phi0) % 1.0) < q
    W = w[inbox].sum()
    print("%4d %3d %9.4f %.4f %.4f %.4f %.9f %.2e %d/%d %.4f" % (seed, nn, ps[k], fr[k], q, phi0, W, 1 - W, inbox.sum(), len(t), pc[k]))
