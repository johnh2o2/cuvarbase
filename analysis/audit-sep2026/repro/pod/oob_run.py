"""On-device check of finding 38 on the default eebls_transit path.
usage: python oob_run.py default            -> eebls_transit(t,y,dy,fmin=0.02,fmax=0.5) at default max_memory
       python oob_run.py plain  FMIN FMAX MAXMEM OUT.npy
       python oob_run.py padded FMIN FMAX MAXMEM OUT.npy   (same call, but device bin buffers over-allocated 1.3x
                                                           via a wrapped gpuarray.zeros -> identical batch boundaries,
                                                           so any difference isolates the out-of-bounds effect)
"""
import sys, numpy as np
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
import cuvarbase.bls as B
from cuvarbase.bls import eebls_transit, compile_bls
mode = sys.argv[1]
rng = np.random.RandomState(7); ndata = 70000
t = np.sort(rng.uniform(0, 1460., ndata))
P, q, depth = 12.3, 0.03, 0.01
y = 1.0 - depth * (((t / P) % 1) < q) + 0.002 * rng.randn(ndata); dy = np.full(ndata, 0.002)
fr = compile_bls()
if mode == 'padded':
    orig = B.gpuarray
    class NS:
        to_gpu = staticmethod(orig.to_gpu)
        @staticmethod
        def zeros(n, dtype=np.float32):
            return orig.zeros(int(n * 1.3) if n > 1_000_000 else n, dtype=dtype)
    B.gpuarray = NS
free = cuda.mem_get_info()[0]
if mode == 'default':
    print("free=%.2f GB -> max_memory=%.3e" % (free / 1e9, 0.9 * free))
    try:
        freqs, p, sols = eebls_transit(t, y, dy, fmin=0.02, fmax=0.5, functions=fr)
        cuda.Context.synchronize()
        print("returned: nfreq=%d max=%.4f at P=%.4f #zero=%d #nonfinite=%d #>1=%d" % (len(freqs), p.max(), 1/freqs[np.argmax(p)], (p == 0).sum(), (~np.isfinite(p)).sum(), (p > 1).sum()))
    except Exception as e:
        print("EXCEPTION:", repr(e))
else:
    fmin, fmax, mm, out = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    try:
        freqs, p, sols = eebls_transit(t, y, dy, fmin=fmin, fmax=fmax, max_memory=mm, functions=fr)
        cuda.Context.synchronize()
        p = np.asarray(p)[:len(freqs)]
        qs = np.array([s[0] for s in sols])[:len(freqs)]
        np.save(out, np.vstack([freqs, p, qs]))
        print("%s returned: nfreq=%d max=%.4f at P=%.4f #zero=%d #nonfinite=%d" % (mode, len(freqs), p.max(), 1/freqs[np.argmax(p)], (p == 0).sum(), (~np.isfinite(p)).sum()))
    except Exception as e:
        print(mode, "EXCEPTION:", repr(e))
