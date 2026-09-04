"""Per-template cost: current path vs memory reuse vs host phasor matmul. Shared GPU: repeated medians only."""
import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
import pycuda.driver as cuda

def med(f, reps=30):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return np.median(ts)*1e3, np.min(ts)*1e3

for n in (600, 5000):
    rng = np.random.RandomState(0)
    t = np.sort(rng.uniform(0, 90, n)); nf = 2*n
    proc = NUFFTLRTAsyncProcess()
    tm = proc._generate_template(t, 5.3, 0.0, 0.22, 1.0); tm -= tm.mean()
    tm32 = tm.astype(np.float32)
    proc.compute_nufft(t, tm, nf)  # warm-up/compile
    cur = med(lambda: proc.compute_nufft(t, tm, nf))
    # memory reuse: allocate once, reuse plan/buffers
    mem = proc.nufft_proc.allocate([(t.astype(np.float32), tm32, nf)])
    def reuse():
        mem[0].y = tm32
        return proc.nufft_proc.run([(t, tm32, nf)], memory=mem)[0]
    ru = med(reuse)
    # host phasor matrix (exact adjoint DFT), complex64 GEMV
    x = t / (t.max() - t.min()); E = np.exp(2j*np.pi*np.outer(np.arange(nf), x)).astype(np.complex64)
    gm = med(lambda: E @ tm32)
    # batched: 48 epochs at once
    M = np.stack([(lambda m: m - m.mean())(proc._generate_template(t, 5.3, e, 0.22, 1.0)) for e in np.linspace(0, 5.3, 48, endpoint=False)], 1).astype(np.float32)
    gb = med(lambda: E @ M, reps=10)
    # per-call allocation share: time allocate() alone
    al = med(lambda: proc.nufft_proc.allocate([(t.astype(np.float32), tm32, nf)]))
    print('n=%d nf=%d: compute_nufft (current, alloc per call) median=%.2f ms (min %.2f) | allocate() alone %.2f ms | memory-reuse run %.2f ms (min %.2f) | host complex64 GEMV %.3f ms | host GEMM 48 templates %.2f ms (%.3f/template)' % (
        n, nf, cur[0], cur[1], al[0], ru[0], ru[1], gm[0], gb[0], gb[0]/48))
    # sanity: reuse result equals current result
    a = proc.compute_nufft(t, tm, nf); b = reuse()
    print('   reuse vs current max|d|=%.2e ; GEMV vs current max|d|/rms=%.2e' % (np.abs(a-b).max(), np.abs(E@tm32 - a).max()/np.sqrt(np.mean(np.abs(a)**2))))
