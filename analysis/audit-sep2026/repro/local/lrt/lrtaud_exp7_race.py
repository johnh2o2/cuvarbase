"""Is the adjoint-NFFT result synchronized before compute_nufft reads it? And corrected reuse timing."""
import numpy as np, sys, time, gc
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
import pycuda.driver as cuda

rng = np.random.RandomState(0)
for n in (600, 5000, 50000):
    t = np.sort(rng.uniform(0, 90, n)); nf = 2*n
    proc = NUFFTLRTAsyncProcess()
    y = rng.randn(n).astype(np.float32); y -= y.mean()
    proc.compute_nufft(t, y, nf)
    # (a) current path: read immediately vs after full sync
    bad = 0
    for i in range(50):
        ghat = proc.nufft_proc.run([(t.astype(np.float32), y, nf)])[0]
        Y1 = ghat.copy()
        cuda.Context.synchronize()
        Y2 = ghat.copy()
        bad += int(not np.array_equal(Y1, Y2))
    # (b) memory kept alive (documented reuse API): read immediately vs after sync
    mem = proc.nufft_proc.allocate([(t.astype(np.float32), y, nf)])
    bad2 = 0
    for i in range(50):
        mem[0].y = (y * (1 + 0.01*i)).astype(np.float32)
        ghat = proc.nufft_proc.run([(t, y, nf)], memory=mem)[0]
        Y1 = ghat.copy(); mem[0].stream.synchronize(); Y2 = ghat.copy()
        bad2 += int(not np.array_equal(Y1, Y2))
    # (c) explain: does dropping the NFFTMemory (cuMemFree) sync? time current path w/ and w/o gc
    def cur(): return proc.compute_nufft(t, y, nf)
    def reuse_sync():
        mem[0].y = y
        g = proc.nufft_proc.run([(t, y, nf)], memory=mem)[0]; mem[0].stream.synchronize(); return g
    ts = []
    for f in (cur, reuse_sync):
        v = []
        for _ in range(40):
            t0 = time.perf_counter(); f(); v.append(time.perf_counter()-t0)
        ts.append((np.median(v)*1e3, np.min(v)*1e3))
    a = cur(); b = reuse_sync()
    print('n=%d nf=%d: immediate-read mismatches: current path %d/50, kept-memory path %d/50 | current %.2f ms (min %.2f) vs reuse+sync %.2f ms (min %.2f) | results equal: %s' % (
        n, nf, bad, bad2, ts[0][0], ts[0][1], ts[1][0], ts[1][1], np.allclose(a, b, rtol=1e-4, atol=1e-6*np.abs(a).max())))
