"""Independent minimal reproduction of finding 34 + test of the proposed fix (monkeypatched, no tree edits)."""
import numpy as np, sys
import pycuda.autoprimaryctx  # noqa
import pycuda.driver as cuda
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.lombscargle import LombScargleAsyncProcess

rng = np.random.RandomState(0)
def lc(N, seed):
    r = np.random.RandomState(seed)
    t = np.sort(r.uniform(0, 100, N)); y = 0.3*np.sin(2*np.pi*t/1.7) + 0.05*r.randn(N); dy = 0.05*np.ones(N)
    return t, y, dy
F = np.linspace(0.05, 5.0, 4000)
B = lc(900, 2); C = lc(300, 5)

def rel(a, b): return np.max(np.abs(a-b))/max(np.max(np.abs(b)), 1e-30)

for fast in (False, True):
    print("=== use_fast=%s ===" % fast)
    proc = ConditionalEntropyAsyncProcess(use_fast=fast)
    fB = np.copy(proc.run([B], freqs=[F])[0][1]); proc.finish()
    fC = np.copy(proc.run([C], freqs=[F])[0][1]); proc.finish()
    print("fresh: std(fB)=%.3g std(fC)=%.3g" % (fB.std(), fC.std()))

    # (1) as shipped: preallocate then run
    proc.preallocate(max_nobs=900, freqs=F, nlcs=1)
    m = proc.memory[0]
    print("[shipped] memory.stream=%r  freqs_g[:3]=%s  freqs_g.max()=%g  mem.freqs[:3]=%s" % (m.stream, m.freqs_g.get()[:3], m.freqs_g.get().max(), m.freqs[:3]))
    rB = np.copy(proc.run([B], freqs=[F])[0][1]); proc.finish(); cuda.Context.synchronize()
    print("[shipped] run(B) after preallocate: std=%.3g  rel-vs-fresh=%.3g  (constant output => every freq evaluated at f=0)" % (rB.std(), rel(rB, fB)))

    # (2) fix A only: upload freqs, keep stream=None  -> check stale read race over repeated alternations
    for mm in proc.memory: mm.transfer_freqs_to_gpu()
    nbad_early = nbad_late = 0
    for k in range(15):
        for data, ref in ((B, fB), (C, fC)):
            r = proc.run([data], freqs=[F]); proc.finish(); early = np.copy(r[0][1]); cuda.Context.synchronize(); late = np.copy(r[0][1])
            nbad_early += rel(early, ref) > 1e-6; nbad_late += rel(late, ref) > 1e-6
    print("[freqs uploaded, stream=None] over 30 runs: stale-after-finish() count=%d ; wrong-after-Context.synchronize() count=%d" % (nbad_early, nbad_late))

    # (3) full fix: freqs uploaded + memory streams = proc.streams  (finish() syncs proc.streams)
    if len(proc.streams) < 1: proc._create_streams(1)
    proc.preallocate(max_nobs=900, freqs=F, nlcs=1, streams=proc.streams)
    for mm in proc.memory: mm.transfer_freqs_to_gpu()
    nbad = 0
    for k in range(15):
        for data, ref in ((B, fB), (C, fC)):
            r = proc.run([data], freqs=[F]); proc.finish(); early = np.copy(r[0][1])
            nbad += rel(early, ref) > 1e-6
    print("[full fix: freqs uploaded + streams=proc.streams] over 30 runs: wrong-after-finish() count=%d" % nbad)

    # (4) does the *fresh* (memory=None) path have any of this?  (default path sanity)
    proc2 = ConditionalEntropyAsyncProcess(use_fast=fast)
    nbad = 0
    for k in range(10):
        for data, ref in ((B, fB), (C, fC)):
            r = proc2.run([data], freqs=[F]); proc2.finish(); nbad += rel(np.copy(r[0][1]), ref) > 1e-6
    print("[default run() path, no preallocate] over 20 runs: wrong-after-finish() count=%d" % nbad)
    # (5) batched_run_const_nfreq sanity
    res = proc2.batched_run_const_nfreq([B, C], freqs=F, batch_size=2)
    print("[batched_run_const_nfreq] rel B=%.3g C=%.3g" % (rel(res[0][1], fB), rel(res[1][1], fC)))

print("=== LombScargle preallocate(streams=None) stale-read check ===")
f = np.linspace(0.05, 5.0, 3000)
proc = LombScargleAsyncProcess()
fB = np.copy(proc.run([B], freqs=[f])[0][1]); proc.finish()
fC = np.copy(proc.run([C], freqs=[f])[0][1]); proc.finish()
proc.preallocate(max_nobs=900, nlcs=1, freqs=f)
print("LS memory.stream=%r" % proc.memory[0].stream)
nbad_early = nbad_late = 0
for k in range(15):
    for data, ref in ((B, fB), (C, fC)):
        r = proc.run([data], freqs=[f]); proc.finish(); early = np.copy(r[0][1]); cuda.Context.synchronize(); late = np.copy(r[0][1])
        nbad_early += rel(early, ref) > 1e-5; nbad_late += rel(late, ref) > 1e-5
print("LS preallocate(streams=None) over 30 runs: stale-after-finish()=%d ; wrong-after-ctx-sync=%d" % (nbad_early, nbad_late))
if len(proc.streams) < 1: proc._create_streams(1)
proc.preallocate(max_nobs=900, nlcs=1, freqs=f, streams=proc.streams)
nbad = 0
for k in range(15):
    for data, ref in ((B, fB), (C, fC)):
        r = proc.run([data], freqs=[f]); proc.finish(); nbad += rel(np.copy(r[0][1]), ref) > 1e-5
print("LS preallocate(streams=proc.streams) over 30 runs: wrong-after-finish()=%d" % nbad)
