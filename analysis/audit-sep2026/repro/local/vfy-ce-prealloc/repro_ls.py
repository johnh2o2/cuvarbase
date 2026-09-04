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

B = lc(900, 2); C = lc(300, 5)
print("=== LombScargle preallocate(streams=None) stale-read check ===")
f = 0.001*(50+np.arange(3000))
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
