import numpy as np
import cuvarbase.ce as ce
from pycuda import gpuarray
rng = np.random.RandomState(3)
N = 100
t = np.sort(rng.uniform(0, 30, N)); y = 0.3*np.sin(2*np.pi*t/1.7) + 0.1*rng.randn(N); dy = np.full(N, 0.1)
imax = np.argmax(y); tt = np.float32(t - t.mean())
def nm(f):
    ph = tt[imax]*np.float32(f); ph = ph - np.floor(ph); return int(ph*np.float32(10)) % 10
cands = [f for f in np.linspace(0.3, 1.3, 4000) if nm(f) == 9]
freqs = np.concatenate([np.linspace(0.5, 0.9, 63), [cands[0]]])
proc = ce.ConditionalEntropyAsyncProcess()
mems = proc.allocate([(t, y, dy)], freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
m = mems[0]
nb = m.nbins
big = gpuarray.zeros(nb + 8, dtype=np.uint32)   # 8 guard elements past the end
big.fill(np.uint32(0xDEAD))
m.bins_g = big[:nb]                             # view: same base pointer, length nb
proc.run([(t, y, dy)], memory=m if False else mems, freqs=[freqs]); proc.finish()
full = big.get()
print("nbins =", nb, "; guard elements after bins_g:", full[nb:nb+8], "(0xDEAD=%d untouched)" % 0xDEAD)
print("per-freq total of last freq:", full[:nb].reshape(64, 50)[-1].sum(), "(N=100)")
