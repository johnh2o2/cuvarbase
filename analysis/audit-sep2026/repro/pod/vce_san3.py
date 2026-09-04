import numpy as np
import cuvarbase.ce as ce
rng = np.random.RandomState(3)
N = 100
t = np.sort(rng.uniform(0, 30, N)); y = 0.3*np.sin(2*np.pi*t/1.7) + 0.1*rng.randn(N); dy = np.full(N, 0.1)
imax = np.argmax(y); tt = np.float32(t - t.mean())
def nm(f):
    ph = tt[imax]*np.float32(f); ph = ph - np.floor(ph); return int(ph*np.float32(10)) % 10
cands = [f for f in np.linspace(0.3, 1.3, 4000) if nm(f) == 9]
import sys
nf=int(sys.argv[1]); freqs = np.concatenate([np.linspace(0.5, 0.9, nf-1), [cands[0]]])
print("last freq", freqs[-1], "-> max point phase bin", nm(freqs[-1]))
proc = ce.ConditionalEntropyAsyncProcess()
mems = proc.allocate([(t, y, dy)], freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
print("y idx max:", mems[0].y[:N].max(), " y[imax]=", mems[0].y[imax])
b = mems[0].bins_g.get().reshape(len(freqs), 10, 5)
print("bins_g len", mems[0].bins_g.size, "== nf*NB", len(freqs)*50, "; per-freq sums", b.sum(axis=(1,2)), "total", b.sum(), "expected", N*len(freqs))
print("freq0 bin(0,0)=", b[0,0,0], " freq1 bin(0,0)=", b[1,0,0], " freq2 bin(0,0)=", b[2,0,0])
