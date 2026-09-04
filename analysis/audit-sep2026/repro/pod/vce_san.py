import numpy as np, sys
import cuvarbase.ce as ce
rng = np.random.RandomState(3)
N = 100
t = np.sort(rng.uniform(0, 30, N)); y = 0.3*np.sin(2*np.pi*t/1.7) + 0.1*rng.randn(N); dy = np.full(N, 0.1)
mode = sys.argv[1]
if mode == 'std':
    # put the brightest point in the LAST phase bin of the LAST frequency -> index nf*NB (one past bins_g)
    imax = np.argmax(y); tt = t - t.mean(); f_last = None
    freqs = np.linspace(0.3, 1.2, 50)
    # choose last freq so that phase of the max point is in bin 9
    for f in np.linspace(1.2, 1.3, 2000):
        ph = np.float32(tt[imax])*np.float32(f); ph = ph - np.floor(ph)
        if int(ph*10) == 9: f_last = f; break
    freqs[-1] = f_last
    proc = ce.ConditionalEntropyAsyncProcess()
    mems = proc.allocate([(t, y, dy)], freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
    b = mems[0].bins_g.get().reshape(len(freqs), 10, 5)
    print("bins_g alloc len", mems[0].bins_g.size, "= nf*NB", len(freqs)*50, "; per-freq sums first/last:", b.sum(axis=(1,2))[[0, -1]], "total", b.sum(), "expected", N*len(freqs))
elif mode == 'wt':
    for dyv in (0.2, 0.05, 0.01):
        dy = np.full(N, dyv)
        freqs = np.linspace(0.3, 1.2, 20)
        proc = ce.ConditionalEntropyAsyncProcess(weighted=True)
        mems = proc.allocate([(t, y, dy)], freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
        proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
        b = mems[0].bins_g.get().reshape(len(freqs), 10, 5)
        print(f"weighted dy={dyv}: per-freq weight sums min/max = {b.sum(axis=(1,2)).min():.3f}/{b.sum(axis=(1,2)).max():.3f} (N={N}); bins_g len {mems[0].bins_g.size}")
