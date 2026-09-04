import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc
import cuvarbase.memory.lombscargle_memory as lsm
import cuvarbase.lombscargle as lsmod
from cuvarbase.lombscargle import LombScargleAsyncProcess

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def bench(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.median(ts)

nfx = 219000
print("np.array([True]*%d): %.1f ms ; np.ones(nf, bool): %.3f ms" % (nfx, 1e3 * bench(lambda: np.array([True] * nfx)), 1e3 * bench(lambda: np.ones(nfx, bool))))

for label, N, T, fmax, nlc in [("ZTF-like", 300, 3650.0, 20.0, 50), ("Kepler-like", 65000, 1400.0, 50.0, 10)]:
    freqs = grid(1.0 / (3 * T), fmax, T, spp=3)
    lcs = [make_lc(N=N, T=T, f0=1 + i, seed=i) for i in range(nlc)]
    proc = LombScargleAsyncProcess(use_double=False, sigma=4, m=8, autoset_m=False)
    proc.batched_run_const_nfreq(lcs[:1], freqs=freqs)
    res0 = proc.batched_run_const_nfreq(lcs, freqs=freqs)
    t_ship = bench(lambda: proc.batched_run_const_nfreq(lcs, freqs=freqs))
    # host-side fix: numpy reductions instead of Python builtins in weights()/setdata()/get_k0
    lsm.weights = lambda err: np.power(err, -2) / np.sum(np.power(err, -2))
    import builtins
    _mn = lambda *a, **k: np.min(a[0]).item() if len(a) == 1 and not k else builtins.min(*a, **k)
    _mx = lambda *a, **k: np.max(a[0]).item() if len(a) == 1 and not k else builtins.max(*a, **k)
    lsm.min = _mn; lsm.max = _mx; lsmod.min = _mn; lsmod.max = _mx
    res1 = proc.batched_run_const_nfreq(lcs, freqs=freqs)
    t_fix = bench(lambda: proc.batched_run_const_nfreq(lcs, freqs=freqs))
    same = max(np.abs(a[1] - b[1]).max() for a, b in zip(res0, res1))
    print("%s N=%d nf=%d, %d LCs/call: shipped %.2f ms/LC -> numpy reductions %.2f ms/LC (%.1fx); max|diff|=%.1e" % (label, N, len(freqs), nlc, 1e3 * t_ship / nlc, 1e3 * t_fix / nlc, t_ship / t_fix, same))
    # restore
    import builtins
    lsm.weights = lambda err: np.power(err, -2) / sum(np.power(err, -2))
    lsm.min = builtins.min; lsm.max = builtins.max; lsmod.min = builtins.min; lsmod.max = builtins.max
    del proc
