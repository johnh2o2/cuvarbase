import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase import cufinufft_backend as cb

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def run(proc, t, y, dy, freqs, **kw):
    r = proc.run([(t, y, dy)], freqs=freqs, **kw); proc.finish()
    return np.array(r[0][1][:len(freqs)], float)

print("=== cufinufft LS vs astropy: N, band (k0/nf), eps ===")
procC = LombScargleAsyncProcess(use_cufinufft=True)
procN = LombScargleAsyncProcess(sigma=4, m=8, autoset_m=False)
for N in (100, 1000, 20000):
    t, y, dy = make_lc(N=N, T=365.0, f0=7.3, hetero=True, seed=N)
    for (fmin, fmax) in [(1.0 / (5 * 365.0), 20.0), (5.0, 10.0), (20.0, 30.0), (40.0, 50.0)]:
        freqs = grid(fmin, fmax, 365.0)
        ref = LombScargle(t, y, dy).power(freqs, method='cython') if N * len(freqs) < 4e9 else gls_numpy_fast(t, y, dy, freqs)
        for eps in (1e-6, 1e-3):
            p = run(procC, t, y, dy, freqs, eps=eps)
            report("cufinufft N=%d band %.1f-%.1f k0/nf=%.2f eps=%g" % (N, fmin, fmax, get_k0(freqs) / len(freqs), eps), ref, p, freqs)
        p = run(procN, t, y, dy, freqs)
        report("custom    N=%d band %.1f-%.1f k0/nf=%.2f" % (N, fmin, fmax, get_k0(freqs) / len(freqs)), ref, p, freqs)
print("plan cache keys:", list(cb._plan_cache.keys()))

print("=== cufinufft large k*t float32: T=3650 fmax=50 ===")
t, y, dy = make_lc(N=500, T=3650.0, f0=23.456, seed=5)
freqs = grid(1.0 / (5 * 3650.0), 50.0, 3650.0)
ref = gls_numpy_fast(t, y, dy, freqs)
p = run(procC, t, y, dy, freqs)
report("cufinufft T=3650 fmax=50", ref, p, freqs)
d = np.abs(ref - p); q = len(d) // 5
print("   quintile max err:", ["%.1e" % d[i*q:(i+1)*q].max() for i in range(5)])

print("=== plan cache: alternate two LCs with different N, same nf ===")
freqs = grid(1.0 / (5 * 365.0), 20.0, 365.0)
lcs = [make_lc(N=N, T=365.0, f0=3.3, seed=N) for N in (100, 700, 100, 700)]
for i, (t, y, dy) in enumerate(lcs):
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    p = run(procC, t, y, dy, freqs)
    report("alternating N=%d call %d" % (len(t), i), ref, p, freqs)
print("plan cache size:", len(cb._plan_cache))

print("=== cufinufft + use_double=True ===")
try:
    procD = LombScargleAsyncProcess(use_cufinufft=True, use_double=True)
    t, y, dy = lcs[0]
    p = run(procD, t, y, dy, freqs)
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    report("cufinufft use_double=True", ref, p, freqs)
except Exception as e:
    print("cufinufft use_double=True raised: %r" % (e,))

print("=== cufinufft in batched_run_const_nfreq ===")
res = procC.batched_run_const_nfreq(lcs, freqs=freqs, batch_size=2)
for (t, y, dy), (f, p) in zip(lcs, res):
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    report("batched cufinufft N=%d" % len(t), ref, p, freqs)
