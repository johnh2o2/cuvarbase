"""Compact injection-recovery: LRT (null-calibrated) vs BLS, white and red noise."""
import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
from cuvarbase.bls import eebls_gpu_fast

rng = np.random.RandomState(5)
t = make_times(rng); n = len(t)
P, dur = 5.3, 0.22
periods = np.exp(np.linspace(np.log(2), np.log(18), 16)); periods[np.argmin(np.abs(periods-P))] = P
freqs = np.sort(1/periods)
proc = NUFFTLRTAsyncProcess()
def lrt(y, flat=False):
    best = (-np.inf, np.nan)
    kw = {}
    if flat: kw = dict(estimate_psd=False, psd=np.ones(2*n, np.float32), nf=2*n)
    for p in periods:
        ne = int(min(max(round(2*p/dur), 8), 48))
        s = proc.run(t, y, np.array([p]), durations=np.array([dur]), epochs=np.linspace(0, p, ne, endpoint=False), **kw).max()
        if s > best[0]: best = (s, p)
    return best
def bls(y, dy):
    pw = eebls_gpu_fast(t, y, dy, freqs, qmin=0.005, qmax=0.08)
    i = int(np.argmax(pw)); return pw[i], 1/freqs[i]
def hit(p): return any(abs(p-x)/x < 0.01 for x in (P, 2*P, P/2))
sw = 3e-3
for name, sred, depths in (('white', 0.0, [0.006, 0.010]), ('red_3x', 3*sw, [0.016, 0.028])):
    t0 = time.time()
    meth = {'lrt': lambda y, dy: lrt(y), 'bls': bls}
    if sred > 0: meth['lrt_flat'] = lambda y, dy: lrt(y, True)
    nulls = {k: [] for k in meth}
    for i in range(10):
        y = 1 + sw*rng.randn(n) + (ou_noise(rng, t, sred, 0.8) if sred else 0); dy = np.full(n, sw)
        for k, f in meth.items(): nulls[k].append(f(y, dy)[0])
    thr = {k: np.percentile(v, 90) for k, v in nulls.items()}
    res = {}
    for d in depths:
        hits = {k: 0 for k in meth}; stats = {k: [] for k in meth}
        for i in range(10):
            y = 1 + sw*rng.randn(n) + (ou_noise(rng, t, sred, 0.8) if sred else 0) + box(t, P, rng.rand()*P, dur, d); dy = np.full(n, sw)
            for k, f in meth.items():
                s, p = f(y, dy); stats[k].append(s)
                hits[k] += int(s > thr[k] and hit(p))
        res[d] = {k: (hits[k]/10, np.median(stats[k])/thr[k]) for k in meth}
    print('%s: null p90 %s  (%.0fs)' % (name, {k: round(v, 3) for k, v in thr.items()}, time.time()-t0))
    for d in depths: print('   depth %.3f: completeness (median stat/threshold): %s' % (d, {k: (v[0], round(v[1], 2)) for k, v in res[d].items()}))
