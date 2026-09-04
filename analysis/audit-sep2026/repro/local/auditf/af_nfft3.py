"""Which convention does the adjoint NFFT implement? Compare against 4 exact DFT variants,
then re-measure float32 vs double accuracy at large grids using the matching one."""
import numpy as np, json, warnings
warnings.simplefilter('ignore')
from cuvarbase.cunfft import NFFTAsyncProcess
rng = np.random.RandomState(4)
def variants(t, y, ks, tmin, tmax):
    T = tmax - tmin
    out = {}
    for sgn in (+1, -1):
        for ref, tt in (('abs', t), ('tmin', t - tmin)):
            out['sign%+d_%s' % (sgn, ref)] = np.array([np.sum(y*np.exp(sgn*2j*np.pi*k*tt/T)) for k in ks])
    return out
res = {}
n0, nf, T = 2000, 4000, 100.
t = np.sort(rng.rand(n0)*T); y = rng.randn(n0)
ks = np.arange(0, nf//2, 97)
proc = NFFTAsyncProcess(sigma=2, m=8)
g = np.asarray(proc.run([(t, y, nf)])[0]); proc.finish(); g = g.copy()
V = variants(t, y, ks, t.min(), t.max())
for k, v in V.items():
    print("convention %-14s: max|dG|/||y||_1 = %.2e" % (k, np.max(np.abs(g[ks]-v))/np.sum(np.abs(y))))
best = min(V, key=lambda k: np.max(np.abs(g[ks]-V[k])))
print("best-matching convention:", best)
def exact(t, y, ks, tmin, tmax):
    return variants(t, y, ks, tmin, tmax)[best]
for n0, nf, T in [(2000, 4000, 100.), (6000, 200000, 3650.), (6000, 894250, 3650.), (6000, 1788500, 3650.)]:
    t = np.sort(rng.rand(n0)*T); y = rng.randn(n0)
    ks = np.unique(np.concatenate([np.arange(0, nf//2, max(1, nf//13)), [nf//4, nf//2-1]]))
    ex = exact(t, y, ks, t.min(), t.max())
    row = {}
    for dbl in (False, True):
        for m in (8, 12):
            p = NFFTAsyncProcess(use_double=dbl, sigma=2, m=m)
            gg = np.asarray(p.run([(t, y, nf)])[0]).copy(); p.finish()
            err = np.abs(gg[ks]-ex)/np.sum(np.abs(y))
            row['dbl%d_m%d' % (dbl, m)] = float(err.max())
            print("n0=%d nf=%d ng=%d use_double=%-5s m=%d: max rel err=%.2e  median=%.2e  at k=nf/4: %.2e  k=nf/2-1: %.2e" % (n0, nf, 2*nf, dbl, m, err.max(), np.median(err), err[ks==nf//4][0], err[ks==nf//2-1][0]))
    res['n%d_nf%d' % (n0, nf)] = row
json.dump(res, open('/workspace/scratch/af_nfft3.json', 'w'), indent=1)
