import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
import patches
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def run(proc, t, y, dy, freqs, **kw):
    r = proc.run([(t, y, dy)], freqs=freqs, **kw); proc.finish()
    return np.array(r[0][1][:len(freqs)], float)

cases = {}
t, y, dy = make_lc(N=300, T=365.0, f0=3.1, seed=1); fr = grid(1.0 / (5 * 365.0), 20.0, 365.0)
cases['k0=1 N=300 T=365'] = (t, y, dy, fr, LombScargle(t, y, dy).power(fr, method='cython'))
t, y, dy = make_lc(N=1000, T=1000.0, f0=7.3, hetero=True, seed=2); fr = grid(1.0 / (5 * 1000.0), 20.0, 1000.0)
cases['hetero N=1000 T=1000'] = (t, y, dy, fr, LombScargle(t, y, dy).power(fr, method='cython'))
t, y, dy = make_lc(N=300, T=365.0, f0=9.5, seed=3)
for (a, b) in [(5.0, 10.0), (20.0, 30.0), (40.0, 50.0)]:
    fr = grid(a, b, 365.0)
    cases['band %.0f-%.0f k0/nf=%.0f' % (a, b, get_k0(fr) / len(fr))] = (t, y, dy, fr, LombScargle(t, y, dy).power(fr, method='cython'))
t, y, dy = make_lc(N=300, T=3650.0, f0=23.456, seed=5); fr = grid(1.0 / (5 * 3650.0), 50.0, 3650.0)
cases['T=3650 fmax=50 nf=912K'] = (t, y, dy, fr, gls_numpy_fast(t, y, dy, fr))

for fixes in [dict(F=False, P=False, B=False), dict(F=True, P=True, B=False), dict(F=True, P=True, B=True)]:
    patches.apply(**fixes)
    print("=================== fixes: %s ===================" % fixes)
    for (dbl, sigma, m) in [(False, 4, 8), (True, 4, 8), (False, 2, 8), (True, 2, 8)]:
        proc = LombScargleAsyncProcess(use_double=dbl, sigma=sigma, m=m, autoset_m=False)
        for name, (t, y, dy, fr, ref) in cases.items():
            try:
                p = run(proc, t, y, dy, fr)
                report("%-24s dbl=%d sigma=%d m=%d" % (name, dbl, sigma, m), ref, p)
            except Exception as e:
                print("%-24s dbl=%d sigma=%d m=%d RAISED %r" % (name, dbl, sigma, m, e))
        del proc

print("=================== all fixes, double: m sweep at sigma=4 and sigma=2 (k0=1 case) ===================")
patches.apply(F=True, P=True, B=True)
t, y, dy, fr, ref = cases['k0=1 N=300 T=365']
for sigma in (2, 3, 4):
    for m in (2, 4, 6, 8, 10, 12):
        proc = LombScargleAsyncProcess(use_double=True, sigma=sigma, m=m, autoset_m=False)
        p = run(proc, t, y, dy, fr)
        report("FIXED dbl sigma=%d m=%2d" % (sigma, m), ref, p)
        del proc
print("=================== all fixes, float32: m sweep ===================")
for sigma in (2, 4):
    for m in (2, 4, 6, 8, 12):
        proc = LombScargleAsyncProcess(use_double=False, sigma=sigma, m=m, autoset_m=False)
        p = run(proc, t, y, dy, fr)
        report("FIXED f32 sigma=%d m=%2d" % (sigma, m), ref, p)
        del proc
