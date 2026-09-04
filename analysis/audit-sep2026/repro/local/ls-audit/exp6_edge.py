import sys, time, traceback
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0, fap_baluev
from cuvarbase.utils import autofrequency

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

proc = LombScargleAsyncProcess(use_double=True, sigma=4, m=8, autoset_m=False)
def run(t, y, dy, freqs=None, p=proc, **kw):
    r = p.run([(t, y, dy)], freqs=freqs, **kw); p.finish()
    f, pw = r[0]
    return np.asarray(f), np.array(pw[:len(f)], float)

def attempt(name, fn):
    try:
        out = fn()
        print("%-55s -> %s" % (name, out))
    except Exception as e:
        print("%-55s -> RAISED %s: %s" % (name, type(e).__name__, str(e)[:150]))

t, y, dy = make_lc(N=300, T=365.0, f0=3.1, seed=1)
print("=== nf not a multiple of block size ===")
for nf in (1, 2, 3, 255, 257, 1001):
    df = 1.0 / (5 * 365.0); freqs = df * (1 + np.arange(nf))
    def f():
        fr, p = run(t, y, dy, freqs, fast_grid=False)
        ref = LombScargle(t, y, dy).power(freqs, method='cython')
        return "maxabs=%.2e" % np.abs(ref - p).max()
    attempt("nf=%d" % nf, f)

print("=== non-uniform freqs (geomspace) silently accepted? ===")
freqs = np.geomspace(1.0, 20.0, 5000)
print("check_k0 passes on geomspace? df=%.4g k0=%d" % (freqs[1] - freqs[0], get_k0(freqs)))
def f():
    fr, p = run(t, y, dy, freqs, fast_grid=False)
    ref_user = LombScargle(t, y, dy).power(freqs, method='cython')
    uni = (freqs[1] - freqs[0]) * (get_k0(freqs) + np.arange(len(freqs)))
    ref_uni = LombScargle(t, y, dy).power(uni, method='cython')
    return "returned freqs==input: %s; maxabs vs astropy@returned freqs=%.2e ; vs astropy@uniform grid df*(k0+i)=%.2e (grid spans %.2f-%.2f)" % (
        np.allclose(fr, freqs), np.abs(ref_user - p).max(), np.abs(ref_uni - p).max(), uni[0], uni[-1])
attempt("geomspace freqs", f)

print("=== autofrequency conventions ===")
fa = autofrequency(t, samples_per_peak=5, nyquist_factor=5, minimum_frequency=0.1, maximum_frequency=20.0)
print("autofreq(minf=0.1,maxf=20): f[0]=%.5f f[-1]=%.5f nf=%d df=%.6f  (astropy: f[0]=minf, f[-1]~maxf)" % (fa[0], fa[-1], len(fa), fa[1] - fa[0]))
from astropy.timeseries import LombScargle as LS
fap_ = LS(t, y, dy).autofrequency(samples_per_peak=5, nyquist_factor=5, minimum_frequency=0.1, maximum_frequency=20.0)
print("astropy autofrequency: f[0]=%.5f f[-1]=%.5f nf=%d" % (fap_[0], fap_[-1], len(fap_)))
fa0 = autofrequency(t, samples_per_peak=5, nyquist_factor=5)
fap0 = LS(t, y, dy).autofrequency(samples_per_peak=5, nyquist_factor=5)
print("default: cuvarbase f[0]=%.5f f[-1]=%.5f nf=%d ; astropy f[0]=%.5f f[-1]=%.5f nf=%d" % (fa0[0], fa0[-1], len(fa0), fap0[0], fap0[-1], len(fap0)))
res = proc.batched_run_const_nfreq([(t, y, dy)], samples_per_peak=5, nyquist_factor=5)
print("batched_run_const_nfreq(freqs=None) grid: f[0]=%.5f f[-1]=%.5f nf=%d (run() would give nf=%d)" % (res[0][0][0], res[0][0][-1], len(res[0][0]), len(fa0)))

print("=== data edge cases ===")
freqs = grid(1.0 / (5 * 365.0), 5.0, 365.0)
def chk(tt, yy, dd, **kw):
    fr, p = run(tt, yy, dd, freqs, **kw)
    return "finite=%s min=%.3g max=%.3g n_neg1=%d" % (np.all(np.isfinite(p)), np.nanmin(p), np.nanmax(p), np.sum(p == -1))
attempt("dy with one zero", lambda: chk(t, y, np.where(np.arange(300) == 5, 0.0, dy)))
attempt("y with one NaN", lambda: chk(t, np.where(np.arange(300) == 5, np.nan, y), dy))
attempt("t with one NaN", lambda: chk(np.where(np.arange(300) == 5, np.nan, t), y, dy))
attempt("N=2", lambda: chk(t[:2], y[:2], dy[:2]))
attempt("N=3", lambda: chk(t[:3], y[:3], dy[:3]))
attempt("N=1", lambda: chk(t[:1], y[:1], dy[:1]))
attempt("all t identical (N=5)", lambda: chk(np.ones(5), y[:5], dy[:5]))
attempt("constant y", lambda: chk(t, np.ones_like(y), dy))
attempt("unsorted t", lambda: "maxabs vs sorted=%.2e" % np.abs(run(t[::-1], y[::-1], dy[::-1], freqs)[1] - run(t, y, dy, freqs)[1]).max())
def dup():
    t2 = t.copy(); t2[1::2] = t2[::2]  # duplicate every other time
    fr, p = run(t2, y, dy, freqs)
    ref = LombScargle(t2, y, dy).power(freqs, method='cython')
    return "maxabs vs astropy=%.2e" % np.abs(ref - p).max()
attempt("duplicate times (150 pairs)", dup)
attempt("dy=None (unweighted)", lambda: chk(t, y, None))
attempt("freqs list len != data len", lambda: proc.run([(t, y, dy), (t, y, dy)], freqs=[freqs]))
attempt("freqs[0] not k0*df (f0 = 1.37*df)", lambda: run(t, y, dy, (freqs[1]-freqs[0]) * (1.37 + np.arange(100))))

print("=== only_return_best_freqs significance saturation ===")
tb, yb, dyb = make_lc(N=1000, T=365.0, f0=3.1, amp=0.5, noise=0.05, seed=9)
bf, sig = proc.batched_run_const_nfreq([(tb, yb, dyb)], freqs=freqs, only_return_best_freqs=True)
fr, p = run(tb, yb, dyb, freqs)
zbest = p.max()
fap = fap_baluev(tb, dyb, zbest, freqs.max())
print("best f=%.4f z=%.4f fap_baluev=%.3e returned significance=%r  (1-significance=%r)" % (bf[0], zbest, fap, sig[0], 1 - sig[0]))

print("=== -1 sentinel handling in FAP path: fap_baluev on z=-1 ===")
with np.errstate(all='ignore'):
    print("fap_baluev(z=[-1, 0, 0.5, 1.0, 1.5]) =", fap_baluev(tb, dyb, np.array([-1.0, 0, 0.5, 1.0, 1.5]), 5.0))

print("=== floating-mean STANDARD mode on hetero data: what Y (weighted mean of centered y) is ===")
