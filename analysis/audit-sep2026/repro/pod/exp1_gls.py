import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def run_gpu(proc, t, y, dy, freqs, **kw):
    r = proc.run([(t, y, dy)], freqs=freqs, **kw)
    proc.finish()
    return np.array(r[0][1][:len(freqs)], dtype=float)

procs = {}
def P(use_double, sigma, m):
    key = (use_double, sigma, m)
    if key not in procs:
        procs[key] = LombScargleAsyncProcess(use_double=use_double, sigma=sigma, m=m, autoset_m=False)
    return procs[key]

print("=== A. baseline N=300 T=365 k0=1 grid, fmax=20 c/d; vs astropy(fit_mean, standard, cython) ===")
t, y, dy = make_lc(N=300, T=365.0, f0=3.1, seed=1)
freqs = grid(1.0 / (5 * 365.0), 20.0, 365.0)
print("nf=%d k0=%d" % (len(freqs), get_k0(freqs)))
ref = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freqs, method='cython', normalization='standard')
ref2 = gls_numpy_fast(t, y, dy, freqs)
report("astropy vs numpy-ZK09", ref, ref2)
for use_double in (False, True):
    for sigma in (2, 4):
        for m in (4, 8, 12):
            g = run_gpu(P(use_double, sigma, m), t, y, dy, freqs)
            report("fft dbl=%s sigma=%d m=%d" % (use_double, sigma, m), ref, g, freqs)

print("=== B. same, BJD-scale t0=2455000.5 ===")
tb = t + 2455000.5
refb = LombScargle(tb, y, dy).power(freqs, method='cython')
for use_double in (False, True):
    for sigma in (2, 4):
        g = run_gpu(P(use_double, sigma, 8), tb, y, dy, freqs)
        report("fft BJD dbl=%s sigma=%d m=8" % (use_double, sigma), refb, g, freqs)

print("=== C. heteroscedastic dy, N=1000, T=1000 ===")
t, y, dy = make_lc(N=1000, T=1000.0, f0=7.3, hetero=True, seed=2)
freqs = grid(1.0 / (5 * 1000.0), 20.0, 1000.0)
print("nf=%d k0=%d" % (len(freqs), get_k0(freqs)))
ref = LombScargle(t, y, dy).power(freqs, method='cython')
for use_double in (False, True):
    for sigma in (2, 4):
        g = run_gpu(P(use_double, sigma, 8), t, y, dy, freqs)
        report("fft hetero dbl=%s sigma=%d m=8" % (use_double, sigma), ref, g, freqs)

print("=== D. narrow high band: k0 comparable to nf (upper half-band) ===")
t, y, dy = make_lc(N=300, T=365.0, f0=7.3, seed=3)
for (fmin, fmax) in [(0.5, 20.0), (5.0, 20.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (20.0, 50.0), (40, 50)]:
    freqs = grid(fmin, fmax, 365.0)
    k0 = get_k0(freqs); nf = len(freqs)
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    for use_double in (False, True):
        for sigma in (2, 4):
            g = run_gpu(P(use_double, sigma, 8), t, y, dy, freqs)
            report("band %.1f-%.1f k0=%d nf=%d k0/nf=%.2f dbl=%s sigma=%d" % (fmin, fmax, k0, nf, k0 / nf, use_double, sigma), ref, g, freqs)
            if use_double and sigma == 4:
                # error profile across the band in quartiles
                d = np.abs(ref - g); q = len(d) // 4
                print("   quartile max err: %s" % ["%.1e" % d[i * q:(i + 1) * q].max() for i in range(4)])

print("=== D2. signal at the top of a narrow band: does the peak survive? ===")
t, y, dy = make_lc(N=300, T=365.0, f0=9.5, seed=3)
freqs = grid(5.0, 10.0, 365.0)
ref = LombScargle(t, y, dy).power(freqs, method='cython')
for use_double in (False, True):
    for sigma in (2, 4):
        g = run_gpu(P(use_double, sigma, 8), t, y, dy, freqs)
        report("f0=9.5 in band 5-10 dbl=%s sigma=%d" % (use_double, sigma), ref, g, freqs)

print("=== E. direct sums (use_fft=False) float32/64, T=3650, fmax=20 (large phase args) ===")
t, y, dy = make_lc(N=300, T=3650.0, f0=7.3, seed=4)
freqs = grid(19.0, 20.0, 3650.0)
print("nf=%d k0=%d" % (len(freqs), get_k0(freqs)))
ref = LombScargle(t, y, dy).power(freqs, method='cython')
for use_double in (False, True):
    g = run_gpu(P(use_double, 4, 8), t, y, dy, freqs, use_fft=False)
    report("dirsum dbl=%s T=3650 f~20" % use_double, ref, g, freqs)
    g = run_gpu(P(use_double, 4, 8), t, y, dy, freqs)
    report("fft    dbl=%s T=3650 f~20 (k0/nf=%.1f)" % (use_double, get_k0(freqs) / len(freqs)), ref, g, freqs)
g = run_gpu(P(True, 4, 8), t, y, dy, freqs, use_fft=False, python_dir_sums=True)
report("python_dir_sums", ref, g, freqs)

print("=== E2. direct sums, short baseline (T=30), fmax 20 ===")
t, y, dy = make_lc(N=300, T=30.0, f0=7.3, seed=4)
freqs = grid(1.0 / 150, 20.0, 30.0)
ref = LombScargle(t, y, dy).power(freqs, method='cython')
for use_double in (False, True):
    g = run_gpu(P(use_double, 4, 8), t, y, dy, freqs, use_fft=False)
    report("dirsum dbl=%s T=30" % use_double, ref, g, freqs)

print("=== F. float32 large k*t: T=3650 d, fmax=50 c/d, k0=1 grid (nf=%d) ===" % int(50 * 5 * 3650))
t, y, dy = make_lc(N=500, T=3650.0, f0=23.456, seed=5)
freqs = grid(1.0 / (5 * 3650.0), 50.0, 3650.0)
print("nf=%d k0=%d" % (len(freqs), get_k0(freqs)))
ref = gls_numpy_fast(t, y, dy, freqs)
for use_double in (False, True):
    for sigma in (2, 4):
        g = run_gpu(P(use_double, sigma, 8), t, y, dy, freqs)
        report("fft T=3650 fmax=50 dbl=%s sigma=%d" % (use_double, sigma), ref, g, freqs)
        # error vs frequency: split into 5 bins
        d = np.abs(ref - g); q = len(d) // 5
        print("   quintile max err: %s" % ["%.1e" % d[i * q:(i + 1) * q].max() for i in range(5)])

print("=== G. floating_mean=False vs astropy fit_mean=False (hetero dy) ===")
t, y, dy = make_lc(N=300, T=365.0, f0=3.1, hetero=True, seed=6)
freqs = grid(1.0 / (5 * 365.0), 20.0, 365.0)
refA = LombScargle(t, y, dy, fit_mean=False, center_data=True).power(freqs, method='cython')
g = run_gpu(P(True, 4, 8), t, y, dy, freqs, floating_mean=False)
report("floating_mean=False vs astropy fit_mean=False center_data=True", refA, g, freqs)
# alt reference: classic LS on data centered by the UNWEIGHTED mean
yc = y - np.mean(y)
w = dy ** -2; w /= w.sum()
refB = LombScargle(t, yc, dy, fit_mean=False, center_data=False).power(freqs, method='cython')
report("floating_mean=False vs astropy fit_mean=False center_data=False on y-mean(y)", refB, g, freqs)
# note astropy 'standard' with fit_mean=False, center_data=False uses YY = sum w y^2
print("=== H. window=True ===")
gw = run_gpu(P(True, 4, 8), t, y, dy, freqs, window=True)
ones = np.ones_like(y)
refW = LombScargle(t, ones, dy, fit_mean=False, center_data=False).power(freqs, method='cython')
report("window vs astropy LS of ones (fit_mean=False,center=False)", refW, gw, freqs)
report("window vs 4x that", 4 * refW, gw, freqs)
print("ratio window/ref at top-5 peaks:", (gw / refW)[np.argsort(-refW)[:5]])
