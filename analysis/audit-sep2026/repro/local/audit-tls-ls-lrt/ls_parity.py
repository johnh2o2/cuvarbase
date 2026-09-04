"""Lomb-Scargle parity vs astropy across configurations (GPU)."""
import warnings, time
warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
from cuvarbase.utils import autofrequency

rng = np.random.RandomState(42)

def make(N, T, freq=3.1, hetero=True, offset=0.0):
    t = np.sort(rng.rand(N) * T) + offset
    dy = 0.05 * (1 + (rng.rand(N) if hetero else 0))
    y = 12.0 + 0.3 * np.cos(2 * np.pi * freq * (t - offset) - 0.4) + dy * rng.randn(N)
    return t, y, dy

def gpu_freqs(t, spp=5, nyq=5):
    return autofrequency(t, samples_per_peak=spp, nyquist_factor=nyq)

def report(label, freqs, p_gpu, p_ref):
    p_gpu = np.asarray(p_gpu, dtype=np.float64); p_ref = np.asarray(p_ref, dtype=np.float64)
    top = np.argsort(p_ref)[::-1][:5]
    d = np.abs(p_gpu - p_ref)
    print("%-42s nf=%7d  max|d|=%.2e  max|d|@top5=%.2e  peak_gpu=%.6g peak_ref=%.6g  argmax match=%s  corr=%.6f"
          % (label, len(freqs), d.max(), d[top].max(), p_gpu.max(), p_ref.max(),
             np.argmax(p_gpu) == np.argmax(p_ref), np.corrcoef(p_gpu, p_ref)[0, 1]))

def run_gpu(t, y, dy, freqs, **kw):
    proc = LombScargleAsyncProcess(**{k: v for k, v in kw.items() if k in ('sigma', 'm', 'use_double', 'nharmonics', 'use_cufinufft', 'autoset_m', 'tol')})
    runkw = {k: v for k, v in kw.items() if k in ('use_fft', 'floating_mean')}
    res = proc.run([(t, y, dy)], freqs=[freqs], **runkw)
    proc.finish()
    return np.array(res[0][1])

# 1. baseline, heteroscedastic, N=300
t, y, dy = make(300, 100.0)
freqs = gpu_freqs(t)
ref = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freqs, method='cython')
report("N=300 default (sigma=4,m=8,f32)", freqs, run_gpu(t, y, dy, freqs), ref)
report("N=300 use_double", freqs, run_gpu(t, y, dy, freqs, use_double=True), ref)
report("N=300 sigma=2 m=8", freqs, run_gpu(t, y, dy, freqs, sigma=2), ref)
report("N=300 sigma=2 m=6", freqs, run_gpu(t, y, dy, freqs, sigma=2, m=6), ref)
report("N=300 sigma=2 m=4", freqs, run_gpu(t, y, dy, freqs, sigma=2, m=4), ref)
report("N=300 sigma=4 m=6", freqs, run_gpu(t, y, dy, freqs, sigma=4, m=6), ref)
report("N=300 direct sums (use_fft=False)", freqs, run_gpu(t, y, dy, freqs, use_fft=False), ref)
report("N=300 cufinufft", freqs, run_gpu(t, y, dy, freqs, use_cufinufft=True), ref)
# standard (no floating mean) vs astropy fit_mean=False, both centerings
ref_nc = LombScargle(t, y, dy, fit_mean=False, center_data=False).power(freqs, method='cython')
ref_c = LombScargle(t, y, dy, fit_mean=False, center_data=True).power(freqs, method='cython')
p_std = run_gpu(t, y, dy, freqs, floating_mean=False)
report("N=300 floating_mean=False vs fit_mean=F,center=F", freqs, p_std, ref_nc)
report("N=300 floating_mean=False vs fit_mean=F,center=T", freqs, p_std, ref_c)
# 2. BJD-scale times
t2, y2, dy2 = make(300, 100.0, offset=2457000.0)
freqs2 = gpu_freqs(t2)
ref2 = LombScargle(t2, y2, dy2).power(freqs2, method='cython')
report("N=300 BJD times (t+2457000) f32", freqs2, run_gpu(t2, y2, dy2, freqs2), ref2)
report("N=300 BJD times direct sums f32", freqs2, run_gpu(t2, y2, dy2, freqs2, use_fft=False), ref2)
# 3. multiharmonic
ref_mh = LombScargle(t, y, dy, nterms=2).power(freqs, method='chi2')
report("N=300 nharmonics=2 vs astropy nterms=2", freqs, run_gpu(t, y, dy, freqs, nharmonics=2), ref_mh)
# 4. larger: N=3000, long baseline, high nf
t3, y3, dy3 = make(3000, 1000.0, freq=7.3)
freqs3 = gpu_freqs(t3, spp=5, nyq=2)
t0 = time.time(); ref3 = LombScargle(t3, y3, dy3).power(freqs3, method='fast', assume_regular_frequency=True); print("astropy fast took %.1fs" % (time.time() - t0))
report("N=3000 nf~ (default) vs astropy fast", freqs3, run_gpu(t3, y3, dy3, freqs3), ref3)
report("N=3000 sigma=2 m=8", freqs3, run_gpu(t3, y3, dy3, freqs3, sigma=2), ref3)
report("N=3000 sigma=2 m=6", freqs3, run_gpu(t3, y3, dy3, freqs3, sigma=2, m=6), ref3)
report("N=3000 use_double sigma=4", freqs3, run_gpu(t3, y3, dy3, freqs3, use_double=True), ref3)
# exact reference at the top region only (cython on a subset is not possible with regular grid; use slow at 2000 freqs around peak)
ipk = np.argmax(ref3); sl = slice(max(0, ipk - 1000), ipk + 1000)
ref3x = LombScargle(t3, y3, dy3).power(freqs3[sl], method='cython')
for lab, kw in [("default", {}), ("sigma=2 m=8", dict(sigma=2)), ("sigma=2 m=6", dict(sigma=2, m=6)), ("use_double", dict(use_double=True))]:
    p = run_gpu(t3, y3, dy3, freqs3, **kw)[sl]
    print("   N=3000 %-12s vs EXACT (2000 freqs around peak): max|d|=%.2e  rel@peak=%.2e" % (lab, np.abs(p - ref3x).max(), abs(p[np.argmax(ref3x)] - ref3x.max()) / ref3x.max()))
# 5. batched_run_const_nfreq + only_return_best_freqs FAP vs full-array FAP
proc = LombScargleAsyncProcess()
bf, sig = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs, only_return_best_freqs=True)
res = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)
p = res[0][1]; bi = int(np.argmax(p))
fap_full = fap_baluev(t, dy, p, freqs.max())[bi]; fap_one = fap_baluev(t, dy, p[bi], freqs.max())
print("only_return_best_freqs: best f=%.6f (argmax f=%.6f) sig=%.6g ; fap(full arr)[best]=%.3e fap(scalar)=%.3e ; astropy baluev=%.3e"
      % (bf[0], freqs[bi], sig[0], fap_full, fap_one, LombScargle(t, y, dy).false_alarm_probability(ref.max(), method='baluev', minimum_frequency=freqs.min(), maximum_frequency=freqs.max())))
