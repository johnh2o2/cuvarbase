import sys, numpy as np
sys.path.insert(0, '/workspace/scratch')
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
T = 365.0; df = 1.0 / (5 * T)
rng = np.random.RandomState(3)
t = np.sort(rng.rand(300)) * T; dy = 0.1 * np.ones(300)
proc = LombScargleAsyncProcess()   # all defaults: float32, sigma=4, m=8
print("defaults: sigma=%s m=%s use_double=%s" % (proc.nfft_proc.sigma, proc.nfft_proc.m, proc.use_double))
nf = 9125
for r in (0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
    k0 = int(round(r * nf)); fr = df * (k0 + np.arange(nf))
    f0 = fr[int(0.5 * nf)]                      # signal mid-band
    y = 12.0 + 0.3 * np.cos(2 * np.pi * f0 * t - 0.3) + dy * rng.randn(300)
    ref = LombScargle(t, y, dy).power(fr, method='cython')
    res = proc.run([(t, y, dy)], freqs=fr); proc.finish()
    p = np.asarray(res[0][1][:nf], float)
    d = np.abs(ref - p); nbad = int((d > 0.05).sum())
    print("k0/nf=%.2f band %.2f-%.2f topfrac=%.3f maxabs=%.2e n(|err|>0.05)=%d (%.1f%% of band) argmax ref %.4f gpu %.4f %s maxpow=%.3g" % (
        r, fr[0], fr[-1], (k0 + nf) / (4.0 * nf), d.max(), nbad, 100.0 * nbad / nf, fr[np.argmax(ref)], fr[np.argmax(p)],
        "OK" if np.argmax(ref) == np.argmax(p) else "BAD", p.max()), flush=True)
