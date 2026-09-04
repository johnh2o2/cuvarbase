"""LS audit: GPU GLS vs astropy at modest and survey-scale grids (float32 phase/trig
precision), use_double, and fap_baluev vs astropy's Baluev FAP."""
import numpy as np, time, json
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev

rng = np.random.RandomState(3)
res = {}

def compare(label, n, T, fsig, fmin, fmax, spp=5, use_double=False, amp=0.5, sig=0.3, stride=None):
    t = np.sort(rng.rand(n) * T)
    y = 10. + amp*np.sin(2*np.pi*fsig*t + 0.3) + sig*rng.randn(n)
    dy = sig*(0.5 + rng.rand(n))
    df = 1./(spp*T); k0 = max(1, int(round(fmin/df))); nf = int(fmax/df) - k0
    freqs = df*(k0 + np.arange(nf))
    proc = LombScargleAsyncProcess(use_double=use_double)
    r = proc.run([(t, y, dy)], freqs=freqs); proc.finish()
    p = np.array(r[0][1], dtype=np.float64)
    idx = np.arange(nf) if stride is None else np.unique(np.concatenate([np.arange(0, nf, stride), [np.argmax(p)], np.argmin(np.abs(freqs-fsig))+np.arange(-3,4)]))
    idx = idx[(idx >= 0) & (idx < nf)]
    ls = LombScargle(t, y, dy, fit_mean=True, center_data=True)
    pa = ls.power(freqs[idx], method='cython', normalization='standard')
    d = np.abs(p[idx] - pa)
    ipk = np.argmin(np.abs(freqs[idx] - fsig))
    ipk = idx[np.argmax(pa)] if False else np.argmax(pa)
    print("%-38s n=%d T=%g nf=%d k0=%d: max|d|=%.2e  p95|d|=%.2e  peak(astropy)=%.4f gpu=%.4f  rel=%.2e ; argmax match=%s ; gpu min=%.3f"
          % (label, n, T, nf, k0, d.max(), np.percentile(d, 95), pa[ipk], p[idx][ipk], abs(p[idx][ipk]-pa[ipk])/pa[ipk],
             np.argmax(p) == idx[np.argmax(p[idx])] and abs(freqs[np.argmax(p)] - freqs[idx][ipk]) < 2*df, p.min()))
    # error vs frequency (binned in 5 bands)
    bands = np.array_split(np.arange(len(idx)), 5)
    print("      max|d| by frequency band:", ["%.1e@f<%.1f" % (d[b].max(), freqs[idx][b].max()) for b in bands])
    res[label] = dict(n=n, T=T, nf=nf, k0=k0, maxabs=float(d.max()), p95=float(np.percentile(d, 95)),
                      peak_astropy=float(pa[ipk]), peak_gpu=float(p[idx][ipk]),
                      bands=[[float(d[b].max()), float(freqs[idx][b].max())] for b in bands])
    return t, y, dy, freqs, p

compare('modest: 1000pts, 365d, f<20', 1000, 365., 3.3, 0.1, 20.)
compare('modest hi-f signal f=18', 1000, 365., 18.0, 0.1, 20.)
t, y, dy, freqs, p = compare('survey: 6000pts, 3650d, f<50, sig f=2', 6000, 3650., 2.0, 1.0, 50., stride=61)
compare('survey: 6000pts, 3650d, f<50, sig f=40', 6000, 3650., 40.0, 1.0, 50., stride=61)
compare('survey f<50 sig f=40 use_double', 6000, 3650., 40.0, 1.0, 50., use_double=True, stride=61)
compare('survey k0=1 (fmin=df): f<50 sig f=40', 6000, 3650., 40.0, 0.0, 50., stride=61)
compare('kepler-like: 65000pts, 1460d, f<30, sig f=25', 65000, 1460., 25.0, 0.5, 30., stride=97)

# ---- fap_baluev vs astropy
t = np.sort(rng.rand(300) * 100.); y = 10 + 0.3*rng.randn(300); dy = 0.3*np.ones(300)
fmax = 20.; fmin = 0.05
ls = LombScargle(t, y, dy, fit_mean=True, center_data=True)
zs = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.3])
fa = ls.false_alarm_probability(zs, method='baluev', minimum_frequency=fmin, maximum_frequency=fmax)
fc = fap_baluev(t, dy, zs, fmax)
print("Baluev FAP  z:", zs)
print("  astropy  :", np.array2string(fa, precision=3))
print("  cuvarbase:", np.array2string(fc, precision=3))
print("  ratio cuv/astropy:", np.array2string(fc / fa, precision=3))
# single-frequency FAP (astropy 'single') and tau ratio check
fs = ls.false_alarm_probability(zs, method='single')
res['fap'] = dict(z=zs.tolist(), astropy=fa.tolist(), cuvarbase=fc.tolist(), ratio=(fc/fa).tolist())
# with N=1000
t = np.sort(rng.rand(1000) * 100.); y = 10 + 0.3*rng.randn(1000); dy = 0.3*np.ones(1000)
ls = LombScargle(t, y, dy, fit_mean=True, center_data=True)
zs = np.array([0.01, 0.02, 0.03, 0.05])
fa = ls.false_alarm_probability(zs, method='baluev', minimum_frequency=fmin, maximum_frequency=fmax)
fc = fap_baluev(t, dy, zs, fmax)
print("N=1000 ratio cuv/astropy:", np.array2string(fc / fa, precision=3), " astropy:", np.array2string(fa, precision=2))
res['fap_n1000'] = dict(z=zs.tolist(), astropy=fa.tolist(), cuvarbase=fc.tolist())
json.dump(res, open('/workspace/scratch/af_ls.json', 'w'), indent=1)
