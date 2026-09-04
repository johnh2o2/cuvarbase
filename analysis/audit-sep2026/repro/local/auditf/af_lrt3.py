"""NUFFT-LRT sequential detector: OLS cotrend has no intercept; flat-PSD comparison at the
correct noise level; self-whitening loss."""
import numpy as np, json, warnings
warnings.simplefilter('ignore')
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _sequential_detrend
rng = np.random.RandomState(21)
def box(t, P, dur, depth, t0):
    ph = ((t - t0) / P) % 1.0; ph[ph > 0.5] -= 1
    return -depth * (np.abs(ph) < dur / (2*P))
n = 3000; T = 90.; P = 3.3; dur = 0.15; depth = 0.006; sig = 0.003; t0 = 1.1
t = np.sort(rng.rand(n) * T); noise = sig * rng.randn(n)
lrt = NUFFTLRTAsyncProcess(sigma=2)
epochs = np.arange(0, P, dur/4); periods = np.array([P, 2.9, 3.1, 3.5, 3.7, 4.1]); nf = 2*n
def ev(y, det='matched', **kw):
    s = lrt.run(t, y, periods, np.array([dur]), epochs=epochs, detector=det, **kw); m = s.max(axis=(1, 2)); return m[0], m[1:].max()
y_clean = 1.0 + box(t, P, dur, depth, t0) + noise
V = np.stack([np.sin(2*np.pi*t/30.), np.cos(2*np.pi*t/17.)], axis=1); c = np.array([0.004, -0.004]); Cc = np.diag([0.004**2]*2)
y_sys = y_clean + V @ c
psd_flat = np.full(nf, n*sig**2)
res = {}
a, b = ev(y_clean); a2, b2 = ev(y_clean, estimate_psd=False, psd=psd_flat)
print("white noise: matched w/ estimated PSD SNR@P=%.2f (off %.2f) | with TRUE flat PSD n*sig^2: %.2f (off %.2f)" % (a, b, a2, b2)); res['clean'] = [a, b, a2, b2]
coef_nointercept = np.linalg.lstsq(V, y_sys, rcond=None)[0]
coef_intercept = np.linalg.lstsq(np.column_stack([V, np.ones(n)]), y_sys, rcond=None)[0][:2]
print("OLS coefficients: true=%s  no-intercept (as implemented)=%s  with intercept=%s" % (c, np.round(coef_nointercept, 4), np.round(coef_intercept, 4)))
res['coef'] = dict(true=c.tolist(), noint=coef_nointercept.tolist(), int=coef_intercept.tolist())
for label, det, kw in [('matched', 'matched', {}),
                       ('sequential (as implemented, no intercept)', 'sequential', dict(systematics_basis=V)),
                       ('sequential + constant column in basis', 'sequential', dict(systematics_basis=np.column_stack([V, np.ones(n)]))),
                       ('sequential, demeaned V (marginal-style)', 'sequential', dict(systematics_basis=V - V.mean(axis=0))),
                       ('marginal', 'marginal', dict(systematics_basis=V, coeff_prior_cov=Cc))]:
    a, b = ev(y_sys, det, **kw); a2, b2 = ev(y_sys, det, estimate_psd=False, psd=psd_flat, **kw)
    print("  %-44s est-PSD SNR@P=%6.2f (off %5.2f) | true flat PSD: %6.2f (off %5.2f)" % (label, a, b, a2, b2)); res[label] = [a, b, a2, b2]
json.dump(res, open('/workspace/scratch/af_lrt3.json', 'w'), indent=1)
