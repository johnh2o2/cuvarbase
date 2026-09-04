"""Detector A and sequential detector checks."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch')
from lrt_common import *
from cuvarbase.nufft_lrt import (NUFFTLRTAsyncProcess, _marginal_statistic, _sequential_detrend, _whitened_inner)

rng = np.random.RandomState(3)
# --- sequential detrend: OLS with no intercept, basis not demeaned
t = make_times(rng); n = len(t)
V_raw = t[:, None].copy()                      # not zero-mean
V_dm = (t - t.mean())[:, None]
y = 1.0 + 0.004*(t - t.mean())/t.std() + 1e-3*rng.randn(n)
for name, V in (('raw t', V_raw), ('demeaned t', V_dm), ('[1, t]', np.stack([np.ones(n), t], 1))):
    r = _sequential_detrend(t, y, V)
    slope = np.polyfit(t - t.mean(), r, 1)[0] * t.std()
    print('sequential detrend basis=%-12s residual slope*std(t)=%.5f (trend amp 0.004)  resid std=%.5f' % (name, slope, r.std()))

# --- Detector A: singular prior covariance
nf, K = 40, 2
Y = rng.randn(nf) + 1j*rng.randn(nf); T = rng.randn(nf) + 1j*rng.randn(nf)
v1 = rng.randn(nf) + 1j*rng.randn(nf); v2 = rng.randn(nf) + 1j*rng.randn(nf)
psd = 0.5 + rng.rand(nf); w = np.ones(nf)
s_k1 = _marginal_statistic(Y, T, [v1], psd, w, np.array([[1.0]]))
s_sing = _marginal_statistic(Y, T, [v1, v2], psd, w, np.diag([1.0, 0.0]))
s_tiny = _marginal_statistic(Y, T, [v1, v2], psd, w, np.diag([1.0, 1e-12]))
s_flat = _marginal_statistic(Y, T, [v1, v2], psd, w, np.diag([1.0, 1e12]))
print('Detector A prior_cov=diag(1,0): got %.6f | correct limit (v2 fixed, K=1) %.6f | cov=diag(1,1e-12) %.6f | cov=diag(1,1e12) (flat on v2) %.6f' % (s_sing, s_k1, s_tiny, s_flat))
# --- basis scaling invariance
C = np.array([[2.0, 0.3], [0.3, 1.0]])
a = _marginal_statistic(Y, T, [v1, v2], psd, w, C)
S = np.diag([3.0, 0.2])
b = _marginal_statistic(Y, T, [3.0*v1, 0.2*v2], psd, w, np.linalg.inv(S) @ C @ np.linalg.inv(S))
c = _marginal_statistic(Y, T, [3.0*v1, 0.2*v2], psd, w, C)
print('scaling: base %.8f | V S with S^-1 C S^-1 %.8f (invariant) | V S with same C %.8f (differs, expected)' % (a, b, c))
# --- K=0 reduce
print('K=0 reduce: %.8f vs plain %.8f' % (_marginal_statistic(Y, T, [], psd, w, np.zeros((0,0))), _whitened_inner(Y,T,psd,w)/np.sqrt(_whitened_inner(T,T,psd,w))))

# --- device: marginal with user PSD containing zeros -> nan?
proc = NUFFTLRTAsyncProcess()
P, dur = 5.3, 0.22
yy = 1 + 3e-3*rng.randn(n) + box(t, P, 1.0, dur, 0.01)
nf = 2*n; psd0 = np.ones(nf); psd0[::7] = 0.0
sm = proc.run(t, yy, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]),
              estimate_psd=False, psd=psd0, nf=nf, detector='marginal',
              systematics_basis=V_dm, coeff_prior_cov=np.array([[1.0]]))
sp = proc.run(t, yy, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]),
              estimate_psd=False, psd=psd0, nf=nf)
print('user psd with zeros: matched=%s  marginal=%s' % (sp.ravel(), sm.ravel()))

# --- device: sequential vs marginal with non-demeaned basis on a transit + trend + offset
trend = np.sin(2*np.pi*t/40.0) + 0.7          # non-zero mean basis vector
yy = 1.0 + 0.01*trend + 3e-3*rng.randn(n) + box(t, P, 1.0, dur, 0.01)
periods = np.exp(np.linspace(np.log(2), np.log(18), 24)); periods[np.argmin(np.abs(periods-P))] = P
ip = int(np.argmin(np.abs(periods-P)))
def scan(**kw):
    best = []
    for p in periods:
        best.append(proc.run(t, yy, np.array([p]), durations=np.array([dur]), epochs=np.linspace(0, p, 32, endpoint=False), **kw).max())
    best = np.array(best); return periods[best.argmax()], best[ip], best.max()
print('matched   : bestP=%.3f snr@true=%.2f max=%.2f' % scan())
print('sequential raw basis: bestP=%.3f snr@true=%.2f max=%.2f' % scan(detector='sequential', systematics_basis=trend[:,None]))
print('sequential demeaned : bestP=%.3f snr@true=%.2f max=%.2f' % scan(detector='sequential', systematics_basis=(trend-trend.mean())[:,None]))
print('marginal  raw basis : bestP=%.3f snr@true=%.2f max=%.2f' % scan(detector='marginal', systematics_basis=trend[:,None], coeff_prior_cov=np.array([[1.0]])))

# --- PSD double counting: marginal detector's PSD estimated from y containing V c
y_sys = 1.0 + 3e-3*rng.randn(n) + ou_noise(rng, t, 3e-3, 0.8)
c_true = 0.03
yy = y_sys + c_true*(trend - trend.mean()) + box(t, P, 1.0, dur, 0.006)
noise_only = y_sys - y_sys.mean()
Ynoise = proc.compute_nufft(t, noise_only, nf)
psd_clean = np.abs(Ynoise)**2
from cuvarbase.nufft_lrt import _smoothed_periodogram
psd_clean = _smoothed_periodogram(psd_clean.astype(np.float32), 5)
Vb = (trend - trend.mean())[:, None]
for cov in (1e-4, 1e-2, 1.0):
    s_est = proc.run(t, yy, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]), nf=nf,
                     detector='marginal', systematics_basis=Vb, coeff_prior_cov=np.array([[cov]]))[0,0,0]
    s_cln = proc.run(t, yy, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]), nf=nf,
                     estimate_psd=False, psd=psd_clean,
                     detector='marginal', systematics_basis=Vb, coeff_prior_cov=np.array([[cov]]))[0,0,0]
    print('marginal prior_var=%g: SNR@true with PSD estimated from y (contains V c) = %.2f | with noise-only PSD = %.2f' % (cov, s_est, s_cln))
