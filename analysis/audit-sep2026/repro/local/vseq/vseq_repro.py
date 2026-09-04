"""Verifier: detector='sequential' OLS-without-intercept. Analytic bias check + realistic CBV case + device scan."""
import numpy as np, warnings
warnings.simplefilter('ignore')
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _sequential_detrend
rng = np.random.RandomState(7)
def box(t, P, dur, depth, t0):
    ph = ((t - t0) / P) % 1.0; ph[ph > 0.5] -= 1
    return -depth * (np.abs(ph) < dur / (2*P))
n = 3000; T = 90.; t = np.sort(rng.rand(n) * T)
P, dur, depth, t0, sig = 3.3, 0.15, 0.006, 1.1, 0.003

# 1) Analytic: one column v with mean m_v, std s_v; y = ybar + c v + noise.
#    OLS no intercept: c_hat = c + ybar*m_v/(m_v^2+s_v^2); leftover (after demean) = -(c_hat-c)*(v-m_v)
v0 = np.sin(2*np.pi*t/30.0); v0 = (v0 - v0.mean())/v0.std()   # exactly zero-mean, unit std
print("== analytic bias check (CPU, _sequential_detrend as implemented) ==")
for ybar in (1.0, 1e5):
    for m_v in (0.0, 1e-3, 1e-2, 0.5):
        v = v0 + m_v
        y = ybar + 0.004*v + sig*rng.randn(n)*0  # noiseless
        r = _sequential_detrend(t, y, v[:, None])
        r -= r.mean()
        pred = ybar*m_v/(m_v**2 + 1.0)   # coefficient bias
        print("  ybar=%-7g m_v=%-6g  leftover systematic amp (std of demeaned resid)=%.4g  predicted |bias|*s_v=%.4g" % (ybar, m_v, r.std(), abs(pred)))

# 2) Realistic: relative flux (mean 1) + transit + two smooth CBV-like columns with small nonzero means (1%)
print("== device scan: relative flux, 2-column basis, column means 0 vs 0.01 ==")
lrt = NUFFTLRTAsyncProcess(sigma=2)
epochs = np.arange(0, P, dur/4); periods = np.array([P, 2.9, 3.1, 3.5, 3.7, 4.1])
def ev(y, det='matched', **kw):
    s = lrt.run(t, y, periods, np.array([dur]), epochs=epochs, detector=det, **kw); m = s.max(axis=(1, 2)); return m[0], m[1:].max()
V0 = np.stack([np.sin(2*np.pi*t/30.), np.cos(2*np.pi*t/17.)], axis=1); V0 -= V0.mean(axis=0); V0 /= V0.std(axis=0)
c = np.array([0.004, -0.004])
noise = sig*rng.randn(n)
for mean_off in (0.0, 1e-3, 1e-2):
    V = V0 + mean_off
    y = 1.0 + box(t, P, dur, depth, t0) + V @ c + noise
    a, b = ev(y, 'sequential', systematics_basis=V)
    a2, b2 = ev(y, 'sequential', systematics_basis=V - V.mean(axis=0))
    a3, b3 = ev(y, 'sequential', systematics_basis=np.column_stack([V, np.ones(n)]))
    print("  col mean=%-5g  as-implemented SNR@P=%6.2f (off %5.2f) | demeaned V %6.2f (off %5.2f) | +const col %6.2f (off %5.2f)" % (mean_off, a, b, a2, b2, a3, b3))
# 3) Raw flux units (e-/s), exactly zero-mean basis: should be unaffected
y_raw = 5e4*(1.0 + box(t, P, dur, depth, t0) + V0 @ c + noise)
a, b = ev(y_raw, 'sequential', systematics_basis=V0)
print("  raw flux 5e4 e-/s, exactly zero-mean basis: as-implemented SNR@P=%.2f (off %.2f)" % (a, b))
y_raw = 5e4*(1.0 + box(t, P, dur, depth, t0) + (V0+1e-3) @ c + noise)
a, b = ev(y_raw, 'sequential', systematics_basis=V0+1e-3)
a2, b2 = ev(y_raw, 'sequential', systematics_basis=V0+1e-3-(V0+1e-3).mean(axis=0))
print("  raw flux 5e4 e-/s, col mean 1e-3: as-implemented SNR@P=%.2f (off %.2f) | demeaned %.2f (off %.2f)" % (a, b, a2, b2))
