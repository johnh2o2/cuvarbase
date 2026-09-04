"""Detector A: PSD estimated from raw data vs from the cotrended residual vs true."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram, _sequential_detrend
def box(t, P, e, d):
    ph = np.fmod(t - e, P) / P; ph[ph < 0] += 1.0; ph[ph > 0.5] -= 1.0
    tm = np.zeros_like(t); tm[np.abs(ph) <= d / (2.0 * P)] = -1.0; return tm
proc = NUFFTLRTAsyncProcess(); rng = np.random.RandomState(5)
nights = 60; t = np.concatenate([n + 0.05 + 0.3 * np.sort(rng.rand(40)) for n in range(nights)]); N = len(t); Tb = t.max() - t.min(); sigma = 3e-3
m1 = (t - t.mean()) / (0.5 * Tb); tn = t - np.floor(t) - 0.2; m2 = (tn / 0.15) ** 2 - 0.5; m3 = np.sin(2 * np.pi * t / (0.4 * Tb))
M = np.stack([m1, m2, m3], axis=1); M /= M.std(axis=0); amps = np.array([6.0, 3.0, 6.0]) * sigma
pop = np.empty((60, N))
for i in range(60): pop[i] = sigma * rng.randn(N) + M @ (rng.randn(3) * amps)
pop -= pop.mean(axis=1, keepdims=True); _, _, VT = np.linalg.svd(pop, full_matrices=False); V = VT[:3].T
coeffs = pop @ V; mu_c = coeffs.mean(axis=0); cov_c = np.cov(coeffs.T)
P_true, dur_true = 5.3, 0.22; periods = np.exp(np.linspace(np.log(2.0), np.log(18.0), 40)); periods[np.argmin(np.abs(periods - P_true))] = P_true
ef = np.arange(8) / 8.0; nf = 2 * N; psd_true = np.full(nf, N * sigma ** 2)
def psd_from(yres):
    Y = proc.compute_nufft(t, (yres - yres.mean()).astype(np.float32), nf)
    p = _smoothed_periodogram(np.abs(Y) ** 2, 5); return np.maximum(p, 1e-12 * np.median(p[p > 0]))
def search(y, **kw):
    best = -np.inf; bestP = np.nan
    for p in periods:
        s = proc.run(t, y, np.array([p]), durations=np.array([dur_true]), epochs=ef * p, **kw)
        if s.max() > best: best = s.max(); bestP = p
    return best, bestP
base = dict(detector='marginal', systematics_basis=V, coeff_prior_mean=mu_c, coeff_prior_cov=cov_c)
print("marginal detector statistic at the injected signal (depth 5 sigma) and on null, with three PSD choices:")
for kind in ('null', 'inj'):
    rows = []
    for i in range(4):
        y = sigma * rng.randn(N) + M @ (rng.randn(3) * amps)
        if kind == 'inj': y = y + 5 * sigma * box(t, P_true, rng.rand() * P_true, dur_true)
        resid = _sequential_detrend(t, y, V)
        r_raw = search(y, **base)
        r_res = search(y, estimate_psd=False, psd=psd_from(resid), **base)
        r_tru = search(y, estimate_psd=False, psd=psd_true, **base)
        rows.append((r_raw[0], r_res[0], r_tru[0], r_raw[1], r_res[1], r_tru[1]))
    rows = np.array(rows)
    print("  %s: max-stat  psd-from-raw-y: %s | psd-from-cotrended-residual: %s | true white psd: %s" % (kind, np.round(rows[:, 0], 2), np.round(rows[:, 1], 2), np.round(rows[:, 2], 2)))
    if kind == 'inj': print("       best periods: raw %s | resid %s | true %s (P_true=%.1f)" % (np.round(rows[:, 3], 2), np.round(rows[:, 4], 2), np.round(rows[:, 5], 2), P_true))
