"""NUFFT-LRT Detector A / sequential soundness experiments (GPU)."""
import warnings, time, sys
warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase.nufft_lrt import (NUFFTLRTAsyncProcess, _marginal_statistic,
                                 _whitened_inner, _sequential_detrend)

def box(t, P, e, d):
    ph = np.fmod(t - e, P) / P
    ph[ph < 0] += 1.0
    ph[ph > 0.5] -= 1.0
    tm = np.zeros_like(t)
    tm[np.abs(ph) <= d / (2.0 * P)] = -1.0
    return tm

def time_domain_detA(y, tau, V, Cc, sigma):
    """Exact Taaki Detector A in the time domain, white noise sigma,
    with y, tau, V columns all demeaned (matches the freq path's k=0 kill)."""
    n = len(y)
    Vd = V - V.mean(axis=0)
    yd = y - y.mean(); td = tau - tau.mean()
    Cz = sigma**2 * np.eye(n) + Vd @ Cc @ Vd.T
    Wz = np.linalg.inv(Cz)
    return float(yd @ Wz @ td / np.sqrt(td @ Wz @ td))

def time_domain_matched(y, tau, sigma):
    yd = y - y.mean(); td = tau - tau.mean()
    return float(yd @ td / sigma**2 / np.sqrt(td @ td / sigma**2))

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(0)

print("=== (b) null calibration + injection: matched / marginal(est psd) / marginal(true psd) / sequential ===")
# irregular ground-based-like sampling: 60 nights x 40 pts in 0.3 d windows
nights = 60
t = np.concatenate([n + 0.05 + 0.3 * np.sort(rng.rand(40)) for n in range(nights)])
N = len(t); Tb = t.max() - t.min()
sigma = 3e-3
m1 = (t - t.mean()) / (0.5 * Tb)
tn = t - np.floor(t) - 0.2
m2 = (tn / 0.15) ** 2 - 0.5
m3 = np.sin(2 * np.pi * t / (0.4 * Tb))
M = np.stack([m1, m2, m3], axis=1); M /= M.std(axis=0)
amps = np.array([6.0, 3.0, 6.0]) * sigma
# population-estimated basis + prior (as in the harness)
pop = np.empty((60, N))
for i in range(60):
    pop[i] = sigma * rng.randn(N) + M @ (rng.randn(3) * amps)
pop -= pop.mean(axis=1, keepdims=True)
_, _, VT = np.linalg.svd(pop, full_matrices=False)
V = VT[:3].T
coeffs = pop @ V
mu_c = coeffs.mean(axis=0); cov_c = np.cov(coeffs.T)
P_true, dur_true = 5.3, 0.22
periods = np.exp(np.linspace(np.log(2.0), np.log(18.0), 40))
periods[np.argmin(np.abs(periods - P_true))] = P_true
epochs_frac = np.arange(8) / 8.0
psd_true = np.full(2 * N, N * sigma ** 2)

def search(y, **kw):
    # epoch grid scaled per period like the harness (epochs in [0,P))
    best = -np.inf; bestP = np.nan
    for p in periods:
        s = proc.run(t, y, np.array([p]), durations=np.array([dur_true]),
                     epochs=epochs_frac * p, **kw)
        if s.max() > best:
            best = s.max(); bestP = p
    return best, bestP

methods = {
    'matched': dict(),
    'marg_est': dict(detector='marginal', systematics_basis=V, coeff_prior_mean=mu_c, coeff_prior_cov=cov_c),
    'marg_true': dict(detector='marginal', systematics_basis=V, coeff_prior_mean=mu_c, coeff_prior_cov=cov_c,
                      estimate_psd=False, psd=psd_true),
    'seq': dict(detector='sequential', systematics_basis=V),
    'seq_true': dict(detector='sequential', systematics_basis=V, estimate_psd=False, psd=psd_true),
}
n_null, n_inj = 8, 8
t0w = time.time()
nulls = {k: [] for k in methods}
for i in range(n_null):
    y = sigma * rng.randn(N) + M @ (rng.randn(3) * amps)
    for k, kw in methods.items():
        nulls[k].append(search(y, **kw)[0])
print("null done in %.1fs" % (time.time() - t0w))
thr = {k: np.percentile(nulls[k], 95) for k in methods}
for k in methods:
    print("  %-10s null max: median=%.2f p95=%.2f" % (k, np.median(nulls[k]), thr[k]))
for depth in [5 * sigma]:
    hits = {k: 0 for k in methods}; stats = {k: [] for k in methods}
    for i in range(n_inj):
        e = rng.rand() * P_true
        y = sigma * rng.randn(N) + M @ (rng.randn(3) * amps) + depth * box(t, P_true, e, dur_true)
        for k, kw in methods.items():
            s, pf = search(y, **kw)
            stats[k].append(s)
            if s > thr[k] and abs(pf - P_true) / P_true < 0.01:
                hits[k] += 1
    print("depth=%.1f sigma: " % (depth / sigma) + "  ".join("%s=%d/%d(med stat %.2f)" % (k, hits[k], n_inj, np.median(stats[k])) for k in methods))

