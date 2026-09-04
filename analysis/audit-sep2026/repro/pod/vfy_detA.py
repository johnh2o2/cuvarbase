"""Verifier reproduction for 'lrt-detectorA-defeated' (auditor finding 28).

Uses the validation harness's own data model (ground sampling, white+OU
red noise, 3 shared systematics modes at 6/3/6 sigma with random N(0,1)
coefficients, PCA basis + population prior) and compares, at the true
period with a 32-epoch scan (max statistic):

  matched      : default matched filter, PSD from y
  marg_est     : detector='marginal', default estimate_psd=True (PSD from y - V mu)
  marg_resid   : detector='marginal', PSD from the OLS residual y - V c_hat (proposed fix)
  marg_oracle  : detector='marginal', PSD from the systematics-free noise realization
  sequential   : detector='sequential' (OLS detrend, PSD from residual)

Null-calibrated (p95 over n_null systematics-only realizations) completeness
at several depths, plus an amplitude sweep of the single-template SNR.
"""
import sys, warnings, importlib.util
import numpy as np
warnings.filterwarnings('ignore')
spec = importlib.util.spec_from_file_location(
    'harness', '/workspace/cuvarbase/scripts/nufft_lrt_validation.py')
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram

rng = np.random.RandomState(11)
t = H.make_times(rng); n = len(t); nf = 2 * n
sigma_w = 3e-3; sigma_r = 1.0 * sigma_w; tau = 0.8
sys_amps = np.array([6.0, 3.0, 6.0]) * sigma_w
M, V_est, mu_c, cov_c = H.build_basis_from_population(
    rng, t, 90.0, sigma_w, sigma_r, tau, sys_amps)
print('n=%d nf=%d  mu_c=%s  diag(cov_c)=%s' % (n, nf, np.round(mu_c, 4),
                                                np.round(np.diag(cov_c), 6)))
proc = NUFFTLRTAsyncProcess()
P, dur, ep = 5.3, 0.22, 1.0
epochs = np.linspace(0, P, 32, endpoint=False)
epochs[np.argmin(np.abs(epochs - ep))] = ep   # on-grid true epoch


def est_psd(resid, smooth_window=5, eps_floor=1e-12):
    """Replicates run()'s PSD estimate (nufft_lrt.py:475-481) on `resid`."""
    Yr = proc.compute_nufft(t, (resid - resid.mean()).astype(np.float32), nf)
    psd = (np.abs(Yr) ** 2).astype(np.float32)
    psd = _smoothed_periodogram(psd, smooth_window)
    med = np.median(psd[psd > 0])
    return np.maximum(psd, np.float32(eps_floor) * np.float32(med))


MARG = dict(detector='marginal', systematics_basis=V_est,
            coeff_prior_mean=mu_c, coeff_prior_cov=cov_c)
SEQ = dict(detector='sequential', systematics_basis=V_est)


def stat(y, noise=None, single=False, **kw):
    e = np.array([ep]) if single else epochs
    return float(proc.run(t, y, np.array([P]), durations=np.array([dur]),
                          epochs=e, nf=nf, **kw).max())


def variants(y, noise, single=False):
    c_ols, *_ = np.linalg.lstsq(V_est, y - y.mean(), rcond=None)
    out = {
        'matched': stat(y, single=single),
        'marg_est': stat(y, single=single, **MARG),
        'marg_resid': stat(y, single=single, estimate_psd=False,
                           psd=est_psd(y - V_est @ c_ols), **MARG),
        'sequential': stat(y, single=single, **SEQ),
    }
    if noise is not None:
        out['marg_oracle'] = stat(y, single=single, estimate_psd=False,
                                  psd=est_psd(noise), **MARG)
    return out


def make(rng, depth=None, amp_scale=1.0, return_noise=False):
    noise = sigma_w * rng.randn(n) + H.ou_noise(rng, t, sigma_r, tau)
    y = 1.0 + noise + M @ (rng.randn(3) * sys_amps * amp_scale)
    if depth:
        y = y + H.box_transit(t, P, ep, dur, depth)
    return (y, noise) if return_noise else y


# ---- 1. amplitude sweep, single true template, depth 0.008
print('\n== single-template SNR at true (P,dur,epoch), depth=0.008, '
      'averaged over 8 realizations, systematics amplitude scale x{0,0.5,1,2,3}')
keys = ['matched', 'marg_est', 'marg_resid', 'marg_oracle', 'sequential']
print('%-6s ' % 'scale' + ' '.join('%11s' % k for k in keys))
for sc in (0.0, 0.5, 1.0, 2.0, 3.0):
    acc = {k: [] for k in keys}
    for r in range(8):
        y, noise = make(rng, 0.008, sc, True)
        v = variants(y, noise, single=True)
        for k in keys:
            acc[k].append(v[k])
    print('%-6.1f ' % sc + ' '.join('%11.2f' % np.mean(acc[k]) for k in keys))

# ---- 2. null-calibrated completeness at harness amplitudes (scale 1)
n_null, n_inj = 60, 30
print('\n== null-calibrated completeness (max over 32 epochs at true P), '
      'n_null=%d n_inj=%d, harness systematics (6/3/6 sigma)' % (n_null, n_inj))
nulls = {k: [] for k in keys}
for i in range(n_null):
    y, noise = make(rng, None, 1.0, True)
    v = variants(y, noise)
    for k in keys:
        nulls[k].append(v[k])
p95 = {k: np.percentile(nulls[k], 95) for k in keys}
print('null p95 : ' + ' '.join('%s=%.2f' % (k, p95[k]) for k in keys))
print('null std : ' + ' '.join('%s=%.2f' % (k, np.std(nulls[k])) for k in keys))
for depth in (0.004, 0.008, 0.016):
    hits = {k: 0 for k in keys}; vals = {k: [] for k in keys}
    for i in range(n_inj):
        y, noise = make(rng, depth, 1.0, True)
        v = variants(y, noise)
        for k in keys:
            vals[k].append(v[k])
            hits[k] += v[k] > p95[k]
    print('depth %.3f completeness: ' % depth
          + ' '.join('%s=%.2f(med %.1f)' % (k, hits[k] / n_inj,
                                            np.median(vals[k])) for k in keys))
