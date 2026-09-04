"""Exp 3: does eebls_gpu (eebls_transit's default for ndata >= 500)
honor per-frequency q bounds?  Keplerian grid, pure noise + injected
signal; compare returned solution q against [qmin_f, qmax_f], and the
noise floor of eebls_gpu vs eebls_gpu_fast on the same grid.
Also: reduction argmax consistency with many blocks per frequency.
"""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, exact_bls
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu, eebls_transit,
                           transit_autofreq, q_transit, single_bls,
                           count_tot_nbins)

t, y, dy = make_data(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6,
                     snr=12, seed=11)
freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.02, fmax=3.0)
freqs = freqs[::max(1, len(freqs) // 3000)]
q0 = q_transit(freqs)
qmins, qmaxes = 0.5 * q0, 2.0 * q0
print("nfreqs=%d  q0 range [%.4g, %.4g]  global qmin=%.4g qmax=%.4g -> nbins0=%d nbinsf=%d nbins_tot=%d"
      % (len(freqs), q0.min(), q0.max(), qmins.min(), qmaxes.max(),
         int(np.floor(1 / qmaxes.max())), int(np.ceil(1 / qmins.min())),
         count_tot_nbins(int(np.floor(1 / qmaxes.max())), int(np.ceil(1 / qmins.min())), 0.2)))

ps, sols = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, dlogq=0.2, noverlap=3)
qs = np.array([s[0] for s in sols])
out = (qs < qmins * 0.999) | (qs > qmaxes * 1.001)
print("eebls_gpu (array bounds): %d/%d solutions have q OUTSIDE [qmin_f, qmax_f]; "
      "q_sol/q0 range [%.3g, %.3g]" % (out.sum(), len(freqs), (qs / q0).min(), (qs / q0).max()))
ps_scalar, sols_s = eebls_gpu(t, y, dy, freqs, qmin=float(qmins.min()), qmax=float(qmaxes.max()),
                              dlogq=0.2, noverlap=3)
print("eebls_gpu array bounds vs scalar (min,max) bounds: max|diff| = %.3e (identical=%s)"
      % (np.abs(ps - ps_scalar).max(), np.allclose(ps, ps_scalar)))
pf = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, dlogq=0.3, noverlap=2)
i0 = np.argmin(np.abs(freqs - 0.2))
mask = np.abs(freqs - 0.2) > 0.01
print("power at injected f=0.2 (q0=%.4f, q_inj=0.03): eebls_gpu=%.4f (q_sol=%.4f) fast=%.4f exact=%.4f"
      % (q0[i0], ps[i0], qs[i0], pf[i0], exact_bls(t, y, dy, 0.2, 0.03, 0.6)))
print("off-peak noise floor: eebls_gpu median=%.4f p99=%.4f max=%.4f | fast median=%.4f p99=%.4f max=%.4f"
      % (np.median(ps[mask]), np.percentile(ps[mask], 99), ps[mask].max(),
         np.median(pf[mask]), np.percentile(pf[mask], 99), pf[mask].max()))
print("peak / off-peak max: eebls_gpu %.2f  fast %.2f; argmax f: gpu=%.4f fast=%.4f"
      % (ps[i0] / ps[mask].max(), pf[i0] / pf[mask].max(), freqs[np.argmax(ps)], freqs[np.argmax(pf)]))
# high-frequency end: small q boxes searched where Keplerian q is large
hi = freqs > 2.0
print("f>2 (q0>%.3f): eebls_gpu q_sol median=%.4f (min %.4f) vs qmin_f min=%.4f; power median gpu=%.4f fast=%.4f"
      % (q0[hi].min(), np.median(qs[hi]), qs[hi].min(), qmins[hi].min(),
         np.median(ps[hi]), np.median(pf[hi])))

# eebls_transit default path (ndata>=500) = eebls_gpu
fr, pw, so = eebls_transit(t, y, dy, fmin=0.02, fmax=3.0)
qso = np.array([s[0] for s in so])
q0t = q_transit(fr)
outt = (qso < 0.5 * q0t * 0.999) | (qso > 2.0 * q0t * 1.001)
print("eebls_transit default (ndata=%d): nfreqs=%d, %d solutions outside Keplerian q window" % (len(t), len(fr), outt.sum()))

# argmax consistency across many blocks/freq (reduction_max in-place)
sb = np.array([single_bls(t, y, dy, f, qq, pp) for f, (qq, pp) in zip(freqs, sols)])
rel = np.abs(sb - ps) / np.maximum(ps, 1e-3)
print("single_bls(sol) vs power: max rel %.3e, #>2%%: %d/%d" % (rel.max(), (rel > 0.02).sum(), len(freqs)))
reps = [eebls_gpu(t, y, dy, freqs[:500], qmin=qmins[:500], qmax=qmaxes[:500], dlogq=0.2, noverlap=3)
        for _ in range(5)]
sols_rep = np.array([[s for s in r[1]] for r in reps])
print("5 repeats: solutions identical across runs: %s; powers max spread %.2e"
      % (np.all([np.allclose(sols_rep[0], sols_rep[k]) for k in range(1, 5)]),
         max(np.abs(reps[0][0] - reps[k][0]).max() for k in range(1, 5))))
