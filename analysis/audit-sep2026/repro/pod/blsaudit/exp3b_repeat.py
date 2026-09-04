"""Exp 3b: run-to-run solution nondeterminism of eebls_gpu -- near-ties or errors?"""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data
from cuvarbase.bls import eebls_gpu, single_bls, transit_autofreq, q_transit

t, y, dy = make_data(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6, snr=12, seed=11)
freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.02, fmax=3.0)
freqs = freqs[::max(1, len(freqs) // 3000)][:800]
q0 = q_transit(freqs)
reps = [eebls_gpu(t, y, dy, freqs, qmin=0.5 * q0, qmax=2 * q0, dlogq=0.2, noverlap=3) for _ in range(6)]
p0, s0 = reps[0]
s0 = np.array(s0)
for k in range(1, 6):
    pk, sk = reps[k]
    sk = np.array(sk)
    diff = np.where(~np.isclose(sk, s0).all(axis=1))[0]
    if len(diff) == 0:
        print("rep %d: identical solutions" % k); continue
    worst = 0.
    for i in diff:
        a = single_bls(t, y, dy, freqs[i], s0[i][0], s0[i][1])
        b = single_bls(t, y, dy, freqs[i], sk[i][0], sk[i][1])
        worst = max(worst, abs(a - b) / max(p0[i], 1e-6))
    print("rep %d: %d/%d solutions differ; max rel power difference between the two (q,phi) = %.2e; example i=%d sols %s vs %s powers %.6f %.6f"
          % (k, len(diff), len(freqs), worst, diff[0], s0[diff[0]], sk[diff[0]], p0[diff[0]], pk[diff[0]]))
