"""Exp 10b: eebls_gpu results with per-frequency q bounds depend on freq_batch_size;
sparse vs standard path comparability across sparse_threshold."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data
from cuvarbase.bls import eebls_gpu, eebls_transit, transit_autofreq, q_transit, sparse_bls_gpu

t, y, dy = make_data(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6, snr=12, seed=11)
freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.02, fmax=3.0)
freqs = freqs[::max(1, len(freqs) // 2000)]; q0 = q_transit(freqs)
pa, sa = eebls_gpu(t, y, dy, freqs, qmin=0.5 * q0, qmax=2 * q0)
pb, sb = eebls_gpu(t, y, dy, freqs, qmin=0.5 * q0, qmax=2 * q0, freq_batch_size=200)
pc, sc = eebls_gpu(t, y, dy, freqs, qmin=0.5 * q0, qmax=2 * q0, freq_batch_size=20)
print("eebls_gpu array q bounds: freq_batch_size None vs 200: max|diff|=%.3e (#>1e-4: %d/%d); None vs 20: max|diff|=%.3e (#>1e-4: %d)"
      % (np.abs(pa - pb).max(), (np.abs(pa - pb) > 1e-4).sum(), len(freqs), np.abs(pa - pc).max(), (np.abs(pa - pc) > 1e-4).sum()))
print("  median off-peak power: batch=None %.4f, 200 %.4f, 20 %.4f" % (np.median(pa), np.median(pb), np.median(pc)))

# sparse vs standard across the threshold on the SAME data (ndata=499 vs 501 subsets)
rng = np.random.RandomState(3)
t2, y2, dy2 = make_data(ndata=501, baseline=200., freq=0.2, q=0.03, phi0=0.6, snr=10, seed=12)
fr, pw_std, so_std = eebls_transit(t2, y2, dy2, fmin=0.05, fmax=1.0)
fr2, pw_sp, so_sp = eebls_transit(t2, y2, dy2, fmin=0.05, fmax=1.0, use_sparse=True)
i0 = np.argmin(np.abs(fr - 0.2)); mask = np.abs(fr - 0.2) > 0.01
print("ndata=501 same data, standard (eebls_gpu) vs sparse: nfreqs=%d; peak@0.2 std=%.4f sparse=%.4f; off-peak median std=%.4f sparse=%.4f; p99 std=%.4f sparse=%.4f; peak/max-offpeak std=%.2f sparse=%.2f"
      % (len(fr), pw_std[i0], pw_sp[i0], np.median(pw_std[mask]), np.median(pw_sp[mask]),
         np.percentile(pw_std[mask], 99), np.percentile(pw_sp[mask], 99), pw_std[i0] / pw_std[mask].max(), pw_sp[i0] / pw_sp[mask].max()))
qs_std = np.array([s[0] for s in so_std]); qs_sp = np.array([s[0] for s in so_sp]); qk = q_transit(fr)
print("  fraction of solutions with q outside [0.5,2]*q_kep: standard %.3f  sparse %.3f" % (np.mean((qs_std < 0.5 * qk * 0.999) | (qs_std > 2 * qk * 1.001)), np.mean((qs_sp < 0.5 * qk * 0.999) | (qs_sp > 2 * qk * 1.001))))
