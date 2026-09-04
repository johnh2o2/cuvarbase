"""Exp 7: hone_solution behaviour; eebls_gpu_custom repeated compile."""
import sys, time
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, exact_bls, exact_best_at_freq
from cuvarbase.bls import hone_solution, single_bls, eebls_gpu, eebls_gpu_custom

for t0 in [0.0, 2457000.0]:
    t, y, dy = make_data(ndata=1500, baseline=60., freq=0.7123, q=0.04, phi0=0.33, snr=15, seed=21, t0=t0)
    T = t.max() - t.min()
    freqs = np.linspace(0.6, 0.8, 1500)
    p, sols = eebls_gpu(t, y, dy, freqs, qmin=0.01, qmax=0.1, dlogq=0.2, noverlap=3)
    i = int(np.argmax(p)); f0 = freqs[i]; q0, phi0 = sols[i]
    df0 = freqs[1] - freqs[0]
    tt = time.perf_counter()
    f, pn, niter, (q, phi) = hone_solution(t, y, dy, f0, df0, q0, 0.2, phi0)
    dt = time.perf_counter() - tt
    print("t0=%g: coarse best f=%.6f p=%.5f (q=%.4f phi=%.4f); honed f=%.6f p=%.5f q=%.4f phi=%.4f in %d iters (%.1fs); true f=0.7123 q=0.04 phi0=%.4f exact@true=%.5f; single_bls(honed)=%.5f"
          % (t0, f0, p[i], q0, phi0, f, pn, q, phi, niter, dt,
             (0.33 * 1 + 0.0) if t0 == 0 else ((0.33 - (np.floor(t.min()) - 0) * 0.7123 + np.floor(t.min()) * 0.7123) % 1.0),
             exact_bls(t, y, dy, 0.7123, 0.04, 0.33), single_bls(t, y, dy, f, q, phi)))
    print("   |f_honed - f_true| * T = %.3f cycles; exact power at honed (f,q,phi) in float64 = %.5f"
          % (abs(f - 0.7123) * T, exact_bls(t, y, dy, f, q, phi)))
