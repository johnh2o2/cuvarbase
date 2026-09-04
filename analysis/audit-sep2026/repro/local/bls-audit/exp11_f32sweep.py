"""Exp 11 (CPU, uses the fold verified bit-identical to the kernel in exp9):
power loss from float32 phase folding vs float64, as a function of T*f."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from blsref import make_data, ref_fast, exact_bls
from math import asin, pi
def q_transit(f, fmax0=8.6307):
    return asin(min(1., (f / fmax0) ** (2. / 3.))) / pi
print("%8s %8s %8s %10s | %-28s | %-28s" % ("T[d]", "f[1/d]", "T*f", "ulp[cyc]", "Keplerian q0 box: f64 / f32 / exact", "qmin=0.01 box(q=0.01): f64 / f32 / exact"))
rng_seed = 0
for T, f in [(27., 5.), (365., 5.), (1400., 5.), (3650., 2.), (3650., 5.), (3650., 8.), (7300., 8.)]:
    q0 = q_transit(f)
    rows = []
    for q, qmin, qmax in [(q0, 0.5 * q0, 2 * q0), (0.01, 0.01, 0.1)]:
        losses = []
        for seed in range(4):
            t, y, dy = make_data(ndata=5000, baseline=T, freq=f, q=q, phi0=0.37 + 0.1 * seed, snr=25., seed=seed)
            ex = exact_bls(t, y, dy, f, q, 0.37 + 0.1 * seed)
            p64 = ref_fast(t, y, dy, [f], qmin=qmin, qmax=qmax, noverlap=2)[0]
            p32 = ref_fast(t, y, dy, [f], qmin=qmin, qmax=qmax, noverlap=2, f32fold=True)[0]
            losses.append((p64 / ex, p32 / ex))
        l = np.array(losses)
        rows.append("%.3f / %.3f (min %.3f) / 1" % (l[:, 0].mean(), l[:, 1].mean(), l[:, 1].min()))
    print("%8.0f %8.1f %8.0f %10.2e | %-28s | %-28s" % (T, f, T * f, np.spacing(np.float32(T * f)), rows[0], rows[1]))
