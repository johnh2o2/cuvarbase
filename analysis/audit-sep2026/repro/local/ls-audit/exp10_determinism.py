import sys
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc
from cuvarbase.lombscargle import LombScargleAsyncProcess
def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T); k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))
for label, N, T, fmax in [("ZTF-like", 300, 3650.0, 20.0), ("small", 300, 365.0, 20.0), ("Kepler-like", 65000, 1400.0, 50.0)]:
    freqs = grid(1.0 / (3 * T), fmax, T, spp=3)
    lcs = [make_lc(N=N, T=T, f0=1 + i, seed=i) for i in range(5)]
    for dbl in (False, True):
        proc = LombScargleAsyncProcess(use_double=dbl, sigma=4, m=8, autoset_m=False)
        r = [proc.batched_run_const_nfreq(lcs, freqs=freqs) for _ in range(3)]
        d = max(np.abs(a[1] - b[1]).max() for a, b in zip(r[0], r[1]))
        d2 = max(np.abs(a[1] - b[1]).max() for a, b in zip(r[0], r[2]))
        rr = proc.run(lcs, freqs=freqs); proc.finish()
        d3 = max(np.abs(a[1] - np.asarray(b[1][:len(freqs)])).max() for a, b in zip(r[0], rr))
        print("%s dbl=%s: run-to-run max|diff| = %.2e, %.2e ; batched vs run() = %.2e ; ref max power %.3f" % (label, dbl, d, d2, d3, max(a[1].max() for a in r[0])))
        del proc
