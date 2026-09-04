"""Exp 9b: localize the fused-vs-multipass-vs-reference discrepancy at large t*f."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, ref_fast
from cuvarbase.bls import eebls_gpu_fast

for T in [365., 3650.]:
    for t0 in [0.0, 2455000.5]:
        t, y, dy = make_data(ndata=5000, baseline=T, freq=6.3, q=0.06, phi0=0.9, snr=15., seed=5, t0=t0)
        freqs = 6.3 + 0.00013 * np.arange(-40, 40)
        kw = dict(qmin=1e-2, qmax=0.5, dlogq=0.3)
        r0 = ref_fast(t, y, dy, freqs, noverlap=1, dphi=0.0, f32fold=True, **kw)
        r1 = ref_fast(t, y, dy, freqs, noverlap=1, dphi=0.5, f32fold=True, **kw)
        r2 = ref_fast(t, y, dy, freqs, noverlap=2, dphi=0.0, f32fold=True, **kw)
        g0 = eebls_gpu_fast(t, y, dy, freqs, noverlap=1, dphi=0.0, **kw)
        g1 = eebls_gpu_fast(t, y, dy, freqs, noverlap=1, dphi=0.5, **kw)
        gf = eebls_gpu_fast(t, y, dy, freqs, noverlap=2, dphi=0.0, **kw)          # fused
        gm = eebls_gpu_fast(t, y, dy, freqs, noverlap=2, dphi=1e-9, **kw)         # multipass
        gm2 = np.maximum(g0, g1)
        d = lambda a, b: np.abs(a - b).max()
        print("T=%.0f t0=%g (max t*f=%.0f): |g0-r0|=%.1e |g1-r1|=%.1e |max(g0,g1)-r2|=%.1e |gm-max(g0,g1)|=%.1e |gf-r2|=%.1e |gf-gm|=%.1e  #freqs gf!=gm(>1e-6): %d  gf<gm at: %d  gf>gm at: %d"
              % (T, t0, (t.max() - np.floor(t.min())) * freqs.max(), d(g0, r0), d(g1, r1), d(gm2, r2), d(gm, gm2), d(gf, r2), d(gf, gm),
                 int((np.abs(gf - gm) > 1e-6).sum()), int((gf < gm - 1e-6).sum()), int((gf > gm + 1e-6).sum())))
        if d(gf, gm) > 1e-6:
            i = int(np.argmax(np.abs(gf - gm)))
            print("   worst i=%d f=%.5f: fused=%.6f multipass=%.6f ref2=%.6f ref pass0=%.6f pass1=%.6f" % (i, freqs[i], gf[i], gm[i], r2[i], r0[i], r1[i]))
