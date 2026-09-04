"""Exp 2: the q grid actually searched.

(a) enumerate q values tested by the fast path (m/nbf) and eebls_gpu
    (1/nb) for default and Keplerian-style bounds; is qmax ever tested?
(b) host (float64) vs device (float32) dnbins disagreement -> eebls_gpu
    bin-layout mismatch; find dlogq values that trigger it and run it.
(c) injected q near qmax: recovered power vs exact.
"""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, exact_bls, m_sequence, nb_levels, dnbins
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu, single_bls,
                           count_tot_nbins, q_transit)

print("--- (a) tested q values ---")
for qmin, qmax, dlogq in [(1e-2, 0.5, 0.3), (1e-2, 0.5, 0.2), (0.025, 0.1, 0.3),
                          (0.01, 0.04, 0.3), (0.005, 0.02, 0.3)]:
    nbf = int(np.float32(1.) / np.float32(qmin))
    nb0 = int(np.float32(1.) / np.float32(qmax))
    ms = m_sequence(nbf, nb0, dlogq, f32=True)
    qs = np.array(ms) / nbf
    print("fast  qmin=%g qmax=%g dlogq=%g: nbf=%d nb0=%d  q tested (%d) = %s  max=%g (qmax=%g)"
          % (qmin, qmax, dlogq, nbf, nb0, len(qs), np.round(qs, 4), qs.max(), qmax))
    nb0s = int(np.floor(1. / qmax)); nbfs = int(np.ceil(1. / qmin))
    qs2 = 1. / np.array(nb_levels(nb0s, nbfs, dlogq, f32=True))
    print("slow  (eebls_gpu) nb0=%d nbf=%d  q tested (%d) = %s" % (nb0s, nbfs, len(qs2), np.round(qs2, 4)))

print("--- (b) host vs device dnbins disagreement scan ---")
mism = []
for dlogq in np.round(np.arange(0.05, 1.0, 0.01), 2):
    for nb0 in range(1, 6):
        for nbf in [50, 100, 200, 500, 1000, 2000]:
            h = count_tot_nbins(nb0, nbf, dlogq)
            d = sum(nb_levels(nb0, nbf, dlogq, f32=True))
            if h != d:
                mism.append((dlogq, nb0, nbf, h, d))
print("mismatching (dlogq, nb0, nbf, host_tot, device_tot): %d cases; first 15:" % len(mism))
for m in mism[:15]:
    print("   ", m)

if mism:
    dlogq, nb0, nbf, h, d = mism[0]
    qmin, qmax = 1. / nbf, 1. / nb0
    t, y, dy = make_data(ndata=1500, baseline=20., freq=1.1, q=0.05, phi0=0.2, snr=15, seed=2)
    freqs = np.linspace(0.8, 1.4, 400)
    p, sols = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, dlogq=dlogq, noverlap=2)
    sb = np.array([single_bls(t, y, dy, f, qq, pp) for f, (qq, pp) in zip(freqs, sols)])
    rel = np.abs(sb - p) / np.maximum(p, 1e-3)
    print("eebls_gpu with dlogq=%g qmin=%g qmax=%g (host_tot=%d device_tot=%d): "
          "max|single_bls(sol)-power|/power = %.3e, #>2%%: %d/%d, any NaN=%s, power range [%.3g, %.3g]"
          % (dlogq, qmin, qmax, h, d, rel.max(), int((rel > 0.02).sum()), len(freqs),
             np.isnan(p).any(), p.min(), p.max()))
    # a safe dlogq for comparison
    p2, sols2 = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, dlogq=0.2, noverlap=2)
    sb2 = np.array([single_bls(t, y, dy, f, qq, pp) for f, (qq, pp) in zip(freqs, sols2)])
    rel2 = np.abs(sb2 - p2) / np.maximum(p2, 1e-3)
    print("  control dlogq=0.2: max rel dev = %.3e, #>2%%: %d/%d" % (rel2.max(), int((rel2 > 0.02).sum()), len(freqs)))

print("--- (c) injected q near qmax (Keplerian-style bounds) ---")
freq = 1.0
for qinj in [0.05, 0.075, 0.095, 0.1]:
    t, y, dy = make_data(ndata=3000, baseline=40., freq=freq, q=qinj, phi0=0.37, snr=20, seed=4)
    freqs = np.linspace(0.98, 1.02, 401)
    ex = exact_bls(t, y, dy, freq, qinj, 0.37)
    pf = eebls_gpu_fast(t, y, dy, freqs, qmin=0.025, qmax=0.1, dlogq=0.3, noverlap=2)
    ps, sols = eebls_gpu(t, y, dy, freqs, qmin=0.025, qmax=0.1, dlogq=0.2, noverlap=3)
    i0 = np.argmin(np.abs(freqs - freq))
    print("q_inj=%.3f (bounds [0.025,0.1]): exact=%.4f fast@f=%.4f (%.1f%%) slow@f=%.4f (%.1f%%) slow sol q=%.4f"
          % (qinj, ex, pf[i0], 100 * pf[i0] / ex, ps[i0], 100 * ps[i0] / ex, sols[i0][0]))
