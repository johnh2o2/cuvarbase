"""Exp 1: GPU BLS vs independent float64 references and astropy.

(a) eebls_gpu_fast / _optimized (fused and multipass) vs ref_fast
(b) eebls_gpu vs ref_slow (single batch)
(c) single_bls / exact vs astropy 'slow' snr/likelihood at the injected f
(d) t from 0 vs BJD-scale t (2.46e6) vs long-baseline (3650 d)
"""
import sys, time
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import (make_data, ref_fast, ref_slow, exact_bls,
                    exact_best_at_freq, m_sequence, nb_levels)
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu_fast_optimized,
                           eebls_gpu, eebls_gpu_custom, single_bls,
                           convert_bls_power)


def stats(a, b, name):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    rel = np.abs(d) / np.maximum(np.abs(b), 1e-3)
    c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    print("  %-46s maxabs=%.3e maxrel=%.3e (at i=%d) corr=%.7f argmax_eq=%s"
          % (name, np.abs(d).max(), rel.max(), int(np.argmax(np.abs(d))),
             c, np.argmax(a) == np.argmax(b)))
    return d


def run_case(label, t, y, dy, freqs, freq, q, phi0, settings):
    print("=== %s: ndata=%d T=%.1f tmin=%.1f f=%.4g q=%.4g phi0=%.3g"
          % (label, len(t), t.max() - t.min(), t.min(), freq, q, phi0))
    for st in settings:
        qmin, qmax, dlogq, nov, dphi = st
        kw = dict(qmin=qmin, qmax=qmax, dlogq=dlogq, noverlap=nov, dphi=dphi)
        print(" settings qmin=%s qmax=%s dlogq=%s noverlap=%d dphi=%s"
              % (qmin, qmax, dlogq, nov, dphi))
        r64 = ref_fast(t, y, dy, freqs, **kw)
        r32 = ref_fast(t, y, dy, freqs, f32fold=True, **kw)
        stats(r32, r64, "ref_fast(f32 fold) vs ref_fast(f64 fold)")
        g = eebls_gpu_fast(t, y, dy, freqs, **kw)
        stats(g, r64, "eebls_gpu_fast vs ref_fast(f64)")
        stats(g, r32, "eebls_gpu_fast vs ref_fast(f32 fold)")
        go = eebls_gpu_fast_optimized(t, y, dy, freqs, **kw)
        stats(go, r32, "eebls_gpu_fast_optimized vs ref_fast(f32)")
        stats(go, g, "optimized vs standard kernel")
        # multipass fallback (fused only for pow2 noverlap & dphi==0)
        kw2 = dict(kw); kw2['dphi'] = 1e-9
        gm = eebls_gpu_fast(t, y, dy, freqs, **kw2)
        stats(gm, g, "multipass(dphi=1e-9) vs fused")
        # power at the injected frequency
        i0 = int(np.argmin(np.abs(freqs - freq)))
        ex = exact_bls(t, y, dy, freq, q, phi0)
        print("  at injected f: exact(true q,phi)=%.5f gpu_fast=%.5f ref=%.5f  "
              "argmax f: gpu=%.6g ref=%.6g" % (ex, g[i0], r64[i0],
                                               freqs[np.argmax(g)],
                                               freqs[np.argmax(r64)]))


def slow_case(label, t, y, dy, freqs, freq, q, phi0, qmin=1e-2, qmax=0.5,
              dlogq=0.2, noverlap=3):
    print("=== eebls_gpu %s" % label)
    kw = dict(qmin=qmin, qmax=qmax, dlogq=dlogq, noverlap=noverlap)
    p, sols = eebls_gpu(t, y, dy, freqs, **kw)
    r, rs = ref_slow(t, y, dy, freqs, f32fold=True, **kw)
    stats(p, r, "eebls_gpu vs ref_slow(f32 fold)")
    r64, _ = ref_slow(t, y, dy, freqs, **kw)
    stats(p, r64, "eebls_gpu vs ref_slow(f64 fold)")
    # solution consistency: single_bls at the returned (q, phi)
    sb = np.array([single_bls(t, y, dy, f, qq, pp)
                   for f, (qq, pp) in zip(freqs, sols)])
    d = stats(sb, p, "single_bls(sol) vs eebls_gpu power")
    i0 = int(np.argmin(np.abs(freqs - freq)))
    print("  at injected f: gpu=%.5f ref=%.5f exact(true)=%.5f sol=(q=%.4f, phi=%.4f) ref_sol=(q=%.4f, phi=%.4f)"
          % (p[i0], r[i0], exact_bls(t, y, dy, freq, q, phi0),
             sols[i0][0], sols[i0][1], rs[i0][0], rs[i0][1]))
    bad = np.where(np.abs(d) > 0.02 * np.maximum(p, 1e-3))[0]
    print("  #freqs where single_bls(sol) deviates >2%%: %d / %d" % (len(bad), len(freqs)))
    return p, sols


def astropy_case(t, y, dy, freq, q, phi0):
    from astropy.timeseries import BoxLeastSquares
    m = BoxLeastSquares(t, y, dy=dy)
    P = 1. / freq
    dur = q * P
    for obj in ('snr', 'likelihood'):
        res = m.power(np.array([P]), np.array([dur]), method='slow',
                      objective=obj)
        ph_start = ((res.transit_time[0] - 0.5 * dur) * freq) % 1.0
        ex = exact_bls(t, y, dy, freq, q, ph_start)
        ours = convert_bls_power(ex, y, dy,
                                 'snr' if obj == 'snr' else 'loglik')
        w = dy ** -2.
        r = w[(((t * freq) - ph_start) % 1.0) < q].sum() / w.sum()
        extra = "" if obj == 'snr' else "  ours/(1-r)=%.5f (r=%.4f)" % (ours / (1 - r), r)
        print("  astropy %-10s at its own best phase: astropy=%.5f ours=%.5f%s"
              % (obj, res.power[0], ours, extra))
    # exact best over phase grid at true q vs astropy snr^2/chi2_0
    eb, bp = exact_best_at_freq(t, y, dy, freq, q)
    print("  exact best over phases at f,q: %.5f at phi=%.4f (true phi0=%.4f); depth astropy=%.5g"
          % (eb, bp, phi0, res.depth[0]))


if __name__ == '__main__':
    freq, q, phi0 = 1.37, 0.04, 0.31
    settings = [(1e-2, 0.5, 0.3, 2, 0.0),   # eebls_gpu_fast defaults
                (1e-2, 0.5, 0.3, 4, 0.0),
                (5e-3, 0.2, 0.1, 3, 0.0),
                (2e-2, 0.5, -1.0, 1, 0.25),
                (1e-2, 0.5, 0.5, 2, 0.0)]
    # short baseline
    t, y, dy = make_data(ndata=2000, baseline=30., freq=freq, q=q,
                         phi0=phi0, snr=15., seed=3)
    T = t.max() - t.min()
    df = 0.25 * 0.01 / T
    freqs = np.concatenate((np.arange(0.5, 3.0, 40 * df), np.arange(freq - 60 * df, freq + 60 * df, df)))
    freqs = np.sort(freqs)
    print("nfreqs=%d" % len(freqs))
    run_case("T=30d t0=0", t, y, dy, freqs, freq, q, phi0, settings)
    astropy_case(t, y, dy, freq, q, phi0)
    slow_case("T=30d t0=0", t, y, dy, freqs[::2], freq, q, phi0)

    # BJD scale
    tb = t + 2.46e6
    run_case("T=30d BJD", tb, y, dy, freqs, freq, q, phi0, settings[:2])
    slow_case("T=30d BJD", tb, y, dy, freqs[::2], freq, q, phi0)
    # custom path on BJD: power at true (q,phi) must equal single_bls
    q_values = np.array([0.02, 0.04, 0.08])
    phi_values = np.linspace(0, 1, 200, endpoint=False)
    fsub = freqs[::40]
    pc, sc = eebls_gpu_custom(tb, y, dy, fsub, q_values, phi_values)
    sb = np.array([single_bls(tb, y, dy, f, qq, pp) for f, (qq, pp) in zip(fsub, sc)])
    stats(sb, pc, "custom(BJD): single_bls(sol) vs power")
    pc0, sc0 = eebls_gpu_custom(t, y, dy, fsub, q_values, phi_values)
    stats(pc, pc0, "custom BJD vs t0=0")

    # long baseline, high frequency: float32 fold stress
    t, y, dy = make_data(ndata=5000, baseline=3650., freq=6.3, q=0.06,
                         phi0=0.9, snr=15., seed=5)
    T = t.max() - t.min()
    freqs = np.arange(6.2995, 6.3005, 0.25 * 0.02 / T)
    print("nfreqs=%d" % len(freqs))
    run_case("T=3650d t0=0 f~6.3", t, y, dy, freqs, 6.3, 0.06, 0.9,
             [(1e-2, 0.5, 0.3, 2, 0.0), (2e-3, 0.5, 0.3, 2, 0.0)])
    run_case("T=3650d BJD f~6.3", t + 2455000.5, y, dy, freqs, 6.3, 0.06, 0.9,
             [(1e-2, 0.5, 0.3, 2, 0.0)])
    slow_case("T=3650d t0=0", t, y, dy, freqs[::8], 6.3, 0.06, 0.9)
