#!/usr/bin/env python
"""v1.0.0 release-gate checks that go beyond the pytest suite.

Run on a GPU machine:

    python tools/check_release_gate.py

Checks:
  0. preflight -- every dependency the zero-skip suite run needs
     (pycuda, batman, transitleastsquares, nfft, astropy, cufinufft)
     imports; the gate FAILS if any is missing, so a "0 skipped" suite
     run is actually possible on this environment
  1. reduction_max equivalence — eebls_gpu_fast with use_optimized=True
     (bls_optimized.cu) agrees with the standard kernel (validates the
     s >= 32 reduction fix end-to-end)
  2. kernel-cache timing — second call of eebls_gpu_fast /
     eebls_gpu_fast_optimized skips compilation
  3. lomb_scargle_simple + batched_run_const_nfreq — validates the
     weights fix and the PR #59 memory_requirement (cufftEstimate1d) path
  4. CE compute_log_prob smoke + guard checks
  5. PDM: new (t, y, err) API vs deprecated path; fast vs reference kernels

Exits nonzero if any check fails.
"""
import sys
import time

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("[%s] %s%s" % (status, name, (" — " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def fake_transit(n=300, baseline=365.0, freq=1.0 / 2.5, q=0.05, depth=0.05,
                 sigma=0.01, seed=42):
    rand = np.random.RandomState(seed)
    t = np.sort(baseline * rand.rand(n))
    phase = (t * freq) % 1.0
    y = np.zeros_like(t)
    y[phase < q] -= depth
    y += sigma * rand.randn(n)
    dy = sigma * np.ones_like(y)
    return (t.astype(np.float32), y.astype(np.float32),
            dy.astype(np.float32))


def fake_sine(n=300, baseline=365.0, freq=1.0 / 5.0, sigma=0.1, seed=7):
    rand = np.random.RandomState(seed)
    t = np.sort(baseline * rand.rand(n))
    y = 12 + 0.1 * np.cos(2 * np.pi * freq * t) + sigma * rand.randn(n)
    dy = sigma * np.ones_like(y)
    return t, y, dy


def ce_numpy_reference(t, y, freqs, phase_bins=10, mag_bins=5):
    """Plain-numpy Graham et al. (2013) conditional entropy (unweighted,
    no bin overlap) for gating the default CE kernel."""
    yi = np.digitize(y, np.linspace(y.min(), y.max(), mag_bins + 1)[1:-1])
    out = np.zeros(len(freqs))
    n = len(t)
    for k, f in enumerate(freqs):
        phi = (t * f) % 1.0
        pi = np.minimum((phi * phase_bins).astype(int), phase_bins - 1)
        hist, _, _ = np.histogram2d(pi, yi, bins=[phase_bins, mag_bins],
                                    range=[[0, phase_bins], [0, mag_bins]])
        p = hist / n
        p_phi = p.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = p * np.log(p_phi / p)
        out[k] = np.nansum(np.where(p > 0, term, 0.0))
    return out


# Every optional dependency a zero-skip run of cuvarbase/tests needs:
# module name -> pip distribution name.
PREFLIGHT_MODULES = [
    ('pycuda', 'pycuda'),
    ('batman', 'batman-package'),
    ('transitleastsquares', 'transitleastsquares'),
    ('nfft', 'nfft'),
    ('astropy', 'astropy'),
    ('cufinufft', 'cufinufft'),
]


def preflight():
    """Import every dependency the zero-skip suite needs and print its
    version; a missing one fails the gate (it would silently turn into
    pytest skips otherwise)."""
    import importlib
    ok = True
    for module, dist in PREFLIGHT_MODULES:
        try:
            mod = importlib.import_module(module)
        except Exception as e:
            check("preflight: import %s" % module, False,
                  "%s: %s (pip install %s)" % (type(e).__name__, e, dist))
            ok = False
            continue
        version = getattr(mod, '__version__', None)
        if version is None:
            try:
                from importlib.metadata import version as _v
                version = _v(dist)
            except Exception:
                version = '?'
        check("preflight: import %s" % module, True,
              "version %s" % version)
    return ok


def main():
    # --- 0. preflight -------------------------------------------------
    if not preflight():
        print()
        print("RELEASE GATE: preflight FAILED -- install the missing "
              "dependencies above; a zero-skip suite run is not possible "
              "without them")
        return 1

    from cuvarbase.bls import eebls_gpu_fast, eebls_gpu_fast_optimized

    t, y, dy = fake_transit()
    f_inj = 1.0 / 2.5
    freqs = np.linspace(0.05, 1.0, 5000).astype(np.float32)

    # --- 1. reduction_max equivalence ---------------------------------
    t0 = time.time()
    p_std = eebls_gpu_fast(t, y, dy, freqs)
    t_std_first = time.time() - t0

    t0 = time.time()
    p_opt = eebls_gpu_fast_optimized(t, y, dy, freqs)
    t_opt_first = time.time() - t0

    corr = np.corrcoef(p_std, p_opt)[0, 1]
    denom = max(np.max(np.abs(p_std)), 1e-30)
    max_rel = np.max(np.abs(p_std - p_opt)) / denom
    same_peak = np.argmax(p_std) == np.argmax(p_opt)
    check("reduction_max equivalence (standard vs optimized kernel)",
          corr > 0.9999 and same_peak,
          "corr=%.6f max_rel_diff=%.2e argmax %s (std=%d opt=%d)"
          % (corr, max_rel, "same" if same_peak else "DIFFERS",
             np.argmax(p_std), np.argmax(p_opt)))

    f_best = freqs[np.argmax(p_std)]
    check("BLS recovers injected transit",
          abs(f_best - f_inj) < 0.01,
          "best=%.4f injected=%.4f" % (f_best, f_inj))

    # --- 2. kernel-cache timing ---------------------------------------
    t0 = time.time()
    eebls_gpu_fast(t, y, dy, freqs)
    t_std_second = time.time() - t0

    t0 = time.time()
    eebls_gpu_fast_optimized(t, y, dy, freqs)
    t_opt_second = time.time() - t0

    check("kernel cache: eebls_gpu_fast 2nd call faster",
          t_std_second < t_std_first / 2,
          "first=%.0fms second=%.0fms" % (1e3 * t_std_first,
                                          1e3 * t_std_second))
    check("kernel cache: eebls_gpu_fast_optimized 2nd call faster",
          t_opt_second < t_opt_first / 2,
          "first=%.0fms second=%.0fms" % (1e3 * t_opt_first,
                                          1e3 * t_opt_second))

    # --- 3. Lomb-Scargle ----------------------------------------------
    from cuvarbase.lombscargle import (lomb_scargle_simple,
                                       LombScargleAsyncProcess)

    ts, ys, dys = fake_sine()
    f_sine = 1.0 / 5.0
    ls_freqs, ls_power = lomb_scargle_simple(ts, ys, dys,
                                             samples_per_peak=10)
    f_ls = ls_freqs[np.argmax(ls_power)]
    check("lomb_scargle_simple recovers injected frequency",
          abs(f_ls - f_sine) / f_sine < 0.01,
          "best=%.4f injected=%.4f" % (f_ls, f_sine))

    # batched_run_const_nfreq exercises memory_requirement, which now
    # calls cufft.cufft.cufftEstimate1d (PR #59) — would crash if that
    # API path were wrong.
    proc = LombScargleAsyncProcess()
    batch = [fake_sine(seed=s) for s in (1, 2, 3)]
    results = proc.batched_run_const_nfreq(batch, batch_size=3,
                                           samples_per_peak=5)
    proc.finish()
    ok = (len(results) == 3 and
          all(np.all(np.isfinite(p)) for _, p in results))
    check("batched_run_const_nfreq (memory_requirement/cufftEstimate1d)",
          ok, "%d result sets, all finite" % len(results))

    # --- 4. Conditional entropy ----------------------------------------
    # NOTE: CE recovery is checked on a strong sinusoid, not the transit.
    # For the q=0.05 transit above, the CE global minimum legitimately
    # lands on the 2*f harmonic (folding a transit at 2f superimposes the
    # dip on itself), so transit argmin-recovery is not a valid CE gate.
    # Kernel correctness is instead gated by correlation against a plain
    # numpy conditional-entropy reference on identical (normalized) data.
    from cuvarbase.ce import ConditionalEntropyAsyncProcess

    ce_freqs = np.linspace(0.05, 1.0, 2000)
    rand = np.random.RandomState(13)
    tc = np.sort(365.0 * rand.rand(300))
    yc = 12 + 0.5 * np.cos(2 * np.pi * 0.2 * tc) + 0.05 * rand.randn(300)
    dyc = 0.05 * np.ones_like(yc)
    proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
    r = proc.run([(tc, yc, dyc)], freqs=ce_freqs)
    proc.finish()
    fr, cper = r[0]
    f_ce = fr[np.argmin(cper)]
    check("CE recovers strong sinusoid frequency",
          abs(f_ce - 0.2) < 0.01,
          "best=%.4f injected=%.4f" % (f_ce, 0.2))

    # GPU vs numpy reference on the transit data (run() mean-subtracts
    # t and y first, so the reference uses the same normalization).
    # The kernel's statistic is an offset/scaled variant of the textbook
    # CE, so gate on shared global minimum plus rank correlation rather
    # than numerical agreement.
    proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
    r = proc.run([(t, y, dy)], freqs=ce_freqs)
    proc.finish()
    fr, cper = r[0]
    ref = ce_numpy_reference(t - np.mean(t), y - np.mean(y), ce_freqs,
                             phase_bins=10, mag_bins=5)
    ce_corr = np.corrcoef(ref, cper)[0, 1]
    same_min = np.argmin(ref) == np.argmin(cper)
    check("CE periodogram matches numpy reference",
          ce_corr > 0.9 and same_min,
          "corr=%.4f argmin %s (ref=%.4f gpu=%.4f)"
          % (ce_corr, "same" if same_min else "DIFFERS",
             ce_freqs[np.argmin(ref)], fr[np.argmin(cper)]))

    proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5,
                                          compute_log_prob=True)
    r = proc.run([(t, y, dy)], freqs=ce_freqs)
    proc.finish()
    fr, logp = r[0]
    check("CE compute_log_prob=True returns finite periodogram",
          bool(np.all(np.isfinite(logp))),
          "min=%.3g max=%.3g" % (np.min(logp), np.max(logp)))

    try:
        ConditionalEntropyAsyncProcess(weighted=True, use_fast=True)
        check("CE rejects use_fast + weighted", False, "no exception")
    except Exception as e:
        check("CE rejects use_fast + weighted", True, type(e).__name__)

    # --- 5. PDM ---------------------------------------------------------
    import warnings
    from cuvarbase.pdm import PDMAsyncProcess
    from cuvarbase.utils import weights as make_weights

    pdm_freqs = np.linspace(0.05, 1.0, 2000).astype(np.float32)
    pdm_freqs += 0.5 * (pdm_freqs[1] - pdm_freqs[0])

    proc = PDMAsyncProcess()
    res_new = proc.run([(ts, ys, dys)], freqs=pdm_freqs,
                       kind='binned_linterp', nbins=20)
    proc.finish()
    frqs_new, p_new = res_new[0]

    proc = PDMAsyncProcess()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        res_dep = proc.run([(ts, ys, make_weights(dys), pdm_freqs)],
                           kind='binned_linterp', nbins=20)
    proc.finish()
    p_dep = res_dep[0]

    corr = np.corrcoef(p_new, p_dep)[0, 1]
    check("PDM new (t,y,err) API matches deprecated path",
          corr > 0.999, "corr=%.6f" % corr)

    proc = PDMAsyncProcess()
    res_fast = proc.run([(ts, ys, dys)], freqs=pdm_freqs,
                        kind='binned_linterp_fast', nbins=20)
    proc.finish()
    _, p_fast = res_fast[0]
    corr = np.corrcoef(p_new, p_fast)[0, 1]
    check("PDM fast kernel matches reference kernel",
          corr > 0.999, "corr=%.6f" % corr)

    f_pdm = frqs_new[np.argmax(p_new)]
    check("PDM recovers injected frequency",
          abs(f_pdm - f_sine) / f_sine < 0.01,
          "best=%.4f injected=%.4f" % (f_pdm, f_sine))

    try:
        proc = PDMAsyncProcess()
        proc.run([(ts, ys, dys)], freqs=pdm_freqs, block_size=512)
        check("PDM rejects block_size > 256", False, "no exception")
    except ValueError as e:
        check("PDM rejects block_size > 256", True, "ValueError")

    # --------------------------------------------------------------------
    print()
    if FAILURES:
        print("RELEASE GATE: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("RELEASE GATE: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
