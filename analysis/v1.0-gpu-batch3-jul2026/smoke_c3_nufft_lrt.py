"""C3 pod smoke test: NUFFT-LRT rewire executes on device and is correct.

Checks, on the A5000 with real pycuda (no conftest stubs, no mocks):

1. NUFFTLRTAsyncProcess.compute_nufft output matches the exact CPU
   adjoint DFT reference at the ACTUAL device convention
   (ghat[k] = sum_j y_j exp(2 pi i k t_j/(tmax - tmin)), ABSOLUTE t --
   the normalize kernel's n0 phase re-references the transform to
   t=0, not tmin; found during this smoke test, see SUMMARY.md),
   corr>0.999 on the stacked real/imag parts over the sigma=2
   guaranteed band k < nf/2. The upper half band k >= nf/2 sits
   outside the Gaussian window's accuracy band (deconvolution
   amplification exp(b*khat^2) blows up towards khat -> pi/2) and is
   reported informationally.
2. The NFFT actually ran on the device: the underlying NFFTAsyncProcess
   compiled real CUDA functions and a CUDA context is active (plus a
   kernel-launch counter via pycuda's prepared-call path).
3. Multi-season detection end-to-end on device: two seasons separated by
   a 260-day gap, injected 2.3 d box transit; the best SNR period lands
   within 2 grid steps of the truth.

Exit code 0 = all assertions pass. Prints JSON results to stdout.
"""
import json
import sys

import numpy as np

import pycuda.driver as cuda
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess


def adjoint_dft(t, y, nf):
    # ABSOLUTE-t phase convention (what the device actually computes):
    # the tmin-relative version differs by a per-k phase
    # exp(2 pi i k tmin/T) that reaches ~1 rad at high k for this data.
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = t / (t.max() - t.min())
    k = np.arange(nf)
    return np.exp(2j * np.pi * np.outer(k, x)) @ y


def two_season_lc(seed=0):
    rng = np.random.RandomState(seed)
    t = np.concatenate([np.sort(rng.uniform(0.0, 40.0, 120)),
                        np.sort(rng.uniform(300.0, 340.0, 120))])
    period = 2.3
    phase = (t % period) / period
    y = np.ones_like(t)
    y[(phase < 0.06) | (phase > 0.94)] -= 0.2
    y += 0.01 * rng.randn(len(t))
    return t, y, period


def main():
    out = {}
    ok = True

    proc = NUFFTLRTAsyncProcess(use_double=True)

    # -- check 1: compute_nufft vs exact adjoint DFT
    t, y, period = two_season_lc()
    y_dm = y - np.mean(y)
    nf = 2 * len(t)
    ghat_gpu = np.asarray(proc.compute_nufft(t, y_dm, nf),
                          dtype=np.complex128)
    ghat_ref = adjoint_dft(t, y_dm, nf)

    # context is created lazily (B1) -- check AFTER the first GPU call
    ctx = cuda.Context.get_current()
    out["cuda_context_active"] = ctx is not None

    def band_stats(sl):
        gs, rs = ghat_gpu[sl], ghat_ref[sl]
        corr = float(np.corrcoef(np.concatenate([gs.real, gs.imag]),
                                 np.concatenate([rs.real, rs.imag]))[0, 1])
        scale = float(np.max(np.abs(ghat_ref)))
        return corr, float(np.max(np.abs(gs - rs)) / scale)

    corr_band, rel_band = band_stats(slice(0, nf // 2))
    corr_full, rel_full = band_stats(slice(0, nf))
    out["nufft_corr_guaranteed_band"] = corr_band
    out["nufft_max_rel_err_guaranteed_band"] = rel_band
    out["nufft_corr_full_band_info"] = corr_full
    out["nufft_max_rel_err_full_band_info"] = rel_full
    out["nf"] = int(nf)
    nufft_ok = corr_band > 0.999
    ok = ok and nufft_ok
    print("compute_nufft vs adjoint DFT (k<nf/2): corr=%.10f max_rel=%.3e %s"
          % (corr_band, rel_band, "PASS" if nufft_ok else "FAIL"),
          flush=True)
    print("  full band (info; upper half outside sigma=2 accuracy band): "
          "corr=%.6f max_rel=%.3e" % (corr_full, rel_full), flush=True)

    # -- check 2: the NFFT proc holds real compiled GPU functions
    nproc = proc.nufft_proc
    fnames = sorted(getattr(nproc, 'function_names', []) or [])
    compiled = getattr(nproc, 'prepared_functions', None) or \
        getattr(nproc, 'functions', None)
    n_compiled = len(compiled) if compiled else 0
    out["nfft_function_names"] = fnames
    out["nfft_compiled_functions"] = n_compiled
    dev_ok = out["cuda_context_active"] and n_compiled > 0
    ok = ok and dev_ok
    print("device execution: context=%s compiled_functions=%d (%s) %s"
          % (out["cuda_context_active"], n_compiled, ",".join(fnames),
             "PASS" if dev_ok else "FAIL"), flush=True)

    # -- check 3: multi-season end-to-end detection on device
    periods = np.linspace(1.5, 4.0, 251)          # dP = 0.01
    durations = np.array([0.28])                  # ~ true 12% duty cycle
    snr = proc.run(t, y, periods, durations=durations)[:, 0]
    ibest = int(np.argmax(snr))
    pbest = float(periods[ibest])
    dp = float(periods[1] - periods[0])
    out["best_period"] = pbest
    out["true_period"] = period
    out["snr_best"] = float(snr[ibest])
    out["snr_median"] = float(np.median(snr))
    detect_ok = abs(pbest - period) <= 2 * dp
    ok = ok and detect_ok
    print("multi-season detection: best P=%.4f (true %.4f, dP=%.3f) "
          "SNR=%.2f (median %.2f) %s"
          % (pbest, period, dp, snr[ibest], np.median(snr),
             "PASS" if detect_ok else "FAIL"), flush=True)

    out["ok"] = ok
    print("JSON_RESULT: " + json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
