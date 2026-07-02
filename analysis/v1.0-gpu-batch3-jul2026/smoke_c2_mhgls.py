"""C2 pod smoke test: multiharmonic GLS ghat_g spectrum layout.

Runs LombScargleAsyncProcess(nharmonics=H) for H=2,3 on the A5000 and
asserts corr>0.999 against the pure-Python lomb_scargle_direct_sums
(nharms=H) reference on the same frequency grid. This validates that the
real (GPU) ghat_g spectrum layout read back by _mh_power_from_spectra
matches the convention verified by inspection in the Jul 2 audit
((m-1)*k0 + m*i indexing, H*(nf+k0)-k0 / 2H*(nf+k0)-k0 sizing).

Exit code 0 = all assertions pass. Prints JSON results to stdout.
"""
import json
import sys

import numpy as np

from cuvarbase.lombscargle import (LombScargleAsyncProcess,
                                   lomb_scargle_direct_sums)
from cuvarbase.utils import weights

NFFT_SIGMA = 5
SPP = 3
NYQUIST_FACTOR = 3


def make_data(seed=100, ndata=150, freq=3.0, sigma=0.1):
    rand = np.random.RandomState(seed)
    t = np.sort(rand.rand(ndata))
    # non-sinusoidal signal: fundamental + strong 2nd/3rd harmonics so the
    # multiharmonic model has real structure to fit at H=2,3
    y = (np.cos(2 * np.pi * freq * t)
         + 0.6 * np.cos(2 * np.pi * 2 * freq * t + 0.3)
         + 0.4 * np.cos(2 * np.pi * 3 * freq * t + 1.1))
    y += sigma * rand.randn(ndata)
    err = sigma * np.ones_like(y)
    return t, y, err


def cpu_reference(t, y, err, freqs, nharms):
    w = weights(err)
    ybar = np.dot(w, y)
    yw = w * (y - ybar)          # same centered convention as the GPU memory
    yy = np.dot(w, (y - ybar) ** 2)
    return lomb_scargle_direct_sums(t, yw, w, freqs, yy, nharms=nharms)


def run_case(H, use_double):
    t, y, err = make_data()
    proc = LombScargleAsyncProcess(use_double=use_double,
                                   sigma=NFFT_SIGMA,
                                   nharmonics=H)
    results = proc.run([(t, y, err)],
                       nyquist_factor=NYQUIST_FACTOR,
                       samples_per_peak=SPP,
                       use_fft=True)
    proc.finish()
    fgpu, pgpu = results[0]
    pcpu = cpu_reference(t, y, err, fgpu, H)

    corr = float(np.corrcoef(pgpu, pcpu)[0, 1])
    maxabs = float(np.max(np.abs(np.asarray(pgpu, dtype=np.float64) - pcpu)))
    peak_gpu = float(fgpu[np.argmax(pgpu)])
    peak_cpu = float(fgpu[np.argmax(pcpu)])
    return dict(H=H, use_double=use_double, nf=int(len(fgpu)),
                corr=corr, max_abs_diff=maxabs,
                peak_freq_gpu=peak_gpu, peak_freq_cpu=peak_cpu)


def main():
    out = {"cases": []}
    ok = True
    for H in (2, 3):
        for use_double in (True, False):
            case = run_case(H, use_double)
            out["cases"].append(case)
            passed = case["corr"] > 0.999
            # the assertion gates on the double-precision path; float32 is
            # informational (NFFT float32 vs float64 reference)
            if use_double and not passed:
                ok = False
            case["pass"] = bool(passed)
            print("H=%d double=%s nf=%d corr=%.6f max|diff|=%.3e "
                  "peak_gpu=%.4f peak_cpu=%.4f %s"
                  % (H, use_double, case["nf"], case["corr"],
                     case["max_abs_diff"], case["peak_freq_gpu"],
                     case["peak_freq_cpu"],
                     "PASS" if passed else "FAIL"), flush=True)
    out["ok"] = ok
    print("JSON_RESULT: " + json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
