"""A3 diagnosis: why does the realized NFFT error floor at ~1e-3 when the
implemented L1 truncation bound promises ~1e-7..1e-11 at large m/float64?

Code-reading suspects (cunfft.cu):
  S1. `#define PI 3.14159...f` -- a FLOAT literal, used in every phase
      computation (nfft_shift, normalize) even under DOUBLE_PRECISION.
      Predicted error: |G(k)| * |theta_k| * 2.8e-8, with
      theta_k = 2*pi*n0*(k0+k)/ng (normalize) and phi up to 2*pi*|k0|
      (nfft_shift). For the standard test (x0=-0.5, k0=-nf/2, nf=500,
      ||y||_1 ~ 67) this is ~1e-3 absolute: m-independent, nearly
      precision-independent -- matching the observed floor.
  S2. `modflt` has a hardcoded `float` return type -> fractional grid
      position truncated to float32 in precompute_psi (fast grid path).
  S3. --use_fast_math (default True) degrades float32 trig further.
  S4. Deconvolution amplification exp(b*khat^2) at band edges amplifies
      grid roundoff (relevant at float32 only).

Discriminating experiments (all vs exact CPU direct sums, float64):
  A. m-sweep at float64, sigma=2, centered band (x0=-0.5, k0=-nf/2):
     does the error track the bound 4*exp(-m*D)*||y||_1 or floor?
  B. same sweep at float32.
  C. sigma-sweep at m=12, float64: aliasing hypothesis predicts strong
     sigma dependence; phase hypothesis predicts none.
  D. use_fast_math=False at m=12: isolates S3.
  E. THE SMOKING GUN: identical transform content with phase factors
     on vs off.
       E1: t in [-0.5,0.5), k0=-nf/2  -> theta_k = -pi*(k0+k) != 0
       E2: t in [0,1),      k0=0      -> n0=0, k0=0: NO phase factors
     |G(k)| is identical between E1(mode j) and E2(mode j), j=|k0+k|,
     and the deconvolution khat range is identical. If E2 hits the
     truncation bound while E1 floors at 1e-3 -> the phase factors
     (float PI) are the cause, not anything "inherent".
     Per-k error profiles recorded for both.
  F. fast_grid=False at m=12, float64: isolates S2 (slow path never
     calls modflt/precompute_psi).
  G. nf scaling at m=12, float64, centered: phase-error model predicts
     floor ~ linear in nf (theta_max = pi*nf/2); truncation bound is
     nf-independent.

Prints JSON to stdout (JSON_RESULT line). Runtime ~2-4 min (one kernel
compile per unique (sigma, double, fast_math) combo).
"""
import json
import sys

import numpy as np

from cuvarbase.cunfft import NFFTAsyncProcess

SPP = 1


def direct_sums(t, y, freqs):
    def sfunc(func):
        return [np.sum(y * func(2 * np.pi * t * f)) for f in freqs]
    return np.asarray(sfunc(np.cos)) + 1j * np.asarray(sfunc(np.sin))


def make_data(seed=100, ndata=100, sigma_noise=0.1, x0=-0.5):
    rand = np.random.RandomState(seed)
    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (3. / (max(t) - min(t))) * t)
    y += sigma_noise * rand.randn(len(t))
    # scale to [x0, x0 + 1)
    tsc = (t - min(t)) / (max(t) - min(t)) + x0
    return tsc, y


def gpu_nfft(tsc, y, nf, m, sigma, use_double, k0, fast_math=True,
             fast_grid=True):
    proc = NFFTAsyncProcess(sigma=sigma, m=m, autoset_m=False,
                            use_double=use_double,
                            use_fast_math=fast_math)
    results = proc.run([(tsc, y, int(nf))],
                       minimum_frequency=float(k0),
                       samples_per_peak=SPP,
                       fast_grid=fast_grid)
    proc.finish()
    return np.asarray(results[0])


def run_case(label, m=12, sigma=2, use_double=True, x0=-0.5, k0=None,
             nf=500, ndata=100, fast_math=True, fast_grid=True,
             keep_profile=False):
    tsc, y = make_data(ndata=ndata, x0=x0)
    if k0 is None:
        k0 = -nf // 2
    ghat = gpu_nfft(tsc, y, nf, m, sigma, use_double, k0,
                    fast_math=fast_math, fast_grid=fast_grid)
    freqs = k0 + np.arange(nf)
    ref = direct_sums(tsc, y, freqs)

    err = np.abs(np.asarray(ghat, dtype=np.complex128) - ref)
    l1 = float(np.sum(np.abs(y)))
    D = np.pi * (1. - 1. / (2. * sigma - 1.))
    bound = 4. * np.exp(-m * D) * l1

    case = dict(label=label, m=int(m), sigma=float(sigma),
                use_double=bool(use_double), x0=float(x0), k0=int(k0),
                nf=int(nf), ndata=int(ndata), fast_math=bool(fast_math),
                fast_grid=bool(fast_grid),
                l1=l1, bound=float(bound),
                max_abs_err=float(err.max()),
                rms_err=float(np.sqrt(np.mean(err ** 2))),
                err_over_bound=float(err.max() / bound))
    if keep_profile:
        case["err_profile"] = [float(e) for e in err]
        case["absG_profile"] = [float(a) for a in np.abs(ref)]
        case["mode_profile"] = [int(f) for f in freqs]
    print("%-28s m=%2d sig=%g dbl=%d fm=%d fg=%d nf=%5d x0=%+.1f k0=%+5d"
          "  max_err=%.3e  bound=%.3e  ratio=%.2e"
          % (label, m, sigma, use_double, fast_math, fast_grid, nf, x0,
             k0, case["max_abs_err"], bound, case["err_over_bound"]),
          flush=True)
    return case


def main():
    cases = []

    # A: m-sweep, float64
    for m in (2, 4, 6, 8, 10, 12, 14, 16):
        cases.append(run_case("A_msweep_f64", m=m))

    # B: m-sweep, float32
    for m in (2, 4, 6, 8, 10, 12, 14, 16):
        cases.append(run_case("B_msweep_f32", m=m, use_double=False))

    # C: sigma-sweep at m=12, float64
    for sig in (2, 3, 4, 5):
        cases.append(run_case("C_sigsweep_f64", m=12, sigma=sig))

    # D: fast-math off, m=12
    cases.append(run_case("D_nofastmath_f64", m=12, fast_math=False))
    cases.append(run_case("D_nofastmath_f32", m=12, use_double=False,
                          fast_math=False))

    # E: smoking gun -- phase factors on (E1) vs off (E2)
    cases.append(run_case("E1_phases_on_f64", m=12, x0=-0.5, k0=-250,
                          keep_profile=True))
    cases.append(run_case("E2_phases_off_f64", m=12, x0=0.0, k0=0,
                          keep_profile=True))

    # F: slow-grid path (no modflt/precompute_psi), m=12, float64
    cases.append(run_case("F_slowgrid_f64", m=12, fast_grid=False))

    # G: nf scaling, m=12, float64, centered
    for nf in (200, 500, 1000, 2000):
        cases.append(run_case("G_nfscale_f64", m=12, nf=nf, k0=-nf // 2))

    print("JSON_RESULT: " + json.dumps({"cases": cases}), flush=True)


if __name__ == '__main__':
    main()
