# NUFFT-LRT: whitened matched-filter transit detection (Taaki)

> **⚠️ EXPERIMENTAL** — this module emits a `UserWarning` on import.
> The statistic and its implementation are audited correct
> (`analysis/nufft-lrt-audit-jul2026.md`) and an injection-recovery
> characterization exists (below), but the method has far less
> operational mileage than cuvarbase's BLS/TLS and its thresholds must
> be calibrated empirically per dataset (see "Statistical caveats").

## What this is

A frequency-domain **likelihood-ratio / matched-filter transit search
for correlated ("red") noise**, contributed by **Jamila Taaki**
([@xiaziyna](https://github.com/xiaziyna)). The lightcurve and each box
transit template are transformed with the GPU adjoint NFFT directly at
the observed (irregular, gappy) times over the full baseline, and the
detection statistic is the noise-whitened correlation

```
SNR = Re Σ_k [ Y_k T_k* / P(k) ]  /  sqrt( Σ_k |T_k|² / P(k) )
```

with the noise power spectrum `P(k)` either supplied or estimated from
the data (smoothed periodogram). Whitening by `P(k)` is what
distinguishes it from BLS/TLS, which weight points by their individual
error bars and otherwise assume *white* noise.

## Provenance, and exactly what is implemented

The method family is published in:

1. **Taaki, Kamalabadi & Kemball (2020), AJ 159, 283** ([arXiv:2004.14893](https://arxiv.org/abs/2004.14893)) — joint Bayesian
   transit detection + systematic-noise characterization on Kepler
   long-cadence data.
2. **Taaki, Kemball & Kamalabadi (2025), AJ 170, 14** ([arXiv:2504.18706](https://arxiv.org/abs/2504.18706)) — the TESS 2-min
   application.
3. Kay (1998/2002)-style adaptive detection under unknown noise PSDs is
   the signal-processing foundation.
4. Reference NUFFT prototype: [`code_nova_exoghosts`](https://github.com/star-skelly/code_nova_exoghosts).

`cuvarbase.nufft_lrt` implements, selectable via
`run(..., detector=...)`:

- **`'matched'`** (default) — the stationary PSD-whitened matched
  filter.
- **`'marginal'`** — **Detector A** of the 2020 paper: the joint
  detector with a Gaussian prior on systematics coefficients
  marginalized in closed form. Computed in the whitened frequency
  domain via the Woodbury identity, so the systematics basis costs one
  NFFT per basis vector per lightcurve and K-dimensional algebra per
  template. Supply `systematics_basis` (e.g. instrument cotrending
  vectors, or PCA modes of a lightcurve population) and
  `coeff_prior_cov` (+ optional `coeff_prior_mean`), estimated from
  population fits as in the paper.
- **`'sequential'`** — the papers' "standard" baseline: least-squares
  cotrend against the basis in the time domain, then the stationary
  filter on the residual.

Not implemented (deliberately): **Detector B** (joint MAP plug-in over
a depth grid) — the 2020 paper found it comparable to Detector A and
describes it as exploratory; the closed-form marginalization supersedes
the plug-in. The papers' phase-correlation epoch pre-estimation trick
(2020, Appendix A) is also not implemented — epochs are searched on an
explicit grid.

**Honesty note on citing the papers:** the published validations cover
*uniformly sampled* Kepler/TESS data, and the published gains of the
joint detectors are modest (~2% detection efficiency on Kepler; 0.2%
and not statistically significant on TESS). The NUFFT /
irregular-sampling variant in this module appears in no publication —
its characterization is the cuvarbase injection-recovery study below.
Do not cite the papers' numbers as this module's performance.

## When is this the right tool?

Decision guide, based on the measured injection-recovery study
(`analysis/nufft-lrt-audit-jul2026.md`, validation section, and
`benchmarks/results/nufft_lrt_validation_jul2026/`):

**Reach for NUFFT-LRT when all of these hold:**

1. **Your noise is genuinely correlated** on timescales comparable to
   transit durations (stellar activity, unmodeled instrument drift) —
   the whitening is the entire advantage; in white noise it can only
   tie BLS at best (and in practice pays a small penalty for
   estimating the PSD from the data).
2. **You are scoring a bounded set of candidates**, not running a blind
   survey: the cost is one adjoint NFFT *per template*
   (period × duration × epoch), so ~10³–10⁴ templates is comfortable
   and survey-scale grids (10⁶+) are not. Typical fits: vetting/
   re-ranking BLS or TLS candidates under a realistic noise model,
   or focused searches around known ephemerides.
3. **You can calibrate thresholds empirically** (see caveats).

**Prefer BLS** for blind box searches at scale (it is thousands of
times cheaper per trial and its white-noise statistic is
well-understood), **TLS** when limb-darkened template fidelity matters
for small planets. (Lomb-Scargle is not a transit competitor at all — a
short-duty-cycle box leaves only a small fraction of its power in the
sinusoidal fundamental, which is why box searches exist.)

**Use `detector='marginal'`** when you additionally have a shared
systematics basis (CBVs, PCA modes of a population) whose overfitting
during pre-detrending you want to avoid — this is the regime the 2020
paper targets.

## Statistical caveats (measured)

- **The "SNR" is not N(0,1).** With the PSD estimated from the data,
  the null distribution of the statistic is over-dispersed
  (measured std ≈ 1.7 on white noise with the default settings — the
  estimated-PSD modes are correlated and shared between numerator and
  normalization). **Never apply a textbook SNR≳7 threshold; calibrate
  the detection threshold on signal-free or scrambled data**, as the
  validation harness does (null-percentile calibration).
- **Self-whitening**: with `estimate_psd=True`, a strong transit
  inflates the PSD estimate at its own harmonic frequencies and
  partially suppresses itself. Provide `psd=` from a transit-free
  noise model when you have one.
- **Frequency resolution**: the default `nf = 2·len(t)` gives a
  maximum template frequency `nf / T_span`. Resolving a transit of
  duration `d` wants `nf ≳ a few × T_span / d` — raise `nf` for short
  transits on long sparse baselines.

## Usage

```python
import numpy as np
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

proc = NUFFTLRTAsyncProcess()

# 1) stationary whitened matched filter over a small grid
periods = np.linspace(1.0, 10.0, 100)
durations = np.linspace(0.1, 0.5, 5)
snr = proc.run(t, y, periods, durations=durations)   # (100, 5)

# 2) with an epoch axis (epoch grid should scale ~ P/duration)
snr = proc.run(t, y, np.array([P]), durations=np.array([d]),
               epochs=np.linspace(0, P, 40, endpoint=False))

# 3) Detector A (joint marginalized) with a systematics basis V (n, K)
#    and a coefficient prior estimated from population fits
snr = proc.run(t, y, periods, durations=durations,
               detector='marginal', systematics_basis=V,
               coeff_prior_mean=mu_c, coeff_prior_cov=cov_c)

# 4) known noise PSD (recommended when available)
snr = proc.run(t, y, periods, durations=durations,
               estimate_psd=False, psd=my_psd, nf=len(my_psd))
```

Threshold calibration sketch (do this for your dataset):

```python
null_maxima = []
for y_null in signal_free_or_scrambled_lightcurves:
    null_maxima.append(proc.run(t, y_null, periods, ...).max())
threshold = np.percentile(null_maxima, 95)   # 5% per-search FAR
```

## Validation summary (July 2026)

<!-- VALIDATION_RESULTS -->

Full protocol, raw JSON, and the audit:
`scripts/nufft_lrt_validation.py`,
`benchmarks/results/nufft_lrt_validation_jul2026/`,
`analysis/nufft-lrt-audit-jul2026.md`.

## Citation

If you use this module, please cite Taaki, Kamalabadi & Kemball (2020,
AJ 159, 283) for the method, Taaki, Kemball & Kamalabadi (2025, AJ 170,
14) for the space-photometry application, the reference prototype
(`code_nova_exoghosts`), and cuvarbase itself (see the main README).
