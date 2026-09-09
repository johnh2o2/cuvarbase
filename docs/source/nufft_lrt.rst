NUFFT-LRT: whitened matched-filter transit detection (experimental)
*******************************************************************

.. warning::

   **EXPERIMENTAL.** ``cuvarbase.nufft_lrt`` is importable only by name
   (it is deliberately *not* exported from the top-level ``cuvarbase``
   namespace) and emits an ``EXPERIMENTAL`` ``UserWarning`` when
   :class:`~cuvarbase.nufft_lrt.NUFFTLRTAsyncProcess` is first
   constructed. It *is* validated: the September 2026 correctness fixes
   (below) were re-measured by the injection-recovery campaign of
   2026-09-06 (*Validation status*), and the public default path is
   correct on absolute BJD timestamps and recovers random-epoch
   transits. It stays experimental because that campaign also showed
   that what a 1.x freeze would lock in should still change: the
   default epoch grid costs 4-9 % of completeness against a finer one,
   PSD whitening -- the default detector's distinguishing feature --
   gave no gain over a flat PSD, and ``run()`` returns a tuple or an
   array depending on ``epochs``. So the module and its ``run()``
   signature are **outside the 1.x API-stability promise** and may
   change incompatibly in a 1.x release. Its thresholds must be
   calibrated empirically per dataset (*Statistical caveats*), and it
   has far less operational mileage than cuvarbase's BLS and TLS. Use
   it with those caveats, and quote only the measured numbers below.

What this is
============

A frequency-domain **likelihood-ratio / matched-filter transit search
for correlated ("red") noise**, contributed by **Jamila Taaki**
(`@xiaziyna <https://github.com/xiaziyna>`_). The lightcurve and each
box transit template are transformed with the GPU adjoint NFFT directly
at the observed (irregular, gappy) times over the full baseline, and the
detection statistic is the noise-whitened correlation

.. math::

   S = \frac{\mathrm{Re}\sum_k Y_k T_k^{*} / P(k)}
            {\sqrt{\sum_k |T_k|^2 / P(k)}}

with the noise power spectrum :math:`P(k)` either supplied or estimated
from the data (smoothed periodogram). Whitening by :math:`P(k)` is what
distinguishes it from BLS/TLS, which weight points by their individual
error bars and otherwise assume *white* noise.

Provenance, and exactly what is implemented
===========================================

The method family is published in:

1. **Taaki, Kamalabadi & Kemball (2020), AJ 159, 283**
   (`arXiv:2004.14893 <https://arxiv.org/abs/2004.14893>`_) -- joint
   Bayesian transit detection + systematic-noise characterization on
   Kepler long-cadence data.
2. **Taaki, Kemball & Kamalabadi (2025), AJ 170, 14**
   (`arXiv:2504.18706 <https://arxiv.org/abs/2504.18706>`_) -- the TESS
   2-min application.
3. Kay (1998/2002)-style adaptive detection under unknown noise PSDs is
   the signal-processing foundation.
4. Reference NUFFT prototype: `code_nova_exoghosts
   <https://github.com/star-skelly/code_nova_exoghosts>`_.

``cuvarbase.nufft_lrt`` implements, selectable via
``run(..., detector=...)``:

* ``'matched'`` (default) -- the stationary PSD-whitened matched filter.
* ``'marginal'`` -- **Detector A** of the 2020 paper: the joint detector
  with a Gaussian prior on systematics coefficients marginalized in
  closed form. Computed in the whitened frequency domain via the
  Woodbury identity, so the systematics basis costs one NFFT per basis
  vector per lightcurve and K-dimensional algebra per template (the
  template-independent K x K algebra is computed once per search).
  Supply ``systematics_basis`` (e.g. instrument cotrending vectors, or
  PCA modes of a lightcurve population) and ``coeff_prior_cov``
  (+ optional ``coeff_prior_mean``), estimated from population fits as
  in the paper. With ``estimate_psd=True`` (default) the PSD is
  estimated from the basis-projected residual ``y - V c_ols``, not from
  ``y - V mu``: the latter still contains the realized systematics,
  whose power the spectral window spreads across the whole band and
  which then whitens the transit away (confirmed defect, Sep 2026;
  fixed).
* ``'sequential'`` -- the papers' "standard" baseline: least-squares
  cotrend (with an intercept: basis columns and data are centred, so
  columns need not be zero-mean) against the basis in the time domain,
  then the stationary filter on the residual.

Not implemented (deliberately): **Detector B** (joint MAP plug-in over a
depth grid) -- the 2020 paper found it comparable to Detector A and
describes it as exploratory; the closed-form marginalization supersedes
the plug-in. The papers' phase-correlation epoch pre-estimation trick
(2020, Appendix A) is also not implemented -- epochs are searched on a
grid (automatic or explicit, see *Usage*).

**Honesty note on citing the papers:** the published validations cover
*uniformly sampled* Kepler/TESS data, and the published gains of the
joint detectors are modest (~2% detection efficiency on Kepler; 0.2% and
not statistically significant on TESS). The NUFFT / irregular-sampling
variant in this module appears in no publication -- its characterization
is the cuvarbase injection-recovery study
(``benchmarks/nufft_lrt/validate.py``; see *Validation status*). Do not
cite the papers' numbers as this module's performance.

When is this the right tool?
============================

The evidence is the Sep-2026 injection-recovery re-validation of the
fixed code (*Validation status* below: 200 injections per depth, 200
null light curves per threshold, 600-point ground-based sampling over
90 d, every arm searching the same period grid; one NVIDIA A40):

* **Absolute (BJD-scale) timestamps and the default epoch search are
  correct.** The same light curves at ``t + 2457000.5`` d give the same
  statistic to 5e-8 and the same 800 detection decisions; the
  ``epochs=None`` default finds the injected transit (99 % of its
  detections within half a duration of the true mid-time, median error
  0.01-0.03 d). Its automatic per-cell epoch grid is coarser than the
  explicit grid the harness uses for the longer durations, which costs
  it 4-9 % of completeness at the transition depths (raise
  ``epoch_oversample`` to buy it back, at proportional cost).
* **In white noise BLS and TLS are more complete than the whitened
  filter** at the transition depths (BLS by 10-12 +- 3 % at depths
  0.003-0.004 on the same light curves; TLS similarly). Part of that is
  the template grid (a 3-duration ladder and an epoch step of up to
  0.028 d against BLS's finer q ladder and P/200 phase bins), the rest
  is the statistic itself.
* **In OU red noise the whitened filter is more complete than BLS by
  6-10 +- 3 %** at the transition depths (1x and 3x the white level),
  and about as complete as TLS (+3 to -6 %). But **a flat-PSD matched
  filter does as well (1x: +2-3 +- 3 % for whitening, not significant)
  or better (3x: the flat filter wins by 6 +- 2 %)** -- the estimated
  PSD partly whitens the transit away. The gain over BLS in red noise
  comes from the full-baseline matched-filter form, not from the PSD
  whitening, and whitening does not stabilize the false-alarm threshold
  (null p95 8.6 -> 11.4 -> 14.1 from white to 3x red, as BLS's rises
  0.038 -> 0.124 -> 0.204).
* **With a shared-systematics basis the basis-aware detectors are the
  only thing that works**: Detector A and the sequential cotrend + filter
  recover 3/44/98/100 % of transits at depths 0.004/0.008/0.016/0.032
  where the basis-free whitened filter recovers 0/0/6/34 %, BLS
  0/0/2/16 % and TLS nothing. **Detector A equals the sequential
  baseline exactly** (zero discordant decisions out of 800): after the
  PSD fix it no longer trails it, but it does not beat it either. A
  non-zero-mean basis changes nothing (5.5e-7).
* **Cost**: 3.4-5.7 s per search of 32 periods x 3 durations on the A40
  (~7,500 templates; 0.23 ms per template single-process) against 1.2 ms
  for BLS and 9 ms for TLS.

So, based on that evidence, **reach for NUFFT-LRT when all of these
hold:**

1. **You have a systematics basis** (CBVs, PCA modes of a population)
   and want the cotrend and the search in one statistic -- the one
   regime with a decisive gain over basis-free BLS/TLS. Note that the
   simpler sequential detector delivered the same completeness as
   Detector A.
2. **You are scoring a bounded set of candidates**, not running a blind
   survey: the cost is one adjoint NFFT *per template* (period x
   duration x epoch), so ~10^3-10^5 templates is comfortable and
   survey-scale grids (10^6+) are not. Typical fits: vetting/re-ranking
   BLS or TLS candidates under a realistic noise model, or focused
   searches around known ephemerides. Mind the period step: a box of
   duration :math:`d` drifts by :math:`T\,\delta P / P` over the
   baseline :math:`T` when the trial period is off by :math:`\delta P`,
   so the grid needs :math:`\delta P \lesssim d P / (2T)` or an on-grid
   harmonic alias (:math:`P/2`, :math:`2P`) beats the off-grid true
   period.
3. **You can calibrate thresholds empirically** (see the caveats).

**BLS** is designed for blind box searches over large period grids;
**TLS** uses limb-darkened transit templates. The validation tables
below compare recovery and cost for the specified NUFFT-LRT experiment.
Use the current transit benchmark for BLS/TLS release speed claims.
(Lomb-Scargle is not a transit
competitor at all -- a short-duty-cycle box leaves only a small
fraction of its power in the sinusoidal fundamental, which is why box
searches exist.)

Statistical caveats
===================

* **The statistic is not N(0, 1) and is not an SNR.** Under irregular
  sampling the NFFT modes are not orthogonal, so the frequency-diagonal
  whitened correlation is over-dispersed *even with the true noise
  PSD*: its null standard deviation is 1.8-2.7 for ground-based sampling
  at the default ``nf = 2 * len(t)`` (1.81 measured for the validation
  harness's sampling with the estimated PSD, 5000 draws; about 1.4 for
  uniform sampling) and grows with ``nf`` (28 -> 51 at a fixed resolved
  template for ``nf`` = n -> 8n). This is intrinsic to the statistic (an exact
  float64 DFT reproduces it), not an NFFT accuracy or PSD-estimation
  artefact. **Never apply a textbook SNR >~ 7 threshold; calibrate the
  detection threshold per (sampling, ``nf``, PSD estimator)
  configuration on signal-free or scrambled data**, as the validation
  harness does (null-percentile calibration). Raising ``nf`` inflates
  the raw value without adding information -- pick ``nf`` once and
  calibrate at it.
* **Self-whitening**: with ``estimate_psd=True``, a strong transit
  inflates the PSD estimate at its own harmonic frequencies and
  partially suppresses itself (24-28% of the statistic at threshold in
  the audit's white-noise runs). Provide ``psd=`` from a transit-free
  noise model when you have one.
* **PSD convention** (for ``psd=``): ``psd[k]`` is the expected squared
  modulus of the noise's *unnormalized* adjoint NFFT at mode ``k``,

  .. math::

     P(k) = \mathrm{E}\,\Bigl|\sum_j s_j\, e^{2\pi i f_k t_j}\Bigr|^2,
     \qquad f_k = \frac{k}{\max t - \min t},\quad k = 0 \ldots n_f - 1 .

  White noise of variance :math:`\sigma^2` per point has
  :math:`P(k) = n\sigma^2` at every :math:`k`. ``psd = np.ones(nf)``
  therefore returns a statistic in *data units*. Bins are floored at
  ``eps_floor`` (default 1e-3) times the positive median, for supplied
  and estimated PSDs alike.
* **``dy`` is not used** by any detector (a ``UserWarning`` is emitted
  if it is passed); the noise model is the PSD.
* **Detector A's prior is effectively wider than you specify.** The
  Gram matrix :math:`G_{ij} = \langle v_i, v_j \rangle_W` is accumulated
  over the ``nf`` (default ``2n``) non-orthogonal NFFT modes, which
  overcounts the corresponding time-domain inner products by ~2.2-2.4x
  for the samplings measured in the Sep-2026 audit, so
  ``coeff_prior_cov`` behaves as though it were about that much wider.
  The effect on the statistic is small, but calibrate the prior and the
  detection threshold on the same footing.
* **Frequency resolution**: the default ``nf = 2 * len(t)`` gives a
  maximum template frequency ``nf / T_span``. Resolving a transit of
  duration :math:`d` wants ``nf`` :math:`\gtrsim` a few
  :math:`\times\, T_{\rm span} / d` -- but see the first caveat before
  raising ``nf``.

Input validation
================

``run`` validates its inputs on the host before any GPU work (equal-length
finite ``t``/``y``, at least three observations, positive finite periods
and durations, finite epochs, a finite basis) and raises ``ValueError``
otherwise; see :ref:`Input validation <input-validation>` for the rules
shared with the other methods.

Usage
=====

.. code-block:: python

    import numpy as np
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

    proc = NUFFTLRTAsyncProcess()      # sigma=4: full-band-accurate NFFT

    # Times may be absolute (BJD): floor(min(t)) is subtracted in float64
    # internally; epochs in and out are in YOUR time scale.

    # 1) focused period search with the automatic epoch grid (epochs=None):
    #    per (period, duration) cell, clip(ceil(2 P / duration), 8, 96)
    #    epochs are scanned and the max over epochs is returned together
    #    with the epoch that attains it -> two (nP, nD) arrays. The period
    #    step follows the drift criterion dP <~ dur * P / (2 T).
    durations = np.array([0.12, 0.25])
    T = t.max() - t.min()
    periods = np.arange(5.0, 5.6, durations.min() * 5.0 / (2 * T))
    snr, best_epoch = proc.run(t, y, periods, durations=durations)
    i, j = np.unravel_index(np.argmax(snr), snr.shape)
    print(periods[i], durations[j], best_epoch[i, j])
    # cost: ~2P/duration transforms per cell (max_epochs=96 caps it; raise
    # it for long periods, where P/96 exceeds the duration)

    # 2) explicit epochs -> one (nP, nD, nE) array, no reduction
    snr = proc.run(t, y, np.array([P]), durations=np.array([d]),
                   epochs=np.linspace(0, P, 40, endpoint=False))

    # 3) Detector A (joint marginalized) with a systematics basis V (n, K)
    #    and a coefficient prior estimated from population fits
    snr, best_epoch = proc.run(t, y, periods, durations=durations,
                               detector='marginal', systematics_basis=V,
                               coeff_prior_mean=mu_c, coeff_prior_cov=cov_c)

    # 4) known noise PSD (recommended when available; convention above)
    snr, best_epoch = proc.run(t, y, periods, durations=durations,
                               estimate_psd=False, psd=my_psd, nf=len(my_psd))

Threshold calibration sketch (do this for your dataset, at the ``nf``,
sampling and PSD estimator you will search with):

.. code-block:: python

    null_maxima = []
    for y_null in signal_free_or_scrambled_lightcurves:
        snr, _ = proc.run(t, y_null, periods, durations=durations)
        null_maxima.append(snr.max())
    threshold = np.percentile(null_maxima, 95)   # 5% per-search FAR

A runnable example with absolute timestamps and a transit injected at a
random epoch is ``examples/nufft_lrt_example.py``.

Sep-2026 correctness fixes (all result-changing)
================================================

1. **BJD-scale times**: ``run()`` and ``compute_nufft`` cast times to
   float32 before folding/gridding; absolute BJD input returned a
   different statistic (corr ~0.5, wrong argmax). Times are now
   epoch-subtracted in float64 first.
2. **``epochs=None``** evaluated a single phase-0 template per cell
   (0/12 random-epoch transits recovered) while being documented as a
   period search. It is now an automatic epoch grid with a max
   reduction (see *Usage*) and returns ``(snr, best_epoch)``; the
   shipped example and the old README used to show that non-search as a
   detection.
3. **``detector='sequential'``** fitted the basis without an intercept:
   a 1% column mean on relative flux dropped the statistic at the true
   period from ~25 to ~5. The fit is now centred.
4. **``detector='marginal'``** estimated the PSD from ``y - V mu`` (see
   above): SNR at the true template 2.3 vs 8.9 for the sequential
   baseline; now from the basis-projected residual (8.9 vs 8.9).
5. **NFFT upper half band**: the default ``sigma = 2`` left modes
   ``k >= nf/2`` aliased at O(1) (in double precision too, and
   non-deterministic in float32); ``sigma = 4`` (the library's NFFT
   default) makes every returned mode accurate (~4e-4 relative in
   float32, ~1e-6 in float64 vs the exact adjoint DFT).
6. Also: one NFFT buffer set per ``run()`` instead of one per template
   (the per-template allocation was ~90% of the campaign's GPU time),
   the NFFT reuse path is zeroed and synchronized, user PSDs are floored
   and length-checked, singular coefficient priors give the correct
   pinned-to-mean limit (a zero variance used to become a *flat* prior)
   and non-PSD priors raise.

Validation status
=================

**Re-validated after the Sep-2026 fixes** (Phase 4 of the 1.0 release
plan; campaign JSON, per-process logs and the full-suite log under
``benchmarks/results/nufft_lrt_validation_2026-09-06/``; harness
``benchmarks/nufft_lrt/validate.py`` at commit 2f9736a; tables rendered
by ``benchmarks/nufft_lrt/summarize.py --rst``). Measured on one
NVIDIA A40 (CUDA 12.4); the numbers are completeness fractions and
per-search costs, not absolute timings for any other GPU.

Protocol
--------

600-point ground-based sampling over 90 d (nightly windows with
per-night jitter, 35 % weather loss); a shared grid of 32 log-spaced
trial periods in 2-18 d with the injected 5.3 d period *and its 2P
alias* placed on the grid; box transits of duration 0.22 d at random
epochs; formal errors sigma_white = 3e-3. Per configuration and arm, the
detection threshold is the 95th percentile of the search maximum over
200 signal-free light curves (a 5 % per-search false-alarm rate), then
200 injections per depth; a detection is a statistic above that
threshold with the best period within 1 % of P, 2P or P/2. Noise:
white; white + Ornstein-Uhlenbeck red (tau = 0.8 d) at 1x and 3x the
white level; white + 1x red + three shared systematics modes (6/3/6
sigma_white) searched with a PCA basis and coefficient prior estimated
from a 60-light-curve population, as in Taaki et al. (2020).

Arms: ``lrt`` = the whitened matched filter over an explicit epoch grid
(2 P / 0.12 d epochs per period, clipped to 8..96); ``lrt_auto`` = **the
public default path**, one ``run(t, y, periods, durations=...)`` call
with ``epochs=None`` and every other argument at its default;
``lrt_flat`` = PSD set to ones (no whitening); ``lrt_marg`` = Detector
A; ``lrt_seq`` = least-squares cotrend then the filter; ``bls`` =
``eebls_gpu_fast`` (q in 0.005..0.08); ``tls`` = ``tls_search_batch``
scored by its un-normalized delta-chi-squared statistic (an SDE over a
32-point spectrum is bounded by sqrt(31) and would saturate). The LRT
arms search durations {0.12, 0.21, 0.30} d; against the 0.22 d box the
nearest template recovers 97.7 % of the matched statistic when centred.
The explicit arm uses round(2 P / 0.12 d) epochs for every duration
(88 at P = 5.3 d, up to 0.030 d of misalignment); the default path's
own grid, ceil(2 P / duration), gives 89/51/36 epochs for the three
durations (up to 0.030/0.052/0.074 d), which is where its 4-9 %
deficit against the explicit arm comes from. BLS's q ladder happens to
sit closer to the injected duration (0.2385 d, P/200 phase bins), so
the comparators are slightly *better* matched to the injection than
the LRT grid is.

Two configurations are *paired* with an existing one on identical light
curves (same random draws): ``white_bjd`` is the white-noise data on
absolute timestamps, ``t + 2457000.5`` d (2457000 d on top of the 0.5 d
every configuration carries, so each method's ``floor(min t)``-anchored
grid keeps its phase and any difference is a time-scale defect, not
grid alignment); ``red_sys_nzm`` is the systematics data searched with
the same PCA basis plus constant column offsets (0.12-0.49 of a
column's rms) and the unchanged prior.

Resolution: a completeness cell carries the binomial error of 200
injections (0.035 at p = 0.5) and the sampling error of its arm's
threshold from 200 null maxima (a common shift for all injections of
that arm); the tables quote both in quadrature, from a bootstrap of the
null set and a Wilson interval. Arm-vs-arm differences within a
configuration are paired on the same light curves (McNemar).

Results
-------

Completeness per depth (fraction of the flux) with its 1-sigma
uncertainty; ``ms/search`` is the per-search cost on the A40 under the
campaign's 8-process split (single-process LRT costs are ~2.5x lower).
The paired rows give A minus B on the same light curves.

**White noise**

.. list-table::
   :header-rows: 1

   * - arm
     - null p95
     - depth 0.002
     - depth 0.003
     - depth 0.004
     - depth 0.008
     - ms/search
   * - LRT (explicit epoch grid)
     - 8.588
     - 13 +- 3%
     - 47 +- 5%
     - 82 +- 3%
     - 99 +- 1%
     - 5737
   * - LRT, default path (epochs=None)
     - 8.719
     - 10 +- 3%
     - 42 +- 5%
     - 74 +- 4%
     - 99 +- 1%
     - 4491
   * - BLS (eebls_gpu_fast)
     - 0.038
     - 13 +- 3%
     - 60 +- 4%
     - 91 +- 2%
     - 100 +- 0%
     - 1.55
   * - TLS (tls_search_batch, delta-chi2)
     - 4.662
     - 16 +- 3%
     - 65 +- 4%
     - 90 +- 2%
     - 100 +- 1%
     - 11.4

Paired differences A - B on the same light curves:

.. list-table::
   :header-rows: 1

   * - A - B
     - depth 0.002
     - depth 0.003
     - depth 0.004
     - depth 0.008
   * - lrt - bls
     - +0 +- 2%
     - -12 +- 3%
     - -10 +- 2%
     - -1 +- 1%
   * - lrt_auto - lrt
     - -4 +- 1%
     - -5 +- 3%
     - -8 +- 2%
     - +0 +- 1%
   * - lrt - tls
     - -4 +- 2%
     - -18 +- 3%
     - -9 +- 3%
     - -0 +- 1%

**Red noise, sigma_red = sigma_white**

.. list-table::
   :header-rows: 1

   * - arm
     - null p95
     - depth 0.004
     - depth 0.006
     - depth 0.008
     - depth 0.016
     - ms/search
   * - LRT (explicit epoch grid)
     - 11.364
     - 4 +- 2%
     - 25 +- 5%
     - 56 +- 5%
     - 100 +- 1%
     - 4383
   * - LRT, default path (epochs=None)
     - 11.186
     - 4 +- 2%
     - 24 +- 4%
     - 52 +- 4%
     - 99 +- 1%
     - 3412
   * - LRT, flat PSD
     - 1.779
     - 4 +- 2%
     - 22 +- 4%
     - 54 +- 5%
     - 100 +- 1%
     - 4377
   * - BLS (eebls_gpu_fast)
     - 0.124
     - 2 +- 1%
     - 17 +- 3%
     - 47 +- 5%
     - 100 +- 0%
     - 1.22
   * - TLS (tls_search_batch, delta-chi2)
     - 12.283
     - 4 +- 1%
     - 22 +- 3%
     - 62 +- 4%
     - 100 +- 0%
     - 8.56

Paired differences A - B on the same light curves:

.. list-table::
   :header-rows: 1

   * - A - B
     - depth 0.004
     - depth 0.006
     - depth 0.008
     - depth 0.016
   * - lrt - bls
     - +2 +- 2%
     - +8 +- 3%
     - +10 +- 3%
     - -0 +- 0%
   * - lrt_auto - lrt
     - +0 +- 1%
     - -1 +- 2%
     - -4 +- 2%
     - -0 +- 0%
   * - lrt - lrt_flat
     - -0 +- 1%
     - +2 +- 2%
     - +3 +- 3%
     - +0 +- 0%
   * - lrt - tls
     - -0 +- 2%
     - +3 +- 3%
     - -5 +- 3%
     - -0 +- 0%

**Red noise, sigma_red = 3 sigma_white**

.. list-table::
   :header-rows: 1

   * - arm
     - null p95
     - depth 0.008
     - depth 0.016
     - depth 0.024
     - depth 0.032
     - ms/search
   * - LRT (explicit epoch grid)
     - 14.121
     - 0 +- 0%
     - 12 +- 4%
     - 57 +- 6%
     - 89 +- 3%
     - 4383
   * - LRT, default path (epochs=None)
     - 13.972
     - 0 +- 0%
     - 10 +- 3%
     - 48 +- 5%
     - 83 +- 4%
     - 3413
   * - LRT, flat PSD
     - 5.097
     - 0 +- 1%
     - 18 +- 5%
     - 63 +- 6%
     - 90 +- 3%
     - 4376
   * - BLS (eebls_gpu_fast)
     - 0.204
     - 0 +- 1%
     - 7 +- 2%
     - 48 +- 4%
     - 88 +- 2%
     - 1.2
   * - TLS (tls_search_batch, delta-chi2)
     - 34.507
     - 0 +- 1%
     - 17 +- 3%
     - 62 +- 4%
     - 93 +- 2%
     - 8.55

Paired differences A - B on the same light curves:

.. list-table::
   :header-rows: 1

   * - A - B
     - depth 0.008
     - depth 0.016
     - depth 0.024
     - depth 0.032
   * - lrt - bls
     - -0 +- 0%
     - +6 +- 2%
     - +10 +- 3%
     - +1 +- 2%
   * - lrt_auto - lrt
     - +0 +- 0%
     - -2 +- 1%
     - -9 +- 2%
     - -6 +- 2%
   * - lrt - lrt_flat
     - -0 +- 0%
     - -6 +- 2%
     - -6 +- 3%
     - -0 +- 1%
   * - lrt - tls
     - -0 +- 0%
     - -4 +- 2%
     - -6 +- 3%
     - -4 +- 1%

**Red noise + shared systematics (PCA basis + population prior)**

.. list-table::
   :header-rows: 1

   * - arm
     - null p95
     - depth 0.004
     - depth 0.008
     - depth 0.016
     - depth 0.032
     - ms/search
   * - LRT (explicit epoch grid)
     - 11.965
     - 0 +- 0%
     - 0 +- 0%
     - 6 +- 2%
     - 34 +- 4%
     - 5735
   * - LRT, default path (epochs=None)
     - 11.616
     - 0 +- 0%
     - 0 +- 0%
     - 5 +- 2%
     - 34 +- 4%
     - 4482
   * - LRT Detector A (marginal)
     - 12.104
     - 3 +- 1%
     - 44 +- 5%
     - 98 +- 1%
     - 100 +- 0%
     - 5524
   * - LRT sequential cotrend
     - 12.220
     - 3 +- 2%
     - 43 +- 5%
     - 98 +- 1%
     - 100 +- 0%
     - 5412
   * - BLS (eebls_gpu_fast)
     - 0.188
     - 0 +- 0%
     - 0 +- 0%
     - 2 +- 1%
     - 16 +- 3%
     - 1.51
   * - TLS (tls_search_batch, delta-chi2)
     - 114.971
     - 0 +- 0%
     - 0 +- 0%
     - 0 +- 0%
     - 0 +- 0%
     - 11.4

Paired differences A - B on the same light curves:

.. list-table::
   :header-rows: 1

   * - A - B
     - depth 0.004
     - depth 0.008
     - depth 0.016
     - depth 0.032
   * - lrt - bls
     - +0 +- 0%
     - +0 +- 0%
     - +3 +- 1%
     - +17 +- 3%
   * - lrt_auto - lrt
     - +0 +- 0%
     - +0 +- 0%
     - -0 +- 0%
     - +0 +- 2%
   * - lrt_marg - lrt_seq
     - +0 +- 0%
     - +0 +- 0%
     - +0 +- 0%
     - +0 +- 0%
   * - lrt_seq - bls
     - +3 +- 1%
     - +43 +- 5%
     - +95 +- 7%
     - +84 +- 6%
   * - lrt - tls
     - +0 +- 0%
     - +0 +- 0%
     - +6 +- 2%
     - +34 +- 4%

**white vs white_bjd** (same light curves)

.. list-table::
   :header-rows: 1

   * - arm
     - searches
     - max rel. diff of the statistic
     - best period differs
     - detection differs
   * - LRT (explicit epoch grid)
     - 1000
     - 5.2e-08
     - 3 / 800
     - 0 / 800
   * - LRT, default path (epochs=None)
     - 1000
     - 3.5e-08
     - 2 / 800
     - 0 / 800
   * - BLS (eebls_gpu_fast)
     - 1000
     - 1.3e-06
     - 0 / 800
     - 0 / 800
   * - TLS (tls_search_batch, delta-chi2)
     - 1000
     - 1.4e-07
     - 0 / 800
     - 0 / 800

**red_sys vs red_sys_nzm** (same light curves)

.. list-table::
   :header-rows: 1

   * - arm
     - searches
     - max rel. diff of the statistic
     - best period differs
     - detection differs
   * - LRT Detector A (marginal)
     - 1000
     - 5.5e-07
     - 3 / 800
     - 0 / 800
   * - LRT sequential cotrend
     - 1000
     - 8.4e-08
     - 2 / 800
     - 0 / 800

Epoch recovery (arms that return a best epoch): per configuration,
99-100 % of the explicit-grid and default-path detections lie within
half a duration of the injected mid-time (the smallest per-depth cell
is 94 %, 45 of 48), with median errors of 0.01-0.03 d -- the grid
resolution.

Null calibration of the single-template statistic on white noise: the
campaign's 200 draws give mean 0.348, std 1.579; 5000 draws from the
same seed give mean 0.030 +- 0.026, std 1.808 (the statistic is exactly
odd in the data, so its null mean is zero by construction; 1.81 is the
calibration constant of this sampling, the same value the pre-fix
campaign measured, and independent of ``sigma``).

What the numbers say
--------------------

* The two configurations that exercise the Sep-2026 fixes on the public
  default path pass exactly: BJD-scale times reproduce the relative-time
  results to float32 rounding for every method, and the non-zero-mean
  basis reproduces the zero-mean results for both basis-aware detectors.
  ``epochs=None`` is a working epoch search.
* What a *default* call delivers, against BLS on the same light curves:
  -17 +- 3 % at depths 0.003 and 0.004 in white noise, +7 +- 3 % and
  +5 +- 4 % at 1x red (depths 0.006/0.008), +4/+1/-5 +- 2-3 % at 3x red
  (0.016/0.024/0.032), i.e. indistinguishable from BLS there; against
  TLS -23 +- 4 % (white, 0.003) and -7/-15/-10 % at 3x red. The
  explicit-grid numbers in the bullets above are the method's; these
  are the default's.
* The systematics-basis gain is a gain over *basis-free* BLS and TLS:
  no cotrend-then-BLS/TLS comparator was run, so the campaign does not
  show that the LRT detectors beat cotrending first and searching with
  BLS or TLS afterwards.
* Against the explicit epoch grid, the default path loses 4-9 % of
  completeness at the transition depths (paired, 2-4 sigma) because its
  per-cell grid, ``ceil(2 P / duration)`` epochs, is coarser for the
  longer durations (51 and 36 epochs at P = 5.3 d for 0.21 and 0.30 d,
  against 89); it is a resolution setting, not a defect.
* White noise: BLS and TLS beat the whitened filter by 10-12 +- 3 % at
  depths 0.003-0.004. Red noise: the whitened filter beats BLS by 6-10
  +- 3 % at the transition depths and is within +3/-6 % of TLS; the
  flat-PSD filter is as good (1x) or better (3x, +6 +- 2 %). Shared
  systematics: only the basis-aware detectors work (98 % vs <= 6 % at
  depth 0.016), and Detector A equals the sequential baseline exactly.
* Compared with the pre-fix campaign (60 injections, explicit epochs
  only, ``sigma = 2``, the `archived pre-fix campaign <https://github.com/johnh2o2/cuvarbase/tree/f0dc98136ae34b34465b152be1af84faf063eb44/analysis/audit-sep2026/campaign>`_): the
  qualitative picture in white and red noise is unchanged (no
  whitening gain over a flat PSD; thresholds rise with red noise), the
  Detector A row now measures the detector instead of the PSD defect,
  and the default path and BJD-scale times are measured for the first
  time.

Citation
========

If you use this module, please cite Taaki, Kamalabadi & Kemball (2020,
AJ 159, 283) for the method, Taaki, Kemball & Kamalabadi (2025, AJ 170,
14) for the space-photometry application, the reference prototype
(``code_nova_exoghosts``), and cuvarbase itself (see the landing page).

API reference
=============

The canonical API entry is on the :doc:`API page <cuvarbase>`; it is
repeated here for convenience.

.. automodule:: cuvarbase.nufft_lrt
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:
