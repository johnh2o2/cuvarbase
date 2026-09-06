NUFFT-LRT: whitened matched-filter transit detection (experimental)
*******************************************************************

.. warning::

   **EXPERIMENTAL.** ``cuvarbase.nufft_lrt`` is importable only by name
   (it is deliberately *not* exported from the top-level ``cuvarbase``
   namespace) and emits an ``EXPERIMENTAL`` ``UserWarning`` when
   :class:`~cuvarbase.nufft_lrt.NUFFTLRTAsyncProcess` is first
   constructed. The statistic's algebra has CPU and GPU unit tests, but
   the module's injection-recovery re-validation after the September
   2026 correctness fixes (below) is still pending, the method has far
   less operational mileage than cuvarbase's BLS and TLS, and its
   thresholds must be calibrated empirically per dataset (see
   *Statistical caveats*). It is **outside the 1.x API-stability
   promise** and may change incompatibly in a 1.x release. Do not use it
   for publishable science yet.

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
(``scripts/nufft_lrt_validation.py``; see *Validation status*). Do not
cite the papers' numbers as this module's performance.

When is this the right tool?
============================

What the Sep-2026 injection-recovery campaign (run *before* the fixes
below, with an explicit epoch grid, epoch-relative times and a zero-mean
basis, so it exercised none of the defects except the Detector A one)
showed, at 60 injections per depth on 600-point ground-based sampling
over 90 d:

* The whitened NUFFT matched filter **matched BLS's completeness** in
  white noise and in OU red noise at 1x and 3x the white level
  (differences <= 0.08) and showed **no measurable gain over a flat-PSD
  matched filter**; PSD whitening does not stabilize the false-alarm
  threshold (null p95 8.4 -> 12.4 with red noise, as for BLS).
* With a **shared-systematics basis** the sequential cotrend + matched
  filter recovered 0.57/0.95/1.00 of transits at depths
  0.008/0.016/0.032 where BLS and TLS without a basis recovered
  0.00/0.05/0.15 and 0.00/0.00/0.02. **Detector A results are pending
  re-measurement** after the PSD fix (the campaign's Detector A arm
  measured the PSD defect, not the detector; with the fix it matches --
  but does not beat -- the sequential baseline in the verifier's runs).

So, based on the evidence in hand, **reach for NUFFT-LRT when all of
these hold:**

1. **You have a systematics basis** (CBVs, PCA modes of a population)
   and want the cotrend and the search in one statistic -- this is where
   the campaign showed a gain over basis-free BLS/TLS, and it comes from
   the basis, not from the whitening.
2. **You are scoring a bounded set of candidates**, not running a blind
   survey: the cost is one adjoint NFFT *per template* (period x
   duration x epoch; 0.2-0.4 ms each on an A40 after the per-run buffer
   reuse), so ~10^3-10^5 templates is comfortable and survey-scale grids
   (10^6+) are not. Typical fits: vetting/re-ranking BLS or TLS
   candidates under a realistic noise model, or focused searches around
   known ephemerides. Mind the period step: a box of duration :math:`d`
   drifts by :math:`T\,\delta P / P` over the baseline :math:`T` when the
   trial period is off by :math:`\delta P`, so the grid needs
   :math:`\delta P \lesssim d P / (2T)` or an on-grid harmonic alias
   (:math:`P/2`, :math:`2P`) beats the off-grid true period.
3. **You can calibrate thresholds empirically** (see the caveats).

**Prefer BLS** for blind box searches at scale (it is thousands of times
cheaper per trial, its white-noise statistic is well understood, and in
white or OU red noise it was as complete as this filter), **TLS** when
limb-darkened template fidelity matters for small planets.
(Lomb-Scargle is not a transit competitor at all -- a short-duty-cycle
box leaves only a small fraction of its power in the sinusoidal
fundamental, which is why box searches exist.)

Statistical caveats
===================

* **The statistic is not N(0, 1) and is not an SNR.** Under irregular
  sampling the NFFT modes are not orthogonal, so the frequency-diagonal
  whitened correlation is over-dispersed *even with the true noise
  PSD*: its null standard deviation is 1.8-2.7 for ground-based sampling
  at the default ``nf = 2 * len(t)`` (about 1.4 for uniform sampling)
  and grows with ``nf`` (28 -> 51 at a fixed resolved template for
  ``nf`` = n -> 8n). This is intrinsic to the statistic (an exact
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

**Pending.** Re-validation of the fixed code -- all four noise
configurations (white; OU red at 1x and 3x the white level; red noise
plus shared systematics) and all arms, plus a BJD-offset configuration,
an ``epochs=None`` arm and a non-zero-mean basis, at >= 200 injections
per depth -- is scheduled as Phase 4 of the 1.0 release plan and has not
run yet. When it has, the completeness tables rendered by
``scripts/summarize_lrt_validation.py`` from the campaign JSON replace
this paragraph; until then the only measured evidence is the pre-fix
campaign summarized in *When is this the right tool?* (its JSON is
archived under ``analysis/audit-sep2026/campaign/``). Full protocol:
``scripts/nufft_lrt_validation.py``; the audit that motivated the fixes:
``analysis/audit-sep2026/ALGORITHM_AUDIT.md`` (section 6).

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
