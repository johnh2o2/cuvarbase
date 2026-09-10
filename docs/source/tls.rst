Transit Least Squares (TLS)
===========================

The standard ``cuvarbase.tls`` search evaluates individual observations using
the numerical search implemented by public GTLS. It retains the transit-shaped
template, broad duration domain, sample-window trials, depth estimation,
spectrum ranking, and full candidate/harmonic refinement. It does not phase-bin
observations, and a GPU memory limit does not reduce the searched durations.

The implementation removes repeated calculation and large intermediate
residual arrays. See the `current benchmark and numerical validation
<https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TRANSIT_BENCHMARKS.md>`_
for the measured speed and the tested regimes.

Installation
------------

For CUDA 12, install the v1 candidate with the TLS extra. PyPI 0.2.5 does not
contain TLS; these measurements use the ``v1.0-fixes`` branch:

.. code-block:: bash

    pip install 'cuvarbase[tls] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'

It supplies CuPy 13 and ``batman-package``; this TLS extra supports Python
3.9–3.13. The current GPU validation uses Python 3.11 and CuPy 13.6. For another CUDA runtime, install
its matching CuPy wheel and ``batman-package`` separately. Install only one
CuPy distribution in an environment; see the `CuPy installation guide
<https://docs.cupy.dev/en/v13.6.0/install.html>`_. PyCUDA and CuPy use the same device's
primary context; select the device with ``CUDA_DEVICE`` before the first call.
The default TLS engine requires batman and does not silently substitute a
box-shaped or analytic template when that dependency is missing.

Single light curves
-------------------

.. code-block:: python

    from cuvarbase.tls import tls_search

    result = tls_search(t, flux, flux_error,
                        R_star=1.0, M_star=1.0,
                        period_min=0.5, period_max=15.0)
    print(result['period'], result['SDE'], result['T0'])

Times and periods are in days. Input arrays must be finite, aligned and contain
at least three observations with positive uncertainties and a positive time
span. Flux must be positive and normalized to an out-of-transit baseline of one;
the search does not fit a free baseline. Normalize each passband before combining
multiband observations. There is no separate per-band depth or baseline model.

A float64 time-origin shift preserves relative, negative and absolute BJD times
without dropping observations. ``T0`` is restored to the input time system and
is the first mid-transit at or after ``min(t)``. ``t0_phase`` is its fold phase
relative to ``floor(min(t))``.

The automatic Ofir period grid uses ``R_star`` and ``M_star`` in solar units,
``n_transits_min=2`` and ``oversampling_factor=3``. An explicit ``periods`` array
retains float64 precision; returned arrays retain its original order.
``tls_search_gpu`` and ``tls_transit`` use the same default policy.

Small requested grids remain small: unlike pinned GTLS, cuvarbase does not
silently replace a grid of fewer than 100 periods with default solar-host
bounds. Automatic grids require ``0.01 <= R_star <= 10000`` and
``0.01 <= M_star <= 1000``; out-of-range values raise ``ValueError`` rather
than being clamped. Explicit periods accept other finite positive stellar
values. These input-policy differences do not change a search supplied with
the same valid period array.

Thin transits and the automatic duration domain
-----------------------------------------------

Omitting duration controls selects the broad native GTLS duration grid. There
is no phase-bin cap and no default half-central-duration cutoff. Narrow
transits therefore do not require a separate accuracy preset.

The numerical search remains a GTLS-style sample-window/template search. It is
not an exposure-integrated physical fit to arbitrary irregular sampling. The
usual limits shared with GTLS remain: a transit must be sampled, lie inside the
period/template domain, and have sufficient signal relative to noise. The
validation compares implementations on the same data and search domain; it
cannot promise that either algorithm detects every physical transit.

Optional controls change the scientific search:

* ``duration_grid_step=1.1`` sets the native duration-grid spacing.
* Scalar or aligned ``qmin`` and ``qmax`` explicitly replace the duration domain.
  They mean nominal duration/period. Their bounds are honored per period,
  including during refinement, independently of GPU workspace chunks.
* ``duration_window='keplerian'`` explicitly selects the older stellar-duration
  prior, with default factors 0.5 and 2.0. Use it only when that prior is intended.
* ``n_durations`` with an explicit window sets a minimum geometric density;
  additional admissible integer sample widths can be shared across periods.
* ``u``, ``limb_dark`` and ``transit_template`` configure the native template.
  Template choices are ``'default'``, ``'grazing'`` and ``'box'``.

Surveys
-------

.. code-block:: python

    from cuvarbase.tls import tls_search_batch

    results = tls_search_batch(lightcurves,
                               period_min=0.5, period_max=15.0)

Each light curve is a ``(t, flux, flux_error)`` tuple. The standard batch wrapper
processes curves sequentially while parallelizing each period search on the
GPU and reusing bounded scan plans. The longest input baseline determines the
shared automatic grid. Pass ``periods`` to fix it explicitly.
``return_arrays=True`` includes every period spectrum. ``work_chunk=256`` sets
an upper limit on periods in a physical workspace; it changes allocation and
runtime while retaining the duration and epoch trial policies. The native
floating-point scans do not guarantee bitwise repeatability on every input;
the `numerical accuracy guide
<https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TLS_NUMERICS.md>`_
records that shared limit.

The estimated workspace budget is the smaller of 512 MiB and one quarter
of free device memory. If one period's full preparation exceeds that budget,
the call raises ``MemoryError``; reducing ``work_chunk`` cannot solve that
single-period case. Memory limits never silently narrow the duration search.

Results and significance
------------------------

Results include ``period``, ``period_uncertainty``, ``T0``, ``t0_phase``,
``duration``, ``depth``, ``chi2_min``, ``SDE``, ``SDE_raw``, ``SNR`` and
``n_transits``. ``search_configuration`` records the numerical policy.

SDE follows the native GTLS arithmetic and full refinement policy. Invalid
masked/nonfinite candidates are excluded before ranking; this corrects a native
host-mask defect without changing the template or its resolution. The coarse
scan uses its epoch stride; the top-ranked candidates and harmonics are searched
at every sample start and their residuals enter the final detection spectrum.
``full=False`` explicitly selects GTLS's fast-mode detection policy. cuvarbase
additionally fits the selected winner to supply its parameter-result contract;
public GTLS fast mode returns only the coarse periodogram.

The default SDE median window derives from ``30 * oversampling_factor`` and
must have an integer width. For other fractional factors, supply an explicit
positive integer ``sde_kernel_size``; even widths are increased by one. Such
an override changes the detection statistic, so use the same setting for
observed and null searches.

SNR retains cuvarbase's definition, the square root of the nonnegative
improvement in chi-squared over the fixed baseline, in the supplied uncertainty
units. It is different from GTLS's historically reported SNR formula. Compare
detection decisions using the full period spectrum and a common detection rule,
not equality of differently defined scalar SNR fields.

SDE summarizes the full detection spectrum; ``period``, ``T0`` and ``SNR``
describe the selected full-stage fit. Native harmonic selection can make
these correspond to different peaks.

Per-period duration, depth and epoch arrays describe nominal sample-window
search diagnostics. On an irregular cadence, the native final duration/T0
estimator can differ from these nominal values. ``parameter_valid_periods``
distinguishes fitted windows from skipped/gated numerical sentinels;
``n_masked_periods`` counts periods excluded by the spectrum mask.
A degenerate spectrum returns SDE=SNR=0, NaN fitted parameters and an ``error``.
If only the final physical-duration estimator fails, the selected period and
spectrum remain available with ``parameter_error`` and NaN duration/T0.

SDE is not a universal false-alarm probability. An optional white-noise
permutation calibration runs the same complete search on every null:

.. code-block:: python

    results = tls_search_batch(lightcurves, periods=periods,
                               fap_null_draws=200, fap_seed=1)
    results[0]['FAP']
    results[0]['SDE_null']

The add-one estimator is ``(1 + n_exceed) / (1 + fap_null_draws)``. Permuting the
flux/error pairs preserves sampling and the marginal error distribution but
destroys correlated noise. This is not a systematics model. FAP controls are
available only on the batch wrapper.

Older engines
-------------

``method='binned'`` explicitly selects the earlier phase-binned batch engine.
It accepts ``nbins``, ``block_size``, ``t0_oversample``, ``n_durations``,
``refine_top_k`` and ``refine_oversample``. Its coarse statistic approximates
the observation-level search; its sensitivity limits and historical large
speed ratios apply only to that engine. See the `binned-engine audit
<https://github.com/johnh2o2/cuvarbase/tree/v1.0-fixes/benchmarks/results/tls_accuracy_2026-09-09>`_.

``method='legacy'`` on the single-curve wrapper selects the old shared-memory
per-observation kernel, including its light-curve-length limit and low-level
memory/stream controls. The deprecated ``use_fast`` keyword selects these older
engines: True means binned and False means legacy.

References
----------

* Hippke & Heller (2019), A&A 623, A39, `Transit Least Squares
  <https://doi.org/10.1051/0004-6361/201834672>`_.
* Ofir (2014), A&A 561, A138, `Optimizing the search for transiting planets
  <https://doi.org/10.1051/0004-6361/201220860>`_.
* `Pinned GTLS implementation
  <https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111>`_.
