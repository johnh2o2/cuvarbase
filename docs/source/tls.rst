Transit Least Squares (TLS)
===========================

Transit Least Squares [HH2019]_ searches for periodic transits with a
physically-motivated, limb-darkened transit template instead of the box
of :doc:`BLS <bls>`. The template matters most for small planets: the
smooth ingress/egress of a real transit is a measurably better match to
the data than a box, which translates into a higher detection
significance at fixed depth.

``cuvarbase.tls`` implements a GPU TLS with two execution paths:

* **The fast path (default)** — a batch-native, phase-binned kernel:
  each (lightcurve, period) pair is one CUDA block that folds the
  lightcurve once into shared-memory phase bins and evaluates every
  (duration, epoch) trial against precomputed integrated-template
  tables, with a closed-form :math:`\chi^2`. A second kernel then
  re-fits the best ``refine_top_k`` candidate periods per lightcurve
  *exactly* (per-point template evaluation) on a finer local grid.
  There is **no cap on the number of points per lightcurve**, absolute
  BJD-scale timestamps are safe (the epoch is subtracted in float64
  internally), and whole surveys can be searched in one call.
* **The legacy path** (``use_fast=False``) — the original per-point
  kernel. It caps lightcurves at ~3,500 points (48 KB shared-memory
  budget) and folds float32 times directly, so it should not be used
  with raw BJD timestamps. It remains available as a reference
  implementation and for the low-level plumbing (custom streams,
  pre-compiled kernels, externally-managed memory) that the batch
  engine does not expose.

Accuracy is validated two ways in the test suite: golden tests against
the reference `transitleastsquares
<https://github.com/hippke/tls>`_ package, and injected-transit
recovery tests across cadence regimes. On the identical SDE statistic,
the default configuration recovers the reference package's detection
significance to within a few percent at a small fraction of the cost;
see ``docs/BENCHMARK_RESULTS.md`` for measured numbers.

Input conventions
-----------------

* ``t``: observation times in days. BJD-scale absolute times are safe
  on the default fast path.
* ``y``: fluxes **normalized so the out-of-transit baseline is ~1.0**.
  The transit model is :math:`1 - \delta\,T(x)`; no TLS path rescales
  the input, so unnormalized fluxes (e.g. raw counts) produce
  meaningless depths.
* ``dy``: per-point flux uncertainties (same units as ``y``).

Searching a single lightcurve
-----------------------------

.. code-block:: python

    import numpy as np
    from cuvarbase.tls import tls_search_gpu

    # t (days), y (normalized flux), dy (uncertainties)
    results = tls_search_gpu(t, y, dy)

    print(results['period'])    # best-fit period (days)
    print(results['T0'])        # transit epoch (phase in [0, 1))
    print(results['duration'])  # transit duration (days)
    print(results['depth'])     # fractional transit depth
    print(results['SDE'])       # signal detection efficiency

The trial period grid is generated automatically following [Ofir2014]_
(pass ``period_min``/``period_max`` to bound it, or ``periods`` for an
explicit grid). Keplerian per-period duration windows are used when
``qmin``/``qmax`` arrays are supplied — :func:`cuvarbase.tls.tls_transit`
wraps this, deriving the windows from stellar parameters:

.. code-block:: python

    from cuvarbase.tls import tls_transit

    results = tls_transit(t, y, dy, R_star=1.0, M_star=1.0)

Searching many lightcurves (surveys)
------------------------------------

:func:`cuvarbase.tls.tls_search_batch` is the survey entry point: all
lightcurves share one trial-period grid and are searched together with
a small number of kernel launches, which is what the fast path is
optimized for.

.. code-block:: python

    from cuvarbase.tls import tls_search_batch

    lightcurves = [(t1, y1, dy1), (t2, y2, dy2), ...]
    results = tls_search_batch(lightcurves,
                               period_min=0.5, period_max=15.0)

    for r in results:
        print(r['period'], r['SDE'], r['T0'])

Each result dict carries the best-fit parameters (``period``,
``period_uncertainty``, ``T0`` — the absolute mid-transit time near the
lightcurve's epoch — ``duration``, ``depth``, ``chi2_min``) and the
detection statistics (``SDE``, ``SDE_raw``, ``SNR``, ``FAP``,
``n_transits``). Pass ``return_arrays=True`` to also get the per-period
:math:`\chi^2` spectrum and derived quantities.

Detection statistics and refinement
-----------------------------------

The per-period spectrum that feeds the SDE and FAP statistics comes
from the *coarse* phase-binned scan at uniform fidelity. The exact
refinement pass only sharpens the reported best-fit parameters (period
choice among the top candidates, ``T0``, ``duration``, ``depth``,
``chi2_min``) — refined :math:`\chi^2` values are never mixed into the
spectrum. A finer trial grid digs deeper minima *everywhere, noise
included*, so refining only the peak would inflate the SDE and bias the
false-alarm calibration; keeping the spectrum uniform preserves the
statistic's scale. Consequently ``chi2_min`` can sit slightly below the
minimum of the returned spectrum — that is by design.

Tuning
------

``t0_oversample`` (default 3)
    Trial epochs per transit duration in the coarse scan. The default
    favors speed; the reference ``transitleastsquares`` package steps
    ~33× finer. Because the SDE is a period-space contrast, the coarse
    epoch grid costs only a few percent of detection significance
    (measured), while the exact refinement restores full parameter
    precision at the candidates. Raise it (e.g. to 33) for
    sensitivity-critical searches at a roughly proportional increase
    in kernel time.
``refine_top_k`` (default 50) / ``refine_oversample`` (default 33)
    How many candidate periods per lightcurve are re-fit exactly, and
    the epoch resolution of that re-fit.
``n_durations`` (default 15)
    Log-spaced trial durations per period within the (Keplerian or
    fixed) duration window.
``nbins`` / ``block_size``
    Phase-bin and CUDA block-size overrides. By default the period
    grid is split into bands that each compile with their own bin
    count (long-period bands need fewer bins), sized to the device's
    shared-memory limit — overriding is rarely necessary.

References
----------

.. [HH2019] Hippke & Heller (2019), "Optimized transit detection
    algorithm to search for periodic transits of small planets", A&A
    623, A39

.. [Ofir2014] Ofir (2014), "Optimizing the search for transiting
    planets in long time series", A&A 561, A138
