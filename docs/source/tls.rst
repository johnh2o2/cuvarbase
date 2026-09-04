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
  budget). The epoch ``floor(min(t))`` is subtracted in float64 before
  the float32 cast (so BJD-scale timestamps are safe), but the fold
  itself is float32, so its phase precision degrades with the baseline
  (about 1e-4 d at 1400 d). It remains available as a reference
  implementation and for the low-level plumbing (custom streams,
  pre-compiled kernels, externally-managed memory) that the batch
  engine does not expose.

Accuracy is validated two ways in the test suite: golden tests against
the reference `transitleastsquares
<https://github.com/hippke/tls>`_ package, and injected-transit
recovery tests across cadence regimes. The SDE is defined exactly as
in the reference package (see below), and on the reference's own
period grid the default configuration reports the same SDE for the
same detection to within the coarse-vs-fine epoch grid difference
(measured 5-15%) at a small fraction of the cost; see
``docs/BENCHMARK_RESULTS.md`` for measured numbers.

.. note::

   **Input validation.** Since 1.0 every entry point rejects
   non-finite ``t``/``y``/``dy``, ``dy <= 0``, mismatched array
   lengths, too-short light curves and non-finite or non-positive
   frequency grids with a ``ValueError`` raised on the host, before
   any GPU work. See :ref:`Input validation <input-validation>` for
   the full rules and the pre-1.0 behaviour they replace.

Input conventions
-----------------

* ``t``: observation times in days. BJD-scale absolute times are safe
  on the default fast path.
* ``y``: fluxes **normalized so the out-of-transit baseline is ~1.0**.
  The transit model is :math:`1 - \delta\,T(x)` with the out-of-transit
  level **fixed at exactly 1** — there is no free baseline term (as in
  the reference package). No TLS path rescales the input, so
  unnormalized fluxes (e.g. raw counts) produce meaningless depths, and
  even a small normalization offset matters: for a P = 7.3 d transit at
  sigma = 1e-3 per point, an offset of +5e-4 raised the SDE from 21.5 to
  29.3 with the depth 20% low, -5e-4 halved it to 10.1, and -1e-3 gave
  the wrong period. Normalize to a *median* out-of-transit level of 1
  (to ~0.1 sigma per point) before searching.
* ``dy``: per-point flux uncertainties (same units as ``y``).
* ``periods``: any order is accepted (the grid is sorted internally and
  every per-period output array is returned in the caller's order).

Searching a single lightcurve
-----------------------------

.. code-block:: python

    import numpy as np
    from cuvarbase.tls import tls_search_gpu

    # t (days), y (normalized flux), dy (uncertainties)
    results = tls_search_gpu(t, y, dy)

    print(results['period'])    # best-fit period (days)
    print(results['T0'])        # mid-transit time (days): first transit
                                # at or after min(t)
    print(results['t0_phase'])  # the same epoch as a phase in [0, 1)
                                # relative to floor(min(t))
    print(results['duration'])  # transit duration (days)
    print(results['depth'])     # fractional transit depth
    print(results['SDE'])       # signal detection efficiency

    # fold so the transit sits at phase 0
    phase = ((t - results['T0']) / results['period']) % 1.0

``T0`` is an absolute time on the same scale as ``t`` on every path
(``min(t) <= T0 < min(t) + period``, the convention of the reference
package).

The trial period grid is generated automatically following [Ofir2014]_
(pass ``period_min``/``period_max`` to bound it, or ``periods`` for an
explicit grid). At every trial period the search scans ``n_durations``
log-spaced durations inside a **Keplerian duration window**
``[0.5, 2] x q_kep(P; R_star, M_star, R_planet)`` built by
:func:`cuvarbase.tls_grids.duration_window` from the stellar parameters
the function takes (``qmin_fac``/``qmax_fac``/``R_planet`` adjust it;
explicit per-period ``qmin``/``qmax`` arrays override it). The window
follows :math:`P^{-2/3}` and stays physical out to any period. Before
1.0 the default was a constant window ``[0.005, 0.15]`` at every period,
which excludes the Keplerian duration beyond P ~ 60 d for a Sun-like
star (18.5 d for an M dwarf) — a P = 365 d transit on a 1400-d baseline
came back at 182.5 d with half the depth. That window is still available
as ``duration_window='fixed'`` and warns whenever it is unphysical for
the grid. :func:`cuvarbase.tls.tls_transit` is the explicit-name wrapper
for the same Keplerian search:

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
``period_uncertainty``, ``T0`` — the absolute time of the first
mid-transit at or after ``min(t)`` — ``t0_phase``, ``duration``,
``depth``, ``chi2_min``) and the detection statistics (``SDE``,
``SDE_raw``, ``SNR``, ``n_transits``). Pass ``return_arrays=True`` to
also get the per-period :math:`\chi^2` spectrum and derived quantities.
A lightcurve with no valid solution at any trial period (flat or
noiseless flux) gets ``SDE = 0``, NaN best-fit parameters and the
message under ``'error'``, with a warning.

Detection statistics and refinement
-----------------------------------

**SDE.** The signal residue is :math:`\mathrm{SR} = \chi^2_{\min} /
\chi^2` (1 at the best trial period), and

.. math::

    \mathrm{SDE}_{\rm raw} = \frac{1 - \langle \mathrm{SR} \rangle}
    {\sigma(\mathrm{SR})}, \qquad
    \mathrm{SDE} = \frac{\max(D) - \langle D \rangle}{\sigma(D)},
    \quad D = \mathrm{SR} - \mathrm{runmed}(\mathrm{SR}),

with an edge-extended running median of 91 points (``sde_kernel_size``;
length-scaled below 910 periods). These are exactly the statistics of
the reference ``transitleastsquares`` package, so its published SDE
thresholds apply to cuvarbase's numbers. (Before 1.0 the SR was
:math:`1 - \chi^2/\max\chi^2`, which agrees under the null but gave
about half the SDE for strong signals, and the running median was
zero-padded, which inflated the detrended power at the grid edges.)

**SNR** is :math:`\sqrt{\chi^2_0 - \chi^2_{\min}}`, the
delta-chi-squared significance of the best fit over the constant model
(``chi2_min`` from the exact refinement); it is not the reference's
``depth / std * sqrt(n_in_transit)``.

**There is no calibrated FAP.** The SDE is a contrast statistic whose
null distribution moves with the number of trial periods and the
baseline: for pure noise the audit measured a mean of 6.4 and std 1.0
(6157 periods, 60 d), with 23% of noise-only lightcurves above SDE = 7,
and 92% above 7 at 365 d with 43,780 periods; the reference package's
fixed SDE-to-FAP table is miscalibrated for the same reason. Before 1.0
every result carried a ``'FAP'`` computed from a fixed function of the
SDE; that key is gone. For an honest number use the opt-in null
bootstrap of :func:`cuvarbase.tls.tls_search_batch`:

.. code-block:: python

    results = tls_search_batch(lightcurves, periods=periods,
                               fap_null_draws=200, fap_seed=1)
    r = results[0]
    r['FAP']       # (1 + #null SDE >= observed) / (fap_null_draws + 1)
    r['SDE_null']  # the null SDEs, for choosing your own threshold

Each draw permutes a lightcurve's (y, dy) pairs over its times (a
white-noise null: same sampling and noise distribution, no coherent
signal, no red noise) and searches the identical grid with the same
settings; 400 such searches of 2880 points x 6157 periods took 1.6 s
on an A40. The smallest resolvable FAP is ``1 / (fap_null_draws + 1)``.

**Refinement.** The per-period spectrum that feeds the SDE comes from
the *coarse* phase-binned scan at uniform fidelity. The exact
refinement pass only sharpens the reported best-fit parameters (period
choice among the top candidates, ``T0``, ``duration``, ``depth``,
``chi2_min``) — refined :math:`\chi^2` values are never mixed into the
spectrum. A finer trial grid digs deeper minima *everywhere, noise
included*, so refining only the peak would inflate the SDE; keeping the
spectrum uniform preserves the statistic's scale. Consequently
``chi2_min`` can sit slightly below the minimum of the returned
spectrum — that is by design.

Tuning
------

``t0_oversample`` (default 3)
    Trial epochs per transit duration in the coarse scan. The default
    favors speed; the reference ``transitleastsquares`` package steps
    ~100× finer (every cadence for dense data). Measured cost of the
    default: the SDE of a P = 7.3 d, q = 0.021 transit varies by 17%
    (19.8-23.4) with where the true epoch falls relative to the coarse
    grid (6.5% at 33), and for a narrow transit (M dwarf, 3.4 cadences
    of 30 min) the SDE is 29.1 at 3 vs 32.9 at 10 and 32.7 at 33 (-11%).
    The exact refinement restores full parameter precision at the
    candidates but does not enter the SDE. Raise it to 10 (matched 33
    within 1% in those runs) for sensitivity-critical or
    narrow-transit searches, at a roughly proportional increase in
    kernel time.
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
