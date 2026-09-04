Box least squares (BLS) periodogram
***********************************

The box-least squares periodogram [BLS]_ searches for the periodic dips in brightness that occur when, e.g., a planet passes in front of its host star. The algorithm fits
a `boxcar function <https://en.wikipedia.org/wiki/Boxcar_function>`_ to the data. The parameters used are

- ``q``: the transit duration as a fraction of the period :math:`t_{\rm trans} / P`
- ``phi0``: the phase offset of the transit (from 0)
- ``delta``: the difference between the out-of-transit brightness and the brightness during transit 
- ``y0``: The out-of-transit brightness


.. plot:: plots/bls_transit_diagram.py


Using ``cuvarbase`` BLS
-----------------------


.. plot:: plots/bls_example.py
	:include-source:


A shortcut: assuming orbital mechanics
--------------------------------------

The derivation below follows Seager & Mallén-Ornelas (2003) [SM03]_: their eq. (3) relates the transit duration to the orbital period for a body transiting a star of a given mean density, and eq. (4) is the Kepler's-third-law step used here.

If you assume :math:`R_p\ll R_{\star}`, :math:`M_p\ll M_{\star}`, :math:`L_p\ll L_{\star}`, and :math:`e\ll 1`,  where :math:`e` is the ellipticity of the planetary orbit, :math:`L` is the luminosity, :math:`R` is the radius, and :math:`M` mass, you can eliminate a free parameter.

This is because the orbital period obeys `Kepler's third law <https://en.wikipedia.org/wiki/Kepler's_laws_of_planetary_motion#Third_law>`_,

.. math::
	P^2 \approx \frac{4\pi^2a^3}{G(M_p + M_{\star})}

.. plot:: plots/planet_transit_diagram.py


The angle of the transit is

.. math::

	\theta = 2{\rm arcsin}\left(\frac{R_p + R_{\star}}{a}\right)

and :math:`q` is therefore :math:`\theta / (2\pi)`. Thus we have a relation between :math:`q` and the period :math:`P`

.. math::

	\sin{\pi q} = (R_p + R_{\star})\left(\frac{4\pi^2}{P^2 G(M_p + M_{\star})}\right)^{1/3}

By incorporating the fact that

.. math::
	
	R_{\star} = \left(\frac{3}{4\pi\rho_{\star}}\right)^{1/3}M_{\star}^{1/3}

where :math:`\rho_{\star}` is the average stellar density of the host star, we can write

.. math::

	\sin{\pi q} = \frac{(1 + r)}{(1 + m)^{1/3}} \left(\frac{3\pi}{G\rho_{\star}}\right)^{1/3} P^{-2/3}

where :math:`r = R_p / R_{\star}` and :math:`m = M_p / M_{\star}`. We can get rid of the constant factors and convert this to more intuitive units to obtain

.. math::

	\sin{\pi q} \approx 0.238 (1 + r - \frac{m}{3} + \dots{}) \left(\frac{\rho_{\star}}{\rho_{\odot}}\right)^{-1/3} \left(\frac{P}{\rm day}\right)^{-2/3}

where here we've expanded :math:`(1 + r) / (1 + m)^{1/3}` to first order in :math:`r` and :math:`m`.


Using the Keplerian assumption in ``cuvarbase``
-----------------------------------------------

.. plot:: plots/bls_example_transit.py
	:include-source:


Period spacing considerations
-----------------------------

The frequency spacing :math:`\delta f` needed to resolve a BLS signal with width :math:`q`, is

.. math::
	\delta f \lesssim \frac{q}{T}

where :math:`T` is the baseline of the observations (:math:`T = {\rm max}(t) - {\rm min}(t)`). This can be especially problematic if no assumptions are made about the nature of the signal (e.g., a Keplerian assumption). If you want to resolve a transit signal with a few observations, the minimum :math:`q` value that you would need to search is :math:`\propto 1/N` where :math:`N` is the number of observations.

For a typical Lomb-Scargle periodogram, the frequency spacing is :math:`\delta f \lesssim 1/T`, so running a BLS spectrum with an adequate frequency spacing over the same frequency range requires a factor of :math:`\mathcal{O}(N)` more trial frequencies, each of which requiring :math:`\mathcal{O}(N)` computations to estimate the best fit BLS parameters. That means that BLS scales as :math:`\mathcal{O}(NN_f)` in the number of trial frequencies actually searched -- a grid that is itself a factor :math:`\mathcal{O}(N)` denser than the corresponding Lomb-Scargle grid -- while Lomb-Scargle only scales as :math:`\mathcal{O}(N_f\log N_f)`

However, if you can use the assumption that the transit is caused by an edge-on transit of a circularly orbiting planet, we not only eliminate a degree of freedom, but (assuming :math:`\sin{\pi q}\approx \pi q`)

.. math::

	\delta f \propto q \propto f^{2/3}

This duty-cycle-aware spacing :math:`\delta f \approx q(f) / (\mathrm{OS}\,T)` is the optimal transit-search grid of Ofir (2014) [O2014]_ (his eq. 4, with oversampling :math:`\mathrm{OS}`); it is implemented in :func:`cuvarbase.bls.transit_autofreq` and :func:`cuvarbase.bls_frequencies.keplerian_freq_grid`.

The minimum frequency you could hope to measure a transit period would be :math:`f_{\rm min} \approx 2/T` (Ofir 2014, Sect. 3.1 [O2014]_), and the maximum frequency is determined by :math:`\sin{\pi q} < 1` which implies

.. math::

	f_{max} = 8.612~{\rm c/day}~\times \left(1 - \frac{3r}{2} + \frac{m}{2} -\dots{}\right) \sqrt{\frac{\rho_{\star}}{\rho_{\odot}}}

The leading coefficient is the surface-orbit frequency :math:`f_{\max,0} = \sqrt{G\rho_\star / 3\pi}` evaluated at solar mean density (the :math:`r, m \to 0` limit). ``cuvarbase`` uses the value ``8.6307`` c/day for this constant (see :func:`cuvarbase.bls.fmax_transit0`); the ``8.612`` here is the same derived quantity, the ~0.2% difference being the precision of the adopted :math:`G` and :math:`\rho_\odot`. It is a *derived* constant, not a literature value.


For a 10 year baseline, this translates to :math:`2.7\times 10^5` trial frequencies. The number of trial frequencies needed to perform Lomb-Scargle over this frequency range is only about :math:`3.1\times 10^4`, so 8-10 times less. However, if we were to search the *entire* range of possible :math:`q` values at each trial frequency instead of making a Keplerian assumption, we would instead require :math:`5.35\times 10^8` trial frequencies, so the Keplerian assumption reduces the number of frequencies by over 1,000.


Sparse BLS for small datasets
------------------------------

For datasets with a small number of observations, the standard BLS algorithm that bins observations and searches over a grid of transit parameters can be inefficient. The "Sparse BLS" algorithm [SparseBLS]_ avoids this redundancy by directly testing all pairs of observations as potential transit boundaries.

At each trial frequency, the observations are sorted by phase. Then, instead of searching over a grid of (phase, duration) parameters, the algorithm considers each pair of consecutive observations (i, j) as defining:

- Transit start phase: :math:`\phi_0 = \phi_i`
- Transit duration: :math:`q = \phi_j - \phi_i`

This approach has complexity :math:`\mathcal{O}(N_{\rm freq} \times N_{\rm data}^2)` compared to :math:`\mathcal{O}(N_{\rm freq} \times N_{\rm data} \times N_{\rm bins})` for the standard gridded approach. ``cuvarbase`` selects it for small datasets (by default :math:`N_{\rm data} < 500`) for its detection properties -- every candidate transit is tested exactly, with no binning or phase-grid loss -- not for speed: on the GPU the sparse kernel is slower than the binned fast kernel at every :math:`N_{\rm data}` (its per-frequency work grows as :math:`N_{\rm data}^2`), and it needs :math:`\mathcal{O}(N_{\rm data})` shared memory per block, which limits it to roughly 2,000 points.

Using Sparse BLS in ``cuvarbase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``eebls_transit`` function automatically selects between sparse BLS (for small datasets) and the GPU-accelerated standard BLS (for larger datasets):

.. code-block:: python

    from cuvarbase.bls import eebls_transit
    import numpy as np
    
    # Generate small dataset (e.g., 100 observations)
    t = np.sort(np.random.rand(100)) * 365  # 1 year baseline
    # ... (generate y, dy from your data)
    
    # Automatically uses sparse BLS for ndata < 500
    freqs, powers, solutions = eebls_transit(
        t, y, dy,
        fmin=0.1,  # minimum frequency
        fmax=10.0  # maximum frequency
    )
    
    # Or explicitly control the method:
    freqs, powers, solutions = eebls_transit(
        t, y, dy,
        fmin=0.1, fmax=10.0,
        use_sparse=True  # Force sparse BLS
    )

Both paths apply the same per-frequency Keplerian duration bounds
(``qmin_fac``/``qmax_fac`` times the fiducial ``q_transit`` value),
exactly per frequency, so results are comparable across the
``sparse_threshold`` boundary up to the two algorithms' different
candidate sets (a binned box grid vs observation pairs). For
:math:`N_{\rm data} \ge` ``sparse_threshold`` the periodogram comes
from the fast shared-memory kernel (:func:`cuvarbase.bls.eebls_gpu_fast`)
and the best-fit ``(q, phi0)`` is recovered at the ``n_solutions``
(default 10) highest peaks; the remaining entries of ``solutions`` are
``None``. For a solution at every frequency, run the full binned
search with :func:`cuvarbase.bls.eebls_transit_gpu` or
:func:`cuvarbase.bls.eebls_gpu` (which also honour per-frequency
``qmin``/``qmax`` arrays exactly, independently of ``freq_batch_size``
and of the free device memory; before 1.0 the standard path collapsed
them to one batch-wide window).

The shared-memory kernels do not search a continuum of durations.
Phase is binned into :math:`n_f = \lfloor 1/q_{\rm min} \rfloor`
bins and a trial box spans :math:`m` of them, so the durations
actually searched are :math:`q = m / n_f` for
:math:`m = 1, 1 + \Delta(1), \ldots` (``dlogq`` sets the geometric
step :math:`\Delta`) up to and including
:math:`\lfloor n_f / \lfloor 1/q_{\rm max} \rfloor \rfloor`, the
widest box with :math:`q \le q_{\rm max}`. Before 1.0 the loop
stopped one rung short and never tested ``qmax`` itself -- with
``qmin=0.025``, ``qmax=0.1`` the widest box searched was ``q=0.075``,
and an on-grid ``q=0.1`` transit was recovered at ~73% of its exact
power. The geometric step can still overshoot the last rung: with the
defaults (``qmin=0.01``, ``qmax=0.5``, ``dlogq=0.3``) the ladder ends
at ``q=0.48``. Box start phases step one fine bin divided by
``noverlap``, so a box of :math:`m` bins can be misaligned by up to
:math:`1/(2 m\,{\rm noverlap})` of its width; boxes near ``qmin``
therefore recover only part of their exact power (49-90% in the Sep
2026 audit). Raise ``noverlap`` (nearly free on the fused kernel) or
lower ``qmin`` before comparing fast-path power with an exact box fit.

You can also use sparse BLS directly with ``sparse_bls_cpu`` (or
``sparse_bls_gpu``). By default all durations :math:`q \in (0, 0.5]`
are searched; the optional ``qmin``/``qmax`` arguments (scalar or
per-frequency arrays) restrict the candidate durations:

.. code-block:: python

    from cuvarbase.bls import sparse_bls_cpu, q_transit

    # Define trial frequencies
    freqs = np.linspace(0.1, 10.0, 1000)

    # Run sparse BLS (unconstrained durations)
    powers, solutions = sparse_bls_cpu(t, y, dy, freqs)

    # ... or restrict durations to a Keplerian band
    qvals = q_transit(freqs)
    powers, solutions = sparse_bls_cpu(t, y, dy, freqs,
                                       qmin=0.5 * qvals,
                                       qmax=2.0 * qvals)

    # solutions is a list of (q, phi0) tuples for each frequency
    best_idx = np.argmax(powers)
    best_freq = freqs[best_idx]
    best_q, best_phi0 = solutions[best_idx]


.. [BLS] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_
.. [SparseBLS] `Panahi & Zucker 2021 <https://arxiv.org/abs/2103.06193>`_

Power-spectrum convention
-------------------------

By default, all BLS functions in cuvarbase report

.. math::

    P(f) = 1 - \chi^2(f) / \chi^2_0

where :math:`\chi^2(f)` is the weighted sum of squared residuals of
the best-fit box at frequency :math:`f` and :math:`\chi^2_0` is that
of a constant (weighted-mean) model. :math:`P` is dimensionless and
lies in :math:`[0, 1]`, with 1 meaning the box model fits perfectly.

The BLS entry points accept a ``convention=`` keyword (issue
`#17 <https://github.com/johnh2o2/cuvarbase/issues/17>`_) selecting
among exact transformations of this quantity:

* ``'chi2ratio'`` (default): :math:`P` as above.
* ``'snr'``: :math:`\sqrt{\chi^2_0\,P}`, the (unsigned)
  signal-to-noise ratio of the best-fit transit depth,
  :math:`|\hat{\delta}|/\sigma_{\hat\delta}`. At the same
  (period, duration, phase) this equals the power returned by
  ``astropy.timeseries.BoxLeastSquares`` with ``objective='snr'``
  (astropy reports it signed and only keeps flux dips).
* ``'loglik'``: :math:`\chi^2_0\,P / 2`, the improvement in Gaussian
  log-likelihood of the best two-level (in/out-of-transit) model over
  the constant weighted-mean model. Astropy's
  ``objective='likelihood'`` instead measures the improvement against
  the *out-of-transit level* reference, which equals this value
  divided by :math:`(1 - r)` where :math:`r` is the in-transit
  fraction of the total statistical weight; for transit-like signals
  (:math:`q \ll 1`) the two agree closely.

These equivalences are verified against astropy in the test suite on
shared (period, duration, phase) solutions. Standalone conversion is
available via :func:`cuvarbase.bls.convert_bls_power`:

.. code-block:: python

    from cuvarbase.bls import eebls_transit, convert_bls_power

    freqs, p, sols = eebls_transit(t, y, dy, convention='snr')
    # ... or convert an existing chi2ratio periodogram:
    p_loglik = convert_bls_power(p_chi2ratio, y, dy, 'loglik')

Reported ``phi0`` values are transit *start* phases measured relative
to ``floor(min(t))`` (observation times are epoch-subtracted internally to
preserve float32 precision).


.. _input-validation:

Input validation
----------------

Every public entry point in cuvarbase -- BLS, TLS, Lomb-Scargle,
conditional entropy, PDM, the NFFT and NUFFT-LRT -- validates its
light curve and its trial grid on the host before any GPU work
(kernel compilation included) and raises ``ValueError`` when

* ``t``, ``y`` or ``dy`` contains a NaN or an infinity,
* any ``dy`` is zero or negative (uncertainties become
  inverse-variance weights ``dy**-2``),
* ``t``, ``y`` and ``dy`` do not all have the same length,
* the light curve has fewer points than the method needs (four for
  Lomb-Scargle, three for NUFFT-LRT, two elsewhere),
* the frequency grid is empty or contains a non-finite or
  non-positive frequency,
* the transit-duration bounds are not ``0 < qmin <= qmax <= 1`` (the
  binned kernels) or not finite (all paths).

The error message names the array, the number of offending entries and
the first few of their indices::

    >>> eebls_gpu_fast(t, y, dy, freqs)
    ValueError: eebls_gpu_fast: t contains 1 non-finite value(s)
    (NaN or inf) out of 600; first at index/indices 137. Remove or
    interpolate the bad samples before searching.

Before 1.0 these inputs were accepted silently and produced a finite
but wrong periodogram, an all-NaN spectrum, or a kernel crash that
left the process's CUDA context unusable. Because the checks run on
the host, a rejected call is *safe*: the context is untouched and the
next call in the same process succeeds. Nothing changes for valid
finite input.

Filter your data before searching::

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    freqs, power, sols = eebls_transit(t[m], y[m], dy[m])

The two helpers are public and can be reused in your own pipeline:
:func:`cuvarbase.utils.check_lightcurve` and
:func:`cuvarbase.utils.check_freqs`.


Data hygiene: near-zero uncertainties
-------------------------------------

BLS is a *weighted* least-squares fit: each observation enters with
weight :math:`w_i = dy_i^{-2} / \sum_j dy_j^{-2}`. A lightcurve point
with a near-zero reported uncertainty (a common artifact of pipeline
glitches, sentinel values, or unit mistakes) therefore concentrates
essentially *all* of the statistical weight in a single observation.
The box that covers that one point's phase bin then absorbs essentially
all of the weighted variance, so :math:`\chi^2 \approx 0` for the box
model and the reported power :math:`P = 1 - \chi^2/\chi^2_0` saturates
near 1 (typically :math:`\sim 0.99` after binning) — *deterministically*,
in pure noise. Because every trial frequency has some phase bin
containing the dominant point, the result is a spuriously high,
nearly frequency-independent periodogram rather than an isolated peak.

How to recognize it:

- one point dominates the statistical weight: ``max(dy**-2) / sum(dy**-2)``
  is close to 1 (anything above ~0.1 deserves scrutiny);
- suspiciously high BLS power (:math:`\sim 0.99`) on data you expect to
  be noise, roughly flat across trial frequencies.

The recommended guard is an *error floor*: clip the reported
uncertainties from below at a percentile-based floor (and/or clip the
weights from above) before running BLS:

.. code-block:: python

    import numpy as np

    # Error floor: clip dy from below at a percentile-based floor
    # before computing BLS weights.
    dy_floor = np.percentile(dy, 10)  # or a survey-specific value
    dy_safe = np.clip(dy, dy_floor, None)

    # Sanity check: no single point should dominate the total weight.
    w = dy_safe ** -2
    w = w / w.sum()
    if w.max() > 0.1:
        raise ValueError("one point holds {:.0%} of the statistical "
                         "weight; check dy for near-zero values"
                         .format(w.max()))

    freqs, power, sols = eebls_transit(t, y, dy_safe, fmin=0.1, fmax=10.0)

``cuvarbase`` does not apply such a floor automatically — reported
uncertainties are taken at face value — so this check belongs in your
pre-processing.


Precision and reproducibility
-----------------------------

**The phase fold is float32.** Every GPU BLS kernel folds with
``mod1(t * f)`` in single precision, so the phase grid it can resolve is
quantized at :math:`\mathrm{ulp}(T f_\mathrm{max})`, where :math:`T` is
the baseline after epoch subtraction (:math:`1.95\times10^{-3}` cycles
at :math:`T f = 23{,}019`; 5000 points then take only 1977 distinct
phases). For the box edges to land where they should, the narrowest
phase step the search actually uses,

.. math::

    \frac{q_\mathrm{min}}{n_\mathrm{overlap}} \quad\text{(in cycles)},

must stay well above that ulp. When it does not, the transit's power
leaks across bin edges: measured against a float64 replica, ``q =
0.01`` boxes recover 0.968 / 0.924 / 0.901 of the exact power at
:math:`T f = 7000` / 18,250 / 58,400 (worst case 0.846), i.e. a 3-15 %
loss, and a 10-year baseline searched to 20 c/d loses 22 % (a 1-year
baseline at the same frequency loses 6 %). Keplerian ``q0`` boxes --
what :func:`~cuvarbase.bls.eebls_transit` searches by default -- are
much wider and are not affected. If you need ``q ~ 0.01`` at
:math:`T f_\mathrm{max} \gtrsim 7000`, split the baseline into shorter
segments or restrict ``fmax``; the periodogram peak is still found, but
its height (and the depth inferred from it) is biased low.

**Binned results depend on the time origin.** Times are epoch-subtracted
with ``floor(min(t))``, so the *fractional* part of ``min(t)`` shifts
where the phase-bin edges fall relative to the data. Binned BLS powers
move by up to ~10 % with that fraction (11 / 9 / 7.6 % at
``noverlap = 1 / 4 / 8``), occasionally moving the ``eebls_transit``
argmax. This is discretization, not precision loss -- ``t + 2457000.5``
and ``t + 0.5`` agree to 1e-8 -- but it means a periodogram is only
reproducible for a fixed time origin. Sparse BLS, which uses no bins,
is invariant to 5e-4.

**Run-to-run reproducibility.** The fast shared-memory kernels
(:func:`~cuvarbase.bls.eebls_gpu_fast`,
:func:`~cuvarbase.bls.eebls_gpu_fast_optimized`) and the batch kernels
accumulate through float32 atomics, whose summation order is not fixed,
so two identical calls differ by ~1e-8 to 1e-7 in power. Compare
periodograms with a tolerance at that level, not with
``array_equal``. Sparse BLS (:func:`~cuvarbase.bls.sparse_bls_gpu`) and
the conditional-entropy and PDM kernels use no such accumulation and are
bitwise reproducible.


References
----------

.. [SM03] Seager, S. & Mallén-Ornelas, G. (2003), "A Unique Solution of
   Planet and Star Parameters from an Extrasolar Planet Transit Light
   Curve", ApJ 585, 1038 (DOI 10.1086/346105).
.. [O2014] Ofir, A. (2014), "Optimizing the search for transiting
   planets in long time series", A&A 561, A138
   (DOI 10.1051/0004-6361/201220860; arXiv:1307.7330; corrigendum
   A&A 597, C2).
