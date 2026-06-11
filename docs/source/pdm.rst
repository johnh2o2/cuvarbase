Phase Dispersion Minimization
=============================

Phase dispersion minimization [S1978]_ phase-folds the data at each trial
frequency and measures how "dispersed" the folded lightcurve is. If the
trial frequency is close to the true frequency of a stationary signal, the
folded data trace out a coherent curve and the scatter around that curve is
small; at an unrelated frequency the fold looks like noise and the scatter
is comparable to the total variance of the data.

Classically, PDM bins the folded data into :math:`M` phase bins and
computes the statistic

.. math::
    \Theta(f) = \frac{s^2(f)}{\sigma^2},

where :math:`s^2(f)` is the (weighted) variance of the data around the
per-bin means at trial frequency :math:`f` and :math:`\sigma^2` is the
total (weighted) variance. :math:`\Theta \approx 1` for noise and
:math:`\Theta \ll 1` near the true frequency.

``cuvarbase`` returns the equivalent *peak-finding* statistic

.. math::
    P(f) = 1 - \Theta(f),

so the best candidate frequencies appear as **maxima** of the returned
power array, consistent with the other periodograms in this package.

To our knowledge this is the only GPU implementation of PDM currently
available. It is used in production-scale searches but receives
maintenance-level development; if you find problems, please open an issue.

PDM variants
------------

The ``kind`` argument of :func:`PDMAsyncProcess.run` selects the dispersion
model:

* ``binned_step`` — classic Stellingwerf PDM: the model is the (weighted)
  mean in each of ``nbins`` phase bins.
* ``binned_linterp`` (default) — like ``binned_step``, but the model is
  linearly interpolated between bin centers (a "PDM2"-style refinement
  that reduces binning artifacts).
* ``binless_tophat`` — no binning; each point is compared against a local
  mean computed from all points within a phase distance ``dphi``.
* ``binless_gauss`` — like ``binless_tophat``, but neighbors are weighted
  by a Gaussian in phase distance with width ``dphi``.

Each variant also has a ``*_fast`` version (``binned_linterp_fast``,
``binned_step_fast``, ``binless_tophat_fast``, ``binless_gauss_fast``)
that computes the same statistic with shared-memory tiling and a one-pass
sum-of-squares formulation. The fast kernels are substantially quicker on
large datasets and are numerically equivalent up to single-precision
round-off; results may differ from the reference kernels at the
:math:`\sim 10^{-6}` level.

An example with ``cuvarbase``
-----------------------------

.. code-block:: python

    import numpy as np
    from cuvarbase.pdm import PDMAsyncProcess

    # make some fake data
    t = np.sort(365 * np.random.rand(300))
    y = 12 + 0.1 * np.cos(2 * np.pi * t / 5.0)
    y += 0.1 * np.random.randn(len(t))
    dy = 0.1 * np.ones_like(y)

    # start a PDM process
    proc = PDMAsyncProcess()

    # format your data as a list of (t, y, err) lightcurves
    data = [(t, y, dy)]

    # run PDM; a frequency grid is generated automatically
    # if ``freqs`` is not given
    results = proc.run(data, kind='binned_linterp', nbins=20)
    proc.finish()

    # results is a list of (freqs, power) tuples, one per lightcurve
    freqs, power = results[0]
    best_freq = freqs[np.argmax(power)]
    print(1.0 / best_freq)  # ~5.0

You can supply your own frequency grid (or one per lightcurve), and any
keyword arguments accepted by :func:`cuvarbase.utils.autofrequency`
(``samples_per_peak``, ``nyquist_factor``, ``minimum_frequency``,
``maximum_frequency``) are forwarded when the grid is generated
automatically:

.. code-block:: python

    freqs = np.linspace(0.01, 10.0, 100000)
    results = proc.run(data, freqs=freqs, kind='binless_gauss_fast',
                       dphi=0.05)

API notes
---------

* ``run(data, freqs=None, kind='binned_linterp', nbins=10, dphi=0.05)``
  takes ``data`` as a list of ``(t, y, err)`` tuples. Observation
  uncertainties ``err`` are converted to normalized inverse-variance
  weights internally, and ``t`` and ``y`` are mean-centered before
  transfer to the GPU.
* ``nbins`` controls the number of phase bins for the ``binned_*``
  variants; ``dphi`` controls the phase window/width for the
  ``binless_*`` variants.
* The legacy input format ``[(t, y, w, freqs), ...]`` (weights and
  frequencies packed into the data tuples) is still accepted for
  backward compatibility but is **deprecated** and emits a
  ``DeprecationWarning``; it returns bare power arrays instead of
  ``(freqs, power)`` tuples.

.. [S1978] `Stellingwerf 1978 <https://ui.adsabs.harvard.edu/abs/1978ApJ...224..953S/abstract>`_
