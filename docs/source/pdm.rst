Phase Dispersion Minimization
=============================

Phase dispersion minimization [S1978]_ phase-folds the data at each trial
frequency and measures how "dispersed" the folded lightcurve is. If the
trial frequency is close to the true frequency of a stationary signal, the
folded data trace out a coherent curve and the scatter around that curve is
small; at an unrelated frequency the fold looks like noise and the scatter
is comparable to the total variance of the data.

Classically [S1978]_, PDM bins the folded data into :math:`M` phase bins
and computes

.. math::
    \Theta(f) = \frac{s^2(f)}{\sigma^2}
              = \frac{\sum_i \left(y_i - m_i(f)\right)^2 / (N - M)}
                     {\sum_i \left(y_i - \bar{y}\right)^2 / (N - 1)},

where :math:`m_i(f)` is the mean of the bin that observation :math:`i`
falls in at trial frequency :math:`f`, :math:`N` is the number of
observations and :math:`M` the number of occupied bins.
:math:`\Theta \approx 1` for noise and :math:`\Theta \ll 1` near the true
frequency.

The statistic ``cuvarbase`` computes
------------------------------------

The kernels return the *peak-finding* sum-of-squares ratio

.. math::
    P(f) = 1 - \frac{\sum_i w_i \left(y_i - m_i(f)\right)^2}
                    {\sum_i w_i \left(y_i - \bar{y}\right)^2},
    \qquad \bar{y} = \sum_i w_i y_i,

with weights :math:`w_i \propto 1/\sigma_i^2` normalized to
:math:`\sum_i w_i = 1` and :math:`m_i(f)` the model of the folded
lightcurve at the phase of observation :math:`i` (a bin mean, an
interpolation between bin means, or a local mean, depending on the
variant; see below). The best candidate frequencies appear as **maxima**
of the returned power array, consistent with the other periodograms in
this package.

:math:`P(f)` is **not** :math:`1 - \Theta(f)`: the degrees-of-freedom
factors :math:`N - M` and :math:`N - 1` are not applied (for uniform
weights, :math:`1 - P(f) = \frac{N - M}{N - 1}\,\Theta(f)`). Keep the
consequences in mind:

* **Noise floor.** For pure noise :math:`\Theta \approx 1`, but the
  expected value of :math:`P` is :math:`(M - 1)/(N - 1)` (exact for
  ``binned_step`` with Gaussian noise and uniform weights;
  ``binned_linterp`` behaves similarly): up to 0.47 for :math:`N = 20`
  observations in 10 bins, about 0.18 for :math:`N = 50` and 0.01 for
  :math:`N = 1000`. Judge a peak against this floor, not against zero.
* **Comparability.** Values are only comparable between runs with the
  same ``nbins`` (or ``dphi``) and the same :math:`N`; more bins raise
  the whole periodogram.
* **Gappy data.** :math:`M` counts *occupied* bins, so with incomplete
  phase coverage the floor varies along the periodogram and, for small
  :math:`N`, the ranking of candidate peaks can differ from that of
  :math:`\Theta`.
* **Binless kinds.** ``binless_tophat`` and ``binless_gauss`` use the
  same ratio with :math:`m_i(f)` the kernel-weighted local mean (which
  includes the point itself); their noise floor depends on ``dphi`` and
  :math:`N`.

To recover Stellingwerf's :math:`\Theta` for a binned kind, rescale
:math:`1 - P(f)` by :math:`(N - 1)/(N - M(f))` with :math:`M(f)` counted
on the host (the kernels do not return it); ``cuvarbase`` does not do
this for you.

To our knowledge this is the only GPU implementation of PDM currently
available. As of v1.0 it has fast kernels for all variants, unit tests,
and this documentation; if you find problems, please open an issue.

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
  ``(freqs, power)`` tuples. The weights ``w`` may have any scale (raw
  :math:`1/\sigma^2`, all ones, ...): they are normalized to sum to one
  internally, exactly like the weights derived from ``err``.

.. [S1978] `Stellingwerf 1978 <https://ui.adsabs.harvard.edu/abs/1978ApJ...224..953S/abstract>`_
