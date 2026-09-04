Conditional Entropy
===================

The conditional entropy period finder [G2013]_ phase-folds the data at each trial frequencies and estimates
the conditional entropy :math:`H(m|\phi)` of the data. The idea is that the data with the least entropy (intuitively: the greatest "structure" or "non-randomness"), should correspond to the correct frequency of a stationary signal.

Here,

.. math::
	H(m|\phi) = H(m, \phi) - H(\phi) = \sum_{m,\phi}p(m, \phi)\log\left(\frac{p(\phi)}{p(m, \phi)}\right)


where :math:`p(m, \phi)` is the density of points that fall within the bin located at phase :math:`\phi` and magnitude :math:`m` and :math:`p(\phi) = \sum_m p(m, \phi)` is the density of points that fall within the phi range.

.. note::

	**What the returned value is.** ``cuvarbase`` returns
	:math:`H(m|\phi) + \sum_m p(m) \log \Delta m_m`, where
	:math:`\Delta m_m` is the width of magnitude bin :math:`m` in units of
	the (normalized) magnitude range and :math:`p(m)` is the fraction of
	the histogram mass in that bin -- i.e. the entropy of the magnitude
	*density* rather than of the bin probabilities. With the default
	``mag_overlap=0`` every bin has :math:`\Delta m_m = 1/\mathrm{mag\_bins}`
	and the offset is :math:`\log(1/5) = -1.609` for the default
	``mag_bins=5``. With ``mag_overlap > 0`` the unweighted kernels use
	:math:`\Delta m_m = \min(\mathrm{mag\_overlap} + 1,\,
	\mathrm{mag\_bins} - m) / \mathrm{mag\_bins}` (the top bins are
	truncated at the brightest magnitude cell), whereas the weighted
	kernel integrates every bin over the full window and uses the
	constant :math:`(\mathrm{mag\_overlap} + 1) / \mathrm{mag\_bins}`, so
	``weighted=True`` and ``weighted=False`` spectra then differ by a
	constant. With ``balanced_magbins=True`` each bin uses its own width.
	In every case the offset is the same at every frequency (the
	per-magnitude-bin totals do not depend on the trial frequency), so
	the location of the minimum is unaffected; subtract it to recover
	Graham et al.'s normalization. Lower values mean more structure: the
	best frequency is the **argmin** of the periodogram.

	With ``compute_log_prob=True`` the returned quantity is instead the
	Poisson log-likelihood of the phase-folded histogram under the
	phase-independent null model,
	:math:`\sum_{\phi, m} [N_{\phi m} \log N^{\rm exp}_{\phi m} -
	N^{\rm exp}_{\phi m} - \log\Gamma(N_{\phi m} + 1)]` with
	:math:`N^{\rm exp}_{\phi m} = N_\phi\, p(m)`. It is likewise
	**minimized** at the true frequency.

.. plot:: plots/ce_example.py


.. note::

   **Input validation.** Since 1.0 every entry point rejects
   non-finite ``t``/``y``/``dy``, ``dy <= 0``, mismatched array
   lengths, too-short light curves and non-finite or non-positive
   frequency grids with a ``ValueError`` raised on the host, before
   any GPU work. See :ref:`Input validation <input-validation>` for
   the full rules and the pre-1.0 behaviour they replace.

An example with ``cuvarbase``
-----------------------------

.. code-block:: python
	
	import cuvarbase.ce as ce
	import numpy as np

	# make some fake data
	t = np.sort(np.random.rand(100))
	y = np.cos(2 * np.pi * 10 * t)
	y += np.random.randn(len(t))
	dy = np.ones_like(t)

	# start a conditional entropy process
	proc = ce.ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)

	# format your data as a list of lightcurves (t, y, dy)
	data = [(t, y, dy)]

	# run the CE process with your data
	results = proc.run(data)

	# finish the process (necessary: the results are filled in
	# asynchronously and are only complete after finish())
	proc.finish()

	# Results is a list of [(freqs, CE), ...] for each lightcurve
	# in ``data``.
	freqs, ce_spectrum = results[0]


If you want to run CE on large datasets, you can do

.. code-block:: python
	
	proc.large_run(data, max_memory=1e9)

instead of ``run``, which will ensure that the memory limit (1 GB in this case) is not exceeded on the GPU (unless of course you have other processes running). 

The frequency grid can be any 1-D numeric array (``float32``,
``float64``, integers or a Python list); to use a different grid for
each lightcurve pass a list with one grid per lightcurve.

Reusing memory across many lightcurves
--------------------------------------

For many lightcurves on the same frequency grid, ``batched_run_const_nfreq``
allocates the GPU buffers once. The lower-level equivalent is
``preallocate``, which uploads the frequency grid and binds each memory
object to one of the process streams so that ``finish()`` synchronizes
the result transfers:

.. code-block:: python

	proc = ce.ConditionalEntropyAsyncProcess()
	proc.preallocate(max_nobs=1000, freqs=freqs, nlcs=1)
	for t, y, dy in lightcurves:      # each with <= 1000 observations
	    results = proc.run([(t, y, dy)], freqs=freqs)
	    proc.finish()
	    ce_spectrum = np.copy(results[0][1])

Passing a different grid of the same length to ``run`` re-uploads it; a
grid of a different length raises ``ValueError``.

Binning details
---------------

* Magnitudes are normalized to :math:`[0, 1]` over the lightcurve's
  range and binned into ``mag_bins`` uniform bins; the brightest point
  (normalized magnitude exactly 1) belongs to the last bin.
* ``weighted=True`` spreads each point over the magnitude bins according
  to the Gaussian probability mass implied by its uncertainty. A bin is
  skipped only when the *whole* bin lies more than ``max_phi`` sigma
  from the point (the point's own bin is always kept), so every point
  retains essentially all of its mass. ``widen_mag_range=True`` pads the
  normalized range by ``max_phi`` median uncertainties on each side.
* ``balanced_magbins=True`` uses ``mag_bins`` bins holding the same
  number of points each, to within one (each group holds
  :math:`\lfloor N/\mathrm{mag\_bins}\rfloor` or one more point).
  Bin edges lie at the midpoints between adjacent
  sorted groups, so the widths tile :math:`[0, 1]`; a width is floored at
  :math:`10^{-6}` of the range so quantized magnitudes (bins made of a
  single repeated value) cannot make the entropy :math:`-\infty`.

.. [G2013] `Graham et al. 2013 <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1306.6664>`_

Unsupported option combinations
-------------------------------

CE is in maintenance mode (see the module notice), and the following
option combinations are **not implemented** — they raise
``ValueError`` (from the constructor, or from ``run``/``preallocate``
when passed as per-call keyword arguments) rather than silently
misbehaving:

* ``use_fast=True`` with ``weighted=True`` — the fast shared-memory
  kernels have no weighted variant.
* ``use_fast=True`` with ``balanced_magbins=True`` — the fast kernels
  only implement uniform magnitude bins.
* ``balanced_magbins=True`` with ``compute_log_prob=True``.
* ``mag_overlap > 0`` with ``balanced_magbins=True`` — overlapping
  magnitude bins are incompatible with the balanced-bin layout.
* ``weighted=True`` with ``balanced_magbins=True`` or
  ``compute_log_prob=True``.

``use_fast=True`` with ``use_double=True`` is supported (in single and
double precision, for any ``phase_bins``/``mag_bins``).

For an actively developed GPU conditional-entropy implementation, see
`periodfind <https://github.com/scope-ml/periodfind>`_.
