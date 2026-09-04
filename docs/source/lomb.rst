Lomb-Scargle periodogram
************************

The Lomb-Scargle periodogram ([Barning1963]_, [Vanicek1969]_, [Scargle1982]_, [Lomb1976]_) is one of the best known and most popular period finding algorithms used in astrononomy. If you would like to learn more about least-squares methods for periodic signals, see the review article by [VanderPlas2017]_.

The LS periodogram is a least-squares estimator for the following model

.. math:: 
	
	\hat{y}(t|\omega, \theta) = \theta_1\cos{\omega t} + \theta_2\sin{\omega t}

and it is equivalent to the Discrete Fourier Transform in the regularly-sampled limit. For irregularly sampled data, LS is a maximum likelihood estimator for the parameters :math:`\theta` in the case where the noise is Gaussian. The periodogram has many normalizations in the literature, but ``cuvarbase`` adopts

.. math::

	P(\omega) = \frac{\chi^2_0 - \chi^2(\omega)}{\chi^2_0}

where 

.. math::
	
	\chi^2(\omega) = \sum_i \left(\frac{y_i - \hat{y}(t_i|\omega, \theta)}{\sigma_i}\right)^2

is the goodness-of-fit statistic for the optimal parameters :math:`\theta` and

.. math::
	
	\chi^2_0 = \sum_i \left(\frac{y_i - \bar{y}}{\sigma_i}\right)^2

is the goodness-of-fit statistic for a constant fit, and :math:`\bar{y}` is the weighted mean, 


.. math::

	\bar{y} = \sum_i w_i y_i

where :math:`w_i \propto 1/\sigma_i^2` and :math:`\sum_iw_i = 1`. 

The closed form of the periodogram is given by

.. math::

	P(\omega) = \frac{1}{\chi^2_0}\left(\frac{YC_{\tau}^2}{CC_{\tau}} + \frac{YS_{\tau}^2}{SS_{\tau}}\right)

Where

.. math::
	
	YC_{\tau} &= \sum_i w_iy_i\cos{\omega (t_i - \tau)}\\

	YS_{\tau} &= \sum_i w_iy_i\sin{\omega (t_i - \tau)}\\

	CC_{\tau} &= \sum_i w_i\cos^2{\omega (t_i - \tau)}\\

	SS_{\tau} &= \sum_i w_i\sin^2{\omega (t_i - \tau)}\\

	\tan{2\omega\tau} &= \frac{\sum_i w_i \sin{2\omega t_i}}{\sum_i w_i \cos{2\omega t_i}}

For the original formulation of the Lomb-Scargle periodogram without the constant offset term. 

Adding a constant offset
------------------------

Lomb-Scargle can be extended in many ways, most commonly to include a constant offset [ZK2009]_.

.. math::

	\hat{y}^{\rm GLS}(t|\omega, \theta) = \theta_1\cos{\omega t} + \theta_2\sin{\omega t} + \theta_3

This protects against cases where the mean of the data does not correspond with the mean of the underlying
signal, as is usually the case with sparsely sampled data or for signals with large amplitudes that become
too bright or dim to be observed during part of the signal phase. 

With the constant offset term, the closed-form solution to :math:`P(\omega)` is the same, but the terms
are slightly different. Derivations of this are in [ZK2009]_.

Getting :math:`\mathcal{O}(N\log N)` performance
------------------------------------------------

The secret to Lomb-Scargle's speed lies in the fact that computing it requires evaluating sums that, for regularly-spaced data, can be evaluated with the fast Fourier transform (FFT), which scales as :math:`\mathcal{O}(N_f\log N_f)` where :math:`N_f` is the number of frequencies. For *irregularly* spaced data, however, we can employ tricks to get to this scaling.

1. We can "extirpolate" the data with Legendre polynomials to a regular grid and then perform the FFT [PressRybicki1989]_, or,
2. We can use the non-equispaced fast Fourier transform (NFFT) [DuttRokhlin1993]_, which is tailor made for this exact problem.

The latter was shown by [Leroy2012]_ to give roughly an order-of-magnitude speed improvement over the [PressRybicki1989]_ method, with the added benefit that the NFFT is a rigorous extension of the FFT and has proven error bounds.

It's worth mentioning the [Townsend2010]_ CUDA implementation of Lomb-Scargle, however this uses the :math:`\mathcal{O}(N_{\rm obs}N_f)` "naive" implementation
of LS without any FFT's.

Estimating significance
-----------------------

``cuvarbase`` implements the [Baluev2008]_ analytic upper bound on the
false-alarm probability of a periodogram peak, which accounts for the
effective number of independent frequencies searched without resorting
to bootstrap simulations:

.. code-block:: python

    from cuvarbase.lombscargle import fap_baluev

    # t, dy: observation times and uncertainties
    # z:     the periodogram value of the peak
    # fmax:  the maximum frequency searched
    fap = fap_baluev(t, dy, z, fmax)

:func:`cuvarbase.lombscargle.LombScargleAsyncProcess.batched_run_const_nfreq`
applies the same bound when called with ``only_return_best_freqs=True``,
returning ``(best_freqs, best_freq_faps)``: the frequency of each
lightcurve's best peak and the false-alarm probability of that peak
(``d_K = 2 * nharmonics + 1``). Small is significant; an overwhelming
peak can underflow to exactly ``0.0``. *Changed in 1.0:* earlier
versions returned ``1 - FAP``, which rounds to exactly ``1.0`` for every
FAP below 1e-16 and so could not rank detections. Two caveats: the
bound is one-sided (an upper limit on the false-alarm probability,
tight in the interesting low-FAP regime), and it assumes uncorrelated
Gaussian noise -- correlated ("red") noise or strong aliasing can make
the true false-alarm rate higher than the bound suggests. The FAP is
exponentially sensitive to the peak power (:math:`d\ln{\rm FAP}/dP \sim
-N/2`), so for FAP-grade work on large :math:`f T` grids use
``use_double=True`` (see *Precision* below).

Frequency grids, conventions and precision
------------------------------------------

**Uniform grids only.** Every GPU kernel evaluates the periodogram on
``freqs = df * (k0 + np.arange(nf))`` with an integer ``k0 >= 1`` and
``nf >= 2``; the array you pass only labels the output. ``run``,
``batched_run_const_nfreq`` and ``preallocate`` validate the grid with
:func:`cuvarbase.lombscargle.check_k0` and raise ``ValueError`` naming
the first offending point for anything else -- two concatenated
``arange`` segments, a uniform grid with points removed, ``geomspace``,
or a ``linspace`` whose start is not a multiple of its step. (Before 1.0
only the first two points were inspected and such grids were silently
evaluated on the implied uniform grid.) Build one uniform grid per band
instead; ``cuvarbase.utils.autofrequency`` and
``run(minimum_frequency=..., maximum_frequency=...)`` produce valid
grids. Bands that start far from zero (``fmin >= fmax / 2``, say) are
fine: the NFFT grids are sized from the highest mode used. The NFFT
oversampling factor must be ``sigma >= 3`` (default 4); smaller values
alias the top of every band and are rejected.

**Model conventions.** ``floating_mean=True`` (default) is the
generalized Lomb-Scargle of [ZK2009]_ (astropy's ``fit_mean=True``) and
is what all accuracy statements below refer to. ``floating_mean=False``
is the classic periodogram of the data centred on the *unweighted* mean,
which differs from astropy's ``fit_mean=False, center_data=True`` for
heteroscedastic errors. ``window=True`` returns the spectral window as
the periodogram of ``y = 1`` in the classic normalization, which is
**4x** astropy's ``LombScargle(t, ones, fit_mean=False,
center_data=False)``. ``nharmonics > 1`` (the multiharmonic GLS) is
floating-mean only and is honoured on every path: the NFFT path solves
the small per-frequency system on the host from the GPU spectra, and
``use_fft=False`` / ``python_dir_sums=True`` run float64 direct sums on
the host (correct but O(N nf)). ``amplitude_prior`` is the standard
deviation of a Gaussian prior on the harmonic amplitudes (a ridge term
``1 / amplitude_prior**2``) and is applied on every path. ``dy=None``
gives unit weights.

**The -1 sentinel.** A power of exactly ``-1`` is the kernels' marker
for a non-finite or negative value at that frequency. **It should not
occur.** Since 1.0 every entry point validates the light curve before
any GPU work (:func:`cuvarbase.utils.check_lightcurve`), so the inputs
that used to fill a whole periodogram with ``-1`` -- non-finite ``y``
or ``dy``, ``dy = 0``, mismatched array lengths, fewer than four
observations -- raise ``ValueError`` instead. The kernel branch is
kept as a last-resort guard against a degenerate grid (e.g.
all-identical ``t``); a ``-1`` in a returned periodogram is a bug
report, not a valid power.

**Precision.** The default float32 pipeline agrees with the exact
float64 generalized Lomb-Scargle to about 1e-4 in power for
:math:`f T \lesssim 10^4` (e.g. 300 points over a year to 20 cycles/day)
and to about 1e-3 at survey scale (:math:`f T \sim 10^5`--:math:`10^6`,
ten-year baselines to 50 cycles/day); the limit is the float32 storage
of the (epoch-subtracted) times. ``use_double=True`` reaches ~1e-7 and
is recommended whenever the *value* of the power matters -- false-alarm
probabilities, amplitude estimates -- rather than the location of the
peak, which float32 recovers identically in all tests. Times are
mean-centred on the host in float64 before any cast, so absolute (BJD)
timestamps are safe.


Example: Basic
--------------

.. plot::
	:include-source:

	import cuvarbase.lombscargle as gls
	import numpy as np
	import matplotlib.pyplot as plt


	t = np.sort(np.random.rand(300))
	y = 1 + np.cos(2 * np.pi * 100 * t - 0.1)
	dy = 0.1 * np.ones_like(y)
	y += dy * np.random.randn(len(t))

	# Set up LombScargleAsyncProcess (compilation, etc.)
	proc = gls.LombScargleAsyncProcess()

	# Run on single lightcurve
	result = proc.run([(t, y, dy)])

	# Synchronize all cuda streams
	proc.finish()

	# Read result!
	freqs, ls_power = result[0]

	############
	# Plotting #
	############

	f, ax = plt.subplots()
	ax.set_xscale('log')

	ax.plot(freqs, ls_power)
	ax.set_xlabel('Frequency')
	ax.set_ylabel('Lomb-Scargle')
	plt.show()

Example: Batches of lightcurves
-------------------------------


.. plot::
	:include-source:

	import cuvarbase.lombscargle as gls
	import numpy as np
	import matplotlib.pyplot as plt

	nlcs = 9

	def lightcurve(freq=100, ndata=300):
		t = np.sort(np.random.rand(ndata))
		y = 1 + np.cos(2 * np.pi * freq * t - 0.1)
		dy = 0.1 * np.ones_like(y)
		y += dy * np.random.randn(len(t))
		return t, y, dy

	freqs = 200 * np.random.rand(nlcs)
	data = [lightcurve(freq=freq) for freq in freqs]

	# Set up LombScargleAsyncProcess (compilation, etc.)
	proc = gls.LombScargleAsyncProcess()

	# Run on batch of lightcurves
	results = proc.batched_run_const_nfreq(data)

	# Synchronize all cuda streams
	proc.finish()

	############
	# Plotting #
	############
	max_n_cols = 4
	ncols = max([1, min([int(np.sqrt(nlcs)), max_n_cols])])
	nrows = int(np.ceil(float(nlcs) / ncols))
	f, axes = plt.subplots(nrows, ncols,
	                       figsize=(3 * ncols, 3 * nrows))

	for (frqs, ls_power), ax, freq in zip(results,
	                                      np.ravel(axes),
	                                      freqs):
		ax.set_xscale('log')
		ax.plot(frqs, ls_power)
		ax.axvline(freq, ls=':', color='r')

	f.text(0.05, 0.5, "Lomb-Scargle", rotation=90, 
	       va='center', ha='right', fontsize=20)
	f.text(0.5, 0.05, "Frequency", 
	       va='top', ha='center', fontsize=20)


	for i, ax in enumerate(np.ravel(axes)):
		if i >= nlcs:
			ax.axis('off')
	f.tight_layout()
	f.subplots_adjust(left=0.1, bottom=0.1)
	plt.show()


.. [DuttRokhlin1993] `Dutt, A., & Rokhlin, V. 1993, SIAM J. Sci. Comput., 14(6), 1368–1393. <http://epubs.siam.org/doi/abs/10.1137/0914081>`_
.. [PressRybicki1989] `Press, W. H., & Rybicki, G. B. 1989, ApJ, 338, 277 <http://adsabs.harvard.edu/abs/1989ApJ...338..277P>`_
.. [Baluev2008] `Baluev, R. V. 2008, MNRAS, 385, 1279 <http://adsabs.harvard.edu/abs/2008MNRAS.385.1279B>`_
.. [ZK2009] `Zechmeister, M., & Kürster, M. 2009, AAP, 496, 577 <http://adsabs.harvard.edu/abs/2009A%26A...496..577Z>`_
.. [VanderPlas2017] `VanderPlas, J. T. 2017, arXiv:1703.09824 <http://adsabs.harvard.edu/abs/2017arXiv170309824V>`_
.. [Leroy2012] `Leroy, B. 2012, AAP, 545, A50 <http://adsabs.harvard.edu/abs/2012A%26A...545A..50L>`_
.. [Townsend2010] `Townsend, R. H. D. 2010, ApJS, 191, 247 <http://adsabs.harvard.edu/abs/2010ApJS..191..247T>`_
.. [Barning1963] `Barning, F. J. M. 1963, BAN, 17, 22 <http://adsabs.harvard.edu/abs/1963BAN....17...22B>`_
.. [Vanicek1969] `Vaníček, P. 1969, APSS, 4, 387 <http://adsabs.harvard.edu/abs/1969Ap&SS...4..387V>`_
.. [Scargle1982] `Scargle, J. D. 1982, ApJ, 263, 835 <http://adsabs.harvard.edu/abs/1982ApJ...263..835S>`_
.. [Lomb1976] `Lomb, N. R. 1976, APSS, 39, 447 <http://adsabs.harvard.edu/abs/1976Ap%26SS..39..447L>`_

Power-spectrum convention
-------------------------

The GPU Lomb-Scargle returns the standard normalized periodogram

.. math::

    P(f) = 1 - \chi^2(f) / \chi^2_0

(equivalently the ``normalization='standard'`` convention of
``astropy.timeseries.LombScargle``), where :math:`\chi^2(f)` is the
best-fit sinusoid's weighted residual sum and :math:`\chi^2_0` that of
the constant model. With ``floating_mean=True`` (the default) this is
the *generalized* (floating-mean) Lomb-Scargle of Zechmeister &
Kürster (2009). Values are directly comparable to astropy's defaults;
see the unit tests (``test_lombscargle.py``) which assert agreement.
