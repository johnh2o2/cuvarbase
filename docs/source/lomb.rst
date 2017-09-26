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

	\tan{2\omega\tau} &= \frac{\sum_i w_i \sin{2\omega t_i}}{\sum_i w_i \sin{2\omega t_i}}

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

See [Baluev2008]_ for more information (TODO.)


Example: Basic
--------------

.. plot::
	:include-source:

	import skcuda.fft
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

	import skcuda.fft
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