What's new in cuvarbase
***********************

* **0.1.9**

  * Added Sphinx documentation
  * **Now Python 3 compatible!**
  * Miscillaneous bug fixes
  * CE
	* Run functions for ``ConditionalEntropyAsyncProcess`` now allow for a ``balanced_magbins`` argument to set the magnitude bins to have widths that vary with the distribution of magnitude values. This is more robust to outliers, but performance comparisons between the usual CE algorithm indicate that you should use care.
	* Added ``precompute`` function to ``ConditionalEntropyAsyncProcess`` that allows you to speed up computations without resorting to the ``batched_run_constant_nfreq`` function. Currently it still assumes that the frequencies used will be the same for all lightcurves.
  * GLS
	* Added ``precompute`` function to ``LombScargleAsyncProcess``.
	* Avoids allocating GPU memory for NFFT when ``use_fft`` is ``False``.
	* ``LombScargleAsyncProcess.memory_requirement`` is now implemented.
  * BLS
	* ``eebls_gpu``, ``eebls_transit_gpu``, and ``eebls_custom_gpu`` now have a ``max_memory`` option that allows you to automatically set the ``batch_size`` without worrying about memory allocation errors.
	* ``eebls_transit_gpu`` now allows for a ``freqs`` argument and a ``qvals`` argument for customizing the frequencies and the fiducial ``q`` values
	* Fixed a small bug in ``fmin_transit`` that miscalculated the minimum frequency.

* **0.1.8**
    * Removed gamma function usage from baluev 2008 false alarm probability (``use_gamma=True`` will override this)
    * Fixed a bug in the GLS notebook

* **0.1.6/0.1.7**
    * Some bug fixes for GLS
    * ``large_run`` function for Conditional Entropy period finder allows large frequency grids
      without raising memory allocation errors.
    * More unit tests for conditional entropy
    * Conditional entropy now supports double precision with the ``use_double`` argument

* **0.1.5**
	* Conditional Entropy period finder now unit tested
		* Weighted variant also implemented -- accounts for heteroskedasticity if
		  that's important
	* BLS
		* New unit tests
		* A new transiting exoplanet BLS function: ``eebls_transit_gpu``
			* Only searches plausible parameter space for Keplerian orbit
	* GLS
		* False alarm probability: ``fap_baluev``
			* Implements `Baluev 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.385.1279B>`_ false alarm probability measure based on extreme value theory

