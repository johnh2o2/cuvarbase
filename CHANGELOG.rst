What's new in cuvarbase
***********************
* **0.2.6** (In Development)
    * Added Sparse BLS implementation for efficient transit detection with small datasets
        * New ``sparse_bls_cpu`` function that avoids binning and grid searching by testing all pairs of observations
        * New ``eebls_transit`` wrapper that automatically selects between sparse (CPU) and standard (GPU) BLS based on dataset size
        * Based on algorithm from Burdge et al. 2021 (https://arxiv.org/abs/2103.06193)
        * More efficient for datasets with < 500 observations
        * Default threshold is 500 observations (configurable with ``sparse_threshold`` parameter)

* **0.2.5**
    * swap out pycuda.autoinit for pycuda.autoprimaryctx to handle "cuFuncSetBlockShape" error
    
* **0.2.4**
    * bugfix for pytest (broke b/c of incorrect fixture usage)
    * added ``ignore_negative_delta_sols`` option to BLS to ignore inverted dips in the lightcurve

* **0.2.1**
    * bugfix for memory leak in BLS
    * contact email changed in setup

* **0.2.0**
	* Many more unit tests for BLS and CE.
	* BLS
		* Now several orders of magnitude faster! Use ``use_fast=True`` in ``eebls_transit_gpu`` or use ``eebls_gpu_fast``.
		* Bug-fix for boost-python error when calling ``eebls_gpu_fast``.
  	* CE
		* New ``use_fast`` parameter in ``ConditionalEntropyAsyncProcess``; if selected will use a kernel that should be substantially more efficient and that requires no memory overhead. If selected, you should use the ``run`` function and not the ``large_run`` function. Currently the ``weighted`` option is not supported when ``use_fast`` is ``True``.
		* Bug-fix for ``mag_overlap > 0``.

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

