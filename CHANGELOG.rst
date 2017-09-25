Change log
----------
* **0.1.9**
	* Added Sphinx docuemntation
	* CE now allows for a ``balanced_magbins`` argument to 
	

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

