What's new in cuvarbase
***********************
* **1.0.0**
    * First major release. Supersedes the unreleased internal 0.4.0 (below); everything since the last PyPI release (0.2.6) ships here.
    * **BLS**
        * Optimized kernel variant (``bls_optimized.cu``) with bank-conflict fixes and warp shuffles; ``eebls_gpu_fast_optimized()`` and ``eebls_gpu_fast_adaptive()`` (automatic block sizing — 1.4-5.3x on realistic grids, larger gains for very small lightcurves)
        * Thread-safe kernel caching with LRU eviction
        * Sparse BLS (Panahi & Zucker 2021) on GPU and CPU, with ground-truth correctness tests; ``eebls_transit`` auto-selects sparse vs standard BLS by dataset size
        * ``sparse_bls_cpu`` vectorized with prefix sums (the previous pure-Python pair loop recomputed slice sums, O(N³) — minutes per frequency at the ndata=500 sparse threshold; now ~3 ms)
        * Multi-lightcurve batch mode: ``eebls_gpu_batch()`` + ``BLSBatchMemory`` (best for ndata < ~1000 per lightcurve)
        * Keplerian frequency grids: ``cuvarbase.bls_frequencies.keplerian_freq_grid()`` — 4-37x fewer frequencies than uniform grids at survey baselines
        * Fixed ``mod1_fast`` integer overflow for t*f >= 2^31 (corrupted phases on long-baseline data)
        * **Fixed silent accuracy loss for absolute timestamps (e.g. BJD ~2.45e6 days):** all BLS paths now subtract ``min(t)`` in float64 before casting times to float32; previously the float32 phase fold lost nearly all phase information at BJD scale. **Convention change:** reported ``phi0`` solutions are now relative to ``min(t)``
        * Fixed ``reduction_max`` in the optimized kernel silently dropping half the per-block candidates (``use_optimized=True`` paths)
        * Fixed ``eebls_transit`` sparse path crashing with TypeError on documented kwargs (rho, samples_per_peak, ...); it now also warns that the sparse search ignores qmin_fac/qmax_fac
        * ``compile_bls`` validates block_size (power of 2, >= 32) and raises a clear error when no requested kernel functions are loadable; ``_reduction_max`` now applies the same validation (its old power-of-two assert was always true under Python 3 division)
    * **Lomb-Scargle / NFFT**
        * Memory classes refactored into ``cuvarbase.memory`` (behavior-preserving)
        * Optional cuFINUFFT backend (``use_cufinufft=True``) as a cross-check; the custom NFFT kernel remains the default. cufinufft Plans are now cached per problem shape (creation dominated the per-call cost, making the backend 0.63-0.84x the custom kernel's speed); ``free_plan_cache()`` releases the cached GPU resources
        * Fixed ``lomb_scargle_simple`` double-applying inverse-variance weights (largest-error points previously got the most weight)
        * Fixed ``fap_baluev`` returning exactly 0 for significant peaks (issue #14): the false-alarm probability is now evaluated in log space with ``expm1``, staying positive down to the float64 limit instead of underflowing at FAP ≲ 1e-16
        * Fixed ``lomb_scargle_async`` (direct-sums branch) gating the device→host result copy on ``transfer_to_device`` instead of ``transfer_to_host``: callers with data already on the GPU got a stale/empty periodogram back, and the copy could not be suppressed
        * ``lomb_scargle_async(use_cufinufft=True)`` now raises ImportError when cufinufft is not installed instead of silently running the custom NFFT path
        * Improved ``memory_requirement`` estimation (PR #59; fixes the previous NameError and now accounts for cuFFT work areas and per-batch buffers)
        * Lightcurves are normalized (mean-subtracted ``t`` and ``y``) before processing for numerical stability (PRs #57/#60)
    * **PDM** (community contribution by @astrobatty — PR #62)
        * Fast shared-memory CUDA kernels for all four variants: ``binned_step_fast``, ``binned_linterp_fast``, ``binless_tophat_fast``, ``binless_gauss_fast``
        * Backward-compatible ``(t, y, err)`` input API for ``PDMAsyncProcess.run()`` with automatic frequency grids; the legacy ``(t, y, w, freqs)`` format is deprecated (emits DeprecationWarning)
        * Unit tests for all kernel variants and new Sphinx documentation (``docs/source/pdm.rst``)
        * Fixed the CPU reference functions (``binless_pdm_cpu``, ``pdm2_cpu``, ``pdm2_single_freq``) mutating the caller's ``t``/``y`` arrays in place
    * **Conditional Entropy** (community contribution — PR #61)
        * Optional log-probability periodogram via ``compute_log_prob=True``
        * Lightcurves normalized before processing; 32-bit overflow guard for large ``nfreq x ndata`` runs; clear error for the unsupported ``use_fast`` + ``weighted`` combination
        * CE is now in **maintenance mode**: it keeps working, but no new development is planned — for an actively developed GPU CE/AOV search see `periodfind <https://github.com/scope-ml/periodfind>`_
    * **Experimental** (UserWarning on import; not recommended for science use yet)
        * GPU Transit Least Squares (``cuvarbase.tls``) with Ofir (2014) period grids
        * TLS epoch (t0) grid is now duration-scaled (stride = duration / 3, floor 30, cap 20,000 epochs): the previous fixed 30-epoch grid missed transits narrower than ~1/30 of the period entirely, which broke Keplerian-mode searches for most periods > ~3.5 d. Mirrored in ``tls_grids.t0_grid_size()``
        * Removed the TLS kernels' bitonic phase sort: it was incomplete for non-power-of-2 sizes and its output order was never consumed — pure wasted per-period work; results are unchanged
        * Added golden accuracy tests against the reference ``transitleastsquares`` package (``test_tls_golden.py``)
        * TLS hardening: ``tls_search_gpu`` now raises ValueError when the shared-memory layout exceeds the 48 KB budget (~3,500 points) instead of failing at kernel launch; failed trial periods (1e30 chi2 sentinel) are masked out of the best-fit search and SDE/FAP statistics (previously they collapsed SDE and drove FAP to 1); ``signal_to_noise`` no longer inflates by sqrt(n_transits); ``false_alarm_probability``'s heuristic is no longer misattributed to Hippke & Heller (2019); batman template failures now warn instead of silently substituting a trapezoid
        * NUFFT-LRT matched filter (contributed by Jamila Taaki) — **removed from the released package**: the implementation computed on the CPU (its CUDA kernels were compiled but never invoked) and silently ignored data beyond ``median(dt) * nf`` from the first observation, truncating multi-season baselines. Source preserved on the ``feature/nufft-lrt-experimental`` branch pending a GPU rewire
    * **Packaging / infrastructure**
        * **BREAKING:** requires Python 3.9+
        * Fixed wheel/sdist omitting the ``base``/``memory`` subpackages (pip installs of the v1.0 branch were unimportable)
        * Lazy module imports: ``import cuvarbase`` and BLS/CE/PDM no longer require scikit-cuda; a numpy>=1.24 compatibility shim is applied automatically before skcuda loads
        * Fixed CUDA kernel lookup crashing for editable installs (``pip install -e .``) on Python < 3.12 when cuvarbase is imported from outside the source tree; kernel paths now resolve relative to the package directory
        * GitHub Actions CI: CPU test suite (108 tests; GPU tests stubbed/skipped) on Python 3.9-3.12 + build-wheel-install-import packaging check; flake8 error class enforced
        * Root ``conftest.py`` stubs pycuda/skcuda so the suite runs on GPU-less machines
        * Removed vestigial ``cuvarbase.periodograms`` scaffolding
        * Benchmark suite (``scripts/benchmark_*.py``) and multi-GPU results in ``docs/BENCHMARK_RESULTS.md``
    * **Docs**
        * Performance claims re-grounded in measured data (257-354x vs astropy BoxLeastSquares across 7 GPU architectures for standard BLS; honest small-problem caveats for LS)
        * Corrected the nifty-ls reference to Garrison et al. (arXiv:2409.08090)

* **0.4.0** *(never released — folded into 1.0.0)*
    * **BREAKING CHANGE:** Dropped Python 2.7 support - now requires Python 3.9+ (importlib.resources.files)
    * Removed ``future`` package dependency and all Python 2 compatibility code
    * Modernized codebase: removed ``__future__`` imports and ``builtins`` compatibility layer
    * Updated minimum dependency versions: numpy>=1.17, scipy>=1.3
    * Added modern Python packaging with ``pyproject.toml``
    * Added Docker support for easier installation with CUDA 11.8
    * Added GitHub Actions CI: CPU test suite on Python 3.9-3.12 + packaging smoke test (GPU validation remains manual)
    * Updated classifiers to reflect Python 3.9-3.12 support
    * Cleaner, more maintainable codebase (89 lines of compatibility code removed)
    * Includes all features from 0.2.6:
        * Added Sparse BLS implementation for efficient transit detection with small datasets
        * New ``sparse_bls_cpu`` function that avoids binning and grid searching
        * New ``eebls_transit`` wrapper that automatically selects between sparse (CPU) and standard (GPU) BLS
        * Based on algorithm from Panahi & Zucker 2021 (https://arxiv.org/abs/2103.06193)
        * More efficient for datasets with < 500 observations
        * NUFFT LRT implementation for transit detection
        * Refactored codebase organization with base/, memory/, and periodograms/ modules

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

