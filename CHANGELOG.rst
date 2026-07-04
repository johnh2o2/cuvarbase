What's new in cuvarbase
***********************
* **1.0.0**
    * First major release, and the first release published to PyPI since 0.2.5 (2023). Supersedes the unreleased internal 0.4.0 and the tagged-but-never-published 0.2.6 (below); everything since 0.2.5 ships here.
    * Measured head-to-head against the previous cuvarbase on an RTX A5000 (raw data in ``benchmarks/results/v026_head_to_head_jul2026/``): steady-state kernel throughput is unchanged, but real pipelines are much faster — the previous release rebuilt its CUDA module on *every* call (~0.25-0.4 s), so a call-per-lightcurve loop runs **34x faster** in 1.0.0 (kernel caching), a 100-lightcurve run ~10x; survey-scale Lomb-Scargle is 2.85x faster; and BLS on BJD-scale timestamps now actually works (the old float32 fold silently lost the transit)
    * **BLS**
        * Optimized kernel variant (``bls_optimized.cu``) with bank-conflict fixes and warp shuffles; ``eebls_gpu_fast_optimized()`` and ``eebls_gpu_fast_adaptive()`` (automatic block sizing; the v1.0 re-benchmark with warm kernel cache measures ~1.0-1.3x over fixed blocks — earlier 1.4-5.3x gains were dominated by per-call kernel handling that the cache now amortizes)
        * Thread-safe kernel caching with LRU eviction
        * Selectable power conventions (issue #17): all BLS entry points accept ``convention=`` ('chi2ratio' default, 'snr', 'loglik') and ``convert_bls_power()`` converts standalone periodograms. 'snr' equals astropy's ``objective='snr'`` power at the same solution; 'loglik' is the log-likelihood gain over the constant weighted-mean model (astropy's ``objective='likelihood'`` equals it divided by 1 - r, r = in-transit weight fraction) — both relations verified against astropy in the test suite
        * ``eebls_gpu_fast`` (and ``_optimized``/``_adaptive``): the ``noverlap`` parameter is now honored — the periodogram is the elementwise max over ``noverlap`` passes with the phase-bin grid shifted by ``1/noverlap`` of the finest bin between passes. Previously ``noverlap`` was silently ignored on the fast path (its docstring recommended a manual ``dphi`` re-run workaround, now removed). Runtime scales linearly with ``noverlap`` (default 2); pass ``noverlap=1`` for the old single-pass behavior
        * Sparse BLS (Panahi & Zucker 2021) on GPU and CPU, with ground-truth correctness tests; ``eebls_transit`` auto-selects sparse vs standard BLS by dataset size. The sparse path (kernels + CPU) honors per-frequency ``qmin``/``qmax`` duration bounds, and ``eebls_transit`` passes its Keplerian ``qmin_fac``/``qmax_fac`` constraints through, so results are comparable across the ``sparse_threshold`` boundary. **BREAKING:** all ``sparse_bls_cpu``/``sparse_bls_gpu`` arguments after ``freqs`` are keyword-only — pre-1.0 positional calls (e.g. passing ``ignore_negative_delta_sols`` positionally) would have silently landed on the new ``qmin`` parameter and returned an all-zero periodogram; they now raise TypeError. Bound values are validated (finite, ``qmin >= 0``, ``qmax > 0``, ``qmin <= qmax``) instead of silently rejecting every candidate box
        * ``sparse_bls_cpu`` vectorized with prefix sums (the previous pure-Python pair loop recomputed slice sums, O(N³) — minutes per frequency at the ndata=500 sparse threshold; now ~3 ms)
        * Multi-lightcurve batch mode: ``eebls_gpu_batch()`` + ``BLSBatchMemory``
        * **Fixed nondeterministic bogus BLS peaks from degenerate all-weight boxes (Jul 2026, root cause of the instability reported in PR #65):** the ``bls_value`` upper bound ``w < 1.f - 1e-10f`` was a float32 no-op (1e-10 underflows against 1.0f), so a trial box capturing all the statistical weight — routine for single-site data near cycles-per-day aliases with wide boxes — passed the guard with ``1 - w`` equal to atomicAdd-roundoff noise and ``ybar`` roundoff around zero, producing run-to-run-varying spurious power (``sparse_bls.cu``'s ``MAX_W_COMPLEMENT = 1e-9`` had the same underflow). The bound is now a float32-meaningful ``1e-4`` complement across ``bls_common.cuh``/``bls_batch.cu``/``sparse_bls.cu`` and the CPU mirrors (``single_bls`` returned a literal NaN on an all-weight box; ``sparse_bls_cpu`` uses the same complement for GPU/CPU parity). Regression tests cover the deterministic CPU case, repeat-stability on single-site data, and 500 ppm shallow-transit recovery (guarding against absolute-amplitude thresholds as an alternative "fix")
        * **Fixed two ``eebls_gpu_batch`` defects (Jul 2026):** (a) the batch path was single-pass while the fast/adaptive paths do ``noverlap`` phase-shifted passes (the batch kernel's ``noverlap`` argument was a silent no-op, like the single-LC fast kernels' before this release) — the batch periodogram diverged from ``eebls_gpu_fast`` at small ndata (corr 0.77 at ndata=200); it now runs the same host-side multi-pass + elementwise max and matches at corr>0.999 with identical peaks. (b) The batch kernel was recompiled on every call (~0.6–0.9 s vs 2–10 ms of kernel work) — the entire "~12x slower at TESS scale" regression; it now goes through the same LRU kernel cache as the single-LC paths, and with a warm cache batch beats a single-LC ``eebls_gpu_fast`` loop at every measured scale (~10x at ndata=200, ~5x at ndata=20,000; RTX A5000). The large-ndata inefficiency UserWarning is retired
        * Fixed ``convention='snr'``/``'loglik'`` scaling on the fast path's memory-reuse pattern: ``BLSMemory`` now records the :math:`\\chi^2_0` of the data loaded at ``setdata`` time and the conversion uses it, so calls that reuse a preloaded memory (``transfer_to_device=False``) while passing different ``y``/``dy`` arguments no longer scale the power by the wrong null model
        * Keplerian frequency grids: ``cuvarbase.bls_frequencies.keplerian_freq_grid()`` — 4-37x fewer frequencies than uniform grids at survey baselines; ``return_qvals=True`` also returns the per-frequency Keplerian duration fraction, which ``eebls_gpu_batch`` accepts as array ``qmin``/``qmax`` for duration-constrained batch searches
        * Fixed ``mod1_fast`` integer overflow for t*f >= 2^31 (corrupted phases on long-baseline data)
        * **Fixed silent accuracy loss for absolute timestamps (e.g. BJD ~2.45e6 days):** all BLS paths now subtract ``floor(min(t))`` in float64 before casting times to float32; previously the float32 phase fold lost nearly all phase information at BJD scale. **Convention:** ``phi0`` phases (both reported solutions and inputs to ``single_bls``/``eebls_gpu_custom``/``hone_solution``) are in the ORIGINAL input timescale — internally phases are folded relative to ``floor(min(t))`` and re-referenced as ``(phi ± epoch*freq) % 1`` in float64 (PR #65, @astrobatty). An earlier iteration reported phases relative to ``floor(min(t))`` itself
        * **Fixed a float32 fold-order precision loss in ``single_bls``** (exposed by PR #65's non-zero-epoch tests): the reference folded as ``(t*f - phi0) mod 1``, subtracting at magnitude ``t*f`` where float32 resolution is only ``ulp(t*f)/2`` (~1.5e-5 phase for a 1-yr baseline, ~2.4e-4 for 10 yr), so points within that fuzz of a box edge could get the wrong membership relative to the GPU kernels, which wrap into [0, 1) *before* binning (~1e-7 resolution; hardware-probed: nvcc does not FMA-contract the kernels' ``mod1(t*f)``, so wrap-first is bit-identical to the kernel fold). ``single_bls`` now wraps first. Also: ``bin_and_phase_fold_custom`` folds with the float32-cast frequency (double freqs are used only for epoch re-referencing) and takes float64 ``phi_values`` so its epoch conversion matches ``single_bls`` bit for bit, and ``sparse_bls_cpu``/``sparse_bls_gpu`` re-reference solution phases with the caller's float64 frequencies (the float32 copies put phases off by ``epoch*|f64-f32|``, up to ~0.07 cycles at BJD epochs). ``eebls_transit_gpu`` now always returns a 3-tuple (``sols=None`` on the fast/optimized paths) and ``eebls_transit(use_optimized=True)`` respects an explicit ``block_size``
        * Fixed ``reduction_max`` in the optimized kernel silently dropping half the per-block candidates (``use_optimized=True`` paths)
        * Fixed ``eebls_transit`` sparse path crashing with TypeError on documented kwargs (rho, samples_per_peak, ...)
        * ``compile_bls`` validates block_size (power of 2, >= 32) and raises a clear error when no requested kernel functions are loadable; ``_reduction_max`` now applies the same validation (its old power-of-two assert was always true under Python 3 division)
    * **Lomb-Scargle / NFFT**
        * Multiharmonic generalized Lomb-Scargle on GPU (``LombScargleAsyncProcess(nharmonics=H)`` for ``H>1`` no longer raises ``NotImplementedError``). The GPU NFFT already produces the weight spectrum to 2H harmonics and the ``w*(y-ybar)`` spectrum to H; the per-frequency 2H x 2H generalized-LS solve runs on the host in float64 (reusing the tested ``mhdirect_sums``/``mhgls_from_sums`` math), which matches the ``lomb_scargle_direct_sums`` reference to machine precision for H=2,3 in CPU tests. Suited to occasional multiharmonic searches rather than survey-scale throughput
        * **Dropped the abandoned ``scikit-cuda`` dependency** (`issue #63 <https://github.com/johnh2o2/cuvarbase/issues/63>`_): the cuFFT calls (the only thing scikit-cuda 0.5.3 was used for) now go through a minimal in-house ``ctypes`` binding, ``cuvarbase._cufft`` (Plan/fft/ifft/cufftEstimate1d, lazily loaded). No cuvarbase module imports scikit-cuda anymore, and its numpy>=1.24 compatibility shim is gone. Validated on an RTX A5000: full LS/NFFT suite green, FFT matches scipy, and the binding is within ~2% of the old scikit-cuda cuFFT performance (both call ``cufftExecC2C``)
        * Memory classes refactored into ``cuvarbase.memory`` (behavior-preserving)
        * ``NFFTAsyncProcess.estimate_m``/``get_m`` now implement the rigorous L1-norm *truncation* bound (NFFT3 guide p. 11: ``max|E| <= 4 exp(-m pi (1 - 1/(2 sigma - 1))) ||y||_1``) when the data is available — with ``autoset_m=True`` the filter radius is the smallest ``m`` whose truncation-error bound meets the requested tolerance, replacing the jakevdp/nfft ``N``-based heuristic (which guaranteed the tolerance only for ``max|y| <= 1``; it remains the fallback when ``m`` is sized before the data is seen, e.g. the Lomb-Scargle buffer layouts). Resolves the package's only TODO. In double precision the realized error tracks this bound down to ~1e-10 absolute (A5000-validated); in single precision a genuine ~1e-3 absolute floor remains (float32 trig on large phase arguments) — use ``use_double=True`` for tolerances below ~1e-2
        * **Fixed a float32 ``PI`` literal in ``cunfft.cu``'s phase-factor kernels** (``nfft_shift``/``normalize``): its 2.8e-8 relative error, multiplied by un-reduced phase arguments up to ``2*pi*|k0|`` and amplified by the Gaussian deconvolution, imposed an m-independent ~1e-3 absolute error floor on the NFFT *even in double precision* (an earlier note here described that floor as inherent — it was this bug). After the fix the float64 NFFT error follows the truncation bound over 9 decades (m=12 reference config: 3.4e-3 → 1.2e-10); float32 behavior is unchanged. Also typed the ``modflt``/``diffmod`` device helpers with ``FLT`` (they hardcoded float32 in double mode)
        * NUFFT-LRT ``compute_nufft`` docstring/pipeline-test mock corrected to the transform's actual phase convention (``exp(2*pi*i*f_k*t)`` with absolute ``t``, not ``t - min(t)``; device-verified at corr=1.0 vs the exact adjoint DFT). The matched filter is unaffected — data and template share the transform, so the common phase cancels
        * ``batched_run_const_nfreq``'s ``batch_size>1`` "multi-stream overhead" diagnosed (Jul 2026): the method builds ``batch_size`` memory sets (pinned buffers + cuFFT plan each) on every call while a single survey-scale periodogram already saturates the GPU, so the setup cost scales with ``batch_size`` with little compute to gain. Amortized over large calls, ``batch_size=4`` is ~10% faster per lightcurve than 1; the default stays 1 and the docstring now carries the guidance
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
        * Batch APIs (issue #33): ``PDMAsyncProcess.batched_run_const_nfreq`` processes a lightcurve collection in memory-bounded chunks that share one frequency grid (peak GPU memory scales with ``batch_size``, not the number of lightcurves), and ``large_run`` auto-picks ``batch_size`` from the free GPU memory. A ``scripts/benchmark_pdm.py`` GPU-vs-CPU benchmark + correctness check was added
        * Fixed the CPU reference functions (``binless_pdm_cpu``, ``pdm2_cpu``, ``pdm2_single_freq``) mutating the caller's ``t``/``y`` arrays in place
    * **Conditional Entropy** (community contribution — PR #61)
        * Optional log-probability periodogram via ``compute_log_prob=True``
        * Lightcurves normalized before processing; 32-bit overflow guard for large ``nfreq x ndata`` runs; clear error for the unsupported ``use_fast`` + ``weighted`` combination
        * CE is now in **maintenance mode**: it keeps working, but no new development is planned — for an actively developed GPU CE/AOV search see `periodfind <https://github.com/scope-ml/periodfind>`_
    * **Experimental** (UserWarning on import; not recommended for science use yet)
        * GPU Transit Least Squares (``cuvarbase.tls``) with Ofir (2014) period grids
        * TLS epoch (t0) grid is now duration-scaled (stride = duration / oversample, floor 30, cap 20,000 epochs): the previous fixed 30-epoch grid missed transits narrower than ~1/30 of the period entirely, which broke Keplerian-mode searches for most periods > ~3.5 d. The oversample factor is caller-tunable via ``t0_oversample`` on ``tls_search``/``tls_search_gpu``/``compile_tls`` (default 3.0, favoring speed; the reference ``transitleastsquares`` steps ~33x finer — raise it for sensitivity-critical searches). Mirrored in ``tls_grids.t0_grid_size()``
        * Removed the TLS kernels' bitonic phase sort: it was incomplete for non-power-of-2 sizes and its output order was never consumed — pure wasted per-period work; results are unchanged
        * Added golden accuracy tests against the reference ``transitleastsquares`` package (``test_tls_golden.py``)
        * TLS hardening: ``tls_search_gpu`` now raises ValueError when the shared-memory layout exceeds the 48 KB budget (~3,500 points) instead of failing at kernel launch; failed trial periods (1e30 chi2 sentinel) are masked out of the best-fit search and SDE/FAP statistics (previously they collapsed SDE and drove FAP to 1); ``signal_to_noise`` no longer inflates by sqrt(n_transits); ``false_alarm_probability``'s heuristic is no longer misattributed to Hippke & Heller (2019); batman template failures now warn instead of silently substituting a trapezoid
        * NUFFT-LRT matched filter (``cuvarbase.nufft_lrt``, contributed by **Jamila Taaki** / @xiaziyna) — **reinstated** with a GPU rewire. The data and each transit template are now transformed with the GPU adjoint NFFT (``NFFTAsyncProcess``), which takes the raw non-uniform times directly over the full baseline — fixing both defects that got it cut (the earlier path computed a uniform-grid RFFT on the host, never invoking the GPU, and its ``median(dt)*nf`` grid silently truncated multi-season/gappy data). The per-template matched-filter combination still runs on the host. CPU tests verify the rewired pipeline is sensitive to data across the full baseline; it remains EXPERIMENTAL pending a full injection-recovery validation
    * **Known limitations and deferred work**
        * No benchmark against CETRA (the PLATO mission's GPU transit-detection code) exists yet, so cuvarbase makes **no comparative performance claims** against GPU transit searches; the published comparisons cover astropy, nifty-ls, and the CPU fBLS numbers only
    * **Packaging / infrastructure**
        * **BREAKING:** requires Python 3.9+
        * Lazy CUDA context: ``import cuvarbase`` no longer creates a CUDA context or requires a GPU. The eager ``import pycuda.autoprimaryctx`` (which retained+pushed the primary context at package import) is gone; the context is now retained on first GPU use via ``cuvarbase.base.ensure_context`` — wired into every kernel-compile function, ``GPUAsyncProcess.__init__``, and each ``*Memory`` class's ``__init__``. ``import cuvarbase`` and the CPU-only helpers (``sparse_bls_cpu``, ``single_bls``, ``fap_baluev``) therefore run on GPU-less machines. The ``pycuda`` package remains an import dependency of the GPU modules (they ``import pycuda.driver``), but importing them allocates no context. ``CUDA_DEVICE`` is now read at first GPU use rather than at import. The packaging smoke test proves the GPU-less import (pycuda absent)
        * True pinned host buffers: host transfer arrays in every ``*Memory`` class (BLS, batch BLS, NFFT, Lomb-Scargle, Conditional Entropy, TLS) and the PDM result buffer are now page-locked (pinned) by default via ``cuvarbase.memory._host.host_array``, so ``set_async``/``get_async`` host<->device copies overlap with computation instead of staging through a synchronous bounce buffer. If pinning fails (e.g. the OS locked-memory limit is hit) it warns once and falls back to page-aligned memory; pass ``pinned=False`` to opt out. Previously these were only page-aligned (``cuda.aligned_zeros``), so async transfers silently ran synchronously. **Migration note:** device-to-host copies into these buffers are now *genuinely* asynchronous — results from ``run()`` on the async processes (LS/CE/PDM/NFFT) must not be read before calling ``finish()`` (the batched entry points synchronize internally; the internal BLS/TLS consumers that relied on the old effectively-synchronous copies now sync before reading)
        * Fixed wheel/sdist omitting the ``base``/``memory`` subpackages (pip installs of the v1.0 branch were unimportable)
        * Lazy module imports via PEP 562 ``__getattr__`` in ``cuvarbase/__init__.py`` (importing the package does not import the GPU modules). Historical note: the interim scikit-cuda numpy shim this enabled was removed along with the scikit-cuda dependency itself (see the Lomb-Scargle/NFFT section)
        * Fixed CUDA kernel lookup crashing for editable installs (``pip install -e .``) on Python < 3.12 when cuvarbase is imported from outside the source tree; kernel paths now resolve relative to the package directory
        * GitHub Actions CI: CPU test suite (108 tests; GPU tests stubbed/skipped) on Python 3.9-3.12 + build-wheel-install-import packaging check; flake8 error class enforced
        * Root ``conftest.py`` stubs pycuda/skcuda so the suite runs on GPU-less machines
        * Removed vestigial ``cuvarbase.periodograms`` scaffolding
        * Single-sourced the device/global functions shared by ``bls.cu`` and ``bls_optimized.cu`` into ``bls_common.cuh``, inlined via a ``//{INCLUDE ...}`` directive expanded at load time (``_module_reader``). Removes the drift hazard that once let the ``reduction_max`` s>32 bug be fixed in only one copy; the kernel-drift test now asserts the include mechanism. Functionally equivalent; not bit-identical for the standard kernel — the shared header adopted the optimized variant's float literals, so ``store_best_sols``/``bls_value`` in the standard kernel now do a few divisions in float32 (under fast-math) instead of double-then-truncate, shifting reported solutions by ~1-2 ulp at most
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
    * Includes the post-0.2.6 development that never shipped in any release:
        * Added Sparse BLS implementation for efficient transit detection with small datasets
        * New ``sparse_bls_cpu`` function that avoids binning and grid searching
        * New ``eebls_transit`` wrapper that automatically selects between sparse (CPU) and standard (GPU) BLS
        * Based on algorithm from Panahi & Zucker 2021 (https://arxiv.org/abs/2103.06193)
        * More efficient for datasets with < 500 observations
        * NUFFT LRT implementation for transit detection
        * Refactored codebase organization with base/, memory/, and periodograms/ modules

* **0.2.6** *(tagged May 2025, never published to PyPI)*
    * pycuda 2025 compatibility fixes; content folded into 1.0.0

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

