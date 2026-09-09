# cuvarbase 1.0.0

**First major release.** cuvarbase provides GPU-accelerated period-finding and transit-detection algorithms for astronomical time series: Box Least Squares (BLS), Transit Least Squares (TLS), Lomb–Scargle (including multiharmonic), Phase Dispersion Minimization (PDM), Conditional Entropy (CE), and the non-uniform FFT (NFFT) that powers them.

This is the first release published to PyPI since **0.2.5 (October 2023)** — it contains everything from the tagged-but-never-published 0.2.6 maintenance release (May 2025) plus all of the 1.0 development work. If you `pip install cuvarbase` today you get 0.2.5; 1.0.0 is a substantially different, faster, and more correct package.

In production: cuvarbase's BLS has powered the TESS Quick-Look Pipeline's planet search since Sector 59 (Kunimoto et al. 2023, RNAAS 7, 28).

## Highlights

- **New GPU Transit Least Squares:** a phase-binned batch engine with exact candidate refinement. The [current ZTF/TESS benchmark](TRANSIT_BENCHMARKS.md) reports its timing advantage over public GTLS together with independent recovery and false-positive qualifications.
- **Faster BLS searches and grid construction:** compare actual PyPI 0.2.5, v1 and tested CPU/GPU alternatives in the [current benchmark](TRANSIT_BENCHMARKS.md).
- **Versus actual PyPI 0.2.5:** fused phase searches, conflict-scatter staging, reusable batch memory, vectorized host scans and grid construction, plus support for the current NumPy/PyCUDA stack. Both releases receive warmed kernels and reusable PyPI memory in the new comparison; its warm speedup is not attributed entirely to compilation caching.
- **Correct results on absolute (BJD-scale) timestamps.** Pre-1.0, feeding BLS raw BJD times (~2.45 million days) silently destroyed the phase fold in float32. Measured: an injected P=3.46 d transit recovered at power 0.30 on near-zero timestamps collapses to power 0.089 at the wrong frequency when the same data carries BJD timestamps in 0.2.6 — no error, no warning. 1.0.0 returns identical periodograms on both timescales (r=1.000000); all BLS paths epoch-subtract in float64 first.
- **Deterministic periodograms.** A float32 guard bug let degenerate trial boxes produce run-to-run-varying spurious peaks on single-site ground-based data (reported by @astrobatty against HATPI light curves). Fixed at the root, with regression tests proving 500 ppm transits still survive.
- **New algorithms and APIs**: sparse BLS for small datasets (Panahi & Zucker 2021), batched multi-lightcurve BLS, Keplerian frequency grids with stellar-density and duration constraints, multiharmonic generalized Lomb–Scargle on GPU, fast PDM kernels, CE log-probability periodograms, and an experimental NUFFT matched-filter transit search.
- **Modern, lighter install**: Python 3.9–3.14, numpy 2.x, no more scikit-cuda or `future`; `import cuvarbase` works on GPU-less machines (the pure helpers need no pycuda at all; the method modules need the pycuda package but no device until the first GPU call).
- **Trustworthy by construction**: the GPU test suite grew from 37 test functions with no CI (0.2.5) to **1,785 passed + 1 xfailed of 1,786 collected** (0 failed, 0 skipped; full suite, NVIDIA A40, 6 September 2026), plus a 14-check on-GPU release gate, CPU CI across Python 3.9–3.14, and a published benchmark methodology with archived raw results. The expected failure is `test_examples_compile.py::test_notebook_code_cells_compile_without_warnings[Phase Dispersion Minimization.ipynb]`, for known non-raw TeX label strings. The release gate on the frozen tree must reproduce these measured Phase 4 counts before tagging.

## Performance

The [current transit benchmark](TRANSIT_BENCHMARKS.md) is the source for BLS/TLS release claims: one figure, single-source and batch timing, independent recovery, null false positives, and search-cost projections. Equal scalar SDE is not an equal-sensitivity guarantee.

The published upgrade baseline in this campaign is 0.2.5; the 0.2.6 tag was not published to PyPI. The [benchmark index](BENCHMARK_RESULTS.md) links the current report, component evidence and historical-claim audit.

## New features

### BLS
- **Sparse BLS** (Panahi & Zucker 2021) on GPU and CPU (`sparse_bls_gpu`, `sparse_bls_cpu`) for small datasets (≲500 points); `eebls_transit` auto-selects it by dataset size and applies Keplerian duration constraints consistently on both paths.
- **Batched BLS**: `eebls_gpu_batch()` processes many light curves per kernel launch and accepts per-frequency `qmin`/`qmax` arrays.
- **Keplerian frequency grids**: `cuvarbase.bls_frequencies.keplerian_freq_grid()` (with `return_qvals=True` feeding duration bounds straight into the batch API).
- **Selectable power conventions**: `convention='chi2ratio' | 'snr' | 'loglik'` on all BLS entry points (+ `convert_bls_power()`); `'snr'` verified equal to astropy's `objective='snr'`.
- **Optimized/adaptive kernels**: `eebls_gpu_fast_optimized()` and `eebls_gpu_fast_adaptive()` provide warp-shuffle reductions and automatic block sizing. Their benefit depends on workload and settings.
- `noverlap` is now honored on the fast path (elementwise max over phase-shifted passes; default 2).
- **BLS throughput features (July 2026):** fused phase histograms, observation-scatter staging, frequency chunking, and host overhead fixes. The [current benchmark](TRANSIT_BENCHMARKS.md) measures their practical upgrade effect and diagnostic ablations; scattering does not demonstrate a benefit on its three selected cases.

### Lomb–Scargle & NFFT
- **Multiharmonic generalized Lomb–Scargle on GPU** (`nharmonics>1`). The per-frequency solve runs on the host in float64; on device, after the Sep-2026 psi-table and grid-sizing fixes, the NFFT path agrees with the float64 `lomb_scargle_direct_sums` reference to 5.7e-7 in float32 and 7.4e-10 with `use_double=True` for H=2,3 (the host solve itself is exact to float64 roundoff).
- **scikit-cuda dependency removed**: cuFFT is called through a minimal in-house ctypes binding that preserves the cuFFT execution path. This unblocks numpy ≥1.24 / 2.x environments.
- **Optional cuFINUFFT backend** (`pip install cuvarbase[cufinufft]`, `use_cufinufft=True`) as a numerical cross-check; the built-in kernel remains the default.
- **Rigorous NFFT accuracy control**: `autoset_m` now uses the L1-norm truncation bound, and a float32 π-literal bug that imposed a ~1e-3 error floor on *double-precision* NFFTs is fixed — float64 error now tracks theory down to ~1e-10.
- Baluev false-alarm probability evaluates in log space (no more `FAP == 0` underflow for significant peaks).

### PDM (community contribution: @astrobatty)
- Fast shared-memory CUDA kernels for all four PDM variants; modern `(t, y, err)` API with automatic frequency grids (legacy format deprecated, not removed).
- Batch processing: `batched_run_const_nfreq()` and memory-auto-sized `large_run()`.

### Conditional Entropy (community contribution: @astrobatty)
- `compute_log_prob=True` log-probability periodograms, input normalization, overflow guards, and an implemented `memory_requirement()`. CE is otherwise in maintenance mode — for an actively developed GPU CE/AOV search see the `periodfind` package.

### Transit Least Squares (new survey-scale engine)
- **`tls_search_batch()`** searches whole surveys against a shared period grid: one block per (light curve, period) folds into shared-memory phase bins and scans every (duration, epoch) trial against integrated-template tables with a closed-form χ²; a second kernel re-fits the best `refine_top_k` candidates exactly. The fast path is the default for `tls_search`/`tls_search_gpu`/`tls_transit` (`use_fast=False` keeps the legacy per-point kernel and its ~3,500-point cap).
- No cap on points per light curve; BJD-scale timestamps are safe (float64 epoch subtraction); the period grid is banded by required phase resolution so long-period searches don't pay the finest band's cost.
- Limb-darkened templates (optional batman-package), Ofir (2014) period grids, Keplerian per-period duration windows.
- **Statistics:** SDE uses the coarse spectrum while refinement sharpens candidate parameters. Null calibration and independent recovery are required to compare detection performance; a scalar SDE difference or successful golden tests do not establish population sensitivity. An opt-in null bootstrap is available on `tls_search_batch(fap_null_draws=...)`.
- Golden-tested against `transitleastsquares`; validated on RTX A5000 (sm86), RTX 4000 Ada (sm89), and V100 (sm70).

### Experimental (quarantined; not yet recommended for science use)
- **NUFFT-LRT likelihood-ratio transit search** (`cuvarbase.nufft_lrt`), contributed by Jamila Taaki (@xiaziyna): a frequency-domain matched filter for box transits in correlated noise, whitened by a noise PSD that is supplied or estimated from the data. `NUFFTLRTAsyncProcess.run(t, y, periods, durations=..., epochs=None, detector='matched' | 'marginal' | 'sequential', systematics_basis=None, coeff_prior_mean=None, coeff_prior_cov=None, ...)` selects the stationary whitened filter (default), Detector A of Taaki, Kamalabadi & Kemball (2020) — systematics coefficients marginalized under a Gaussian prior, computed in the whitened frequency domain via the Woodbury identity — or the papers' sequential baseline (least-squares cotrend with an intercept, then the filter). With `epochs=None` an automatic epoch grid is scanned per (period, duration) cell and `(snr, best_epoch)` is returned; explicit `epochs` return the `(nP, nD, nE)` array.
- **Status, honestly**: the module emits an `EXPERIMENTAL` `UserWarning` when `NUFFTLRTAsyncProcess` is first constructed (not at import) and is deliberately *not* exported from the top-level `cuvarbase` namespace (`import cuvarbase.nufft_lrt` explicitly). Its statistic is a whitened correlation, not an N(0,1) SNR, and thresholds must be calibrated per dataset. Test coverage: CPU tests of the Detector-A algebra (Woodbury path against a dense inverse) and of the pipeline, plus GPU behavioural tests (NFFT against the exact adjoint DFT, multi-season detection, BJD-scale invariance, the Sep-2026 regression tests). Its injection-recovery re-validation after the September 2026 fixes ran on 2026-09-06 (200 injections per depth, one A40; `benchmarks/results/nufft_lrt_validation_2026-09-06/`): the public default path is correct on BJD-scale times (identical statistics to 5e-8) and recovers random-epoch transits; with a systematics basis the Detector A and sequential detectors recover 3/44/98/100% of transits at depths 0.004/0.008/0.016/0.032 where basis-free BLS recovers 0/0/2/16% and TLS none; in OU red noise the whitened filter is 6-10 ± 3% more complete than BLS at the transition depths but a flat-PSD matched filter does as well or better; in white noise BLS and TLS are 10-12 ± 3% more complete. It stays **outside the 1.x API-stability promise** because that campaign showed its defaults (automatic epoch grid, whitening) and `run()` return conventions should still change before the API is frozen, so it may change incompatibly in a 1.x release. See the [NUFFT-LRT page](https://johnh2o2.github.io/cuvarbase/nufft_lrt.html) of the documentation.

### Usability & infrastructure
- `import cuvarbase` no longer requires a GPU or creates a CUDA context; CPU-only helpers work on laptops.
- All host transfer buffers are genuinely page-locked, so async GPU transfers actually overlap compute.
- Typed exceptions (`ValueError`/`RuntimeError`) with clear messages replace bare `Exception`s and `assert`s; validation survives `python -O`.

## Notable correctness fixes

Beyond the highlights above (BJD epoch handling, nondeterministic degenerate-box peaks, `noverlap`):

- `mod1_fast` integer overflow corrupted phases when `t × f ≥ 2³¹` (long baselines × high frequencies).
- The CPU reference `single_bls` folded phases in an order that lost up to ~1.5e-5 of phase precision per year of baseline (it subtracted the trial phase before wrapping); it now wraps first, bit-identically to the GPU kernels.
- The optimized kernel's block-level max reduction dropped half the per-block candidates.
- `eebls_gpu_batch` results now match the single-LC path exactly (it was silently single-pass, and recompiled kernels every call).
- `lomb_scargle_simple` double-applied inverse-variance weights (inverted weighting for heteroskedastic errors).
- The direct-sums LS path returned stale results for GPU-resident workflows (`transfer_to_host` was gated on the wrong flag).
- `eebls_transit`'s sparse path crashed on documented kwargs and silently dropped Keplerian duration constraints.
- PDM CPU reference functions no longer mutate caller arrays in place.
- Wheels/sdists now include all subpackages; editable installs resolve kernel files correctly.

## September 2026 audit fixes

A read-only algorithm audit of the release candidate (September 2026, on-device) found a set of default-path defects that changed *results*, and a performance pass followed. Every item is reproduced on device before its fix and carries a regression test; the full per-item list with root causes is in the 1.0.0 section of [CHANGELOG.rst](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/CHANGELOG.rst). The condensed list:

**Correctness (result-changing):**
- **Input validation (BREAKING)** — every entry point rejects non-finite `t`/`y`/`dy`, `dy <= 0`, mismatched lengths, too-short light curves, bad frequency grids and inverted duration bounds with `ValueError` on the host, before any GPU work (see the migration table below). Previously a NaN gave a finite-but-wrong periodogram, and a bad `q` bound crashed the kernel and destroyed the process's CUDA context.
- **BLS**: 64-bit thread indexing in the phase-fold kernels (`eebls_gpu`/`eebls_transit` on > 2³¹ threads silently returned zeros, powers above 1 and the wrong peak); per-frequency `qmin`/`qmax` arrays are now honoured per frequency by `eebls_gpu` (they collapsed to one grid-wide window) and its bin buffers are sized correctly for Keplerian grids (out-of-bounds writes); the fast kernels evaluate the widest box allowed by `qmax` (the loop stopped one rung short); the sparse path centres the flux in float64; `eebls_transit` uses the fused fast kernel above the sparse threshold and recovers solutions at the top peaks; the Keplerian grid recursion of `transit_autofreq`/`keplerian_freq_grid` is solved with numpy — grids change at float64 rounding only.
- **TLS**: the default duration window is the per-period Keplerian one (the old constant `[0.005, 0.15]` window excluded physical durations beyond P ≈ 60 d for a Sun-like star); `'T0'` is the absolute mid-transit time of the first transit at or after `min(t)` on every path, with `'t0_phase'` alongside; SDE/SNR use the reference package's definitions; the fixed SDE→FAP table is gone (opt-in null bootstrap instead); period grids in any order; flat light curves return SDE = 0.
- **Lomb–Scargle / NFFT**: the w-spectrum was gridded with the psi tables of the differently sized yw grid; `floorf()` on the double-precision grid coordinate; aliased garbage for bands that do not start near zero (grid sizing); wrong NFFT magnitudes for absolute-time input; `nharmonics > 1` and `amplitude_prior` ignored on some paths; non-uniform frequency grids are now rejected instead of silently evaluated on the implied uniform grid; stale results after `preallocate()`; `only_return_best_freqs=True` returns the FAP itself (it returned `1 - FAP`); cuFINUFFT in double precision.
- **Conditional entropy**: the brightest point fell into an out-of-range magnitude bin (clamped now); weighted-CE `max_phi` truncation; `use_double=True, use_fast=True` crash; constructor `balanced_magbins`/`widen_mag_range` ignored; `preallocate()` never uploaded the grid; recompilation on every call; histogram accumulation across `set_data=False` calls; float32 frequency arrays rejected.
- **PDM**: out-of-bounds bin read in the `binned_step` kernel; the deprecated 4-tuple format returned a flat spectrum for unnormalized weights.
- **NUFFT-LRT**: BJD-scale times; `epochs=None` is a real epoch search (returns a tuple — breaking); the sequential detector fits an intercept; Detector A estimates its PSD from the basis-projected residual; NFFT `sigma = 4`; PSD validation and flooring; singular priors handled in the correct limit.

**Additional implementation changes:** performance comparisons for release advertising are in the [current transit benchmark](TRANSIT_BENCHMARKS.md).
- **BLS**: `eebls_gpu`, `eebls_gpu_custom`, `hone_solution` and `sparse_bls_gpu` take their kernels from the LRU cache instead of compiling per call; the adaptive/optimized paths run the fused-`noverlap` kernel; no per-call `BLSMemory` on the single-call paths; vectorized solution re-phasing and `einsum` prologues.
- **Lomb–Scargle**: `batched_run_const_nfreq` reuses its memory set, cuFFT plans and pinned buffers across calls; the multiharmonic host solve is one stacked `numpy.linalg.solve`; vectorized NumPy reductions on the host path.
- **Conditional entropy / PDM**: `use_fast=True` sizes its grid from the device and no longer allocates the global histogram it never read; PDM `run()` reuses its device buffers across same-shape calls.
- **TLS**: `tls_transit` builds only the duration bounds; `tls_search_batch` computes its statistics sequentially (the thread pool was GIL-bound and slower); memoized template tables.

## Breaking changes & migration

| Change | Migration |
|---|---|
| **Every entry point now validates its input and raises `ValueError`** — non-finite `t`/`y`/`dy`, `dy <= 0`, mismatched lengths, an empty or too-short light curve (4 points for Lomb–Scargle, 3 for NUFFT-LRT, 2 elsewhere), non-finite/non-positive frequencies, and transit-duration bounds outside `0 < qmin <= qmax <= 1`. These used to be accepted silently: a NaN timestamp gave a finite BLS/CE periodogram with the wrong peak, `dy = 0` gave an all-NaN PDM spectrum or a Lomb–Scargle power of `-1` everywhere, and a NaN q bound or an under-populated Keplerian grid crashed the kernel and killed the process's CUDA context. Checks run on the host before any GPU work, so a rejected call leaves the context usable. Valid finite input is bit-identical. | Filter first: `m = np.isfinite(t) & np.isfinite(y) & (dy > 0)`. Pipelines that read an all-zero or `-1` periodogram as “no detection” must now catch `ValueError`. Helpers: `cuvarbase.utils.check_lightcurve` / `check_freqs`. |
| **Python ≥ 3.9 required** (was 2.7–3.6); numpy ≥ 1.22, scipy ≥ 1.8 (the oldest releases that install on 3.9; the previously declared 1.17/1.3 could not be installed on any supported interpreter) | Upgrade the interpreter; numpy 2.x is supported. |
| **BLS results on absolute (BJD-scale) timestamps change** — they were silently wrong before. Reported `phi0` stays referenced to your original input timescale (no convention change; internally times are epoch-subtracted in float64 for precision — thanks @astrobatty, #65) | Re-baseline stored results from absolute-timestamp runs; data starting near t=0 is numerically unaffected. |
| **`noverlap` now works** on fast BLS paths (default 2): peaks can rise, runtime ~doubles at defaults | Pass `noverlap=1` for old behavior/timing. |
| **Truly async results**: reading `run()` outputs before synchronizing is now a race | Call `proc.finish()` first (batched entry points synchronize internally); `pinned=False` opts out. |
| **`import cuvarbase` no longer creates a CUDA context** | Call `cuvarbase.base.ensure_context()` (or any GPU function) before raw pycuda work; set `CUDA_DEVICE` before first GPU use, not import. |
| **`sparse_bls_cpu`/`sparse_bls_gpu`: args after `freqs` are keyword-only**; q bounds validated | Pass `qmin=`, `qmax=`, etc. by keyword. Legacy positional calls now fail loudly instead of silently returning zeros. |
| `LombScargleAsyncProcess.batched_run_const_nfreq` default `batch_size` 10 → 1 (the PDM and CE `batched_run_const_nfreq` keep 10) | Pass `batch_size=10` to restore old Lomb–Scargle chunking. |
| PDM legacy `(t, y, w, freqs)` input deprecated (still works, warns) | Move to `(t, y, err)` tuples + `freqs=`. |
| `BLSMemory.allocate_pinned_arrays` → `allocate_host_arrays` (alias warns) | Rename the call. |
| scikit-cuda is no longer installed transitively | `pip install scikit-cuda` yourself if *your* code needs it. |
| Small numerical shifts everywhere (shared kernel literals, input normalization, degenerate-box guard, NFFT π fix) | Re-baseline golden outputs; parity with old results is >0.999 correlation in our tests, and the shifts are fixes, not drift. |

## Packaging

- `pyproject.toml` (PEP 517/621) is the only packaging file (`setup.py`, `setup.cfg`, `requirements*.txt` removed); `setuptools>=77` backend with PEP 639 license metadata (`License-Expression: GPL-3.0-only`, `LICENSE.txt` shipped); Python 3.9–3.14 classifiers; dynamic versioning; wheel tag `py3-none-any`.
- Dependencies removed: `scikit-cuda`, `future`. Floors: `numpy>=1.22`, `scipy>=1.8`. Pins: `pycuda>=2017.1.1,!=2024.1.2`.
- Optional extras: `cuvarbase[test]` (pytest, nfft, astropy, batman-package, transitleastsquares — matplotlib is no longer required for the tests), `cuvarbase[cufinufft]`, `cuvarbase[docs]` (sphinx, matplotlib); batman-package enables limb-darkened TLS templates.
- pytest is configured in `pyproject.toml` (`testpaths`, `-rs --strict-markers`, `gpu` marker); `cuvarbase/kernels/wavelet.cu` (never loaded) no longer ships, guarded by an orphan-kernel test.
- GitHub Actions CI: the CPU suite on Python 3.9–3.14, wheel and sdist install legs (including `pytest --pyargs cuvarbase` from the installed wheel), a docs build, and flake8. The repository's Dockerfile was removed: it never installed cuvarbase (a rebuilt image is queued for 1.1).
- **If you fetched the earlier `v1.0.0` tag (June 2026) from this repository:** it was deleted and re-created on the 1.0.0 release commit; run `git fetch --tags --force` to replace your stale copy (a plain `git fetch` keeps the old one).

## Credits

Major community contributions to this release from **Attila Bódi (@astrobatty)** — fast PDM kernels and batch APIs, Conditional Entropy enhancements, Lomb–Scargle normalization and memory-estimation improvements, and the BLS epoch/phase-reporting work (PRs #57–#62, #65) — and **Jamila Taaki (@xiaziyna)** — the NUFFT-LRT matched-filter transit search. Thanks also to the TESS QLP team for production adoption and feedback.

## Known limitations

- NUFFT-LRT is experimental (`UserWarning` at first construction; not in the top-level namespace; outside the 1.x stability promise). It has been re-validated by injection-recovery (see its docs page for the measured numbers), but its defaults and `run()` conventions may still change in 1.x; calibrate thresholds empirically and cite the measured numbers, not the papers'.
- No benchmark against CETRA (PLATO's GPU transit code, a different algorithm family) exists yet; the GPU-vs-GPU transit-search comparison published here covers GTLS.
- float32 NFFT has a genuine ~1e-3 accuracy floor from single-precision trig on large phases; pass `use_double=True` for tight tolerances.
- Conditional Entropy is maintained but not actively developed.
