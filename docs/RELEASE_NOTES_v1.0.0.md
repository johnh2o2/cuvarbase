# cuvarbase 1.0.0

**First major release.** cuvarbase provides GPU-accelerated period-finding and transit-detection algorithms for astronomical time series: Box Least Squares (BLS), Transit Least Squares (TLS), Lomb–Scargle (including multiharmonic), Phase Dispersion Minimization (PDM), Conditional Entropy (CE), and the non-uniform FFT (NFFT) that powers them.

This is the first release published to PyPI since **0.2.5 (October 2023)** — it contains everything from the tagged-but-never-published 0.2.6 maintenance release (May 2025) plus all of the 1.0 development work. If you `pip install cuvarbase` today you get 0.2.5; 1.0.0 is a substantially different, faster, and more correct package.

In production: cuvarbase's BLS has powered the TESS Quick-Look Pipeline's planet search since Sector 59 (Kunimoto et al. 2023, RNAAS 7, 28).

## Highlights

- **New: survey-scale GPU Transit Least Squares — the fastest TLS available.** A batch-native phase-binned kernel with exact top-K refinement searches a TESS-FFI-sector light curve in ~1.2 ms (a Kepler 4-year light curve, 65k points × 172k trial periods, in 0.17 s), with no cap on points per light curve and safe BJD-scale timestamps. Head-to-head on the *same* GPU at matched search settings and equal detection significance (SDE within 1–3% under the pre-1.0 SDE definition, 100% injected recovery), it is **30–171× faster than GTLS** (arXiv:2607.00348) — the only other GPU TLS — and thousands of times faster than the reference CPU `transitleastsquares` package, whose results it reproduces in golden tests.
- **Standard BLS runs 257–354× faster than astropy's `BoxLeastSquares`** (measured across 7 GPU architectures, V100 through H200; 10,000 observations × 5,000 frequencies). At cloud spot prices that is roughly **$0.14–0.50 per million light curves** (RTX 4000 Ada / V100 / L40).
- **Versus the previous cuvarbase:** the GPU kernels were already fast and their steady-state throughput is unchanged — the wins are in everything around them. 0.2.6 recompiled its CUDA kernels on **every single call** (~0.25–0.4 s, forever); 1.0.0 compiles once and caches, measuring **34× higher per-lightcurve throughput in a call-per-lightcurve loop** (10× over a 100-lightcurve run including the first compile). Survey-scale Lomb–Scargle is **2.9× faster**, the BLS survey path is a further **2.0–12.7× faster end-to-end** on realistic Keplerian grids (fused-`noverlap` kernels, conflict-scatter staging, occupancy-aware chunking — July 2026), and 0.2.6's LS/PDM paths segfault outright on modern pycuda (≥2025.1) — on a current software stack, 1.0.0 is effectively the only version that runs.
- **Survey-scale Lomb–Scargle beats the fastest CPU package.** At realistic survey frequency grids, batched GPU LS is 1.5× (TESS-like) to 12.6× (Kepler-like) faster per light curve than nifty-ls, and >15–27× on ZTF/HAT-Net-scale grids where nifty-ls exceeded the benchmark timeout. (Honesty note: for a single light curve at small frequency grids, nifty-ls on CPU is still the better tool — see [docs/BENCHMARK_RESULTS.md](https://github.com/johnh2o2/cuvarbase/blob/v1.0.0/docs/BENCHMARK_RESULTS.md).)
- **Correct results on absolute (BJD-scale) timestamps.** Pre-1.0, feeding BLS raw BJD times (~2.45 million days) silently destroyed the phase fold in float32. Measured: an injected P=3.46 d transit recovered at power 0.30 on near-zero timestamps collapses to power 0.089 at the wrong frequency when the same data carries BJD timestamps in 0.2.6 — no error, no warning. 1.0.0 returns identical periodograms on both timescales (r=1.000000); all BLS paths epoch-subtract in float64 first.
- **Deterministic periodograms.** A float32 guard bug let degenerate trial boxes produce run-to-run-varying spurious peaks on single-site ground-based data (reported by @astrobatty against HATPI light curves). Fixed at the root, with regression tests proving 500 ppm transits still survive.
- **New algorithms and APIs**: sparse BLS for small datasets (Panahi & Zucker 2021), batched multi-lightcurve BLS, Keplerian frequency grids (4–37× fewer trial frequencies at survey baselines), multiharmonic generalized Lomb–Scargle on GPU, fast PDM kernels, CE log-probability periodograms, and an experimental NUFFT matched-filter transit search.
- **Modern, lighter install**: Python 3.9–3.14, numpy 2.x, no more scikit-cuda or `future`; `import cuvarbase` works on GPU-less machines (the pure helpers need no pycuda at all; the method modules need the pycuda package but no device until the first GPU call).
- **Trustworthy by construction**: the GPU test suite grew from 37 test functions with no CI (0.2.5) to **1,785 passed + 1 xfailed of 1,786 collected** (0 failed, 0 skipped; full suite, NVIDIA A40, 6 September 2026), plus a 14-check on-GPU release gate, CPU CI across Python 3.9–3.14, and a published benchmark methodology with archived raw results. The expected failure is `test_examples_compile.py::test_notebook_code_cells_compile_without_warnings[Phase Dispersion Minimization.ipynb]`, for known non-raw TeX label strings. The release gate on the frozen tree must reproduce these measured Phase 4 counts before tagging.

## Performance

All numbers are measured, with configs and raw JSON archived in [benchmarks/results/](https://github.com/johnh2o2/cuvarbase/blob/v1.0.0/benchmarks/results/) and summarized in [docs/BENCHMARK_RESULTS.md](https://github.com/johnh2o2/cuvarbase/blob/v1.0.0/docs/BENCHMARK_RESULTS.md).

| Comparison | Result | Setup |
|---|---|---|
| BLS vs astropy `BoxLeastSquares` (CPU) | **257–354× faster** | 10k obs × 5k freqs, 7 GPUs (V100→H200), astropy 7.2.0 |
| TLS vs GTLS (the only other GPU TLS), same GPU, equal SDE | **30–171× faster**, growing with baseline | 200–2000-d baselines, matched grids + epoch density, RTX A5000 |
| TLS vs reference `transitleastsquares` (CPU, all cores) | **~10³× at matched SDE fidelity** | Same light curves and period grid, single RTX A5000 |
| TLS survey throughput | **TESS-FFI 1.2 ms/LC; Kepler-4yr 0.17 s/LC** | 100% injected recovery; RTX A5000. V100 within ~1.3× either way; the RTX 4000 Ada workstation card is 1.2–2.4× slower (2.4× on the TESS-FFI row) |
| BLS survey path vs pre-optimization v1.0 | **2.0–12.7× end-to-end; 2.9–9.2× kernel-only** | ZTF/HAT-Net/TESS/Kepler-shaped Keplerian grids, RTX A5000 |
| Lomb–Scargle vs nifty-ls (CPU), survey grids | **1.5× (TESS) → 12.6× (Kepler); >15–27× (HAT-Net/ZTF, timeout)** | Realistic per-survey frequency grids, batched, RTX A5000 |
| Batched BLS vs looping single light curves | **2.2–10× faster** | 2–10 LCs/batch, ndata 200–20,000, RTX A5000 |
| Keplerian vs uniform frequency grid | **4–37× fewer frequencies; 1.5–24× wall-time** | ZTF/HAT-Net/TESS/Kepler-shaped surveys, identical recovery |
| Kernel caching (all BLS entry points) | **first call 1.67 s → 7.6 ms thereafter** | Measured on the RTX A5000 (table below); previously *every* call paid CUDA compilation |
| Estimated survey costs | ZTF 10M LCs ≈ $0.69 (3.5 h); LS+BLS on ZTF+HAT-Net+TESS+Kepler ≈ $33 | Projection from measured throughput, RTX A5000 @ $0.20/hr |

### Measured head-to-head vs cuvarbase 0.2.6 (RTX A5000, CUDA 12.4, July 2026)

Identical inputs on both sides; v1.0 run at `noverlap=1` for apples-to-apples because 0.2.6 silently ignores `noverlap` (v1.0's default `noverlap=2` buys a finer phase search for ~2× kernel work). Raw JSON, scripts, and full periodograms in `benchmarks/results/v026_head_to_head_jul2026/`.

| Measurement | 0.2.6 | 1.0.0 | Change |
|---|---|---|---|
| BLS kernel-only, TESS-scale (20k obs × 13.5k freqs) | 9.8 ms | 9.8 ms | **1.00× — kernel throughput unchanged** |
| BLS per-call in a lightcurve loop (steady state) | 261 ms | 7.6 ms | **34× faster** (kernel cached vs recompiled every call) |
| BLS 100-lightcurve run, incl. first compile | 28.4 s | 2.8 s | **10× faster** |
| BLS cold first call (empty caches) | 2.37 s | 1.67 s | 1.4× faster |
| BLS warm single call, 10k×5k, end-to-end | 5.9 ms | 7.8 ms | 0.76× — see note |
| Lomb–Scargle, survey grid (3k obs × 100k freqs) | 33.3 ms | 11.7 ms | **2.85× faster** |
| Lomb–Scargle, small (10k × 5k) | 8.8 ms | 9.2 ms | parity |
| PDM (same algorithm / new `_fast` kernel) | 3.7 ms | 3.4 / 2.8 ms | 1.08× / 1.33× |

Honesty notes: we claim **no** raw-kernel speedup — the kernel-only decomposition is identical, and the 34×/10× are architectural wins (compile-once vs compile-always) that any real pipeline experiences. The warm single-call row shows v1.0 spending ~2 ms more host-side work per call (float64 epoch handling, χ²₀ bookkeeping, convention support — the price of the correctness fixes); millisecond-scale timings jitter 2–4× between rounds on cloud pods, so pooled medians are reported. One GPU model; the earlier "21–390× vs pre-v1.0" figures from Feb 2026 conflated compile overhead and are retracted — do not cite them. Running the 0.2.6 baseline at all required numpy 1.23 and a 2022-era pycuda for LS/PDM (segfaults on pycuda 2025.1).

## New features

### BLS
- **Sparse BLS** (Panahi & Zucker 2021) on GPU and CPU (`sparse_bls_gpu`, `sparse_bls_cpu`) for small datasets (≲500 points); `eebls_transit` auto-selects it by dataset size and applies Keplerian duration constraints consistently on both paths.
- **Batched BLS**: `eebls_gpu_batch()` processes many light curves per kernel launch and accepts per-frequency `qmin`/`qmax` arrays.
- **Keplerian frequency grids**: `cuvarbase.bls_frequencies.keplerian_freq_grid()` (with `return_qvals=True` feeding duration bounds straight into the batch API).
- **Selectable power conventions**: `convention='chi2ratio' | 'snr' | 'loglik'` on all BLS entry points (+ `convert_bls_power()`); `'snr'` verified equal to astropy's `objective='snr'`.
- **Optimized/adaptive kernels**: `eebls_gpu_fast_optimized()` and `eebls_gpu_fast_adaptive()` (warp-shuffle reductions, automatic block sizing). With a warm kernel cache these measure ~1.0–1.3× over the standard fast kernel — the real win for everyone is the cache itself.
- `noverlap` is now honored on the fast path (elementwise max over phase-shifted passes; default 2).
- **Survey-speed kernels (July 2026)**: fused-`noverlap` histograms, conflict-scatter staging of dense cadences, occupancy-aware frequency chunking, and host-path overhead fixes — end-to-end **2.0–12.7×** on realistic Keplerian survey grids, kernel-only 2.9–9.2× (the TESS-scale 12.7× includes curing a default-environment BLAS threadpool pathology in-library; 5.8× against an already-tuned baseline). Periodograms unchanged (parity correlation 1.0000000, identical peaks).

### Lomb–Scargle & NFFT
- **Multiharmonic generalized Lomb–Scargle on GPU** (`nharmonics>1`). The per-frequency solve runs on the host in float64; on device, after the Sep-2026 psi-table and grid-sizing fixes, the NFFT path agrees with the float64 `lomb_scargle_direct_sums` reference to 5.7e-7 in float32 and 7.4e-10 with `use_double=True` for H=2,3 (the host solve itself is exact to float64 roundoff).
- **scikit-cuda dependency removed**: cuFFT is called through a minimal in-house ctypes binding at performance parity (±2%). This unblocks numpy ≥1.24 / 2.x environments.
- **Optional cuFINUFFT backend** (`pip install cuvarbase[cufinufft]`, `use_cufinufft=True`) as a numerical cross-check; the built-in kernel remains default and faster.
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
- **Statistics discipline**: the SDE comes from the uniform coarse spectrum while refinement sharpens only the reported parameters; there is no fixed SDE→FAP table (an opt-in null bootstrap on `tls_search_batch(fap_null_draws=...)` replaces it). On the reference package's own period grid the default epoch grid reports the same SDE for the same detection to within the coarse-vs-fine epoch-grid difference (measured 5–15% under the 1.0 SDE definition; the July-2026 “within 1–3%, within 1% at `t0_oversample=33` at ~5–13× cost” figures were measured under the pre-1.0 signal-residue definition), with 100% injected recovery in every tested regime.
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

A read-only algorithm audit of the release candidate (September 2026, on-device) found a set of default-path defects that changed *results*, and a performance pass followed. Every item is reproduced on device before its fix and carries a regression test; the full per-item list with root causes is in the 1.0.0 section of [CHANGELOG.rst](https://github.com/johnh2o2/cuvarbase/blob/v1.0.0/CHANGELOG.rst). The condensed list:

**Correctness (result-changing):**
- **Input validation (BREAKING)** — every entry point rejects non-finite `t`/`y`/`dy`, `dy <= 0`, mismatched lengths, too-short light curves, bad frequency grids and inverted duration bounds with `ValueError` on the host, before any GPU work (see the migration table below). Previously a NaN gave a finite-but-wrong periodogram, and a bad `q` bound crashed the kernel and destroyed the process's CUDA context.
- **BLS**: 64-bit thread indexing in the phase-fold kernels (`eebls_gpu`/`eebls_transit` on > 2³¹ threads silently returned zeros, powers above 1 and the wrong peak); per-frequency `qmin`/`qmax` arrays are now honoured per frequency by `eebls_gpu` (they collapsed to one grid-wide window) and its bin buffers are sized correctly for Keplerian grids (out-of-bounds writes); the fast kernels evaluate the widest box allowed by `qmax` (the loop stopped one rung short); the sparse path centres the flux in float64; `eebls_transit` uses the fused fast kernel above the sparse threshold and recovers solutions at the top peaks; the Keplerian grid recursion of `transit_autofreq`/`keplerian_freq_grid` is solved with numpy — grids change at float64 rounding only.
- **TLS**: the default duration window is the per-period Keplerian one (the old constant `[0.005, 0.15]` window excluded physical durations beyond P ≈ 60 d for a Sun-like star); `'T0'` is the absolute mid-transit time of the first transit at or after `min(t)` on every path, with `'t0_phase'` alongside; SDE/SNR use the reference package's definitions; the fixed SDE→FAP table is gone (opt-in null bootstrap instead); period grids in any order; flat light curves return SDE = 0.
- **Lomb–Scargle / NFFT**: the w-spectrum was gridded with the psi tables of the differently sized yw grid; `floorf()` on the double-precision grid coordinate; aliased garbage for bands that do not start near zero (grid sizing); wrong NFFT magnitudes for absolute-time input; `nharmonics > 1` and `amplitude_prior` ignored on some paths; non-uniform frequency grids are now rejected instead of silently evaluated on the implied uniform grid; stale results after `preallocate()`; `only_return_best_freqs=True` returns the FAP itself (it returned `1 - FAP`); cuFINUFFT in double precision.
- **Conditional entropy**: the brightest point fell into an out-of-range magnitude bin (clamped now); weighted-CE `max_phi` truncation; `use_double=True, use_fast=True` crash; constructor `balanced_magbins`/`widen_mag_range` ignored; `preallocate()` never uploaded the grid; recompilation on every call; histogram accumulation across `set_data=False` calls; float32 frequency arrays rejected.
- **PDM**: out-of-bounds bin read in the `binned_step` kernel; the deprecated 4-tuple format returned a flat spectrum for unnormalized weights.
- **NUFFT-LRT**: BJD-scale times; `epochs=None` is a real epoch search (returns a tuple — breaking); the sequential detector fits an intercept; Detector A estimates its PSD from the basis-projected residual; NFFT `sigma = 4`; PSD validation and flooring; singular priors handled in the correct limit.

**Performance (measured on one shared NVIDIA A40 — read every ratio as indicative of that machine, not as a portable number; bit-neutral unless the CHANGELOG says otherwise):**
- **BLS**: `eebls_gpu`, `eebls_gpu_custom`, `hone_solution` and `sparse_bls_gpu` take their kernels from the LRU cache instead of compiling per call; the adaptive/optimized paths run the fused-`noverlap` kernel; no per-call `BLSMemory` on the single-call paths; vectorized solution re-phasing and `einsum` prologues.
- **Lomb–Scargle**: `batched_run_const_nfreq` reuses its memory set, cuFFT plans and pinned buffers across calls; the multiharmonic host solve is one stacked `numpy.linalg.solve`; numpy reductions on the host path (2.0× per light curve at N = 65,000).
- **Conditional entropy / PDM**: `use_fast=True` sizes its grid from the device (it is now the faster single-precision path: 1.2×/1.9×/8× at 300/2,000/10,000 observations × 10⁵ frequencies) and no longer allocates the global histogram it never read; PDM `run()` reuses its device buffers across same-shape calls.
- **TLS**: `tls_transit` builds only the duration bounds; `tls_search_batch` computes its statistics sequentially (the thread pool was GIL-bound and slower); memoized template tables — combined 1.44–1.62× for `tls_transit` and 2.41× for a 64-light-curve `tls_search_batch`.

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
| `LombScargleAsyncProcess.batched_run_const_nfreq` default `batch_size` 10 → 1 (measured faster; the PDM and CE `batched_run_const_nfreq` keep 10) | Pass `batch_size=10` to restore old Lomb–Scargle chunking. |
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
