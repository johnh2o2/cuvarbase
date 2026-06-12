# v1.0 packed-release punchlist (June 11, 2026)

Source of truth for "fix ALL known errors before PyPI". Compiled by a
4-source sweep (audit doc, GitHub tracker, code markers, benchmark
records; 60 raw findings -> 45 deduped items) at HEAD 5553248 (= tag
v1.0.0), plus a completeness pass over closed issues, CHANGELOG, test
skip patterns, recent commit messages, and merged-PR threads.

Rules of engagement:
- **PyPI publish is blocked until every A item is fixed** and every
  B/C item is either fixed or *formally* dispositioned (cut from the
  wheel / documented as unsupported with no contradicting claims).
- One packed release: work lands on v1.0-fixes; the v1.0.0 tag moves
  to the final commit at publish time (nothing external references
  today's tag).
- Checking a box requires: the fix committed, a test that fails
  before / passes after (where testable), the commit hash noted next
  to the item, and any GPU-dependent verification added to the queue
  below rather than provisioning a pod per item.

## GPU verification queue

Items whose verification needs real hardware. Worked in batches: when
~5+ accumulate (or all CPU-side work is done), provision ONE pod, run
the full suite + check_release_gate.py + everything queued here,
terminate, archive, and check these off.

- [ ] (standing) full pytest suite + check_release_gate.py +
      benchmark_new_features.py --tests-only must pass on the final
      release candidate, with batman-package installed this time so
      the 5 TLS accuracy tests actually run
- [ ] re-run scripts/benchmark_adaptive_bls.py and commit the output
      JSON (backs the 1.4-5.3x README claim — bucket D traceability)
- [ ] LS comparison vs astropy 8.0 (LRA default) or version-caveat
      the published tables — bucket D, audit §6 risk 1
- [ ] nsys profile of eebls_gpu_batch at TESS scale (bucket C
      regression diagnosis)
- [ ] BJD epoch-subtraction fix: run the 3 GPU tests in
      test_bls.py::TestEpochHandling (BLSMemory/BLSBatchMemory storage +
      eebls_gpu BJD invariance) — they skip on CPU
- [ ] TLS phase-1 hardening end-to-end: shared-mem guard does NOT fire
      for ndata ~3,000 (kernel launches OK), and a run with some failed
      periods produces masked NaNs + sane SDE on hardware
- [ ] TLS duration-scaled t0 grid: kernel compiles; narrow-transit
      recovery on the audit scenario (P~100 d injection that the old
      30-epoch grid missed 8/8); runtime sanity with n_t0 up to 20k
      (cap) at the narrowest durations
- [ ] TLS golden tests: pip install transitleastsquares (in addition
      to batman-package) on the pod; run test_tls_golden.py (4 tests:
      2 recovery + 2 reference comparisons)
- [ ] cuFINUFFT plan caching: re-run the cufinufft_vs_custom section
      of benchmark_new_features.py (pip install cufinufft) — record
      whether caching moves the 0.63-0.84x ratio past 1x; correctness
      cross-check vs custom NFFT still passes


## A. Errors — wrong results, crashes, broken API (publish blockers)

- [x] **BLS float32 phase-fold degradation for BJD-scale timestamps (no t.min() subtraction)** — FIXED: `utils.subtract_epoch()` applied in float64 before every float32 cast across all 7 BLS folding paths (BLSMemory.setdata, BLSBatchMemory.set_lightcurve, eebls_gpu, eebls_gpu_custom, single_bls, sparse_bls_cpu, sparse_bls_gpu); phi0 convention now relative to min(t), documented in docstrings + CHANGELOG; regression tests in TestEpochHandling (CPU test reproduced 0.961→0.002 power collapse before fix). Commit: a987987
  - Evidence: cuvarbase/bls.py:455-470 — verified at HEAD 5553248: setdata does `self.t[:len(t)] = np.asarray(t).astype(self.rtype)[:]` with no epoch subtraction; audit §3 quick wins
  - BLSMemory.setdata / BLSBatchMemory cast raw times to float32 without subtracting t.min(). Audit demonstrated 0.705 -> 0.285 power loss with BJD-scale timestamps (~2.4e6 days) — silent accuracy loss on the most common real-world input format. No commit in v0.2.6..HEAD touches the memory path. Single sweep, no dupes.
- [x] **fap_baluev returns exactly 0 for significant peaks (issue #14, numerical underflow)** — FIXED: log-space/expm1 formulation (`-expm1(-tau) + exp(log(1-Psing) - tau)`); stays positive to the float64 limit, matches the naive formula to 1e-8 where that formula is accurate; TestFapBaluev covers underflow, monotonicity, and z∈{0,1} edges (CPU-only, no GPU queue needed). Close issue #14 at release. Commit: dc52b6a
  - Evidence: cuvarbase/lombscargle.py:896-923 — verified at HEAD: `return 1 - Psing * np.exp(-tau)` with Psing = 1-(1-z)**(0.5*N_K), eZ2 = (1-z)**(0.5*(N_K-1)); invoked from run() at lombscargle.py:841; GitHub issue #14 open
  - For z near 1 both Psing and exp(-tau) underflow to 1.0, so the function returns FAP == 0.0 exactly instead of a small positive value. Only the gammaln overflow half was fixed (line 904); the docstring's 'should be stable now' is wrong for the tail. User-facing via only_return_best_freqs=True. The one genuine numerical bug on the open tracker; fix with a log-FAP formulation before PyPI.
- [x] **lomb_scargle_async direct-sums branch gates host transfer on the wrong flag (transfer_to_device instead of transfer_to_host)** — FIXED: dirsum branch now gates the copy on transfer_to_host (matching the FFT branch); fake-memory unit tests assert the copy happens/suppresses on the right flag (failed pre-fix). Commit: 7647d3e
  - Evidence: cuvarbase/lombscargle.py:366-369 — verified at HEAD: `lomb_dirsum.prepared_async_call(*args); if transfer_to_device: memory.transfer_lsp_to_cpu()`; FFT branch correctly uses transfer_to_host (line ~403)
  - In the use_fft=False branch a caller passing transfer_to_device=False (data already on GPU) gets a stale/empty lsp_c back, and transfer_to_host=False cannot suppress the copy. Defaults (both True) mask the bug. One-line fix.
- [x] **use_cufinufft=True silently ignored by module-level lomb_scargle_async when cufinufft is not installed** — FIXED: ImportError raised at the top of lomb_scargle_async (same message as the class init); monkeypatched HAS_CUFINUFFT test asserts the raise. Commit: 7647d3e
  - Evidence: cuvarbase/lombscargle.py:380 — verified at HEAD: `if use_cufinufft and HAS_CUFINUFFT:`; the ImportError guard exists only in LombScargleAsyncProcess.__init__ (lines 451-454)
  - A direct call to the module-level function with use_cufinufft=True on a system without cufinufft silently runs the custom-NFFT path with no warning — silent behavior substitution rather than an error. Fix: raise or warn at the module-level gate.
- [x] **PDM CPU reference functions mutate caller's input arrays in place** — FIXED: `t = t - np.mean(t)` (copies) in binless_pdm_cpu, pdm2_cpu, pdm2_single_freq; TestCpuFunctionsDoNotMutateInputs asserts inputs unchanged (failed pre-fix). Commit: 047cb65
  - Evidence: cuvarbase/pdm.py:87-88, 100-101, 112-113 — verified at HEAD: binless_pdm_cpu, pdm2_cpu, pdm2_single_freq all do `t -= np.mean(t); y -= np.mean(y)` on their arguments
  - Public module-level functions modify the user's float arrays as a side effect (no copy). The GPU run() path is unaffected (normalize_light_curves copies). Fix is `t = t - np.mean(t)` in three places.
- [x] **Dead always-true 'power of 2' assert in _reduction_max leaves an unguarded silent-wrong-results path** — FIXED: dead assert replaced with _validate_block_size() (ValueError on non-power-of-2/<32/non-int); also fixed latent float division `grid_size / nfreq` → `//`; TestReductionMaxValidation with fake kernel (failed pre-fix). Commit: 047cb65
  - Evidence: cuvarbase/bls.py:159-161 — verified at HEAD: `assert(block_size - 2 * (block_size / 2) == 0)` (always true under Python 3 true division); acknowledged in commit 66739b4 message
  - The main validation hole was fixed by _validate_block_size in compile_bls (bls.py:277-289), but _reduction_max still trusts its block_size argument: a caller passing precompiled `functions` with a mismatched block_size kwarg gets silently wrong tree reductions (kernels require the compiled power-of-two size). Low severity (expert-path misuse only) but a trivial fix: delete the dead assert and validate at the call site.

## B. Experimental debt — fix or formally cut/document

- [x] **TLS: hard-coded n_t0=30 epoch grid misses narrow transits (Keplerian mode effectively broken for P > ~3.5 d)** — FIXED: duration-scaled t0 grid in both kernels (device t0_grid_size(): stride = duration/3, floor 30, cap 20,000), Python mirror tls_grids.t0_grid_size() as the documented contract; TestT0GridDurationScaled (scaling, circular coverage guarantee, kernel-source check; 3/3 fail pre-fix). GPU recovery test on the audit scenario queued. Commit: 77e32b9
  - Evidence: cuvarbase/kernels/tls.cu:277-279 and 433-435 (int n_t0 = 30); disclosed in import warning tls.py:19-28 and README.md:180-185
  - Both TLS kernels test only 30 epochs per period; transit windows narrower than 1/30 of phase mostly never overlap a tested epoch (audit simulation: 8/8 epochs missed at P=100d). Fix (duration-scaled t0 stride) deferred to the v1.1 rework. Dedupe note: the consolidated 'TLS module' finding was folded into this and the two following items.
- [x] **TLS: shared-memory layout caps ndata at ~3,500 with no launch-time guard; TLS_GPU_README body still claims 100,000-point support** — FIXED: ValueError guard before kernel compile (accounts for ndata, n_template, block_size); both TLS_GPU_README claim lines corrected; TestSharedMemoryGuard (2 tests, CPU). Commit: fdfd01a
  - Evidence: cuvarbase/tls.py:536 (shared_mem_size = (3*ndata + n_template + 4*block_size)*4); verified at HEAD: docs/TLS_GPU_README.md:128 ('Support datasets up to ~100,000 points') and :224 still assert the claim the line-7 banner calls aspirational
  - TESS (~20K) and Kepler (~65K) light curves exceed the 48KB shared-memory budget and fail at kernel launch; no Python-side ValueError guard exists. Minimum v1.0 action even if TLS stays experimental: add the guard and fix the two README body lines.
- [x] **TLS: chi2=1e30 sentinel for failed periods corrupts SDE/FAP (no host-side masking)** — FIXED: _mask_failed_periods() warns + excludes sentinels from argmin/SDE/FAP (raises if all fail); failed periods are NaN in returned chi2/power/SR with valid_periods + n_failed_periods keys; TestFailedPeriodMasking (4 tests incl. SDE-restoration). End-to-end GPU check queued. Commit: fdfd01a
  - Evidence: cuvarbase/kernels/tls.cu:265,417; no 1e30/isfinite/mask handling in cuvarbase/tls.py or cuvarbase/tls_stats.py
  - Failed periods write the 1e30 initializer into the chi2 output; audit reproduced SDE collapsing 15.3 -> 0.06 and FAP -> 1.0. Disclosed in the import warning; masking fix deferred to v1.1 — but host-side masking is cheap and would defuse the worst statistic corruption now.
- [x] **TLS: signal_to_noise inflated by sqrt(n_transits)** — FIXED: factor removed (chi2-based depth_err already covers all in-transit points); n_transits param retained but documented deprecated/unused; TestSnrNotInflated. Commit: fdfd01a
  - Evidence: cuvarbase/tls_stats.py:177 (snr = depth / depth_err * np.sqrt(n_transits))
  - Audit-confirmed SNR inflation unchanged at HEAD. Part of the descoped TLS stats surface; on the v1.1 rework list.
- [x] **TLS: FAP 'empirical calibration' constants are invented approximations attributed to Hippke & Heller** — FIXED: attribution removed; docstring now carries a warning block stating the heuristic is hand-rolled/uncalibrated and recommends injection-recovery (docs change, no test). Commit: fdfd01a
  - Evidence: cuvarbase/tls_stats.py:214-225 (piecewise 10**(-0.5*(SDE-5)) / 10**(-(SDE-5)) attributed to Hippke & Heller 2019 Fig 5)
  - false_alarm_probability ships hand-rolled constants the audit found invented. Docstring says 'approximate' and recommends injection-recovery, but the function returns authoritative-looking numbers with a false citation. If TLS ships experimental, at least remove the attribution.
- [x] **TLS: bitonic sort provably incomplete for non-power-of-2 sizes (wasted GPU work, misleading naming)** — REMOVED: the sort's output order was never consumed (depth/chi2 accumulations are order-independent), so the O(N log²N)-per-period sort was deleted from both kernels along with the misleading *_sorted names and the unused MAX_NDATA=100000 define; results unchanged up to float summation order. TestNoBitonicSort guards the removal; kernel-compile check covered by the queued TLS items. Commit: c964ead
  - Evidence: cuvarbase/kernels/tls.cu:47-55 (comparator-skipping bounds check 'ixj < ndata && i < ndata' breaks the bitonic network invariant)
  - Arrays are only permuted, not sorted, for non-power-of-2 ndata. Audit found downstream code permutation-invariant (harmless to results) — pure wasted work plus misleading names. Lowest-priority TLS item.
- [x] **TLS: no golden accuracy test vs transitleastsquares; the 5 batman-dependent tests were skipped in the v1.0.0 GPU gate** — TESTS WRITTEN: test_tls_golden.py compares period/depth/SDE against the reference transitleastsquares package on identical data (2 configs incl. the narrow-transit case) plus a no-reference narrow-transit recovery test on the audit scenario; skip cleanly on CPU. EXECUTION is in the pod batch (install batman-package + transitleastsquares). Commit: 9329ac1
  - Evidence: analysis/v1.0.0-gpu-validation/README.md — verified: '568 passed, 5 skipped... The 5 skips are batman-package tests'; skipif markers at cuvarbase/tests/test_tls_basic.py:127,136,149,170
  - test_tls_basic.py exists but the promised accuracy comparison against the reference transitleastsquares package does not, and the batman tests never ran on GPU (optional dep not installed on the pod). TLS shipped experimental with zero end-to-end GPU accuracy validation. If TLS is not cut, install batman on the validation pod and rerun before release.
- [x] **tls_models silently swallows all batman exceptions and substitutes a trapezoid template** — FIXED: _warn_template_fallback() warns with the failure reason in all four silent-fallback paths (broad except + 3 degenerate-model cases); TestTemplateFallbackWarns (monkeypatched batman failure). Commit: fdfd01a
  - Evidence: cuvarbase/tls_models.py:355-356 (broad `except Exception:` returning trapezoid fallback); silent fallback also at :319; only the missing-package case warns (import-time, :22)
  - Any batman failure at call time (bad params, numerical issue) silently degrades template quality with no warning or log. The only broad except in the package outside tests. Add a warnings.warn in the except before release or as part of the v1.1 TLS rework.
- [x] **NUFFT-LRT: all computation on CPU — 6 compiled CUDA kernels never invoked (GPU+nvcc required for nothing); README body still says 'GPU Accelerated'** — CUT from the wheel per standing decision: module/kernel/tests/examples/docs removed, source preserved on feature/nufft-lrt-experimental (pushed), README+CHANGELOG updated (credit kept, points at branch), close-out note posted on issue #36, lazy-import regression test asserts the package no longer exposes it. Commit: 96b6f7b
  - Evidence: cuvarbase/nufft_lrt.py:240-243 (np.interp + np.fft.rfft), 178-204 (_compile_and_prepare_functions still called from run()); docs/NUFFT_LRT_README.md:40,92-93 body text contradicts its own banner; disclosed in import warning nufft_lrt.py:16-24
  - compute_nufft is host interpolation+rfft while the module compiles unused kernels at run time. Rewire-to-cunfft deferred (planned with contributor Taaki). The formerly circular tests WERE fixed (136b06b). Dedupe note: consolidated 'NUFFT-LRT module' finding folded into this and the next item. Note: GitHub issue #36 that motivated the feature was closed at merge, so no open tracker item records this commitment.
- [x] **NUFFT-LRT: uniform grid spans only median(dt)*2N from t.min(), silently ignoring later data — wrong results for the advertised multi-season use case** — RESOLVED by the cut (see previous item); the grid-span defect is documented in the #36 close-out note and the CHANGELOG. Commit: 96b6f7b
  - Evidence: cuvarbase/nufft_lrt.py:306 (nf = 2*len(t) default), 234-240 (tu = t0 + dt*arange(nf); np.interp with left/right=0)
  - For gappy/multi-season baselines most of the light curve never enters the computation — audit reproduced output unchanged when season-2 data was perturbed. Silently wrong, not erroring; disclosed only in the import warning. This is the strongest argument for cutting the module from the wheel rather than fencing it.
- [ ] **eebls_transit sparse/standard discontinuity: sparse path ignores qmin_fac/qmax_fac/use_fast and searches all q in (0, 0.5]**
  - Evidence: cuvarbase/bls.py:1755-1763 (docstring warning), 1798-1816 (implementation + runtime UserWarning + kwargs whitelist silently dropping others)
  - Merged two sweep findings (audit + code-marker). The kwargs TypeError crash was fixed (ae0af5d) and the loud UserWarning works as intended, but per-frequency q bounds are never passed to the sparse kernels, so power values and best solutions change qualitatively at the arbitrary ndata=500 threshold. Functional fix (q bounds in sparse kernels) outstanding; 'loudly document' was the accepted v1.0 remedy — confirm that stance or fix.
- [ ] **keplerian_freq_grid does not return q values; per-frequency q bounds not wired into eebls_gpu_batch**
  - Evidence: cuvarbase/bls_frequencies.py:90 (returns freqs only); cuvarbase/bls.py:1886 (eebls_gpu_batch takes scalar qmin/qmax); audit §3 quick wins
  - The GPU batch kernel already supports per-frequency q bounds, but the grid helper returns only frequencies, so batch users cannot run duration-constrained Keplerian searches. Audit-listed quick win, never done.
- [ ] **Eager `import pycuda.autoprimaryctx` makes `import cuvarbase` require a working GPU, contradicting advertised CPU fallbacks**
  - Evidence: cuvarbase/__init__.py:2; cuvarbase/bls.py:14; cuvarbase/ce.py:19; cuvarbase/tls.py:30; README.md:127,132,167 advertise sparse_bls_cpu / use_gpu=False fallbacks
  - Importing any module creates a CUDA primary context, so the README-advertised CPU paths are unreachable on GPU-less machines (CI sidesteps via pycuda stubs). Also pins device 0 at import, complicating multi-GPU/fork. The skcuda lazy-import work explicitly does not cover this. Minimum release action: correct the README claims; real fix is lazy context creation. Related: closed issue #31 (re-evaluate PyCUDA) was closed with this debt outstanding.
- [ ] **GPU Lomb-Scargle supports only 1 harmonic ('right now' admission) despite CPU multiharmonic helpers in the same file**
  - Evidence: cuvarbase/lombscargle.py:449 — `raise Exception("Only 1 harmonic is supported right now")`; mhdirect_sums/mhgls_from_sums implement multiharmonic GLS on CPU; README.md:200 lists multiharmonic GLS under Planned Features
  - Feature gap honestly listed as planned; the bare Exception type is covered by the error-handling hygiene item in bucket D. Keep as formally-deferred or implement in the packed release.
- [ ] **eebls_gpu_fast has no noverlap parameter ('yet') — phase-binning bias when optimal q is near qmin**
  - Evidence: cuvarbase/bls.py:536-542 — verified docstring admission at HEAD ('There is no noverlap parameter here yet...'); related admitted limits: shared-memory lower bound on q (515-517), OS kernel-time-limit timeouts (523-529)
  - Workaround (re-running noverlap times with shifted dphi) is pushed onto the user. Documented limitation of the fast path; fix or keep formally documented.
- [ ] **NFFT filter radius m chosen by a heuristic the code itself flags as wrong**
  - Evidence: cuvarbase/cunfft.py:270-274 — TODO: should use L1 norm of true Fourier coefficients (NFFT3 guide p.11); verified the only TODO/FIXME marker in the entire package
  - estimate_m()'s truncation-error bound is admitted inaccurate, affecting custom-NFFT Lomb-Scargle accuracy when autoset_m is in effect. Long-standing; document the limitation or implement the proper bound.
- [ ] **Public API stubs raising NotImplementedError: ConditionalEntropyAsyncProcess.memory_requirement and LombScargleMemory.is_ready**
  - Evidence: cuvarbase/ce.py:287-292 (docstring: 'Will throw a NotImplementedError if called, so ... don't call it.'); cuvarbase/memory/lombscargle_memory.py:192-194 (vs implemented NFFTMemory.is_ready at memory/nfft_memory.py:120-127)
  - Visible API asymmetry (the LS memory_requirement counterpart is implemented). Implement, remove, or document before the packed release.
- [ ] **Fast CE kernels incompatible with weighted CE; mag_overlap incompatible with balanced_magbins (loud guards over real feature gaps)**
  - Evidence: cuvarbase/ce.py:222-228 — guards raise on use_fast+weighted and mag_overlap+balanced_magbins; CE maintenance notice at ce.py:5-11
  - Guards are correct behavior and are gate-tested, but the combinations remain unimplemented. Since CE is formally in maintenance mode with a periodfind referral, the realistic disposition is 'formally cut': document the unsupported combinations in ce.rst and move on.
- [ ] **#33 PDM follow-ups publicly promised in our own issue comment: batched multi-LC API, memory-capped large_run, perf benchmark — all absent**
  - Evidence: Issue #33 comment checklist (2026-06-11); cuvarbase/pdm.py has no batched_run_const_nfreq or large_run equivalent at HEAD
  - Three unchecked public commitments verified still absent. Feature gaps, not bugs (shipped PDM kernels pass the gate). Either implement in the packed release or amend the issue comment to re-scope. Note: the fourth checkbox (GPU validation) is verified DONE — see the tracker-reconciliation item in bucket D.
- [ ] **FFA-BLS: formally cut (negative result, ~14x slower); only residual decision is whether the spec doc ships**
  - Evidence: docs/FBLS_GPU_SPEC.md:3-11 STATUS banner ('EXPERIMENT COMPLETED — NEGATIVE RESULT'); no cuvarbase/ffa_bls.py or kernels/ffa_bls.cu at HEAD; code lives on feature/ffa-bls-experimental
  - Verified correctly descoped — nothing user-facing advertises FFA. This item is already in the 'formally cut' end-state; the only release decision is whether docs/FBLS_GPU_SPEC.md belongs in the sdist/docs build.

## C. Performance / undiagnosed — diagnose, fix, or document honestly

- [ ] **eebls_gpu_batch large-ndata regression: ~12x slower than the single-LC loop for TESS-like data — undiagnosed, unguarded, and contradicted by its own docs**
  - Evidence: benchmarks/results/benchmark_results_new_features.json (TESS-1sector batch_speedup=0.0847, Kepler 0.87); cuvarbase/bls.py:1886-1900 — verified at HEAD the docstring advertises only benefits, no caveat; docs/BENCHMARK_RESULTS.md:127-131 prose says 'as fast or faster' while its own table shows 12x slower; CHANGELOG.rst soft hint only; eebls_gpu_batch absent from docs/source/bls.rst
  - Merged three sweep findings (audit-disputed diagnosis, perf records, docs gap). Root cause never profiled (nsys/ncu planned for v1.1; the launch-config hypothesis was refuted). No commits to bls_batch.cu or eebls_gpu_batch since the Feb 2026 benchmark. Minimum v1.0 action: add docstring + Sphinx warning, fix the contradictory BENCHMARK_RESULTS prose, and consider a runtime warning or ndata-based fallback to the single-LC path. Diagnosis itself can stay v1.1 if documented.
- [x] **cuFINUFFT LS backend 0.63-0.84x the speed of the custom NFFT: per-call Plan creation, no caching, never destroyed — and its module docstring claims the opposite** — FIXED: LRU plan cache keyed on (nf_total, eps, n_pts, gpu_method) with cap 8 + free_plan_cache() for eager release; gpu_method exposed as a documented kwarg; module docstring rewritten honestly (cross-check backend, custom kernel faster in benchmarks); use_cufinufft documented in LombScargleAsyncProcess. TestCufinufftPlanCache (3 tests w/ fake plans, all fail pre-fix). Speedup re-measurement queued for the pod batch. Commit: 62ce387
  - Evidence: cuvarbase/cufinufft_backend.py:118-130 — verified at HEAD: fresh gpuarray output + cufinufft.Plan per invocation (called twice per periodogram, lombscargle.py:382-383); benchmark_results_new_features.json cufinufft_vs_custom 0.63-0.84 across 8 configs; cufinufft_backend.py:4-7 docstring claims '~10-100x faster spreading throughput'; use_cufinufft documented in no docstring or Sphinx page
  - Merged four sweep findings (audit, code-marker x2, perf records). Plan caching keyed on (nf_total, eps, n_pts) is the identified fix to flip the backend past 1x; gpu_method=1 is hard-coded and Plans rely on GC for GPU resource release. Docs side: BENCHMARK_RESULTS.md/CHANGELOG are honest but the module docstring is misleading and the parameter is undocumented in the API. Either implement caching or ship clearly labeled as a cross-check backend with the docstring corrected.
- [ ] **Page-locked (pinned) host buffers never restored — allocate_pinned_arrays is a misnomer and async transfers silently serialize**
  - Evidence: cuvarbase/bls.py:395-417 — 'allocate_pinned_arrays' uses cuda.aligned_zeros (aligned, NOT page-locked); no pagelocked_*/register_host_memory anywhere in cuvarbase/; audit attributes removal to a false premise in commit 4e6e232
  - set_async/get_async fall back to synchronous staged copies, defeating the multi-stream architecture. Restore cuda.pagelocked_* (or register_host_memory) or rename and document the behavior.
- [x] **sparse_bls_cpu still pure-Python O(N^2 x Nf) nested loops — unusable CPU fallback** — FIXED: prefix-sum + broadcast vectorization (the old loop was actually O(N^3): per-pair slice sums); now ~3 ms/freq at ndata=500 vs minutes before. Passes all existing brute-force equivalence/wrapping/optimality tests unchanged; TestSparseBlsCpuVectorized adds a perf regression guard (ndata=250 in seconds; old code times out). Commit: d7e2b43
  - Evidence: cuvarbase/bls.py:1406, 1470-1507 (for i in range(ndata) / for j in range(i+1, ndata+1)); audit §3 medium item 'vectorize (~100x)'
  - This is the CPU path backing sparse ground-truth comparisons and the README-advertised no-GPU fallback (which is itself unreachable — see the eager-pycuda item in B). Numpy-cumsum vectorization is a known ~100x, contained change.
- [ ] **Adaptive block-size heuristic considers only ndata, not nbins**
  - Evidence: cuvarbase/bls.py:45-67 (_choose_block_size(ndata)); audit §3; block_size validation was centralized (66739b4) but the heuristic is unchanged
  - The adaptive kernel selection (headline 1.4-5.3x claim) keys solely on ndata; nbins-driven occupancy effects are ignored. Diagnose/extend or document the heuristic's domain.
- [ ] **Published LS survey throughput requires batch_size=1, but batched_run_const_nfreq defaults to batch_size=10 and no doc says so**
  - Evidence: cuvarbase/lombscargle.py:754 — verified at HEAD: `def batched_run_const_nfreq(self, data, batch_size=10,`; scripts/benchmark_new_features.py:860 ('batch_size=1 is fastest — avoids multi-stream overhead'); docs/BENCHMARK_RESULTS.md:38
  - All published numbers (4.4 ms/LC ZTF, 19.8 ms/LC Kepler, nifty-ls comparisons in README) were measured at batch_size=1; the only record is a benchmark-script comment. Users following the docs (lomb.rst uses defaults) will not reproduce published throughput. Fix: change the default or add explicit guidance; the underlying multi-stream overhead at batch_size>1 is itself undiagnosed.

## D. Process, docs, and claims

- [ ] **LS benchmark claims not re-validated against astropy 8.0 (LRA-NUFFT default); astropy version unpinned in benchmark docs**
  - Evidence: Audit §6 risk 1; analysis/v1.0.0-gpu-validation/README.md — verified gate env ran astropy 7.2.0; docs/BENCHMARK_RESULTS.md pins no astropy version
  - The risk register requires re-running LS comparisons on astropy 8.0 before publishing v1.0 claims, and pinning versions in benchmark docs. Feb-2026 numbers stand un-rebenchmarked. Must resolve (re-run or version-caveat the tables) before PyPI publicizes the comparisons.
- [ ] **No benchmark vs CETRA — comparative GPU-transit-search claims must stay off the table**
  - Evidence: Audit §6 risk 2; no CETRA benchmark anywhere in the repo
  - CETRA (PLATO's named detection algorithm) is the live competitor. Docs currently avoid comparative claims, so this constrains release messaging rather than code. Keep the constraint explicit in release notes/announcements until a benchmark exists.
- [ ] **scikit-cuda remains the cuFFT backend for LS/NFFT; full replacement publicly promised 'post-1.0' (issue #63)**
  - Evidence: Issue #63 open (our 2026-06-11 comment promises replacement via cupy.cuda.cufft or direct cuFFT binding); cuvarbase/_skcuda_compat.py shim + PEP 562 lazy imports already shipped and CI-verified
  - Merged two sweep findings. Mitigations (numpy-2.x shim, lazy imports so BLS/CE/PDM never touch skcuda) are done — do not re-fix. The open commitment is the actual replacement for the LS/NFFT runtime dependency on abandoned scikit-cuda 0.5.3. Decide: land in the packed release or let the post-1.0 promise stand and say so in release notes.
- [ ] **bls_optimized.cu remains a near-duplicate of bls.cu — kernel-drift hazard with a prior shipped-bug precedent**
  - Evidence: Verified at HEAD: cuvarbase/kernels/bls.cu and cuvarbase/kernels/bls_optimized.cu both present; precedent: reduction_max s>32 bug fixed in only one copy originally (77b4333)
  - Audit medium item to merge the kernels not done. The duplication already produced one silent-wrong-results bug; the structural risk is live. Merge via templating or add a CI check diffing the shared sections.
- [ ] **Error-handling hygiene: assert-based validation vanishes under python -O; user input errors raised as bare Exception**
  - Evidence: Asserts: cuvarbase/lombscargle.py:43,335,590,721; cuvarbase/ce.py:146,482,561; cuvarbase/memory/nfft_memory.py:113,122-127. Bare Exceptions: lombscargle.py:449; ce.py:224,228; bls.py:1788
  - Merged two sweep findings (same cleanup pass). Under -O all input validation in the main LS/CE run paths silently disappears; when asserts fire users get messageless AssertionError; bare Exception cannot be caught precisely. Convert to ValueError/RuntimeError with messages.
- [ ] **Bus-factor/release-ritual items pending: JOSS paper, ASCL record update, co-maintainer onboarding**
  - Evidence: Audit §5 Phase 3 step 13 and §6 risk 5; no JOSS/ASCL artifacts in repo at HEAD
  - Explicit gameplan items tied to the release: ASCL record update, JOSS submission for citability, engaging astrobatty (PDM contributor, PR #62) as co-maintainer. All open.
- [ ] **v1.0.0 tag not merged to master; PyPI publish deferred; local master stale behind origin**
  - Evidence: git: origin/master at 060d839 (~176 commits behind v1.0.0 tag/HEAD 5553248); local master at ec53ae8, behind origin/master
  - Intentional per the one-packed-release strategy, but the default GitHub branch serves pre-v1.0 code and local master needs a fast-forward at release. Phase 3 step 12 (merge, build wheel, smoke-test, publish) is the closing move.
- [ ] **Sphinx autodoc covers no post-0.2.6 modules**
  - Evidence: docs/source/cuvarbase.rst — verified at HEAD: automodule entries only for bls, ce, core, cunfft, lombscargle, pdm, utils
  - tls/tls_grids/tls_models/tls_stats, nufft_lrt, bls_frequencies, cufinufft_backend, and the base/ and memory/ subpackages have no API pages (pdm.rst content itself is done and wired). If TLS/NUFFT-LRT stay experimental, at least add bls_frequencies and the memory subpackage; conf.py builds cleanly so this is mechanical.
- [ ] **Benchmark claim traceability: adaptive-kernel numbers (1.4-5.3x; 90x for ndata<64) and the '~1 second' fBLS comparison have no raw data in the repo**
  - Evidence: README.md:103-104 + docs/BLS_OPTIMIZATION.md:15,51-65 (prose tables only; scripts/benchmark_adaptive_bls.py output JSON never committed); docs/BENCHMARK_RESULTS.md:98 (~1s figure, no provenance; closest repo data is ~0.18 s/LC Kepler single-LC)
  - Merged two sweep findings. README.md:60 promises all headline numbers are 'traceable to benchmark data in this repository' — these two are not. Fix by committing the adaptive-vs-fixed JSON (rerun scripts/benchmark_adaptive_bls.py on the next pod session) and either sourcing or rephrasing the fBLS sentence (the real number is conservative in cuvarbase's favor).
- [ ] **GitHub tracker reconciliation at release: #33 checkbox (verified done), #15 (verified done, close after docs rebuild), #19 (verified: tables but no plots), #32 (close-as-wontfix per CE descope), #28 (status comment)**
  - Evidence: Verified this session: scripts/check_release_gate.py:208-253 + analysis/v1.0.0-gpu-validation/README.md (gate 14/14 on RTX A5000 2026-06-11; GPU pytest suite ran the 9 PDM tests covering all four fast kernels) satisfies #33's validation checkbox. docs/source/pdm.rst exists and is wired (index.rst:20 toctree; cuvarbase.rst:57 automodule) satisfying #15's content. grep of docs/BENCHMARK_RESULTS.md finds zero image/figure references — #19's literal 'plots' ask is unmet (tables only). ce.py:5-11 maintenance notice contradicts open #32.
  - Five tracker items collapse into one release-day hygiene pass: tick #33's validation checkbox; close #15 once the docs site rebuilds from the released tag; close #19 (decide: close-with-tables or add 2-3 figures from the by_gpu JSONs); close #32 as wontfix/descoped referencing the periodfind referral; post a #28 status comment summarizing v1.0 against each modernization goal. Also consider a note on closed #36 pointing to the NUFFT-LRT experimental status (see gaps). The two 'possibly-fixed-verify' raw items (#33 checkbox, #15) were verified fixed in substance and dropped as standalone entries.
- [ ] **#17: BLS/LS power-spectrum convention — docstrings now state 1 - chi2/chi2_0, but bls.rst has no convention discussion and selectable conventions were never implemented**
  - Evidence: Issue #17 open; cuvarbase/bls.py:979,1195,1770,2138 (docstring convention statements); docs/source/bls.rst lacks any convention/astropy comparison
  - Clarity half is partially done (docstrings only); the user-selectable-conventions half is unimplemented. Cheap pre-release win: a short convention section in bls.rst (and lomb.rst) comparing to astropy; defer selectable conventions explicitly on the issue.
- [ ] **#29: Documentation refactor umbrella (docstring audit, example notebooks) incomplete and un-scoped**
  - Evidence: Issue #29 open, no comments narrowing scope
  - Real progress shipped (PDM docs, README significance pass, benchmark docs) but the full ask (all-public-API docstring audit, updated example notebooks) is not done. Pre-release action: comment on the issue stating what v1.0 delivered and what is deferred.
- [ ] **#30: Code-convention standardization + CONTRIBUTING guidelines untouched**
  - Evidence: Issue #30 open, no comments, no targeted work on the branch
  - Renaming public API immediately before a major release would be churn; the realistic pre-release action is shipping a CONTRIBUTING file and an explicit deferral comment. No defect attached.

## Completeness-pass notes (sources checked beyond the 4 sweeps)

- Closed GitHub issues (gh issue list --state closed, 11 issues reviewed): the sweeps covered only the open tracker. Findings: #36 (NUFFT-LRT) was closed when the feature merged even though the shipped module is CPU-only and experimental — the closure hides the GPU-rewire commitment, and the contributor's comment ('the adaptive spectral estimator component needs some work though') is an additional admission tracked nowhere; recommend a closing note or reopen at release. #18 (pdot) cleanly closed as abandoned with rationale; #54 (kernel path) fixed by commit 6e9c127; #31 (re-evaluate PyCUDA) closed while the eager-import debt persists (covered by an open B item). No new errors found.
- CHANGELOG.rst full read for a hidden 'Known Issues' section: none exists; caveats are inline ('best for ndata < ~1000 per lightcurve', 'the custom NFFT kernel remains faster', experimental warnings) and consistent with the inventory. The CHANGELOG's claims themselves check out against the GPU validation record. No new items.
- Test-suite skip/xfail patterns (grep over cuvarbase/tests + conftest): only importorskip/skipif for nfft (installed in CI per 19e37cc), batman (the 5 GPU-gate skips already inventoried in bucket B), pycuda, and NUFFT_LRT_AVAILABLE; the GPU validation record confirms NUFFT-LRT and PDM tests actually ran on hardware. No xfail markers and no skips hiding known failures beyond the batman/TLS gap already captured.
- Adversarial-review commit 39d3367 message body: the 10-agent verification pass 'confirmed five issues; all fixed here' — pdm block_size OOB, normalize_light_curves None crash, ce.large_run overflow, pdm.rst claim, missing per-arch benchmark table. No deferred leftovers declared. Gates re-run after fixes (112 passed CPU).
- Full code-marker rescan (TODO/FIXME/XXX/HACK across cuvarbase/*.py, kernels/*.cu, memory/, base/): exactly one marker exists — cunfft.py:270 (estimate_m), already in the inventory. The code-marker sweep was complete.
- Review comments on merged community PRs #57-62 (gh pr view): only 'LGTM'-class comments; no acknowledged-but-deferred defects hiding in PR threads.
- NOT checked (recommend before release): side branches feature/ffa-bls-experimental and tls-gpu-implementation for known-issue notes that never made it to the tracker; readthedocs build warnings for the new pages; a clean-environment Sphinx build of pdm.rst (verified wired into the toctree but not rendered this session, no sphinx locally); docs/source/whatsnew.rst vs CHANGELOG.rst consistency.
