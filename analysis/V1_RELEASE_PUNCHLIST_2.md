# v1.0 release punchlist #2 (June 12, 2026)

Maintainer decision 2026-06-12: **all previously-deferred "v1.1+"
debt items are now v1.0 requirements.** This punchlist absorbs
analysis/V1_FINAL_TASKS.md (T1 docs, T2 benchmarks) and promotes the
entire T4 deferred-debt inventory. Predecessor:
analysis/V1_RELEASE_PUNCHLIST.md (closed, 45/45; GPU validation
608/608 in analysis/v1.0-rc-gpu-validation/).

Rules of engagement (carried from punchlist #1):
- Checking a box requires: fix committed, a test that fails before /
  passes after (where testable), commit hash noted, GPU-dependent
  verification queued below (batched pod sessions — never one pod per
  item).
- Work lands on v1.0-fixes; fast-forward v1.0; CI green before the
  next item.
- NEVER without explicit maintainer go: publish to PyPI, merge
  master, move the v1.0.0 tag, touch b8-* RunPod pods.

⚠️ DECISIONS NEEDED (conflict with earlier standing decisions — set
these in the loop prompt before starting):
- **D1 NUFFT-LRT**: earlier decision was CUT from the wheel. The
  deferred-item promotion implies reinstating it after a GPU rewire
  (item C3). Confirm: rework & reinstate for v1.0, or keep cut?
- **D2 API renaming (#30)**: earlier disposition (posted publicly on
  the issue hours ago) deferred renaming to a major cycle with
  deprecation aliases. Confirm: actually rename for v1.0, or keep
  deferred?
- **D3 Benchmark protocol**: analysis/BENCHMARK_PROTOCOL_V1.md awaits
  sign-off; pod spend gated on it. Confirm approved (≲$5, ~1 day).
- **D4 CETRA benchmark**: promoting the deferred "benchmark vs CETRA"
  item adds a CETRA comparison to the campaign (different algorithm —
  framed as time-to-equivalent-detection, not power comparison).
  Confirm in/out of scope for v1.0.

## GPU verification queue
(batch on ONE pod when ~5+ accumulate or all CPU-side work done;
include `pip install batman-package transitleastsquares cufinufft`,
`apt-get install rsync` before setup-remote.sh; terminate + verify
via API; archive in analysis/)
**Batch 1 (Jun 13 2026, RTX A5000, pod ydqi9luioem03s — terminated +
verified; fixes in c6baf13): results in analysis/v1.0-gpu-batch-jun2026/.**
Suite 660 passed / 2 failed (the 2 = A1+A3, found + fixed here); gate
ALL PASSED.
- [~] (standing) full suite + check_release_gate.py +
      benchmark_new_features.py --tests-only — suite green (after A1/A3
      fixes) + gate green; benchmark_new_features **A) BLS batch
      correctness FAILS** at small ndata (pre-existing, routed to E1).
      Re-confirm on the final RC.
- [x] A1: sparse q-bounds GPU parity — FOUND BUG: sparse_bls_simple.cu
      didn't compile (qmin_f/qmax_f undefined; missing qmin_arr/qmax_arr
      in signature). Fixed (mirror full kernel); both
      test_sparse_bls_gpu_q_bounds[True/False] pass on A5000.
- [x] A2: noverlap multi-pass — green in suite.
- [x] A3: autoset-m tolerance — FOUND OVER-CLAIM: realized NFFT error
      floors ~1e-3 (deconv/precision), so tol=1e-6 unachievable; L1
      bound governs truncation only. Fixed test (achievable tol +
      assert closed-form m) + docstring + CHANGELOG; passes on A5000.
- [x] A4: benchmark_block_size.py full grid run → A4 CLOSED (document;
      median 3.9%, >10% only at atypical qmin>=0.02). JSON in
      benchmark_results_by_gpu/block_size_a5000.json.
- [x] A5: power conventions — green in suite.
- [x] A6: bls_common.cuh single-source — both kernels compile + run;
      include mechanism works on pod. Green in suite + gate.
- [x] B1: lazy CUDA context — import creates no context; context on
      first GPU use across every path; full suite green with real
      pycuda. No "no active context" errors.
**Batch 2 (Jun 13 2026, RTX A5000, pod 2baj5kk9z5p0zq — terminated +
verified): all GREEN. Results in analysis/v1.0-gpu-batch2-jun2026/.**
Full suite 671 passed / 7 skipped; release gate ALL PASSED.
- [x] B2: in-house cuFFT binding — LS/NFFT suite green; test_nfft
      FFT-vs-fftpack passes (binding ifft correct); gate
      cufftEstimate1d path passes; perf vs scikit-cuda max |ratio-1| =
      2.4% (within ±10%). → scikit-cuda dep + _skcuda_compat + numpy
      shim DROPPED; #63 resolvable.
- [x] B3: pinned host buffers — suite green with pinned=True (no
      regression); pinned H2D 1.4–2.85x faster than page-aligned
      (overlap demonstrated); fallback path intact.
- [x] C1: PDM batch — batched_run_const_nfreq + large_run match per-LC
      run() at corr=1.000000; GPU PDM == CPU pdm2_cpu (corr=1.0);
      benchmark JSON committed (benchmark_results_by_gpu/pdm_a5000.json,
      1006–12622x vs pure-Python CPU).
  (bug found+fixed: benchmark_pdm.py queried the device before B1's lazy
  context existed.)

**Batch 3 queue (accumulating; CPU-side landed, GPU-pending):**
- [ ] C2: multiharmonic GLS — on the A5000, confirm the real ghat_g
      spectrum layout matches the verified convention: run
      LombScargleAsyncProcess(nharmonics=H) for H=2,3 and assert
      corr>0.999 vs lomb_scargle_direct_sums(nharms=H) on the same grid.
- [ ] C3: NUFFT-LRT rewire — on the A5000: NUFFTLRTAsyncProcess.run
      actually executes the NFFT on device (no GPUStubError; profile/
      confirm kernels invoked); compute_nufft output matches the CPU
      adjoint-DFT reference (corr>0.999); the restored GPU tests pass;
      multi-season detection works end-to-end on device.

## A. Contained code items (do first)

- [x] **A1. Sparse-path per-frequency q bounds** — wire qmin/qmax
      (per-frequency arrays) into sparse_bls.cu kernels AND
      sparse_bls_cpu, so eebls_transit's sparse path honors
      qmin_fac/qmax_fac/Keplerian constraints; remove the
      discontinuity UserWarning once behavior matches across the
      sparse_threshold boundary. Accept: CPU sparse honors q bounds
      (brute-force test with bounded q); GPU matches CPU; the
      eebls_transit warning is retired; docs updated (bls.rst,
      docstrings). GPU queue: sparse kernel parity test.
      **DONE 665dbbd** — qmin/qmax (scalar or per-frequency) added to
      both kernels + sparse_bls_cpu/gpu; eebls_transit passes
      Keplerian bounds through and the UserWarning is retired;
      bounded brute-force parity + per-frequency + validation +
      no-warning tests added; bls.rst sparse section updated. GPU
      parity queued (signature change → pod must recompile kernels).
- [x] **A2. noverlap for eebls_gpu_fast** — add the noverlap
      parameter (phase-offset oversampling) to the fast path,
      removing the documented dphi re-run workaround. Accept:
      eebls_gpu_fast(noverlap=k) matches the k-shifted-dphi manual
      procedure; docstring admission removed.
      **DONE 9fb7b1e** — the kernels' noverlap arg was a silent no-op
      in the compiled (non-LOG) branch; implemented as noverlap
      dphi-shifted passes combined by on-GPU elementwise max in a
      shared _eebls_gpu_fast_impl (standard + optimized now share one
      body). Docstring admission removed; noverlap documented +
      validated (ValueError, CPU-side test); CHANGELOG entry (also
      retired the stale A1 sparse-warning line). Behavior note:
      default noverlap=2 now really does 2 passes (~2x kernel time);
      noverlap=1 restores the old behavior.
- [x] **A3. estimate_m L1-norm truncation bound** — implement the
      NFFT3-guide bound (the package's only TODO, cunfft.py); keep
      the old heuristic as fallback flag if the bound is costly.
      Accept: unit test comparing achieved NFFT error vs requested
      tol on synthetic data (CPU nfft reference); docstring warning
      replaced with the real bound's statement.
      **DONE 6df6bfb** — estimate_m(N=None, y=None): with y, m = the
      smallest integer with 4·exp(-m·D)·||y||_1 <= tol (rigorous
      absolute bound; the bound is O(N) so no cost flag needed); the
      N-based heuristic remains the no-data fallback (used by the LS
      buffer-sizing call sites, documented). cunfft.allocate passes y.
      Local tests: bound rigor + minimality across tol/sigma/scale,
      fallback equivalence, monotonicity, zero-data, validation
      (test_nfft_m.py, 23 cases). GPU queue: autoset-m tolerance test
      vs direct sums. TODO + docstring warning replaced.
- [x] **A4. nbins-aware block-size heuristic** — extend
      _choose_block_size to consider nbins (qmin) occupancy; or
      demonstrate empirically (pod microbenchmark) that ndata-only is
      within ~10% of best and document that instead. Accept: data-
      backed either way; heuristic doc updated.
      **PREPPED 90bac8d** — scripts/benchmark_block_size.py sweeps
      (ndata × qmin × block_size) on both fast kernels with
      preallocated memory (kernel-only timing), reports per-cell
      heuristic-vs-best penalty + >10% offenders. Decision (extend
      heuristic vs document) and the box close on the pod data —
      queued below.
      **CLOSED (Jun 13, A5000) — DECISION: DOCUMENT.** Full-grid run
      (benchmark_results_by_gpu/block_size_a5000.json): median penalty
      **3.9%**, max **30%**, 13/40 cells >10% — ALL at qmin>=0.02
      (mostly qmin=0.1, i.e. large duration fraction / few bins; best
      block 64 vs heuristic 256). Typical transit search (q~0.01-0.05)
      stays within ~10%. Chose to document rather than add a qmin-aware
      heuristic (extra complexity + GPU re-validation for atypical-only
      gain); block_size is user-overridable. Docstring note pending in
      G-phase docs pass (analysis/v1.0-gpu-batch-jun2026/SUMMARY.md has
      the data). No code change.
- [x] **A5. Selectable power conventions (#17)** — add a
      `convention=` kwarg ('chi2ratio' default, 'snr', 'loglik'?)
      to the BLS entry points mapping the existing outputs;
      document equivalences vs astropy objectives in bls.rst.
      Accept: conversions unit-tested against astropy on shared
      grids; issue #17 closable at release.
      **DONE 4f82e24** — convert_bls_power() + convention= on
      eebls_gpu, eebls_gpu_custom, fast impl (+3 wrappers via
      kwargs), eebls_gpu_batch (per-LC), sparse_bls_cpu/gpu,
      eebls_transit (both paths; transit_gpu inherits via kwargs).
      Derived + verified vs astropy method='slow' on shared
      solutions: 'snr' = sqrt(chi2_0·P) EXACTLY equals astropy
      objective='snr'; 'loglik' = chi2_0·P/2 (constant-mean
      reference); astropy objective='likelihood' = ours/(1-r)
      (out-of-transit reference) — relation tested with r from the
      solution mask. bls.rst section rewritten; CHANGELOG; #17
      closable at release (H2). GPU queue: kwarg-flow smoke test.
- [x] **A6. Kernel templating merge (bls.cu/bls_optimized.cu)** —
      single-source the shared device functions (Jinja-style include
      via _module_reader cpp_defs or a common .cuh inlined at load);
      keep the drift-guard test as the invariant. Accept: shared
      functions defined once; both kernels compile + gate passes on
      pod; drift test simplified to assert the include mechanism.
      **DONE 5ee3a88** — added a Python-side `//{INCLUDE bls_common.cuh}`
      directive (expanded by utils._module_reader at load time; nvcc
      never sees an #include since pycuda compiles from the assembled
      string). The 13 shared device/global functions now live once in
      kernels/bls_common.cuh; bls.cu keeps only full_bls_no_sol +
      full-tree reduction_max, bls_optimized.cu only
      full_bls_no_sol_optimized + warp-shuffle reduction_max
      (mod1_fast→mod1, identical body). Drift test rewritten to assert
      the include mechanism (directive present, shared funcs defined
      once and never redefined in either .cu, directive expands). No
      behavior change: every assembled function body is byte-identical
      (normalized) to the pre-refactor HEAD (verified in-script). GPU
      queue: both kernels must compile + gate pass on pod (the assembled
      source changed shape).

## B. Architecture items

- [x] **B1. Lazy CUDA context creation** — remove eager
      `import pycuda.autoprimaryctx` from cuvarbase/__init__.py and
      module tops; initialize the primary context on first GPU use
      (helper in core/base; honor CUDA_DEVICE). Accept:
      `import cuvarbase` + sparse_bls_cpu/single_bls/fap_baluev run
      on a GPU-less machine WITHOUT the conftest stubs (new CI job
      proves it: pip install pycuda is still required at import? —
      goal: no CUDA context, document whether pycuda-the-package
      remains an import dependency); all GPU paths still pass on pod;
      README CPU-helper caveat updated/removed.
      **DONE ccd5bc9** — new helper `cuvarbase.base.ensure_context()`
      (base/context.py) retains the primary context lazily on first GPU
      use (defers to pycuda.autoprimaryctx; CUDA_DEVICE honored via its
      make_default_context). Eager import removed from __init__.py +
      bls/ce/tls tops; wired ensure_context() into GPUAsyncProcess
      .__init__ (covers ce/pdm/cunfft/lombscargle processes), the 4 BLS
      compile fns incl. _get_cached_kernels (cache-hit self-guarantee),
      compile_tls, all 6 *Memory __init__ (BLS/BLSBatch/CE/NFFT/LS/TLS),
      cufinufft_nfft_adjoint, and the .device reads. An 11-module
      adversarial gap-hunt (analysis/b1-lazy-context-audit-jun2026.json)
      confirmed NO module does GPU work at import and found 2 blockers
      (LombScargleMemory/TLSMemory direct construction) + memory
      edge-cases, all closed by the *Memory __init__ guards. pycuda
      package still required by GPU modules (import pycuda.driver),
      documented in README + CHANGELOG. New CPU contract tests
      (test_lazy_imports): import without pycuda; no context until first
      GPU use (CPU helper single_bls verified context-free). Packaging
      smoke (ci_wheel_smoke.py) now proves GPU-less import with pycuda
      genuinely absent. GPU queue: full GPU suite must pass on pod with
      real pycuda (context lifecycle exercised).
- [x] **B2. scikit-cuda replacement (#63)** — replace skcuda.fft
      (cuFFT) in cunfft.py/lombscargle.py with cupy.cuda.cufft OR a
      minimal direct cuFFT ctypes binding (decide by spike: cupy adds
      a heavy dep; direct binding is ~200 lines for C2C 1D batched).
      Keep _skcuda_compat shim until removal is complete, then drop
      skcuda from deps. Accept: LS/NFFT suite green on pod with
      scikit-cuda UNINSTALLED; perf within ±10% of skcuda baseline
      (measure both); #63 closable; CHANGELOG known-limitation
      removed. Supersedes the deferral comment posted on #63
      (post follow-up at release).
      **IMPLEMENTED (CPU) 1194127 — box open pending GPU validation +
      dep drop.** DECISION: direct ctypes binding over cupy (cupy is a
      heavy CUDA-version-specific dep; the cuFFT surface used is 3 calls;
      skcuda was itself a ctypes binding). New cuvarbase/_cufft.py binds
      libcufft (Plan/fft/ifft/cufftEstimate1d, C2C+Z2Z, lazy lib load);
      cunfft/lombscargle/nfft_memory + test_nfft now use it, so NO module
      imports scikit-cuda (lazy-import test flipped: LS/NFFT import with
      skcuda broken). Adversarially reviewed vs the cuFFT C API (5 dims,
      web-verified, 0 blockers; analysis/b2-cufft-binding-review-jun2026
      .json) — all signatures/constants/pointers correct; 4 minor
      lib-discovery findings fixed (glob pip-wheel + toolkit lib64,
      RTLD_GLOBAL, LD_LIBRARY_PATH-aware error, atexit __del__ guard).
      Suite 197 passed; flake8 clean. **VALIDATED on A5000 (batch 2,
      c77bd86):** test_nfft FFT-vs-fftpack passes, full LS/NFFT suite
      green, gate cufftEstimate1d path passes, perf vs scikit-cuda max
      |ratio-1| = 2.4% (within ±10%). → scikit-cuda DROPPED from
      pyproject + setup.py; cuvarbase/_skcuda_compat.py removed; CHANGELOG
      limitation removed + LS/NFFT feature bullet added; #63 closable at
      release (H2). Wheel imports cleanly without scikit-cuda.
- [x] **B3. True pinned host buffers** — restore page-locked memory
      (cuda.pagelocked_empty or register_host_memory) in
      BLSMemory/BLSBatchMemory/NFFT/LS/CE memory classes behind a
      `pinned=True` default with graceful fallback; rename docs
      accordingly (allocate_host_arrays docs already honest).
      Accept: async transfer overlap demonstrated on pod (CUDA-event
      timeline or bandwidthTest-style measurement showing
      async-vs-sync delta); suite green; no regression for
      non-pinned fallback.
      **DONE c40f9be** — new cuvarbase/memory/_host.py:host_array(shape,
      dtype, pinned=True) uses cuda.pagelocked_zeros with graceful
      fallback to cuda.aligned_zeros (warns once). Wired into all 6
      *Memory classes (BLS/BLSBatch/NFFT/LS/CE/TLS — each gains a
      pinned=True kwarg) + the PDM result buffer; removed the now-dead
      `import resource` from those modules. Fallback design: attempts the
      aligned allocator inside the pinned except-handler and only warns
      if it succeeds, so a GPU-less run (both stubbed) propagates + skips
      cleanly with no spurious warning. test_host_array.py (4) covers
      pinned-default / fallback+warn / no-warn-when-fallback-fails /
      pinned=False; the batch-API honesty test flipped from "NOT
      page-locked" to page-locked-by-default-with-fallback. Suite 201
      passed; flake8 clean. GPU queue: demonstrate async-vs-sync overlap
      + confirm no fallback regression on pod.

## C. Feature completion items

- [x] **C1. PDM batch API + large_run + benchmark (#33)** —
      batched_run_const_nfreq-equivalent for PDMAsyncProcess,
      memory-capped large_run, and a PDM GPU-vs-CPU benchmark
      (add to campaign scenarios). Accept: batch matches per-LC
      results; large_run respects max_memory on pod; benchmark JSON
      committed; #33 checkboxes closable (supersedes the re-scope
      comment — post follow-up at release).
      **DONE 254f219** — PDMAsyncProcess.batched_run_const_nfreq
      (chunked, shared-grid, memory-bounded: peak mem ~ batch_size, not
      len(data); correct-by-construction = per-chunk run() with results
      copied out) + large_run (auto batch_size from 90% free GPU mem via
      cuda.mem_get_info; _bytes_per_lc/_batch_size_from_memory factored).
      scripts/benchmark_pdm.py: GPU(PDMAsyncProcess) vs CPU(pdm2_cpu)
      correctness (theta-corr + recovery) + (ndata×nfreq) throughput,
      JSON out, --tests-only. test_pdm_batch.py (4 CPU tests via mocked
      run): batch-size arithmetic, chunking+const-freq reuse, empty,
      large_run dispatch. Suite 205 passed; flake8 clean. NOTE: chose a
      chunked (reallocate-per-batch) design over LS-style buffer reuse —
      lower risk given no local GPU, same memory-bound + const-grid win;
      buffer reuse is a possible future optimization. GPU queue +
      benchmark-JSON commit pending batch 2; #33 closable at release.
- [x] **C2. Multiharmonic GLS on GPU** — extend the LS kernel to
      nharmonics>1 (the CPU helpers mhdirect_sums/mhgls_from_sums
      already define the math; kernel computes the 2H-sums via NFFT
      of higher harmonics — same NFFT plan at h*f). Accept: GPU
      multiharmonic matches the existing CPU mhgls reference
      (corr>0.999) for H=2,3 on pod; NotImplementedError removed;
      README planned-features updated.
      **DONE 8d5a1aa** — scoping spike (Workflow, analysis/c2-
      multiharmonic-gls-spike-jun2026.json) confirmed the math + that the
      memory grids are ALREADY sized for it (w to 2H, yw to H) and yw is
      already mean-centered. DECISION: HYBRID — GPU NFFT emits the
      spectra (validated), host does the per-freq 2H×2H solve in float64
      via the existing tested mhdirect_sums/mhgls_from_sums (avoids an
      untestable float32 in-kernel Cholesky; H>1 not a hot path).
      Factored _mh_assemble_from_centered (shared by mhdirect_sums + the
      new _mh_power_from_spectra which reads moments at index
      (m-1)*k0+m*i); wired into lomb_scargle_async for nharmonics>1;
      NotImplementedError → ValueError(nharmonics<1); removed dead lomb_mh
      kernel + README planned line. test_mhgls_hybrid.py (CPU): hybrid ==
      lomb_scargle_direct_sums to machine precision for H=2,3 + refactor
      equivalence + construct-without-raise. Suite 209 passed; flake8
      clean. GPU queue: one smoke-test of the real ghat_g layout.
- [x] **C3. ⚠️ D1: NUFFT-LRT GPU rewire** (only if D1=reinstate) —
      wire the existing compiled kernels (preserved on
      feature/nufft-lrt-experimental) into compute_nufft via cunfft;
      fix the grid-span defect (uniform grid must cover the full
      baseline or use the NFFT path); restore module + tests to the
      wheel; coordinate/credit @xiaziyna. Accept: GPU path actually
      executes on device (profiled); multi-season test (perturbing
      late-season data changes output); accuracy vs CPU reference.
      **DONE 83d5356** — scoping spike (analysis/c3-nufft-lrt-spike-
      jun2026.json) → rewire compute_nufft to the GPU adjoint NFFT
      (self.nufft_proc.run([(t,y,nf)])[0]), which fixes BOTH cut defects
      at once: it runs on-device (the old path computed a host
      uniform-grid RFFT; kernels never invoked) AND covers the full
      non-uniform baseline (the old median(dt)*nf grid truncated
      multi-season data). Weights→all-ones + PSD over all nf bins (NFFT
      modes are all physical, freq k/(tmax-tmin)); dead matched-filter
      kernels left unwired (host combine is O(nf)) + no longer compiled
      in run(). Restored module+kernel+3 tests+docs+example to the wheel;
      __init__ re-exposes NUFFTLRTAsyncProcess/Memory; removal test
      inverted; README/CHANGELOG reframed + @xiaziyna credited. CPU
      verification (test_nufft_lrt_pipeline.py): full pipeline via a
      direct adjoint-DFT (exact NFFT math) is sensitive to late-season
      data (grid-span fix) + runs end-to-end. Suite 229 passed; flake8
      clean. Still EXPERIMENTAL (warning) pending injection-recovery.
      GPU queue: NFFT executes on device + accuracy vs the CPU
      adjoint-DFT reference.

## D. TLS science-ready (beyond punchlist-1 fixes)

- [ ] **D1. Expose t0 fidelity** — make T0_OVERSAMPLE a Python-level
      parameter (kernel #define via cpp_defs); document the
      sensitivity/speed trade (reference TLS uses ~33x finer
      stepping). Accept: parameter plumbed + tested; default
      documented.
- [ ] **D2. Lift the ~3,500-point cap** — tile the shared-memory
      layout (chunked data passes or global-memory fallback kernel)
      so native TESS 10-min/200-s cadence fits; keep the fast path
      for small ndata. Accept: ndata=12,000 runs on pod, matches
      binned-equivalent results within tolerance; guard message
      updated to the new bound; QLP-feasibility note updated.
- [ ] **D3. TLS injection-recovery validation** — campaign on pod:
      injected transits across (P, depth, ndata) grid, recovery vs
      reference transitleastsquares at matched fidelity; decide
      experimental-flag removal on results. Accept: validation
      report in analysis/; warning text updated to reflect validated
      domain (or kept with documented gaps).

## E. Diagnosis items

- [ ] **E1. eebls_gpu_batch large-ndata regression** — profile on pod
      (nsys via pip nvidia-nsight-systems or apt cuda-nsight-systems;
      fallback: CUDA-event stage timing inside the batch path);
      identify root cause; fix it OR implement automatic
      single-LC-path fallback above the crossover; update the
      runtime warning/docs to the diagnosis. Accept: TESS-scale
      batch ≥ parity with single-LC loop, or auto-fallback +
      documented root cause.
      **SCOPE EXPANDED (Jun 13 GPU batch):** also a *correctness*
      divergence at SMALL ndata, not just large-ndata perf —
      benchmark_new_features.py A) BLS batch correctness fails:
      eebls_gpu_batch vs eebls_gpu_fast_adaptive give ndata=200
      corr=0.77 peak_match=5/10, ndata=2000 corr=0.97 peak_match=9/10
      (ndata=20000 passes). Pre-existing (bls_batch.cu untouched this
      session). E1 must explain + fix the batch path's small-ndata
      disagreement too (or document the regime where batch is valid).
      Details: analysis/v1.0-gpu-batch-jun2026/SUMMARY.md.
- [ ] **E2. LS batch_size>1 multi-stream overhead** — same treatment:
      stage timing, root cause, fix or document; revisit the
      batch_size=1 default if fixed.

## F. Benchmark campaign (T2; gated on D3 sign-off)

- [ ] **F1. Execute analysis/BENCHMARK_PROTOCOL_V1.md** (after B/C/D
      items that affect perf land — campaign measures the final RC):
      7 scenarios, BLS v1.0 vs origin/master vs astropy; TLS vs
      transitleastsquares at two fidelities; QLP tables; raw JSON +
      env pins committed.
- [ ] **F2. ⚠️ D4: CETRA comparison** (only if in scope) — install
      CETRA, design time-to-equivalent-detection framing (different
      algorithm: no power comparison), add as scenario S8.
- [ ] **F3. Rewrite README/BENCHMARK_RESULTS claims from new data**;
      retire superseded numbers (21-390x pre-v1.0 note, adaptive
      claims already corrected).

## G. Documentation refresh (T1; finale after APIs settle)

- [x] **G1. Keplerian citations** (T1.a — full insertion list in
      V1_FINAL_TASKS.md): SM03 + Ofir 2014 across bls.py,
      bls_frequencies.py, bls.rst; fix 4 wrong Ofir titles; reconcile
      fmax0 8.6307 vs 8.612 + derived-constant note. (Independent of
      API changes — can run early.)
      **DONE b13deec** — both titles web-verified: SM03 = "A Unique
      Solution of Planet and Star Parameters from an Extrasolar Planet
      Transit Light Curve" (ApJ 585, 1038); Ofir 2014 = "Optimizing the
      search for transiting planets in long time series" (A&A 561, A138,
      arXiv:1307.7330) — the wrong title that 3 spots carried is
      actually Hippke & Heller 2019's real title. Added [SM03]_/[O2014]_
      to bls.py module docstring + new docstrings on
      q_transit/freq_transit/fmax_transit0/fmin_transit/fmax_transit
      (with the fmax0 = sqrt(G·rho/3pi) derivation) + transit_autofreq
      Notes (Ofir eq. 4); bls_frequencies.py module/_q_transit/
      keplerian_freq_grid; bls.rst (shortcut→SM03, period-spacing→O2014,
      8.612↔8.6307 reconciliation, References section with targets).
      Fixed wrong Ofir titles: tls_grids.py, TLS_GPU_README.md (×2),
      TLS_GPU_IMPLEMENTATION_PLAN.md (was a GW title). New test
      test_keplerian_relations.py (3): fmax0 derived-constant check,
      q/freq round-trip (SM03 inverse), and a citation guard that fails
      if H&H's title is pasted on Ofir again (fails-before: tls_grids.py
      had it). Suite 192 passed; flake8 clean. No GPU dep.
- [x] **G2. README content fixes** (T1.b): PyPI v0.2.5 blocker
      handling, selling-point reorder (QLP/257-354x/$33 to first
      screenful; BibTeX + personal note down), periodograms/ claim,
      notebooks/ pointer, Testing-section fix, misc. (Reorder only;
      no voice changes.)
      **DONE 2cbe81c** — full-file rewrite verified to preserve every
      prose block verbatim (content-diff vs HEAD: 19 dropped lines all
      intentional edits, 22 added all intentional — big blocks moved
      intact). Performance/QLP/257-354x/$33 now the first screenful;
      Citation BibTeX + Personal Note + Future Plans moved below the
      technical sections (promoted to ## since no longer under About).
      Fixes: removed stale PyPI badge + replaced `pip install cuvarbase`
      with `git+...@v1.0` (the 0.2.5 blocker); dropped the nonexistent
      `periodograms/` module claim; examples→notebooks/ pointer;
      Testing section now says CPU-runnable (conftest stubs, CI);
      "What's New" no longer framed relative to master; citation-count
      de-dated (~two dozen, late 2025); added cufinufft optional extra;
      singular "module" for experimental TLS; 3 http→https ADS links;
      and corrected the sparse-BLS bullet that still claimed import
      requires a GPU (now reflects B1). New test_readme_consistency.py
      (5 guards, fails-before: README had all 5 issues). Suite 197
      passed; flake8 clean. No GPU dep.
- [ ] **G3. Sphinx sources + conf.py** (T1.c): drop
      only_directives, add autodoc mocks, fix install.rst/ce.rst/
      lomb.rst/figure scripts/tau typo/fap stub/complexity claim,
      modernize conf.py vestiges; add pages for new v1.0 APIs.
- [ ] **G4. gh-pages rebuild + clean republish**: modern Sphinx,
      SAME alabaster theme/logo/sidebars; orphan commit purging the
      169 MB of junk; regenerate the 4 GPU figures on the final pod
      session; keep .nojekyll; THEN update README/pyproject doc
      links. Clean-env render check.
- [ ] **G5. Docstring audit + notebooks (#29 full scope)** — all
      public APIs docstring-audited; the 3 walkthrough notebooks
      re-run against v1.0 APIs (on pod) and committed with outputs;
      #29 closable.

## H. Release closing moves (unchanged; explicit go required)
- [ ] H1. master merge, tag move, wheel build + smoke test, PyPI
- [ ] H2. tracker pass: #14 #15 #17 #19 #28 #29 #30 #32 #33 #63
      closures/updates per landed work
- [ ] H3. JOSS (now further motivated: QLP DRN 003 credits cuvarbase
      only via a GitHub footnote — nothing citable), ASCL update
      (ascl:2210.030 exists), co-maintainer invite

## Explicitly NOT in scope (aspirational roadmap — say the word)
GPU Fast Template Periodogram; astropy method= registration; LSDB
worked example; wavelet transforms; PDM/GLS spectrograms; astropy-8.0
LS re-run (blocked: not released).
