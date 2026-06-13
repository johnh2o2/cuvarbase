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
- [ ] (standing) full suite + check_release_gate.py +
      benchmark_new_features.py --tests-only green on the final RC
- [ ] A1: sparse q-bounds GPU parity — test_sparse_bls_gpu_q_bounds
      (full + simple kernels) and the full sparse GPU test group must
      pass on pod (kernel signature changed: +qmin_arr/+qmax_arr)
- [ ] A2: noverlap multi-pass — TestEeblsGpuFastNoverlap GPU tests
      (manual-dphi equivalence for standard + optimized, monotonic
      power) + full fast-path test group (refactor touched both entry
      points)
- [ ] A3: autoset-m tolerance — test_autoset_m_l1_bound_meets_tolerance
      (float64 tol=1e-6, float32 tol=1e-2) vs direct sums; validates
      the tighter-m direction (||y||_1 < N)
- [ ] A4: run scripts/benchmark_block_size.py (full grid, both
      kernels) on the A5000; commit JSON to
      benchmark_results_by_gpu/; then close A4 (extend heuristic if
      any cell >10%, else document)
- [ ] A5: test_gpu_entry_points_convention (kwarg flows through
      eebls_gpu + eebls_gpu_fast chains; host-side conversion
      identity) + TestPowerConventions group on pod (pip install
      astropy there)
- [ ] A6: both BLS kernels must compile after the bls_common.cuh
      single-source refactor — run the full BLS GPU test group
      (standard + optimized + adaptive paths) and check_release_gate.py
      on pod; confirm `//{INCLUDE}` expands correctly under the editable
      install path (the MultiplexedPath gotcha from memory)
- [ ] items accumulate here as work proceeds

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
- [ ] **A4. nbins-aware block-size heuristic** — extend
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
      **DONE bf2c34b** — added a Python-side `//{INCLUDE bls_common.cuh}`
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

- [ ] **B1. Lazy CUDA context creation** — remove eager
      `import pycuda.autoprimaryctx` from cuvarbase/__init__.py and
      module tops; initialize the primary context on first GPU use
      (helper in core/base; honor CUDA_DEVICE). Accept:
      `import cuvarbase` + sparse_bls_cpu/single_bls/fap_baluev run
      on a GPU-less machine WITHOUT the conftest stubs (new CI job
      proves it: pip install pycuda is still required at import? —
      goal: no CUDA context, document whether pycuda-the-package
      remains an import dependency); all GPU paths still pass on pod;
      README CPU-helper caveat updated/removed.
- [ ] **B2. scikit-cuda replacement (#63)** — replace skcuda.fft
      (cuFFT) in cunfft.py/lombscargle.py with cupy.cuda.cufft OR a
      minimal direct cuFFT ctypes binding (decide by spike: cupy adds
      a heavy dep; direct binding is ~200 lines for C2C 1D batched).
      Keep _skcuda_compat shim until removal is complete, then drop
      skcuda from deps. Accept: LS/NFFT suite green on pod with
      scikit-cuda UNINSTALLED; perf within ±10% of skcuda baseline
      (measure both); #63 closable; CHANGELOG known-limitation
      removed. Supersedes the deferral comment posted on #63
      (post follow-up at release).
- [ ] **B3. True pinned host buffers** — restore page-locked memory
      (cuda.pagelocked_empty or register_host_memory) in
      BLSMemory/BLSBatchMemory/NFFT/LS/CE memory classes behind a
      `pinned=True` default with graceful fallback; rename docs
      accordingly (allocate_host_arrays docs already honest).
      Accept: async transfer overlap demonstrated on pod (CUDA-event
      timeline or bandwidthTest-style measurement showing
      async-vs-sync delta); suite green; no regression for
      non-pinned fallback.

## C. Feature completion items

- [ ] **C1. PDM batch API + large_run + benchmark (#33)** —
      batched_run_const_nfreq-equivalent for PDMAsyncProcess,
      memory-capped large_run, and a PDM GPU-vs-CPU benchmark
      (add to campaign scenarios). Accept: batch matches per-LC
      results; large_run respects max_memory on pod; benchmark JSON
      committed; #33 checkboxes closable (supersedes the re-scope
      comment — post follow-up at release).
- [ ] **C2. Multiharmonic GLS on GPU** — extend the LS kernel to
      nharmonics>1 (the CPU helpers mhdirect_sums/mhgls_from_sums
      already define the math; kernel computes the 2H-sums via NFFT
      of higher harmonics — same NFFT plan at h*f). Accept: GPU
      multiharmonic matches the existing CPU mhgls reference
      (corr>0.999) for H=2,3 on pod; NotImplementedError removed;
      README planned-features updated.
- [ ] **C3. ⚠️ D1: NUFFT-LRT GPU rewire** (only if D1=reinstate) —
      wire the existing compiled kernels (preserved on
      feature/nufft-lrt-experimental) into compute_nufft via cunfft;
      fix the grid-span defect (uniform grid must cover the full
      baseline or use the NFFT path); restore module + tests to the
      wheel; coordinate/credit @xiaziyna. Accept: GPU path actually
      executes on device (profiled); multi-season test (perturbing
      late-season data changes output); accuracy vs CPU reference.

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

- [ ] **G1. Keplerian citations** (T1.a — full insertion list in
      V1_FINAL_TASKS.md): SM03 + Ofir 2014 across bls.py,
      bls_frequencies.py, bls.rst; fix 4 wrong Ofir titles; reconcile
      fmax0 8.6307 vs 8.612 + derived-constant note. (Independent of
      API changes — can run early.)
- [ ] **G2. README content fixes** (T1.b): PyPI v0.2.5 blocker
      handling, selling-point reorder (QLP/257-354x/$33 to first
      screenful; BibTeX + personal note down), periodograms/ claim,
      notebooks/ pointer, Testing-section fix, misc. (Reorder only;
      no voice changes.)
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
