# cuvarbase 1.0: merged execution plan (September 2026)

Single entry point for the release work. It merges the two read-only audits run
on `v1.0-fixes` @ `7d55ea2` (Sep 1-4 2026):

- `RELEASE_READINESS.md`: release hygiene (173 findings; 100 two-lens verified,
  7 confirmed inline, 66 finder-only). Data: `release_findings.json`.
- `ALGORITHM_AUDIT.md`: per-method soundness and speed (163 findings; 25 of 26
  blocker/high correctness topics confirmed on an RTX 4090, 1 downgraded,
  0 refuted; performance and medium/low items are auditor-measured only).
  Data: `algorithm_findings.json`, `algorithm_verify_topics.json`.
- `campaign/`: the NUFFT-LRT injection-recovery campaign (seed 20260711, all
  four configurations, run to completion on the pod).
- `repro/`: every auditor and verifier experiment script (`repro/pod/` ran on
  the pod against `/workspace/cuvarbase`; `repro/local/` are the local copies
  and CPU checks). They are the starting point for the regression tests below.

Numbers in this file are quoted from those reports; "finding N" refers to
`RELEASE_READINESS.md` appendix ids, "defect N" to `ALGORITHM_AUDIT.md`
section 2, and "id N" to `algorithm_findings.json`.

Standing rules (maintainer directives, unchanged): no merge to `master`, no tag
moves, no PyPI upload, no GitHub Release, no `gh-pages` push until the explicit
go coordinated with @astrobatty. Everything else below is authorized
("all of this sounds fine to fix + do", Sep 3).

Pod recipe that worked (RTX A5000 preferred; 4090 fallback landed at $0.74/hr):
`scripts/runpod-create.sh "NVIDIA RTX A5000"` (run unpiped; check "SSH ready"),
then on the pod `apt-get install -y rsync`, then `scripts/setup-remote.sh`,
then `pip install batman-package transitleastsquares cufinufft`. Drive it with
a thin ssh wrapper that exports the CUDA env and never rsyncs
(the audit used `pod.sh '<cmd>'`); clone by SHA for the gate, never rsync.
Terminate with `scripts/runpod-stop.sh --terminate` and confirm `myself{pods}`
is empty.

---

## Phase 0: decisions (assumed defaults, override any of them)

| # | Question | Assumed answer (panel recommendation) |
|---|---|---|
| D1 | NUFFT-LRT in 1.0 | Aim for an official release: fix defects 5, 6, 21, 22, 24 and the per-template allocation, re-run the campaign (Phase 4), then decide official vs experimental on the numbers. If re-validation does not land before the freeze: ship import-warned EXPERIMENTAL, quarantined (not in `__all__`), with the BJD and `epochs=None` caveats in the docs. Message the contributor before tagging either way. |
| D2 | Archive strategy | Prune in place; one annotated tag `archive/pre-1.0-process` on the last pre-prune commit; `analysis/README.md` index; no archive folder, no orphan branch. |
| D3 | Compatibility shims and API | Keep-with-DeprecationWarning only what shipped in 0.2.5 (`core.py`, `allocate_pinned_arrays`, PDM 4-tuple, `GPUAsyncProcess(reader=, function_kwargs=, device=)`); remove what never shipped (`bls` `__getattr__` fallback, `tls_stats` `window_length` / `n_transits`, `tls_search_gpu(durations=)`, `pink_noise_correction`); `T0` = absolute time everywhere plus `t0_phase`; one top-level namespace; input validation that raises. |
| D4 | Gate and branches | Full gate re-run on a frozen commit T cloned by SHA; gate record as the only later commit T' (analysis/ only); `--no-ff` merge into `origin/master` taking `v1.0-fixes` for all four conflicts, asserted tree-identical; delete and recreate `v1.0.0`; fast-forward `origin/v1.0`; keep `master` as default. |

---

## Phase 1: correctness fixes (CPU + pod; about 3 agent-days)

Each item: fix, then a regression test derived from the named `repro/` script,
then a CHANGELOG line when the default-path result changes ("changes results").
Work in dependency order inside each module. Every item that changes results is
why Phase 5 must re-run the whole gate.

### BLS

| Defect | Root cause | Fix | Test from | Changes results |
|---|---|---|---|---|
| 1 `bls-overflow-oob` (blocker) | `bls_common.cuh:12-14` `get_id()` is `unsigned int`; `:304`, `:349` bound `i < ndata*nfreq` in 32 bits; host launches the exact product (`bls.py:1581`, `:1320`) and never caps the auto batch (`:1504`, `:1247`); bin buffers sized by `count_tot_nbins(global nbins0_max, nbinsf_max)`, not an upper bound over batches | 64-bit index (or cap `freq_batch_size <= (2**31-1)//ndata` and `<= len(freqs)`); size bin buffers by the maximum over the actual batches; raise a clear error instead of overrunning | `repro/pod/exp4_overflow.py`, `repro/local/blsperf/*` (262,800-pt 2-min year; `fmin=0.02, fmax=0.5` on 70k points) | only calls that overflow or crash today |
| 7 `bls-q-collapse` (high) | `eebls_gpu` reduces per-frequency q arrays to one batch-wide (min, max) | Preferred: make `eebls_transit` default to the fused fast kernel (honours per-frequency bounds) with a top-K `eebls_gpu_custom`/`single_bls` pass for (q, phi); otherwise pass the per-frequency bounds to the multifreq kernel | `repro/pod/exp3_qcollapse.py`, `exp10b_batchdep.py` | yes (default `eebls_transit` for ndata >= 500) |
| 8 `bls-sparse-uncentered` (high) | `bls.py:2080` casts uncentered y to float32; `sparse_bls.cu:214-217` float32 prefix sums of uncentered w*y | Center y in float64 in the wrapper (and accumulate centered in the kernel) | `repro/local/sparse-batch/sparse_exp.py` | yes (default `eebls_transit` for ndata < 500) |
| 20 `bls-sparse-simple` (high) | `sparse_bls_simple.cu:4` `MAX_W_COMPLEMENT 1E-9` (pre-PR#65) | Delete `sparse_bls_simple.cu` and the `use_simple` plumbing (`bls.py:1985-2124`, `2305-2306`, `test_bls.py:817`); generalize `test_kernel_drift.py` to all kernel files (same-named `#define` comparison) | existing sparse tests | no (opt-in path removed) |

Also from the release audit (BLS): `eebls_gpu` allocates ~90% of free memory
per call (finding 135); fix together with defect 1 (perf BLS-2).

### TLS

| Defect | Root cause | Fix | Test from | Changes results |
|---|---|---|---|---|
| 2 `tls-duration-window` (blocker) | `tls_search_gpu`/`tls_search` without qmin/qmax use a fixed q window [0.005, 0.15] while the Ofir grid runs to span/2 | Default to Keplerian duration bounds per period (`tls_grids.q_transit`, stellar parameters) or cap the default period grid where the window becomes unphysical; document | `repro/pod/tlsaud_*` / `repro/local/tls-audit/*` (P = 365 d on a 1400-d light curve) | yes |
| 10 `tls-fap` (high) | `FAP` is a fixed piecewise function of SDE (discontinuous at SDE = 7) | Remove the key (or rename `fap_heuristic` with a docstring warning); document the null-SDE behaviour vs grid and baseline | 400-noise-LC null from the TLS audit | yes (key) |
| 11 `tls-T0` (high) | `tls.py:631` phase, `:800` phase relative to 0, `:1589` absolute; docstrings `:526-534`, `:906`; `docs/source/tls.rst:62`; `examples/tls_example.py:174, 225`; `test_tls_fast.py:128` | `T0` = absolute mid-transit time on every path plus `t0_phase`; fix docs, example, test; GPU assertion `T0` in `[min(t), min(t)+P]` | `repro/local/verify-tls-T0/*` | yes (key semantics) |

Medium items worth taking with these (report 2.4): SR definition vs the
reference (ids 81, 146), unsorted period grids corrupt SDE (id 82), medfilt
edge bias (id 83), coarse t0 grid losses documented (id 84), no free baseline
term (id 80, document).

### Lomb-Scargle and NFFT

| Defect | Root cause | Fix | Test from | Changes results |
|---|---|---|---|---|
| 3 `nfft-psi-table` (blocker) | `memory/lombscargle_memory.py:186-190` reuses the yw grid's psi tables for the 2x larger w grid | Precompute psi per grid size | `repro/local/verify-psi/*`, `repro/pod/exp1_gls.py` (target: float32 <= 2e-4, float64 <= 2e-8 vs astropy) | yes (every default LS call, 3e-3..2.4e-2) |
| 4 `nfft-k0-size` (blocker) | `lombscargle_memory.py:193-194` sizes grids `sigma*nf` while modes `k0..k0+nf-1` are read | Size grids from `k0 + nf` (or shift the band); reject grids the transform cannot represent | `repro/local/verify-k0size/*` (`fmin >= fmax/2`) | yes for such grids (today garbage) |
| 12 `nfft-absolute-time` (high) | `memory/nfft_memory.py:246` stores absolute t as float32 | Subtract a float64 epoch on the host before the cast (as BLS does via `utils.subtract_epoch`); document the phase convention | `repro/local/vfy-nfft-abs/*` | yes for absolute-time input |
| 13 `ls-nharmonics-nofft` (high) | `use_fft=False`/`python_dir_sums=True` ignore `nharmonics` | Implement the direct-sums multiharmonic path or raise `NotImplementedError` | `repro/local/verify-ls-nharm/*` | yes (that path) |
| 14 `ls-amplitude-prior` (high) | `amplitude_prior` not wired into the GPU multiharmonic path | Wire it (ridge term as in `add_regularization`) or raise | `repro/local/verify-ls-ampprior/*` | yes (that option) |
| 15 `ls-nonuniform-grid` (high) | `check_k0` (`lombscargle.py:35-43`) inspects `freqs[0:2]`; kernels evaluate `fmin + i*df` | Validate the whole grid is uniform (tolerance), raise otherwise | `repro/local/verify-ls-nonuniform/*` | no (raises where it was silently wrong) |
| `nfft-floorf-double` (high, defect in 2.2) | `cunfft.cu:151` `floorf()` on a double coordinate | `floor()` under `DOUBLE_PRECISION` (FLT-typed) | `repro/local/verify-nfft-floorf/*` | yes (`use_double=True`, 1.5e-2 -> 2e-8) |

Also: `LombScargleAsyncProcess.preallocate` leaves `memory.stream=None`
(stale reads 14/30; defect 19's sibling); `significance = 1 - FAP` saturates
(ids 96, 129, 144); `batched_run_const_nfreq(freqs=None)` drops the last
frequency (id 147); `NFFTAsyncProcess` docstring says sigma=2, code uses 4
(id 148); `NFFTAsyncProcess.run(memory=)` returns unsynchronized, un-zeroed
data (ids 118, 155; blocks perf LRT-1).

### Conditional entropy

| Defect | Root cause | Fix | Test from | Changes results |
|---|---|---|---|---|
| 9 `ce-brightest-bin` (high) | `setdata` normalizes y to [0, 1] and takes `floor(y*mag_bins)`; no kernel clamps index `mag_bins` | Clamp to `mag_bins - 1` at binning time (all kernels), guard in `setdata` | `repro/local/verify-ce-bin/*`, `repro/local/pdm-ce/*` (compare with the independent CPU CE) | yes (every unweighted run, O(1/N)) |
| 16 `ce-weighted-asym` (high) | `ce.cu:68-69` skips a bin by distance to its lower edge only | Use distance to the nearest edge (symmetric), keep a mass floor | `repro/local/verify-ce-weighted/*` | yes (weighted CE) |
| 17 `ce-double-fast-crash` (high) | shared-memory offset uses a byte remainder as an element offset when `(mag_bins+1)*phase_bins` is odd | Round the offset up to the element alignment (8 bytes for double) | `repro/local/verify-ce-dblfast/*` | no (crash only) |
| 18 `ce-balanced-ignored` (high) | constructor drops `balanced_magbins` / `widen_mag_range` | Plumb them to `run`/`setdata`; make weighted+balanced raise as documented; fix the test parametrization | `repro/local/verify-ce-balanced/*` | yes for callers who passed them |
| 19 `ce-preallocate` (high) | `preallocate()` never calls `transfer_freqs_to_gpu()` and leaves `memory.stream=None` | Upload freqs, set the stream; same for LS `preallocate` | `repro/local/vfy-ce-prealloc/*` | yes (was constant output) |

Also: CE recompiles its module on every call (`ce.py:495-498`, `:576-579`
`'ce_wt'` gate; perf CE-1); `set_data=False` accumulates histograms (id 112);
zero-width balanced bins give -inf (id 106).

### NUFFT-LRT

| Defect | Root cause | Fix | Test from | Changes results |
|---|---|---|---|---|
| 5 `lrt-bjd-float32` (blocker) | `nufft_lrt.py:411` and `compute_nufft` `:333` cast t to float32 before folding/gridding | Subtract a float64 epoch first (like BLS); BJD invariance test on all three detectors | `repro/local/vfy-lrt-bjd/*`, `repro/pod/lrtaud_exp3_epoch_bjd.py` | yes for absolute-time input |
| 6 `lrt-epochs-none` (blocker) | `epochs=None` evaluates one phase-0 template per (P, duration) | Build a duration-scaled epoch grid by default (or refuse `None`); fix the class docstring, README example 1 and `examples/nufft_lrt_example.py` which present it as a search | `repro/local/verify-lrt-epochs/*` (random-epoch recovery 12/12) | yes (default) |
| 21 `lrt-sequential-intercept` (high) | `_sequential_detrend` (`nufft_lrt.py:101-103`) OLS without intercept on un-centred y and V, before the demean at `:465-466` | Center V and y (or add an intercept column) before the fit | `repro/local/vseq/*` | yes (`detector='sequential'`) |
| 22 `lrt-detectorA-defeated` (high) | `estimate_psd=True` estimates the PSD from `y - V mu`, which still contains the realized systematics | Estimate the PSD from the basis-projected residual | `repro/local/vfy-detA/*` (completeness 0.13 -> 0.47 at depth 0.008 in the harness's data model) | yes (`detector='marginal'` default) |
| 24 `lrt-upper-half-band` (medium) | default `sigma=2` (`nufft_lrt.py:232`) leaves modes `k >= nf/2` aliased | `sigma=4` default (matches `NFFTAsyncProcess`) or a `2*nf` transform | `repro/local/verify-lrt-band/*` | small (< 2% z-scores) |

Also: zero and synchronize the NFFT buffer on the reuse path (ids 118, 155),
floor user PSDs and validate length (id 121), positive-definite prior check
instead of `pinv` (ids 122, 156), document the PSD convention and the 2-D
return shape when `epochs is None` (ids 107, 123, 173), warn that `dy` is
ignored, and state in the docs that the statistic's null std is 1.4-2.7 under
irregular sampling even with the true PSD (section 6.3): it is not N(0, 1).

### Cross-cutting

| Defect | Fix | Test from | Changes results |
|---|---|---|---|
| 23 `input-validation` (high) | `utils.check_lightcurve(t, y, dy)` and `check_freqs` raising `ValueError` with counts of offending entries, called before any GPU work in every BLS/TLS/LS/PDM/CE/LRT entry point; kernel-side guard for NaN q bounds; `N < 5` handled before launch (today `use_fast=True` with `N <= 4` or a NaN timestamp kills the CUDA context for the process) | `repro/local/vfy-inputval/*`, `repro/local/xcut/*` | no for valid input; raises where it was silently wrong |

Also: worker threads cannot launch kernels (id 124) and poisoned-context errors
are not translated (id 125); document both, and translate the sticky
`illegal memory access` into a clear message.

### PDM (no confirmed defects)

Document that the statistic is `1 - SS_within/SS_tot` without Stellingwerf's
degrees-of-freedom correction (pure-noise mean `(M-1)/(N-1)`), fix the notebook
and `docs/source/pdm.rst` formulas (id 110), and note the float32-only fold
(id 109). `binned_step` reads `bin_means[NBINS]` at phase exactly 1.0 (id 113):
clamp.

---

## Phase 2: performance quick wins (pod for measurement; about 1.5 agent-days)

Take the bit-neutral rows first; measure before/after on the pod with the
auditors' profiling scripts (`repro/pod/exp6_perf.py`, `ls_profile*.py`,
`tls_profile*.py`, `af_*.py`); keep a parity test for each. All gains below
are auditor-measured on a shared 4090 and must be re-measured.

Bit-neutral (do first): BLS-1 route `compile_bls`/`compile_sparse_bls` through
the kernel LRU cache (0.4 s and 1.2-1.6 s per call today); BLS-2 cap
`freq_batch_size` at `len(freqs)` and bound the memory budget (fixes the
20 GB transient); BLS-5 use the fused kernel in `eebls_gpu_fast_adaptive` and
`eebls_transit(use_optimized=True)` (2-2.5x); BLS-6 pin only t/yw/w, small
`BLSMemory` LRU (6x at ZTF scale); BLS-8 einsum instead of `np.dot` under the
OpenBLAS throttle; BLS-9 vectorize solution re-phasing, `chi2_0` from yy;
LS-1 numpy reductions instead of Python builtins in `weights()`/`setdata`
(3x per LC at N = 65K); LS-2 FAP at the best index only (35-111 ms per LC);
LS-4 cache `LombScargleMemory`/cuFFT plans across `batched_run_const_nfreq`
calls (up to 20x for one LC per call); LS-5 stacked multiharmonic solve
(96x); TLS-1 `q_transit` instead of `duration_grid_keplerian` in
`tls_transit` (2x); TLS-2 drop the ThreadPoolExecutor in `tls_search_batch`
(2x); TLS-3 memoize template tables and pool device buffers; CE-1 fix the
`'ce_wt'` compile gate (recompile every run today: 3.5-100x); CE-2 size the
fast grid from the SM count (5 blocks on 128 SMs today: 65x kernel); CE-3 /
PDM-1 skip unused `allocate_bins`, pinned staging; LRT-1 one `NFFTMemory` and
cuFFT plan per `run()` (2.5-12 ms -> 0.08 ms per template; campaign 1.5 h ->
~15 min; needs the buffer zeroing fix); LRT-2 hoist the Detector A matrices
out of the template loop.

Result-changing (need the decision and a CHANGELOG line): BLS-3 default
`eebls_transit` to the fused fast kernel (47-520x; ties to defect 7); BLS-4
vectorized `transit_autofreq` (grid count differs by ~2 at the top; keep the
recursion behind a flag); LS-3 `next_fast_len` grid padding (results move
~1e-3 toward exact; do together with defect 4).

Not low-hanging (do not spend time): TLS coarse kernel re-fold per period
(large rewrite), packed-CAS BLS atomics (5x slower), sigma=2 or m<8 for LS,
`ndimage.median_filter`, PDM `_fast` on Ada.

---

## Phase 3: release hygiene (CPU; about 2 agent-days)

Follow `RELEASE_READINESS.md` sections 2-6 in this order; item numbers are that
report's blockers.

1. API freeze (D3): blocker 13 namespace (`__init__.py:66-77` fallback deleted,
   `__all__` == `_LAZY_ATTRS` keys minus NUFFT-LRT while it is experimental,
   `__dir__` fixed, CPU test); blocker 5 `T0` (done in Phase 1); shim hygiene
   per section 6 (internal imports off `core.py`, `stacklevel=2`, "removed in
   2.0" wording, `utils.weights` canonical); delete `pink_noise_correction`,
   `estimate_n_evaluations`, `_next_pow2`, `utils.tophat_window/gaussian_window/
   get_autofreqs` (changelog "Removed"); keyword-only markers on the 1.0-new
   signatures (finding 136); explicit `__all__` per user-facing module.
2. Packaging (blocker 6, section 5): delete `setup.cfg`; `setup.py` -> two-line
   shim or delete (update `scripts/benchmark_all_gpus.sh:286`); `MANIFEST.in`
   `LICENSE.txt`, drop `requirements.txt`; `pyproject.toml` `setuptools>=77`,
   `license = "GPL-3.0-only"` + `license-files`, drop the `License ::`
   classifier, floors `numpy>=1.22`/`scipy>=1.8`, classifiers 3.13/3.14,
   `test` extra = pytest, nfft, astropy, batman-package, transitleastsquares,
   `[tool.pytest.ini_options]` (testpaths, `-rs --strict-markers`, `gpu`
   marker, filterwarnings); delete `cuvarbase/kernels/wavelet.cu` + orphan-
   kernel guard test; `.gitignore` negations for `analysis/**/*.log` and the
   tracked PNGs, then `git add` the seven cited gate logs (blocker 9); fix the
   `\l` escape at `bls.py:1437`; remove `import resource` x3 and the unused
   imports/locals; mutable default `u=[...]` -> `None`.
3. Tests (blockers 7, 8; section 4): move the stub conftest into
   `cuvarbase/tests/conftest.py`, guard the repo-relative tests, make
   `pytest --pyargs cuvarbase` pass from an installed wheel; `-rs` everywhere;
   `test_nfft.py` importorskip scoped to the two tests that need it; fix the
   two `except Exception: pass` tests in `test_bls.py:998-1004, 1038-1045`;
   seed the RNG in `test_nufft_lrt.py`; port `scripts/test_kernel_cache.py`,
   `tls_fast_smoke.py`, `test_adaptive_correctness.py` into the suite; add
   tests for `tls_transit`, `tls_search` dispatch, `duration_grid_keplerian`;
   a cufinufft on-device test; generalized kernel-drift guard; a compile-check
   of examples/notebook cells; CI: `twine check`, sdist install leg, wheel
   `--pyargs` run, docs job, 3.13/3.14, `permissions:`.
4. Docs consistency (blocker 10, section 6 table): delete `docs/TLS_GPU_README.md`
   and `docs/TLS_GPU_IMPLEMENTATION_PLAN.md`; fix `docs/source/bls.rst:233-235`
   (phi0 convention) and the two stale `test_bls.py` comments; delete the
   `Dockerfile` and its four mentions (or rebuild and smoke it on the pod);
   `INSTALL.rst` CUDA 12.4 only, astropy is test-only, TLS not experimental,
   the `--no-deps` install path; `CONTRIBUTING.md` (3.9+, `.base`, README.md,
   `master`, archive pointer); release-notes rows `:38, :43, :127`;
   `BENCHMARK_RESULTS.md:138-145` batch table rewritten from the July data;
   `docs/source/cuvarbase.rst` gains `nufft_lrt` (labelled experimental) and
   `base.context`, drops `core`; `index.rst` landing blurb; `conf.py` copyright
   `2017-2026`; delete `docs/source/plots/benchmarks.py`; fix the unused import
   in `bls_transit_diagram.py`; CHANGELOG/release-notes bullets for every
   Phase 1/2 result change, the NUFFT-LRT `detector=` API, and tag-pinned URLs
   instead of `analysis/` paths; docstring default mismatches (finding 134).
5. Repo prune (D2; section 3 table): create `archive/pre-1.0-process` locally on
   the pre-prune commit, then one prune commit: delete `test_python_versions.sh`,
   `publish_docs.sh`, `requirements*.txt`, `docs/BENCHMARKING.md`,
   `docs/FBLS_GPU_SPEC.md`, `docs/BLS_OPTIMIZATION.md`, `examples/benchmark_
   results/`, `analysis/TESS_*`, `analysis/V1_*`, `analysis/BENCHMARK_PROTOCOL_
   V1.md`, the six `analysis/*-jun2026.json`, the superseded gate/batch
   folders, `cuvarbase/base/README.md`, `cuvarbase/memory/README.md`, the 15
   one-off scripts, the empty CE notebook; move `GTLS_COMPARISON.md` (+png) and
   `TLS_COST_ANALYSIS.md` to `docs/`, `PROFILE_RANKING.md` to its campaign,
   `benchmark_results_by_gpu/*.json` into `benchmarks/results/`; rewrite
   `docs/NUFFT_LRT_README.md` into `docs/source/nufft_lrt.rst`; merge
   `RUNPOD_DEVELOPMENT.md` into `scripts/README.md`; parity `.npz`: keep the 8
   endpoint files or delete all 24; add `analysis/README.md`. This directory
   (`analysis/audit-sep2026/`) stays as the audit of record for 1.0; prune
   `repro/` to the scripts that became tests once Phase 1 lands.
6. Runbook rewrite (blockers 1, 2, 3, 6, 12): `git fetch` + reset local master;
   four conflicts, take `v1.0-fixes`, assert `git diff --quiet v1.0-fixes
   master`; README flip BEFORE the freeze; wheel glob; gate provenance and
   log archiving; `gh-pages-staging` rebuilt at T; post-release branch sweep.
7. Last content commit: README flip (blocker 3: drop the banner, `pip install
   cuvarbase`, absolute links, invert `test_readme_consistency`, PKG-INFO grep),
   strip the release-notes DRAFT comment, write the expected test count.
8. Local verification: CPU suite green; flake8 hard select green; `python -m
   build && twine check`; wheel and sdist into fresh venvs with
   `ci_wheel_smoke.py` and `pytest --pyargs cuvarbase`; CPU Sphinx build with
   only the plot-directive warnings; push; GitHub Actions green.

---

## Phase 4: NUFFT-LRT re-validation (pod, 2-3 pod-hours after LRT-1 lands)

With Phase 1 fixes in: extend `scripts/nufft_lrt_validation.py` with (a) a
configuration with `t + 2457000.5`, (b) an arm using the public default
`epochs=None` (now an epoch grid), (c) a non-zero-mean basis, and (d) at least
200 injections per depth so 0.05-level differences resolve. Re-run all four
existing configurations and all arms. Archive the JSON under
`benchmarks/results/nufft_lrt_validation_<date>/` and fill
`docs/source/nufft_lrt.rst` from `scripts/summarize_lrt_validation.py`.
Then D1: official only if the fixed default path passes and the docs quote
the measured numbers honestly (baseline text in `ALGORITHM_AUDIT.md` 6.4).
The Sep-2026 campaign JSON is in `campaign/` for comparison.

---

## Phase 5: freeze and gate (pod, about 3 pod-hours; CPU rehearsal 2 hours)

Freeze T = `origin/v1.0-fixes` tip. On a fresh pod, clone T by SHA; install
`.[test]` + cufinufft; run the full suite (`-v -rs`, expect all passed,
0 skipped, collected count recorded), `scripts/check_release_gate.py` (14/14),
`SPHINXOPTS='-E -a -W --keep-going' make -C docs html` with all figures,
`python -m build && twine check`, wheel and sdist smoke from outside the tree
including `pytest --pyargs cuvarbase`, one `eebls_transit`/`tls_search_batch`/
`lomb_scargle_simple` run from the installed wheel; rebuild `gh-pages-staging`
from T (strip `.doctrees`/`.buildinfo`, keep `.nojekyll`, SHA in the commit
message). Copy logs back; terminate the pod. Commit T' = gate record only
(`analysis/v1.0-release-gate-<date>/` with SHA, tree hash, logs) and assert
`git diff --quiet T T' -- . ':!analysis'`. Rehearse the merge on a throwaway
branch (expect the four conflicts; assert tree identity; build; check
PKG-INFO carries the flipped README); delete the rehearsal branch.

---

## Phase 6: release day (waits for the coordinated go; about 2 hours)

Fetch; local `master` = `origin/master`; `git merge --no-ff v1.0-fixes` with
the four `--theirs` resolutions; assert `git diff --quiet v1.0-fixes master`;
delete and recreate `v1.0.0` on the merge commit; push `master`, the tag, and
`archive/pre-1.0-process`; build from the tag in a clean venv, `twine check`,
`twine upload`; `pip install cuvarbase==1.0.0` smoke; `gh release create`
from the release notes; force-push `gh-pages-staging` to `gh-pages`;
fast-forward `origin/v1.0`; contributor messages; issue sweep.

## Phase 7: after release

Delete `origin/v1.0-fixes` after a grace period, the four `worktree-*` and the
~24 merged local branches, the obsolete remote branches (tell the contributor
before deleting `fix/BLS-kernel`); remove `analysis/release-staging-v1.0.0/`;
queue 1.0.1/1.1: TLS coarse-kernel rewrite, Detector A promotion after
re-validation, Dockerfile rebuild, `_cufft.py` hardening, stellar-parameter
overrides, CE float32 grids, float64 grid builders, thread-safety.

---

## Known gaps in the audits (what a completeness pass would look at)

- 66 release-hygiene findings and all 25 performance + 101 medium/low
  algorithm findings are single-auditor evidence; treat each as a hypothesis
  when acting on it.
- The release audit's "what did we miss" critic never ran (usage limits). Not
  examined: secrets in git history (`.runpod.env` is ignored; check
  `git log -p --all -S RUNPOD_API_KEY`), `CITATION.cff`, `SECURITY.md`, issue
  templates, README badges (PyPI badge shows 0.2.5, no CI badge), GitHub repo
  description/topics, the 0.2.6/0.4.0 lineage story in the CHANGELOG, and
  whether closing all ten open issues at release is still right after the
  algorithm audit (several findings map onto issues #14, #17, #32).
- Performance numbers are from a shared RTX 4090; the archived benchmarks are
  RTX A5000. Re-measure on an A5000 before quoting any new speed claim.

## Rough effort

| Phase | Effort | Where |
|---|---|---|
| 1 correctness fixes + tests | ~3 agent-days | CPU, pod for device tests |
| 2 performance quick wins | ~1.5 agent-days | pod |
| 3 release hygiene | ~2 agent-days | CPU |
| 4 NUFFT-LRT re-validation | 2-3 pod-hours + 0.5 day docs | pod |
| 5 freeze, gate, rehearsal | 3 pod-hours + 2 hours | pod, CPU |
| 6 release day | 2 hours | needs the go |
