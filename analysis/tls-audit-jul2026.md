# TLS Fast-Path Audit (PR #68) — July 2026

**Scope**: pre-release audit of the Opus-4.8-authored TLS survey-scale rewrite
(PR #68, `feature/tls-fast-survey` → `v1.0-fixes`, merged 72f3663). Library
surface reviewed line-by-line: `cuvarbase/kernels/tls_fast.cu` (576 lines, both
kernels), `cuvarbase/tls.py` (fast infra + `tls_search_batch` + `tls_search_gpu`
dispatch), `tls_grids.py`, `tls_models.py`, `tls_stats.py`,
`tests/test_tls_fast.py`, `tests/test_tls_basic.py`. Performance-claims trace
reported separately (see the claims-trace section of this audit when appended).

**Method**: adversarial read targeting the trap-prone invariants: period
banding + `period_map` scatter, shared-memory layout arithmetic, float-float
fold under `--use_fast_math`, block/warp reductions, the
SDE-from-uniform-coarse-spectrum invariant, the refinement fallback path
(site of the one defect already caught in PR #68 review, 51c2cb3), chunking /
int32 offset rebasing, and host↔kernel constant agreement.

## Verdict

**No correctness defects found.** The items below are cleanup, documentation,
and test-coverage findings. This is consistent with the branch's independent
validation record (golden tests vs the reference `transitleastsquares` package
run through the fast path by default; SDE parity measured on A5000 / RTX 4000
Ada / V100 with 100% injected-transit recovery).

Invariants explicitly verified:

- **Shared-memory arithmetic** host↔kernel: `_tls_fast_shared_size` =
  `2*NBINS + 2*(NTEMPLATE+1) + 4*BLOCK_SIZE + MAX_DURATIONS` floats +
  `MAX_DURATIONS+1` ints — matches the kernel's layout exactly, including the
  int-aliased `dur_cum` tail. Refine layout `(NTEMPLATE+1) + 4*(bs/32)` matches.
  >48KB opt-in (`MAX_DYNAMIC_SHARED_SIZE_BYTES`) applied to the search kernel;
  the refine kernel's footprint (~4.2 KB) never needs it.
- **Banding**: `np.unique` bands partition the period grid exactly; every
  `(lc, period)` output slot is written exactly once per chunk (kernel writes
  score or the −1 sentinel unconditionally at thread 0); `period_map` scatter
  indices are the band's own global indices. Bin-count capping loop respects
  the device shared cap; explicit `nbins`/`block_size` overrides fail loudly in
  `compile_tls_fast` rather than silently shrinking.
- **Float-float fold**: hi/lo split of both `t` and `1/P`; residual via exact
  FMA. `--use_fast_math` does not break it (nvcc keeps explicit `fmaf`
  intrinsics; no cross-statement algebraic re-simplification), and the fold
  error is empirically pinned at ~3e-8 phase on three GPU families. Bin index
  masked (`& (NBINS-1)`) so a phase that rounds to 1.0 wraps to bin 0 —
  correct periodic behavior. Negative `kk` in the window walk wraps correctly
  through the two's-complement mask.
- **Validation**: `n_durations ∈ [2, 64]` enforced (kernel `MAX_DURATIONS`);
  `0 < qmin ≤ qmax < 1` enforced (prevents `logf(0)` and window wrap /
  double-count); power-of-two `block_size ≥ 32` and power-of-two `nbins`
  enforced at compile; `refine_nd ≥ 2` enforced (kernel divides by
  `REFINE_ND-1`); `refine_oversample > 0` enforced when refinement is on.
- **Reductions**: block max-reduction requires power-of-two blockDim
  (enforced); final 32 lanes reduce via `__shfl_down_sync(0xffffffff, ...)`
  with the whole first warp active — no partial-warp mask hazard. Refine
  kernel's per-trial warp reduction converges (shuffle is outside the
  divergent point loop); warps with no trials publish the −1 sentinel and are
  ignored.
- **SDE invariant**: refined values go only to compact per-candidate arrays;
  `_finish_lc` computes SDE/FAP from the coarse `score_h` spectrum
  exclusively. Regression-tested (`test_spectrum_is_coarse_and_uniform`).
- **Refinement fallback** (the 51c2cb3 fix): coarse t0/dur/depth arrays are
  fetched iff `return_arrays or K == 0 or any LC's refinements all failed`,
  which exactly covers every path into the coarse-fallback branch; a refined
  slot with positive score implies its coarse period was valid, so the
  `searchsorted` valid-index mapping is exact. Regression-tested with a
  monkeypatched always-fail refine kernel.
- **Chunking**: grid.y ≤ 65535 and per-chunk output caps enforced; chunk
  offsets rebased to the chunk base before the int32 cast (a single chunk is
  bounded by the 16M-point budget except when one lightcurve alone exceeds
  it, which works — buffers size to the actual max chunk).
- **Statistics**: SDE median-filter kernel capped at 91 (reference TLS
  convention) with the reference's skip-detrend-when-short behavior;
  `window_length` kept as a deprecated alias. Template integral tables
  (trapezoid on an 8× fine grid) align exactly with the
  `linspace(-1, 1, n)` template convention; `S1/S2` endpoint saturation
  implements the zero-outside-[-1,1] template correctly.

## Findings

| # | Severity | Where | Finding | Disposition |
|---|----------|-------|---------|-------------|
| 1 | Low (cleanup) | `tls.py` `_auto_nbins` | Dead code: never called — `tls_search_batch` inlines the per-period banding logic instead (an evolution of this single-nbins helper). | **Fix now**: delete. |
| 2 | Low (docs) | `tls_search_gpu` docstring | Says flux "will be normalized" — neither path normalizes; both assume baseline ≈ 1.0 (`a = (1−y)/σ²`). Wrong user expectation → silently wrong results for raw-counts flux. Pre-existing text, but worth fixing before release. | **Fix now**: docstring corrected; narrative docs (tls.rst) to state the convention. |
| 3 | Low (API hygiene) | `tls_search_gpu(durations=...)` | Parameter accepted and silently ignored (pre-existing — dead in the legacy path too, and undocumented). | **Fix now**: warn when passed non-None. Removal would break positional callers; deferred to the v1.1 naming/API pass. |
| 4 | Doc (pre-existing) | legacy path (`use_fast=False`) | Legacy path casts `t` to float32 with no epoch subtraction — BJD-scale times silently lose phase precision on the legacy path only (same class as the BLS bug fixed in PR #65). Fast path (default) epoch-subtracts in float64 and is regression-tested at BJD scale. | Docstring + tls.rst note ("use the default fast path for absolute BJD times"). Legacy kernel rework deferred (v1.1 list). |
| 5 | Low (edge case) | auto Keplerian bounds | For ultra-short periods near the stellar-surface limit (P ≲ 3.5 h, q → 0.5), `qmax_fac=2` can push `qmax ≥ 1` → hard ValueError (clear message). Legacy path tolerated such grids. Realistic USP searches (P ≥ 4 h) are unaffected. | Documented here; error message already explains the fix. No code change. |
| 6 | **Moderate (test gap)** | multi-band parity | No test forces the period grid into multiple NBINS bands and checks banded vs single-band parity — banding + `period_map` scatter is the most intricate host logic and was only exercised implicitly. | **Fix now**: GPU test added (`TestBanding::test_banded_matches_single_band`); runs in the Phase-3 gate. |
| 7 | Low (robustness) | `_preprocess_batch` | A single lightcurve with > 2³¹ points would wrap in the int32 `lens` cast (needs ≳ 34 GB of device buffers first, so effectively unreachable). | **Fix now**: explicit guard with a clear error. |
| 8 | Note | NaN/Inf inputs | Non-finite `y`/`dy` propagate NaN into the affected phase bins; windows touching them fail the validity gate and the corresponding trials are skipped silently (matches legacy behavior). | Recorded; input validation would cost a pass over the data — v1.1 candidate. |
| 9 | Note | `--use_fast_math` | The float-float fold and `logf/expf` duration grid run under fast-math intrinsics; correctness is empirically pinned by the 3-GPU validation + golden tests. Any future `tls_fast.cu` change must re-run those (pycuda disk cache invalidates all variants on any source change). | Recorded as a maintenance caution in the kernel header comment (already present). |
| 10 | **Medium (release blocker, docs)** | `tls.py` import-time warning | The module still emitted "cuvarbase.tls is EXPERIMENTAL and not recommended for science use in this release ... For validated transit searches use cuvarbase.bls" on every import — contradicting the v1.0 release story (TLS fast path is golden-tested against the reference package and ships as a headline feature), and it pointed users at an internal `analysis/` doc. | **Fix now**: warning removed. The legacy-path ndata cap stays documented in the docstrings and raises its own clear ValueError. |
| 11 | Low (lint) | `bls.py` docstrings | Five `:math:` docstring lines used single-backslash `\chi`/`\omega` in non-raw strings (invalid escape sequences → DeprecationWarning today, SyntaxWarning on newer Pythons). | **Fix now**: doubled, matching the file's existing house style. |

## Test-coverage assessment

`test_tls_fast.py` covers: batch↔single consistency, injected recovery, noise
ordering, coarse/refined statistics separation, refined-chi2 ≤ coarse-min,
ndata beyond the legacy cap, BJD-scale times, chunk-boundary integrity, mixed
lengths/offsets, empty/mismatched/invalid input validation, and the
refinement-fallback regression. `test_tls_basic.py` keeps the legacy
shared-memory guard (now `use_fast=False`) and adds a no-cap fast-path check;
`test_tls_golden.py` runs the reference-package comparison through the fast
path (it is the default). Gap #6 (multi-band parity) addressed in this audit;
CPU-side tests for the `tls_stats` kernel-size cap added alongside.
