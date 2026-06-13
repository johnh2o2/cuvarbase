# v1.0 GPU validation batch — June 13, 2026 (RTX A5000)

Pod: RunPod `ydqi9luioem03s`, NVIDIA RTX A5000 (24 GB), CUDA 12.4,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
numpy 2.4.6, pycuda 2026.1, scikit-cuda 0.5.3 (numpy-2.x patched),
batman 2.5.1, transitleastsquares, cufinufft 2.5.1. Pod terminated +
verified ($0/hr, 0 pods remaining) after the session.

## Validated punchlist GPU-queue items
- **Standing full suite**: `pytest cuvarbase/tests/` → **660 passed,
  2 failed** initially (`gpu_suite_full.log`). The 2 failures were the
  queued A1 and A3 items (found here, fixed below); after the fixes
  both pass (4 passed on re-run). B1 lazy-context exercised across every
  GPU path (import + all process/standalone paths) — green.
- **A2 noverlap**, **A5 power conventions**, **A6 kernel templating**
  (bls_common.cuh single-source — both kernels compile and run) — all
  green in the suite.
- **B1 lazy CUDA context**: `import cuvarbase` creates no context;
  context retained on first GPU use; every GPU path works with real
  pycuda. Green.
- **check_release_gate.py**: ALL CHECKS PASSED (reduction_max
  equivalence, BLS transit recovery, kernel cache, LS/CE/PDM recovery +
  references, API guards).

## Bugs found by GPU validation (fixed this session)
- **A1 — `sparse_bls_simple.cu` did not compile.** The q-bounds wiring
  (commit 665dbbd) used `qmin_f`/`qmax_f` but never added
  `qmin_arr`/`qmax_arr` to the *simple* kernel signature nor declared
  them — nvcc error (`identifier "qmin_f" is undefined`). The Python
  launch already passed `qmin_g`/`qmax_g` for both kernels, so this was
  also a latent arg-misalignment. CPU suite could not catch it
  (SourceModule is stubbed). Fixed by mirroring the full kernel; both
  `test_sparse_bls_gpu_q_bounds[True/False]` pass.
- **A3 — autoset-m tolerance over-claim.** `test_..._meets_tolerance[True-1e-06]`
  asserted total NFFT error <= 1e-6, but the realized error vs the exact
  DFT **floors at ~1e-3 absolute regardless of m** (and *grows* for very
  large m as the wide Gaussian amplifies grid noise) — measured at
  sigma=2,3,5 (see diagnostic in the commit message / below). The L1
  bound governs only the truncation term, not total accuracy. Fixed:
  docstring softened to "truncation component" + documents the floor;
  the GPU test now asserts the closed-form bound `m` and realized error
  at an achievable tol (1e-2, both precisions).

### A3 error-vs-m diagnostic (double precision, ||y||_1=66.9, nf=500)
```
sigma=2  autoset m(tol=1e-6)=10
  m= 4 err=2.1e-03   m=8 err=1.5e-03   m=10 err=2.2e-03   m=16 err=8.0e-03  m=32 err=3.4e-01
sigma=3  m=4 err=9.9e-04 ... m=10 err=9.8e-04 ... m=32 err=2.9e-03
sigma=5  m=4..32 err≈1.0e-03 (flat)
```
Error floors ~1e-3 and is independent of m; L1 bound at m=10 is ~1e-27.

## A4 — nbins-aware block-size heuristic: DECISION = DOCUMENT
`benchmark_block_size.py` full grid (ndata × qmin × block_size, both
fast kernels), A5000 (`benchmark_results_by_gpu/block_size_a5000.json`):
- median penalty **1.039** (3.9%), max **1.300** (30%).
- **13/40 cells > 10%**, ALL at qmin >= 0.02 (mostly qmin=0.1) — i.e.
  large transit-duration fractions (few bins). At qmin=0.1 the best
  block is 64 while the ndata heuristic picks 256.
- Typical transit search (q ~ 0.01–0.05) stays within ~10%; qmin=0.1 is
  atypical. Decision: **document** that the ndata-only heuristic is
  within ~4% median / ~10% typical (up to ~30% for atypical large-qmin
  + large-ndata), and `block_size` is user-overridable — rather than add
  a qmin-aware heuristic (extra complexity + its own GPU re-validation)
  for the atypical-only gain.

## NEW pre-existing finding (NOT from this session) → routed to E1
`benchmark_new_features.py --tests-only`: **A) BLS batch correctness
FAILS** for small ndata — batch (`eebls_gpu_batch`) vs single-LC
(`eebls_gpu_fast_adaptive`) diverge: ndata=200 corr=0.77 peak_match=5/10;
ndata=2000 corr=0.97 peak_match=9/10; ndata=20000 PASS (corr=0.997).
The batch kernel (`bls_batch.cu`) was untouched by this session's work
(A6/B1/A1/A3), so this is pre-existing. cuFINUFFT LS (B) PASSED
(corr 0.99999). Routed to **E1** (batch-path diagnosis), scope expanded
from "large-ndata perf regression" to also cover this small-ndata
correctness divergence.
