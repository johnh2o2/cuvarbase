# v1.0 release-candidate GPU validation (June 12, 2026)

Validation of the packed-release punchlist work (branch `v1.0-fixes`,
code state = commit 53ee37b: commit 6a01439 + the floor-epoch fix and
test calibrations landed as 53ee37b after this run exposed them).

## Environment

- RunPod RTX A5000 (24 GB), driver 580.126.09, CUDA 12.4 (nvcc)
- Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- python 3.11.10, numpy 2.4.6, scipy 1.17.1, astropy 7.2.0,
  pycuda 2026.1, skcuda 0.5.3 (+ compat shim), cufinufft 2.5.1,
  batman-package, transitleastsquares — pod `qikjnntc3qzoqu`
  (terminated and verified deleted via the RunPod API afterwards)

## Results

1. **Full pytest suite: 608 passed, 0 failed, 0 skipped**
   (`pytest_full_suite.log`). This includes, for the first time on
   hardware: the 5 batman TLS accuracy tests (previously skipped in
   the v1.0.0 gate) and the 4 new golden tests vs the reference
   `transitleastsquares` package.
2. **check_release_gate.py: 14/14 PASS** (reduction_max equivalence,
   BLS/LS/CE/PDM recovery, kernel caching, guard checks).
3. **benchmark_new_features.py --tests-only: ALL PASS** (batch BLS
   consistency, cuFINUFFT correctness vs custom NFFT, Keplerian-grid
   transit detection).
4. **GPU queue items** (see analysis/V1_RELEASE_PUNCHLIST.md):
   - BJD epoch tests (TestEpochHandling, 5 tests): pass. The first
     run exposed a systematic phase-0.0 bin-edge artifact of the
     epoch = min(t) convention; fixed by switching to
     epoch = floor(min(t)) (commit 53ee37b).
   - TLS phase-1 hardening + duration-scaled t0 grid: kernels
     compile and run; the narrow-transit audit scenario (P=15 d,
     q=0.012 — invisible to the old 30-epoch grid) is recovered with
     period error < 1%, correct depth, SDE 5.75.
   - TLS golden vs transitleastsquares: both configs agree with the
     reference on period (<1%) and depth; both packages mark the
     detections significant.
   - Batch Keplerian per-frequency q bounds: recovery test passes.
   - cuFINUFFT plan caching: 10-call warm timings —
     ndata=1e3/nf=12.5e3: 1.95x FASTER than custom NFFT (was
     0.63-0.84x); ndata=1e4/nf=125e3: 0.96x; ndata=3e3/nf=75e3:
     0.88x. Caching flips small/medium problems past 1x; large
     problems are spreading-dominated. Module docstring stance
     ("cross-check backend; custom kernel default") remains accurate.
   - Adaptive BLS re-benchmark: committed as
     `benchmarks/results/bls_adaptive_keplerian_benchmark_rtxa5000_jun2026.json`.
     Measures ~1.0-1.3x over fixed blocks (0.48-0.92x for ndata<=64)
     with warm kernel cache — the published 1.4-5.3x / 90x claims did
     NOT reproduce and have been corrected in README /
     BLS_OPTIMIZATION / CHANGELOG (the old gains were dominated by
     per-call kernel handling that the kernel cache now amortizes).
   - nsys profile of eebls_gpu_batch: NOT RUN — nsys is not on the
     pod image. The regression remains documented + runtime-guarded
     (docstring warning, UserWarning above ndata=10,000); diagnosis
     stays a v1.1 item.
   - LS vs astropy 8.0: astropy 8.0 is NOT released on PyPI as of
     2026-06-12 (latest: 7.2.0). LS test suite passes against 7.2.0
     (21/21). Published tables now carry an explicit astropy-version
     pin; re-comparison against 8.0's LRA default is deferred until
     it ships.
