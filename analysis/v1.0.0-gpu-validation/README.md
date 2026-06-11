# v1.0.0 GPU validation record

GPU validation gate for the v1.0.0 release (see
`analysis/V1_AUDIT_AND_GAMEPLAN.md` §7), run 2026-06-11 on a RunPod
on-demand instance.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX A5000 (24 GB), driver 570.211.01 |
| CUDA | 12.4 (nvcc V12.4.131) |
| Image | runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 |
| Python | 3.11.10 |
| Key packages | numpy 2.4.6, pycuda 2026.1, scikit-cuda 0.5.3 (numpy-2.x patched), astropy 7.2.0, scipy 1.17.1, nfft 0.1 |
| cuvarbase | 1.0.0 (commit recorded in the release tag) |

## Contents

- `gpu_test_results.xml` — junit output of the full pytest suite on GPU:
  **568 passed, 5 skipped, 0 failed**. The 5 skips are batman-package
  tests (optional TLS dependency, not installed). All GPU kernel tests
  ran for real (no pycuda stubbing), including the 9 PDM tests covering
  the four fast shared-memory kernels.
- `pytest_output.txt` — verbose pytest log for the same run.
- `release_gate_output.txt` — `scripts/check_release_gate.py`, all 14
  checks PASS: reduction_max equivalence (standard vs optimized BLS
  kernel, corr=1.000000), kernel-cache timing (second call ~5 ms vs
  ~1.3 s first), lomb_scargle_simple weights fix,
  batched_run_const_nfreq via the PR #59 cufftEstimate1d
  memory-estimation path, CE sinusoid recovery + numpy-reference
  agreement + compute_log_prob + use_fast/weighted guard, PDM new
  (t,y,err) API vs deprecated path (corr=1.000000), PDM fast vs
  reference kernels (corr=1.000000), PDM frequency recovery and
  block_size validation.
- `benchmark_new_features_tests.txt` —
  `scripts/benchmark_new_features.py --tests-only`, all correctness
  tests PASS (batch BLS vs single-LC equivalence, cuFINUFFT vs custom
  NFFT cross-check, Keplerian frequency grids 4.4–37.2x reduction with
  injected-transit recovery).

## Notes

Two defects were found and fixed during validation (commit 6e9c127):

1. `find_kernel` crashed for PEP-660 editable installs on Python < 3.12
   when cuvarbase was imported from outside the source tree
   (`MultiplexedPath` misresolution). Kernel paths now resolve relative
   to the package `__file__`.
2. The original CE gate check ("recovers transit period") was
   mis-conditioned, not a kernel bug: a plain-numpy CE reference puts
   the global minimum of the q=0.05 box transit at the 2f harmonic
   (raw times) or 0.9363 (mean-subtracted times, as the PR #57/#61
   normalization applies) — in both cases matching the GPU kernel
   exactly. The check now gates on strong-sinusoid recovery plus
   GPU-vs-numpy-reference agreement on identically normalized data.
