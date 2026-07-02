# GPU batch 3 — RTX A5000, Jul 2 2026

Pod: RunPod `vma81x9ssaaf92` (runpod/pytorch:2.4.0-py3.11-cuda12.4.1,
$0.27/hr, terminated + verified after the run). Branch under test:
`v1.0-fixes` @ ed96347 (all Jul-2 audit fixes) unless noted. Extra deps
installed for this batch: batman-package, transitleastsquares, cufinufft
(so the 7 batch-2 skips now execute).

## Full suite + release gate (v1.0-fixes)

- **Full GPU suite: 721 passed, 0 failed, 0 skipped** in 8:40.
  (Batch 2 was 671 passed / 7 skipped; the delta is the new audit
  regression tests + the previously-skipped batman/TLS/cufinufft tests.)
- **check_release_gate.py: ALL 14 CHECKS PASSED** (reduction_max
  equivalence corr=1.0, BLS/LS/CE/PDM recovery, kernel cache, batched
  cufftEstimate1d path, CE validation errors, PDM API parity).

## AUDIT: stream-parity regression tests (B3 race fixes)

All three pass on device (run verbosely, in-suite as well):

- `test_bls.py::TestPinnedBufferStreamParity::test_fast_path_stream_matches_default[False]` PASS
- `test_bls.py::TestPinnedBufferStreamParity::test_fast_path_stream_matches_default[True]` PASS
- `test_tls_basic.py::TestTLSStreamParity::test_stream_matches_default` PASS

These exercise user-supplied streams against the default-stream path and
would race (stale/zero host buffers) without the Jul-2 sync fixes
(07ce10e).

## C2: multiharmonic GLS ghat_g layout (smoke_c2_mhgls.py)

GPU `LombScargleAsyncProcess(nharmonics=H, use_fft=True)` vs pure-Python
`lomb_scargle_direct_sums(nharms=H)` on the same grid (nf=675),
harmonic-rich signal (P1 + 0.6·P2 + 0.4·P3), ndata=150:

| H | precision | corr | max abs diff | peak match |
|---|-----------|------|--------------|-----------|
| 2 | float64 | 0.999951 | 4.5e-3 | exact (3.0074 = 3.0074) |
| 2 | float32 | 0.999951 | 4.5e-3 | exact |
| 3 | float64 | 0.999943 | 4.7e-3 | exact |
| 3 | float32 | 0.999929 | 1.1e-2 | exact |

**PASS** (criterion corr>0.999 on the float64 path). The real ghat_g
spectrum layout read back by `_mh_power_from_spectra` matches the
audited convention on device. Raw JSON in `smoke_c2_result.json`.

## C3: NUFFT-LRT on-device checks (smoke_c3_nufft_lrt.py)

Two-season light curve (120+120 pts, 260-day gap), injected 2.3 d box
transit, `NUFFTLRTAsyncProcess(use_double=True)`:

- `compute_nufft` vs exact adjoint DFT: **corr=1.0000000000,
  max rel err 1.3e-7** over the sigma=2 guaranteed band (k < nf/2). PASS.
- Device execution confirmed: lazy CUDA context active after first call,
  all 5 NFFT kernels compiled+prepared (no stubs, no mocks). PASS.
- Multi-season end-to-end detection: **best period 2.3000 (exact truth),
  SNR 20.4 vs median −0.05**. PASS.
- All 19 CPU/GPU `test_nufft_lrt*` tests pass on device.

**Finding (fixed in this batch): the documented phase convention was
wrong.** The device transform is `ghat[k] = Σ y_j exp(2πi k t_j/T)`
with ABSOLUTE t (the `normalize` kernel re-references to t=0), not
`t_j − tmin` as the `compute_nufft` docstring and the pipeline-test mock
claimed. Against the tmin-relative reference the full-band corr is 0.55
(per-k phase error `2πk·tmin/T`, up to ~0.9 rad on this data) — this is
what the first smoke run caught. Docstring + mock corrected; the
matched filter is unaffected (data and template share the transform, so
the common phase cancels — detection was exact all along).

**Also documented:** modes k ≥ nf/2 sit outside the σ=2 Gaussian
window's guaranteed-accuracy band (deconvolution amplification
exp(b·khat²) → the full-band max rel err 0.37 is band-edge error, not a
defect). The LRT consumes the full band but whitens by the empirical
PSD, which absorbs this; noted in `compute_nufft`.

## AUDIT: PDM benchmark recovery re-run (argmin→argmax fix)

`scripts/benchmark_pdm.py --tests-only` after the 9ef3306 argmax fix:

- ndata=300, P=2.5 d: corr=1.000000, argmax match, f_best=0.39990 vs
  f_inj=0.40000 → **recovers**
- ndata=1000, P=5 d: corr=1.000000, argmax match, 0.19995/0.20000 → **recovers**
- ndata=3000, P=10 d: corr=1.000000, argmax match, 0.09997/0.10000 → **recovers**

ALL PASS. The batch-2 "PDM sparse-bin high-frequency artifact" recovery
failure was entirely the benchmark's argmin-on-maximize-convention bug
(C1 audit finding); PDM itself recovers the injected signal at every
config. `benchmark_results_by_gpu/pdm_a5000.json` updated with the
recovery table (batch-2 throughput grid retained unmodified).

## A3: NFFT error-floor diagnosis

**Cause found and fixed: float32 `PI` literal in cunfft.cu's phase
kernels.** The float64 error now tracks the L1 truncation bound
(m=12: 3.4e-3 → 1.2e-10). See `A3_DIAGNOSIS.md`; raw sweeps in
`a3_sweep_before_fix.txt` / `a3_sweep_after_fix.txt` (JSON_RESULT lines
include per-mode error profiles for the phase-on/phase-off cases).

## PR #65 (astrobatty, bugfix/BLS-kernel @ c959d51)

Run in a separate clone (`/workspace/cuvarbase-pr65`, editable install
repointed) so the branch was tested exactly as submitted:

- **Full GPU suite: 916 passed, 0 failed** in 9:29 (his
  use_fast × use_optimized parametrization adds ~190 cases over our
  723).
- **check_release_gate.py: ALL CHECKS PASSED** on his branch.
- **noverlap × use_optimized interaction** (flagged in the Jul-1 review
  notes as needing a pod check): `eebls_transit(use_optimized=True,
  noverlap=3)` really does multi-pass (max |p3−p1| = 0.059 > 0),
  satisfies the elementwise-max property (p3 ≥ p1 everywhere), and
  recovers the injected signal exactly. CLEARED.
- **Shallow-transit regression from the `fabs(ybar) > 1e-5f` guard
  CONFIRMED** (`pr65_shallow_transit_repro.py`): 500 ppm, q=0.01
  transit in normalized flux (per-point σ=1e-4, in-transit SNR ~27,
  ndata=3000, 20001 freqs). The kernel-internal s ≈ q·depth = 5e-6 is
  below the guard for *every* box, so on his branch the entire
  periodogram is identically zero (peak power = 0, NOT RECOVERED);
  on v1.0-fixes the same data yields peak f=0.40002 vs injected
  0.40000, power 0.137, RECOVERED. This is review ask #1.
- `git merge-tree` check: his branch merges into the current
  v1.0-fixes head with no conflicts.

Review comment posted on the PR (Jul 2) with these results and four
asks (guard justification/relative threshold, single_bls/hone_solution
phi convention + JD round-trip test, use_optimized docstring claim,
return-arity/block_size-override consistency).

## Batch-4 pod (same day, tw642fncf2qsvu): E1 + E2 + A5

A second short pod session closed the remaining diagnosis items:

- **E1** (eebls_gpu_batch): correctness divergence = missing A2
  noverlap multi-pass; perf "regression" = per-call kernel compilation.
  Both fixed; batch now beats the single-LC loop at every scale.
- **E2** (LS batch_size>1): per-call allocation of batch_size memory
  sets with no compute headroom; documented, default unchanged.
- **A5** (chi2_0 on memory reuse): BLSMemory records chi2_0 at setdata
  time; snr/loglik conversion uses it.

Details + tables in `E1_E2_DIAGNOSIS.md`; scripts `e1_batch_profile.py`,
`e2_ls_profile.py`, tests `TestBatchFastParity`,
`test_snr_uses_loaded_data_on_memory_reuse`,
`test_batch_kernels_are_cached`. Full suite re-run green on the batch-4
pod (see tracker).

## Batch-5 pod (Jul 2, 0edjptn1mlfkgb): PR #65 instability root-caused

Attila's response to the review supplied a HATPI reproducer for the
instability his `fabs(ybar) > 1e-5f` guard was added for
(nondeterministic run-to-run periodogram deviations, transient bogus
peaks). Root cause found in OUR kernel: `bls_value`'s upper bound
`w < 1.f - 1e-10f` is a float32 no-op (1e-10 < ulp(1)/2 → compiles to
`w < 1.f`), so an all-weight trial box divides atomic roundoff by
atomic roundoff; `sparse_bls.cu`'s `MAX_W_COMPLEMENT=1e-9` had the same
underflow, and the CPU `single_bls` returned literal NaN on the same
box (deterministically verified pre-fix on the pod).

Fixed with a float32-meaningful 1e-4 complement in bls_common.cuh /
bls_batch.cu / sparse_bls.cu / single_bls / sparse_bls_cpu. New tests
(`TestAllWeightBoxStability`): deterministic CPU zero, GPU repeat
stability on single-site data, 500 ppm shallow-transit recovery.

Caveat, recorded honestly: the GPU run-to-run instability itself could
NOT be reproduced synthetically (tried HATPI-scale n=98K + his exact
eebls_transit call + extreme weights + outliers + phase clustering —
all stable pre-fix at ≤1e-6, scripts in `pr65_instability_repro.py`).
The reply asks attila to re-run his 50-iteration HATPI check on
v1.0-fixes and to share one light curve's columns if any deviation
survives. Side observation: a few near-zero-error points make one box
carry ~all weight deterministically (power ≈ 0.99 in pure noise) —
data-hygiene footgun, possibly worth a docs note (not a code bug).
