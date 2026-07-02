# v1.0 GPU validation batch 2 — June 13, 2026 (RTX A5000)

Pod: RunPod `2baj5kk9z5p0zq`, NVIDIA RTX A5000 (24 GB), CUDA 12.4, numpy
2.4.6, pycuda 2026.1. Validated B2 (in-house cuFFT binding), B3 (pinned
host buffers), C1 (PDM batch API). Pod terminated + verified ($0/hr, 0
pods) after the session.

## Full suite
`pytest cuvarbase/tests/` → **671 passed, 7 skipped** in 8:09
(`gpu_suite2.log`). The 7 skips are TLS-golden (batman not installed this
session) + cufinufft tests (cufinufft not installed) — neither needed for
B2/B3/C1. Clean sweep (batch 1 had 2 failures; those — A1, A3 — are
fixed). The new CPU-side tests (citation, README, host_array, PDM batch)
all pass on GPU too.

## B2 — in-house cuFFT binding (cuvarbase._cufft)
- **Suite**: all LS/NFFT tests pass; `test_nfft` FFT-vs-fftpack (now
  exercising `_cufft.ifft`) passes — the binding's IFFT is correct.
- **Release gate**: `batched_run_const_nfreq (memory_requirement/
  cufftEstimate1d)` PASS — confirms `_cufft.cufftEstimate1d` works; LS
  recovery via `_cufft` PASS.
- **Perf vs scikit-cuda** (`_b2_perf`, identical complex64 ifft):
  ```
  n        _cufft(ms) skcuda(ms) ratio
  4096      0.0053    0.0054     0.976
  65536     0.0075    0.0076     0.982
  365000    0.0409    0.0410     0.997
  1000000   0.0766    0.0754     1.015
  ```
  max |ratio-1| = **2.4%** → WITHIN ±10% (both call cufftExecC2C).
- **RESULT**: B2 complete. Dropped `scikit-cuda` from deps + removed
  `cuvarbase/_skcuda_compat.py` + the numpy shim. Issue #63 resolvable.

## B3 — pinned host buffers
- **Suite**: all memory-using tests pass with `pinned=True` default
  (no transfer regression).
- **Overlap demo** (`_b3_overlap`, H2D bandwidth pinned vs page-aligned):
  ```
  n         pinned(GB/s)  pageable(GB/s)  speedup
  10000      1.17          0.83           1.42x
  100000     12.09         4.25           2.85x
  1000000    24.05         15.76          1.53x
  4000000    25.28         16.32          1.55x
  ```
  Pinned H2D is **1.4–2.85× faster** — the async-transfer win demonstrated.

## C1 — PDM batch API
- **Batch parity** (GPU): `batched_run_const_nfreq` and `large_run` match
  per-LC `run()` at **min corr = 1.000000** (6 lightcurves, batch_size=2,
  memory-capped large_run forcing chunking).
- **GPU-vs-CPU** (`benchmark_pdm.py`): GPU PDM == CPU `pdm2_cpu` exactly
  (corr=1.0000, identical theta-argmin, all 3 configs). Throughput
  (`benchmark_results_by_gpu/pdm_a5000.json`): GPU **1006–12622×** the
  pure-Python `pdm2_cpu` reference (modest grid; CPU side is the slow part).
- Release gate `PDM recovers injected frequency` PASS (best=0.1999,
  injected=0.2000) — PDM recovery works; the benchmark's original
  recovery sanity-check tripped a known PDM sparse-bin high-frequency
  artifact (hit identically by GPU and CPU), so the benchmark's pass
  criterion was corrected to GPU-vs-CPU agreement (the real test).

  **CORRECTION (2026-07-02 audit): the "sparse-bin high-frequency
  artifact" diagnosis above is FALSE.** The benchmark selected the best
  frequency with `np.argmin` on a spectrum that PEAKS at the true
  period (the kernels and `pdm2_cpu` return `1 - var/var_tot`; the
  release gate correctly uses argmax) — i.e. it reported the
  worst-fitting frequency and unsurprisingly failed to "recover" on
  healthy data. Verified on all three benchmark configs: argmax
  recovers the injected frequency within 5·df every time. Fixed in
  `scripts/benchmark_pdm.py` (argmax + recovery restored to the pass
  criterion); the `recovers=false` fields in
  `benchmark_results_by_gpu/pdm_a5000.json` are artifacts of the
  argmin bug (the throughput numbers are unaffected). Re-run queued
  for the next pod session.

## Bug found + fixed during the session
- `scripts/benchmark_pdm.py` queried `cuda.Context.get_device()` before
  any context existed — broken by B1's lazy context. Fixed to
  `ensure_context()` first (a class of script other than the suite/gate
  that assumed an eager import-time context).
