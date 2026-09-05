# E1 + E2: batch-path diagnoses — both root-caused (batch-4 pod, Jul 2 2026)

> The probe scripts and raw JSON/txt named below were pruned before 1.0; see `analysis/README.md` (tag `archive/pre-1.0-process`).

Pod: RTX A5000 `tw642fncf2qsvu` (terminated + verified). Scripts:
`e1_batch_profile.py`, `e2_ls_profile.py`.

## E1. eebls_gpu_batch: correctness divergence + "12x slower at TESS scale"

Two independent defects, both fixed:

**E1a (correctness, small ndata).** The batch kernel (`bls_batch.cu`)
accepts `noverlap` but never uses it — the same silent no-op the A2 audit
found in the single-LC kernels. A2 fixed the fast paths *host-side*
(elementwise max over `noverlap` dphi-shifted passes) but the batch path
never got that treatment, so `eebls_gpu_batch` was effectively
`noverlap=1` while the `eebls_gpu_fast`/adaptive reference multi-passes.
Phase-bin quantization is exactly what oversampling fixes, so the
divergence was worst at small ndata — reproducing the Jun benchmark's
corr=0.77 / peak-match 5/10 at ndata=200.
**Fix:** `eebls_gpu_batch` now wraps its kernel launch in the same
dphi-shifted multi-pass + on-GPU elementwise max as
`_eebls_gpu_fast_impl`. Regression tests (`TestBatchFastParity`): batch
vs `eebls_gpu_fast` corr>0.999 with identical argmax at ndata=200 and
2000 (previously 0.77/0.97), plus the noverlap=1 ≤ noverlap=3
elementwise-max property. All pass on the A5000.

**E1b (performance, large ndata).** Stage profiling shows the actual
kernel work is 2–10 ms per call across all regimes; **per-call kernel
compilation was 0.58–0.89 s** — `compile_bls_batch` ran `SourceModule`
on every `eebls_gpu_batch` call, unlike the single-LC paths which use
the `_kernel_cache` LRU. The "~12x slower at ndata=20,000" measurement
was per-call compilation, the same artifact class as the pre-v1.0
adaptive-BLS speedup claims.
**Fix:** `_get_cached_batch_kernels` routes the batch kernel through the
same thread-safe LRU cache (key `(block_size, 'batch')`). Warm-cache
measurements after the fix (10 LCs, correctness-parity multi-pass
included):

| config | single-LC loop | batch | batch/single |
|---|---|---|---|
| ndata=200, nfreq=5000 | 63 ms | 6 ms | **0.10x** |
| ndata=2000, nfreq=5000 | 44 ms | 7 ms | **0.16x** |
| ndata=20000, nfreq=1788 (TESS) | 996 ms | 195 ms | **0.20x** |
| 2 LCs, ndata=20000 | 207 ms | 94 ms | **0.45x** |

Batch now beats the single-LC loop at every measured scale — the E1
acceptance criterion ("TESS-scale batch ≥ parity") is exceeded, so the
`_warn_if_batch_inefficient` UserWarning (and its test) is retired and
the docstring rewritten with the new numbers. A cache regression test
(`test_batch_kernels_are_cached`) guards the fix.

## E2. Lomb-Scargle batched_run_const_nfreq: batch_size>1 "multi-stream overhead"

Stage timing (32 LCs, ndata=3000, nf=100,000, warm process):

| batch_size | total | alloc | setdata | launch | finish | other |
|---|---|---|---|---|---|---|
| 1 | 67 ms | 9 ms | 16 ms | 10 ms | 10 ms | 21 ms |
| 2 | 71 ms | 17 ms | 16 ms | 8 ms | 8 ms | 22 ms |
| 4 | 82 ms | 31 ms | 16 ms | 8 ms | 7 ms | 21 ms |
| 8 | 236 ms | 106 ms | 16 ms | 7 ms | 6 ms | 101 ms |

**Root cause:** the method constructs `batch_size` separate
`LombScargleMemory` sets — pinned host buffers, device arrays, and a
cuFFT plan each — on **every call**; that setup cost scales with
`batch_size` (and jumps super-linearly at 8), while the compute stages
barely improve because a single survey-scale LS already saturates the
device (launch/finish actually shrink slightly with more streams — the
overlap machinery works; there is just no idle GPU to fill).

Steady state (256 LCs/call, setup amortized): batch_size=4 is ~10%
faster per LC than 1 (1.44 vs 1.62 ms/LC); 8 is net slower (1.93).

**Action: documented, default unchanged.** With ≤10% best-case upside,
a cross-call memory-reuse redesign is not v1.0 material; the
`batch_size` docstring now carries the diagnosis and amortization
guidance instead of "cause undiagnosed". (A5 fixed in the same pass:
`BLSMemory` records `chi2_0` of its loaded data at `setdata` time and
the fast path's `convention='snr'/'loglik'` scaling uses it, so
memory-reuse calls with stale `y`/`dy` arguments convert correctly —
regression test `test_snr_uses_loaded_data_on_memory_reuse`.)
