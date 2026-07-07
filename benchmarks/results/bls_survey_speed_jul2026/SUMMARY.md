# BLS survey-speed campaign — results (Jul 4 2026)

Pod: RunPod RTX A5000, $0.27/hr, CUDA 12.4, driver 570.211.01.
Branch `feature/bls-survey-speed` off 89d5481 (v1.0).
All numbers: warm-cache medians of 5 runs of a multi-LC loop
(cold/first-call totals recorded separately in the raw JSON);
before/after in the same pod session; Keplerian frequency grids
(oversampling=2, qmin=0.5 q_kep, qmax=2 q_kep):

| survey  | ndata  | nfreq   |
|---------|--------|---------|
| ZTF     | 150    | 60,121  |
| HAT-Net | 6,000  | 300,592 |
| TESS    | 20,000 | 1,788   |
| Kepler  | 65,000 | 130,597 |

Method + bottleneck ranking: `analysis/bls_survey_speed_jul2026/PROFILE_RANKING.md`
(ncu blocked by RunPod -> nsys + event decomposition + one-axis sweeps).

## Changes (each fully gated: full GPU suite + release gate 14/14 + parity)

1. **opt1_fused** — fused-noverlap kernels (`full_bls_no_sol_fused`,
   `full_bls_batch_fused`): one launch histograms at noverlap-x finer
   phase resolution, derives every dphi-shifted pass from fine-bin
   runs. Power-of-two noverlap + dphi=0 only (bit-identical bins);
   host multi-pass loop otherwise.
2. **opt2_scatter** — deterministic golden-stride permutation of the
   staged (t, yw, w): kills the same-phase-bin shared-atomic
   serialization of time-sorted dense cadences.
3. **opt3_host** — np.dot->einsum in per-LC host path (BLAS threadpool
   tripped CFS throttling on quota-limited containers: ~90 ms freeze
   per 100 ms period); builtin max()->np.max over nbins arrays (2.4-9 ms
   per call at survey nfreq, hidden in every launch); `eebls_gpu_batch`
   allocates one BLSBatchMemory per call instead of per chunk, uploads
   the grid once, prefix-only transfers, and accepts `memory=` for
   cross-call reuse.
4. **opt4_chunk** — occupancy-aware frequency chunking (8192/launch,
   per-chunk shared sizing) when shared memory is the occupancy
   limiter; batch kernels gained an explicit output-row-pitch arg
   (`bls_stride`).

## End-to-end per-lightcurve (ms/lc; RTX A5000)

`baseline` = v1.0 (89d5481) as-shipped under the pod's default
environment. `base_envfix` = same code with BLAS threadpools pinned
(the honest reference for the code changes; the env effect itself is
Finding 0 and is fixed IN the library by opt3). `final` = opt4_chunk.

### Best available path per survey ("survey mode")

| survey  | baseline best  | final best                              | speedup |
|---------|----------------|-----------------------------------------|---------|
| ZTF     | 1.62  (batch)  | **0.83** (fast_reuse; batch_reuse 0.84) | **2.0x** |
| HAT-Net | 58.17 (batch)  | **26.98** (batch_reuse)                 | **2.2x** |
| TESS    | 10.13 (batch)  | **0.80** (batch_reuse)                  | **12.7x** |
| Kepler  | 355.62 (batch) | **117.71** (batch_reuse)                | **3.0x** |

vs the thread-pinned baseline (isolating the code changes alone):
ZTF 1.45->0.83 (1.7x), HAT-Net 57.92->26.98 (2.1x), TESS 4.63->0.80
(5.8x), Kepler 351.87->117.71 (3.0x).

### Kernel-only (data resident; measures the GPU work itself)

| survey  | base_envfix | final  | speedup |
|---------|-------------|--------|---------|
| ZTF     | 5.63        | 0.63   | 8.9x    |
| HAT-Net | 76.26       | 26.10  | 2.9x    |
| TESS    | 4.49        | 0.49   | 9.2x    |
| Kepler  | 367.40      | 114.98 | 3.2x    |

(Pre-opt3, "kernel" timings silently included 0.1-9 ms/call of
Python-level `max()` over the bin-count arrays; the 8.9x ZTF row is
~5x GPU + host-in-the-loop removal. kernel_1pass shows the pure
per-pass GPU scaling.)

### Single-LC convenience path (`eebls_gpu_fast`, fresh call)

| survey  | baseline (default env) | final  | speedup |
|---------|------------------------|--------|---------|
| ZTF     | 7.91                   | 4.77   | 1.7x    |
| HAT-Net | 90.44                  | 32.84  | 2.8x    |
| TESS    | 59.80                  | 2.86   | **20.9x** |
| Kepler  | 375.38                 | 121.91 | 3.1x    |

## $/lightcurve (RTX A5000 @ $0.27/hr, best path, USD per million LCs)

| survey  | baseline | final      |
|---------|----------|------------|
| ZTF     | $0.12    | **$0.062** |
| HAT-Net | $4.36    | **$2.02**  |
| TESS    | $0.76    | **$0.060** |
| Kepler  | $26.67   | **$8.83**  |

## Full stage-by-stage table (ms/lc)

See `raw/bench_*.json`; condensed:

```
ZTF          baseline envfix opt1  opt2  opt3  opt4
 fast_naive     7.91   9.20  6.08  6.66  2.73  4.77
 fast_reuse     6.13   6.09  3.36  3.67  0.86  0.83
 kernel         5.70   5.63  3.15  3.33  0.62  0.63
 batch          1.62   1.45  1.24  1.22  1.22  2.05
 batch_reuse       -      -     -     -  0.76  0.84
HAT-Net
 fast_naive    90.44  94.52 66.24 54.70 32.12 32.84
 fast_reuse    79.74  76.92 37.83 35.94 26.67 26.79
 kernel        77.23  76.26 36.44 35.34 26.32 26.10
 batch         58.17  57.92 30.20 34.46 31.72 28.44
 batch_reuse       -      -     -     - 27.00 26.98
TESS
 fast_naive    59.80   6.16  4.26  6.83  3.27  2.86
 fast_reuse    40.03   4.90  2.07  1.25  1.15  1.16
 kernel         4.88   4.49  1.66  0.56  0.48  0.49
 batch         10.13   4.63  2.00  2.02  1.05  1.45
 batch_reuse       -      -     -     -  0.85  0.80
Kepler
 fast_naive   375.38 369.43 178.9 167.1 158.3 121.9
 fast_reuse   374.49 365.57 173.9 158.9 153.3 117.1
 kernel       370.75 367.40 172.4 155.4 151.5 115.0
 batch        355.62 351.87 172.5 157.8 158.0 119.0
 batch_reuse       -      -     -     - 156.4 117.7
```

Non-reuse `batch` bounces (1.22->2.05 ZTF, 1.05->1.45 TESS between
sessions): per-call cuMemHostAlloc latency on the shared host varies
by several ms; `batch_reuse`/`fast_reuse` are the stable survey paths
and the recommended usage.

## Correctness gates per change

| change | BLS tests (pod) | full suite             | release gate | parity vs base_envfix |
|--------|-----------------|------------------------|--------------|------------------------|
| opt1   | 438/438         | 759 passed / 7 skipped | 14/14        | corr=1.0000000, peaks identical, max diff 5.6e-6 |
| opt2   | 444/444         | 761 passed / 7 skipped | 14/14        | corr=1.0000000, peaks identical |
| opt3   | 447/447         | 764 passed / 7 skipped | 14/14        | corr=1.0000000, peaks identical |
| opt4   | 443/443 (bls)   | 766 passed / 7 skipped | 14/14        | corr=1.0000000, peaks identical |

Parity arrays: fast, fast+BJD (t+2455197.5), batch — for all four
surveys (raw/parity/*.npz; comparator benchmarks/compare_parity.py).
Baseline suite on 89d5481: 752 passed / 7 skipped, gate 14/14.

Flagged (non-silent) numerical notes:
- einsum vs BLAS ddot changes float64 summation order of the
  normalization scalars (last-ulp, ~1e-16 relative).
- fused kernel + scatter permutation change float32 accumulation
  ORDER of bin/box sums (max observed periodogram delta 5.6e-6 on
  Kepler, i.e. the same class as the run-to-run atomic
  nondeterminism both old and new kernels already have).
- No accuracy-for-speed trades were taken.

## Default-environment robustness verification (post-opt3)

Re-ran the original CFS-throttling reproducer (TESS-scale reuse loop,
8 LCs) with NO threadpool env vars on the same pod after the einsum
change: **3.10 ms/lc, 0 throttle events, 0 throttled time** (was
~52 ms/lc with +5 throttle events per loop and +23 s cumulative
thread-throttle time on v1.0 code under the same default env).
cuvarbase no longer needs OPENBLAS_NUM_THREADS pinning on
CPU-quota-limited hosts.
