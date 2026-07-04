# cuvarbase v1.0.0 vs 0.2.6 — measured head-to-head (July 2026)

**Purpose.** Release-notes performance numbers for v1.0.0 against the last
official release, 0.2.6. Replaces the retracted "21-390x vs pre-v1.0" numbers,
whose baseline paid per-call nvcc compilation (see
`analysis/BENCHMARK_PROTOCOL_V1.md` sections 4.1 and 8, which ban their reuse).
This campaign follows that protocol's fairness rules (section 4) with the
maintainer-approved amendment that the baseline is the 0.2.6 release, not a
master checkout.

**Date:** 2026-07-04. **GPU:** NVIDIA RTX A5000 (RunPod, driver 570.195.03,
CUDA driver API 12080, nvcc 12.4.131), single otherwise-idle GPU.
**v1.0.0 tree:** branch `v1.0-fixes` @ `2cc1f96` (the v1.0.0 release
candidate). **Baseline tree:** git tag `v0.2.6` @ `0d97ae1`.

## Release-notes finding #0: 0.2.6 was never on PyPI

`pip install cuvarbase==0.2.6` fails: **the 0.2.6 release (GitHub, May 2025)
was never uploaded to PyPI**. The last PyPI artifacts are **0.2.5
(2023-10-23)**. The baseline here was therefore installed from the `v0.2.6`
git tag (master = 0.2.6 + 13 non-user-facing maintenance commits, so this is
equivalent and closer to what users actually have). Release notes should
mention that v1.0.0 is the first PyPI release since 0.2.5.

## Environments (separate venvs; 0.2.6 creates a CUDA context at import)

| stack | python | numpy | pycuda | used for |
|---|---|---|---|---|
| v1.0.0 | 3.11.10 | 2.4.6 | 2026.1 | everything |
| 0.2.6 (venv A) | 3.11.10 | 1.23.5 | 2025.1 | BLS (cold/warm/loop/correctness), decomposition |
| 0.2.6 (venv B) | 3.11.10 | 1.23.5 | 2022.2.2 | LS, PDM, BLS warm cross-check |

Two 0.2.6 stacks were required because **0.2.6's Lomb-Scargle and PDM
segfault under pycuda 2025.1 + numpy 1.23.5** (bare
`pycuda.driver.register_host_memory` on an `aligned_zeros` array dumps core;
0.2.6's BLS never registers host memory, so it is unaffected). The BLS warm
cross-check confirmed pycuda version does not move the BLS numbers at the
100 ms scale (100.1 ms on both stacks at the TESS config).

Getting 0.2.6 running at all required: numpy pinned to 1.23.5 (scikit-cuda
0.5.3 is broken on numpy >= 1.24: `np.typeDict`/`np.float` removed),
`setuptools < 81` (0.2.6 uses `pkg_resources`), `pytest-runner` + `future`
(archaic `setup_requires`), and pycuda built from source against numpy 1.23
headers (`--no-build-isolation`). None of this is needed for v1.0.0.

## Methodology (protocol section 4)

- Identical seeded float64 inputs both sides (each version does its own
  float32 cast); identical `freqs` grid, `qmin=0.01`, `qmax=0.5`,
  `dlogq=0.3`, `block_size=256` (both defaults).
- **noverlap fairness rule:** 0.2.6's `eebls_gpu_fast` signature accepts
  `noverlap` and passes it to the kernel, but the compiled (linear-bin)
  kernel branch never uses it — **noverlap is silently ignored** (one phase
  pass regardless). The apples-to-apples row is therefore v1.0 at
  `noverlap=1`. v1.0's default `noverlap=2` (two dphi-shifted passes,
  elementwise max — the fix) is a separate row.
- **Warm** = compile excluded on both sides: 0.2.6 gets precompiled
  `functions=` handles (its supported API); v1.0 uses its LRU kernel cache.
  >= 2 discarded warmups, medians of >= 7 timed calls per round,
  `Context.synchronize()` inside every timing. Warm rows pool 2-3
  independent rounds run in alternating version order (14-21 samples/cell).
- **Cold** = fresh process, `~/.cache/pycuda` and `~/.nv/ComputeCache`
  deleted first; first-call time includes nvcc compilation.
- **Loop** = fresh process, product defaults (no `functions=` handle), 20
  distinct seeded light curves; 100-LC figure extrapolated as
  `first_call + 99 x steady_median` (stated, not measured).
- Configs: **canonical** = ndata 10,000, nfreq 5,000 (freqs = k x 2.0/5000,
  10-yr baseline; matches the 7-GPU campaign config); **small** = ndata 500,
  same grid; **tess** = ndata 20,000, nfreq 13,500 (27.4-d baseline, P >=
  0.5 d).

## 1. Standard BLS — warm / steady state (the kernel-vs-kernel comparison)

| config | 0.2.6 warm (functions= precompiled) | v1.0 noverlap=1 | ratio | v1.0 noverlap=2 (default) |
|---|---|---|---|---|
| canonical (10k x 5k) | 5.9 ms [5.1, 7.0] | 7.8 ms [6.9, 10.1] | 0.76x | 9.4 ms |
| small (500 x 5k) | 3.5 ms [3.4, 3.5] | 4.2 ms [2.9, 5.8] | 0.84x | 7.8 ms |
| tess (20k x 13.5k) | 100.1 ms [97.5, 107.2] | 113.5 ms [99.9, 201.4] | 0.88x | 100.5 ms |

**Steady-state GPU throughput is unchanged.** The decomposition below shows
the kernels are identical to within 0.5%; the 0.76-0.88x warm ratios are a
few ms of added v1.0 host work per call (float64 epoch subtraction, chi2_0
for the new power conventions, cache lookup, output conversion) plus
run-to-run jitter at the ms scale (pod CPU; same-stack 0.2.6 rounds varied
3.5 -> 7.2 ms between rounds, and 5.9 vs 13.2 ms across its two pycuda
stacks at the canonical config).

### Warm-call decomposition (tess config, 15 reps, interleaved order)

| variant | v1.0 (run 1) | 0.2.6 | v1.0 (run 2, drift check) |
|---|---|---|---|
| A: product call (as users call it) | 112.3 ms | **500.0 ms** | 112.5 ms |
| B: `functions=` precompiled | 100.0 ms | 100.1 ms | 100.0 ms |
| C: B + `memory=` reused | 100.0 ms | 100.0 ms | 100.0 ms |
| D: C without H2D/D2H (kernel only) | **9.8 ms** | **9.8 ms** | 9.9 ms |

Kernel-only time (D) is identical — the v1.0 correctness fixes (degenerate
all-weight-box guard, etc.) cost nothing measurable. The ~90 ms gap between
C and D is per-call host data preparation + transfers **inherited unchanged
from 0.2.6** (identical both sides). Row A is what a user actually pays:
0.2.6 rebuilds/reloads the module every call even after nvcc's disk cache is
warm (~400 ms of `SourceModule` machinery per call); v1.0 pays a ~12 ms
cache-hit path.

## 2. Standard BLS — cold / out-of-the-box (fresh process, compiler caches cleared)

| config | version | import+ctx | first call (incl. compile) | second call |
|---|---|---|---|---|
| canonical | 0.2.6 | 0.91 s | 2.37 s | 293.7 ms |
| canonical | **v1.0** | 0.49 s | **1.67 s** | **13.8 ms** |
| small | 0.2.6 | 0.84 s | 2.42 s | 195.9 ms |
| small | **v1.0** | 0.36 s | **1.73 s** | **10.8 ms** |
| tess | 0.2.6 | 1.02 s | 2.89 s | 338.4 ms |
| tess | **v1.0** | 0.54 s | **2.00 s** | **43.9 ms** |

Cold first call is ~1.4x faster; the second call tells the real story
(21x/18x/8x): 0.2.6 keeps paying a few hundred ms per call forever.

## 3. The headline: naive per-lightcurve loop (canonical config)

What a pipeline that loops `eebls_gpu_fast` over light curves pays
(product defaults, no expert `functions=` plumbing):

| version | first call | steady per-call | 20-LC loop total | 100-LC (extrapolated) | effective per LC |
|---|---|---|---|---|---|
| 0.2.6 | 2.56 s | 261.1 ms | 7.53 s | 28.4 s | 284 ms |
| **v1.0** | 2.07 s | **7.6 ms** | **2.28 s** | **2.8 s** | **28 ms** |

**34x faster per light curve at steady state; 10x for a 100-LC batch
including one-time compile.** This is the honest replacement for the
retracted numbers: the gain is real but it is a *compile/caching-architecture*
fix (0.2.6 recompiled per call), not a kernel speedup — and unlike the
retracted benchmark, both sides here were also compared with compilation
excluded (section 1: parity).

## 4. Correctness — parity + the BJD demo

Injected transit: P = 3.456 d (f = 0.289352 /d), q = 0.03, depth = 0.008,
ndata = 3,000, 27.4-d baseline, noverlap=1 both sides, 7,800 freqs.

| version | timescale | peak freq (/d) | peak power | recovered? |
|---|---|---|---|---|
| 0.2.6 | t near 0 | 0.289231 | 0.3010 | yes |
| 0.2.6 | t + 2457000 (BJD) | 0.290769 | **0.0893** | **NO** |
| v1.0 | t near 0 | 0.289231 | 0.3010 | yes |
| v1.0 | t + 2457000 (BJD) | 0.289231 | 0.3010 | **yes** |

- Parity at near-zero timestamps: periodogram correlation v1.0 vs 0.2.6
  **r = 1.000000** (identical peak to 6 decimals) — the algorithm is
  unchanged where 0.2.6 was correct.
- **BJD-scale timestamps silently destroy 0.2.6's periodogram** (float32
  phase fold: at t ~ 2.457e6 the float32 grid is 0.25 d wide): peak power
  drops 3.4x, the peak lands off the injected frequency, correlation with
  the near-zero periodogram falls to r = 0.806. v1.0 epoch-subtracts in
  float64 before the float32 cast and returns bit-comparable results at both
  timescales (r = 1.000000). No warning, no error in 0.2.6 — just a wrong
  answer on the time convention every modern survey uses.
- v1.0's `noverlap=2` default additionally fixes undersampling of boxes near
  the finest phase bin — 0.2.6 accepted the argument and did nothing.

## 5. Lomb-Scargle (process reused, warm; median of 7)

| config | 0.2.6 (pycuda 2022.2.2) | v1.0 | ratio |
|---|---|---|---|
| ndata=10,000, nf=5,000 | 8.8 ms [7.6, 11.2] | 9.2 ms [7.0, 9.7] | 0.96x (parity) |
| ndata=3,000, nf=100,000 | 33.3 ms [17.7, 43.3] | 11.7 ms [11.2, 12.6] | **2.85x** |

Identical peak frequency recovered in all four cells (injected 0.7431/d).
At survey-realistic grid sizes v1.0's reworked NFFT path is ~3x faster and
far less jittery. **On the modern stack 0.2.6's LS does not run at all**
(scikit-cuda segfault path above) — it needed a 2022-era pycuda to produce
these numbers.

## 6. PDM (binned_linterp, nbins=10, ndata=3,000, nf=10,000)

| variant | 0.2.6 | v1.0 | ratio |
|---|---|---|---|
| binned_linterp (same algorithm) | 3.7 ms | 3.4 ms | 1.08x |
| binned_linterp_fast (new in v1.0) | 3.7 ms | 2.8 ms | 1.33x |

## Caveats a release-notes author must state

1. Single GPU model (RTX A5000); ratios at other GPUs will differ mainly via
   compile/launch overhead vs kernel time balance.
2. The 100-LC loop figure is extrapolated from a measured 20-LC loop
   (`first + 99 x steady`); the 20-LC totals are directly measured.
3. Steady-state kernel throughput is **not** improved (0.76-0.88x warm
   ratios with precompiled handles, identical kernel-only time); do not
   claim a kernel speedup. The 34x/10x numbers are the product path
   (compile-once + cache vs recompile-per-call).
4. v1.0's default `noverlap=2` does ~2x the kernel work of 0.2.6's silent
   single pass; at ndata >= 10k configs the wall-time impact is small
   because per-call host prep dominates, but it is visible at small ndata
   (small config: 3.5 ms -> 7.8 ms). It is a correctness feature, priced
   honestly.
5. 0.2.6 BLS numbers were obtained with pycuda 2025.1 and cross-checked
   with 2022.2.2 (identical at the 100 ms scale; few-ms configs jitter
   2-4x between rounds on the pod regardless of stack).
6. These numbers supersede and must not be mixed with the retracted
   "21-390x" table (protocol sections 4.1/8).

## Reproduction

- `scripts/bench_v026_head_to_head.py` (version-agnostic; run under each
  venv), `scripts/decomp_v026_head_to_head.py` (warm decomposition),
  `scripts/summarize_v026_head_to_head.py` (tables from raw JSON).
- Raw JSON (all timings, per-call samples, env versions, full periodograms
  for the correctness demo): `raw/` in this directory; driver logs in
  `logs/`.
