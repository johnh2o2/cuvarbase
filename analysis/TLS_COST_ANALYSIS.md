# TLS cost analysis: GPU vs CPU, and vs the only other GPU TLS

**Question.** With the survey-scale fast TLS path (`tls_search_batch`), is it
cheaper to run a Transit Least Squares search on a rented GPU than on a CPU?
And is cuvarbase now not just the fastest but the *cheapest* TLS available?

**Short answer.** Yes on both counts, by a wide margin. At its default
fidelity, cuvarbase GPU TLS costs **$0.06–$7.45 per million light curves**
depending on regime, versus **$11,000–$99,000 per million** for the reference
CPU `transitleastsquares` package — a **13,000× to 200,000×** cost reduction.
It is also ~600× cheaper than GTLS, the only other GPU TLS. Read the fidelity
caveat at the end before quoting the largest ratios.

## Method

Throughput is the measured end-to-end survey wall time (grid generation +
preprocessing + host↔device transfer + kernels + per-LC statistics),
`scripts/benchmark_tls_survey.py`, 100% injected-transit recovery in every
regime on every GPU. Raw JSON in `benchmarks/results/tls_survey_jul2026/`.

Cost per million light curves is `(ms_per_lc × 1000 / 3600) × $/hr`.

- **GPU $/hr** are the prices actually paid on RunPod this session: A5000
  **$0.16**, RTX 4000 Ada **$0.20**, Tesla V100 **$0.23**.
- **CPU $/hr** uses AWS on-demand as a defensible public anchor: 64 vCPU =
  `c6i.16xlarge` **$2.72/hr** (the reference runs used `use_threads=cpu_count()`
  = 64 on the pod); the compute-bound Kepler row uses 16 vCPU `c6i.4xlarge`
  **$0.68/hr** to match the published 522 s / 16-core Kepler baseline.
- **Reference CPU** = the `transitleastsquares` package (Hippke & Heller 2019),
  same forced Ofir period grid, `oversampling_factor=3`, all cores. Measured on
  the pod for four regimes; the 4-year Kepler point (>15 min/LC) uses the
  published 522 s figure (Ryzen 9 7950X, 16 cores; GTLS paper arXiv:2607.00348).

## Cost per million light curves

| Regime (ndata, periods) | A5000 | Ada | V100 | Cheapest GPU | Reference CPU | GPU savings |
|---|---:|---:|---:|---|---:|---:|
| TESS FFI (1.3K, 2.5K)   | **$0.056** | $0.170 | $0.091 | A5000 $0.056 | $11,110 | ~199,000× |
| K2 90-d (4.3K, 9.7K)    | **$0.138** | $0.356 | $0.247 | A5000 $0.138 | $16,226 | ~118,000× |
| TESS 2-min (19.7K, 2.5K)| **$0.125** | $0.283 | $0.197 | A5000 $0.125 | $15,643 | ~125,000× |
| TESS 1-yr (17K, 42K)    | **$0.819** | $1.519 | $1.054 | A5000 $0.819 | $83,930 | ~102,000× |
| Kepler 4-yr (65K, 172K) | **$7.45**  | $10.99 | $9.30  | A5000 $7.45  | $98,600 | ~13,000× |

Two robust conclusions:

1. **GPU TLS is dramatically cheaper than CPU TLS.** The ratio is the
   throughput advantage (~3,000–12,000×) multiplied by the hourly-cost
   advantage (a $0.16/hr GPU beats a multi-core CPU box), so it holds under any
   reasonable CPU price — even pricing the CPU at the GPU's $0.16/hr leaves the
   throughput gap intact.

2. **The RTX A5000 is the cost sweet spot.** It is not always the fastest
   (the V100 edges it on the biggest regime), but at $0.16/hr it is the
   cheapest to operate in every regime. Fastest-per-dollar ≠ fastest.

At A5000 rates, a full **TESS FFI sector–scale run of ~1 million light curves
costs about 6 cents** of GPU time; a **Kepler-depth 4-year, 65K-point, 172K-period
search of a million targets costs about $7.45** — versus roughly $100,000 for
the same million on the reference CPU pipeline.

## Versus the only other GPU TLS (GTLS)

GTLS (arXiv:2607.00348, submitted 1 Jul 2026; CuPy) is the sole other GPU TLS.
On a ~1500-day / 67K-point / 190K-period Kepler-class light curve it reports
**33.3 s/LC on an RTX 4090** (15.7× over reference TLS). cuvarbase does the
comparable Kepler-4yr configuration in **168 ms/LC on an A5000**.

| | Time/LC | GPU $/hr | $/million LC |
|---|---:|---:|---:|
| GTLS, RTX 4090 | 33.3 s | ~$0.50 | ~$4,625 |
| cuvarbase, A5000 | 0.168 s | $0.16 | **$7.45** |

≈ **620× cheaper** than GTLS, on a cheaper GPU. cuvarbase wins on hardware-hours
(hand-written kernels + phase-binned scan vs CuPy per-point) and on hardware
price (A5000 < 4090).

## The honest caveat: fidelity

The largest ratios are partly a fidelity trade, and the comparison is only fair
if that is stated:

- **Default epoch grid.** cuvarbase's default coarse scan steps the transit
  epoch at `t0_oversample=3` (~3 positions per transit duration), then runs an
  **exact per-point refinement** at `refine_oversample=33` on the top candidate
  periods. The reference package steps ~100× finer (`T0_FIT_MARGIN=0.01`)
  *uniformly*. So cuvarbase evaluates far fewer coarse trials, which is a large
  part of why it is faster — not implementation efficiency alone.
- **What we verified.** 100% injected-transit recovery in all five regimes on
  all three GPUs; agreement with the reference package on the golden configs
  (period error <1%, both packages flag the detection significant); coarse
  chi² spectrum correlated 0.998 with the legacy per-point cuvarbase kernel.
  This is strong evidence the default fidelity is science-useful, but it is not
  a bit-for-bit statistical match to the reference's ~100× epoch grid, and no
  full injection–recovery completeness campaign has been run yet (that is the
  open D3 validation item; the module remains flagged EXPERIMENTAL).
- **Matched fidelity is still a win.** Raising `t0_oversample` toward the
  reference grid costs roughly linearly in the coarse scan (~70% of Kepler-4yr
  time). A reference-matched run is an estimated ~5–10× slower — order
  **1–1.7 s/LC on an A5000, ~$40–75/million** — still **>20× faster/cheaper
  than GTLS** and **>300× cheaper than the reference CPU**. (Estimate from the
  stage profile; not yet measured end-to-end.)

## Bottom line

At default fidelity cuvarbase is, on the evidence here, both the fastest and the
cheapest TLS available — thousands of times cheaper than CPU TLS and ~600×
cheaper than the only other GPU TLS. Even conservatively adjusted to the
reference's finer epoch grid, it remains the cheapest by a large margin. The one
thing still owed before dropping the EXPERIMENTAL flag is a full
injection–recovery validation at matched fidelity (item D3), not a speed or cost
result.
