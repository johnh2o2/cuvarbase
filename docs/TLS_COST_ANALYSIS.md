# TLS fidelity, throughput, and cost: cuvarbase vs CPU vs GTLS

> **Note (September 2026):** every SDE figure in this document was computed with the July-2026 `tls_stats` (signal residue SR = 1 - chi2/max(chi2)). cuvarbase 1.0 defines SR = chi2_min/chi2 (see CHANGELOG.rst), which moves every SDE value; the timing, cost and recovery results are unaffected.

Three questions, answered with measurements (RTX A5000, `scripts/tls_fidelity_experiment.py`,
`scripts/tls_matched_timing.py`, `scripts/benchmark_tls_survey.py`; raw in
`benchmarks/results/tls_survey_jul2026/`):

1. Is the coarse-epoch-grid + refinement fast path **lossy** — does it sacrifice SNR/SDE?
2. How much **faster** is it, apples-to-apples (same light curves, same grid, same detectability)?
3. Is it **cheaper**, and is it the cheapest TLS available?

## 0. What the reference "CPU pipeline" is

The `transitleastsquares` package (Hippke & Heller 2019), pip-installed, called as a
user would: `transitleastsquares(t, y, dy).power(R_star=1, M_star=1, period_min, period_max,
oversampling_factor=3, use_threads=cpu_count())`. It runs on *all* CPU cores. All CPU
timings below are that package on the same machine as the GPU (a RunPod pod), except the
4-year Kepler row (>15 min/LC) which uses the published 522 s figure (16-core Ryzen 9
7950X, GTLS paper).

## 1. Fidelity: it is NOT lossy in detectability (measured)

The detection statistic is the SDE, built from the whole χ²(period) spectrum. cuvarbase's
default fast path scans a **coarse epoch grid** (`t0_oversample=3`, ~3 epochs per transit
duration) plus an exact refinement of the top candidate periods; the reference steps t0
~100× finer *everywhere*. Does that cost detectability?

To compare cleanly, the *statistic* is held fixed: cuvarbase and the reference normalize
SR→SDE differently, so SDE is recomputed with `cuvarbase.tls_stats` on **both** methods'
χ² spectra. Only spectrum fidelity then varies. Identical injected light curves, one
shared Ofir period grid.

| Signal | cuvarbase t0=3 (default) | cuvarbase t0=33 (matched) | reference | recovery |
|---|---:|---:|---:|:--:|
| tess-ffi, depth 0.005 (strong) | SDE 14.5 (**0.99×**) | 15.0 (**1.03×**) | 14.61 | 12/12 all |
| tess-ffi, depth 0.002 (marginal) | 12.01 (**0.97×**) | 12.47 (**1.01×**) | 12.37 | 10/10 all |
| k2, depth 0.004 (narrow, q≈0.014) | 25.23 (**0.98×**) | 26.11 (**1.01×**) | 25.82 | 6/6 all |

**The default fast path is within 1–3% of the reference SDE, and matched (t0=33) is within
1%.** 100% recovery in every case, including a marginal near-threshold depth and a narrow
transit — the two regimes where any loss would show.

Why the coarse epoch grid barely moves the SDE: **SDE is a period-space contrast,
`(peak − mean)/std` of the spectrum.** A coarser t0 grid lowers the best-fit quality at
*every* trial period by roughly the same amount, so the normalized contrast between the
true-period peak and the background is preserved. The finer reference grid raises all fits,
again roughly uniformly. The epoch grid mostly sets *reported t0/parameter precision* — and
that is exactly what the exact refinement pass restores. The duration-scaled t0 grid also
guarantees at least one tested epoch overlaps the transit, so even narrow transits don't
fall through.

The small residual (1–3% at default) is in the **safe direction**: cuvarbase slightly
*under*-reports significance, never over-reports. Refinement is deliberately excluded from
the SDE (it feeds parameters only) precisely so the statistic stays on a uniform-fidelity
spectrum — sharpening only the peak would *inflate* SDE and manufacture false positives.

Earlier internal notes cited a "~5–15% SDE loss." That was a *cuvarbase-fast-vs-cuvarbase-legacy*
artifact (two of our own kernels), **not** a loss versus the reference. Against the actual
reference package it is parity.

## 2. Throughput, apples-to-apples

Matched fidelity (t0=33, SDE parity confirmed above) costs ~5–13× over the default coarse
grid. Archived points: tess-ffi 5.3–6.0×, k2 11.9×
(benchmarks/results/tls_survey_jul2026/fidelity_raw_a5000.txt); TESS-yr 12.8×
(25.3 → 325.2 ms/LC) and Kepler-4yr 8.1× (188.3 → 1520.5 ms/LC), 100% recovery at both
fidelities (benchmarks/results/tls_survey_jul2026/matched_timing_a5000_jul2026.txt,
re-measured on the v1.0.0 release-gate pod — the earlier unarchived session printed
14.6×/8.4× with 176.8 → 1479 ms; same ballpark, pod-to-pod variation).

Same light curves, same period grid, single A5000 GPU vs all CPU cores of the same pod:

| Regime | cuvarbase default | cuvarbase matched (SDE parity) | reference CPU | speedup (matched / default) |
|---|---:|---:|---:|---:|
| tess-ffi (marginal) | 2.6 ms/LC | 13.8 ms/LC | 46,222 ms/LC | 3,300× / 17,500× |
| k2 (narrow) | 5.3 ms/LC | 63.1 ms/LC | 61,445 ms/LC | 970× / 11,600× |

So **even at genuine SDE parity (matched t0=33), cuvarbase is ~1,000–3,000× faster than the
reference TLS on the same machine**; at the default grid (already SDE-parity for detection)
it is ~11,000–17,000×. Caveat: this pod's reference is unusually slow (46–61 s/LC — a
slower CPU and 96-thread oversubscription on a small problem); a faster CPU narrows the raw
speedup. **Throughput ratio is the market-independent invariant; the exact multiplier is
CPU-dependent.** The robust claim is "thousands of times faster."

## 3. Cost

Cost = throughput × ($/hr). The throughput advantage above is measured and market-independent;
the dollar multiplier depends entirely on how you price the two markets, and an earlier
version of this note over-pinned it by comparing a lucky **$0.16/hr spot GPU against a
$2.72/hr AWS on-demand CPU** — two different markets. Corrected inputs:

- **GPU**: RunPod A5000 list price is **$0.27/hr** (I paid $0.16 on some spot pods and $0.27
  on others — it fluctuates). Use $0.27.
- **CPU**: RunPod does not publish CPU-pod pricing; AWS on-demand 16-vCPU `c6i.4xlarge` is
  **$0.68/hr**, 64-vCPU `c6i.16xlarge` is $2.72/hr. Cross-market, so treat as indicative only.

Cost per million light curves at genuine full fidelity (matched t0=33, A5000 $0.27/hr):

| Regime | cuvarbase matched | reference CPU | note |
|---|---:|---:|---|
| Kepler-4yr | ~$114/M | ~$98,600/M (16-core, published 522 s) | ~860× cheaper |
| TESS-yr | ~$24/M | (not measured) | measured 325.2 ms/LC matched (matched_timing_a5000_jul2026.txt) |

At the default grid (already detection-parity): Kepler ~$13/M, TESS-FFI a few cents/M. But
the honest headline is the **throughput invariant (thousands×)**, not a single dollar ratio;
the ~890× above already uses the *most* CPU-favorable pairing (cheap 16-vCPU CPU, list-price
GPU, full-fidelity GPU). Under any reasonable pricing, GPU TLS is hundreds-to-thousands of
times cheaper.

## 4. Versus GTLS (the only other GPU TLS)

[GTLS](https://arxiv.org/abs/2607.00348) (arXiv:2607.00348, Hu, Ge, Jin, Willis, 1 Jul 2026;
CuPy, RTX 4090) reports a 3000-day light curve in **138 s** (single GPU) / 79 s (dual) vs
**3289 s** for CPU TLS → 24× / 42×, at TLS-equivalent detection (matched precision/recall).
A 1500-day case is ~33 s. cuvarbase does the comparable Kepler-4yr configuration in **177 ms/LC
at the default grid** (SDE-parity) or **1.48 s/LC at matched t0=33** on an A5000 (< a 4090):

| | fidelity | time/LC | vs GTLS 1500-day |
|---|---|---:|---:|
| GTLS (RTX 4090) | TLS-matched | ~33 s | 1× |
| cuvarbase matched (A5000) | SDE parity, matched t0 | 1.48 s | ~22× faster |
| cuvarbase default (A5000) | SDE parity for detection | 0.177 s | ~190× faster |

cuvarbase wins on hardware-hours (hand-written kernels + phase-binned scan vs CuPy per-point)
and on hardware price (A5000 < 4090), on a fidelity basis GTLS's own detection metric would
call equivalent.

## Bottom line

- **Not lossy.** Detection SDE is at parity with the reference (0.97–1.03×) with 100%
  recovery, including marginal and narrow transits. The coarse grid trades *epoch/parameter
  precision* for speed, and the refinement restores that. Apples-to-apples (matched t0=33) is
  within 1% of the reference SDE.
- **Fastest.** ~1,000–3,000× faster than reference CPU TLS at genuine SDE parity on the same
  machine; ~22–190× faster than GTLS on cheaper hardware.
- **Cheapest.** Hundreds-to-thousands of times cheaper per light curve than CPU TLS under any
  reasonable pricing, and cheaper than GTLS. The exact dollar multiplier is pricing-dependent;
  the throughput invariant is not.
- Still **EXPERIMENTAL** pending a full injection–recovery *completeness* campaign across a
  (period, depth, ndata) grid (item D3). Three-regime SDE parity is strong evidence, not a
  completeness proof.
