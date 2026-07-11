# cuvarbase vs GTLS — apples-to-apples reproduction of the GTLS Fig. 7 benchmark

**What this is.** GTLS (Hu, Ge, Jin & Willis, arXiv:2607.00348, submitted 1 Jul 2026)
is the first and only *other* GPU implementation of Transit Least Squares — a CuPy
reimplementation of Hippke & Heller's (2019) TLS (`pip install gputls`, v0.5.1).
Their Fig. 7 reports single-light-curve search time vs light-curve baseline for
GTLS, reference CPU-TLS, and cuvarbase's GPU-BLS. This document reproduces that
figure **on one GPU, holding the search fair**, using our improved TLS
(`feature/tls-fast-survey`) and improved BLS (`feature/bls-survey-speed`).

**Figure:** `gtls_fig7_reproduction.png` (this directory). Benchmark: `scripts/gtls_benchmark/`. Raw data: `benchmarks/results/gtls_comparison_jul2026/`.

All measurements: single RTX A5000 (24 GB, sm_86), CUDA 12.x, cupy 13.6,
one injected batman transit per baseline (P=8.13 d, depth=4e-3, 110–400 ppm-class
noise, Keplerian-consistent duration so both grids bracket it), 30-min cadence.
GTLS ran on the *same* A5000 as cuvarbase, so all ratios below are same-hardware.

---

## 1. The fairness protocol (what "apples-to-apples" required)

The GTLS paper's absolute numbers are on an RTX 4090 (GTLS/BLS) and a Ryzen 7950X
(CPU-TLS). Rather than trust cross-hardware ratios, we run **every method on the
same A5000** and equalize the *search*, not just the hardware. Five knobs had to
be matched (each was a real gap):

| axis | GTLS | cuvarbase default | how we matched it |
|---|---|---|---|
| **period grid** | Ofir, os=3, Pmax=S/2 | Ofir, os=3 | identical: the *same* array passed to all methods (grids already agreed to 0.05%: 191,837 vs 191,742 at 1500 d) |
| **epoch (T0) density** | `T0_fit_margin` → SKIP_POINT = 8 epochs/duration (default); =0 → every cadence (paper Fig 7) | `t0_oversample`=3 | cuvarbase-matched uses `t0_oversample=8`; GTLS run at both settings |
| **duration grid** | ~36/period over a q-window of ratio ~31 (log-1.1) | 15 over [0.5q,2q] | cuvarbase-matched uses `n_durations=38` and per-period `qmin/qmax` = GTLS's own kernel window |
| **template** | Hippke reference LD (a=23.1, b≈0.32) | LD (a=15, b=0) | left as-is — measured to cost <3% SDE (below) |
| **light curve / SNR** | — | — | one injected transit per baseline, fed to *all* methods → identical SNR by construction |

Every method's chi²(P) (or BLS power) spectrum is additionally re-scored with **one
identical SDE routine**, so "detection significance" means the same thing for all.

**On the epoch axis (the crux).** GTLS exposes epoch density through
`T0_fit_margin`: the default 0.125 compiles to `SKIP_POINT=8` = **8 trial epochs
per transit duration** in the coarse SDE scan; `T0_fit_margin=0` scans **every
cadence** (its most expensive O(N²) mode). We measured both. Our A5000
`gtls_full` numbers (393 s at 1000 d) extrapolate to ~1200 s at 1500 d — 35× the
paper's 33.3 s, implausible even after hardware — whereas `gtls_skip8` (76 s at
1000 d → ~155 s at 1500 d, ≈60 s hardware-adjusted for a 4090) lands within ~2× of
the paper. **So the paper's Fig. 7 used GTLS's *default* (skip=8), not full-scan.**
The true apples-to-apples is therefore **cuvarbase-TLS at `t0_oversample=8` vs
GTLS-skip8** (both = 8 epochs/duration); `gtls_full` is shown only as a "finest
epoch" upper curve.

---

## 2. Results — runtime (per light curve, same A5000)

Per-light-curve search time, all on the same A5000 (GTLS `full`/`skip8` measured
directly through 1000 d; `skip8` also at the paper's 1500/2000/3000 d anchors;
`full` beyond 1000 d omitted — it reaches ~20 min/point):

| baseline | GTLS full | **GTLS skip8 (paper cfg)** | **cuv TLS matched** | cuv TLS default | cuv BLS (Kunimoto) | cuv BLS (sensible) |
|---:|---:|---:|---:|---:|---:|---:|
| 200 d  | 5.9 s  | 4.1 s   | **0.138 s** | 0.041 s | 0.538 s | 0.011 s |
| 500 d  | 60.2 s | 22.3 s  | **0.402 s** | 0.119 s | 1.485 s | 0.032 s |
| 1000 d | 392.6 s| 75.8 s  | **0.883 s** | 0.232 s | 3.279 s | 0.101 s |
| 1500 d | (~1200 s*) | 177.9 s | **1.437 s** | 0.409 s | 5.292 s | 0.207 s |
| 2000 d | —      | 348.3 s | 2.037 s | 0.627 s | 7.499 s | 0.346 s |
| 3000 d | —      | (~830 s*) | 3.460 s | 1.162 s | 12.626 s | 0.730 s |

\* extrapolated. GTLS's full-scan mode scales **super-quadratically** (measured exponent ≈2.5–2.7; the skip-8 mode used for the headline comparison measures ≈1.9–2.2),
because on a 24 GB GPU long light curves force tiny period batches → thousands of
Python-driven per-batch kernel launches. cuvarbase scales cleanly ~linearly.
(For reference the paper's own 4090 GTLS points are 33.3 s @1500 d and 138 s
@3000 d — i.e. skip=8 on faster hardware.)

**Speedup, cuvarbase-TLS-matched vs GTLS-skip8 (same A5000, matched 8
epochs/duration, matched durations & period grid, equal SDE):**

| baseline | 200 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|
| **speedup** | **30×** | **55×** | **86×** | **124×** | **171×** |

The epoch-matched speedup *grows monotonically* with baseline (GTLS's per-call
recompile + launch overhead compound); cuv-TLS *default* is a further ~3–4× on top,
and vs GTLS-*full* the ratio is 43× → 150× → 445×.

**Cross-check against the paper's own hardware (immune to the A5000-vs-4090
question).** Take the paper's *published* GTLS numbers on its RTX 4090 and compare
to cuvarbase on our *slower* A5000:

| baseline | paper GTLS (RTX 4090) | cuvarbase-TLS-matched (A5000) | cuvarbase wins by |
|---|---|---|---|
| 1500 d | 33.3 s | 1.44 s | **23×** |
| 3000 d | 138 s | 3.46 s | **40×** |

cuvarbase on the weaker GPU already beats GTLS on the stronger GPU by 23–40× — and
would widen further on matched hardware. (Our *same-GPU* GTLS is ~5× slower than
the paper's 4090 GTLS, more than the ~2× hardware gap: GTLS's runtime is dominated
by per-batch kernel-launch overhead that is very GPU/driver/CuPy-version-sensitive.
We anchor on both the same-GPU ratio and this paper-hardware cross-check so the
conclusion holds either way.)

**Bonus — improved BLS.** At the paper's *exact* Kunimoto BLS config, our July
`feature/bls-survey-speed` batched BLS runs **5.3 s @1500 d on the A5000 vs the
paper's reported 121.1 s cuvarbase-BLS on a 4090 — ~23× faster on weaker
hardware** (opt1–opt4 + batched kernel; the paper's exact cuvarbase entry point /
version is unspecified).

## 2b. Single light curve — GTLS's home turf, and the cold-start case

Every number above is already **single-light-curve** (GTLS has no batch API, so
cuvarbase was timed one LC at a time too — batching would only widen the gap). The
warm speedups assume the kernel JIT is compiled, which amortizes across any real
workload. For the strict **cold single shot** — one star, a fresh process, kernel
compile *included*, and the on-disk pycuda/cupy kernel cache *cleared* before every
run (first-run / fresh-container worst case) — full launch-to-answer wall time on a
second A5000:

| baseline | cuvarbase-TLS (matched) | GTLS-skip8 | cold ratio |
|---:|---:|---:|---:|
| 200 d  | 4.1 s | 10.7 s | **2.6×** |
| 500 d  | 4.5 s | 27.8 s | **6.1×** |
| 1000 d | 4.8 s | 83.8 s | **17×** |
| 1500 d | 5.6 s | 191.0 s | **34×** |

cuvarbase's cold cost is a ~fixed **~3–4 s kernel compile** that barely grows with
baseline (its search is 0.04–1.4 s); GTLS's cost is its *search*, which explodes —
so the ratio grows from 2.6× (both fixed-cost-bound at short baselines) to 34× at
Kepler length. This is the pessimistic floor: from the **2nd star onward** (disk
kernel cache warm) cuvarbase drops to ~0.5–2 s and the ratio snaps back toward the
warm 30–171×, while GTLS recompiles *and* re-searches on every call. SDE parity
holds cold too. (Raw: `benchmarks/results/gtls_comparison_jul2026/cold_single_shot_a5000.txt`;
harness: `scripts/gtls_benchmark/cold_shot.py` + `cold_driver.sh`.)

## 3. Results — detection significance (SDE parity)

Scored by the one identical statistic, **every method agrees closely at every
baseline** — GTLS vs cuvarbase-TLS to ~1–3%, and the full 6-method spread (which
includes BLS, whose box template scores marginally higher on this signal) ≤~10%:

| baseline | SDE: GTLS-skip8 / cuv-TLS-matched | full 6-method spread |
|---:|---|---|
| 200 d  | 34.2 / 33.8  (−1.4%) | 33.4 – 35.2 |
| 500 d  | 53.4 / 53.2  (−0.5%) | 53.1 – 56.5 |
| 1000 d | 89.5 / 88.7  (−0.9%) | 86.6 – 93.4 |
| 1500 d | 104.2 / 103.5  (−0.6%) | 99.9 – 110.9 |
| 3000 d | — / 150.4 | 150.2 – 161.7 |

100% recovery of the injected period in all cells. So the large speed gaps are
**not** bought with sensitivity — the whole point of the fair comparison. This
independently corroborates the parallel session's finding (commit c4d10ff) that
cuvarbase's coarse fast path sits within 1–3% of *reference CPU-TLS* SDE; here we
see the same ≤3% parity against *GTLS*.

---

## 4. What GTLS does differently from cuvarbase

Both implement the same TLS math (fold → limb-darkened template → χ² → SDE), and
several high-level strategies match (Ofir period grid; hierarchical coarse-then-
refine T0; a moving-average depth estimate). The differences that matter:

**GTLS design choices**
- **Single-light-curve, per-call CuPy JIT.** `gtls(t,y).power()` compiles its
  CUDA (`cp.RawModule(...).compile()`) on *every* call — no cross-call caching,
  no batch API. Fine for one star, costly for a survey.
- **float32 throughout + `(int)` phase fold.** The fold is
  `phase = t/P − (int)(t/P)` (truncation, not floor → wrong for t<0 / raw BKJD),
  and cumulative sums / residuals accumulate in float32 over up to ~150k points.
- **cumsum moving average** for O(1) in-window depth at any duration; a global
  log-1.1 duration grid masked per-period; edge padding + an explicit
  edge-effect χ² subtraction for wrap-around transits.
- **Multi-GPU** via `subprocess` per device splitting the period grid (their 79 s
  dual-4090 number). Only the coarse scan is parallelized; refinement is 1-GPU.

**cuvarbase design choices (why it wins)**
- **Batch-native + cached kernels.** One kernel launch over *all* light curves,
  one block per (period, LC); LRU-cached compiled kernels. Amortizes launch and
  compile — the dominant survey costs.
- **Float-float (t_hi, t_lo) fold**: pure-FP32 FMA fold with 3e-8 phase error at a
  1400-d baseline, vs GTLS's float32/truncation fold (which drifts and mishandles
  negative epochs).
- **Fold-once, phase-binned scan** with integrated-template tables (S1=∫T,
  S2=∫T²): all durations and epochs come from a single fold, so finer duration
  grids are nearly free. This is the same asymptotic trick as GTLS's cumsum but
  applied inside a batched, bank-conflict-aware shared-memory kernel.
- **Clean ~linear scaling** in baseline; no period-batch/launch cliff.
- **Exact top-K refinement** kept off the SDE spectrum (SDE from the uniform
  coarse grid), so precision is refined without deflating significance.

Net: cuvarbase and GTLS share the *algorithm*; cuvarbase's *engineering*
(batching, kernel caching, FF fold, single-fold scan) is a generation ahead, and
that shows up as 1–2 orders of magnitude in wall-clock at equal detection.

---

## 5. The BLS comparison — a fairness caveat in the paper

The GTLS paper concludes "GTLS is 3.6× faster than GPU-BLS" (33.3 s vs 121.1 s at
1500 d). Its BLS is **cuvarbase** run with **Kunimoto et al. (2023, QLP DR notes
003, RNAAS 7:28)** parameters: `qmin=2e-4, qmax=0.15, dlogq=0.1, noverlap=3`.

That comparison flatters GTLS:
- `qmin=2e-4` searches transit durations down to 0.02% of the period — *sub-
  cadence* for 30-min data (~2.9 min at P=10 d). GTLS's own grid also goes that
  fine, but its cumsum moving-average makes fine durations O(1); cuvarbase's BLS
  kernel re-bins the folded curve into up to `1/qmin = 5000` phase bins per
  duration level, so its cost scales with `1/qmin`. Same qmin, wildly different
  cost. (Measured at 500 d: BLS 1.48 s at qmin=2e-4 vs **0.032 s** at the archived qmin=2e-3 config — results_cuv.json.)
- `noverlap=3` is not a power of two, so it bypasses cuvarbase's fastest *fused*
  BLS kernel (opt1) and runs 3 separate phase passes.
- The paper predates our July BLS optimizations (opt1–opt4).

With a physically sensible BLS config for 30-min data (`qmin=2e-3` ≈ one cadence,
fused `noverlap=2`), cuvarbase-BLS runs **0.01–0.73 s** across 200–3000 d — faster
than TLS (as expected: box < template) and faster than GTLS. So the paper's BLS
result is config- and version-contingent, not fundamental. The clean, meaningful
comparison is **TLS-vs-TLS** (GTLS vs cuvarbase-TLS), where cuvarbase wins outright.

---

## 6. What (if anything) to adopt from GTLS

- **Multi-GPU scale-out.** The one capability GTLS has that cuvarbase TLS lacks.
  Low priority (cuvarbase is already ~100× faster single-GPU and batches many LCs
  per launch), but a clean win for the very largest surveys — and easy, since
  cuvarbase's batch grid splits trivially across devices.
- **Richer SNR outputs** (GTLS returns snr / snrPink / snrFit / snrFitPink). Nice-
  to-have reporting, not performance.
- **Nothing algorithmic.** GTLS's core tricks (Ofir grid, cumsum depth, coarse+
  refine T0) are already present in cuvarbase, generally in a more robust form.
  Their float32/truncation fold and per-call recompile are things to *avoid*, not
  adopt.

## 7. Bottom line

At **matched search space, matched epoch density, and equal SDE**, cuvarbase's TLS
is **tens to >100× faster than GTLS on the same GPU**, and its advantage grows with
baseline because GTLS's per-call recompile and period-batch launch overhead scale
super-quadratically while cuvarbase scales linearly. cuvarbase is also numerically
more robust (FF fold vs float32/int-truncation). The GTLS paper's BLS comparison is
not cost-matched and flatters GTLS; the honest, apples-to-apples story is that
cuvarbase is the faster GPU TLS by a wide, sensitivity-neutral margin.
