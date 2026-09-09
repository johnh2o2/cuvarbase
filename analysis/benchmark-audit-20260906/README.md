# Benchmark provenance and fairness audit

Audit target: the frozen v1.0 source at `1032caf029570dc4841db1c594a2cbb1654e8fd8`
(tree `b023c3e8d163010dbae2fc0b7cd5204ca04384d1`), reviewed from `de0037d`.
Audit started 6 September 2026. Archived July/February results and new measurements
are separate evidence sets.

**The historical headline ratios mostly have traceable arithmetic. They do not
establish the advertised apples-to-apples performance or equivalent sensitivity
of the final v1.0 release. The release claims need qualification.**

The most consequential distinction is between matching some input parameters,
matching the actual search, and matching detection completeness at a specified
false-positive rate. Those are three different tests. The old TLS comparison
passes parts of the first; it does not establish the latter two.

## Deliverables

- [Claim-by-claim findings](#findings), below.
- [Historical TLS numbers and grid-limit calculations](historical_tls.csv).
- [Historical BLS CPU ratios](historical_bls_cpu.csv) and
  [BLS optimization ratios](historical_bls_optimization.csv).
- [Exact version/source pins](sources/pins.json) and
  [hashes of the historical evidence](historical_sources_sha256.json).
- [Fresh measurement tables](NEW_RESULTS.md), [selected timing records](selected_timings.csv),
  and [numerical validation](validation.json).
- [Four-way LS comparison, float64](figures/ls_comparison_float64.png)
  ([PDF](figures/ls_comparison_float64.pdf), [SVG](figures/ls_comparison_float64.svg)).
- [LS with default cuvarbase precision](figures/ls_comparison_default.png) and
  [the effect of shared timestamps](figures/ls_shared_times.png).
- [TLS single versus batch timing](figures/tls_comparison.png),
  [fresh TLS baseline scaling](figures/tls_baseline_scaling.png), and
  [the sensitivity diagnostic](figures/tls_sensitivity_diagnostic.png).
- [Historical TLS timing and grid-limit plot](figures/historical_tls_audit.png).
- [Reproduction instructions and runners](../../scripts/benchmark_audit/README.md).

The complete four-role comparison is LS, which exists in both cuvarbase versions.
The TLS chart explicitly marks PyPI 0.2.5 as unavailable. Every figure has PNG,
SVG, and PDF exports; none substitutes a different statistic for missing TLS.

## What the new measurements establish

The frozen release has a substantial TLS timing advantage on these examples.
The advantage is workload dependent, and these measurements still do not
establish equal detection sensitivity. The current GTLS source is 0.5.1.

| Workload | GTLS ms/LC | v1.0 wide-window ms/LC | GTLS/v1.0 | Exact periods: GTLS / v1.0 |
|---|---:|---:|---:|---|
| 27 days, one LC | 332.710 | 14.560 | 22.85× | 1/1 / 1/1 |
| 27 days, 16-LC workload | 357.723 | 5.822 | 61.45× | 4/16 / 5/16 |
| 200 days, one LC | 3,408.473 | 83.375 | 40.88× | 1/1 / 1/1 |
| 1,500 days, one LC | 153,192.137 | 942.809 | 162.48× | 1/1 / 1/1 |

These are medians of three warm API calls on the same A40. Each comparison
uses identical input bytes and trial periods, with the remaining search
differences described below. “Exact” means within 0.2% in period. The 16-LC row
is measured total workload time divided by 16; it is not single-call latency.
The 27-day timing inputs have low aggregate signal strength, and the table
deliberately retains unsuccessful period recoveries in its throughput accounting.

PyPI GTLS 0.4.4 was also measured. At 200 days it took 3,034.942 ms, faster than
current upstream on that case. On the 27-day inputs its native period/SDE
diagnostics were non-finite; those results are retained but excluded from
comparative speedups. This is an observation about these inputs and this
environment, not a diagnosis of the underlying GTLS issue.

For **float64 LS on 32 distinct lightcurves**, v1.0 was **2.72–5.27× faster than
the best tested CPU configuration** across the four array shapes. On the
TESS-size and Kepler-size batches it was **7.99× and 4.06× faster than actual
PyPI 0.2.5**, respectively. Several other legacy results exceed the sampled
accuracy tolerance, so no upgrade speedup is claimed for those cells.

Single-LC and shared-time results change the ranking. CPU nifty-ls wins the
small and TESS-size single-LC cases. GPU nifty-ls wins the ZTF-size and
Kepler-size single-LC cases. With **32 shared-time TESS-size lightcurves**,
nifty-ls GPU took **0.421 ms/LC**, versus **0.910 ms/LC** for float64 v1.0.
A general claim that v1.0 always beats the best GPU competitor is unsupported.

The LS quality screen checks sampled normalized powers against direct float64
GLS, with absolute tolerance 0.001 and peak agreement within one frequency bin.
Of 97 configurations, 95 completed; **84 passed and 11 exceeded that screen**.
Two large Astropy batch jobs timed out at the controller level. All compared
input hashes match. This screen is an exploratory approximation check, not a
guarantee of calibrated false-alarm probabilities. In particular, the default
float32 v1.0 ZTF-size batch exceeds the screen (maximum sampled power error
0.001527); the float64 result passes. `validation.json` retains all errors,
including those unfavorable to the new release. Missing LS bars are excluded
accuracy results, not zero runtimes or missing implementations.

The separate 64-LC diagnostic recovered exact periods for **15/48 CPU TLS,
13/48 GTLS, 12/48 wide-window v1.0, and 14/48 default-window v1.0**. These counts
are too small to establish a sensitivity ranking or equivalence. The 16 nulls
also do not calibrate a 1% false-positive rate. Recovery depends on the tolerance;
the raw periods and injected values are saved so it can be evaluated differently.

All 432 transferred evidence files passed checksum verification. Installed
v1.0 and GTLS runtime sources match the pinned archives, and installed PyPI
package files match the archived wheels. The disposable GPU was terminated
after verification; estimated compute cost was **$0.51**, not an invoiced amount
([cost record](sources/compute-cost.json)). These are observations from one
cloud host, with a fixed run order and a small number of repetitions; the ranges
shown are observed variation, not confidence intervals or cross-host validation.

## Findings

| Claim | Evidence and verdict | Appropriate use |
|---|---|---|
| TLS is 30–171× faster than GTLS at matched settings and equal significance | The archived ratios are **30.02, 55.33, 85.88, 123.75, 171.02** at 200/500/1000/1500/2000 days. However, the effective grids, templates, refinement, and statistical validation differ. | Historical warm single-lightcurve timing observations, with the limitations below. Do not describe them as proven equivalent sensitivity. |
| Same-GPU GTLS provenance | The cuvarbase JSON names an A5000 and CuPy 13.6.0, but both `cuvarbase` and `gputls` versions are `?`. The two GTLS JSONs lack their final `env` sections. No exact implementation commit, input-array hashes, GPU UUID, or CPU allocation was saved. | The same-host account is plausible and documented in prose; the raw files alone cannot independently verify all of it. |
| Eight epochs per duration on both TLS implementations | cuvarbase's coarse phase bins stop at 8,192; its epoch count separately stops at 20,000. GTLS samples indices in a sorted folded lightcurve, with integer-rounded skip counts. | Nominal oversampling is similar; the effective grids are not identical. |
| Matched duration search | cuvarbase uses a per-period log grid; GTLS uses a global grid of integer sample widths masked per period. The copied window omits GTLS's upper-bound factor `1 + P / baseline`, plus its floor/ceil rounding. | Similar physical bounds, not the same set of templates/durations. |
| Equal sensitivity / 100% recovery / SDE within 1–3% | One fixed, central, circular, 8.13-day transit per baseline; identical noise type, limb darkening, phase, and depth. No null calibration, weak-signal population, impact-parameter sweep, or realistic gaps. Search templates were explicitly **not** matched. | Recovery of those particular strong injections. It does not bound completeness loss at fixed false-alarm rate. |
| Realistic GTLS comparison noise | The prose says 110–400 ppm-class noise. The input code **and raw metadata use 0.004 = 4,000 ppm**, with nominal depth 4,000 ppm. Aggregate nominal SNR rises from 13.7 to 43.7 over the headline range. | Correct the input description. The strong aggregate detections are unsurprising. |
| Old SDE evidence applies to v1.0 | The archived re-scorer uses `1 - chi2/max(chi2)`. v1.0 uses `min(chi2)/chi2`. Raw chi-square spectra were not saved in those benchmark JSONs. | The old SDEs cannot be converted reliably into current SDEs from the saved scalar values. |
| GTLS recompiles on every call and this explains the speedup | GTLS creates `RawModule` objects and calls `compile()`, but CuPy caches compiled binaries. A tiny-grid full search is not an isolated compilation measurement. | Attribute the observed wall time to the API path measured. A compilation/launch bottleneck claim needs profiling. |
| Paper Figure 7 definitely used skip=8; the A5000-vs-4090 cross-check is decisive | The skip setting is inferred from runtime, not established by an archived paper runner. The paper and repository runs have different hardware, software, input details, and timing boundaries. | Treat the paper numbers as external context, not a controlled speedup comparison. |
| GTLS uses float32 throughout | The currently available upstream source uses `double` for input times, trial periods, and folded phases. | This description is false for the current comparator; exact archived-source identity is missing. |
| TLS searches one TESS lightcurve in ~1.2 ms | The authoritative A5000 JSON divides a **100-lightcurve batch** by 100; `n_iter=1`. It is a warm throughput result. | Label it amortized batch time per lightcurve, not single-call latency. |
| Final release timings are unaffected by intervening fixes | Since the July campaign, TLS statistics changed, per-lightcurve statistics were made sequential (`9f1540e`), input validation was added, and other API changes landed. | Retest the frozen code. Do not assume timing invariance from a correctness gate. |
| Standard BLS is 257–354× faster than Astropy | Ratios recompute, but Astropy scans five fixed **absolute durations** of 0.01–0.2 days; GPU BLS uses default **fractional durations**, about 0.01–0.5 of each period. Phase coverage/density is also different. | These are different searches. Withdraw the unqualified apples-to-apples interpretation; the direction of bias is not uniform across periods. |
| Survey LS beats the fastest CPU by 1.5–12.6×, or >15–27× | The old benchmark loops over CPU calls with default FINUFFT threading. It does not evaluate CPU worker/thread tuning, shared-epoch batching, or nifty-ls's own GPU backend. CPU quota/model/thread settings are missing. | A measured comparison with that particular CPU invocation, not a general best-competitor result. |
| LS timeout is evidence of a lower bound | The archive stores `None` for both timeout and exception. The helper checks a time limit after calls finish and does not retain the elapsed warmup. | The raw `None` cells alone cannot establish the published timeout lower bounds. |
| Slow Astropy LS baselines represent its fast method | Some runners construct frequencies in float32, cast them to float64, then use `method='auto'`. That rounded grid fails Astropy's regular-grid test and selects the direct `cython` method. The local dispatch check reproduces this at 5K and 50K frequencies; an original float64 grid selects `fast`. | Do not present this as the fastest Astropy implementation. The new runner uses an original float64 grid and explicitly selects its fast method. This issue is separate from the nifty-ls headline comparison. |
| Previous PyPI cuvarbase is 0.2.6 | **PyPI's latest release is 0.2.5**, dated 23 October 2023. The old comparison uses the unpublished 0.2.6 tag. | Name the actual version. Use a real PyPI 0.2.5 install for an upgrade comparison. |
| 34× per-lightcurve loop, 10× for 100 stars | 34.16× from saved medians; the 100-star totals are an extrapolation of a **20-star** run. Old/new stacks also differ in NumPy and PyCUDA. Precompiled-handle paths already existed in the old version. | Describe the API usage and stack, not a GPU-kernel speedup. Label 100-star totals as extrapolated. |
| Survey BLS improved 2.0–12.7× | Broadly traceable as a July development optimization. TESS is 12.68× against the initial environment but **5.80×** against the thread-pinned baseline. These JSONs record `git_sha: unknown`. | Useful historical engineering evidence, not a measured 0.2.5-to-frozen-v1.0 ratio. |
| fBLS ~6 s for 65K points / 100K periods, “their table 1” | Table 1 of the cited paper contains planet candidates, not runtime data. Runtime measurements are in Figure 4 under another grid/phase-resolution protocol. No matched fBLS run is archived here. | Remove the unsupported numeric comparison. fBLS remains a relevant CPU candidate to evaluate. |
| Keplerian grids give fewer frequencies “with no loss” | The grid reduction is real for the chosen stellar/duration assumptions. This restricts the search family and requires a corresponding injection/recovery study to quantify completeness. | State the physical assumptions and comparison grid. Avoid a universal no-loss claim. |
| Entire surveys cost cents/dollars | These are arithmetic projections of warm search throughput at historical pod rates, without data access, detrending, calibration, candidate vetting, or measured sustained survey execution. | Label compute-only projections, date the rate, specify the search and batching, and avoid implying a measured total survey bill. |

### TLS: what was actually held constant?

The period array was supplied explicitly to both methods, as were the same
synthetic observations. Synchronization surrounds wall-clock timing. Both sides
receive a warmup, and the ratio uses full API time rather than subtracting the
tiny-grid probe. Those are useful controls.

However, a full GTLS call also returns diagnostics and performs refinement
differently from cuvarbase (`refine_top_k=50`). Matching a period grid and two
nominal knobs does not make the objectives or computational work identical.
The documented template mismatch is not isolated by an experiment that changes
only the template, so the observed total SDE difference cannot be causally
assigned to it.

The following limits are computed directly from the archived grid generator,
copied GTLS duration bounds, and the shipping cuvarbase constants. “Affected
periods” means the **narrow edge** of the duration window is affected; it does
not mean every duration, or every planet at that period, is missed.

| Baseline | Periods with narrow edge under-resolved by bins | Periods with narrow edge hitting the epoch cap | Maximum bin-width / requested epoch spacing |
|---:|---:|---:|---:|
| 200 d | 9.60% | 0.00% | 2.05 |
| 500 d | 14.60% | 3.77% | 3.78 |
| 1,000 d | 17.24% | 6.75% | 5.99 |
| 1,500 d | 18.47% | 8.13% | 7.85 |
| 2,000 d | 19.23% | 8.98% | 9.52 |

The old runner starts with `warnings.filterwarnings("ignore")`, hiding the
library's warning about this bin limit. Refining top-ranked candidates improves
their parameters, but cannot by itself prove that a weak signal lost in the
coarse scan will make the candidate list.

The old `recovered()` helper accepts 1/3, 1/2, 1, 2, or 3 times the injected
period within 2%. The archived headline examples do in fact return the true
period closely; nevertheless, future recovery reports should distinguish exact
period recovery, harmonic recovery, and merely significant peaks.

Sources: [historical runner](../../scripts/gtls_benchmark/gtls_apples_bench.py),
[data/SDE helper](../../scripts/gtls_benchmark/bench_core.py),
[shipping TLS host code](../../cuvarbase/tls.py),
[shipping TLS kernel](../../cuvarbase/kernels/tls_fast.cu),
[historical writeup](../../docs/GTLS_COMPARISON.md), and
[raw campaign](../../benchmarks/results/gtls_comparison_jul2026/).

### Competitors and the meaning of “best”

For LS, the new comparison evaluates nifty-ls/FINUFFT on CPU, nifty-ls/cuFINUFFT
on GPU, and Astropy's fast CPU method. CPU configurations cover 1/4/8 internal
threads and 4/8 concurrent single-threaded workers where appropriate. Shared
timestamps permit a different, explicitly separate native batch workload.
“Best” in a figure means the fastest **successfully measured and validated
candidate/configuration in this campaign**, not a proof of global optimality.
Approximation parameters otherwise use library defaults. Matching float64 mode
does not make approximation errors identical, and this campaign does not trace
each implementation's best time at every allowed error tolerance. CPU
thread/worker choices and the explicit precision/batching choices are the
configuration tuning performed here; cuvarbase's distinct-time batch size is 8.

For TLS, the reference CPU implementation is `transitleastsquares` 1.32. The GPU
competitors are GTLS from PyPI (0.4.4) and the upstream source (0.5.1 at
`74e449c`). PyPI 0.2.5 has no TLS implementation and cannot appear as a TLS timing.
Plotting its BLS timing as TLS would compare different statistical models.

CETRA is a relevant GPU competitor for the broader task of finding transits,
even though its search algorithm/statistic differs. Likewise fBLS is a CPU BLS
competitor. A claim about the fastest *transit detector* requires evaluating
these at calibrated completeness/false-positive rates; this audit's TLS timing
plot does not establish that broader ranking. We did not locate a runnable fBLS
release in the cited paper/poster links and do not substitute a literature time.

Primary sources checked:

- [cuvarbase PyPI release](https://pypi.org/project/cuvarbase/0.2.5/),
  [GTLS PyPI release](https://pypi.org/project/gputls/0.4.4/), and
  [GTLS upstream source at the measured commit](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111).
- [nifty-ls source and documented CPU/GPU/batch support](https://github.com/flatironinstitute/nifty-ls).
- [CuPy RawModule documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html)
  explicitly describes binary caching.
- [GTLS paper, sections 2.5 and 3.4](https://arxiv.org/html/2607.00348v1):
  the published 33.3/138-second GTLS points are on a 4090; the CPU times use a
  Ryzen 7950X. They are external context, not new observations on our host.
- [fBLS paper](https://arxiv.org/pdf/2204.02398), Figure 4 and Table 1;
  [the cited Zenodo record](https://zenodo.org/records/5559886) is a poster.
- [CETRA paper](https://arxiv.org/abs/2503.20875), a distinct, publicly available
  GPU transit detector that would matter to a broader ranking.

## New measurement protocol

One disposable NVIDIA A40 (48 GB), CUDA toolkit 12.4, driver 580.159.03,
Intel Xeon Gold 6342 host. The container exposes 96 logical CPUs but its CPU
quota is **7.65 core equivalents**, recorded in `cpu.max`. Eight threads/workers
is therefore the upper CPU tuning point, not 96. No benchmark methods run
concurrently with one another.

The final source is installed from a `git archive` of `1032caf`; PyPI 0.2.5 runs
in a separate NumPy 1.23.5 / PyCUDA 2022.2.2 environment so its LS path can run.
No legacy algorithm is patched. Modern packages and GTLS source pins are recorded
along with environment freezes. This comparison necessarily includes the
supported dependency stacks; it does not isolate every dependency's effect.

LS uses 1,000×5,000, TESS-size 20,000×13,500, ZTF-size 150×365,000, and
Kepler-size 65,000×730,000 observation/frequency counts. The timestamps are
synthetic and irregular; these names describe array sizes, not actual survey
data or a full survey pipeline. Single-LC and 32-LC workloads share the first
lightcurve. A separate 32-LC TESS-size workload has common observation times.

The first LS pilot exposed different input hashes between the NumPy versions,
despite identical random seeds. Those pilot measurements are retained separately
and excluded from the final plots. The final campaign loads arrays generated
once in the modern environment, with identical-byte input hashes required across
all methods in each comparison.

All final timing uses wall clocks from host arrays to host output spectra,
including API preprocessing, allocation and transfers. Data generation, imports,
context initialization, and external validation are outside the timed interval.
The first API call is saved separately; an additional warmup is discarded, then
five LS or three TLS samples are saved. Error bars show the observed min/max,
not confidence intervals. These are **warm API timings**, not fresh-process
cold-start measurements. The historical cold-start experiment is kept separate.

TLS inputs use 30-minute exposures with finite-exposure transit integration.
The wide-window comparison shares an explicit Ofir period grid and similar
physical duration bounds and nominal epoch oversampling. It deliberately does
not call that an exact-grid comparison. CPU TLS lacks an explicit-period API,
so the adapter replaces only its period-grid factory and checks returned periods;
its search kernel is unmodified. Default-window v1.0 results are a separate
configuration, not used as the denominator of the wide-window speedup.

A small 64-LC sensitivity experiment contains 16 nulls and 48 injections, with
varied phases, periods, impact parameters, depths, missing observations,
heteroscedastic uncertainties, finite exposures, and white/correlated noise.
It distinguishes native SDE, one shared current-definition score, exact period
recovery, and aliases, and saves full chi-square spectra. With only 16 nulls it
cannot validate a 1% false-positive rate or establish percent-level sensitivity
equivalence. It is an adversarial diagnostic, not a production completeness study.

The appropriate publication-quality follow-up is a larger, preregistered
injection/null study on real survey noise, stratified by period/duration/SNR and
stellar assumptions, comparing throughput at a fixed false-positive rate and
fixed parameter-recovery tolerances. Include false candidates and refinement
costs in the throughput accounting.

## Release-claim changes this audit supports

1. In [README.md](../../README.md) and
   [the release notes](../../docs/RELEASE_NOTES_v1.0.0.md), replace the unqualified
   BLS/GTLS/fastest-CPU headlines with dated, workload-specific observations and
   links to their actual measurement records. The new figures give an honest
   single-call/batch comparison for the frozen release.
2. In [GTLS_COMPARISON.md](../../docs/GTLS_COMPARISON.md), correct the noise level,
   effective-grid mismatch, compilation-cache explanation, and current precision
   description. Keep the July timing data as historical evidence; remove the
   inference that close SDE values establish equal sensitivity.
3. In [BENCHMARK_RESULTS.md](../../docs/BENCHMARK_RESULTS.md), distinguish 0.2.6
   development-tag results from the actual 0.2.5 PyPI upgrade comparison. Remove
   unsupported timeout lower bounds and the incorrectly attributed fBLS number.
4. In [TLS_COST_ANALYSIS.md](../../docs/TLS_COST_ANALYSIS.md), distinguish single
   lightcurve latency from amortized batch time and label all whole-survey costs
   as compute-only projections. A measured 16- or 32-LC workload is not evidence
   that an entire survey achieves the same sustained rate.

This audit adds evidence and reviewable recommendations without changing the
frozen numerical implementation or rewriting the existing release documents.

Git includes the reports, figures, measurement records and verification receipts. Full input/output arrays for this earlier campaign remain in the local archive; see [archive contents](ARCHIVE.md).
