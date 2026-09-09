# Independent TLS study: methods and scope

The question is whether cuvarbase can search faster while retaining recovery within a stated tolerance at independently calibrated false-alarm thresholds. It is a comparison of complete numerical searches, not a claim that cuvarbase and GTLS implement identical computations. [design.json](design.json) records the frozen settings, seeds, counts, decision rule and amendments.

## Observations and injections

| Cadence example | Original observations | Baseline | Common TLS trial periods | Period range |
|---|---:|---:|---:|---:|
| TESS sector 67, 200-second exposures | 9,736 | 25.76 days | 3,084 | 0.6003–12.8784 days |
| TESS sectors 1 and 27, 30/10-minute exposures | 4,295 | 734.85 days | 99,043 | 0.6000–27.4579 days |
| ZTF g/r, sparse seasonal sampling | 1,317 | 2,743.77 days | 312,064 | 0.6000–10 days |

The [cadence manifest](cadences/manifest.json) identifies the original files and their hashes in the [earlier evidence archive](../transit_2026-09-08/ARCHIVE.md). These are three observed cadence examples, including a deliberately separated pair of TESS sectors. They are not random samples of their surveys. The long TESS gap matters computationally: maintaining transit alignment over a longer baseline requires a finer period grid.

Each cadence has 4,096 calibration nulls, 2,048 independent injections and 4,096 independent test nulls. Injections are balanced at white-noise oracle SNR 6, 8, 10 and 14, with 512 at each level. This SNR describes the injected signal and white uncertainties; it is neither native SDE nor a correlated-noise significance estimate.

Fluxes are newly simulated on the retained observing times. Each case independently drops 0–3% of observations. Periods are log-uniform from 0.8 days to the smaller of 12 days and 0.8 times the search maximum. Radius ratios are 0.025, 0.05 or 0.10; impact parameters are uniform from 0 to 0.85. The exposure-integrated batman model uses seven sub-exposures, solar stellar density, circular orbits and quadratic limb darkening `[0.4804, 0.1867]`. Draws must contain at least five in-transit observations and two observed events; rejection counts are retained.

Noise combines heteroscedastic independent Gaussian errors and an Ornstein–Uhlenbeck residual with amplitude 0.25 times the median uncertainty and correlation time 0.15 days for TESS or one day for ZTF. The noise-scale mixture is also used for nulls. Known unit band baselines and achromatic transits are supplied. This tests controlled recovery conditional on observability, not real-flux survey completeness, chromatic modeling or a complete QLP pipeline.

## Search definitions and execution

Numerical sources are cuvarbase [`1032caf`](https://github.com/johnh2o2/cuvarbase/tree/1032caf029570dc4841db1c594a2cbb1654e8fd8) and public GTLS [`74e449c`](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111). [Source verification](source-verification.json) checks all 69 installed cuvarbase files and 19 GTLS files against their Git archives. The [search dependency pins](requirements-search.txt) describe the Python 3.11 / CUDA 12.4 environment; [analysis versions](analysis-environment.json) are recorded separately. GTLS's source installation omitted its CUDA resource files; the pinned, unmodified `.cu` files were copied into the installed package. UTF-8 locale and the CUDA library path were set explicitly.

All methods receive byte-identical observations and trial periods within a case. cuvarbase searches durations 0.5–2 times the central circular duration, with these predeclared alternatives:

| Configuration ID | Display name | Phase bins | Epoch oversampling | Durations |
|---|---|---:|---:|---:|
| `v1_original` | Original | Automatic, 256–1,024 here | 4 | 16 |
| `v1_resolved` | Intermediate | 4,096 | 8 | 16 |
| `v1_fine` | Fine reference | 8,192 | 16 | 32 |

All retain top-50 observation-level refinement. The fine reference is not assumed exact. [The numerical explanation](../../../docs/TLS_NUMERICS.md) distinguishes compression, epoch sampling, duration sampling and refinement.

GTLS uses public fast mode, `duration_grid_step=1.1`, `T0_fit_margin=0.125`, stellar-radius bounds 0.5–2 solar radii and a fixed solar mass. These approximately align the physical search window, but the actual duration/epoch grids and objectives differ. Its selected recovery schedules use one worker for dense TESS and two for separated TESS and ZTF, on one A40 per shard.

Old-input probes exposed GPU memory failures with four concurrent ZTF GTLS calls. The corrected policy uses two workers and releases unused CuPy memory-pool blocks before and after successful calls through the [public CuPy API](https://docs.cupy.dev/en/stable/user_guide/memory.html). This changes client memory management; GTLS's numerical source is unmodified. All initial four-worker ZTF calibration results were superseded and recomputed. This amendment preceded generation or inspection of the new held-out cohorts. Remaining failures are retained, not removed.

Four-case old-input probes were repeated on 20 additional GPUs. [Their records](probe-records.json.gz) and [identical input files](probes) support the [cross-node comparison](cross-node-probes.json): primary periods agree; dense-TESS and ZTF scores agree exactly in those probes; separated-TESS scores differ by at most 0.00941 native SDE. GTLS's memory-dependent chunking means this does not guarantee universal bitwise repeatability.

## Calibration and statistical decision

Each method/cadence has its own threshold: the higher empirical 95th percentile of its 4,096 calibration-null scores, with strict exceedance. [The freeze receipt](calibration-freeze.json) records the threshold file's hash before held-out outcomes were examined. Native SDE values are not equated across algorithms.

A detection must exceed its threshold and return a primary period whose accumulated phase drift over the full cadence baseline is at most half the injected duration. Half/double/third-period aliases are recorded separately. Failed injections count as misses; failed null scores are minus infinity. Partial spectra and all execution failures are reported.

[Execution outcomes](execution-outcomes.json) count invalid candidates and masked trial periods separately for calibration, injections and test nulls. The original ZTF attempts superseded by the pre-test memory-policy amendment are excluded from the frozen cohorts.

For each of nine v1-setting/cadence comparisons with GTLS, require a lower confidence bound on the recovery difference greater than −5 percentage points, and both false-positive difference bounds inside ±2 points. Paired discordant-cell Clopper–Pearson bounds use `alpha = 0.05/27` per primary one-sided difference bound, divided between its two cell bounds. This accounts for the nine recovery lower bounds and 18 false-positive bounds together. Marginal Wilson intervals, per-SNR counts and nominal diagnostic contrasts are reported separately.

The target population is the equally weighted four-SNR mixture. Passing means supported noninferiority within the stated margins on these cases at the nominal 5% false-alarm operating point. It does not establish exact equality, a per-SNR guarantee, performance at every detection threshold or unconditional survey sensitivity. The main figure may select the fastest predeclared setting that passes for each cadence; all three settings remain reported. If none passes, the timing remains explicitly unqualified.

The secondary BLS control uses the same observations and TLS-restricted period grid, `noverlap=4`, `qmin_fac=0.5`, `qmax_fac=2` and `dlogq=0.1`, with its own null calibration. It compares complete searches, including different ranking statistics; it does not isolate template shape or replace the earlier BLS competitor benchmark. A box's optimal width can be shorter than a transit's contact duration; the earlier BLS benchmark includes separately tuned duration bounds. Secondary contrasts are nominal and outside the primary decision family.

## Timing and retained evidence

Final timings run sequentially on one otherwise idle A40, with randomized configuration order, explicit synchronization, workload warmup and five repetitions. The fixed earlier-data subset contains eight injections, two per SNR, and eight nulls. Single-source latency averages 16 separate API calls per repetition; batch throughput divides a 16-source call by 16. GTLS single-source latency uses one worker and is contextual when recovery was calibrated for a concurrent batch schedule.

Timed primary periods remain unchanged versus workload warmup. The largest score change is below 0.00001 for cuvarbase, 0.04511 for concurrent GTLS on separated TESS and 0.08654 for concurrent GTLS on ZTF. Single-worker GTLS scores are unchanged in these repetitions. The [timing analysis](timing_analysis.json) reports each configuration separately.

The boundary starts at prepared host observations and an explicit grid and ends with host periodograms and native candidates/scores. It includes transfers and API postprocessing. Imports, context initialization, grid creation, synthetic data generation, disk I/O, preprocessing and vetting are excluded; initialization and first API calls are retained separately. Disk caches may already be populated. The injection/null mixture is a controlled timing workload, not a survey occurrence-rate model.

Distributed recovery timings are diagnostic only. They never enter the headline speed ratios. Cost projections use measured throughput and the recorded $0.49/hour A40 bundle; they are search-stage projections, not measured million-source runs or a survey's full bill.

The compact evidence deduplicates input truth and source maps while preserving all scalar search records and hashes. The exporter verifies all prepared input arrays and the retained spectra for the first four cases per shard against the larger measurement archive. Other spectra were hashed during execution and discarded. Published receipts distinguish those original byte checks from summary-only reanalysis; omitted arrays cannot be re-verified from the compact checkout alone.

[The execution-source archive](execution-harness.json.gz) stores exact UTF-8 harness sources indexed by SHA256. It includes both adapter revisions and both runner revisions found in the records. The adapter added opt-in cache release; the runner later corrected a local module-lookup collision and supplied BLS's missing validity flag. Primary GPU workers always loaded the intended generator, and TLS already supplied that flag. Numerical package sources stayed fixed. The maintained tools provide the portable reproduction interface; the archive preserves the bytes actually executed.
