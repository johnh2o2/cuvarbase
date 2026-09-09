# Independent TLS recovery and timing study

The predeclared recovery / false-positive matching criterion passes for **separated TESS with the original grid** and **dense TESS with the fine grid**. The ZTF comparison remains inconclusive under the strict two-sided false-positive margin. Every v1 setting passes the recovery-loss bound on every cadence.

These results support bounded, workload-specific comparisons of complete searches. They do not establish identical algorithms or exactly equal detection sensitivity. The study uses 4,096 calibration nulls, 2,048 independent injections and 4,096 independent test nulls per cadence, with four primary methods: **122,880 search outcomes**. The secondary BLS control adds 30,720 outcomes.

## Speed at the predeclared decision

| Cadence | Displayed v1 grid | v1 batch time / source | GTLS batch time / source | GTLS / v1 | Recovery + false-positive match |
|---|---|---|---|---|---|
| TESS 200 s | Fine | 37.4 ms | 0.446 s | 11.9× | Pass |
| Separated TESS sectors | Original | 26.4 ms | 4.64 s | 175.5× | Pass |
| ZTF g/r | Original | 67.9 ms | 10.6 s | 155.6× | Inconclusive |

The [combined timing figure](../../../docs/TRANSIT_BENCHMARKS.md) uses the fastest predeclared passing v1 setting for each TESS cadence. ZTF retains an explicitly unqualified original-grid timing. The old 93–284× TLS headline is superseded by these settings and measurements. BLS competitor measurements remain in the [earlier experiment](../transit_2026-09-08/README.md).

## Independent detection results

A detection requires the primary period to align the injected transits over the full observing baseline and a native score above the independently frozen null threshold. Native SDE values are never equated between packages. Counts pool an equal mixture of white-noise oracle SNR 6, 8, 10 and 14; [per-SNR results](recovery_by_snr.csv) show the individual strata.

| Cadence | Method / grid | Detected injections | False positives | Invalid injections / nulls |
|---|---|---|---|---|
| TESS 200 s | Original | 881/2,048 (43.02%) | 201/4,096 (4.91%) | 0 / 0 |
| TESS 200 s | Intermediate | 898/2,048 (43.85%) | 203/4,096 (4.96%) | 0 / 0 |
| TESS 200 s | Fine | 910/2,048 (44.43%) | 215/4,096 (5.25%) | 0 / 0 |
| TESS 200 s | GTLS | 886/2,048 (43.26%) | 232/4,096 (5.66%) | 0 / 0 |
| Separated TESS sectors | Original | 1137/2,048 (55.52%) | 188/4,096 (4.59%) | 0 / 0 |
| Separated TESS sectors | Intermediate | 1165/2,048 (56.88%) | 179/4,096 (4.37%) | 0 / 0 |
| Separated TESS sectors | Fine | 1172/2,048 (57.23%) | 175/4,096 (4.27%) | 0 / 0 |
| Separated TESS sectors | GTLS | 1118/2,048 (54.59%) | 195/4,096 (4.76%) | 0 / 0 |
| ZTF g/r | Original | 1620/2,048 (79.10%) | 201/4,096 (4.91%) | 0 / 0 |
| ZTF g/r | Intermediate | 1629/2,048 (79.54%) | 210/4,096 (5.13%) | 0 / 0 |
| ZTF g/r | Fine | 1626/2,048 (79.39%) | 202/4,096 (4.93%) | 0 / 0 |
| ZTF g/r | GTLS | 1546/2,048 (75.49%) | 235/4,096 (5.74%) | 12 / 32 |

All three original-grid cuvarbase false-positive rates are lower than GTLS's observed rates. The dense-TESS and ZTF original-grid comparisons miss the two-sided matching rule because their lower confidence bounds extend beyond −2 percentage points. That is uncertainty about how much *lower* cuvarbase's false-positive rate could be, not evidence of an excess of false positives or an established recovery loss. The fine grid's dense-TESS pass does not prove the coarse grid is scientifically inadequate.

GTLS has **34/4,096 calibration failures, 12/2,048 injection failures and 32/4,096 test-null failures on ZTF** after the documented memory-policy amendment; all are retained. Other primary configurations have no invalid API outcomes. Failed injections count as misses and failed null scores as minus infinity. Some valid cuvarbase TESS outputs mask individual trial periods with no admissible fit; partial-spectrum counts are in [recovery_analysis.json](recovery_analysis.json). These differ from missing candidates or failed API calls.

## Confidence bounds and decision rule

All differences below are **cuvarbase minus GTLS, in percentage points**. Require the recovery lower bound to exceed −5 and both false-positive bounds to lie inside ±2. The bounds account jointly for all 27 predeclared one-sided checks; the [methods](METHODS.md) give the construction. No threshold, sample size, setting or primary criterion was changed after viewing the new test outcomes.

| Cadence | v1 grid | Recovery difference | Recovery lower bound | False-positive difference | False-positive bounds | Joint decision |
|---|---|---|---|---|---|---|
| TESS 200 s | Original | -0.24 | -2.26 | -0.76 | [-2.26, +0.75] | Inconclusive |
| TESS 200 s | Intermediate | +0.59 | -1.57 | -0.71 | [-2.21, +0.80] | Inconclusive |
| TESS 200 s | Fine | +1.17 | -1.02 | -0.42 | [-1.97, +1.14] | Pass |
| Separated TESS sectors | Original | +0.93 | -1.06 | -0.17 | [-1.69, +1.35] | Pass |
| Separated TESS sectors | Intermediate | +2.29 | +0.35 | -0.39 | [-1.83, +1.06] | Pass |
| Separated TESS sectors | Fine | +2.64 | +0.73 | -0.49 | [-1.95, +0.97] | Pass |
| ZTF g/r | Original | +3.61 | +1.33 | -0.83 | [-2.79, +1.14] | Inconclusive |
| ZTF g/r | Intermediate | +4.05 | +1.97 | -0.61 | [-2.59, +1.37] | Inconclusive |
| ZTF g/r | Fine | +3.91 | +1.85 | -0.81 | [-2.75, +1.15] | Inconclusive |

Passing applies to the specified mixture at a nominal 5% false-alarm operating point, conditional on the injection being observable. It does not guarantee every SNR stratum, stellar geometry, observing pattern or detection threshold. Inconclusive matching is not a demonstrated performance loss. [Frozen design](design.json) · [Threshold freeze receipt](calibration-freeze.json) · [All analysis values](recovery_analysis.json).

## What finer sampling costs

Moving from original to fine adds a net **29, 35 and 6 detections out of 2,048** for dense TESS, separated TESS and ZTF respectively: about **1.4, 1.7 and 0.3 percentage points**. Individual outcomes are not monotonic with resolution. The fine setting changes bins, epoch steps and duration sampling together; it is a diagnostic reference, not exact canonical TLS.

| Cadence | Method / grid | One-source latency | Batch time / source | Batch time / original v1 | Batch repetition range |
|---|---|---|---|---|---|
| TESS 200 s | Original | 4.3 ms | 1.59 ms | 1.00× | 1.58 ms–1.83 ms |
| TESS 200 s | Intermediate | 8.15 ms | 6.09 ms | 3.83× | 5.95 ms–6.18 ms |
| TESS 200 s | Fine | 41.7 ms | 37.4 ms | 23.55× | 37.4 ms–37.7 ms |
| TESS 200 s | GTLS | 0.45 s | 0.446 s | 280.61× | 0.445 s–0.453 s |
| Separated TESS sectors | Original | 32.4 ms | 26.4 ms | 1.00× | 25.6 ms–26.7 ms |
| Separated TESS sectors | Intermediate | 0.168 s | 0.167 s | 6.33× | 0.166 s–0.168 s |
| Separated TESS sectors | Fine | 1.19 s | 1.18 s | 44.65× | 1.17 s–1.18 s |
| Separated TESS sectors | GTLS | 7.14 s | 4.64 s | 175.53× | 4.61 s–4.65 s |
| ZTF g/r | Original | 100 ms | 67.9 ms | 1.00× | 66.6 ms–68.9 ms |
| ZTF g/r | Intermediate | 0.531 s | 0.511 s | 7.53× | 0.508 s–0.515 s |
| ZTF g/r | Fine | 3.74 s | 3.72 s | 54.79× | 3.71 s–3.72 s |
| ZTF g/r | GTLS | 17.8 s | 10.6 s | 155.57× | 10.3 s–10.7 s |

The original grid uses automatic 256–1,024 bins here, epoch oversampling 4 and 16 durations; intermediate uses 4,096 / 8 / 16; fine uses 8,192 / 16 / 32. All retain top-50 fits against individual observations. [The phase-binning explanation](../../../docs/TLS_NUMERICS.md) shows the retained shape and measures compression alone at known ephemerides. Those bin-only SNR losses are separate from this complete-search result.

## Secondary box-search control

Each cell gives **detected injections / 2,048; test false-positive rate**. The BLS control receives the exact TLS-study observations and period grid, with its own independent null calibration.

| Cadence | BLS control | Original TLS | Intermediate TLS | Fine TLS |
|---|---|---|---|---|
| TESS 200 s | 765/2,048; 4.79% | 881/2,048; 4.91% | 898/2,048; 4.96% | 910/2,048; 5.25% |
| Separated TESS sectors | 1142/2,048; 4.71% | 1137/2,048; 4.59% | 1165/2,048; 4.37% | 1172/2,048; 4.27% |
| ZTF g/r | 1532/2,048; 5.10% | 1620/2,048; 4.91% | 1629/2,048; 5.13% | 1626/2,048; 4.93% |

This is one fixed BLS setting, not the strongest possible BLS configuration. BLS's ranking statistic and epoch/duration search differ from TLS, and a box's optimal width can be shorter than a transit's contact duration. The control compares complete searches; it cannot attribute a difference solely to template shape. It also does not replace the earlier, separately tuned BLS-versus-PyPI/CPU/GPU experiment. [Secondary analysis and nominal paired bounds](bls_analysis.json) · [BLS calibration freeze](bls-calibration-freeze.json).

## Timing boundary and provenance

Timing uses one otherwise idle A40 with a 7.65-CPU-equivalent allocation on a Xeon Gold 6342 host. Twenty-four isolated configuration processes run sequentially in randomized order. Each uses five synchronized, warmed repetitions on the same 16 earlier lightcurves: eight injections and eight nulls. Single latency is the mean of 16 separate calls per repetition; batch throughput is a 16-source call divided by 16. GTLS single-source latency uses one worker and is contextual where recovery was calibrated for concurrent batch execution.

The timer includes API host work, transfers, periodograms, native candidates and synchronization from prepared arrays and an explicit grid. Imports, context initialization, grid construction, simulation, disk I/O, preprocessing and vetting are excluded. Initialization and first-call times are retained separately. Disk caches may already be populated. Distributed recovery runtimes never enter these speed ratios. The fixed timing mixture is not a survey occurrence-rate model, and 16-source throughput is not a measured million-source job.

All primary periods stay unchanged across timed repetitions versus workload warmup. cuvarbase's largest native-score change is below 0.00001. Concurrent GTLS scores change by up to 0.04511 on separated TESS and 0.08654 on ZTF; its single-worker scores stay unchanged in these repetitions. The reported recovery applies to the documented execution policy, which includes GTLS's memory-dependent behavior.

Numerical source pins are cuvarbase `1032caf029570dc4841db1c594a2cbb1654e8fd8` and GTLS `74e449c325792a763dde4fbffab98039c5e8c111`. The [source receipt](source-verification.json) verifies every installed numerical file against its Git archive. GTLS numerical code is unmodified; the ZTF client releases unused CuPy blocks and limits concurrency to two after preflight memory failures. [Cross-node probes](cross-node-probes.json) record small GTLS score changes from memory-dependent chunking. The independent study uses the same frozen numerical versions throughout.

[Timing records](timing) · [Timing analysis](timing_analysis.json) · [Machine-readable timing table](timing_analysis.csv) · [Methods and limitations](METHODS.md).

## Evidence and reproduction

The [compact evidence](evidence) retains all scalar outcomes, truth, paired input hashes, output hashes and installed-source maps. Its receipt records original verification of every prepared input array and retained sampled spectrum, plus exact reconstruction of the full scalar summaries. Full observations and sampled periodograms remain in the larger measurement archive; unretained spectra were hashed during execution and discarded. A compact checkout can repeat summary analysis, not verify omitted bytes.

The [execution-source archive](execution-harness.json.gz) preserves measured harness revisions by SHA256; [maintained tools and commands](../../tls_sensitivity/README.md) provide portable regeneration and analysis. The three cadence files and the seeds specify new input generation, subject to recorded software versions and floating-point reproducibility. [Analysis verification](analysis-verification.json) records exact agreement between the original and compact analyses.

[HATPI cost pilot](HATPI.md) · [Rental and termination ledger](rental-ledger.json) · [Publication checks](validation.json) · [File hashes](SHA256SUMS.json).

All 37 study/preflight pods are terminated and confirmed absent. Estimated rental is **$35.77 for this study**, or **$43.80 including the earlier campaigns**. New-study container storage adds about **$0.81** at the documented rate; earlier storage is additional. These are elapsed-time estimates, not an invoice, and remain within the original $50 allowance.
