# Observation-level TLS: validation and timing

This is the evidence for cuvarbase v1's new default TLS engine. It evaluates individual observations with pinned GTLS's templates, sample windows and full refinement. The earlier phase-binned engine and its larger historical speed ratios are separate studies.

**The independent 160-case study and separately sealed 24-case null supplement match corrected GTLS exactly.** All APIs succeeded. Untouched GTLS has nine cases with different final spectra and SDE values caused by its invalid-candidate mask defect; all 184 selected periods and the studies' descriptive SDE > 8 recovery/null decisions still agree. The [benchmark report](../../../docs/TRANSIT_BENCHMARKS.md) explains the numerical comparison and speed results; [TLS numerics](../../../docs/TLS_NUMERICS.md) explains the algorithm.

**TLS is 3.6–4.6× faster for one lightcurve and 1.5–2.4× faster per lightcurve in 16-source batches** than the qualifying GTLS comparisons. All times below are median seconds per lightcurve, including each API's normal output work.

| Cadence | Single v1 / GTLS | Single speedup | Batch v1 / GTLS | Batch speedup | GTLS batch workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TESS: dense sector | 0.149 / 0.533 s | 3.58× | 0.161 / 0.319 s | 1.98× | 4 |
| TESS: separated sectors | 1.554 / 6.037 s | 3.88× | 1.534 / 3.684 s | 2.40× | 2 |
| ZTF g/r | 3.231 / 14.899 s | 4.61× | 3.116 / 4.545 s | 1.46× | 4 |

The original campaign **failed its all-configurations gate** because four-worker GTLS exhausted GPU memory during the separated-TESS warmup, before any measured repetitions. This report uses a separate, explicitly **post hoc assessment of the 11 completed configurations**, retaining the original numerical checks and fastest-eligible-pool rule. The original failure is preserved; failed or incomplete calls never supply a speed denominator. [Original gate](timing/acceptance.json) · [Reporting assessment](reporting_acceptance.json).

| Evidence | Contents |
| --- | --- |
| [Main validation](validation/README.md) | 160 independent inputs across eight TESS/ZTF regimes; original acceptance, per-regime outcomes, complete comparisons and output hashes |
| [Supplementary nulls](supplement/README.md) | 24 separately sealed null inputs, extending the three timing cohorts to 16 each |
| [Numerical stress tests](stress/README.md) | Selected edge cases and annual-period thin transits, with shared misses and input-handling differences retained |
| [Timing records](timing/README.md) | Five single calls, three 16-source batch repetitions, GTLS pools of one/two/four workers, and separate common-search components |
| [Exact inputs](inputs/README.md) | A portable 209-case array bank, original metadata and byte-identity verification |
| [Executed sources](sources/README.md) | Original scientific and timing source snapshots, seals and production-test source identities |
| [Rental ledger](rental-ledger.json) | Actual rental intervals, storage estimates, interrupted-run accounting and verified termination |

The single timing source is selected by its declared input identity and paired API success. It is never selected for its elapsed time, recovery or SNR. Each measured search included in the report must reproduce its own frozen scientific outputs, and complete returned-object hashes must repeat. The 184-case sensitivity study uses the single-worker reference; GTLS pools qualify on the 16-source timing cohort, which is not a separate pooled injection/recovery study. Failures and incomplete calls are excluded from successful timing denominators and retained in the evidence. Common-search components are measured separately; GTLS's extra SNR/pink-noise diagnostics are not attributed to a slower fitting kernel.

The main validation ran on an A40. Supplementary validation and TLS timing ran on an RTX A6000, selected for availability before timing. Within the TLS comparison, both implementations use the same A6000, inputs, period grids and CPU allocation. Environment receipts record the allocations, package pins and numerical-library thread settings. The batch comparison uses the fastest eligible tested GTLS pool. Repetitions on fixed inputs do not estimate runtime variation across an entire source population; cost projections retain that limit.

A fixed SDE of 8 is not a calibrated false-alarm threshold across these cadences. Whole-spectrum equivalence is the primary accuracy evidence; identical decisions at that descriptive threshold are an additional check. Small cohorts do not establish a one- or two-percentage-point completeness margin. Both engines retain GTLS's sample-window approximation, and neither can recover unobserved transits or guarantee detection through arbitrary noise.

The exact production sources passed [265 TLS tests on an A40](../../../docs/validation/tls-default-20260910/README.md). The pinned native reference is [GTLS 74e449c](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111); its MIT notices are retained. The host-mask correction and literal-native comparisons are explicit in the [implementation comparison](../../../docs/GTLS_COMPARISON.md).

Input and result collection interruptions are documented in the collection receipts. The completed main acceptance is original; it was recovered from complete members of a truncated download and was not reconstructed. Reexecuting the same inputs does not create additional independent samples. Large output arrays remain outside this repository, with their numerical identities, retained/removed/missing status and reproduction route preserved.

The [figure provenance](figure-provenance.json) records the plotted data, renderer and output hashes.

To reproduce the study, start with the [maintained validation tools](../../tls_reference/README.md) and [timing protocol](../../tls_reference/timing/README.md). The [topline figure](../../../docs/figures/transit_benchmarks_20260910.png) combines this TLS campaign with the separately dated [BLS evidence](../transit_2026-09-08/README.md).

The [long-control diagnostic](stress/diagnostic/README.md) reproduces the original coarse discrepancy through shared float32 cumulative-sum variability. Native repeats can change masks and SDE; identical saved intermediates give identical native and fused scores. All nine repeats retain the selected period and final fit, with the true annual period tied with the one-third alias. The original stress comparison stays failed, and no universal bitwise-repeatability claim is made.
