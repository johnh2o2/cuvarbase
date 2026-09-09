# cuvarbase TLS and GTLS: speed, recovery and implementation

The [current transit benchmark](TRANSIT_BENCHMARKS.md) compares exclusive single-source and batch timings on one A40, together with independent recovery and null tests on observed ZTF and TESS cadences. The independent follow-up supports bounded recovery / false-positive matching for both TESS examples: fine sampling for dense TESS and original sampling for separated TESS. ZTF remains inconclusive under the strict matching rule, despite more recovered injections and fewer observed false positives. [Full study](../benchmarks/results/tls_sensitivity_2026-09-09/README.md).

| Stage | cuvarbase v1 TLS | Pinned public GTLS |
|---|---|---|
| Coarse search | Fold into weighted phase bins; reuse the bins across template trials | Sort individual observations by phase; template widths use observation counts |
| Depth / objective | Analytic weighted template-depth fit, with unit baseline | Unweighted window-mean depth estimate with template overshoot, followed by weighted residuals |
| Candidate precision | Observation-level refinement of selected top candidates; SDE still uses the coarse spectrum | Different epoch/duration sampling and refinement policy; fast mode returns an SDE spectrum |
| Significance | Native SDE calibrated on independent nulls | Its own native SDE calibrated on the same independent null inputs |

These are related transit-template algorithms with different numerical searches. A common trial-period array and limb-darkening coefficients do not make them identical. Similar scalar SDE values, including values recomputed with one formula, do not establish equivalent recovery or false-alarm behavior.

[The phase-binning explanation](TLS_NUMERICS.md) illustrates what information the bins retain and measures the isolated SNR cost on the earlier injections. It also explains why refinement cannot repair every detection loss from the coarse search.

The speed difference combines cuvarbase’s phase-bin architecture with GTLS host orchestration overhead. Measured diagnostic changes batch GTLS’s per-period flux-prefix-sum loop and repeated duration-mask union operations. Full output comparisons and synchronized component timings are in the [three-cadence component experiment](../benchmarks/results/transit_2026-09-08/README.md) and the [earlier TLS component audit](../benchmarks/results/tls_profile_2026-09-08/README.md). These diagnostic patches are separate from the public upstream competitor. Warm CUDA module compilation/lookup was negligible in the earlier profiles.

The fast cuvarbase engine predates phase 5; the entire advantage is not a phase-5 gain. The benchmark uses documented GTLS fast mode and density constraints, and measures concurrent throughput with separately calibrated recovery because available GPU memory can change GTLS chunking and its spectrum. Preflight memory failures required two workers and explicit release of unused CuPy memory-pool blocks on ZTF. Remaining failed API calls are retained in the sensitivity outcomes; none supplies a successful timing denominator.

Earlier CPU TLS failures were zero-sample template/model edge cases. Some happened before the search; the ZTF/Rubin cases completed the period search and failed during output-model construction. Failed API times are excluded from speedup claims.

The [provenance audit](BENCHMARK_PROVENANCE.md) records why the July equal-sensitivity headline was withdrawn and how to recover the original documents from Git history. The archived evidence also does not establish the GTLS paper’s exact hidden settings.
