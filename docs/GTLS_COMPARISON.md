# cuvarbase TLS and GTLS: speed, recovery and implementation

The [current transit benchmark](TRANSIT_BENCHMARKS.md) compares exclusive single-source and batch timings on one A40, together with independent recovery and null tests on observed ZTF and TESS cadences. Its figure and qualifications replace the earlier equal-SDE headline.

| Stage | cuvarbase v1 TLS | Pinned public GTLS |
|---|---|---|
| Coarse search | Fold into weighted phase bins; reuse the bins across template trials | Sort individual observations by phase; template widths use observation counts |
| Depth / objective | Analytic weighted template-depth fit, with unit baseline | Unweighted window-mean depth estimate with template overshoot, followed by weighted residuals |
| Candidate precision | Exact observation-level refinement of selected top candidates | Different epoch/duration sampling and refinement policy; fast mode returns an SDE spectrum |
| Significance | Native SDE calibrated on independent nulls | Its own native SDE calibrated on the same independent null inputs |

These are related transit-template algorithms with different numerical searches. A common trial-period array and limb-darkening coefficients do not make them identical. Similar scalar SDE values, including values recomputed with one formula, do not establish equivalent recovery or false-alarm behavior.

The speed difference combines cuvarbase’s phase-bin architecture with GTLS host orchestration overhead. Measured diagnostic changes batch GTLS’s per-period flux-prefix-sum loop and repeated duration-mask union operations. Full output comparisons and synchronized component timings are in the [current experiment](../analysis/transit-recovery-20260908/README.md) and the [earlier TLS component audit](../analysis/tls-profile-20260908/README.md). These diagnostic patches are separate from the public upstream competitor. Warm CUDA module compilation/lookup was negligible in the earlier profiles.

The fast cuvarbase engine predates phase 5; the entire advantage is not a phase-5 gain. The current benchmark also tunes documented GTLS fast mode and density constraints, and measures concurrent throughput with separately validated recovery because available GPU memory can change GTLS chunking and its spectrum.

Earlier CPU TLS failures were zero-sample template/model edge cases. Some happened before the search; the ZTF/Rubin cases completed the period search and failed during output-model construction. Failed API times are excluded from speedup claims.

The original July comparison and its arithmetic remain in the [preserved document](../analysis/transit-recovery-20260908/sources/claims-before/docs/GTLS_COMPARISON.md) and [provenance audit](../analysis/benchmark-audit-20260906/README.md). In particular, the former 30–171× “equal sensitivity” claim and claims about the GTLS paper’s exact hidden settings are not supported by that evidence.
