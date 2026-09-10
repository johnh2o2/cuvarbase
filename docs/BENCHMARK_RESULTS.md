# Benchmark results

The [current transit benchmark](TRANSIT_BENCHMARKS.md) is the source for cuvarbase v1 performance claims. The standard TLS engine now searches individual observations with full refinement. Earlier measurements of the binned TLS engine remain dated evidence and do not describe the new default.

- [Speed and recovery figure, methods and qualifications](TRANSIT_BENCHMARKS.md): v1 BLS versus actual PyPI 0.2.5 and the strongest tested CPU/GPU settings; v1 TLS versus public GTLS.
- [TLS implementation and component comparison](GTLS_COMPARISON.md): the shared numerical search, execution changes, and separate search/API timing boundaries.
- [Search-cost projections](TLS_COST_ANALYSIS.md): hardware rates, measured workloads and the limits of extrapolating their costs.
- [BLS competitor experiment](../benchmarks/results/transit_2026-09-08/README.md) and [its evidence archive](../benchmarks/results/transit_2026-09-08/ARCHIVE.md): frozen inputs, source pins, selected configurations and results.
- [Archived binned TLS study, 2026-09-09](../benchmarks/results/tls_sensitivity_2026-09-09/README.md): recovery/null cohorts, three binned resolutions, exclusive timing and a secondary BLS control. Its speed ratios and HATPI cost pilot apply to that earlier engine.
- [TLS numerical strategy](TLS_NUMERICS.md): the standard observation-level search and the accuracy audit that motivated replacing the binned default.
- [Historical-claim audit](BENCHMARK_PROVENANCE.md): why earlier claims were retired and how to retrieve their original wording and raw measurements.

There is no current general ranking against the best competitors for Lomb–Scargle, NFFT, CE or PDM. Older timing tables do not establish one. Diagnostic and correctness records for those algorithms remain available in the [benchmark index](../benchmarks/README.md).
