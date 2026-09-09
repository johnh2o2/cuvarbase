# Benchmark results

The [September 2026 transit benchmark](TRANSIT_BENCHMARKS.md) is the current source for cuvarbase v1 performance claims. Its timing figure and recovery tables report single-source latency, batch throughput, independent transit recovery and false-positive checks on observed TESS and ZTF cadences with synthetic flux and noise.

- [Speed and recovery figure, methods and qualifications](TRANSIT_BENCHMARKS.md): v1 BLS versus actual PyPI 0.2.5 and the strongest tested CPU/GPU settings; v1 TLS versus public GTLS.
- [TLS implementation and component comparison](GTLS_COMPARISON.md): which computations and overheads differ, and why timing alone does not establish equivalent sensitivity.
- [Search-cost projections](TLS_COST_ANALYSIS.md): measured A40 throughput, timing boundaries and CPU break-even prices.
- [BLS competitor experiment](../benchmarks/results/transit_2026-09-08/README.md) and [its evidence archive](../benchmarks/results/transit_2026-09-08/ARCHIVE.md): frozen inputs, source pins, selected configurations and results.
- [Independent TLS study](../benchmarks/results/tls_sensitivity_2026-09-09/README.md): larger recovery/null cohorts, three numerical resolutions, exclusive timing and a secondary BLS control.
- [TLS phase binning](TLS_NUMERICS.md): retained transit shape, approximation costs and why candidate refinement does not establish complete-search equivalence.
- [Historical-claim audit](BENCHMARK_PROVENANCE.md): why earlier claims were retired and how to retrieve their original wording and raw measurements.

There is no current general ranking against the best competitors for Lomb–Scargle, NFFT, CE or PDM. Older timing tables do not establish one. Diagnostic and correctness records for those algorithms remain available in the [benchmark index](../benchmarks/README.md).
