# Benchmark results

The [September 2026 transit benchmark](TRANSIT_BENCHMARKS.md) is the current source for cuvarbase v1 performance claims. Its timing figure and recovery tables report single-source latency, batch throughput, independent transit recovery and false-positive checks on observed TESS and ZTF cadences with synthetic flux and noise.

- [Speed and recovery figure, methods and qualifications](TRANSIT_BENCHMARKS.md): v1 BLS versus actual PyPI 0.2.5 and the strongest tested CPU/GPU settings; v1 TLS versus public GTLS.
- [TLS implementation and component comparison](GTLS_COMPARISON.md): which computations and overheads differ, and why timing alone does not establish equivalent sensitivity.
- [Search-cost projections](TLS_COST_ANALYSIS.md): measured A40 throughput, timing boundaries and CPU break-even prices.
- [Full experiment](../benchmarks/results/transit_2026-09-08/README.md) and [evidence archive](../benchmarks/results/transit_2026-09-08/ARCHIVE.md): frozen inputs, source pins, selected configurations, results and verification scope.
- [Historical-claim audit](BENCHMARK_PROVENANCE.md): why earlier claims were retired and how to retrieve their original wording and raw measurements.

There is no current general ranking against the best competitors for Lomb–Scargle, NFFT, CE or PDM. Older timing tables do not establish one. Diagnostic and correctness records for those algorithms remain available in the [analysis index](../benchmarks/README.md).
