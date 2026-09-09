# Transit-search rental cost

The [current benchmark report](TRANSIT_BENCHMARKS.md) reports measured execution time and independent recovery. Cost savings have the same recovery qualifications as speedups. The A40 bundle used here costs $0.49/hour, including its CPU allocation.

| Observing pattern | v1 BLS / million | PyPI BLS / million | v1 TLS / million | GTLS / million | CPU BLS hourly break-even |
|---|---:|---:|---:|---:|---:|
| TESS 200 s | $0.21 | $0.90 | $0.21 | $61.03 | $0.0111/h |
| Separated TESS sectors | $2.14 | $5.86 | $3.08 | $635.07 | $0.0086/h |
| ZTF g/r | $7.50 | $13.63 | $8.54 | $789.81 | $0.0261/h |

These are linear projections of the median 16-source search throughput, not measured million-source jobs. The boundary includes transfers, periodograms and candidate ranking from prepared arrays; preprocessing, imports, grid construction, I/O, idle time and vetting are excluded. Fresh-grid timings are reported separately. A complete QLP or survey bill cannot be inferred from these values.

CPU-only break-even price = $0.49 / (CPU time ÷ v1 GPU time), for a CPU service delivering the measured throughput. No standalone CPU rental was benchmarked. The measurement used a 7.65-CPU-equivalent quota on the same Xeon Gold 6342 host; 96 host logical CPUs were not the allocation.

The [full report](../benchmarks/results/transit_2026-09-08/README.md) contains recovery qualifications, repetitions, hardware, pinned versions and the experiment rental ledger. These projections apply to the measured workloads. The [provenance audit](BENCHMARK_PROVENANCE.md) explains why earlier whole-survey cost claims were retired.
