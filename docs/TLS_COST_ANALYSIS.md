# Transit-search rental cost

The [current benchmark report](TRANSIT_BENCHMARKS.md) links measured execution time to independent recovery. Cost comparisons have the same recovery qualifications as speedups. The measured A40 bundle is $0.49/hour, including its CPU allocation.

| Observing pattern | v1 BLS / million | PyPI BLS / million | v1 TLS / million | GTLS / million | CPU BLS hourly break-even |
|---|---:|---:|---:|---:|---:|
| TESS 200 s | $0.21 | $0.90 | $5.10 | $60.73 | $0.0111/h |
| Separated TESS sectors | $2.14 | $5.86 | $3.60 | $631.43 | $0.0086/h |
| ZTF g/r | $7.50 | $13.63 | $9.24 | $1437.05 | $0.0261/h |

TLS uses the fine dense-TESS grid and original separated-TESS grid, which pass the independent joint criterion; ZTF retains an unqualified original-grid timing. BLS's separated-TESS upgrade supports its recovery comparison. The full reports give the other settings and confidence bounds.

These are linear projections of median 16-source throughput, not measured million-source jobs. The boundary includes API host work, transfers, periodograms and candidates from prepared arrays. Preprocessing, imports, context setup, grid construction, I/O, idle time and vetting are excluded. Fresh-grid BLS timings are reported separately. A complete survey or QLP bill cannot be inferred from these values.

CPU-only break-even price = $0.49 / (CPU time ÷ v1 GPU time), for a CPU service delivering the measured throughput. No standalone CPU rental was benchmarked. The measurement used a 7.65-CPU-equivalent quota on a Xeon Gold 6342 host; 96 host logical CPUs were not the allocation.

[BLS evidence](../benchmarks/results/transit_2026-09-08/README.md) · [TLS evidence and rental ledger](../benchmarks/results/tls_sensitivity_2026-09-09/README.md) · [Synthetic HATPI cost pilot](../benchmarks/results/tls_sensitivity_2026-09-09/HATPI.md). These projections apply to the measured workloads. The [provenance audit](BENCHMARK_PROVENANCE.md) explains why earlier whole-survey cost claims were retired.
