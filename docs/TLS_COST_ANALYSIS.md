# Transit-search compute cost

The [benchmark report](TRANSIT_BENCHMARKS.md) records the workloads and sensitivity checks. The BLS campaign rented an A40 and its included CPU allocation for **$0.49/hour**. The new TLS campaign rented an RTX A6000 and its included CPU allocation for **$0.53/hour**; cuvarbase and GTLS use that same machine. Compute cost follows measured elapsed time; these measurements do not price a complete survey pipeline.

## BLS: measured batch throughput

| Observing pattern | v1 BLS / million | PyPI 0.2.5 BLS / million | CPU BLS hourly break-even |
|---|---:|---:|---:|
| TESS 200 s | $0.21 | $0.90 | $0.0111/h |
| Separated TESS sectors | $2.14 | $5.86 | $0.0086/h |
| ZTF g/r | $7.50 | $13.63 | $0.0261/h |

These are linear projections of the September 8 median 16-source throughput, not measured million-source jobs. BLS's executable source and kernels remain unchanged apart from a documentation link. The separated-TESS PyPI comparison supports the stated recovery and false-positive criterion; the other timing ratios retain their sensitivity qualifications.

CPU break-even price is `$0.49 / (CPU time / v1 GPU time)`, for a CPU service delivering the measured throughput. No standalone CPU rental was benchmarked. The BLS measurement used a 7.65-CPU-equivalent allocation on a Xeon Gold 6342 host; the host's total logical CPU count was not the allocation.

## TLS: measured batch throughput

| Observing pattern | v1 TLS / million | GTLS / million | Cost ratio | GTLS workers |
| --- | ---: | ---: | ---: | ---: |
| TESS: dense sector | $23.69 | $47.00 | 1.98× | 4 |
| TESS: separated sectors | $225.85 | $542.44 | 2.40× | 2 |
| ZTF g/r | $458.80 | $669.17 | 1.46× | 4 |

The whole GPU/CPU rental is charged once, regardless of worker count. TLS cost is `median batch seconds / 16 × $0.53 / 3600 × 1,000,000`. The table compares the standard cuvarbase batch with the fastest eligible tested GTLS pool. It projects three repetitions of one fixed 16-source cohort, not a measured million-source survey or population-wide cost distribution.

The TLS machine has a 7.65-CPU quota on a Xeon Gold 6342 host; each worker uses one numerical-library thread; the GTLS batch comparison tests one, two and four workers. The standard TLS engine evaluates individual observations with full refinement. The older phase-binned study and synthetic HATPI pilot used a different engine and cannot price this default. The [current TLS evidence](../benchmarks/results/tls_reference_2026-09-10/README.md) records numerical agreement and the separate search/diagnostic timing boundaries.

The original timing campaign failed when four-worker GTLS ran out of memory in the separated-TESS warmup. These projections use the separately audited completed configurations; the two-worker pool was the fastest eligible completed setting for that cadence. [Timing assessment](../benchmarks/results/tls_reference_2026-09-10/reporting_acceptance.json).

## Included work and limits

The API timings include host work, GPU transfers and completed search results from prepared arrays. TLS includes construction, input validation, template preparation, full candidate/harmonic refinement and final fitting. Each API's normal output work is included; GTLS additionally computes SNR and pink-noise diagnostics. Common-search timings are reported separately.

The calculations exclude input loading, preprocessing, imports and CUDA context startup, explicit period-grid construction, idle time, vetting and storage. The recorded 80 GB TLS container adds about $0.0111/hour while running under the ledger's 720-hour monthly conversion; it is separate from the recorded compute rate. Real workloads also vary in source length, stellar parameters, noise and search domain. Fresh-grid BLS timings appear separately in the benchmark report.

[BLS evidence](../benchmarks/results/transit_2026-09-08/README.md) · [Current TLS evidence](../benchmarks/results/tls_reference_2026-09-10/README.md) · [Retired-claim provenance](BENCHMARK_PROVENANCE.md).
