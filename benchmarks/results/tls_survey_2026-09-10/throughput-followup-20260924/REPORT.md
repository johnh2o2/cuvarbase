# September 24 throughput follow-up

These are repeated original timing workloads on one new allocation, with unchanged numerical sources and full grids. Each available panel has three complete queues of at least 96 attempts and 120 seconds, in whole cohort cycles.

The original experimental exactness outcome remains 5,111/5,120; its 9 mismatches still fail the aggregate gate. These timing repetitions do not requalify sensitivity.

![Sustained throughput](throughput.png)

| Workload | Method | Median / second | Observed range | API failures / attempts | Selected discrepancies |
| --- | --- | ---: | ---: | ---: | ---: |
| tess_solar | TLS baseline | unavailable | — | — | — |
| tess_solar | TLS experimental | 8.0722 | 8.0619–8.0813 | 0/2976 | 0 |
| tess_solar | GTLS | 2.476 | 2.4638–2.5016 | 0/928 | 0 |
| tess_solar | BLS execution | 6.3618 | 6.358–6.3626 | 0/2304 | 85 |
| tess_gap_long | TLS baseline | 0.77039 | 0.77038–0.77062 | 0/288 | 0 |
| tess_gap_long | TLS experimental | 0.77554 | 0.77082–0.77567 | 0/288 | 0 |
| tess_gap_long | GTLS | unavailable | — | — | — |
| tess_gap_long | BLS execution | 11.445 | 11.413–11.455 | 0/4128 | 421 |
| ztf_solar | TLS baseline | 0.45455 | 0.45066–0.45549 | 0/288 | 0 |
| ztf_solar | TLS experimental | 0.82349 | 0.82102–0.82982 | 0/384 | 0 |
| ztf_solar | GTLS | 0.12111 | 0.12048–0.12299 | 0/288 | 0 |
| ztf_solar | BLS execution | 29.851 | 29.78–29.88 | 0/10768 | 995 |
| varied | TLS baseline | unavailable | — | — | — |
| varied | TLS experimental | unavailable | — | — | — |
| varied | GTLS | unavailable | — | — | — |
| varied | BLS execution | 10.846 | 10.838–10.866 | 0/4032 | 153 |

BLS rates count successful native completions and include failed-call elapsed time and per-attempt journal overhead. BLS selected discrepancies include the pre/post diagnostic comparisons and measured queues; they introduce no tolerance or numerical passing label. Its original exact qualification remains failed. TLS/GTLS rates require the unchanged strict timing gates.

The CSV retains the selected worker/batch settings, comparison counts, cold preparation, sampled GPU memory, unavailable reasons and cost projections. BLS batches group serial native calls within a worker. Projected costs use the median successful rate at the recorded hourly price; they exclude acquisition, preprocessing and vetting and do not describe an actual million-source run.

Verified evidence archive SHA256: `a0ffd2dcc13829f11eb1ad2355ff1be652d5b137eabf9a32286885934d61029a`.
Frozen follow-up design SHA256: `2165ec272d73b2b7092c5dfec224e2703927be1fec15a521debd811b690d12eb`.
