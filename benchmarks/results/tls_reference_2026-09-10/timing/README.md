# TLS timing results

On the recorded RTX A6000 allocation, cuvarbase's observation-level TLS search
had **3.6–4.6× lower single-source latency** and **1.5–2.4× better throughput**
than the fastest eligible tested GTLS pool on 16 distinct noise-only inputs.
These are warm, complete public API calls. The [figure data](../timing_analysis.json)
is bound to the separate [reporting assessment](../reporting_acceptance.json).

| Cadence | Single cuvarbase / GTLS (s) | Single speedup | Batch cuvarbase / GTLS (s/source) | Batch speedup | GTLS batch workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense TESS | 0.149 / 0.533 | 3.58× | 0.161 / 0.319 | 1.98× | 4 |
| Gapped TESS | 1.554 / 6.037 | 3.88× | 1.534 / 3.684 | 2.40× | 2 |
| Sparse ZTF | 3.231 / 14.899 | 4.61× | 3.116 / 4.545 | 1.46× | 4 |

Each single-source median uses five repetitions of the fixed null0000 input.
Each batch median uses three repetitions of the same 16 distinct null inputs
for every method and pool. Both methods use one worker for the single-source
comparison; cuvarbase uses one worker for batches, while GTLS pools of 1, 2 and
4 are tested. Pool selection uses the lowest eligible batch median. Batch times
are divided by the actual 16 returned sources. No extrapolated survey scaling
or additional worker multiplier enters the denominator.

The allocation had one **NVIDIA RTX A6000**, **7.65 CPU cores of cgroup quota**
on an Intel Xeon Gold 6342 host, and one numerical library thread per worker.
The [hardware receipt](../sources/timing/hardware.json) preserves the actual
GPU identity, quota and hourly bundle rate. Startup and full warmup are recorded
separately. Timed calls include construction, validation, cache creation, full
search, final fit and completed GPU work. File loading, supplied period-grid
generation and result hashing are outside the measured interval.

## A failed configuration remains excluded

The [original full campaign acceptance](acceptance.json) is **failed**. Gapped
TESS with four GTLS workers ran out of GPU memory during warmup while requesting
an additional 1,623,613,440-byte array. It completed no measured repetitions.
Its [complete failure record](public/tess_gap/gtls_graph_4worker/record.json)
remains present; no elapsed time from this failure enters a speed ratio.

After observing that failure, a separately labeled **post hoc reporting
assessment** retained only complete comparisons. All 12 planned configurations
are terminal and accounted for; 11 completed. The original campaign rejection
and [original normalized output](timing_analysis.json) remain unchanged. The
separate assessment replays the original final audit, requires every other
original prerequisite, verifies all source/input/ownership receipts, and checks
complete returned-object stability as well as the original frozen search
fingerprints. It does not rerun measurements or relax numerical rules.

The included data contain 30 measured single calls and 33 measured batches,
covering 558 complete returned objects. The underlying numerical study is
separate: 160 independent cases and 24 additional nulls. These timing receipts
do not establish a universal recovery or false-positive margin. All 48 timing
inputs have a trace proving the disclosed native host correction is a no-op,
so the conditional extra corrected-native timing was unnecessary.

## Where elapsed time goes

Separate instrumented calls measure the search through final window selection.
The native endpoint follows the final GPU argmin; cuvarbase's endpoint also
includes transfer of its compact winner fields. These identify the same search
stage, with a small difference in endpoint scope. They are not headline public
API denominators.

| Cadence | cuvarbase common search (s) | GTLS common search (s) | Search speedup | GTLS after-search work (s) |
| --- | ---: | ---: | ---: | ---: |
| Dense TESS | 0.167 | 0.463 | 2.77× | 0.110 |
| Gapped TESS | 1.563 | 6.249 | 4.00× | 0.076 |
| Sparse ZTF | 3.086 | 16.181 | 5.24× | 0.313 |

All 30 instrumented outputs match their own complete literal API result and
frozen search fingerprint. GTLS's later physical and per-transit SNR/pink-noise
diagnostics explain part of its public-call cost. The common-search comparison
shows that a substantial improvement remains before that work. Stage timings
are inclusive and can overlap; separately computed medians need not add to the
median total. Raw records retain the individual repetitions and stage values.

[raw-files.json](raw-files.json) inventories all 43 original timing files.
[Source and CPU replay instructions](../sources/timing/README.md) reproduce the
reporting assessment from the public evidence without a GPU. Earlier failed
preflight attempts are retained under
[failed-attempts](../sources/timing/failed-attempts/README.md).
