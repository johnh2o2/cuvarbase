# Supplementary null population

All **24 of 24** separately predeclared null inputs
passed the corrected-GTLS/cuvarbase numerical and public-result comparison.
Untouched GTLS matched on **24 of 24** inputs.
This population has its own unchanged manifest, seal, stream and acceptance
receipt; it is not merged into the main 160-case confirmation counts.

Each regime supplies `null_0008` through `null_0015`. Together with the main
study's eight nulls, these define 16 distinct noise-only inputs per timing
regime. These inputs support the planned single-source and 16-source batch
timing campaign. The supplementary numerical comparisons do not themselves
measure latency or throughput.

| Regime | Native / corrected / cuvarbase nulls above SDE 8 (of 8) |
| --- | ---: |
| TESS, ordinary solar | 1 / 1 / 1 |
| Gapped TESS, ordinary solar | 4 / 4 / 4 |
| ZTF, ordinary solar | 7 / 7 / 7 |

SDE 8 is descriptive and uncalibrated. `strata.csv` retains each regime's exact
binomial interval, discordance bound and any failures. The methods' numerical
agreement does not imply that this threshold has the same false-positive rate
on different cadences. See the [main confirmation](../validation/README.md)
for the shared numerical endpoints, noise model, correction and limitations.

The original input arrays are in the compact bank. The
[timing reproduction route](../../../tls_reference/timing/README.md#reproduce-the-published-timing-cohort)
restores both populations and preserves their separate provenance through
validation and timing. `array_digests.json.gz` retains complete numerical
identities and the correction/no-op evidence; `validation.json` documents
verification and the original retention rule.

This population ran on a separate RTX A6000 host; the main confirmation used
an A40. Its original acceptance and complete compact results were recovered
unchanged after the subsequent development-control gate stopped the pipeline.
All 72 raw records and 48 comparisons were independently verified. Nine
retained output NPZ containers were not collected; their complete numerical
identities and prior on-host verification survive. The other 63 archives had
already been removed under the original retention rule. These collection
losses and the later control outcome do not alter this separately sealed
24-case gate. [Collection details](collection.json) and
[execution environment](execution_environment.json) keep those scopes explicit.
