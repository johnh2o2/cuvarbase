# Runtime planning checkpoint — 2026-09-11

At 10:09 UTC the frozen calibration search had recorded 6,152 valid method
outcomes. High-impact calibration was complete, and the first four M-dwarf
calls per method agreed with the development-derived runtime forecast.
These are elapsed API-call timings under four-worker contention, not separately
tuned sustained-throughput measurements or detection-performance results.

| Calibration regime | Calls per method | BLS mean / forecast | TLS mean / forecast |
| --- | ---: | ---: | ---: |
| ZTF high impact | 512 | 126.36 / 126.30 s | 16.17 / 17.76 s |
| ZTF M dwarf | 4 | 197.93 / 198.01 s | 24.51 / 24.79 s |

The [timing snapshot](calibration-timing-checkpoint-20260911T1008Z.json) records
per-regime counts, sums and descriptive timings, source-receipt identities,
stage timestamps and spending. The [remaining-work calculation](remaining-envelope-20260911T1009Z.json)
uses the unchanged [frozen forecast](../runtime-projection-selected-final.json):
512 calibration calls minus completed calls, 256 future injection calls and
256 future independent test-null calls, per regime and selected method.
In-flight calls remain counted in full. The [independent arithmetic review](root-envelope-review-20260911T1009Z.json)
recomputes these counts, costs and allowances; [the copy index](index.json)
identifies the preserved source files by SHA-256.

The [10:22 UTC follow-up](mdwarf-first20-check-20260911T1022Z.json) preserves
the first 20 completed M-dwarf calls per method as 40 timing-only rows. BLS
averaged 196.78 s against a 198.01 s forecast (−0.62%); TLS averaged 25.01 s
against 24.79 s (+0.87%). These five waves support leaving the forecast
unchanged, but do not establish runtime tails. This follow-up used existing
receipts without inspecting detection scores or running additional GPU work.

The remaining planning envelope was **38.32 hours**: 22.20 hours of scientific
searches, 4.84 hours of baseline comparisons, a 10-hour primary throughput
measurement allowance, the one-hour supplementary BLS limit, and an estimated
0.28 hours of further input preparation. The 10-hour measurement allowance is
a planning estimate, not an imposed timeout. Reporting, archive creation,
transfers, cleanup and additional timing variation consume the **11.38 hours**
left between that envelope and the existing study guard.

That accounting projects **$74.58 cumulative compute before unmeasured overheads
and storage**, including the earlier $50.26 expenditure. Using the entire $30
study cap would bring cumulative compute to $80.26, within the existing $100
authorization. Rental estimates are not invoices. The evidence supports keeping
the full workload and current budget controls unchanged.

Four simultaneous M-dwarf calls per method cannot establish runtime tails.
Held-out injections and independent test-null searches are still unmeasured;
input-preparation and baseline-comparison timings remain extrapolations.
This checkpoint neither guarantees a completion time nor changes any scientific
setting, acceptance tolerance, sample count or operational deadline.
