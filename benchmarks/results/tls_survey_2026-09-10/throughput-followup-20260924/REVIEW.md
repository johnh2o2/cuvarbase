# Follow-up review, September 27

The completed follow-up has **11 reportable panels**: seven strict TLS/GTLS
timings and four BLS execution-only timings. All 21,232 measured BLS calls
completed, with no API failures. Its 1,654 selected-output discrepancies include
measured queues and diagnostic comparisons; they do not acquire numerical
qualification. Repeated calls are not independent scientific populations.

Five panels stay unavailable:

| Engine | Workload | Recorded failure |
| --- | --- | --- |
| Baseline TLS | TESS solar | The first measured queue changed the selected SDE for null 0015; its selected-period hash agreed. |
| Baseline TLS | Varied | Pre-queue power, chi2 and SDE differed for varied TESS solar null 0012; selected period agreed. |
| Experimental TLS | Varied | Pre-queue power, chi2 and SDE differed for varied TESS solar null 0011; selected period agreed. |
| GTLS | TESS long gap | Out-of-memory failure in the first measured queue. |
| GTLS | Varied | Pre-queue power, chi2 and SDE differed for varied long-gap null 0011, and two API calls ran out of memory. The compared selected period agreed. |

[Machine-readable review](failure-review.json) binds these observations to the
original receipt hashes. The TLS varied-case SDE changes were approximately
`+4.29e-6` and `-2.38e-6`; the GTLS varied-case change was approximately `+0.0531`.
No tolerance was widened, failure replaced, or scientific experiment repeated.
These observations do not establish a cause or prove an absence of scientific
impact. Baseline mode preserves its implementation without promising bitwise
determinism under every execution history.

The experimental/baseline median throughput ratio is **1.812** for ZTF solar
and **1.007** for long-gap TESS, where the paired complete-spectrum checks
passed. These are timing-cohort ratios. The original study's **5,111/5,120**
exactness outcome and its nine mismatches still fail the aggregate gate.

The full GPU suite passed **2,091 tests**, with one expected notebook failure
and zero skips. Its separate release gate initially failed because the launcher
had not installed the package. The [isolated installed-wheel validation](../release-gate-20260927/README.md)
subsequently passed all 14 numerical/runtime checks and six dependency
preflights. That fixes validation setup and does not alter any timing outcome.

The figure's horizontal margins were corrected so unavailable labels remain
inside their panels. [Review provenance](review.json) verifies that all 16 result
rows and the original exactness evidence stayed unchanged. The original figure
and report remain preserved locally and in the original immutable R2 bundle.

Both rentals are terminated, and both evidence bundles passed full R2 checksum
read-back. Estimated follow-up compute was **$2.8053** for the benchmark and
**$0.0373** for the installed-wheel check. The cumulative conservative ledger,
including retained storage reserves, is **$78.1846** within the existing $100
authorization. These are estimates and reserves, not provider invoices.

Benchmark collection and review are complete. Release publication remains
pending; no tag or published package was changed by this review.
