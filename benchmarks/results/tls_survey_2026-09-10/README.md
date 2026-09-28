# Survey TLS study — collected results; release validated

The frozen science and timing campaigns are complete, their archives were
verified locally, and the original rental was terminated. The
[84-product publication receipt](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/FINAL_PUBLICATION.json") and
[source-to-copy inventory](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/FINAL_ASSEMBLY.json") bind the collected science,
report tables, original failed timing receipts and figures. Execution completion
does not grant numerical qualification: the experimental TLS candidate matched
**5,111/5,120** original held-out results, and the frozen zero-mismatch contract
**failed**. All selected periods, recovery/alias flags and both frozen threshold
decisions agreed. The nine chi2/SDE differences remain preserved in the
[exactness report](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-science/exactness-final.json") and
[mismatch table](final-report/exactness_mismatches.csv). The
[default-preserving release](release-validation/README.md) is now applied:
`execution="baseline"` retains the original default, and
`execution="experimental"` selects the optimization bundle. Separate GPU wiring
validation passed **24/24 paired comparisons and 86/86 device tests** in
**177.825 seconds**, with a passing independent audit. These fixed-case wiring
checks do not requalify experimental sensitivity or historical throughput; the
figure's “Optimized” label refers to the **opt-in experimental candidate**.

The [final recovery report](final-report/RECOVERY.md) compares observation-level,
GTLS-compatible TLS with native GPU BLS selected separately for each regime.
At the independently calibrated 5% target, the existing simultaneous intervals
support a TLS recovery advantage in dense solar, high-impact, eccentric and
M-dwarf TESS populations. They also preserve a severe smeared grazing failure:
TLS recovered **1/256**, versus **109/256** for BLS. The three ZTF populations
and long-gap TESS have negative point differences but simultaneous intervals
crossing zero; the small HATpi-like recovery difference also crosses zero.
The [paired contrasts](final-report/paired_contrasts.csv) retain all ten regimes
at both 5% and 1% targets. These are common target FPRs, not identical realized
FPRs: [independent test-null rates and intervals](final-report/recovery_fpr.csv)
remain explicit. No pooled advantage, universal sensitivity claim or
sub-percentage equivalence follows from this finite experiment.

The [held-out expected-SNR diagnostics](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-science/heldout-snr-final.json")
cover all 2,560 injections, with [descriptive groups](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/snr_descriptive.csv")
and [sampling/target-SNR recovery](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/subgroups.csv"). They use common
matched-filter definitions, not package SDE/SNR equivalence. White-noise responses
are the enumerated template-family ceilings; OU responses evaluate those same
white-selected filters, not independently OU-optimized maxima. Unsampled and
few-event signals remain included. The [frozen science seal](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/seal-final.json")
and [auxiliary plan](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/exactness-plan.json") preceded held-out generation, with
**zero operative allowance** for approximation losses in every regime. Native
GTLS compatibility and this synthetic-flux/cadence coverage do not establish
canonical CPU TLS equivalence or universal physical coverage.

The [final throughput figure](final-figures/survey-throughput-with-native-bls.png)
([PDF](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-figures/survey-throughput-with-native-bls.pdf"),
[values/provenance](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-figures/survey-throughput-with-native-bls.data.json"))
shows **seven available and nine unavailable** backend/panel results. Both local
qualification gates and the unchanged baseline/candidate pairing passed for
ZTF solar and long-gap TESS. Their median-rate ratios are **1.850×** and
**1.007×**, respectively. The [exact values CSV](final-figures/survey-throughput-with-native-bls.csv)
retains independently tuned worker/batch settings, three-repetition ranges,
cold preparation, amortized cost and sampled memory. The
[collected timing note](final-timing/reporting/TIMING_LINKED.md) explains those
boundaries and the unchanged exclusions; its linked edition records a corrected
prose description of the already-correct cost formula. These are qualified
finite timing cohorts, not global sensitivity preservation. Baseline dense
TESS failed its post-queue gate, baseline varied failed its pre-queue gate,
and the candidate varied reference failed before selected-pool measurement.
Public GTLS gap and varied failed with out-of-memory errors in their first
queues. The [original final campaign](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-timing/primary/throughput-final/campaign.json")
and every failed reference/result remain unchanged. The varied-size workload
therefore has no qualifying throughput result.

Native BLS has no qualifying original timing setting. The separate execution
supplement also produced **no rates**: its launcher set four CPU-thread variables
but omitted `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS`. The frozen runner
rejected their recorded unset values before creating workers. All
[three development pilot receipts](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-timing/native-bls/tune/campaign.json")
retain that allocation-precheck failure; the
[measurement campaign](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-timing/native-bls/measure/campaign.json") contains
four explicitly unavailable panels. This launcher/validation integration failure
is separate from BLS's earlier numerical-repeatability failure. The
[launch audit](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-timing/reporting/native-bls-launch-audit.json") pins the actual
launcher source and all failed pilot receipts. No replacement
trial, passing tolerance or BLS speed bar was fabricated. The requested complete
native BLS and varied-queue throughput comparisons remain unfulfilled.

[Primary collection](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/collection/primary-collection-state.json") and
[supplement collection](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/collection/supplement-collection-state.json") both verified
all archived bytes; the supplement handed control back before original teardown.
The [closed original ledger](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/collection/original-rental-closed-ledger.json")
retains the original compute estimate of **$20.9600**. The
[final ledger](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/collection/final-ledger.json"), including release validation and
elapsed container storage, records **$71.85225 cumulative estimated spend** and
**$73.75634 conservatively including reserves**, within the existing **$100
total**, not a new allowance. The conservative total retains the full $1.50
uncertainty reserve for the rejected rental request; this is not an observed
charge. These are estimates, not invoices. Both actual rentals are verified
absent and all owned controls are closed. Bulk arrays and journals remain in the
verified archives identified by [FINAL_ASSEMBLY.json](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/FINAL_ASSEMBLY.json") and the
[release collection](release-validation/README.md#full-outputs-and-reproduction).

## Final requirements and remaining work

| Requirement | Collected evidence and remaining limitation |
| --- | --- |
| Preserve the observation-level default | Baseline-default/explicit-experimental release applied; 24/24 fixed-case wiring pairs and 86/86 device tests passed, independently audited. The experimental candidate still failed aggregate exactness at 5,111/5,120 despite identical stored periods/flags/decisions; release wiring does not requalify it. |
| Compare TLS with strong BLS | Completed blind recovery with separately frozen BLS settings for all ten regimes; final tables preserve positive TLS regimes, uncertain differences and severe grazing failure. No general TLS/BLS ranking is claimed. |
| Freeze accuracy limits before held-out evaluation | Reviewed science/auxiliary identities preceded input generation; all approximation allowances are zero. No post-evaluation gate was relaxed. |
| Independently calibrate and measure detection | 512 calibration nulls, 256 injections and 256 independent test nulls per regime; all 20,480 search outcomes valid. Both FPR targets, existing marginal/simultaneous intervals, discrete threshold limits and sampling subgroups are collected. |
| Explain losses on common inputs | Completed 2,560-injection white/OU diagnostics and physical boundary checks. Family-response diagnostics are descriptive and cannot replace blind recovery or attribute every implementation loss. |
| Measure sustained throughput | Independent tuning, three long-queue repetitions, cold/amortized cost and sampled memory are reported for seven eligible panels. Only ZTF solar and long-gap TESS permit paired TLS speed ratios. Native BLS and all varied-size panels remain unavailable; their failed receipts are preserved. |
| Preserve reproducibility and close spending | Both original archives, 84 compact study products and separate release-validation evidence verified. Release applied; both actual rentals absent and all owned controls closed. Final estimated cumulative cost is $71.85225, or $73.75634 including conservative reserves, within $100. Missing native BLS/varied-queue measurements remain unfulfilled. |

## Dated execution history and prospective evidence

The following material preserves the state and wording of earlier checkpoints.
Statements that measurements, collection or teardown were pending describe those
checkpoints; the collected results and remaining limitations above are current.

Separately, [old-study storage reclamation](../../../docs/STUDY_STORAGE.md)
reduced the retained file footprint by **39.62 GB**: 26.15 GB of archive-backed
NPZ copies, followed by 13.47 GB from exact compression of 489 retained tar
archives. The [independent postcheck](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/storage-archive-compression/summary.json")
passed; restoration starts with the shared archive kit, then the unchanged NPZ
kits. No active survey data was removed.

The [scientific protocol](../../tls_survey/README.md) declares the populations,
development tuning, independent calibration, recovery endpoints, uncertainty,
and approximation limits. The [throughput protocol](../../tls_survey/THROUGHPUT_PROTOCOL.md)
declares separate operating-configuration tuning and long-queue measurements.
Their [scientific seal](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/seal-final.json"), [auxiliary plan](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/exactness-plan.json"),
and [interpretation](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/seal-final-interpretation-v2.json") were reviewed before any
final input generation. The [launch review](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/root-final-launch-review.json")
verifies all 15 scientific sources, 82 candidate package files and 79 immutable
baseline package files. Launching this experiment does not qualify its results.

Development completed with **392 valid injection-search outcomes** and **3,136
valid null-search outcomes**, covering 80 injections and 640 null light curves.
All ten regimes have **zero operative allowance** for expected-SNR loss,
recovery loss or increased false-positive rate. The small development samples
did not establish a positive protected advantage that could fund approximation.
BLS configuration selection maximized development recovery, with finer
resolution breaking ties; speed did not select the control.

The detached workflow started on **2026-09-11 at 02:56 UTC**, initially tuning
each competitor's batch size and concurrency. Its collection controllers must
verify the final evidence before provider termination. The
[selected-configuration projection](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/runtime-projection-selected-final.json")
estimates 28.02 hours for the science searches and 4.84 hours for the additional
baseline comparisons; throughput tuning and measurement have separate planning allowances.
These are planning estimates, not measured final throughput or guaranteed
completion times. The existing $30 study guard remains within the user's $100
cumulative authorization.

A [separately reviewed capacity fallback](capacity-contingency/operational-addendum-v1.md)
was armed at **2026-09-12 06:13 UTC** to protect the same $29.90 trigger and
$30 study cap against local disk-full failures. Its cutoff remains
**2026-09-13 11:51:51 UTC**, with no new allowance. The ordinary collector
retains evidence and teardown ownership; closure must also verify that the
fallback and its independent wake process have exited after provider absence.

A [verified capacity checkpoint](capacity-checkpoint/README.md) was secured locally
at **2026-09-12 06:57 UTC**. It preserves all 10,240 frozen input cases as exact
arrays and original manifests, plus completed calibration/injection receipts
and a partial snapshot of 3,360 null outcomes. This backup does not establish
final science, throughput, collection completion or teardown.

A [reviewed runtime checkpoint](runtime-planning/README.md) records the completed
ZTF high-impact calibration timings and the first M-dwarf calls. Those timings
support keeping the frozen forecast and full workload unchanged. At that
checkpoint, the remaining planning envelope left 11.38 hours before the study
guard for reporting, archives, transfers and overruns; four M-dwarf calls per
method do not establish a runtime bound.

[Development throughput tuning](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-final.json") completed at
**2026-09-11 04:06 UTC**, with 12 of 16 attempted configurations eligible.
The frozen selections are baseline four workers/batch eight, candidate four
workers/batch four, and public GTLS two workers/batch one. All five eligible
candidate settings matched the baseline's complete spectra on the 24-source
tuning cohort. These development rates are not final throughput estimates or
held-out sensitivity qualification.

BLS has no qualifying timing setting: its single-worker trial changed the
selected likelihood score by −0.00003052 at the same period during the queue,
violating the predeclared repeatability gate. GTLS also has retained memory
failures and one post-queue spectrum-repeat failure. Those trials cannot supply
performance denominators; the original qualified figure must show missing BLS
timing panels.
These changes alone do not establish altered calibrated detection decisions.
BLS remains in the independent recovery comparison. The accuracy campaign
started after tuning, generated all 5,120 calibration nulls, and began searching
that bank with four workers at **2026-09-11 04:23 UTC**.
Calibration and threshold calculation completed at **2026-09-11 18:18 UTC**.
The [independent completion audit](calibration-completion-audit/README.md)
verified all 10,240 outcomes: 512 unique cases for each method in every regime,
paired input-file identities, source hashes, and exact shard membership.
All 40 thresholds match independent recomputation of the frozen order-statistic
rule. The 38 valid zero-score TLS grazing nulls remain included. Calibration
exceedance counts are not independent-test false-positive rates; realized FPR,
recovery and their uncertainty still require the held-out searches.
All 2,560 injections finished generation at **18:27 UTC**, followed by the
separate 2,560 test nulls at **18:36 UTC**. The
[bank preparation check](heldout-bank-preparation/README.md) verifies the
completed manifest counts, roles and recorded hash separation; it does not
replace search-time or archive verification of the held-out input bytes.
The blind injection search completed at **2026-09-12 01:32 UTC**, with all
5,120 method outcomes valid and all four workers exiting normally. The
[independent injection audit](injection-completion-audit/README.md) verified
the original bytes of all 2,560 input files, returned period-grid identities,
paired shard coverage, frozen sources/settings, and retention of unsampled,
few-event and few-point cases. Its initial checker hash-convention error and
corrected receipt are both retained. These checks do not establish recovery
or numerical equivalence. The separate test-null search completed at
**2026-09-12 08:28 UTC**, with all 5,120 outcomes valid and all four workers
exiting normally. Its [independent completion audit](null-completion-audit/README.md)
passed in one execution at **13:45 UTC**, verifying all 2,560 original input
files, paired shard coverage, returned grids, frozen source/settings bindings,
and retention of all latent sampling cases. The earlier 31 preparation checks
and original live worker handles remain preserved. This structural audit does
not estimate FPR or recovery. No scientific settings or tolerances changed
after freezing.
The [exclusion audit](throughput-tuning-exclusions-audit/AUDIT.md) links the
original failed receipts and records the exact differences. It also verifies
that public GTLS's automatic internal period batching exposes no supported
override omitted by this queue-batch/concurrency study.

Reporting those missing panels alone does not complete the requested BLS
throughput comparison. A separate [native BLS execution supplement](../../tls_survey/BLS_EXECUTION_PROTOCOL.md)
has therefore been prepared. It keeps the science-selected BLS settings and
grids, tunes execution settings on development inputs, and measures the same
independent timing cohorts after the primary campaign. It records every
numerical discrepancy and API failure; no new numerical passing tolerance is
introduced. Failed API calls consume elapsed time and reduce successful
throughput. The original repeatability failure remains explicit beside any
supplementary execution rates. Additional per-attempt journaling overhead is
included. The supplementary GPU envelope is capped at one hour ($0.49), inside
the existing study guard. Its [prospective seal](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/bls-execution-supplement/seal-v2.json")
and [launch review](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/bls-execution-supplement/root-launch-review-v2.json") were
completed before arming a waiting sidecar at **2026-09-11 05:05 UTC**. No
supplementary GPU work has started. The original collector is paused; after
primary completion, the sidecar must finish its bounded attempt and verify all
supplementary evidence locally before resuming that collector for primary
verification and provider teardown. The independent budget guard remains active.

The integrated checks passed **186 survey tests** and **70 operations tests**;
the [receipt](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/bls-execution-supplement/host-test-receipt-v2.json") retains commands,
source identities and complete logs. A synthetic figure was rendered and
visually checked; its values are not measurement results. An
[unlaunched first plan](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/bls-execution-supplement/rejected-prospective-v1/rejection.json")
was rejected because its heartbeat files could race primary archive collection.
The reviewed replacement keeps every mutable supplementary file outside the
primary archive's input trees. All original scientific and timing definitions
remain unchanged.

The [final handoff guide](FINAL_HANDOFF.md) records the current v2 collection
paths, local archive and design checks, supplementary extraction, combined
figure command, required delivery tables, and provider/ledger closure. It is
an unsealed operational note; its future products remain pending. The original
frozen launch runbooks retain their historical prospective wording.

## Available evidence

- [Host profile](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/host-profile.json"): isolated candidate-ranking and duration-group
  allocation measurements. These are CPU component measurements, not GPU
  end-to-end speedups.
- [Authorization](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/authorization.json"): the user's updated **$100 cumulative**
  ceiling and the preceding ledger's **$50.258718277017** estimated expenditure.
  This is not an additional $100 allowance. Rental estimates are not invoices.
- [Development grid audit](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/development-grid-audit.json"): the rejected original
  coarse-grid design. The final development policy increases period resolution
  for high-impact, eccentric, grazing and HATpi-like strata before held-out
  generation. The failure remains part of the evidence.
- [BLS response diagnostic](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/bls-response-final.json"): all four search resolutions
  evaluated at the known injected period on the original 80 development inputs,
  with reconstructed box responses and common white/OU expected-SNR definitions.
  Unsupported settings remain recorded. This diagnoses discretization; it is
  separate from the blind-search comparison and configuration selection.
- [Expected-SNR diagnostic](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/development-snr-final.json") and
  [physical boundaries](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/boundaries-final.json"): the final cloud development
  cohort, identified by manifest `a1d18d6c…`. The first compares ideal-box and
  native-template filter responses on common inputs; the second checks exposure
  integration and joint physical extremes. Annual-period boundary examples are
  known-transit diagnostics, not annual-period blind recovery.
- [Development cohort provenance](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/development-cohort-provenance.json"): the older
  local manifest `546f8319…` has identical times, bands and exposures, but small
  floating-point differences in periods, physical signals, fluxes and errors.
  Its original diagnostics and inputs remain separate dated evidence.
- [Grazing development diagnosis](grazing-development-diagnosis/README.md): a
  reviewed, reproducible CPU explanation using all eight original smeared,
  grazing development cases. Their noiseless window means fall below the
  native 10-ppm gate throughout the near-truth coarse width envelope; seven
  remain below across all cached widths. These are float64 diagnostics, not
  native GPU gate traces. The retained TLS 0/8 and BLS 5/8 are period-recovery
  counts, not new equal-FPR detection rates. The template-family SNR remains
  substantial, while the actual gate and ranking causes are unresolved.
  This analysis was added after freezing without changing the experiment.
  Its compact artifacts and input hashes are included; the original NPZs
  remain outside git. [Integration verification](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/grazing-development-diagnosis/integration.json")
  checks every original artifact and all 115 sealed local files.
  The [diagnostic figure](grazing-development-diagnosis/figure/grazing-depths.png)
  separates physical depths from the window means used by the gate; its
  [PDF](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/grazing-development-diagnosis/figure/grazing-depths.pdf"),
  [SVG](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/grazing-development-diagnosis/figure/grazing-depths.svg"), and
  [source/data receipt](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/grazing-development-diagnosis/figure/grazing-depths.receipt.json")
  preserve the same development-only scope. This figure does not replace
  the pending sustained-throughput figure.

The BLS response receipt calls its finest configuration `bls_convergence`; its
parameters equal the later blind-search name `bls_strong`. It runs at 78 of 80
known true periods. The full blind grid makes that configuration inapplicable
to all eight separated-TESS development inputs because other trial periods
exceed its shared-memory limit. Those cases still test the three applicable
resolutions. Known-period applicability cannot substitute for full-grid
applicability or justify removing trial periods using the injected truth.

The numerical target is the full observation-level GTLS-compatible default,
including its complete candidate/harmonic refinement. The currently implemented
changes remove host sorting work, bound temporary duration-group allocation,
combine winner transfers, and skip an unused refinement calculation. A guarded
short-row path batches the installed CUB single-tile scan agent while preserving
its floating-point addition tree; unsupported builds and longer rows retain the
native graphs. Its startup canary checks bitwise parity, including subnormals.
No approximate screen is added. Component and development timings are not yet
evidence of sustained production throughput.

The final numerical code passed **342 TLS GPU tests**. The host suite passed
**762 tests**, with 18 skips and one expected failure. The preceding two host
failures exposed the missing declaration of the newly packaged kernel in the
inventory test; both the [failed run](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/host-tests-final.log") and
[corrected run](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/host-tests-final-inventory-fixed.log") are retained.
The scientific and reporting harness passed **130 CPU tests** after the final
launch integration fixes. The [test receipt](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/survey-host-tests-integration-final.json")
identifies the tested Python sources and the
[complete log](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/survey-host-tests-integration-final.log"). A separate
[operations suite](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/ops-host-tests-integration-final.json") passed **45 tests**,
covering orchestration, archive collection and the guarded development probe.
The [integration review](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/integration-review-final.json") records the corrected
output paths, design identities, report/figure artifact checks and timing-source
checks. These are harness checks; they do not supply missing science results.
A later [wording clarification](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/target-snr-label-20260911.json"), checked with
the 19 renderer tests, distinguishes assigned target SNR from realized SNR.
Unsampled injections can realize zero and remain in their original target
groups; the grouping rules and scientific calculations are unchanged.

The [final development baseline comparison](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/development-promoted-baseline-parity.json")
matched **79/80** complete stored TLS fingerprints. All 32 cases using the new
short-row scan matched. HATpi development case 0001, which uses the long-row
fallback, changed its chi-squared hash and SDE by about −0.00000334; its selected
period, finite mask and period-recovery flag matched. This is a retained
numerical discrepancy, not aggregate bitwise qualification. Its cause is not
assigned from the fallback status alone. A [separate repeat diagnostic](../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/hatpi-repeat-diagnostic-summary.json")
completed 24 calls: eight baseline calls in single-worker processes, eight
candidate calls in single-worker processes and eight candidate calls with four
workers. All matched the original baseline, including complete public and
captured internal outputs; none reproduced the original discrepancy. This
finite quiet probe leaves its cause unresolved and does not replace 79/80 with
80/80. The planned held-out comparison checks
all 5,120 injection/test-null outcomes and both frozen threshold decisions.
Those baseline decisions reuse the candidate TLS cuts; there is no separate
baseline calibration pass. TLS and selected BLS share the same calibration
inputs and each receives its own threshold. That calibration bank is independent
of development and the later test-null bank.

The [literature audit](../../../docs/TLS_LITERATURE.md) and existing known-period
template-response diagnostics answer different questions. The diagnostics do not measure TLS's
blind-search advantage. The original TLS publication reports a substantial
recovery advantage on its Kepler-like population; neither that result nor the
small development template-response differences establish the outcome for the
current GTLS-compatible TESS/ZTF implementation. The present study calibrates
each detector separately and reports each physical regime.

A [dated literature addendum](LITERATURE_ADDENDUM.md) records a further
published Kepler population comparison and its methodological limits. It
does not amend the frozen campaign.

Final claims will distinguish exact implementation qualification from finite
population evidence. Sparse sampling, template mismatch, shared native float32
scan variability, and unachievable false-positive targets caused by discrete
scores remain explicit limitations.

