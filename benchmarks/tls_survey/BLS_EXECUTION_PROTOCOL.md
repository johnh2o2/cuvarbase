# Prospective native-BLS execution-throughput supplement

This is a separate timing study. It does not amend the original
[throughput protocol](THROUGHPUT_PROTOCOL.md), its exact selected-score gate,
its failed results, its frozen selections, or its missing qualified BLS bars.
The original BLS one-worker pilot changed the selected likelihood score on
gapped-TESS development case 0006, as recorded in the
[retained exclusion audit](../results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/AUDIT.md).
The original BLS exact-repeatability qualification remains **failed**, even if
every subsequent supplemental output happens to match.

The supplement measures how many unchanged native BLS searches execute per
second and records their numerical variability. It introduces no numerical
passing tolerance. A changed score or period is recorded as a discrepancy;
it is never relabeled as passing. Neither a small discrepancy nor an unchanged
period establishes unchanged calibrated detection. TLS's scientific settings,
zero approximation allowances, calibration, and held-out analysis are unchanged.

## Freeze and execution boundary

Approve this protocol and the supplemental implementation, renderer, and
sidecar sources now, while the primary pipeline runs. Create an immutable
supplemental seal at `evidence/bls-execution-supplement/seal.json` at that review,
before any supplemental search.
Record its SHA256 in every supplemental campaign and report. The seal binds:

- This protocol and every executed supplemental source and imported scientific
  or timing dependency by SHA256.
- The unchanged scientific seal, original development-tuning receipt, and
  exact original development cohort by SHA256.
- A mechanical post-primary binding rule for the completed primary measurement,
  its verified archive/inventory, resource receipts, and every fixed measurement
  cohort, including original input and varied-cohort retained-index hashes.
- The tuning, measurement, accounting, discrepancy, allocation, and budget
  rules below, including the permanently failed original qualification.

The completed primary measurement hash cannot be known beforehand. After
verified primary completion, create a separate immutable binding receipt
containing its actual SHA256 and the cohort/resource identities required by
the sealed rule. Bind that receipt's SHA256 in every supplemental campaign and
report. This step copies and verifies identities mechanically; it cannot change
algorithms, gates, settings, sources, cohort definitions, or case selection in
response to primary outcomes. Root approval of the prospective seal authorizes
this later mechanical binding. Do not overwrite either receipt. Any inconsistency
requires a preserved failed preparation receipt and root review before work.

The supplement starts only after the primary pipeline has completed
successfully and its verified archive is available. A rescue archive or a
terminal parent process alone does not establish successful completion.
Verify that primary workers and their descendant GPU contexts have exited.
The original collector must not terminate the rental during this separate
authorized study. The sidecar retains supplementary artifacts on success,
failure, or timeout, then returns control to the original collection and
termination procedure. This protocol grants no provisioning authority.

## Fixed searches, cohorts, and allocation

Call the existing native BLS search and the science seal's selected ranker for
each regime. Preserve the complete supplied period grids, duration/phase
settings, errors, source arrays, and normal compact timed output policy.
Do not change numerical kernels, determinize or round outputs, replace the
selected ranker, shorten grids, screen cases, or alter a scientific source.
Full power/period arrays, masks, and all ranker diagnostics are collected
outside measured queues, using the existing full diagnostic call.

Use exactly the original 24 development inputs for tuning: the first eight
manifest-order inputs from TESS solar, gapped TESS, and ZTF solar. Measurement
uses exactly the original independent timing populations: the first 16 nulls
per cadence for three panels, plus the existing 96-input varied-size cohort
derived from the first 32 nulls per cadence by the original deterministic
retained-fraction/index rule. Bind the primary measurement's actual manifests
and retained indices. Do not replace a failed case or select a subset according
to timing, numerical stability, detection, or API success.

Run only the supplemental BLS processes on the same single GPU and container
CPU/memory allocation used by the primary comparison. Preserve one numerical
library thread per worker. Record GPU UUID, CPU quota, memory limit, source
and input hashes, process identities, and context ownership. Input generation
and other scientific work on the rental must have finished. An ownership or
allocation violation invalidates the affected measurement; it cannot become
a fast execution result.

## Independent development tuning

Use the original staged five-configuration policy: one, two, and four persistent
workers at batch size one; then batch sizes four and eight at the selected
worker count. Task batches group serial calls within a worker and never mix
different grids or search options. Each pilot attempts at least 24 sources and
runs for at least 30 seconds, completing whole fixed-cohort cycles, once.

Among operationally valid, complete pilots with positive successful-completion
rates, select the worker count with the highest successful lightcurves/second.
Then choose among its batch sizes one, four, and eight by the same metric.
Exact speed ties prefer fewer workers, then smaller batches. API failures
consume time and reduce the success numerator. Always display failure counts
and completion fractions beside rates; a selected setting with API failures
does not establish successful processing of the entire workload.

A numerical mismatch never removes a timing result or chooses an alternative
reference. There are no retries until a numerical gate passes. If no worker
setting has an operationally valid positive rate, report that no execution
configuration was selected and leave subsequent panels unavailable. Freeze the
winner and the original campaign's unchanged hourly price before supplemental
measurement. Record the separate development tuning seal's path and SHA256;
measurement and rendering must verify its campaign, selection, and artifacts.
No measurement input or held-out
detection result may influence that selection.

## Sustained queues and accounting

For each of the four fixed panels, execute three queues with the frozen winner.
Each queue attempts at least 96 lightcurves and lasts at least 120 seconds,
finishing whole cohort cycles with at most one task per worker in flight.
The minimum is **attempted inputs**, not successful outputs: API failures must
not cause retries until 96 successes occur. Report the actual number of
successes, which can be below 96. Cycle scheduling and stopping depend only on
the fixed input order, attempted count, and wall time, never on numerical
agreement, recovery, or successful output count. Repetitions reuse inputs and
are not additional independent astrophysical samples.

Every assigned case receives one attempt during its scheduled cycle. Catch
ordinary native API errors per case, retain the error, and attempt the remaining
scheduled members once. Do not skip the rest of a batch because one case failed.
A successful completion must return the selected period and score required by
the unchanged adapter in a valid finite form. A backend exception or invalid
selected output is an API/output failure. A changed finite value is a completed
search and a numerical discrepancy, not an API failure.

For every complete operationally valid queue, require:

- `attempts = successful_completions + API_or_output_failures`, with one retained
  record per attempted case and exact scheduled cohort membership.
- Elapsed time covers all attempts, dispatch, required input validation,
  preparation, transfers, full search, selected ranking, output handling,
  error handling, and worker completion. Failed attempts consume that time.
  The supplement also flushes per-attempt worker journals and a parent task
  journal, and compares selected endpoints inside this clock. This additional
  instrumentation cost is retained and disclosed beside cross-method rates.
- Report attempted and successful lightcurves/second, completion fraction,
  error counts and case identities, plus costs per attempt and per successful
  lightcurve. No-success cost per successful lightcurve is unavailable.
- Keep startup, imports/context creation, grid construction, first-call latency,
  complete diagnostic warmup, and teardown separately. Match the original
  cold-amortization convention: charge input/setup time, worker startup, and
  complete diagnostic warmup in addition to the queues. Separately report total
  configuration time and cost, including post-queue diagnostics and teardown;
  do not label either convention as including costs it omits. Sample device
  memory, worker RSS, and container memory with the original accounting and
  limitations.

Missing, duplicated, foreign, or misidentified results; corrupted source/input
or reference hashes; worker loss; execution timeout; failed instrumentation;
or a resource-ownership/allocation violation invalidate the setting or panel.
Retain all partial records and elapsed time, but do not select it or publish a
complete sustained rate. An incomplete queue is never represented as meeting
the planned duration or attempted-input requirement.

## Numerical discrepancies remain visible

The first scheduled one-worker full outputs on the development cohort are
fixed comparison anchors for all tuning configurations. Do not replace an
unavailable anchor with the first later success. On each independent panel,
take one scheduled full one-worker reference before the selected pool runs;
an unavailable reference stays unavailable. An anchor is a comparison point,
not an accepted numerical truth or passing criterion.

Collect complete outputs before and after queues on every worker, and compare
every measured selected period/score with its assigned anchor. Retain both
values and exact differences for each changed selected endpoint, together with
input, worker, configuration, repetition, and attempt identities. Preserve
full diagnostic arrays and mask/period fingerprints and describe nonwinning
power and other-ranker differences separately. Unavailable comparisons and
API failures must be explicit. Do not suppress zero-difference repeats or
replace an original disagreement with a matching repeat.

Every supplemental record and renderer must carry
`original_qualification_passed = false`. No numerical discrepancy threshold,
new numerical passing label, or inferred SNR/recovery tolerance is introduced.
The supplement does not re-evaluate calibrated false-positive or recovery
performance, and does not imply independently recalibrated baseline cuts.

## Time limit, preservation, and reporting

The sidecar enforces one absolute **3,600-second supplemental execution window**
starting before worker startup. At $0.49/hour this allows approximately $0.49
of additional rental compute, within the already authorized cumulative budget;
the live ledger and remaining guard must be checked before release. No new
allowance or provisioning is authorized. Reserve the final 120 seconds for
worker shutdown and ownership verification. Do not begin another required
queue or stage if its known minimum duration cannot fit before that reserve.
Never shorten a planned queue, reduce repetitions, alter settings, or omit cases
to fit the deadline. Stop and retain incomplete work when the budget is reached.
All GPU workers must exit by the absolute deadline; failure to quiesce is a
resource-safety failure, not a completed benchmark. Archive/transfer costs are
recorded separately and remain subject to the existing rental guard.

Preserve the original qualified figure unchanged. A separate combined figure
may show supplemental native-BLS execution rates as hatched bars, with a
permanent **failed exact-repeatability qualification** label and separate
source/provenance binding. Show medians and observed ranges from the three
complete repetitions, completion fractions, and all unavailable panels.
These bars are not qualified results under the original numerical protocol.
Do not silently merge them into the original qualified CSV or label their
execution rates as proof of equivalent scientific outputs. Use matched cohort
and resource receipts for any cross-method comparison, and keep numerical
qualification, execution completion, and calibrated sensitivity distinct.

Every campaign and rendered artifact binds the supplemental seal, post-primary
binding receipt, scientific seal, primary tuning, primary measurement, exact
cohort identities, and actual resource receipts. Preserve all supplementary
attempts, discrepancies, errors,
partial runs, manifests, arrays, source snapshots, and budget/ownership records
before the original collector resumes and the rental terminates.
