# Separate native-BLS supplement sidecar

Implementation: `bls_supplement_sidecar.py`. This is a new private operational
wrapper. It never edits or invokes the frozen primary controller, changes its
scientific settings, provisions a rental, or directly terminates a provider
resource. The original `collect_on_complete.py` retains teardown ownership.

No sidecar has been armed. The original collector, primary controller, active
science workers, rental guard and wake lock are untouched by this implementation.

## Review and immutable inputs

Root first reviews the finished supplemental protocol, runner, renderer and
sidecar sources. The prospective supplement seal has schema1 and kind
`native_bls_execution_supplement`, the original science/auxiliary identities,
all executed `remote_files` and lifecycle `local_files` hashes, and budget
`{"gpu_cap_seconds":3600,"cleanup_reserve_seconds":120}`. Include the existing
original tuning receipt in `remote_files`:

`/workspace/tls-survey/evidence/throughput-tune-final/campaign.json`

SHA256: `8f32c398b0e6b675b41e89474ff3ca5b3c381ee83e79b96f38d3fd9045de273b`.

Its `binding_rule` names these existing/future primary paths:

- `primary_tuning_path`: `/workspace/tls-survey/evidence/throughput-tune-final/campaign.json`
- `primary_measurement_path`: `/workspace/tls-survey/evidence/throughput-final/campaign.json`
- `primary_state_path`: `/workspace/tls-survey/completion/state.json`
- `primary_bundle_receipt_path`: `/workspace/tls-survey/completion/bundle.json`

The future measurement is never dynamically resealed. A separate binding
receipt records its actual bytes and the original configuration/cohort/resource
identities mechanically, after successful primary completion. Both stage
commands carry the prospective seal SHA and this binding SHA. Missing required
primary results/cohort/resource fields fail preparation; no rows are dropped.

`bls-supplement-sidecar-plan.template.json` provides concrete commands and paths.
It deliberately contains unarmable placeholders. Before reviewing the final
plan SHA, fill the finished source/price/seal hashes, absolute primary-wait
cutoff, and verify the recorded original collector and guard process identities.
The collector PID/start-time/command identities were read without signals.
The template keeps the original collector command and uses `resume_stopped`.
Root may instead explicitly choose `start_absent` only if the original process
has already exited; the sidecar never kills it.

## Conditional launch and ownership

Only root pauses the original collector after reviewing the replacement. The
local sidecar refuses to start unless that exact collector is stopped (or absent
under the explicit alternative), its collection state has not advanced toward
teardown, the reviewed rental is active, and the original budget guard remains
active. It rechecks those conditions while waiting and before handback.

After root has reviewed the final plan and safely paused the collector, the
new operational command is:

```sh
python3 WORK/ops/bls_supplement_sidecar.py local \
  --plan WORK/ops/bls-supplement-sidecar-plan.json \
  --plan-sha256 REVIEWED_PLAN_SHA256 --arm
```

Run that local command in the same kind of persistent process root used for the
original collector. The sidecar uploads only its own identical source and its
reviewed plan; root must have already uploaded the other prospectively sealed
supplemental files. No direct GPU command is required.

The remote sidecar waits for primary `archive_ready`, `outcome=complete`,
`archive_mode=verified_banks`, all seven successful execution stages, matching
science/auxiliary/controller identities and archive receipt, primary controller
exit, absence of all primary-owner descendants, and an empty GPU. An exactness
qualification failure does not masquerade as an execution failure. Primary
execution failure skips the supplement and preserves the failed preparation.

The one-hour clock starts before post-primary supplemental binding/preparation.
It covers primary-archive byte verification, setup, tuning and measurement.
Both stages share the same hard epoch deadline; the sidecar additionally uses
monotonic time. It reserves the final120seconds for cleanup. It does not start a
stage whose fixed minimum runtime cannot fit before that reserve. The runner
also guards every queue, preserves partial attempt journals and closes workers.

Supplemental children inherit only the distinct
`CUVARBASE_SURVEY_BLS_SUPPLEMENT_OWNER` marker. The sidecar checks exact marker,
PID, Linux process start ticks and actual GPU process ownership. Cleanup signals
only those identified descendants, including independently sessioned workers.
Foreign GPU work invalidates the supplement and is never signaled.

## Evidence and restart behavior

All mutable remote output is under
`/workspace/tls-survey/supplementary/bls-execution-supplement-run-v2`.
This directory is outside every input tree recursively archived by the frozen
primary controller. Stage outputs, binding, heartbeats, locks and logs all stay
there. The static prospective seal may remain in `evidence`; it never changes.
Validation rejects mutable output under primary archive trees and rejects stage
arguments that point back into those trees.
Every result, partial journal, spectrum, driver log, binding, reviewed design and
source snapshot is archived. The original reviewed designs are retained before
work and kept separate from changed actual-source snapshots. All sources named
by the prospective seal are retained, including the renderer and imported
frozen dependencies.

Archives contain an exact file inventory. Local verification checks full tar
bytes, exact member coverage, every file hash, the actual reviewed plan against
its externally approved SHA, and the actual retained supplement seal against
that plan's pin. If the reviewed seal is unavailable, only explicitly failed
raw rescue is allowed. All supplement archive validation is labeled byte-only;
none grants numerical or detection qualification.

Only after complete local byte verification does the sidecar resume the exact
original stopped collector with `SIGCONT`, or start its original command under
the reviewed absent-process alternative. The original collector independently
verifies the primary bundle and performs provider termination. The sidecar does
not stop the budget guard or wake lock.

Persistent local/remote locks prevent duplicate controllers. A remote restart
with any persisted launch intent or preparation start preserves an interrupted
attempt and packages its partials; it never repeats GPU work or resets the hour.
Local transfer retries resume collection only. A handback intent prevents a
second collector launch after an ambiguous interruption. Already downloaded
bytes can still be verified after independent guard termination; such rental
loss is reported explicitly and does not become a successful primary collection.

Offline verification:

```sh
WORK/local-env/bin/python -m unittest discover -s WORK/ops \
  -p test_bls_supplement_sidecar.py -v
```

Twenty-five bounded tests currently pass, including ownership gates, source
and design tampering, startup owner reload, interrupted preparation/launch,
partial archives, interrupted transfers, and evidence-before-handback ordering.
