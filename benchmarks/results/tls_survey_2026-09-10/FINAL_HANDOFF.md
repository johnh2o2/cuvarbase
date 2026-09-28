# Final collection and delivery handoff

Prepared on 2026-09-11 from the current collector, sidecar, renderer and reviewed
**v2** plan. This document is an unsealed operational note. It does not amend the
historical prospective runbooks or authorize another run. At preparation, both
final local archives were **pending**; the primary workflow was in accuracy and
the armed v2 sidecar was waiting for primary completion. Check current receipts
before proceeding. Do not start a collector, sidecar, timing runner or GPU job
from this runbook.

Known local roots:

```sh
TLS_STUDY_WORK=/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910
TLS_PRIMARY="$TLS_STUDY_WORK/collected"
TLS_SUPPLEMENT="$TLS_STUDY_WORK/bls-execution-supplement-run-v2"
TLS_EXTRACTED="$TLS_SUPPLEMENT/extracted"
TLS_DELIVERY="$TLS_STUDY_WORK/final-delivery"
df -h "$TLS_STUDY_WORK"
```

The active plan is `ops/bls-supplement-sidecar-plan-v2.json`, SHA256
`683f8a373065cbb96b5afb7ea4d39d4f918f029d137e6fb7744e010492c19a03`.
The prospective seal is `evidence/bls-execution-supplement/seal-v2.json`, SHA256
`20972e579f99e4bb97c45adc7ccdb138be636fc951d57682d675241f132942aa`.
Its remote mutable output is
`/workspace/tls-survey/supplementary/bls-execution-supplement-run-v2`.
The primary measurement is `evidence/throughput-final/campaign.json`, **not**
`evidence/throughput-measure-final/campaign.json`.

**1. Verify collected bytes and execution status separately.**

The original collector writes `collection-state.json`,
`completion-bundle-receipt.json`, `completion-bundle.tar` and `collected/` under
the work root. The sidecar writes its local state and `bundle.tar` under
`bls-execution-supplement-run-v2/`; its archive receipt is the local state's
`receipt` field. It verifies that archive but does **not** extract it.

This local-only check streams both archives and checks every inventoried member.
Importing the sidecar with `runpy` does not run its lifecycle entry point; only
its read-only `verify_archive` function is called. Missing files are not verified
collection; check controller status to distinguish pending work from a failed
collection. A failed assertion needs review, not a replay of the experiment.

```sh
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, runpy, tarfile
w = Path('/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910')
s = w/'bls-execution-supplement-run-v2'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
required = [w/'collection-state.json', s/'sidecar-state.json',
            w/'completion-bundle-receipt.json', w/'completion-bundle.tar',
            w/'collected/completion/inventory.json', s/'bundle.tar']
missing = [str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('NOT COLLECTED; inspect controller status: ' + ', '.join(missing))
primary = json.loads((w/'collection-state.json').read_text())
side = json.loads((s/'sidecar-state.json').read_text())
r = json.loads((w/'completion-bundle-receipt.json').read_text())
assert primary['evidence_verified'] and side['evidence_verified']
assert primary['receipt'] == r and primary['archive_sha256'] == r['sha256']
assert r['reviewed_seal_sha256'] == '1b81c75bd1a498c0dbed607e3221da1f374fc05be765de6dd2670c8d2f2b0807'
assert r['reviewed_exactness_plan_sha256'] == '317177812ddcb9683afadc34c7112e133d85f2c746b50e8ae3256773e0b8c525'
assert sha(w/'collected/completion/inventory.json') == r['inventory_sha256']
if r['archive_mode'] == 'verified_banks':
    assert sha(w/'collected/completion/reviewed-science-seal.json') == r['reviewed_seal_sha256']
    assert sha(w/'collected/completion/reviewed-exactness-plan.json') == r['reviewed_exactness_plan_sha256']
else:
    assert r['archive_mode'] == 'raw_rescue' and r['outcome'] == 'failed_integrity_rescue'
assert (w/'completion-bundle.tar').stat().st_size == r['bytes']
assert sha(w/'completion-bundle.tar') == r['sha256']
def safe_name(name):
    return (bool(name) and not PurePosixPath(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')) and
            chr(92) not in name)
with tarfile.open(w/'completion-bundle.tar', 'r') as archive:
    members = archive.getmembers()
    names = [m.name for m in members]
    assert len(names) == len(set(names))
    assert all(m.isfile() and safe_name(m.name) for m in members)
    raw = archive.extractfile('completion/inventory.json').read()
    assert hashlib.sha256(raw).hexdigest() == r['inventory_sha256']
    inventory = json.loads(raw)
    assert set(names) == set(inventory['files']) | {'completion/inventory.json'}
    for name, expected in inventory['files'].items():
        h, size = hashlib.sha256(), 0
        with archive.extractfile(name) as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block); size += len(block)
        assert (size, h.hexdigest()) == (expected['bytes'], expected['sha256'])
        assert sha(w/'collected'/name) == expected['sha256']
plan_sha = sha(w/'ops/bls-supplement-sidecar-plan-v2.json')
assert plan_sha == '683f8a373065cbb96b5afb7ea4d39d4f918f029d137e6fb7744e010492c19a03'
plan = json.loads((w/'ops/bls-supplement-sidecar-plan-v2.json').read_text())
assert sha(w/'ops/bls_supplement_sidecar.py') == plan['remote_pins']['sidecar']['sha256']
assert sha(w/'ops/cloud.py') == plan['local_pins']['cloud']['sha256']
verifier = runpy.run_path(str(w/'ops/bls_supplement_sidecar.py'))
verified = verifier['verify_archive'](s/'bundle.tar', side['receipt'], plan_sha)
print('Primary:', primary['status'], r['outcome'], r['archive_mode'])
print('Supplement:', side['status'], verified['outcome'], side['receipt']['archive_mode'])
print('Collector handback:', side.get('collector_handed_back'))
print('Archive byte identities verified; numerical qualification is a separate result.')
PY
```

For a completed scientific delivery, the primary receipt must have
`outcome=complete` and `archive_mode=verified_banks`. `raw_rescue` /
`failed_integrity_rescue` preserves evidence only and may lack the reviewed
design files; the check above grants only byte verification in that mode. The supplement can also
preserve `failed_or_partial` outcomes: archive verification or collector
handback does not make its tuning or measurements complete. Keep those labels.

**2. Access supplementary evidence without restarting anything.**

After step 1, extract regular members into a fresh local directory. This refuses
an existing destination and uses neither `extractall` nor archived links. The
JSONs retain their original remote path strings; do not rewrite them or create
a `/workspace` mirror. The existing combined renderer accepts relocated files.

```sh
python3 - <<'PY'
from pathlib import Path
import json, runpy, shutil, tarfile
w = Path('/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910')
s = w/'bls-execution-supplement-run-v2'
v = runpy.run_path(str(w/'ops/bls_supplement_sidecar.py'))
state = json.loads((s/'sidecar-state.json').read_text())
plan_sha = v['sha'](w/'ops/bls-supplement-sidecar-plan-v2.json')
assert plan_sha == '683f8a373065cbb96b5afb7ea4d39d4f918f029d137e6fb7744e010492c19a03'
v['verify_archive'](s/'bundle.tar', state['receipt'], plan_sha)
destination = s/'extracted'
assert not s.is_symlink() and not destination.exists()
destination.mkdir()
with tarfile.open(s/'bundle.tar', 'r') as archive:
    for member in archive.getmembers():
        assert member.isfile() and v['safe_member'](member.name)
        target = destination/member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
print(destination)
PY
```

Inspect `extracted/state-at-archive.json`, `packaging-warnings.json`,
`tune/campaign.json`, `tune/tuning-seal.json` and `measure/campaign.json` for
execution status. For the combined renderer, require supplement receipt
`outcome=complete`, `archive_mode=verified_designs`, empty packaging warnings,
and both stages `complete`. Otherwise retain the existing qualified figure and
the supplemental failure evidence.
Reference-only configurations are diagnostics, not final timing bars. Native
BLS bars require valid ownership/accounting and three completed queues per
panel, each with at least 96 attempts and 120 seconds. API failures reduce the
successful-completion rate. The original BLS numerical qualification remains
**failed**, even when later outputs agree. Missing or interrupted products
remain unavailable; do not rerun them to obtain a figure.

**3. Render the separate combined figure from collected products.**

Run only after the required completed receipts exist. This is CPU rendering,
using the source snapshot preserved by the supplementary archive. It checks
the two supplement seal layers, science identity, separate tuning choice,
original cohorts/resources, queue accounting and full planned TLS exactness.
It does not recalibrate scores or infer detection equivalence. Renderer
rejection is a withheld figure, not permission to change a gate.

```sh
"$TLS_STUDY_WORK/local-env/bin/python" \
  "$TLS_EXTRACTED/source-snapshots/candidate/benchmarks/tls_survey/plot_native_bls_comparison.py" \
  --primary "$TLS_PRIMARY/evidence/throughput-final/campaign.json" \
  --native-bls "$TLS_EXTRACTED/measure/campaign.json" \
  --supplement-seal "$TLS_EXTRACTED/reviewed-supplement-seal.json" \
  --supplement-binding "$TLS_EXTRACTED/binding.json" \
  --native-tuning "$TLS_EXTRACTED/tune/tuning-seal.json" \
  --science-seal "$TLS_PRIMARY/completion/reviewed-science-seal.json" \
  --exactness "$TLS_PRIMARY/evidence/exactness-final.json" \
  --output "$TLS_DELIVERY/survey-throughput-with-native-bls"
```

The output prefix produces `.png`, `.pdf`, `.svg`, `.csv` and `.data.json`.
Keep the original `collected/evidence/survey-throughput.*` unchanged, including
its missing qualified BLS bars. In the new figure, native BLS uses separate
hatched execution bars and a permanent failed-repeatability label.

**4. Assemble the final report and its supporting tables.**

Check every `outputs` entry in each report/figure provenance JSON against the
sibling file's SHA256 before copying. Preserve filenames and relative layout.
Publish compact final products under this study directory in new `final-report/`,
`final-science/`, `final-timing/` and `final-figures/` subdirectories; keep complete
archives, input banks, spectra and attempt journals in the work root.

| Collected source | Required delivery |
| --- | --- |
| `collected/final-campaign/report/` | `RECOVERY.md`, `recovery_fpr.csv`, `paired_contrasts.csv`, `thresholds.csv`, `subgroups.csv`, `exactness.csv`, `exactness_mismatches.csv`, `snr_descriptive.csv`, `snr_cases.csv`, `provenance.json` |
| `collected/final-campaign/` | `detection-results.json`, `thresholds.json` |
| `collected/evidence/` | `exactness-final.json`, `heldout-snr-final.json`, original `survey-throughput.{png,pdf,svg,csv,data.json}` |
| `collected/evidence/throughput-final/` | `campaign.json`, `measurements.csv`, and every referenced result JSON at its existing relative path; retain the full archive for arrays and qualification logs |
| `bls-execution-supplement-run-v2/extracted/` | Binding, reviewed seal/plan, separate tuning seal/campaign, measure campaign and referenced result JSONs, inventory and packaging warnings; retain references and failed configurations |
| `final-delivery/` | New `survey-throughput-with-native-bls.{png,pdf,svg,csv,data.json}` when eligible |

Link the two archive receipts and hashes, v2 seal/plan, source identities and
final ledger from the study README. Label any copied partial evidence explicitly.
Update the README's pending statements only after the corresponding products
verify. Scientific conclusions must use per-regime recovery and realized FPR
with their existing intervals, paired contrasts and sampling/SNR subgroups.
Report TLS held-out exactness as its actual planned count and
`exactness_qualified` value; successful execution is not a passing exactness
result. Preserve discrete-threshold limits, failed executions and unsupported
physics. Assigned target-SNR groups include unsampled injections; white-family
ceilings and OU responses of those white-selected filters are distinct.

**5. Close provider termination and the cumulative ledger.**

The sidecar must have verified its evidence before collector handback. The
original collector performs termination; this runbook does not. Read
`collection-state.json` for `phase=complete`, its `termination` receipt and
guard result, then `nodes/survey01/pod.json` for `termination_verified`, final
timestamps and estimated rental cost. A stopped process or empty GPU alone
does not establish provider termination. This fresh provider query is read-only:

```sh
python3 - <<'PY'
from pathlib import Path
import json, runpy
w = Path('/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910')
cloud = runpy.run_path(str(w/'ops/cloud.py'))
node = json.loads((w/'nodes/survey01/pod.json').read_text())
assert node['termination_verified']
pods = cloud['api']('query {myself {pods {id}}}')['myself']['pods']
assert node['id'] not in {p['id'] for p in pods}
print('Provider confirms the owned rental is absent.')
PY
python3 "$TLS_STUDY_WORK/ops/cloud.py" spend
```

Reconcile `authorization.json`, the prior ledger it identifies,
`budget-amendment-30.json` and all final `nodes/*/pod.json` estimates. Include
recorded storage charges and collection time; do not add projected workload
costs or the supplement's one-hour cap a second time. The user ceiling is
**$100 cumulative**, not another allowance. Record the final cumulative
estimate and provider-confirmed termination separately from invoice evidence.
If collection, termination or a required result remains pending, report that
specific remaining item rather than declaring the study complete.
