# Timing sources and reporting provenance

`execution-sources.tar.gz` preserves the exact measurement driver, tools, tests
and protocol executed for this campaign. Its SHA-256 and member inventory are
in `execution-sources.json`. `protocol.json` is the unchanged full measurement
declaration. The recorded hardware and dependency files are in `environment/`;
`hardware-timing02-allocation.json` and `hardware.json` identify the successful allocation.

The original all-configuration campaign failed because one optional GTLS pool
ran out of memory during warmup. `reporting-protocol.json` explicitly records
the subsequent post hoc scope. `report_completed.py` replays the exact original
audit, accounts for every attempt, checks all original prerequisites, adds full
returned-object repeatability and recomputes medians and pool selection. Its
42 CPU tests and their receipt are retained. The derived figure input remains
bound to the separate root `reporting_acceptance.json`; the failed original
acceptance is never replaced.

`timing-origin-records.tar.gz` contains the exact accepted origin manifests and
compact records for the 48 timing inputs. Numerical vectors are restored from
the shared public input bank, avoiding a second input archive.
`collection-support.tar.gz` contains final collection receipts and supporting
snapshot files; the 43 original timing files live in `timing/`. Each archive
has a corresponding member/hash inventory. Previous preflight failures remain
in `failed-attempts/`, with no headline timing denominator.

## Recompute the report with CPU tools only

The following workflow was tested using only files in this repository. It
restores the original inputs, reconstructs the verified collection layout, and
runs the reporting assessor. It performs no search, GPU or cloud work. Use
NumPy and a new scratch directory from the repository root:

```sh
python - <<'PY'
from pathlib import Path
import json
import shutil
import tarfile
from benchmarks.tls_reference import inputs

published = Path('benchmarks/results/tls_reference_2026-09-10')
scratch = Path('tls-report-replay')
scratch.mkdir()

def unpack(name, target):
    target.mkdir()
    with tarfile.open(published / 'sources/timing' / name) as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError(member.name)
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())

unpack('collection-support.tar.gz', scratch / 'checkpoint')
unpack('timing-origin-records.tar.gz', scratch / 'origin')
index = json.loads((scratch / 'checkpoint/file-index.json').read_text())
prefix = 'timing-continuation/results/timing/'
for name, entry in index.items():
    if name.startswith(prefix):
        source = published / 'timing' / name[len(prefix):]
        if inputs.sha(source) != entry['sha256']:
            raise ValueError(name)
        destination = scratch / 'checkpoint/files' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

for study, manifest in [('main', 'validation/input_manifest.json'),
                        ('supplement', 'supplement/input_manifest.json')]:
    inputs.restore_bank(published / 'inputs', study, published / manifest,
                        scratch / ('restored-' + study))
manifest = scratch / 'origin/diagnostic/results/timing_manifest.json'
for case in json.loads(manifest.read_text())['cases']:
    source = scratch / ('restored-' + case['study_id']) / case['file']
    if inputs.sha(source) != case['sha256']:
        raise ValueError('Original input container differs: ' + case['file'])
    shutil.copy2(source, manifest.parent / case['file'])
PY

python benchmarks/tls_reference/timing/report_completed.py \
  --checkpoint tls-report-replay/checkpoint \
  --sources benchmarks/results/tls_reference_2026-09-10/sources/timing \
  --manifest tls-report-replay/origin/diagnostic/results/timing_manifest.json \
  --output tls-report-replay/assessed
```

The resulting `timing_analysis.json` must have SHA-256
`d428e8aadc337678db1112c6a0adf5de1cdcc2c4a7b4fc8283ce1900f607bb1b`.
The reporting receipt has a fresh timestamp; its checked content is otherwise
identical. [public-replay-proof.json](public-replay-proof.json) records the
successful replay. To execute new measurements instead, follow the
[maintained timing workflow](../../../../tls_reference/timing/README.md).
