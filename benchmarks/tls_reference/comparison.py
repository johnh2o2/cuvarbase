"""Serial reference-correction validation, without cloud or resource actions."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

import corrected_reference
import validate as harness


def run_case(path, repo_root, case_root, *, seal=None, options=None, gates=None, thresholds=(8.,), replay=False):
    """Run literal native, corrected native, then the actual public candidate.

    Corrected execution may reuse literal arrays only after a source-pinned
    complete selection trace proves the host correction is a no-op.
    """
    case_root.mkdir()
    outcomes = {}
    reuse = None
    for backend in ('gtls', 'gtls_corrected', 'candidate'):
        if backend == 'gtls_corrected' and outcomes.get('gtls') == 'ok':
            native_path = case_root/'gtls/record.json'
            literal = json.loads(native_path.read_text())
            with np.load(native_path.parent/literal['arrays_file'], allow_pickle=False) as data:
                reuse = corrected_reference.no_op_receipt(data, literal['result'])
            harness.write(case_root/'correction_trace.json', reuse)
            if reuse['proved_no_op']:
                from gputls import core
                destination = case_root/backend
                destination.mkdir()
                shutil.copy2(native_path.parent/literal['arrays_file'], destination/literal['arrays_file'])
                copied = dict(literal, backend=backend, elapsed_seconds=0.,
                    reused_execution_elapsed_seconds=literal['elapsed_seconds'],
                    reference_reuse=dict(literal_record_sha256=harness.sha(native_path), **reuse),
                    result=dict(literal['result'], reference_correction=corrected_reference.identity(core)))
                harness.write(destination/'record.json', copied)
                outcomes[backend] = 'ok'
                continue
        try:
            harness.run(SimpleNamespace(case=path, backend=backend, mode='full',
                engine_root=repo_root, engine_kind='public', positive_origin=True, auto_grid=False,
                reference_record=case_root/'gtls_corrected/record.json' if backend == 'candidate' else None,
                options=options, work_chunk=256, chunk_policy='default', seal=seal, out=case_root/backend, replay=replay))
            outcomes[backend] = 'ok'
        except Exception as error:
            outcomes[backend] = str(error)
    comparisons = {}
    for backend, filename in (('gtls_corrected', 'compare.json'), ('gtls', 'literal_compare.json')):
        if all((case_root/b/'record.json').exists() for b in (backend, 'candidate')):
            harness.compare(SimpleNamespace(reference=case_root/backend/'record.json', candidate=case_root/'candidate/record.json',
                gates=gates, threshold=list(thresholds), out=case_root/filename))
            comparisons[backend] = json.loads((case_root/filename).read_text())
    return dict(statuses=outcomes, corrected_reference_reused=bool(reuse and reuse['proved_no_op']),
                passed=comparisons.get('gtls_corrected',{}).get('passed'),
                comparison_status=comparisons.get('gtls_corrected',{}).get('status'),
                literal_passed=comparisons.get('gtls',{}).get('passed'))
