"""Select timing inputs using paired API success only, never timing/recovery.

The first 16 cases are fixed in common.selected_names. Failed originals remain
in this receipt. Replacements are the next unused case in manifest order for
the same regime. Independent and reproduced evidence retain separate labels.
"""
import json
from pathlib import Path
import numpy as np

if __package__:
    from .common import NATIVE_BACKENDS, array_hash, selected_names, sha
else:
    from common import NATIVE_BACKENDS, array_hash, selected_names, sha


def _reproduction_chain(manifest_path, acceptance_path, manifest, acceptance):
    """Check original inputs and actual reproduced records without promotion."""
    if manifest.get('suite') != 'reproduction':
        raise ValueError('A reproduction gate requires a reproduction manifest')
    identity = manifest['source_identity']
    actual = acceptance.get('reproduction_sources', {})
    if (acceptance.get('original_source_identity') != identity or
            actual.get('production') != identity['production_sources']):
        raise ValueError('Reproduced production source differs from its original seal')
    tools = actual.get('tools', {})
    if not tools.get('validate.py') or not tools.get('corrected_reference.py'):
        raise ValueError('Reproduction lacks the executing validator/adapter identities')
    original_path = Path(manifest_path).resolve().parent/'original_manifest.json'
    if not original_path.is_file() or sha(original_path) != manifest.get('original_manifest_sha256'):
        raise ValueError('Reproduction needs its unchanged original_manifest.json from cases.py')
    original = json.loads(original_path.read_text())
    if (original.get('source_identity') != identity or
            original.get('seal_sha256') != manifest['seal_sha256'] or
            len(original['cases']) != len(manifest['cases'])):
        raise ValueError('Reproduction differs from its original input population')
    count = len(manifest['cases'])
    counts = acceptance.get('counts', {})
    if (not count or any(counts.get(key) != count for key in
                         ('planned', 'accounted', 'numerical_pairs_passed')) or
            acceptance.get('all_planned_accounted') is not True or
            acceptance.get('unresolved_numerical_cases') or acceptance.get('candidate_regressions')):
        raise ValueError('Reproduction did not account for every numerical pair')
    result_root = Path(acceptance_path).resolve().parent
    for source, case in zip(original['cases'], manifest['cases']):
        if (case['file'] != source['file'] or case['metadata'] != source['metadata'] or
                case.get('arrays') != source.get('arrays') or
                case.get('original_npz_sha256') != source['sha256']):
            raise ValueError('Reproduction changed an original numerical input or its metadata')
        records = {}
        folder = result_root/Path(case['file']).stem
        for backend in ('gtls', 'gtls_corrected', 'candidate'):
            path = folder/backend/'record.json'
            record = json.loads(path.read_text())
            records[backend] = path
            if (record.get('status') != 'ok' or record.get('input_sha256') != case['sha256'] or
                    record.get('seal_sha256') != manifest['seal_sha256'] or
                    record.get('harness_sha256') != tools['validate.py'] or
                    record.get('input_metadata', {}).get('cohort') != 'reproduction'):
                raise ValueError('Reproduced record differs from its executing validator or input')
            if backend == 'candidate':
                if record.get('engine_sources') != actual['production']:
                    raise ValueError('Reproduced candidate used another production source')
            elif record['result']['package_sources'] != acceptance['reference_package_sources']:
                raise ValueError('Reproduced reference used another native package')
            if backend == 'gtls_corrected' and record['result']['reference_correction'].get('adapter_sha256') != tools['corrected_reference.py']:
                raise ValueError('Reproduced reference used another correction adapter')
        comparison = json.loads((folder/'compare.json').read_text())
        if (comparison.get('passed') is not True or
                comparison.get('reference_record_sha256') != sha(records['gtls_corrected']) or
                comparison.get('candidate_record_sha256') != sha(records['candidate'])):
            raise ValueError('Reproduction lacks a passing comparison for its actual records')
    return actual


def _accepted_origin(manifest_path, acceptance_path):
    path = Path(acceptance_path)
    acceptance = json.loads(path.read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if 'publication_gate' in acceptance and 'reproduction_gate' in acceptance:
        raise ValueError('Independent and reproduced acceptance cannot be conflated')
    reproduced = 'reproduction_gate' in acceptance
    gate = 'reproduction_gate' if reproduced else 'publication_gate'
    if acceptance.get(gate, {}).get('pass') is not True:
        raise ValueError('Numerical study has not passed its '+gate.replace('_', ' '))
    if (acceptance['inputs_manifest_sha256'] != sha(manifest_path) or
            acceptance['seal_sha256'] != manifest['seal_sha256']):
        raise ValueError('Accepted study and timing manifest differ')
    if reproduced:
        actual = _reproduction_chain(manifest_path, path, manifest, acceptance)
    else:
        if manifest.get('suite') == 'reproduction' or acceptance['source_identity'] != manifest['source_identity']:
            raise ValueError('Independent acceptance cannot relabel a reproduced source')
        actual = None
    return dict(receipt_sha256=sha(path), counts=acceptance['counts'],
                limits=acceptance.get('limits', []), publication_gate_passed=not reproduced,
                reproduction_gate_passed=reproduced, numerical_validation_passed=True,
                evidence_kind='reproduction' if reproduced else 'independent',
                reproduction_sources=actual,
                original_manifest_sha256=manifest.get('original_manifest_sha256') if reproduced else None,
                manifest_sha256=sha(manifest_path), seal_sha256=manifest['seal_sha256'],
                production_sources=manifest['source_identity']['production_sources'],
                reference_package_sources=acceptance['reference_package_sources'])


def _resolve(manifest_path, value):
    value = Path(value)
    return value if value.is_absolute() else Path(manifest_path).resolve().parent/value


def accepted_study(manifest_path, results_root):
    manifest = json.loads(Path(manifest_path).read_text())
    if 'studies' not in manifest:
        return _accepted_origin(manifest_path, Path(results_root)/'acceptance.json')
    origins, source_entries, reference = {}, {}, None
    for name, study in manifest['studies'].items():
        origin = _resolve(manifest_path, study['manifest_path'])
        receipt = _resolve(manifest_path, study['acceptance_path'])
        validated = _accepted_origin(origin, receipt)
        if (validated['manifest_sha256'] != study['manifest_sha256'] or
                validated['seal_sha256'] != study['seal_sha256']):
            raise ValueError('Merged manifest has another origin study identity')
        if validated['production_sources'] != manifest['source_identity']['production_sources']:
            raise ValueError('Merged timing studies used different production algorithms')
        if reference is not None and reference != validated['reference_package_sources']:
            raise ValueError('Merged timing studies used different native reference packages')
        reference = validated['reference_package_sources']
        origins[name] = validated
        source_entries[name] = {entry['file']: entry for entry in json.loads(origin.read_text())['cases']}
    options = {}
    for entry in manifest['cases']:
        study = manifest['studies'][entry['study_id']]
        original = source_entries[entry['study_id']].get(entry['file'])
        if original is None or original['sha256'] != entry['sha256'] or original['metadata'] != entry['metadata']:
            raise ValueError('Merged timing input differs from its accepted origin manifest')
        expected_root = _resolve(manifest_path, study['results_root'])/Path(entry['file']).stem
        if _resolve(manifest_path, entry['result_root']).resolve() != expected_root.resolve():
            raise ValueError('Merged case result root is outside its declared origin study')
        regime, current = entry['metadata']['regime'], entry['metadata']['search_kwargs']
        if regime in options and options[regime] != current:
            raise ValueError('Merged timing studies used different search options within a regime')
        options[regime] = current
    independent = {name:value for name,value in origins.items() if value['evidence_kind'] == 'independent'}
    reproduced = {name:value for name,value in origins.items() if value['evidence_kind'] == 'reproduction'}
    return dict(publication_gate_passed=not reproduced, reproduction_gate_passed=bool(reproduced),
                numerical_validation_passed=True,
                evidence_kind='reproduction' if reproduced else 'independent',
                merged_manifest_sha256=sha(manifest_path), accepted_studies=origins,
                independently_accepted_studies=independent, reproduced_studies=reproduced,
                scope='Separate main and supplementary studies retain their own source/protocol/seal identities; production and native algorithms and per-regime search settings are shared')


def case_root(manifest_path, results_root, name, manifest=None):
    manifest = json.loads(Path(manifest_path).read_text()) if manifest is None else manifest
    entry = next(value for value in manifest['cases'] if value['file'] == name)
    if 'result_root' in entry:
        return _resolve(manifest_path, entry['result_root'])
    return Path(results_root)/Path(name).stem


def case_seal(manifest, entry):
    return (manifest['studies'][entry['study_id']]['seal_sha256']
            if 'studies' in manifest else manifest['seal_sha256'])


def select(manifest_path, results_root, regime):
    manifest_path, results_root = Path(manifest_path), Path(results_root)
    manifest = json.loads(manifest_path.read_text())
    entries = {entry['file']: entry for entry in manifest['cases']}
    originals = selected_names(regime)
    if any(name not in entries for name in originals):
        raise ValueError('The predeclared 16-case timing cohort is absent from the manifest')
    reserve = [entry['file'] for entry in manifest['cases']
               if entry['metadata']['regime'] == regime and entry['metadata'].get('null') is True
               and entry['file'] not in originals]
    examined, selected, replacements, excluded = [], [], [], []
    slots = []
    expected_candidate = manifest['source_identity']['production_sources']
    expected_native = None
    acceptance = accepted_study(manifest_path, results_root)

    def check(name):
        nonlocal expected_native
        entry = entries[name]
        if entry['metadata'].get('null') is not True:
            raise ValueError('The primary survey timing cohort must contain only null light curves')
        row = dict(case=name, input_sha256=entry['sha256'], backends={})
        for backend in ('gtls', 'candidate'):
            path = case_root(manifest_path, results_root, name, manifest)/backend/'record.json'
            if not path.exists():
                raise ValueError('Paired study result is not complete: ' + str(path))
            record = json.loads(path.read_text())
            if record.get('status') not in ('ok', 'error'):
                raise ValueError('Paired study result is still running: ' + str(path))
            if record['input_sha256'] != entry['sha256']:
                raise ValueError('Study input differs from timing manifest: ' + name)
            if record.get('seal_sha256') != case_seal(manifest, entry):
                raise ValueError('Study and manifest have different seals: ' + name)
            if backend == 'candidate' and record['engine_sources'] != expected_candidate:
                raise ValueError('Candidate study sources differ from the input seal')
            if backend == 'gtls' and record['status'] == 'ok':
                sources = record['result']['package_sources']
                if expected_native is not None and expected_native != sources:
                    raise ValueError('Native package changed within the independent study')
                expected_native = sources
            row['backends'][backend] = dict(status=record['status'], record_sha256=sha(path),
                study_elapsed_seconds=record.get('elapsed_seconds'),
                error=record.get('error') if record['status'] == 'error' else None)
        row['paired_api_success'] = all(value['status'] == 'ok' for value in row['backends'].values())
        examined.append(row)
        return row['paired_api_success']

    for name in originals:
        if check(name):
            slots.append(name)
        else:
            excluded.append(name)
            slots.append(None)
    for name in reserve:
        if all(name is not None for name in slots):
            break
        if check(name):
            position = slots.index(None)
            slots[position] = name
            replacements.append(dict(original_case=originals[position], replacement_case=name))
        else:
            excluded.append(name)
    selected = [name for name in slots if name is not None]
    single_requested = f'{regime}_null_0000.npz'
    single = (single_requested if single_requested in selected else
              next((entry['file'] for entry in manifest['cases'] if entry['file'] in selected), None))
    correction_traces, affected, correction_identity = [], [], None
    for name in selected:
        root = case_root(manifest_path, results_root, name, manifest)
        trace_path = root/'correction_trace.json'
        trace = json.loads(trace_path.read_text())
        if trace.get('correction') != 'finite_candidates_before_ranking_v1':
            raise ValueError('Unknown corrected-native study trace')
        if trace.get('proved_no_op') is not True:
            affected.append(name)
        corrected_path = root/'gtls_corrected/record.json'
        corrected = json.loads(corrected_path.read_text())
        if corrected['input_sha256'] != entries[name]['sha256']:
            raise ValueError('Corrected native used another timing input')
        if corrected['status'] == 'ok':
            identity = corrected['result']['reference_correction']
            if correction_identity is not None and correction_identity != identity:
                raise ValueError('Native host correction changed within the study')
            correction_identity = identity
        correction_traces.append(dict(case=name, proved_no_op=trace.get('proved_no_op'),
                                      corrected_status=corrected['status'],
                                      trace_sha256=sha(trace_path), record_sha256=sha(corrected_path)))
    return dict(regime=regime, requested_batch_size=16, actual_batch_size=len(selected),
                status='complete' if len(selected) == 16 else 'insufficient_paired_successes',
                original_cases=originals, selected_cases=selected, replacement_cases=replacements,
                excluded_cases=excluded, examined=examined,
                single_requested=single_requested, single_case=single,
                single_replaced=single is not None and single != single_requested,
                single_unavailable=single is None,
                single_selection_rule='Requested null0000 when successful; otherwise earliest manifest-order paired-success null in the actual batch cohort',
                selection_rule='Null-only paired API success; original null0000–0015 then next unused manifest-order null in regime. No injections, recovery, SNR output or elapsed time used.',
                study_times_are_not_benchmark_denominators=True,
                accepted_study=acceptance,
                correction_timing=dict(required=bool(affected), affected_cases=affected,
                    traces=correction_traces,
                    rule='Additional one-worker full calls on the same entire selected cohort if any trace cannot prove the host correction is a no-op; never enters literal strongest-pool selection'),
                expected_correction=correction_identity,
                expected_candidate_sources=expected_candidate,
                expected_native_sources=expected_native)


def verify_worker_sources(backend, actual, selection):
    if backend in NATIVE_BACKENDS:
        if actual['files'] != selection['expected_native_sources']:
            raise ValueError('Timed native package differs from the independent study')
        if backend == 'gtls_corrected' and actual.get('reference_correction') != selection['expected_correction']:
            raise ValueError('Timed native correction differs from the independent study')
    else:
        expected = selection['expected_candidate_sources']
        for name, digest in actual['files'].items():
            if expected.get('cuvarbase/' + name) != digest:
                raise ValueError('Timed candidate source differs from the study: ' + name)


def frozen_outputs(results_root, cases, backends=('gtls', 'candidate'), manifest_path=None):
    """Read retained complete-output hashes before timed workers start.

    Pool-width selection still uses the literal one-worker native reference.
    A candidate with a declared native-bug correction must reproduce its own
    frozen public output; this function does not silently approve a scientific
    difference between the two implementations.
    """
    result = {backend: {} for backend in backends}
    for case in cases:
        for backend in result:
            root = (case_root(manifest_path, results_root, case['name']) if manifest_path is not None else
                    Path(results_root)/Path(case['name']).stem)/backend
            record_path = root/'record.json'
            record = json.loads(record_path.read_text())
            if record['status'] != 'ok' or record['input_sha256'] != case['input_sha256']:
                raise ValueError('Cannot freeze a failed or different timing-case output')
            arrays = record['arrays']
            strict = {}
            for key in ('periods', 'power', 'chi2'):
                source = key if backend in NATIVE_BACKENDS else 'public_'+key
                info = arrays[source]
                if len(info['shape']) != 1:
                    raise ValueError('Frozen public spectrum is not one-dimensional')
                mask = (arrays[key+'_mask']['sha256'] if backend in NATIVE_BACKENDS else
                        array_hash(np.zeros(info['shape'], dtype=bool)))
                strict[key] = dict(data=info['sha256'], mask=mask)
            if backend in NATIVE_BACKENDS:
                period, sde = record['result']['period'], record['result']['score']
            else:
                contract = record['result']['public_contract']
                period, sde = contract['period'], contract['SDE']
            if period is None or sde is None or not np.isfinite(period) or not np.isfinite(sde):
                raise ValueError('Frozen public detection is not finite')
            strict.update(period=array_hash(np.array(period, np.float64)),
                          SDE=array_hash(np.array(sde, np.float64)))
            result[backend][case['name']] = dict(strict=strict,
                record_sha256=sha(record_path), arrays_sha256=record['arrays_sha256'],
                representation='Complete per-array dtype/shape/data hashes and masks retained by the independent study; large NPZ retention is not required')
    return result
