"""CPU provenance checks for timing a reproduced population."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from .cohort import accepted_study, select
from .common import sha
from .merge import merge


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reproduced_study(root, regimes=('tess_solar',), indices=range(20), label='main'):
    root.mkdir(parents=True)
    identity = dict(production_sources={'cuvarbase/tls.py': 'production'},
                    generator_sha256=label+'-original-generator')
    native = {'core.py': 'native'}
    tools = {'validate.py': 'public-validator', 'corrected_reference.py': 'public-adapter'}
    original = dict(suite='heldout', source_identity=identity, seal_sha256=label+'-seal', cases=[])
    for regime in regimes:
        for index in indices:
            name = f'{regime}_null_{index:04d}.npz'
            (root/name).write_bytes(('Fictitious CPU provenance fixture: '+name).encode())
            original['cases'].append(dict(file=name, sha256=sha(root/name),
                metadata=dict(name=Path(name).stem, regime=regime, null=True, search_kwargs={}),
                arrays={'periods': 'original-array-digest'}))
    write(root/'original_manifest.json', original)
    manifest = copy.deepcopy(original)
    manifest.update(suite='reproduction', original_manifest_sha256=sha(root/'original_manifest.json'))
    for case in manifest['cases']:
        case['original_npz_sha256'] = case['sha256']
    write(root/'manifest.json', manifest)
    results = root/'results'
    for case in manifest['cases']:
        folder = results/Path(case['file']).stem
        for backend in ('gtls', 'gtls_corrected', 'candidate'):
            write(folder/backend/'record.json', dict(status='ok', input_sha256=case['sha256'],
                seal_sha256=manifest['seal_sha256'], harness_sha256=tools['validate.py'],
                input_metadata=dict(cohort='reproduction'), engine_sources=identity['production_sources'],
                result=dict(package_sources=native, reference_correction=dict(
                    correction='finite_candidates_before_ranking_v1', adapter_sha256=tools['corrected_reference.py']))))
        write(folder/'compare.json', dict(passed=True,
            reference_record_sha256=sha(folder/'gtls_corrected/record.json'),
            candidate_record_sha256=sha(folder/'candidate/record.json')))
        write(folder/'correction_trace.json', dict(correction='finite_candidates_before_ranking_v1', proved_no_op=True))
    count = len(manifest['cases'])
    write(results/'acceptance.json', dict(reproduction_gate={'pass': True},
        inputs_manifest_sha256=sha(root/'manifest.json'), seal_sha256=manifest['seal_sha256'],
        original_source_identity=identity, reproduction_sources=dict(production=identity['production_sources'], tools=tools),
        reference_package_sources=native, all_planned_accounted=True,
        unresolved_numerical_cases=[], candidate_regressions=[],
        counts=dict(planned=count, accounted=count, numerical_pairs_passed=count)))
    return root/'manifest.json', results


def test_reproduction_can_time_without_independent_promotion(tmp_path):
    manifest, results = reproduced_study(tmp_path/'replay')
    accepted = accepted_study(manifest, results)
    assert accepted['reproduction_gate_passed']
    assert not accepted['publication_gate_passed']
    assert accepted['evidence_kind'] == 'reproduction'
    assert accepted['reproduction_sources']['tools']['validate.py'] == 'public-validator'
    selected = select(manifest, results, 'tess_solar')
    assert selected['actual_batch_size'] == 16
    assert selected['accepted_study']['evidence_kind'] == 'reproduction'


@pytest.mark.parametrize('change', ('failed_gate', 'dual_gates', 'production', 'validator', 'adapter',
                                  'original_manifest', 'population', 'comparison', 'missing_case', 'count'))
def test_reproduction_requires_original_and_actual_execution_chains(tmp_path, change):
    manifest_path, results = reproduced_study(tmp_path/'replay')
    receipt_path = results/'acceptance.json'
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    folder = results/Path(manifest['cases'][0]['file']).stem
    if change == 'failed_gate':
        receipt['reproduction_gate']['pass'] = False
    elif change == 'dual_gates':
        receipt['publication_gate'] = {'pass': True}
    elif change == 'production':
        receipt['reproduction_sources']['production'] = {'engine': 'changed'}
    elif change == 'validator':
        receipt['reproduction_sources']['tools']['validate.py'] = 'changed'
    elif change == 'adapter':
        receipt['reproduction_sources']['tools']['corrected_reference.py'] = 'changed'
    elif change == 'original_manifest':
        (manifest_path.parent/'original_manifest.json').unlink()
    elif change == 'population':
        manifest['cases'][0]['arrays']['periods'] = 'changed'
        write(manifest_path, manifest)
        receipt['inputs_manifest_sha256'] = sha(manifest_path)
    elif change == 'comparison':
        comparison = json.loads((folder/'compare.json').read_text())
        comparison['candidate_record_sha256'] = 'changed'
        write(folder/'compare.json', comparison)
    elif change == 'missing_case':
        (folder/'candidate/record.json').unlink()
    elif change == 'count':
        receipt['counts']['numerical_pairs_passed'] -= 1
    write(receipt_path, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        accepted_study(manifest_path, results)


def test_reproduced_merge_is_portable_and_keeps_two_original_studies(tmp_path):
    campaign = tmp_path/'campaign'
    regimes = ('tess_solar', 'tess_gap', 'ztf_solar')
    studies = []
    for label, indices in (('main', range(8)), ('supplement', range(8, 16))):
        manifest, results = reproduced_study(campaign/label, regimes, indices, label)
        studies.append((label, manifest, results))
    receipt = merge(studies, campaign/'timing-inputs')
    assert receipt['input_count'] == 48
    assert receipt['evidence_kind'] == 'reproduction'
    shutil.move(str(campaign), str(tmp_path/'moved'))
    output = tmp_path/'moved/timing-inputs'
    manifest = json.loads((output/'manifest.json').read_text())
    accepted = accepted_study(output/'manifest.json', output)
    assert accepted['independently_accepted_studies'] == {}
    assert set(accepted['reproduced_studies']) == {'main', 'supplement'}
    for entry in manifest['cases']:
        assert not Path(entry['result_root']).is_absolute()
        assert sha(output/entry['file']) == entry['sha256']
    for regime in regimes:
        assert select(output/'manifest.json', output, regime)['actual_batch_size'] == 16
    gate = json.loads((output/'acceptance.json').read_text())
    assert gate['reproduction_gate']['pass']
    assert 'publication_gate' not in gate


def test_merge_requires_the_original16_nulls_in_all_regimes(tmp_path):
    manifest, results = reproduced_study(tmp_path/'one-regime')
    with pytest.raises(ValueError, match='incomplete'):
        merge([('main', manifest, results)], tmp_path/'timing-inputs')
    assert not (tmp_path/'timing-inputs').exists()
