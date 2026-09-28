"""Resume guards must reject incomplete or foreign campaign artifacts."""
import copy
import json
from pathlib import Path

import pytest

import campaign


@pytest.fixture
def artifacts(tmp_path):
    seal = dict(source_identity={'physics': 'fixed'}, production_sources={'kernel': 'fixed'},
                regimes=['tess_solar'], counts={'injections': 2}, execution_shards=2,
                bls_selected={'tess_solar': {'method': 'bls_strong'}})
    entries = [dict(metadata={'name': 'case' + str(i), 'regime': 'tess_solar'}, sha256=str(i))
               for i in range(2)]
    manifest = dict(status='complete', split='injections', seal_sha256='reviewed',
                    source_identity=seal['source_identity'], regimes=seal['regimes'],
                    count_per_regime=2, cases=entries)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    receipt = dict(status='complete', split='injections',
                   manifest_sha256=campaign.sha(manifest_path),
                   production_sources=seal['production_sources'],
                   runner_sha256=campaign.sha(campaign.ROOT / 'benchmarks/tls_survey/run.py'),
                   shard_index=0, shard_count=2,
                   cases=[dict(name='case0', method=m, input_sha256='0')
                          for m in ('tls', 'bls_strong')])
    return seal, manifest, manifest_path, receipt


def test_complete_resume_artifacts(artifacts, tmp_path):
    seal, manifest, path, receipt = artifacts
    assert campaign.check_manifest(path, 'injections', seal, 'reviewed') == manifest
    output = tmp_path / 'results.json'; output.write_text(json.dumps(receipt))
    assert campaign.check_result(output, path, manifest, seal, 0) == receipt


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_method', 'wrong_input',
                                    'wrong_shard', 'wrong_source', 'incomplete'])
def test_resume_rejects_bad_search_receipts(artifacts, tmp_path, change):
    seal, manifest, path, receipt = artifacts
    receipt = copy.deepcopy(receipt)
    if change == 'missing':
        receipt['cases'].pop()
    elif change == 'duplicate':
        receipt['cases'].append(receipt['cases'][0])
    elif change == 'wrong_method':
        receipt['cases'][1]['method'] = 'bls_finest'
    elif change == 'wrong_input':
        receipt['cases'][0]['input_sha256'] = 'other'
    elif change == 'wrong_shard':
        receipt['shard_index'] = 1
    elif change == 'wrong_source':
        receipt['production_sources'] = {'kernel': 'changed'}
    elif change == 'incomplete':
        receipt['status'] = 'running'
    output = tmp_path / 'results.json'; output.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        campaign.check_result(output, path, manifest, seal, 0)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_seal', 'wrong_source'])
def test_resume_rejects_bad_input_manifests(artifacts, change):
    seal, manifest, path, _ = artifacts
    manifest = copy.deepcopy(manifest)
    if change == 'missing':
        manifest['cases'].pop()
    elif change == 'duplicate':
        manifest['cases'][1] = manifest['cases'][0]
    elif change == 'wrong_seal':
        manifest['seal_sha256'] = 'unreviewed'
    elif change == 'wrong_source':
        manifest['source_identity'] = {'physics': 'changed'}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        campaign.check_manifest(path, 'injections', seal, 'reviewed')
