"""The descriptive wrapper must keep frozen definitions and input identities."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import heldout_snr


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = {'scientific': 'fixed'}
    seal = tmp_path / 'seal.json'; seal.write_text(json.dumps(dict(source_identity=source,
        regimes=['tess_solar'], counts={'injections': 1})))
    plan = tmp_path / 'plan.json'; plan.write_text(json.dumps(dict(
        seal_sha256=heldout_snr.sha(seal), planned_campaign_root=str(tmp_path),
        heldout_snr_protocol_sha256=heldout_snr.sha(heldout_snr.__file__))))
    folder = tmp_path / 'inputs-injections'; folder.mkdir()
    manifest = folder / 'manifest.json'; manifest.write_text(json.dumps(dict(status='complete',
        split='injections', seal_sha256=heldout_snr.sha(seal), source_identity=source,
        regimes=['tess_solar'], count_per_regime=1,
        cases=[dict(metadata={'name': 'test', 'regime': 'tess_solar'}, sha256='input')])))
    monkeypatch.setattr(heldout_snr, 'source_identity', lambda: source)
    monkeypatch.setattr(heldout_snr, 'load_case', lambda path, entry: (
        {'t': np.arange(8), 'periods': np.array([1., 2.])}, entry['metadata']))
    monkeypatch.setattr(heldout_snr, 'module', lambda *args: SimpleNamespace(build_cache=lambda *args: {}))
    calls = []
    def row(arrays, metadata, cache):
        calls.append(metadata['name'])
        return dict(name=metadata['name'], regime=metadata['regime'], native_family_white_snr=1.)
    monkeypatch.setattr(heldout_snr, 'diagnostic_row', row)
    return SimpleNamespace(command='run', seal=seal, plan=plan, plan_sha256=heldout_snr.sha(plan),
        manifest=manifest, out=tmp_path / 'heldout-snr.json'), calls


def test_guarded_descriptive_run_and_resume(campaign):
    args, calls = campaign
    heldout_snr.execute(args); heldout_snr.execute(args)
    value = json.loads(args.out.read_text())
    assert calls == ['test']
    assert value['status'] == 'complete' and value['split'] == 'injections'
    assert value['seal_sha256'] == heldout_snr.sha(args.seal)
    assert value['manifest_sha256'] == heldout_snr.sha(args.manifest)
    assert value['rows'][0]['input_sha256'] == 'input'


@pytest.mark.parametrize('field,value', [('split', 'nulls'), ('seal_sha256', 'different'),
                                        ('source_identity', {}), ('count_per_regime', 2)])
def test_foreign_heldout_populations_rejected(campaign, field, value):
    args, calls = campaign
    manifest = json.loads(args.manifest.read_text()); manifest[field] = value
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        heldout_snr.execute(args)
    assert not calls


def test_validation_cannot_relabel_heldout_data(campaign):
    args, calls = campaign; args.command = 'validate'
    with pytest.raises(ValueError, match='development inputs'):
        heldout_snr.execute(args)
    assert not calls
