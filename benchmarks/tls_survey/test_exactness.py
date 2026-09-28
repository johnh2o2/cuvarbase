"""The original held-out implementation outcome must survive every diagnostic."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import exactness


@pytest.fixture
def matching():
    metadata = dict(regime='tess_solar', null=False, name='test')
    candidate = dict(period=3., score=7., successful_no_candidate=False,
                     no_candidate_reason=None, recovered=True, alias_recovered=True)
    row = dict(valid=True, candidates={'native': candidate},
               spectra={'periods': 'grid', 'chi2': 'all-values', 'valid_mask': 'all-mask'},
               name='test', method='tls', input_sha256='input')
    thresholds = {key: {'tess_solar/tls': dict(value=cut, target_fpr=fpr)}
                  for key, cut, fpr in [('thresholds', 6., .05), ('secondary_thresholds', 8., .01)]}
    return row, metadata, thresholds


def test_complete_originals_match_both_operating_points(matching):
    row, metadata, thresholds = matching
    result = exactness.compare(row, copy.deepcopy(row), metadata, thresholds)
    assert result['exact']
    assert result['baseline_decisions']['thresholds']['detected']
    assert not result['baseline_decisions']['secondary_thresholds']['detected']


@pytest.mark.parametrize('change', ['chi2', 'valid_mask', 'periods', 'period', 'score', 'recovered', 'valid'])
def test_every_primary_numerical_difference_is_retained(matching, change):
    row, metadata, thresholds = matching
    baseline = copy.deepcopy(row)
    if change in baseline['spectra']:
        baseline['spectra'][change] = 'different'
    elif change == 'valid':
        baseline['valid'] = False
    else:
        baseline['candidates']['native'][change] = False if change == 'recovered' else 9.
    assert not exactness.compare(row, baseline, metadata, thresholds)['exact']


def test_both_api_failures_do_not_establish_equivalence(matching):
    row, metadata, thresholds = matching
    row.update(valid=False, candidates={}, spectra={})
    assert not exactness.compare(row, row, metadata, thresholds)['exact']


def test_no_candidate_is_valid_equal_nondetection(matching):
    row, metadata, thresholds = matching
    row['candidates']['native'].update(period=None, score=0., successful_no_candidate=True,
                                       recovered=False, alias_recovered=False)
    assert exactness.compare(row, row, metadata, thresholds)['exact']


def test_plan_is_frozen_before_execution(tmp_path, monkeypatch):
    seal = tmp_path / 'seal.json'
    seal.write_text(json.dumps(dict(source_identity={'science': 'fixed'},
        production_sources={'candidate': 'fixed'}, regimes=['tess_solar'] * 10,
        counts={'injections': 256, 'nulls': 256})))
    monkeypatch.setattr(exactness, 'source_identity', lambda: {'science': 'fixed'})
    monkeypatch.setattr(exactness, 'package_identity', lambda root: {
        'candidate' if Path(root) == exactness.ROOT else 'baseline': 'fixed'})
    args = SimpleNamespace(seal=seal, baseline_root=tmp_path / 'baseline', out=tmp_path / 'plan.json',
                           campaign=tmp_path / 'final-campaign')
    exactness.freeze(args)
    plan = json.loads(args.out.read_text())
    assert plan['expected_cases'] == 5120
    assert plan['workers'] == 1
    assert plan['baseline_sources'] == {'baseline': 'fixed'}
    assert plan['estimated_extra_hours'] == pytest.approx(4.837752061155108)
    with pytest.raises(ValueError, match='overwrite'):
        exactness.freeze(args)
    args.out = tmp_path / 'second-plan.json'
    data = args.campaign / 'inputs-calibration'; data.mkdir(parents=True)
    (data / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='before any final'):
        exactness.freeze(args)


@pytest.mark.parametrize('split', ['calibration', 'injections', 'nulls'])
@pytest.mark.parametrize('partial_file', [False, True])
def test_plan_rejects_partial_final_input_directory(tmp_path, split, partial_file):
    args = SimpleNamespace(campaign=tmp_path / 'final-campaign')
    data = args.campaign / ('inputs-' + split)
    data.mkdir(parents=True)
    if partial_file:
        (data / 'case.npz').write_bytes(b'partially generated input')
    assert not (data / 'manifest.json').exists()
    with pytest.raises(ValueError, match='before any final'):
        exactness.freeze(args)


def test_interrupted_repeat_cannot_replace_primary_mismatch(tmp_path, monkeypatch, matching):
    original, metadata, thresholds = matching
    baseline_root = tmp_path / 'baseline'; (baseline_root / 'cuvarbase').mkdir(parents=True)
    source = {'source': 'fixed'}
    seal = tmp_path / 'seal.json'; seal.write_text(json.dumps(dict(source_identity=source,
        production_sources=source, execution_shards=1)))
    plan = tmp_path / 'plan.json'; plan.write_text(json.dumps(dict(seal_sha256=exactness.sha(seal),
        protocol_sha256=exactness.sha(exactness.__file__), baseline_root=str(baseline_root),
        baseline_sources=source, expected_cases=1, splits=['injections'], planned_campaign_root=str(tmp_path),
        repeat_diagnostics={'first_mismatching_cases': 10, 'additional_baseline_runs': 2})))
    thresholds['seal_sha256'] = exactness.sha(seal)
    (tmp_path / 'thresholds.json').write_text(json.dumps(thresholds))
    (tmp_path / 'injections-search-0.json').write_text('original immutable receipt')
    entry = {'metadata': metadata, 'sha256': 'input', 'file': 'test.npz'}
    monkeypatch.setattr(exactness, 'source_identity', lambda: source)
    monkeypatch.setattr(exactness, 'package_identity', lambda root: source)
    monkeypatch.setattr(exactness, 'check_manifest', lambda *args: {'cases': [entry]})
    monkeypatch.setattr(exactness, 'check_result', lambda *args: {'cases': [original]})
    monkeypatch.setattr(exactness, 'load_case', lambda *args: ({}, metadata))
    monkeypatch.setitem(sys.modules, 'cuvarbase', SimpleNamespace(
        __file__=str(baseline_root / 'cuvarbase/__init__.py')))
    monkeypatch.setitem(sys.modules, 'cuvarbase.base', SimpleNamespace(ensure_context=lambda: None))
    driver = SimpleNamespace(Context=SimpleNamespace(get_device=lambda: SimpleNamespace(name=lambda: 'test')))
    monkeypatch.setitem(sys.modules, 'pycuda', SimpleNamespace(driver=driver))
    monkeypatch.setitem(sys.modules, 'pycuda.driver', driver)
    baseline = copy.deepcopy(original); baseline['spectra']['chi2'] = 'primary mismatch'
    calls = []
    def measure(*args):
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt('interrupted diagnostic')
        return baseline, {}
    monkeypatch.setattr(exactness, 'measured_search', measure)
    args = SimpleNamespace(plan=plan, plan_sha256=exactness.sha(plan), seal=seal,
                           campaign=tmp_path, out=tmp_path / 'exactness.json')
    with pytest.raises(KeyboardInterrupt):
        exactness.run(args)
    persisted = json.loads(args.out.read_text())
    assert persisted['mismatches'] == 1
    assert persisted['cases'][0]['baseline']['spectra']['chi2'] == 'primary mismatch'
    assert persisted['cases'][0]['original_candidate'] == original
    dumped = persisted['cases'][0]['original_baseline_arrays']
    assert exactness.sha(dumped['path']) == dumped['sha256']
    # Resume never executes the original input a second time or erases failure.
    exactness.run(args)
    final = json.loads(args.out.read_text())
    assert len(calls) == 2
    assert final['status'] == 'complete' and not final['exactness_qualified']
    args.campaign = tmp_path / 'other-existing-campaign'
    with pytest.raises(ValueError, match='pre-input frozen plan'):
        exactness.run(args)
