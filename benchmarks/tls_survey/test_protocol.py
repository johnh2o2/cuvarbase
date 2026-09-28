"""Exercise the freeze/calibrate/analyze boundary without running detectors."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


HERE = Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _module('survey_protocol_common', HERE / 'common.py')
with patch.dict(sys.modules, {'common': common}):
    analysis = _module('survey_protocol_analysis', HERE / 'analyze.py')


def _receipt(tmp_path, split, count, methods):
    rows, planned = [], []
    for index in range(count):
        name = '%s_%04d' % (split, index)
        identity = hashlib.sha256(name.encode()).hexdigest()
        planned.append(dict(name=name, sha256=identity))
        for method in methods:
            injected = split in ('development', 'injections')
            score = 1000. if injected else float(index)
            candidate = dict(score=score, period=2., recovered=injected,
                             alias_recovered=injected)
            rankers = ('native',) if method == 'tls' else ('raw', 'likelihood', 'detrended')
            rows.append(dict(name=name, input_sha256=identity, method=method,
                regime='tess_solar', valid=True, elapsed_s=1.,
                candidates={ranker: dict(candidate) for ranker in rankers},
                white_oracle_snr=common.SNRS[index % 4], observed_events=3,
                in_transit_observations=20, grid_reachable=True))
    path = tmp_path / (split + '.json')
    common.write(path, dict(status='complete', split=split, cases=rows,
        planned_inputs=planned, planned_cases=count, methods=methods,
        manifest_sha256='manifest-' + split, production_sources={'engine.py': 'frozen'},
        runner_sha256='runner', shard_count=1))
    return path


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'source_identity', lambda: {'science.py': 'frozen'})
    methods = ['tls', *common.BLS_CONFIGS]
    dev = _receipt(tmp_path, 'development', 8, methods)
    devnull = _receipt(tmp_path, 'development_nulls', 64, methods)
    snr = tmp_path / 'snr.json'
    common.write(snr, dict(manifest_sha256='manifest-development', rows=[
        dict(regime='tess_solar', native_white_advantage=.01, native_ou_advantage=-.002)
        for unused in range(8)]))
    seal = tmp_path / 'seal.json'
    freeze = argparse.Namespace(development=[dev], development_nulls=[devnull],
        snr=snr, regimes='tess_solar', fpr=.05, calibration_count=128,
        injection_count=8, null_count=64, exposure_nodes=64,
        execution_shards=1, out=seal)
    analysis.freeze(freeze)
    selected = ['tls', 'bls_strong']
    calibration = _receipt(tmp_path, 'calibration', 128, selected)
    thresholds = tmp_path / 'thresholds.json'
    calibrate = argparse.Namespace(seal=seal, results=[calibration], out=thresholds)
    injections = _receipt(tmp_path, 'injections', 8, selected)
    nulls = _receipt(tmp_path, 'nulls', 64, selected)
    analyze = argparse.Namespace(seal=seal, thresholds=thresholds,
        injections=[injections], nulls=[nulls], out=tmp_path / 'recovery.json')
    return dict(freeze=freeze, calibrate=calibrate, analyze=analyze)


def _change(path, function):
    value = json.loads(path.read_text())
    function(value)
    common.write(path, value)


def test_full_protocol_keeps_tight_tolerances_and_separate_operating_points(protocol):
    sealed = json.loads(protocol['freeze'].out.read_text())
    assert sealed['bls_selected']['tess_solar'] == dict(method='bls_strong', ranker='likelihood')
    tolerance = sealed['tolerances']['tess_solar']
    assert tolerance['expected_snr_fractional_loss'] == 0.
    assert tolerance['recovery_absolute_probability_loss'] == 0.
    assert tolerance['fpr_absolute_increase_max'] == 0.
    analysis.calibrate(protocol['calibrate'])
    analysis.analyze(protocol['analyze'])
    result = json.loads(protocol['analyze'].out.read_text())
    assert len(result['methods']) == 4
    assert {row['target_fpr'] for row in result['methods']} == {.05, .01}
    assert all(row['detected'] == 8 for row in result['methods'])
    assert all(row['n_nulls'] == 64 for row in result['methods'])
    assert all(row['tls_minus_bls_recovery']['difference'] == 0 for row in result['contrasts'])
    # Exact agreement on eight cases must still retain finite-sample uncertainty.
    assert all(row['tls_minus_bls_recovery']['interval'][0] < 0 for row in result['contrasts'])


def test_seal_and_thresholds_cannot_be_overwritten(protocol):
    with pytest.raises(ValueError, match='overwrite a frozen seal'):
        analysis.freeze(protocol['freeze'])
    analysis.calibrate(protocol['calibrate'])
    with pytest.raises(ValueError, match='overwrite independently frozen thresholds'):
        analysis.calibrate(protocol['calibrate'])


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r['production_sources'].update({'engine.py': 'changed'}), 'numerical sources'),
    (lambda r: r['cases'][0].update(valid=False), 'Failed calibration nulls'),
    (lambda r: r['cases'].pop(), 'expected 128 cases'),
    (lambda r: r['cases'].append(copy.deepcopy(r['cases'][0])), 'Duplicate measured case'),
    (lambda r: r['cases'][0].update(input_sha256='changed'), 'planned identity'),
])
def test_calibration_rejects_incomplete_or_changed_evidence(protocol, mutation, match):
    _change(protocol['calibrate'].results[0], mutation)
    with pytest.raises(ValueError, match=match):
        analysis.calibrate(protocol['calibrate'])
    assert not protocol['calibrate'].out.exists()


def test_analysis_requires_thresholds_from_the_original_seal(protocol):
    analysis.calibrate(protocol['calibrate'])
    _change(protocol['analyze'].thresholds, lambda r: r.update(seal_sha256='another-seal'))
    with pytest.raises(ValueError, match='another design'):
        analysis.analyze(protocol['analyze'])
    assert not protocol['analyze'].out.exists()


def test_failed_heldout_injections_remain_in_the_denominator(protocol):
    analysis.calibrate(protocol['calibrate'])
    _change(protocol['analyze'].injections[0], lambda r: r['cases'][0].update(valid=False))
    analysis.analyze(protocol['analyze'])
    result = json.loads(protocol['analyze'].out.read_text())
    tls = [row for row in result['methods'] if row['method'] == 'tls']
    assert all(row['n_injections'] == 8 and row['detected'] == 7 for row in tls)
    assert all(row['failed_injections'] == 1 for row in tls)
