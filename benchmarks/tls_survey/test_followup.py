"""The September follow-up must fix the actual child-process environment failure."""
import json
import sys

import pytest

from benchmarks.tls_survey import followup


def plan(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(dict(purpose='test', files={}, attempt_timeout_seconds=10)))
    return path, followup.sha(path)


def test_child_receives_all_six_limits_before_imports(monkeypatch, tmp_path):
    for key in followup.THREAD_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '32')
    source, digest = plan(tmp_path)
    command = [sys.executable, '-c',
               'import json,os; print(json.dumps({k:os.environ.get(k) for k in ' +
               repr(followup.THREAD_VARIABLES) + '}))']
    output = tmp_path / 'attempt'
    assert followup.launch(source, digest, command, output) == 0
    actual = json.loads((output / 'stdout.log').read_text())
    assert set(actual.values()) == {'1'}
    assert actual['VECLIB_MAXIMUM_THREADS'] == actual['NUMEXPR_NUM_THREADS'] == '1'


def test_failed_attempt_is_retained_and_cannot_be_overwritten(tmp_path):
    source, digest = plan(tmp_path)
    output = tmp_path / 'attempt'
    assert followup.launch(source, digest, [sys.executable, '-c', 'raise SystemExit(7)'], output) == 7
    receipt = json.loads((output / 'launch.json').read_text())
    assert receipt['status'] == 'failed' and receipt['exit_code'] == 7
    with pytest.raises(FileExistsError):
        followup.launch(source, digest, [sys.executable, '-c', 'pass'], output)


def test_source_drift_stops_before_launch(tmp_path):
    source, digest = plan(tmp_path)
    source.write_text('{}')
    output = tmp_path / 'attempt'
    with pytest.raises(ValueError, match='plan changed'):
        followup.launch(source, digest, [sys.executable, '-c', 'pass'], output)
    assert not output.exists()


@pytest.mark.parametrize('changed_array', [False, True])
def test_preserved_varied_inputs_require_exact_restored_parent_arrays(tmp_path, changed_array):
    import hashlib
    import numpy as np
    from benchmarks.tls_survey import run_strict_followup as strict

    nulls = tmp_path/'inputs/nulls'
    varied = tmp_path/'inputs/varied'
    nulls.mkdir(parents=True)
    varied.mkdir()
    arrays = dict(t=np.arange(1., 7.), y=np.arange(6.), dy=np.ones(6), periods=np.array([1., 2.]))
    np.savez_compressed(nulls/'a.npz', **arrays)
    original_zip_sha = 'a'*64
    original_manifest_sha = 'b'*64
    (nulls/'manifest.json').write_text(json.dumps(dict(
        original_manifest_sha256=original_manifest_sha,
        cases=[dict(file='a.npz', sha256=strict.sha(nulls/'a.npz'), original_npz_sha256=original_zip_sha)])))
    indices = np.array([0, 2, 5])
    derived = {key: value if key == 'periods' else value[indices] for key, value in arrays.items()}
    if changed_array:
        derived['y'][1] += 1.
    np.savez_compressed(varied/'v.npz', **derived, retained_original_indices=indices)
    (varied/'manifest.json').write_text(json.dumps(dict(
        source_manifest_sha256=original_manifest_sha,
        cases=[dict(file='v.npz', sha256=strict.sha(varied/'v.npz'), metadata=dict(
            original_file='a.npz', original_sha256=original_zip_sha,
            index_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))])))
    if changed_array:
        with pytest.raises(ValueError, match='differ from their restored parent'):
            strict.prepare_preserved_varied(tmp_path, tmp_path/'bound')
    else:
        result = strict.prepare_preserved_varied(tmp_path, tmp_path/'bound')
        assert (result.parent/'v.npz').read_bytes() == (varied/'v.npz').read_bytes()
        assert json.loads(result.read_text())['source_manifest_sha256'] == strict.sha(nulls/'manifest.json')
