"""CPU contracts for exact frozen-input restoration; no signal/GPU execution."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


bank = load('frozen_input_bank', 'inputs.py')
reference = load('frozen_input_hash_reference', 'validate.py')


def make_study(root, name, count=2):
    directory = root / name
    directory.mkdir()
    cases = []
    for i in range(count):
        case_name = name + '_%d' % i
        arrays = dict(t=np.array([1., 2., 3.]), y=np.array([1., .9 + .01 * i, 1.]),
                      dy=np.full(3, .01), periods=np.array([1.2, 2.4, 3.6]),
                      signal=np.array([0., .1, 0.]), exposure_days=np.full(3, .001),
                      band=np.array([0, 1, 0], dtype=np.int64))
        metadata = dict(name=case_name, regime='fixture', null=False, truth_period=2.4,
                        nested=dict(values=[1, 2., None, True]))
        target = directory / (case_name + '.npz')
        np.savez_compressed(target, **arrays, metadata=json.dumps(metadata))
        cases.append(dict(file=target.name, sha256=bank.sha(target), metadata=metadata,
                          arrays={key: reference.array_hash(value) for key, value in arrays.items()}))
    manifest = dict(suite='heldout', stream='heldout:frozen-test', source_identity={'generator_sha256': 'fixed'},
                    seal_sha256='a' * 64, status='complete', cases=cases)
    target = directory / 'manifest.json'
    bank.write(target, manifest)
    return target


@pytest.fixture
def frozen(tmp_path):
    main = make_study(tmp_path, 'main')
    supplement = make_study(tmp_path, 'supplement', count=1)
    folder = tmp_path / 'bank'
    bank.export_bank([('main', main), ('supplement', supplement)], folder)
    return folder, main, supplement


def rehash(folder):
    """Simulate corruption with updated container hashes, but original identities."""
    inventory = json.loads((folder / 'bank.json').read_text())
    inventory['arrays_file_sha256'] = bank.sha(folder / 'arrays.npz')
    inventory['arrays_file_bytes'] = (folder / 'arrays.npz').stat().st_size
    bank.write(folder / 'bank.json', inventory)
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (folder / 'SHA256SUMS').write_text(''.join(bank.sha(p) + '  ' + p.relative_to(folder).as_posix() + '\n' for p in paths))


def rewrite_array_archive(folder, mutate):
    target = folder / 'arrays.npz'
    with zipfile.ZipFile(target) as original:
        entries = {name: original.read(name) for name in original.namelist()}
    mutate(entries)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for name, data in entries.items():
            out.writestr(name, data)
    rehash(folder)


def test_identity_matches_frozen_validator_including_dtype_shape_and_layout():
    values = [np.array([0., -0., 1.], dtype=dtype) for dtype in ('<f4', '<f8', '>f8')]
    values.extend([np.arange(6).reshape(2, 3), np.asfortranarray(np.arange(6).reshape(2, 3))])
    for value in values:
        assert bank.array_hash(value) == reference.array_hash(value)
    assert bank.array_hash(values[0]) != bank.array_hash(values[1])
    assert bank.array_hash(np.arange(6)) != bank.array_hash(values[3])
    assert bank.array_hash(values[3]) == bank.array_hash(values[4])


def test_dedup_preserves_complete_original_populations_and_metadata(frozen, tmp_path):
    folder, main, supplement = frozen
    checked = bank.verify_bank(folder)
    assert checked['cases'] == 3
    assert checked['array_uses'] == 21
    assert checked['unique_arrays'] == 8
    for study, path in [('main', main), ('supplement', supplement)]:
        output = tmp_path / (study + '-restored')
        receipt = bank.restore_bank(folder, study, path, output)
        assert receipt['every_restored_array_verified']
        assert (output / 'original_manifest.json').read_bytes() == path.read_bytes()
        original = json.loads(path.read_text())
        restored = json.loads((output / 'manifest.json').read_text())
        assert restored['suite'] == 'reproduction'
        assert restored['source_identity'] == original['source_identity']
        assert restored['seal_sha256'] == original['seal_sha256']
        assert restored['original_manifest_sha256'] == bank.sha(path)
        for before, after in zip(original['cases'], restored['cases']):
            assert after['original_npz_sha256'] == before['sha256']
            assert after['metadata'] == before['metadata']
            assert after['arrays'] == before['arrays']
            actual = bank.source_case(output, after)
            expected = bank.source_case(path.parent, before)
            for key in actual:
                assert actual[key].dtype == expected[key].dtype
                assert actual[key].shape == expected[key].shape
                assert actual[key].tobytes() == expected[key].tobytes()
        report = json.loads((output / 'reproduction.json').read_text())
        assert report['reproduction_method'] == 'exact-array-bank'
        assert all(row['numerical_arrays_equal'] for row in report['cases'])


def test_existing_bank_and_restoration_are_never_overwritten(frozen, tmp_path):
    folder, main, _ = frozen
    before = bank.sha(folder / 'bank.json')
    with pytest.raises(ValueError, match='overwrite'):
        bank.export_bank([('main', main)], folder)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(ValueError, match='overwrite'):
        bank.restore_bank(folder, 'main', main, output)
    assert bank.sha(folder / 'bank.json') == before


@pytest.mark.parametrize('corruption', ['container', 'arrays', 'metadata'])
def test_export_rejects_changed_original_and_leaves_no_published_bank(tmp_path, corruption):
    manifest_path = make_study(tmp_path, 'main')
    manifest = json.loads(manifest_path.read_text())
    case = manifest['cases'][0]
    target = manifest_path.parent / case['file']
    if corruption == 'container':
        target.write_bytes(target.read_bytes() + b'changed')
    else:
        with np.load(target, allow_pickle=False) as original:
            values = {key: original[key] for key in original.files}
        if corruption == 'arrays':
            values['y'] = values['y'] + .1
        else:
            values['metadata'] = json.dumps(dict(case['metadata'], truth_period=1.))
        np.savez_compressed(target, **values)
        case['sha256'] = bank.sha(target)
        bank.write(manifest_path, manifest)
    output = tmp_path / 'bad-bank'
    with pytest.raises(ValueError):
        bank.export_bank([('main', manifest_path)], output)
    assert not output.exists()


def test_restoration_requires_exact_original_manifest(frozen, tmp_path):
    folder, main, _ = frozen
    wrong = tmp_path / 'wrong.json'
    wrong.write_bytes(main.read_bytes() + b'\n')
    output = tmp_path / 'bad-restore'
    with pytest.raises(ValueError, match='original manifest differs'):
        bank.restore_bank(folder, 'main', wrong, output)
    assert not output.exists()


def test_corrupt_bank_container_rejected(frozen):
    folder, _, _ = frozen
    path = folder / 'arrays.npz'
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        bank.verify_bank(folder)


def test_original_numerical_hashes_detect_changed_values_after_container_rehash(frozen):
    folder, _, _ = frozen
    def mutate(entries):
        name = next(iter(entries))
        array = np.load(io.BytesIO(entries[name]), allow_pickle=False).copy()
        array.flat[0] += 1
        out = io.BytesIO()
        np.save(out, array, allow_pickle=False)
        entries[name] = out.getvalue()
    rewrite_array_archive(folder, mutate)
    with pytest.raises(ValueError, match='numerical array identity'):
        bank.verify_bank(folder)


@pytest.mark.parametrize('field,value', [('dtype', '<f4'), ('shape', [99])])
def test_dtype_shape_descriptor_mismatches_rejected(frozen, field, value):
    folder, _, _ = frozen
    inventory = json.loads((folder / 'bank.json').read_text())
    identity = next(k for k, row in inventory['arrays'].items() if row['dtype'] == '<f8')
    inventory['arrays'][identity][field] = value
    bank.write(folder / 'bank.json', inventory)
    rehash(folder)
    with pytest.raises(ValueError, match='dtype/shape'):
        bank.verify_bank(folder)


@pytest.mark.parametrize('name', ['../escape.npy', '/absolute.npy', 'unreferenced.npy'])
def test_archive_paths_and_unreferenced_members_rejected(frozen, name):
    folder, _, _ = frozen
    rewrite_array_archive(folder, lambda entries: entries.update({name: b'not-an-array'}))
    with pytest.raises(ValueError, match='archive members'):
        bank.verify_bank(folder)


def test_manifest_cannot_escape_to_parent_path(frozen):
    folder, _, _ = frozen
    inventory = json.loads((folder / 'bank.json').read_text())
    inventory['studies']['main']['manifest'] = '../outside.json'
    bank.write(folder / 'bank.json', inventory)
    rehash(folder)
    with pytest.raises(ValueError, match='manifest path'):
        bank.verify_bank(folder)


@pytest.mark.parametrize('filename', ['../escape.npz', '/absolute.npz', 'a\\escape.npz'])
def test_source_case_paths_are_validated_before_file_access(tmp_path, filename):
    path = make_study(tmp_path, 'main')
    manifest = json.loads(path.read_text())
    manifest['cases'][0]['file'] = filename
    bank.write(path, manifest)
    with pytest.raises(ValueError, match='filenames'):
        bank.export_bank([('main', path)], tmp_path / 'invalid-bank')


def test_missing_case_array_reference_is_rejected(frozen):
    folder, _, _ = frozen
    inventory = json.loads((folder / 'bank.json').read_text())
    inventory['arrays'].pop(next(iter(inventory['arrays'])))
    bank.write(folder / 'bank.json', inventory)
    rehash(folder)
    with pytest.raises(ValueError, match='complete original populations'):
        bank.verify_bank(folder)


def test_checksum_path_traversal_and_source_symlink_rejected(frozen, tmp_path):
    folder, main, _ = frozen
    with (folder / 'SHA256SUMS').open('a') as stream:
        stream.write('a' * 64 + '  ../outside\n')
    with pytest.raises(ValueError, match='relative artifact path'):
        bank.verify_bank(folder)
    manifest = json.loads(main.read_text())
    target = main.parent / manifest['cases'][0]['file']
    outside = tmp_path / 'outside.npz'
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        bank.export_bank([('main', main)], tmp_path / 'symlink-bank')


def test_control_digest_derivation_is_explicit_and_preserves_original_identity(tmp_path):
    path = make_study(tmp_path, 'controls', count=1)
    original = json.loads(path.read_text())
    original = dict(suite='controlled-development', rule='fixed signal and half noise',
                    endpoints=['period_rank'], cases=original['cases'])
    original['cases'][0].pop('arrays')
    original['cases'][0]['metadata']['purpose'] = 'mathematical_differential'
    target = path.parent / original['cases'][0]['file']
    with np.load(target, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    values['metadata'] = json.dumps(original['cases'][0]['metadata'])
    np.savez_compressed(target, **values)
    original['cases'][0]['sha256'] = bank.sha(target)
    bank.write(path, original)
    folder = tmp_path / 'control-bank'
    with pytest.raises(ValueError, match='complete frozen input manifest'):
        bank.export_bank([('controls', path)], folder)
    receipt = bank.export_bank([('controls', path)], folder, derive_missing_array_hashes=True)
    assert receipt['array_uses_with_original_digest'] == 0
    assert receipt['array_uses_derived_after_original_npz_hash_verification'] == 7
    output = tmp_path / 'control-restored'
    bank.restore_bank(folder, 'controls', path, output)
    restored = json.loads((output / 'manifest.json').read_text())
    assert restored['original_suite'] == 'controlled-development'
    assert 'source_identity' not in restored
    assert 'seal_sha256' not in restored
    assert (output / 'original_manifest.json').read_bytes() == path.read_bytes()
    assert len(restored['cases'][0]['arrays']) == 7
    assert restored['cases'][0]['metadata'] == original['cases'][0]['metadata']


def test_independent_case_cannot_silently_acquire_missing_array_hashes(tmp_path):
    path = make_study(tmp_path, 'main')
    manifest = json.loads(path.read_text())
    manifest['cases'][0].pop('arrays')
    bank.write(path, manifest)
    with pytest.raises(ValueError, match='seven numerical arrays'):
        bank.export_bank([('main', path)], tmp_path / 'bad-bank', derive_missing_array_hashes=True)
