#!/usr/bin/env python3
"""Export and restore exact frozen TLS inputs using a deduplicated array bank.

Only NumPy and the standard library are needed. Array identity uses the frozen
validation convention: SHA-256 of JSON dtype, JSON shape, and contiguous bytes.
Restoration never regenerates a signal, cadence, noise realization, or grid.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile
import zipfile

import numpy as np


ARRAY_KEYS = frozenset(('t', 'y', 'dy', 'periods', 'signal', 'exposure_days', 'band'))
HASH = re.compile(r'[0-9a-f]{64}')
LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')
CASE_FILE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.npz')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_hash(value):
    """The original validate.py identity, including dtype and shape."""
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(json.dumps(value.dtype.descr if value.dtype.names else value.dtype.str).encode())
    result.update(json.dumps(value.shape).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def safe_relative(name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Invalid relative artifact path')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or name in ('', '.'):
        raise ValueError('Invalid relative artifact path')
    return path


def artifact(root, name):
    root = Path(root).resolve()
    path = root / safe_relative(name)
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError('Artifact escapes its directory or is a symlink')
    return path


def check_manifest(manifest, *, allow_derived=False):
    if not isinstance(manifest.get('cases'), list) or not manifest['cases']:
        raise ValueError('A frozen manifest needs at least one case')
    controlled = (allow_derived and manifest.get('suite') == 'controlled-development' and
                  'rule' in manifest and 'endpoints' in manifest)
    if not controlled and (manifest.get('status') != 'complete' or 'source_identity' not in manifest or 'seal_sha256' not in manifest):
        raise ValueError('Use a complete frozen input manifest with its original identity and seal')
    names = set()
    for case in manifest['cases']:
        filename = case['file']
        if not isinstance(filename, str) or not CASE_FILE.fullmatch(filename) or filename in names:
            raise ValueError('Case filenames must be distinct safe plain NPZ filenames')
        names.add(filename)
        if case['metadata'].get('name') != filename[:-4]:
            raise ValueError('Case metadata name differs from its filename')
        if 'arrays' not in case and controlled and case['metadata'].get('purpose') == 'mathematical_differential':
            pass  # Only this explicitly opted-in control format lacks prior array hashes.
        elif set(case.get('arrays', {})) != ARRAY_KEYS or not all(HASH.fullmatch(v) for v in case['arrays'].values()):
            raise ValueError('Frozen TLS case must identify all seven numerical arrays')
        if not HASH.fullmatch(case['sha256']):
            raise ValueError('Invalid original NPZ SHA-256')
    return manifest


def check_arrays(arrays, expected):
    if set(arrays) != set(expected):
        raise ValueError('Numerical array keys differ from the frozen manifest')
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not portable numerical input')
        if array_hash(value) != expected[name]:
            raise ValueError('Numerical array identity mismatch: ' + name)


def source_case(folder, case, *, derive_missing=False):
    path = artifact(folder, case['file'])
    if sha(path) != case['sha256']:
        raise ValueError('Original NPZ bytes differ from the frozen manifest: ' + case['file'])
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != ARRAY_KEYS | {'metadata'}:
            raise ValueError('Unexpected original NPZ fields: ' + case['file'])
        arrays = {name: data[name] for name in sorted(ARRAY_KEYS)}
        if json.loads(str(data['metadata'])) != case['metadata']:
            raise ValueError('Original NPZ metadata differs from the frozen manifest: ' + case['file'])
    if 'arrays' in case:
        check_arrays(arrays, case['arrays'])
    elif not derive_missing or any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Original case has no numerical identities; explicit controlled-development derivation is required')
    return arrays


def new_staging(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Refuse to overwrite an existing output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))


def publish(staging, output):
    if output.exists() or output.is_symlink():
        raise ValueError('Output directory appeared during preparation')
    staging.rename(output)


def export_bank(studies, output, *, derive_missing_array_hashes=False):
    """Verify original cases and store each distinct numerical array once."""
    output, staging = new_staging(output)
    try:
        (staging / 'manifests').mkdir()
        arrays, origins = {}, {}
        original_bytes = array_uses = case_count = derived_uses = 0
        with zipfile.ZipFile(staging / 'arrays.npz', 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as bank:
            for label, path in studies:
                if not LABEL.fullmatch(label) or label in origins:
                    raise ValueError('Study labels must be distinct safe names')
                path = Path(path)
                original = path.read_bytes()
                manifest = check_manifest(json.loads(original), allow_derived=derive_missing_array_hashes)
                stored = 'manifests/' + label + '.json'
                (staging / stored).write_bytes(original)
                origins[label] = dict(manifest=stored,
                                      manifest_sha256=hashlib.sha256(original).hexdigest(),
                                      cases=len(manifest['cases']), derived_array_hashes={})
                for case in manifest['cases']:
                    values = source_case(path.parent, case, derive_missing=derive_missing_array_hashes)
                    expected = case.get('arrays')
                    if expected is None:
                        expected = {name: array_hash(value) for name, value in values.items()}
                        origins[label]['derived_array_hashes'][case['file']] = expected
                        derived_uses += len(expected)
                    original_bytes += (path.parent / case['file']).stat().st_size
                    case_count += 1
                    for name, value in values.items():
                        identity = expected[name]
                        array_uses += 1
                        if identity in arrays:
                            continue
                        value = np.ascontiguousarray(value)
                        data = io.BytesIO()
                        np.save(data, value, allow_pickle=False)
                        member = identity + '.npy'
                        # Fixed ZIP metadata makes the bank itself deterministic
                        # in one compression environment; values, not compressed
                        # container bytes, are the portable numerical contract.
                        info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
                        info.external_attr = 0o600 << 16
                        bank.writestr(info, data.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                                      compresslevel=6)
                        arrays[identity] = dict(member=member, dtype=value.dtype.str,
                                                shape=list(value.shape), nbytes=value.nbytes)
        if not origins:
            raise ValueError('Provide at least one frozen study')
        bank_path = staging / 'arrays.npz'
        inventory = dict(schema_version=1, format='cuvarbase-frozen-tls-arrays',
                         array_hash_convention='sha256(JSON dtype + JSON shape + C-contiguous bytes)',
                         arrays_file='arrays.npz', arrays_file_sha256=sha(bank_path),
                         arrays_file_bytes=bank_path.stat().st_size, arrays=arrays, studies=origins)
        write(staging / 'bank.json', inventory)
        receipt = dict(schema_version=1, exporter_sha256=sha(__file__),
                       python=platform.python_version(), numpy=np.__version__,
                       original_studies={name: origin['manifest_sha256'] for name, origin in origins.items()},
                       verified_original_cases=case_count, verified_original_array_uses=array_uses,
                       array_uses_with_original_digest=array_uses - derived_uses,
                       array_uses_derived_after_original_npz_hash_verification=derived_uses,
                       unique_arrays=len(arrays), unique_array_bytes=sum(row['nbytes'] for row in arrays.values()),
                       original_npz_bytes=original_bytes, bank_npz_bytes=bank_path.stat().st_size,
                       original_npz_metadata_equal=True, all_original_numerical_arrays_equal=True,
                       interpretation='Exact stored numerical inputs; no signal regeneration or new independent cases')
        write(staging / 'export.json', receipt)
        paths = sorted(p for p in staging.rglob('*') if p.is_file())
        (staging / 'SHA256SUMS').write_text(''.join(
            sha(p) + '  ' + p.relative_to(staging).as_posix() + '\n' for p in paths))
        verify_bank(staging)
        publish(staging, output)
        return dict(receipt, output=str(output), total_bundle_bytes=sum(
            p.stat().st_size for p in output.rglob('*') if p.is_file()))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def checksums(folder):
    paths = set()
    for line in artifact(folder, 'SHA256SUMS').read_text().splitlines():
        parts = line.split('  ', 1)
        if len(parts) != 2 or not HASH.fullmatch(parts[0]) or parts[1] in paths:
            raise ValueError('Malformed or duplicate checksum entry')
        path = artifact(folder, parts[1])
        if sha(path) != parts[0]:
            raise ValueError('Bundle file checksum mismatch: ' + parts[1])
        paths.add(parts[1])
    if not {'bank.json', 'arrays.npz', 'export.json'}.issubset(paths):
        raise ValueError('Bundle checksums omit required artifacts')
    return paths


def verify_bank(folder, *, return_arrays=False):
    """Check bundle, original manifests, and every stored dtype/shape/value hash."""
    folder = Path(folder)
    checked = checksums(folder)
    inventory = json.loads(artifact(folder, 'bank.json').read_text())
    if inventory.get('schema_version') != 1 or inventory.get('format') != 'cuvarbase-frozen-tls-arrays':
        raise ValueError('Unknown frozen-input bank format')
    if inventory.get('arrays_file') != 'arrays.npz':
        raise ValueError('Unexpected bank array filename')
    path = artifact(folder, 'arrays.npz')
    if path.stat().st_size != inventory['arrays_file_bytes'] or sha(path) != inventory['arrays_file_sha256']:
        raise ValueError('Numerical bank container checksum mismatch')
    needed, manifests = set(), {}
    for label, origin in inventory['studies'].items():
        if not LABEL.fullmatch(label) or origin['manifest'] != 'manifests/' + label + '.json':
            raise ValueError('Unsafe study manifest path')
        if origin['manifest'] not in checked:
            raise ValueError('Original manifest missing from bundle checksums')
        manifest_path = artifact(folder, origin['manifest'])
        if sha(manifest_path) != origin['manifest_sha256']:
            raise ValueError('Original manifest checksum mismatch')
        derived = origin.get('derived_array_hashes', {})
        manifest = check_manifest(json.loads(manifest_path.read_text()), allow_derived=bool(derived))
        if len(manifest['cases']) != origin['cases']:
            raise ValueError('Original manifest case count mismatch')
        missing = {case['file'] for case in manifest['cases'] if 'arrays' not in case}
        if set(derived) != missing:
            raise ValueError('Derived identities must describe only original controls lacking array hashes')
        for expected in derived.values():
            if set(expected) != ARRAY_KEYS or not all(HASH.fullmatch(v) for v in expected.values()):
                raise ValueError('Incomplete derived numerical identities')
        needed.update(value for case in manifest['cases']
                      for value in case.get('arrays', derived.get(case['file'], {})).values())
        manifests[label] = manifest
    if not manifests or set(inventory['arrays']) != needed:
        raise ValueError('Bank arrays do not match the complete original populations')
    values = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        expected = {identity + '.npy' for identity in needed}
        if len(members) != len(set(members)) or set(members) != expected:
            raise ValueError('Unexpected or duplicate array archive members')
        for identity, descriptor in inventory['arrays'].items():
            if not HASH.fullmatch(identity) or descriptor['member'] != identity + '.npy':
                raise ValueError('Invalid array identity/member')
            with archive.open(descriptor['member']) as source:
                value = np.load(source, allow_pickle=False)
            if (value.dtype.hasobject or value.dtype.str != descriptor['dtype'] or
                    list(value.shape) != descriptor['shape'] or value.nbytes != descriptor['nbytes']):
                raise ValueError('Stored array dtype/shape differs from its descriptor')
            if array_hash(value) != identity:
                raise ValueError('Stored numerical array identity mismatch')
            if return_arrays:
                values[identity] = value
    receipt = dict(studies=len(manifests), cases=sum(len(m['cases']) for m in manifests.values()),
                   array_uses=sum(len(ARRAY_KEYS) for m in manifests.values() for c in m['cases']),
                   unique_arrays=len(needed), all_numerical_identities_verified=True,
                   bank_manifest_sha256=sha(folder / 'bank.json'), arrays_file_sha256=sha(path))
    return (inventory, manifests, values, receipt) if return_arrays else receipt


def restore_bank(folder, study, manifest_path, output):
    """Restore one complete study with cases.py-compatible provenance receipts."""
    inventory, manifests, values, verified = verify_bank(folder, return_arrays=True)
    if study not in manifests:
        raise ValueError('Unknown frozen study: ' + study)
    manifest_path = Path(manifest_path)
    original = manifest_path.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    if original_sha != inventory['studies'][study]['manifest_sha256']:
        raise ValueError('Requested original manifest differs from the frozen bank study')
    derived = inventory['studies'][study].get('derived_array_hashes', {})
    manifest = check_manifest(json.loads(original), allow_derived=bool(derived))
    output, staging = new_staging(output)
    try:
        receipts = []
        reproduced_cases = []
        for case in manifest['cases']:
            expected = case.get('arrays', derived.get(case['file'], {}))
            arrays = {name: values[identity] for name, identity in expected.items()}
            check_arrays(arrays, expected)
            target = staging / case['file']
            np.savez_compressed(target, **arrays, metadata=json.dumps(case['metadata'], sort_keys=True))
            restored_case = dict(case, arrays=expected, original_npz_sha256=case['sha256'], sha256=sha(target))
            if 'arrays' not in case:
                restored_case['array_identity_basis'] = 'Derived from original NPZ after original container hash and metadata verification'
            # Read each written container back and verify metadata plus every
            # original numerical identity before publishing any usable manifest.
            source_case(staging, restored_case)
            reproduced_cases.append(restored_case)
            receipts.append(dict(name=case['metadata']['name'], numerical_arrays_equal=True,
                                 original_npz_sha256=case['sha256'], regenerated_npz_sha256=restored_case['sha256']))
        (staging / 'original_manifest.json').write_bytes(original)
        reproduced = dict(manifest, suite='reproduction', original_suite=manifest['suite'], original_manifest_sha256=original_sha,
                          cases=reproduced_cases)
        write(staging / 'manifest.json', reproduced)
        write(staging / 'reproduction.json', dict(
            original_manifest_sha256=original_sha, generator_sha256=sha(__file__),
            restorer_sha256=sha(__file__), reproduction_method='exact-array-bank',
            original_generator_sha256=manifest.get('source_identity', {}).get('generator_sha256'),
            bank_manifest_sha256=verified['bank_manifest_sha256'],
            bank_arrays_sha256=verified['arrays_file_sha256'], study=study, cases=receipts,
            cases_with_derived_array_hashes=sorted(derived),
            interpretation='Reproduced frozen numerical inputs from exact stored arrays, not newly independent observations'))
        publish(staging, output)
        return dict(study=study, cases=len(receipts), numerical_array_uses=len(receipts) * len(ARRAY_KEYS),
                    every_restored_array_verified=True, output=str(output),
                    manifest_sha256=sha(output / 'manifest.json'), original_manifest_sha256=original_sha)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Build an immutable bank from original NPZs and manifests')
    export.add_argument('--study', nargs=2, action='append', required=True, metavar=('NAME', 'MANIFEST'))
    export.add_argument('--out', type=Path, required=True)
    export.add_argument('--derive-missing-array-hashes', action='store_true',
                        help='Explicitly derive absent array hashes for mathematical controlled-development cases only, after original NPZ hash verification')
    verify = commands.add_parser('verify', help='Verify all bank files and every numerical array identity')
    verify.add_argument('--bank', type=Path, required=True)
    restore = commands.add_parser('restore', help='Restore one complete frozen study without signal regeneration')
    restore.add_argument('--bank', type=Path, required=True)
    restore.add_argument('--study', required=True)
    restore.add_argument('--manifest', type=Path, required=True, help='Original published input manifest; bytes must match bank')
    restore.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        result = export_bank(args.study, args.out, derive_missing_array_hashes=args.derive_missing_array_hashes)
    elif args.command == 'verify':
        result = verify_bank(args.bank)
    else:
        result = restore_bank(args.bank, args.study, args.manifest, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
