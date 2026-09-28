#!/usr/bin/env python3
"""Restore original benchmark evidence after verifying its archive and members."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / 'benchmarks/archives'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_manifest(study, directory=MANIFESTS):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', study):
        raise ValueError('Invalid archive identifier')
    record = json.loads((directory / (study + '.json')).read_text())
    if record.get('schema') != 1 or record.get('id') != study:
        raise ValueError('Unsupported archive manifest')
    files = {}
    for name, size, digest, kept in record['files']:
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name
                or not name.startswith(('benchmarks/results/', 'docs/validation/'))):
            raise ValueError('Unsafe member path: ' + name)
        if name in files or not isinstance(size, int) or size < 0:
            raise ValueError('Invalid or duplicate inventory entry: ' + name)
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or not isinstance(kept, bool):
            raise ValueError('Invalid member identity: ' + name)
        files[name] = (size, digest, kept)
    return record, files


def check_archive(path, record):
    if (Path(path).stat().st_size != record['archive']['bytes']
            or sha256(path) != record['archive']['sha256']):
        raise ValueError('Archive checksum or size differs from the committed manifest')


def safe_destination(destination, name):
    target = destination / name
    for path in (target, *target.parents):
        if path == destination:
            break
        if path.is_symlink():
            raise ValueError('Restore path contains a symlink: ' + str(path))
    return target


def restore(archive, record, files, destination, full=False):
    """Verify every member before installing any missing files; never overwrite."""
    check_archive(archive, record)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = {name: value for name, value in files.items() if full or not value[2]}
    for name, (_, digest, _) in selected.items():
        target = safe_destination(destination, name)
        if target.exists() and (not target.is_file() or sha256(target) != digest):
            raise ValueError('Existing file differs; use an empty destination: ' + name)
    installed = 0
    with tempfile.TemporaryDirectory(prefix='.archive-restore-', dir=destination) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream:
                if member.name not in files or member.name in seen or not member.isfile():
                    raise ValueError('Unexpected, duplicate or non-regular archive member')
                seen.add(member.name)
                size, expected, _ = files[member.name]
                if member.size != size:
                    raise ValueError('Member size differs: ' + member.name)
                digest = hashlib.sha256()
                output = None
                if member.name in selected:
                    staged = stage / member.name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    output = staged.open('wb')
                try:
                    with stream.extractfile(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            digest.update(block)
                            if output is not None:
                                output.write(block)
                finally:
                    if output is not None:
                        output.close()
                if digest.hexdigest() != expected:
                    raise ValueError('Member checksum differs: ' + member.name)
                if output is not None:
                    staged.chmod(0o755 if member.mode & 0o111 else 0o644)
        if seen != set(files):
            raise ValueError('Archive omits inventory members')
        for name, (_, digest, _) in selected.items():
            target = safe_destination(destination, name)
            if target.exists():
                if not target.is_file() or sha256(target) != digest:
                    raise ValueError('Restore destination changed: ' + name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic no-clobber installation on the same filesystem.
            os.link(stage / name, target)
            installed += 1
    return installed


def fetch(record, remote, cache):
    """Download through the caller's configured rclone remote; no credentials stored."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (record['archive']['sha256'] + '.tar.gz')
    if target.exists():
        check_archive(target, record)
        return target
    if not remote:
        raise ValueError('Supply --archive FILE or --remote NAME:BUCKET; see docs/BENCHMARK_ARCHIVES.md')
    fd, temporary = tempfile.mkstemp(prefix='download-', suffix='.part', dir=cache)
    os.close(fd)
    try:
        subprocess.run(['rclone', 'copyto', remote.rstrip('/') + '/' + record['archive']['key'],
                        temporary, '--contimeout', '15s', '--timeout', '60s'], check=True)
        check_archive(temporary, record)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List preserved studies and archive sizes')
    command = sub.add_parser('restore', help='Verify and restore original evidence')
    command.add_argument('study')
    command.add_argument('--archive', type=Path, help='Previously downloaded archive')
    command.add_argument('--remote', default=os.environ.get('CUVARBASE_ARCHIVE_REMOTE'),
                         help='Configured rclone remote and bucket, e.g. archive:cuvarbase')
    command.add_argument('--cache', type=Path, default=ROOT / '.benchmark-archives/downloads')
    command.add_argument('--destination', type=Path, default=ROOT)
    command.add_argument('--full', action='store_true',
                         help='Restore original reports too; use an empty destination')
    args = parser.parse_args()
    if args.command == 'list':
        for path in sorted(MANIFESTS.glob('*.json')):
            record, files = load_manifest(path.stem)
            print('%-40s %7.2f MiB  %4d files' %
                  (path.stem, record['archive']['bytes'] / 1024 ** 2, len(files)))
        return
    record, files = load_manifest(args.study)
    archive = args.archive or fetch(record, args.remote, args.cache)
    installed = restore(archive, record, files, args.destination, args.full)
    print('Verified %d original members; restored %d missing files.' % (len(files), installed))


if __name__ == '__main__':
    main()
