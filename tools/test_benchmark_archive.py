"""Evidence restoration must preserve failures and refuse corrupt or unsafe input."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from tools.benchmark_archive import load_manifest, restore, sha256


def fixture(tmp_path, extra=None, wrong_member_hash=False):
    archive = tmp_path / 'study.tar.gz'
    original = {'benchmarks/results/study/failure.json': b'{"passed":false}\n',
                'benchmarks/results/study/README.md': b'original report\n'}
    with tarfile.open(archive, 'w:gz') as stream:
        for name, value in original.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            stream.addfile(member, io.BytesIO(value))
        if extra:
            member = tarfile.TarInfo(extra)
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp/outside'
            stream.addfile(member)
    files = [[name, len(value), hashlib.sha256(value).hexdigest(), name.endswith('.md')]
             for name, value in original.items()]
    if wrong_member_hash:
        files[-1][2] = '0' * 64
    record = dict(schema=1, id='study', archive=dict(bytes=archive.stat().st_size,
                  sha256=sha256(archive)), files=files)
    (tmp_path / 'study.json').write_text(json.dumps(record))
    record, inventory = load_manifest('study', tmp_path)
    return archive, record, inventory


def test_restore_preserves_failure_and_updated_tracked_report(tmp_path):
    archive, record, inventory = fixture(tmp_path)
    dest = tmp_path / 'checkout'
    report = dest / 'benchmarks/results/study/README.md'
    report.parent.mkdir(parents=True)
    report.write_text('Current report with archive links\n')
    assert restore(archive, record, inventory, dest) == 1
    assert json.loads((report.parent / 'failure.json').read_text()) == {'passed': False}
    assert report.read_text() == 'Current report with archive links\n'
    assert restore(archive, record, inventory, dest) == 0
    complete = tmp_path / 'complete'
    assert restore(archive, record, inventory, complete, full=True) == 2
    assert (complete / report.relative_to(dest)).read_text() == 'original report\n'


@pytest.mark.parametrize('damage', ['archive', 'member', 'symlink', 'traversal'])
def test_corrupt_or_unsafe_archive_installs_nothing(tmp_path, damage):
    extra = {'symlink': 'benchmarks/results/study/link', 'traversal': '../outside'}.get(damage)
    archive, record, inventory = fixture(tmp_path, extra, damage == 'member')
    if damage == 'archive':
        with archive.open('ab') as stream:
            stream.write(b'corruption')
    dest = tmp_path / 'checkout'
    with pytest.raises(ValueError):
        restore(archive, record, inventory, dest)
    assert not (dest / 'benchmarks').exists()


def test_destination_conflict_is_not_overwritten(tmp_path):
    archive, record, inventory = fixture(tmp_path)
    dest = tmp_path / 'checkout'
    conflict = dest / 'benchmarks/results/study/failure.json'
    conflict.parent.mkdir(parents=True)
    conflict.write_text('unrelated existing evidence')
    with pytest.raises(ValueError, match='Existing file differs'):
        restore(archive, record, inventory, dest)
    assert conflict.read_text() == 'unrelated existing evidence'


def test_restore_refuses_a_symlink_destination(tmp_path):
    archive, record, inventory = fixture(tmp_path)
    dest = tmp_path / 'checkout'
    dest.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (dest / 'benchmarks').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        restore(archive, record, inventory, dest)
    assert list(outside.iterdir()) == []


def test_manifest_cannot_escape_destination(tmp_path):
    _, record, _ = fixture(tmp_path)
    record['files'][0][0] = 'benchmarks/results/../../../outside'
    (tmp_path / 'study.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Unsafe member path'):
        load_manifest('study', tmp_path)
