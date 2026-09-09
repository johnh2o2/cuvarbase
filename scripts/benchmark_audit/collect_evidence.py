#!/usr/bin/env python3
"""After all timing finishes, verify installed source and hash the evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path('/tmp/cuvarbase-benchmark-audit')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(archive, prefix, installed, packaging_exclusions=()):
    rows = {}
    with tarfile.open(archive) as src:
        for member in src.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            expected = hashlib.sha256(src.extractfile(member).read()).hexdigest()
            actual_path = installed / rel
            actual = digest(actual_path) if actual_path.exists() else None
            rows[rel] = dict(expected_sha256=expected, installed_sha256=actual,
                             matches=expected == actual)
    if not rows:
        raise RuntimeError('No source files found: ' + prefix)
    return dict(archive_sha256=digest(archive), installed_path=str(installed),
                packaging_exclusions=list(packaging_exclusions),
                all_match=all(row['matches'] or (name in packaging_exclusions and
                                                row['installed_sha256'] is None)
                              for name, row in rows.items()), files=rows)


def main():
    package_path = Path(importlib.util.find_spec('cuvarbase').submodule_search_locations[0])
    checks = {
        'cuvarbase_v1': verify_archive(ROOT/'source-v1.tar', 'cuvarbase/', package_path),
        # Upstream's wheel omits these source-tree files. Runtime CUDA comes
        # from the embedded string in GPUFun.py, which is verified bytewise.
        'gtls_upstream': verify_archive(ROOT/'gtls-head.tar', 'src/gputls/',
                                       ROOT/'gtls-head-install/gputls',
                                       ('GPUFun.cu', 'GPUFun_bak.cu', 'move.sh')),
    }
    (ROOT/'results/source-verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    pypi_hashes = {}
    for env, package in [('legacy', 'cuvarbase'), ('modern', 'gputls')]:
        location = subprocess.check_output([
            str(ROOT/env/'bin/python'), '-c',
            'import importlib.util; print(importlib.util.find_spec(' + repr(package) +
            ').submodule_search_locations[0])'], text=True).strip()
        folder = Path(location)
        pypi_hashes[package] = {str(p.relative_to(folder)): digest(p)
                                 for p in sorted(folder.rglob('*'))
                                 if p.is_file() and '__pycache__' not in p.parts}
    (ROOT/'results/pypi-installed-sha256.json').write_text(json.dumps(pypi_hashes, indent=2)+'\n')
    for env in ['modern', 'legacy']:
        with (ROOT/f'results/{env}-final-freeze.txt').open('w') as out:
            subprocess.run([str(ROOT/env/'bin/python'), '-m', 'pip', 'freeze'],
                           stdout=out, check=True)
    runners = {p.name: digest(p) for p in sorted(ROOT.glob('*.py'))}
    (ROOT/'results/runner-sha256.json').write_text(json.dumps(runners, indent=2)+'\n')
    manifest = {str(p.relative_to(ROOT)): digest(p)
                for folder in ['results', 'inputs']
                for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
    (ROOT/'transfer-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(source_matches={k:v['all_match'] for k,v in checks.items()},
                          evidence_files=len(manifest))))
    if not all(v['all_match'] for v in checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
