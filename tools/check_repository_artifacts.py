#!/usr/bin/env python3
"""Keep generated evidence out of Git while retaining explicit review summaries."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    allowed = set()
    for manifest in (root / 'benchmarks/archives').glob('*.json'):
        record = json.loads(manifest.read_text())
        allowed.update(row[0] for row in record['files'] if row[3])
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    errors = []
    for name in filter(None, files):
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 1024 * 1024:
            errors.append(name + ': tracked file exceeds 1 MiB; archive bulk evidence')
        if name.startswith(('benchmarks/results/', 'docs/validation/')):
            if (name not in allowed and path.suffix not in {'.md', '.rst'}
                    and path.name not in {'.gitignore', '.gitattributes'}):
                errors.append(name + ': raw evidence belongs in the archive')
    return errors


if __name__ == '__main__':
    problems = check()
    if problems:
        raise SystemExit('\n'.join(problems))
    print('Tracked files contain only the selected benchmark reports and small artifacts.')
