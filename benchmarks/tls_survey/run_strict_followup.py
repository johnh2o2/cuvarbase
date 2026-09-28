#!/usr/bin/env python3
"""Run the preserved strict timing harness on a new allocation.

The restored null ZIP containers have new hashes. Bind the original varied
arrays to those restored parents without regenerating or selecting inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_preserved_varied(root, output):
    import numpy as np

    root, output = Path(root), Path(output)
    source = root/'inputs/varied'
    original = json.loads((source/'manifest.json').read_text())
    null_path = root/'inputs/nulls/manifest.json'
    nulls = json.loads(null_path.read_text())
    if original['source_manifest_sha256'] != nulls['original_manifest_sha256']:
        raise ValueError('Varied cohort and restored nulls have different original parents')
    parents = {row['file']: row for row in nulls['cases']}
    output.mkdir(parents=True, exist_ok=False)
    for row in original['cases']:
        path = source/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Archived varied input changed: '+row['file'])
        metadata = row['metadata']
        parent = parents[metadata['original_file']]
        parent_path = null_path.parent/parent['file']
        if (sha(parent_path) != parent['sha256'] or
                metadata['original_sha256'] != parent['original_npz_sha256']):
            raise ValueError('Restored parent identity changed')
        with np.load(path, allow_pickle=False) as varied, np.load(parent_path, allow_pickle=False) as base:
            indices = varied['retained_original_indices']
            if hashlib.sha256(indices.tobytes()).hexdigest() != metadata['index_sha256']:
                raise ValueError('Retained observation indices changed')
            for name in ('t', 'y', 'dy', 'periods'):
                expected = base[name] if name == 'periods' else base[name][indices]
                actual = varied[name]
                if (expected.dtype != actual.dtype or expected.shape != actual.shape or
                        expected.tobytes() != actual.tobytes()):
                    raise ValueError('Archived varied arrays differ from their restored parent')
        shutil.copyfile(path, output/row['file'])
    original.update(
        original_source_manifest_sha256=original['source_manifest_sha256'],
        source_manifest_sha256=sha(null_path),
        preserved_varied_manifest_sha256=sha(source/'manifest.json'),
        followup_parent_binding='Exact original varied ZIP bytes and retained indices; '
                                'all arrays verified against restored parent arrays')
    (output/'manifest.json').write_text(json.dumps(original, indent=2, sort_keys=True)+'\n')
    return output/'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('tune', 'measure'), required=True)
    parser.add_argument('--tuning', type=Path)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--max-hours', type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Fresh follow-up output required')
    if args.stage == 'measure':
        if args.tuning is None:
            parser.error('Measurement requires frozen tuning')
        prepare_preserved_varied(root, output/'varied-inputs')
    command = [sys.executable,
               str(root/'sources/candidate/benchmarks/tls_survey/throughput_campaign.py'),
               '--stage', args.stage,
               '--manifest', str(root/'inputs'/('development' if args.stage == 'tune' else 'nulls')/'manifest.json'),
               '--output', str(output), '--baseline-root', str(root/'sources/baseline'),
               '--candidate-root', str(root/'sources/candidate'),
               '--science-seal', str(root/'evidence/science-seal.json'),
               '--backends', 'baseline', 'candidate', 'gtls',
               '--hourly-usd', str(args.hourly_usd), '--max-hours', str(args.max_hours)]
    if args.tuning:
        command.extend(['--tuning', str(args.tuning.resolve())])
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(main())
