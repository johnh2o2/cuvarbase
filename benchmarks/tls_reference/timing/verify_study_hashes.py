#!/usr/bin/env python3
"""CPU check of record-only expected hashes against retained study arrays.

This verifies dtype/shape encoding, masked hidden storage and scalar hashes.
It does not execute a new public search or make a timing measurement.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from .common import fingerprint, load_cases, sha, write
    from .cohort import case_root, frozen_outputs
else:
    from common import fingerprint, load_cases, sha, write
    from cohort import case_root, frozen_outputs


def verify(manifest, results, name):
    entries = json.loads(Path(manifest).read_text())['cases']
    entry = next(value for value in entries if value['file'] == name)
    case = load_cases(manifest, entry['metadata']['regime'], [name])[0]
    expected = frozen_outputs(results, [case], backends=('gtls', 'gtls_corrected', 'candidate'),
                              manifest_path=manifest)
    checks = {}
    for backend in expected:
        root = case_root(manifest, results, name)/backend
        record = json.loads((root/'record.json').read_text())
        archive = root/record['arrays_file']
        if sha(archive) != record['arrays_sha256']:
            raise ValueError('Retained study array archive differs from its receipt')
        with np.load(archive, allow_pickle=False) as arrays:
            if backend != 'candidate':
                values = {key: np.ma.array(arrays[key], mask=arrays[key+'_mask'])
                          for key in ('periods', 'power', 'chi2')}
                public = SimpleNamespace(**values, period=record['result']['period'],
                                         SDE=record['result']['score'])
                masked_periods = int(np.sum(arrays['chi2_mask']))
            else:
                public = {key: arrays['public_'+key] for key in ('periods', 'power', 'chi2', 'valid_periods')}
                contract = record['result']['public_contract']
                public.update(period=contract['period'], SDE=contract['SDE'])
                masked_periods = int(np.sum(~arrays['public_valid_periods']))
            actual = fingerprint(backend, case, public)
        matches = actual['strict'] == expected[backend][name]['strict']
        checks[backend] = dict(passed=matches, nperiods=actual['nperiods'], masked_periods=masked_periods,
                              expected=expected[backend][name], observed_strict=actual['strict'])
    return dict(passed=all(value['passed'] for value in checks.values()), case=name, checks=checks,
                manifest_sha256=sha(manifest),
                source_hashes={path.name: sha(path) for path in
                               (Path(__file__), Path(__file__).with_name('common.py'),
                                Path(__file__).with_name('cohort.py'))},
                scope='CPU reconstruction from retained study arrays/scalars compared with record-only expected timing fingerprints; no new public call or GPU execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest, args.results, args.case)
    write(args.output, result)
    if not result['passed']:
        raise SystemExit('Study hash conversion failed')
    print(json.dumps(dict(passed=True, case=args.case,
                         masked_periods={key: value['masked_periods'] for key, value in result['checks'].items()})))


if __name__ == '__main__':
    main()
