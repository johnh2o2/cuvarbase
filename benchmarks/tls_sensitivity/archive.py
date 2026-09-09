#!/usr/bin/env python3
"""Export compact, lossless scalar evidence from complete recovery shards.

Input truth and installed-source maps are stored once. Original summary hashes,
all scalar outcomes and every spectrum hash are retained. Prepared lightcurves
and sampled periodograms stay in the larger measurement archive. The exporter
verifies their bytes before writing a verification receipt.
"""
import argparse
import functools
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compressed(path, value):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


@functools.lru_cache(maxsize=2)
def read(root):
    root = Path(root)
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, key in [('inputs.json.gz', 'input_archive_sha256'),
                      ('searches.json.gz', 'search_archive_sha256')]:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != provenance['verification'][key]:
            raise ValueError(f'Compact evidence archive hash mismatch: {root / name}')
    with gzip.open(root / 'inputs.json.gz', 'rt') as f:
        inputs = json.load(f)
    with gzip.open(root / 'searches.json.gz', 'rt') as f:
        searches = json.load(f)
    return inputs, searches, provenance


def reconstruct(inputs, sources, search):
    record = dict(search['metadata'])
    record['installed_sources'] = sources[search['sources_id']]
    truth = inputs[search['input_id']]['metadata']['cases']
    record['cases'] = [dict(**t, **c) for t, c in zip(truth, search['outcomes'])]
    if len(truth) != len(search['outcomes']):
        raise ValueError('Compact evidence has mismatched case counts')
    return record


def records(root, profile, split, method):
    root = Path(root)
    if (root / 'searches.json.gz').exists():
        inputs, searches, provenance = read(str(root.resolve()))
        for name, search in searches.items():
            if name.startswith(f'{profile}_{split}_') and name.endswith('/' + method):
                yield name, reconstruct(inputs, provenance['installed_sources'], search)
    else:
        for path in sorted(root.glob(f'{profile}_{split}_*/{method}/summary.json')):
            yield str(path), json.loads(path.read_text())


def main():
    from generate import array_hash, sha

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        ap.error('Export into an empty directory to preserve published receipts')
    inputs, searches, sources = {}, {}, {}
    n_arrays = n_spectra = 0
    for path in sorted(a.root.glob('*/input.npz')):
        shard = path.parent.name
        if not (path.parent / 'collection.json').exists():
            continue
        with np.load(path) as data:
            metadata = json.loads(str(data['metadata']))
            for i, truth in enumerate(metadata['cases']):
                for key, expected in truth['array_sha256'].items():
                    if array_hash(data[f'{key}_{i}']) != expected:
                        raise ValueError(f'Input array mismatch: {path}/{key}_{i}')
                    n_arrays += 1
            grids = {k: array_hash(data[k]) for k in ('freqs', 'q', 'tls_periods')}
        inputs[shard] = dict(metadata=metadata, input_sha256=sha(path), grid_sha256=grids)
        wanted = ['v1_original', 'v1_resolved', 'v1_fine', f'gtls_{metadata["profile"]}']
        if (path.parent / 'v1_bls_control/summary.json').exists():
            wanted.append('v1_bls_control')
        for method in wanted:
            summary = path.parent / method / 'summary.json'
            record = json.loads(summary.read_text())
            if record['status'] != 'ok' or record['input_sha256'] != inputs[shard]['input_sha256']:
                raise ValueError(f'Incomplete or mismatched search: {summary}')
            source_id = digest(record['installed_sources'])
            sources[source_id] = record['installed_sources']
            outcomes = []
            if len(record['cases']) != len(metadata['cases']):
                raise ValueError(f'Incorrect case count: {summary}')
            for truth, case in zip(metadata['cases'], record['cases']):
                if any(case[k] != v for k, v in truth.items()):
                    raise ValueError(f'Search input truth changed: {summary}')
                if case.get('output_file'):
                    spectrum = summary.parent / case['output_file']
                    if sha(spectrum) != case['output_sha256']:
                        raise ValueError(f'Spectrum file hash mismatch: {spectrum}')
                    with np.load(spectrum) as data:
                        for key, expected in case['spectrum_sha256'].items():
                            if array_hash(data[key]) != expected:
                                raise ValueError(f'Spectrum array hash mismatch: {spectrum}/{key}')
                    n_spectra += 1
                outcomes.append({k: v for k, v in case.items() if k not in truth})
            search = dict(input_id=shard, sources_id=source_id,
                          original_summary_sha256=sha(summary),
                          metadata={k: v for k, v in record.items() if k not in ('cases', 'installed_sources')},
                          outcomes=outcomes)
            if reconstruct(inputs, sources, search) != record:
                raise ValueError(f'Lossy compact export: {summary}')
            searches[shard + '/' + method] = search
        print('Verified ' + shard, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    compressed(a.out / 'inputs.json.gz', inputs)
    compressed(a.out / 'searches.json.gz', searches)
    provenance = dict(installed_sources=sources,
                      verification=dict(input_arrays_verified=n_arrays,
                                        sampled_spectrum_files_verified=n_spectra,
                                        search_summaries_roundtripped=len(searches),
                                        omitted_spectra='All unretained spectra have hashes; their bytes cannot be re-verified from this compact archive.',
                                        input_archive_sha256=sha(a.out / 'inputs.json.gz'),
                                        search_archive_sha256=sha(a.out / 'searches.json.gz')))
    (a.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
