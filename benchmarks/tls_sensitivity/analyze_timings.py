#!/usr/bin/env python3
"""Verify exclusive TLS timings and select the fastest supported frozen setting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import PROFILES, V1, write


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--timing', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True,
                    help='Earlier *_heldout.npz files used for exclusive timing')
    ap.add_argument('--cadences', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--thresholds', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    signatures = json.loads(a.thresholds.read_text())['source_signatures']
    recovery = json.loads(a.recovery.read_text())
    criteria = {(r['profile'], r['method']): r for r in recovery['comparisons']}
    plan = json.loads((a.timing / 'plan.json').read_text())
    statuses = json.loads((a.timing / 'status.json').read_text())
    if [s['job'] for s in statuses] != plan['order'] or any(s['exit_code'] for s in statuses):
        raise ValueError('Incomplete or failed timing configuration')
    for previous, following in zip(statuses[:-1], statuses[1:]):
        if previous['finished_epoch'] > following['started_epoch']:
            raise ValueError('Timing processes overlapped')
    expected = {f'{p}/{m}/{s}' for p in PROFILES for m in (*V1, f'gtls_{p}')
                for s in ('single', 'batch16')}
    if set(plan['order']) != expected or len(plan['order']) != len(expected):
        raise ValueError('Unexpected or duplicated timing jobs')
    rows, references = [], {}
    for job in plan['order']:
        p, m, mode = job.split('/')
        record = json.loads((a.timing / job / 'summary.json').read_text())
        if (record['status'] != 'ok' or record['n_repetitions'] != 5 or record['n_sources'] != 16
                or [r['rep'] for r in record['repetitions']] != list(range(5))):
            raise ValueError('Timing was not five valid repetitions on sixteen sources')
        wanted = json.loads((a.configs / f'{m}.json').read_text())
        if m.startswith('gtls') and mode == 'single':
            wanted['workers'] = 1
        if record['config'] != wanted:
            raise ValueError('Timed configuration differs from the declared execution policy')
        module = 'gputls' if m.startswith('gtls') else 'cuvarbase'
        signature = hashlib.sha256(json.dumps(record['installed_sources'][module], sort_keys=True).encode()).hexdigest()
        if signature != signatures[module]:
            raise ValueError('Timed numerical source differs from sensitivity experiment')
        reference = {k: record[k] for k in ('indices', 'input_sha256', 'input_array_sha256', 'truth', 'grid_sha256')}
        if p in references and reference != references[p]:
            raise ValueError('Timing methods received different inputs')
        references[p] = reference
        input_path = a.inputs / f'{p}_heldout.npz'
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != record['input_sha256']:
            raise ValueError('Timing input file differs from the retained original')
        if record['indices'] != list(range(8)) + list(range(128, 136)):
            raise ValueError('Timing did not use the declared injection/null subset')
        with np.load(input_path) as original:
            truth = json.loads(str(original['metadata']))['cases']
            if record['truth'] != [truth[i] for i in record['indices']]:
                raise ValueError('Timing truth differs from the retained inputs')
            hashes = [{k: array_hash(original[f'{k}_{i}']) for k in ('t', 'y', 'dy')}
                      for i in record['indices']]
            if hashes != record['input_array_sha256']:
                raise ValueError('Timed observation arrays differ from retained inputs')
            if {k: array_hash(original[k]) for k in record['grid_sha256']} != record['grid_sha256']:
                raise ValueError('Timed grid arrays differ from retained inputs')
        with np.load(a.cadences / f'{p}.npz') as cadence:
            grid = np.ascontiguousarray(cadence['tls_periods'])
            h = hashlib.sha256()
            h.update(grid.dtype.str.encode())
            h.update(json.dumps(grid.shape).encode())
            h.update(grid.tobytes())
            if h.hexdigest() != record['grid_sha256']['tls_periods']:
                raise ValueError('Timed TLS period grid differs from sensitivity study')
        baseline = {}
        if mode == 'single':
            for i, warm in zip(record['indices'], record['warmup']):
                baseline[i] = warm['candidates'][0]
        else:
            baseline = dict(zip(record['indices'], record['warmup'][0]['candidates']))
        seconds, period_changes, score_delta = [], 0, 0.
        for rep in record['repetitions']:
            seen, elapsed = [], 0.
            for call in rep['calls']:
                ids = [call['index']] if mode == 'single' else call['indices']
                if len(ids) != len(call['candidates']):
                    raise ValueError('Mismatched timed outputs')
                elapsed += call['elapsed_s']
                for index, candidate in zip(ids, call['candidates']):
                    seen.append(index)
                    if not candidate.get('api_result_valid', False) or candidate['finite_fraction'] != 1:
                        raise ValueError('Invalid timed candidate')
                    if not np.isfinite(candidate['score']) or not np.isfinite(candidate['period']) or candidate['period'] <= 0:
                        raise ValueError('Invalid timed score or period')
                    period_changes += candidate['period'] != baseline[index]['period']
                    score_delta = max(score_delta, abs(candidate['score'] - baseline[index]['score']))
            if sorted(seen) != sorted(record['indices']):
                raise ValueError('A repetition omitted or duplicated a source')
            measured = elapsed / 16
            if not np.isclose(measured, rep['seconds_per_source'], rtol=1e-12, atol=0):
                raise ValueError('Stored timing arithmetic is inconsistent')
            seconds.append(measured)
        median = float(np.median(seconds))
        if not np.isclose(median, record['seconds_per_source'], rtol=1e-12, atol=0):
            raise ValueError('Stored timing median is inconsistent')
        rows.append(dict(profile=p, method=m, mode=mode, seconds_per_source=median,
                         min_seconds_per_source=min(seconds), max_seconds_per_source=max(seconds),
                         initialization_s=record['initialization_s'], first_api_s=record['first_api_s'],
                         period_changes_from_workload_warmup=int(period_changes),
                         max_native_score_change_from_workload_warmup=score_delta,
                         source=str(Path(job) / 'summary.json')))
    by_key = {(r['profile'], r['method'], r['mode']): r for r in rows}
    selected = []
    for p in PROFILES:
        supported = [m for m in V1 if criteria[p, m]['supported']]
        method = min(supported or ['v1_original'], key=lambda m: by_key[p, m, 'batch16']['seconds_per_source'])
        v1, gtls = (by_key[p, m, 'batch16']['seconds_per_source'] for m in (method, f'gtls_{p}'))
        selected.append(dict(profile=p, method=method, recovery_supported=bool(supported),
                             batch_speedup=gtls/v1, v1_batch_seconds=v1, gtls_batch_seconds=gtls,
                             v1_search_usd_per_million=v1*1e6*.49/3600,
                             gtls_search_usd_per_million=gtls*1e6*.49/3600,
                             criterion=criteria[p, method]))
    write(a.out, dict(timings=rows, selected=selected,
                      verification=dict(complete=True, exclusive_processes=True,
                                        identical_inputs=True, timing_input_arrays_verified=True,
                                        sources_match_sensitivity=True,
                                        full_timing_spectra_retained=False),
                      hardware=plan['hardware'], boundary=plan['boundary'], exclusions=plan['exclusions'],
                      cohort='Eight earlier injections (two per SNR) and eight earlier nulls; identical across algorithms.',
                      single_gtls_qualification='One-worker latency is contextual; sensitivity was calibrated for the stated batch execution mode.'))
    csv_path = a.out.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
