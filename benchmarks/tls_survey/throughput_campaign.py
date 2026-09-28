#!/usr/bin/env python3
"""Predeclared sequential tuning and independent sustained-throughput measurement.

Select one operational batch/pool setting per backend on the balanced development
queue. Freeze it before opening timing outcomes from an independent null cohort.
Per-cadence panels characterize that setting; they do not claim separate tuning
of every cadence. Every attempted configuration and failed numerical gate remains
in the campaign receipt. This driver never rents or terminates cloud resources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.tls_reference.timing.common import sha, write

REGIMES = ('tess_solar', 'tess_gap_long', 'ztf_solar')
BACKENDS = ('baseline', 'candidate', 'gtls', 'gtls_corrected', 'bls')
EXECUTED_BACKENDS = ('baseline', 'candidate', 'gtls', 'bls')
VARIED_POLICY = ('First 32 independent manifest-order nulls per regime; retain rounded fractions '
    '0.8 + 0.2*i/31 for positions i=0..31. Retain first/last observations, select remaining indices '
    'without replacement with NumPy default_rng seeded by SHA256 of '
    'tls-survey-throughput-varied-v1 + NUL + original filename. Align time/flux/error slicing, '
    'keep the original period grid, and never use these modified nulls for recovery/FAP inference. '
    'The resulting 96 distinct sources replace the final balanced-mixed workload, exercising '
    'cache preparation across many observation-array lengths throughout each queue.')


def validate_tuning_identity(tuning, measurement):
    """A frozen operating choice must use the same timing definitions at measure."""
    if tuning.get('status') != 'complete':
        raise ValueError('Measurement requires completed tuning')
    for key in ('driver_sha256', 'runner_sha256', 'protocol_sha256', 'harness_dependency_sha256'):
        if key not in tuning or tuning[key] != measurement.get(key):
            raise ValueError('Timing definitions changed after frozen tuning: ' + key)


def select_names(manifest_path, count, *, nulls_only=False):
    manifest = json.loads(Path(manifest_path).read_text())
    names = []
    for regime in REGIMES:
        entries = [entry for entry in manifest['cases']
                   if entry['metadata']['regime'] == regime and
                   (not nulls_only or entry['metadata']['null'])]
        if len(entries) < count:
            raise ValueError(f'{regime} has only {len(entries)} eligible cases; {count} required')
        names.extend(entry['file'] for entry in entries[:count])
    return names


def prepare_varied_manifest(source_manifest, output, count=32):
    """Derive timing-only nulls without reading scientific detection outcomes."""
    source_manifest, output = Path(source_manifest), Path(output)
    if count < 2:
        raise ValueError('Varied timing requires at least two retention levels')
    origin = json.loads(source_manifest.read_text())
    receipt_path = output/'manifest.json'
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing['source_manifest_sha256'] != sha(source_manifest):
            raise ValueError('Varied timing input origin changed')
        if existing['count_per_regime'] != count:
            raise ValueError('Varied timing case count changed')
        if any(sha(output/entry['file']) != entry['sha256'] for entry in existing['cases']):
            raise ValueError('Varied timing input arrays changed')
        return receipt_path
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(purpose='throughput_only_derived_nulls', policy=VARIED_POLICY,
                   source_manifest_sha256=sha(source_manifest), numpy_version=np.__version__,
                   count_per_regime=count, cases=[])
    for regime in REGIMES:
        entries = [entry for entry in origin['cases'] if entry['metadata']['regime'] == regime
                   and entry['metadata']['null']]
        if len(entries) < count:
            raise ValueError(f'Varied timing needs {count} independent nulls in {regime}')
        for index, entry in enumerate(entries[:count]):
            source_path = source_manifest.parent/entry['file']
            if sha(source_path) != entry['sha256']:
                raise ValueError('Original timing-null input hash changed')
            with np.load(source_path, allow_pickle=False) as data:
                original = json.loads(str(data['metadata']))
                nobs = len(data['t'])
                fraction = .8 + .2*index/(count-1)
                retained = min(nobs, max(3, int(np.rint(fraction*nobs))))
                digest = hashlib.sha256(('tls-survey-throughput-varied-v1\0'+entry['file']).encode()).digest()
                rng = np.random.default_rng(np.frombuffer(digest, dtype='<u4'))
                indices = np.sort(np.r_[0, rng.choice(np.arange(1, nobs-1), retained-2, replace=False), nobs-1])
                metadata = {key: original[key] for key in ('regime', 'search_kwargs', 'grid_kwargs', 'baseline_days')}
                name = 'varied_'+entry['file']
                metadata.update(name=Path(name).stem, null=True,
                    purpose='throughput_only_subsampled_null', do_not_use_for_recovery_or_false_alarm=True,
                    original_file=entry['file'], original_sha256=entry['sha256'],
                    original_ndata=nobs, ndata=len(indices), retained_fraction_requested=fraction,
                    retained_fraction_actual=len(indices)/nobs, endpoints_retained=True,
                    index_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
                np.savez_compressed(output/name, t=data['t'][indices], y=data['y'][indices],
                    dy=data['dy'][indices], periods=data['periods'],
                    retained_original_indices=indices, metadata=json.dumps(metadata, sort_keys=True))
            receipt['cases'].append(dict(file=name, sha256=sha(output/name), metadata=metadata))
    write(receipt_path, receipt)
    return receipt_path


def eligible(record):
    return (record.get('status') == 'ok' and record.get('gpu_ownership', {}).get('passed') is True
            and len(record.get('qualification', [])) == 2
            and all(row['gate']['passed'] for row in record['qualification'])
            and bool(record.get('repetitions'))
            and all(row['status'] == 'ok' for row in record['repetitions']))


def winner(records):
    usable = [record for record in records if eligible(record)]
    if not usable:
        return None
    return min(usable, key=lambda record: (
        -record['summary']['median_repetition_lightcurves_per_second'],
        record['workers'], record['batch_size']))


def cross_baseline_qualification(directory, configs):
    """Exact baseline/optimized spectra on the actual timed source cohort."""
    by_scope = {}
    for row in configs:
        if row['eligible'] and row['backend'] in ('baseline', 'candidate'):
            record = json.loads((Path(directory)/row['result']).read_text())
            by_scope.setdefault(row['scope'], {}).setdefault(row['backend'], []).append(record)
    checks = []
    for scope, methods in by_scope.items():
        if not all(backend in methods for backend in ('baseline', 'candidate')):
            continue
        expected = methods['baseline'][0]['qualification'][0]['gate']['strict']
        for record in methods['candidate']:
            actual = record['qualification'][0]['gate']['strict']
            different = sorted(name for name in set(expected) | set(actual)
                               if expected.get(name) != actual.get(name))
            checks.append(dict(scope=scope, workers=record['workers'], batch_size=record['batch_size'],
                               exact=not different, differing_cases=different))
    return dict(passed=bool(checks) and all(check['exact'] for check in checks), checks=checks)


def summary_table(directory, configs, unavailable=()):
    rows = []
    for config in configs:
        path = Path(directory)/config['result']
        record = json.loads(path.read_text()) if path.exists() else {}
        summary = record.get('summary', {}) if config['eligible'] else {}
        memory = record.get('memory', {})
        row = {key: config[key] for key in ('backend', 'scope', 'workers', 'batch_size', 'eligible')}
        row['unavailable_reason'] = config.get('failure_reason')
        row.update({key: summary.get(key) for key in (
            'median_repetition_lightcurves_per_second', 'observed_rate_min', 'observed_rate_max',
            'total_measured_sources', 'total_measured_seconds',
            'total_measured_compute_usd', 'cold_preparation_compute_usd',
            'cold_first_cohort_including_startup_seconds', 'cold_amortized_lightcurves_per_second',
            'usd_per_million_steady', 'usd_per_million_cold_amortized', 'estimated_run_compute_usd')})
        row.update(sampled_gpu_peak_bytes=memory.get('gpu_used_bytes'),
                   sampled_worker_rss_peak_bytes=memory.get('host_pool_rss_bytes'),
                   cpu_quota_cores=record.get('environment', {}).get('cpu_quota_cores'),
                   host_memory_limit_bytes=record.get('environment', {}).get('host_memory_limit_bytes'))
        rows.append(row)
    for missing in unavailable:
        row = {key: None for key in rows[0]} if rows else {}
        row.update(backend=missing['backend'], scope=missing['scope'], eligible=False,
                   unavailable_reason=missing['reason'])
        rows.append(row)
    if rows:
        with (Path(directory)/'measurements.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('tune', 'measure'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tuning', type=Path, help='Frozen tuning campaign.json for --stage measure')
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, default=ROOT)
    parser.add_argument('--science-seal', type=Path, help='Frozen science-selected BLS settings/rankers')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(EXECUTED_BACKENDS))
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--exhaustive', action='store_true',
                        help='Optional full worker×batch sweep; default is the predeclared 5-configuration staged sweep')
    parser.add_argument('--cases-per-regime', type=int)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'measure' and not args.tuning:
        parser.error('measure requires a frozen tuning campaign')
    if args.stage == 'tune' and (1 not in args.workers or 1 not in args.batches):
        parser.error('Tuning includes a one-worker/batch-one reference')
    if 'bls' in args.backends and args.science_seal is None:
        parser.error('The BLS competitor requires --science-seal before timing tuning')
    count = args.cases_per_regime or (8 if args.stage == 'tune' else 16)
    names = select_names(args.manifest, count, nulls_only=args.stage == 'measure')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output/'campaign.json'
    if receipt_path.exists() and not args.resume:
        parser.error('Output campaign exists; use a new directory or --resume')
    plan = dict(schema_version=1, stage=args.stage, manifest_sha256=sha(args.manifest),
        declared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        manifest=str(args.manifest.resolve()), names=names, backends=args.backends,
        science_seal_sha256=sha(args.science_seal) if args.science_seal else None,
        worker_options=args.workers, batch_options=args.batches,
        exhaustive=args.exhaustive,
        varied_final_policy=VARIED_POLICY,
        tuning_policy='Each backend independently maximizes balanced-mixed development queue throughput; '
            'one repetition at >=24 sources and >=30 seconds. Failed configurations excluded but retained. '
            'Exact same-backend one-worker complete spectra qualify TLS/GTLS before/after queues; '
            'BLS requires exact period arrays/masks and selected period/score, with retained full powers '
            'and unused-ranker variation reported separately under its predeclared atomic-repeat amendment. '
            'Default staged search first compares workers 1/2/4 at batch 1, freezes the fastest eligible '
            'worker count, then tests batches 4/8 only at that count (five configurations per backend). '
            'This is a conditional explored space, not an exhaustive global-optimum claim. Optional '
            '--exhaustive evaluates the full declared Cartesian product instead. Ties choose fewer '
            'workers then smaller batches. One operational setting per backend is frozen for all final '
            'cadence and mixed panels.',
        measurement_policy='First 16 manifest-order independent nulls per cadence (or explicit count), '
            'with no selection by outcomes. Frozen setting; per-regime queues and a derived 96-distinct-source '
            'varied-size queue per varied_final_policy; three repetitions each at >=96 sources AND >=120 seconds.',
        driver_sha256=sha(__file__), runner_sha256=sha(Path(__file__).with_name('throughput.py')),
        harness_dependency_sha256={name: sha(ROOT/'benchmarks/tls_reference/timing'/name)
                                   for name in ('common.py', 'benchmark.py')},
        protocol_sha256=sha(Path(__file__).with_name('THROUGHPUT_PROTOCOL.md')),
        failure_policy='Retain every failed configuration and missing competitor panel with its reason; '
            'continue independent competitors. No failed result contributes a performance denominator. '
            'Show every predeclared scope, with missing bars marked no qualifying result. '
            'Compute a baseline/optimized ratio only where the paired complete-spectrum gate passes. '
            'Do not widen gates, substitute post-hoc case subsets, or retune held-out failures.',
        configs=[], unavailable=[], selected={}, status='running')
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        for key in ('stage', 'manifest_sha256', 'names', 'backends', 'worker_options',
                    'batch_options', 'exhaustive', 'science_seal_sha256', 'driver_sha256',
                    'runner_sha256', 'protocol_sha256', 'harness_dependency_sha256'):
            if prior[key] != plan[key]:
                raise ValueError('Resume changes the frozen campaign: ' + key)
        plan = prior
        plan['status'] = 'running'
    tuning = None
    if args.tuning:
        tuning = json.loads(args.tuning.read_text())
        if tuning['stage'] != 'tune' or not tuning.get('selected'):
            raise ValueError('Tuning receipt does not contain frozen selections')
        validate_tuning_identity(tuning, plan)
        if set(names) & set(tuning['names']):
            raise ValueError('Final timing cohort reuses development case identities')
        if tuning.get('science_seal_sha256') != plan['science_seal_sha256']:
            raise ValueError('Final timing changes the frozen BLS selection')
        plan['tuning_sha256'] = sha(args.tuning)
        plan['selected'] = tuning['selected']
    varied_manifest = None
    varied_names = []
    if args.stage == 'measure':
        varied_manifest = prepare_varied_manifest(args.manifest, args.output/'varied-inputs')
        varied_names = [entry['file'] for entry in json.loads(varied_manifest.read_text())['cases']]
        plan.update(varied_manifest=str(varied_manifest.resolve()), varied_names=varied_names,
                    varied_manifest_sha256=sha(varied_manifest))
    declaration = args.output/'declaration.json'
    if not declaration.exists():
        write(declaration, {key: value for key, value in plan.items()
                            if key not in ('configs', 'selected', 'status')})
    elif plan.get('declaration_sha256') not in (None, sha(declaration)):
        raise ValueError('Frozen timing declaration changed')
    plan['declaration_sha256'] = sha(declaration)
    write(receipt_path, plan)
    started = time.perf_counter()
    completed = {row['id']: row for row in plan['configs']}

    def unavailable(backend, scope, reason, reference=None):
        entry = dict(backend=backend, scope=scope, reason=reason)
        if reference is not None:
            entry.update(reference=str(reference.resolve()),
                         reference_sha256=sha(reference) if reference.exists() else None)
        if entry not in plan['unavailable']:
            plan['unavailable'].append(entry)
        write(receipt_path, plan)

    def run(backend, workers, batch, scope, reference=None):
        identifier = f'{backend}-{scope}-w{workers}-b{batch}'
        output = args.output/identifier
        if identifier in completed:
            actual = sha(output/'result.json') if (output/'result.json').exists() else None
            if actual != completed[identifier]['result_sha256']:
                raise ValueError('Saved configuration changed before resume: '+identifier)
            return json.loads((output/'result.json').read_text()) if actual else dict(status='runner_failed_before_receipt')
        if time.perf_counter()-started > args.max_hours*3600:
            raise TimeoutError('Campaign time cap reached before the next configuration')
        selected_names = (varied_names if scope == 'varied' else names if scope == 'mixed' else
                          [name for name in names if name.startswith(scope + '_')])
        input_manifest = varied_manifest if scope == 'varied' else args.manifest
        command = [sys.executable, str(Path(__file__).with_name('throughput.py')),
            '--manifest', str(input_manifest.resolve()), '--output', str(output.resolve()),
            '--backend', backend, '--workers', str(workers), '--batch-size', str(batch),
            '--hourly-usd', str(args.hourly_usd), '--names', *selected_names,
            '--repetitions', '1' if args.stage == 'tune' or scope.endswith('_reference') else '3',
            '--min-sources', '24' if args.stage == 'tune' else '96',
            '--min-seconds', '30' if args.stage == 'tune' else '120']
        if backend in ('candidate', 'baseline', 'bls'):
            source = args.baseline_root if backend == 'baseline' else args.candidate_root
            command += ['--source-root', str(source.resolve())]
        if args.science_seal:
            command += ['--science-seal', str(args.science_seal.resolve())]
        if reference:
            command += ['--reference', str(reference.resolve())]
        log = args.output/(identifier+'.log')
        print(json.dumps(dict(action='start', id=identifier, utc=time.time())), flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        result_path = output/'result.json'
        record = json.loads(result_path.read_text()) if result_path.exists() else dict(status='runner_failed_before_receipt')
        row = dict(id=identifier, backend=backend, workers=workers, batch_size=batch,
                   scope=scope, returncode=result.returncode, result=str(result_path.relative_to(args.output)),
                   result_sha256=sha(result_path) if result_path.exists() else None,
                   eligible=eligible(record), summary=record.get('summary'),
                   failure_reason=None if eligible(record) else record.get('error', record['status']))
        plan['configs'].append(row)
        completed[identifier] = row
        write(receipt_path, plan)
        print(json.dumps(dict(action='finished', **row)), flush=True)
        return record

    try:
        if args.stage == 'tune':
            for backend in args.backends:
                records = []
                initial = run(backend, 1, 1, 'mixed')
                records.append(initial)
                if not eligible(initial):
                    unavailable(backend, 'mixed', 'Development one-worker qualification failed; no eligible tuning setting')
                    continue
                reference = args.output/f'{backend}-mixed-w1-b1/result.json'
                for workers in args.workers:
                    if workers != 1:
                        records.append(run(backend, workers, 1, 'mixed', reference))
                worker_choice = winner(records)
                worker_counts = args.workers if args.exhaustive else [worker_choice['workers']]
                plan.setdefault('worker_stage_selection', {})[backend] = dict(
                    workers=worker_choice['workers'], batch_size=1,
                    pilot_lightcurves_per_second=worker_choice['summary']['median_repetition_lightcurves_per_second'])
                write(receipt_path, plan)
                for workers in worker_counts:
                    for batch in args.batches:
                        if batch != 1:
                            records.append(run(backend, workers, batch, 'mixed', reference))
                chosen = winner(records)
                if chosen is not None:
                    plan['selected'][backend] = dict(workers=chosen['workers'], batch_size=chosen['batch_size'],
                        pilot_lightcurves_per_second=chosen['summary']['median_repetition_lightcurves_per_second'],
                        qualification_reference=str(reference.resolve()))
                write(receipt_path, plan)
        else:
            # A fresh independent cohort requires its own literal single-worker
            # spectra before the selected concurrent configuration can qualify.
            for backend in args.backends:
                chosen = plan['selected'].get(backend)
                if chosen is None:
                    for scope in (*REGIMES, 'varied'):
                        unavailable(backend, scope, 'No qualifying setting in the frozen development tuning')
                    continue
                for scope in (*REGIMES, 'varied'):
                    # Reference generation uses the regular runner, retaining a
                    # complete successful scalar queue; these are labeled and
                    # never included in the selected-setting figure.
                    reference = None
                    if chosen['workers'] > 1:
                        reference_dir = args.output/'qualification'/scope/backend
                        reference_dir.mkdir(parents=True, exist_ok=True)
                        input_manifest = varied_manifest if scope == 'varied' else args.manifest
                        input_names = varied_names if scope == 'varied' else names
                        ref_command = [sys.executable, str(Path(__file__).with_name('throughput.py')),
                            '--manifest', str(input_manifest.resolve()), '--output', str(reference_dir.resolve()),
                            '--backend', backend, '--workers', '1', '--batch-size', '1',
                            '--hourly-usd', str(args.hourly_usd), '--repetitions', '1', '--min-sources', '1',
                            '--min-seconds', '0', '--names',
                            *[name for name in input_names if scope == 'varied' or name.startswith(scope+'_')]]
                        if backend in ('candidate', 'baseline', 'bls'):
                            source = args.baseline_root if backend == 'baseline' else args.candidate_root
                            ref_command += ['--source-root', str(source.resolve())]
                        if args.science_seal:
                            ref_command += ['--science-seal', str(args.science_seal.resolve())]
                        reference = reference_dir/'result.json'
                        if not (args.resume and reference.exists()):
                            with (reference_dir/'run.log').open('w') as stream:
                                subprocess.run(ref_command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                        if not reference.exists() or not eligible(json.loads(reference.read_text())):
                            unavailable(backend, scope, 'Fresh one-worker required-output qualification failed', reference)
                            continue
                    run(backend, chosen['workers'], chosen['batch_size'], scope, reference)
        if 'baseline' in args.backends and 'candidate' in args.backends:
            plan['baseline_candidate_spectra'] = cross_baseline_qualification(args.output, plan['configs'])
            for check in plan['baseline_candidate_spectra']['checks']:
                if not check['exact']:
                    unavailable('candidate', check['scope'], 'Paired baseline/optimized complete spectra differ')
            if args.stage == 'tune' and not plan['baseline_candidate_spectra']['passed']:
                plan['selected'].pop('candidate', None)
        plan['status'] = 'complete'
    except BaseException as error:
        plan.update(status='interrupted', error=repr(error))
        raise
    finally:
        plan['elapsed_this_invocation_seconds'] = time.perf_counter()-started
        write(receipt_path, plan)
        summary_table(args.output, plan['configs'], plan['unavailable'])


if __name__ == '__main__':
    main()
