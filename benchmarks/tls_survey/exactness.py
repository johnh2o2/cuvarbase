#!/usr/bin/env python3
"""Qualify every held-out TLS result against an immutable baseline, separately.

Freeze this operational plan alongside the science seal before held-out inputs
exist. The original candidate receipt is always the primary comparison; repeat
diagnostics never replace either original outcome.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from common import ROOT, array_hash, load_case, now, recovered, sha, source_identity, write
from campaign import check_manifest, check_result


def package_identity(root):
    package = Path(root) / 'cuvarbase'
    return {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.cu', '.cuh')}


def candidate_decisions(row, metadata, thresholds):
    candidate = row.get('candidates', {}).get('native', {})
    value = candidate.get('score')
    decisions = {}
    for key in ('thresholds', 'secondary_thresholds'):
        cut = thresholds[key][metadata['regime'] + '/tls']
        above = bool(row['valid'] and value is not None and value > cut['value'])
        decisions[key] = dict(target_fpr=cut['target_fpr'], threshold=cut['value'], above=above,
                              detected=above and (metadata['null'] or candidate.get('recovered', False)))
    return decisions


def compare(original, baseline, metadata, thresholds):
    differences = []
    if not original['valid'] or not baseline['valid']:
        differences.append('unavailable_valid_execution')
    if original['valid'] != baseline['valid']:
        differences.append('validity')
    for field in sorted(set(original['spectra']) | set(baseline['spectra'])):
        if original['spectra'].get(field) != baseline['spectra'].get(field):
            differences.append('spectrum/' + field)
    if original.get('candidates', {}).get('native') != baseline.get('candidates', {}).get('native'):
        differences.append('candidate_period_score_recovery')
    first = candidate_decisions(original, metadata, thresholds)
    second = candidate_decisions(baseline, metadata, thresholds)
    if first != second:
        differences.append('frozen_threshold_decisions')
    return dict(exact=not differences, differences=differences,
                original_candidate_decisions=first, baseline_decisions=second)


def baseline_search(arrays, metadata):
    """Capture the original full available periodogram before computing hashes."""
    from cuvarbase.tls import tls_search_gpu
    import pycuda.driver as driver
    result = tls_search_gpu(arrays['t'], arrays['y'], arrays['dy'], periods=arrays['periods'],
                            return_arrays=True, **metadata['search_kwargs'])
    driver.Context.synchronize()
    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None
    native = dict(period=finite(result['period']), score=finite(result['SDE']))
    native['successful_no_candidate'] = native['period'] is None and native['score'] == 0.
    native['no_candidate_reason'] = result.get('error') if native['successful_no_candidate'] else None
    native['recovered'] = recovered(native['period'], metadata)
    native['alias_recovered'] = recovered(native['period'], metadata, aliases=True)
    arrays_out = {key: np.asarray(np.ma.filled(result[key], np.nan)) for key in ('periods', 'chi2')}
    arrays_out['valid_mask'] = np.isfinite(np.ma.filled(result['chi2'], np.nan))
    valid = native['score'] is not None and (native['period'] is not None or native['successful_no_candidate'])
    return dict(valid=valid, candidates={'native': native}, spectra={
        key: array_hash(value) for key, value in arrays_out.items()}, error=None), arrays_out


def measured_search(arrays, metadata):
    begin = time.perf_counter()
    try:
        row, spectra = baseline_search(arrays, metadata)
    except Exception:
        row = dict(valid=False, candidates={}, spectra={}, error=traceback.format_exc())
        spectra = {}
    row['elapsed_s'] = time.perf_counter() - begin
    return row, spectra


def freeze(args):
    if any((args.campaign / ('inputs-' + split)).exists()
           for split in ('calibration', 'injections', 'nulls')):
        raise ValueError('Freeze the implementation plan before any final campaign inputs exist')
    seal = json.loads(args.seal.read_text())
    if source_identity() != seal['source_identity'] or package_identity(ROOT) != seal['production_sources']:
        raise ValueError('Candidate sources differ from the proposed science seal')
    baseline = package_identity(args.baseline_root)
    if not baseline or args.baseline_root.resolve() == ROOT.resolve():
        raise ValueError('Provide a separate immutable baseline checkout')
    count = sum(seal['counts'][split] for split in ('injections', 'nulls')) * len(seal['regimes'])
    if args.out.exists():
        raise ValueError('Refuse to overwrite the frozen implementation-qualification plan')
    write(args.out, dict(schema_version=1, created_utc=now(), seal_sha256=sha(args.seal),
        protocol_sha256=sha(__file__), baseline_root=str(args.baseline_root.resolve()),
        heldout_snr_protocol_sha256=sha(ROOT / 'benchmarks/tls_survey/heldout_snr.py'),
        planned_campaign_root=str(args.campaign.resolve()),
        baseline_sources=baseline, expected_cases=count, splits=['injections', 'nulls'], workers=1,
        primary_reference='Original candidate scientific receipts; never replace with reruns',
        acceptance='Every valid execution, full available period/chi2/valid-mask hash, selected period/SDE, recovery, and both frozen-threshold decisions identical. Any mismatch withholds aggregate exactness.',
        repeat_diagnostics=dict(first_mismatching_cases=10, additional_baseline_runs=2,
                                policy='Retain every repeat separately; never overwrite original outcomes'),
        estimated_extra_hours=4.837752061155108 * count / 5120,
        estimated_extra_gpu_usd=.49 * 4.837752061155108 * count / 5120,
        estimate_basis='80 immutable-baseline development calls, one worker; planning estimate, not sustained throughput.'))


def run(args):
    plan = json.loads(args.plan.read_text()); seal = json.loads(args.seal.read_text())
    if sha(args.plan) != args.plan_sha256 or sha(args.seal) != plan['seal_sha256']:
        raise ValueError('Plan or science seal differs from explicitly reviewed identity')
    if sha(__file__) != plan['protocol_sha256']:
        raise ValueError('Implementation-qualification protocol changed after freeze')
    if args.campaign.resolve() != Path(plan['planned_campaign_root']).resolve():
        raise ValueError('Campaign directory differs from the pre-input frozen plan')
    baseline_root = Path(getattr(args, 'baseline_root', None) or plan['baseline_root']).resolve()
    def guard():
        if (source_identity() != seal['source_identity'] or
                package_identity(ROOT) != seal['production_sources'] or
                package_identity(baseline_root) != plan['baseline_sources']):
            raise ValueError('Candidate or immutable baseline sources changed')
    guard()
    thresholds_path = args.campaign / 'thresholds.json'
    thresholds = json.loads(thresholds_path.read_text())
    if thresholds['seal_sha256'] != sha(args.seal):
        raise ValueError('Thresholds belong to another scientific design')
    entries = []; originals = {}; receipt_identities = []
    for split in plan['splits']:
        manifest_path = args.campaign / ('inputs-' + split) / 'manifest.json'
        manifest = check_manifest(manifest_path, split, seal, sha(args.seal))
        entries.extend((split, manifest_path.parent, entry) for entry in manifest['cases'])
        for shard in range(seal['execution_shards']):
            path = args.campaign / (split + '-search-' + str(shard) + '.json')
            receipt = check_result(path, manifest_path, manifest, seal, shard)
            receipt_identities.append(dict(path=str(path), sha256=sha(path)))
            for row in receipt['cases']:
                if row['method'] == 'tls':
                    key = (split, row['name'])
                    if key in originals:
                        raise ValueError('Duplicate original candidate outcome')
                    originals[key] = row
    if len(entries) != plan['expected_cases'] or len(originals) != len(entries):
        raise ValueError('Missing planned held-out implementation comparisons')
    identity = dict(plan_sha256=sha(args.plan), seal_sha256=sha(args.seal),
                    thresholds_sha256=sha(thresholds_path), candidate_receipts=receipt_identities)
    state = json.loads(args.out.read_text()) if args.out.exists() else dict(
        started_utc=now(), identity=identity, protocol_sha256=sha(__file__), cases=[])
    if state['identity'] != identity:
        raise ValueError('Cannot resume qualification with altered original receipts or thresholds')
    done = {(row['split'], row['name']) for row in state['cases']}
    if len(done) != len(state['cases']) or not done.issubset(originals):
        raise ValueError('Duplicate or foreign qualification rows')
    # One isolated process imports only the immutable baseline numerical package.
    sys.path.insert(0, str(baseline_root))
    import cuvarbase
    if Path(cuvarbase.__file__).resolve().parent != (baseline_root / 'cuvarbase').resolve():
        raise ValueError('Numerical import escaped the immutable baseline checkout')
    from cuvarbase.base import ensure_context
    ensure_context()
    import pycuda.driver as driver
    state.update(status='running', gpu=str(driver.Context.get_device().name()), workers=1,
                 baseline_sources=plan['baseline_sources'])
    write(args.out, state)
    for split, folder, entry in entries:
        name = entry['metadata']['name']; key = (split, name)
        if key in done:
            continue
        arrays, metadata = load_case(folder, entry)
        original = originals[key]
        if original['input_sha256'] != entry['sha256']:
            raise ValueError('Original candidate searched another input')
        baseline, spectra = measured_search(arrays, metadata)
        primary = compare(original, baseline, metadata, thresholds)
        row = dict(split=split, name=name, regime=metadata['regime'], input_sha256=entry['sha256'],
                   original_candidate=original, baseline=baseline, comparison=primary, repeats=[],
                   repeat_status='pending' if not primary['exact'] else 'not_required')
        # Publish the original outcome before any optional diagnostic. A stopped
        # diagnostic must never cause its primary mismatch to be replaced.
        state['cases'].append(row)
        state.update(completed_cases=len(state['cases']), mismatches=sum(
            not item['comparison']['exact'] for item in state['cases']), heartbeat_utc=now())
        write(args.out, state)
        if not primary['exact']:
            assets = args.out.parent / (args.out.stem + '-mismatches'); assets.mkdir(exist_ok=True)
            artifact = assets / (name + '-original-baseline.npz')
            np.savez_compressed(artifact, **spectra)
            row['original_baseline_arrays'] = dict(path=str(artifact), sha256=sha(artifact))
            write(args.out, state)
            earlier = state['mismatches'] - 1
            if earlier < plan['repeat_diagnostics']['first_mismatching_cases']:
                for index in range(plan['repeat_diagnostics']['additional_baseline_runs']):
                    repeated, _ = measured_search(arrays, metadata)
                    row['repeats'].append(dict(index=index, baseline=repeated,
                        versus_original_candidate=compare(original, repeated, metadata, thresholds),
                        versus_original_baseline=compare(baseline, repeated, metadata, thresholds)))
                    write(args.out, state)
                row['repeat_status'] = 'complete'
            else:
                row['repeat_omission'] = 'Predeclared first-mismatch diagnostic cap reached'
                row['repeat_status'] = 'capped'
        write(args.out, state)
        print(json.dumps(dict(completed=len(state['cases']), planned=len(entries), mismatches=state['mismatches'])), flush=True)
    guard()
    state.update(status='complete', completed_utc=now(), exactness_qualified=state['mismatches'] == 0,
                 incomplete_repeat_diagnostics=sum(row['repeat_status'] == 'pending' for row in state['cases']),
                 interpretation='Finite held-out implementation qualification; no universal numerical or physical equivalence claim. Original failures remain failures regardless of repeat outcomes.')
    write(args.out, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    frozen = sub.add_parser('freeze'); frozen.add_argument('--baseline-root', type=Path, required=True)
    frozen.add_argument('--campaign', type=Path, required=True,
                        help='Planned final campaign directory, before any final inputs exist')
    executed = sub.add_parser('run'); executed.add_argument('--plan', type=Path, required=True)
    executed.add_argument('--plan-sha256', required=True)
    executed.add_argument('--campaign', type=Path, required=True)
    executed.add_argument('--baseline-root', type=Path,
                          help='Optional relocated checkout; every frozen source hash must still match')
    for command in (frozen, executed):
        command.add_argument('--seal', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'run':
        with args.out.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        freeze(args)


if __name__ == '__main__':
    main()
