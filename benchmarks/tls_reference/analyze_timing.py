#!/usr/bin/env python3
"""Normalize audited TLS timings for the public comparison figure.

Run the accuracy acceptance checks and timing receipt audit first. This step
requires both gates, preserves raw repetitions, and never uses failed calls
or the instrumented search boundary as public API timing denominators.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


PROFILES = {'tess_solar': 'tess_200s', 'tess_gap': 'tess_gap', 'ztf_solar': 'ztf'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_row(profile, method, mode, measured, count, workers):
    values = measured['raw_seconds']
    expected = 5 if mode == 'single' else 3
    if len(values) != expected or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Incomplete or invalid public timing repetitions')
    median = statistics.median(values)
    if median != measured['median_seconds']:
        raise ValueError('Timing median differs from its recorded repetitions')
    return dict(profile=profile, method=method, mode=mode,
                boundary='warm_public_api', n=count, workers=workers,
                repetitions=len(values), total_seconds=values,
                seconds_per_source=median/count,
                min_seconds_per_source=min(values)/count,
                max_seconds_per_source=max(values)/count)


def numerical_kind(manifest, acceptance, manifest_sha256):
    if 'studies' in manifest:
        merged = acceptance.get('merged_origin_checks', {})
        origins = merged.get('accepted_studies', {})
        if (merged.get('numerical_validation_passed') is not True or
                merged.get('merged_manifest_sha256') != manifest_sha256 or
                set(origins) != set(manifest['studies'])):
            raise ValueError('Merged numerical studies have not passed their original gates')
        for name, study in manifest['studies'].items():
            actual = origins[name]
            if (actual.get('manifest_sha256') != study['manifest_sha256'] or
                    actual.get('seal_sha256') != study['seal_sha256'] or
                    actual.get('production_sources') != manifest['source_identity']['production_sources'] or
                    actual.get('numerical_validation_passed') is not True):
                raise ValueError('Merged numerical origin differs from its accepted identity')
        reproduced = any(value['evidence_kind'] == 'reproduction' for value in origins.values())
        if (merged.get('evidence_kind') != ('reproduction' if reproduced else 'independent') or
                merged.get('publication_gate_passed') is not (not reproduced)):
            raise ValueError('Merged evidence cannot relabel a reproduction independent')
        return reproduced
    if 'publication_gate' in acceptance and 'reproduction_gate' in acceptance:
        raise ValueError('Independent and reproduced numerical evidence cannot be conflated')
    reproduced = 'reproduction_gate' in acceptance
    gate = 'reproduction_gate' if reproduced else 'publication_gate'
    if acceptance.get(gate, {}).get('pass') is not True:
        raise ValueError('The numerical-validation gate has not passed')
    if not reproduced and manifest.get('suite') == 'reproduction':
        raise ValueError('A reproduction cannot be relabeled independent')
    if acceptance.get('inputs_manifest_sha256') != manifest_sha256:
        raise ValueError('Numerical validation used a different input manifest')
    if reproduced and (acceptance.get('original_source_identity') != manifest['source_identity'] or
            acceptance.get('reproduction_sources', {}).get('production') !=
            manifest['source_identity']['production_sources']):
        raise ValueError('Reproduced timing evidence lacks its actual production-source identity')
    return reproduced


def analyze(checks, manifest, acceptance, manifest_sha256):
    measurement_scope = checks.get('measurement_scope', 'full')
    if measurement_scope not in ('full', 'single'):
        raise ValueError('Unknown timing measurement scope')
    reproduced = numerical_kind(manifest, acceptance, manifest_sha256)
    entries = {case['file']: case for case in manifest['cases']}
    rows, profiles, speedups, components = [], [], [], []
    for regime, profile in PROFILES.items():
        result = checks['regimes'][regime]
        single, batch = result['public_single'], result['public_batch']
        if single['eligible'] is not True or (measurement_scope == 'full' and batch['eligible'] is not True):
            raise ValueError('Public timing checks did not pass: ' + regime)
        if measurement_scope == 'single' and (batch.get('eligible') is not False or
                batch.get('status') != 'not_measured' or batch.get('source_count') != 0):
            raise ValueError('Single-source evidence cannot imply measured batch throughput')
        selection = checks['cohort_selection'][regime]
        if 'studies' in manifest and selection.get('accepted_study') != acceptance['merged_origin_checks']:
            raise ValueError('Timing selection and current accepted origins differ')
        if not reproduced and selection.get('accepted_study', {}).get('evidence_kind') == 'reproduction':
            raise ValueError('Reproduced timing selection cannot be relabeled independent')
        if reproduced:
            evidence = selection.get('accepted_study', {})
            if (evidence.get('evidence_kind') != 'reproduction' or
                    evidence.get('numerical_validation_passed') is not True or
                    evidence.get('publication_gate_passed') is not False):
                raise ValueError('Timing selection did not retain its reproduced-evidence classification')
        if selection['expected_candidate_sources'] != manifest['source_identity']['production_sources']:
            raise ValueError('Timed candidate sources differ from the validated manifest')
        examined = {case['case']: case for case in selection['examined']}
        cohort_names = selection['selected_cases']
        count = len(cohort_names)
        if (count < 1 or count > 16 or len(set(cohort_names)) != count or
                (measurement_scope == 'full' and batch['source_count'] != count) or
                selection['actual_batch_size'] != count):
            raise ValueError('Batch size must equal its distinct successful sources')
        if single['case'] != selection['single_case'] or single['case'] not in cohort_names:
            raise ValueError('Single-source identity differs from the selected cohort')
        names = [single['case']] if measurement_scope == 'single' else cohort_names
        metadata = entries[single['case']]['metadata']
        for name in names:
            entry = entries[name]
            if examined[name]['input_sha256'] != entry['sha256']:
                raise ValueError('Timing cohort used a different input array archive')
            if entry['metadata']['regime'] != regime:
                raise ValueError('Timing cohort contains another regime')
            for key in ('ndata', 'baseline_days', 'period_count', 'search_kwargs'):
                if entry['metadata'][key] != metadata[key]:
                    raise ValueError('Shared-grid timing cohort differs in ' + key)
            if entry['arrays']['periods'] != entries[single['case']]['arrays']['periods']:
                raise ValueError('Timing sources do not share an identical period array')
        workers = None
        if measurement_scope == 'full':
            workers = batch['strongest_tested_native_workers']
            if workers not in (1, 2, 4):
                raise ValueError('Native worker selection was not tested')
            native_pool = batch['native_pool_configurations'][str(workers)]
            if native_pool['eligible'] is not True:
                raise ValueError('Selected native pool did not preserve its search results')
            eligible = [(pool['elapsed']['median_seconds'], int(width))
                        for width, pool in batch['native_pool_configurations'].items()
                        if pool['eligible']]
            if min(eligible)[1] != workers:
                raise ValueError('Selected native pool is not the fastest eligible configuration')
        rows.extend([
            timing_row(profile, 'tls_v1', 'single', single['candidate'], 1, 1),
            timing_row(profile, 'gtls', 'single', single['native'], 1, 1),
        ])
        if measurement_scope == 'full':
            rows.extend([
                timing_row(profile, 'tls_v1', 'batch', batch['candidate'], count, 1),
                timing_row(profile, 'gtls', 'batch', batch['strongest_tested_native'], count, workers)])
        profiles.append(dict(profile=profile, regime=regime, n_samples=metadata['ndata'],
            baseline_days=metadata['baseline_days'], n_periods=metadata['period_count'],
            batch_size=count if measurement_scope == 'full' else None,
            supporting_cohort_size=count, gtls_workers=1 if measurement_scope == 'single' else workers,
            single_case=single['case'],
            input_sha256=entries[single['case']]['sha256']))
        speedups.append(dict(profile=profile, single=single['speedup'],
                             batch=batch['speedup'] if measurement_scope == 'full' else None,
                             gtls_batch_workers=workers))
        component = result.get('common_search_components')
        if component is None or component['eligible'] is not True:
            raise ValueError('Separate search/component validation did not pass: ' + regime)
        components.append(dict(profile=profile, **component))
        if selection.get('correction_timing', {}).get('required'):
            corrected = result.get('corrected_native_crosscheck') or {}
            kinds = ('single',) if measurement_scope == 'single' else ('single', 'batch')
            if any(corrected.get(kind, {}).get('eligible') is not True for kind in kinds):
                raise ValueError('Required corrected-native public timing did not pass: ' + regime)
            if result.get('corrected_common_search_components', {}).get('eligible') is not True:
                raise ValueError('Required corrected-native component timing did not pass: ' + regime)
    return dict(measurement_scope='single' if measurement_scope == 'single' else 'single_and_batch',
                verification=dict(complete=True, numerical_validation_complete=True,
                    numerical_evidence_kind='reproduction' if reproduced else 'independent',
                    exclusive_processes=True,
                    scope='Completed scientific acceptance and exclusive public-call receipt gates; '
                          'not a universal population recovery or false-positive guarantee.'),
                profiles=profiles, timings=rows, speedups=speedups, components=components,
                environment=checks['environment'], native_extras=checks['native_extras'],
                source_scope=checks['scientific_scope'],
                corrected_native_crosschecks={PROFILES[regime]: dict(
                    public=result.get('corrected_native_crosscheck'),
                    components=result.get('corrected_common_search_components'))
                    for regime, result in checks['regimes'].items()
                    if result.get('corrected_native_crosscheck') is not None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, acceptance = read(args.manifest), read(args.acceptance)
    if 'studies' in manifest:
        if __package__:
            from .timing.cohort import accepted_study
        else:
            from timing.cohort import accepted_study
        merged = accepted_study(args.manifest, args.acceptance.parent)
        if sha(args.acceptance) not in {value['receipt_sha256'] for value in merged['accepted_studies'].values()}:
            raise ValueError('The supplied acceptance is not one of the merged source studies')
        acceptance = dict(merged_origin_checks=merged)
    result = analyze(read(args.checks), manifest, acceptance, sha(args.manifest))
    result['sources'] = {name: dict(file=path.name, sha256=sha(path))
                         for name, path in (('timing_checks', args.checks),
                                            ('inputs', args.manifest),
                                            ('numerical_acceptance', args.acceptance))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
