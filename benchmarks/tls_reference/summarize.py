#!/usr/bin/env python3
"""Summarize every planned pair, retaining failures and per-regime uncertainty."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from validate import sha, write


def binomial_interval(successes, count, alpha=.05):
    if not 0 <= successes <= count:
        raise ValueError('Require 0 <= successes <= count')
    if count == 0:
        return [None, None]
    return [0. if successes == 0 else float(beta.ppf(alpha/2, successes, count-successes+1)),
            1. if successes == count else float(beta.ppf(1-alpha/2, successes+1, count-successes))]


def discordance_upper(discordances, count, alpha):
    if not 0 <= discordances <= count or not 0 < alpha < 1:
        raise ValueError('Invalid binomial count or tail probability')
    if count == 0:
        return None
    return 1. if discordances == count else float(beta.ppf(1-alpha, discordances+1, count-discordances))


def recovered(period, metadata, alias=1.):
    if period is None or not np.isfinite(period):
        return False
    truth = metadata['truth_period']*alias
    drift = abs(float(period)-truth)/truth*metadata['baseline_days']
    return drift <= .5*metadata['duration_days']


def read_case(case, root, threshold, reference_backend='gtls', comparison_file='compare.json'):
    metadata = case['metadata']
    name = metadata['name']
    row = dict(name=name, regime=metadata['regime'], cohort=metadata['cohort'], purpose=metadata['purpose'],
               null=metadata['null'], snr=metadata['latent_white_oracle_snr'],
               threshold=threshold, input_sha256=case['sha256'], observable=metadata['observable'],
               accepted_proposal=metadata['accepted_proposal'],
               below_approx_native_duration_envelope=metadata['below_approx_native_duration_envelope'])
    row['reference_backend'] = reference_backend
    for backend in ('gtls', 'candidate'):
        source_backend = reference_backend if backend == 'gtls' else backend
        path = root/name/source_backend/'record.json'
        if not path.exists():
            row[backend+'_status'] = 'missing'
            continue
        record = json.loads(path.read_text())
        if record['input_sha256'] != case['sha256']:
            raise ValueError('Recorded input hash differs from planned case: '+str(path))
        row[backend+'_record_sha256'] = sha(path)
        row[backend+'_status'] = record['status']
        row[backend+'_seconds'] = record.get('elapsed_seconds')
        if record['status'] != 'ok':
            row[backend+'_error'] = record.get('error')
            continue
        result = record['result']
        score, period = result.get('score'), result.get('period')
        detected = score is not None and score > threshold
        row.update({backend+'_mode': record['mode'], backend+'_score': score, backend+'_period': period,
                    backend+'_detected': detected,
                    backend+'_primary_period_match': recovered(period, metadata),
                    backend+'_alias_match': any(recovered(period, metadata, a) for a in (.5, 2., 2/3, 1.5)),
                    backend+'_recovered': detected and recovered(period, metadata),
                    backend+'_native_snr': result.get('final_fit', {}).get('native_gtls_snr')})
    compare = root/name/comparison_file
    if compare.exists():
        result = json.loads(compare.read_text())
        for backend, field in (('gtls', 'reference_record_sha256'), ('candidate', 'candidate_record_sha256')):
            if field in result and result[field] != row.get(backend+'_record_sha256'):
                raise ValueError('Comparison refers to different backend record: '+str(compare))
        row['comparison_passed'] = result['passed']
        row['comparison_sha256'] = sha(compare)
    row['paired_success'] = row.get('gtls_status') == row.get('candidate_status') == 'ok'
    row['reference_failure_extension'] = row.get('gtls_status') == 'error' and row.get('candidate_status') == 'ok'
    if metadata['purpose'] != 'recovery_full_grid':
        # No truth-inserted stress case contributes to a reported recovery rate.
        for backend in ('gtls', 'candidate'):
            row.pop(backend+'_recovered', None)
    return row


def strata(rows, alpha=.05):
    groups = defaultdict(list)
    for row in rows:
        if row['purpose'] == 'recovery_full_grid':
            key = (row['regime'], 'null_mixture_6_8_10_12' if row['null'] else 'snr_%g' % row['snr'])
            groups[key].append(row)
    family = len(groups)
    summaries = []
    for (regime, label), group in sorted(groups.items()):
        paired = [r for r in group if r['paired_success']]
        null = group[0]['null']
        outcome = 'detected' if null else 'recovered'
        rcount = sum(r['gtls_'+outcome] for r in paired)
        ccount = sum(r['candidate_'+outcome] for r in paired)
        reference_evaluable = [r for r in group if r['gtls_status'] == 'ok']
        reference_all_count = sum(r['gtls_'+outcome] for r in reference_evaluable)
        discord = sum((not r['gtls_detected'] and r['candidate_detected']) if null else
                      (r['gtls_recovered'] and not r['candidate_recovered']) for r in paired)
        recovery = dict(reference_backend=group[0].get('reference_backend','gtls'), regime=regime, stratum=label, outcome='false_positive' if null else 'recovery',
            planned=len(group), paired_success=len(paired),
            reference_failures=sum(r['gtls_status'] == 'error' for r in group),
            candidate_failures=sum(r['candidate_status'] == 'error' for r in group),
            missing=sum('missing' in (r['gtls_status'], r['candidate_status']) for r in group),
            reference_count=rcount, candidate_count=ccount,
            reference_evaluable_count=len(reference_evaluable),
            reference_all_successful_search_outcome_count=reference_all_count,
            reference_all_successful_search_outcome_rate=reference_all_count/len(reference_evaluable) if reference_evaluable else None,
            reference_rate=rcount/len(paired) if paired else None,
            candidate_rate=ccount/len(paired) if paired else None,
            reference_two_sided_95_interval=binomial_interval(rcount, len(paired)),
            candidate_two_sided_95_interval=binomial_interval(ccount, len(paired)),
            added_fp_or_lost_recovery=discord,
            discordance_simultaneous_upper=discordance_upper(discord, len(paired), alpha/max(1, family)),
            simultaneous_family=family, simultaneous_tail_alpha=alpha/max(1, family),
            complete=len(paired) == len(group),
            both_zero_observed_recovery=bool(not null and paired and rcount == 0 and ccount == 0),
            comparison_failures=sum(r.get('comparison_passed') is False for r in group),
            comparisons_missing=sum('comparison_passed' not in r for r in group))
        summaries.append(recovery)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--records-root', type=Path, required=True)
    parser.add_argument('--threshold', type=float, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; summaries are immutable')
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True)
    rows, summaries = [], []
    for threshold in args.threshold:
        selected = [read_case(c, args.records_root, threshold) for c in manifest['cases']]
        rows.extend(selected)
        # Multiple predeclared thresholds enlarge the simultaneous family.
        table = strata(selected, alpha=.05/len(args.threshold))
        summaries.extend([dict(r, threshold=threshold) for r in table])
    with (args.out/'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    write(args.out/'summary.json', dict(manifest_sha256=sha(args.manifest), thresholds=args.threshold,
         cohort=manifest['suite'], strata=summaries,
         mathematical_stress_cases=sum(c['metadata']['purpose'] == 'mathematical_differential' for c in manifest['cases']),
         warning='Rates use paired successful full-grid searches only; failures and missing planned cases remain explicit. '
                 'A small zero-discordance sample does not establish a tight recovery margin. '
                 'Descriptive thresholds are not a calibrated fixed false-positive rate.'))


if __name__ == '__main__':
    main()
