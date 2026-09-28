#!/usr/bin/env python3
"""Validate and format existing survey inference and implementation receipts.

This renderer imports no scientific or GPU code and calculates no new intervals,
tests, thresholds, or pooled detection rates. Counts and hashes validate the
reported design; all inferential values are copied from the analysis JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

SNRS = (6., 8., 10., 12.)
SAMPLING = ('unsampled', 'one_event', 'two_events', 'three_plus_events',
            'one_to_four_points', 'grid_unreachable')
SNR_METRICS = ('native_family_white_snr', 'ideal_box_white_snr', 'native_family_ou_snr',
               'ideal_box_ou_snr', 'native_white_advantage', 'native_ou_advantage')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(value, lower=0., upper=1.):
    require(len(value) == 2 and all(math.isfinite(v) for v in value) and
            lower <= value[0] <= value[1] <= upper, 'Invalid stored interval')


def count(value, maximum, label):
    require(type(value) is int and 0 <= value <= maximum, 'Invalid count: ' + label)


def unique(rows, fields):
    values = {tuple(row[key] for key in fields): row for row in rows}
    require(len(values) == len(rows), 'Duplicate report rows: ' + '/'.join(fields))
    return values


def validate(recovery, seal, exactness, seal_hash, synthetic=False):
    require(all(bool(value.get('synthetic_fixture')) == synthetic
                for value in (recovery, seal, exactness)), 'Synthetic fixture requires explicit --synthetic mode')
    require(recovery['seal_sha256'] == seal_hash == exactness['identity']['seal_sha256'],
            'Recovery/exactness scientific seal identities differ')
    require(recovery['thresholds_sha256'] == exactness['identity']['thresholds_sha256'],
            'Recovery/exactness threshold identities differ')
    require(exactness['status'] == 'complete', 'Exactness execution is incomplete')
    regimes = seal['regimes']
    require(regimes and len(set(regimes)) == len(regimes), 'Missing/duplicate planned regimes')
    targets = (seal['target_fpr'], seal['secondary_target_fpr'])
    require(len(set(targets)) == 2 and all(0 < value < 1 for value in targets), 'Invalid planned FPRs')
    require(set(seal['bls_selected']) == set(regimes), 'Missing/extra frozen BLS selections')
    counts = seal['counts']
    require(all(type(counts[key]) is int and counts[key] > 0 for key in ('calibration', 'injections', 'nulls')),
            'Invalid planned population sizes')
    require(counts['injections'] % len(SNRS) == 0, 'Report schema requires the sealed balanced four-SNR design')
    methods = unique(recovery['methods'], ('regime', 'target_fpr', 'method'))
    required = {(regime, target, method) for regime in regimes for target in targets for method in ('tls', 'bls')}
    require(set(methods) == required, 'Missing/extra planned regime/method/FPR rows')
    strata = {}
    for (regime, target, label), row in methods.items():
        selection = dict(method='tls', ranker='native') if label == 'tls' else seal['bls_selected'][regime]
        require((row['configuration'], row['ranker']) == (selection['method'], selection['ranker']),
                'Reported method differs from frozen science selection')
        for denominator, numerator, rate, bounds, failures, planned in (
                ('n_injections', 'detected', 'recovery', 'recovery_interval95', 'failed_injections', 'injections'),
                ('n_nulls', 'false_positives', 'fpr', 'fpr_interval95', 'failed_nulls', 'nulls')):
            require(row[denominator] == counts[planned], 'Main report denominator differs from planned count')
            count(row[numerator], row[denominator], numerator)
            count(row[failures], row[denominator], failures)
            require(row[numerator] + row[failures] <= row[denominator], 'Failed execution counted as detection')
            require(math.isclose(row[rate], row[numerator]/row[denominator], abs_tol=1e-12),
                    'Stored rate disagrees with its counts')
            interval(row[bounds])
        count(row['aliases_including_fundamental'], row['n_injections'], 'aliases')
        calibration = row['calibration']
        require(calibration['n'] == counts['calibration'] and calibration['target_fpr'] == target and
                calibration['value'] == row['threshold'] and math.isfinite(row['threshold']),
                'Calibration identity/count/threshold differs from planned row')
        require(calibration['decision'] == 'strict exceedance', 'Unexpected threshold decision policy')
        require(1 <= calibration['rank_1based'] <= counts['calibration'], 'Invalid stored threshold rank')
        for key in ('calibration_scores_above', 'calibration_scores_at_threshold', 'calibration_zero_scores'):
            count(calibration[key], counts['calibration'], key)
        require(0 <= calibration['attainable_marginal_fpr'] <= target and
                calibration['attainable_marginal_fpr'] == calibration['marginal_fpr_upper_bound'],
                'Invalid stored attainable FPR bound')
        levels = unique(row['strata'], ('kind', 'level'))
        require(set(levels) <= {('snr', value) for value in SNRS} | {('sampling', value) for value in SAMPLING},
                'Unexpected subgroup level')
        require({key for key in levels if key[0] == 'snr'} == {('snr', value) for value in SNRS},
                'Missing planned SNR subgroup')
        for (kind, level), subgroup in levels.items():
            require(0 < subgroup['n'] <= counts['injections'], 'Invalid nonempty subgroup size')
            count(subgroup['detected'], subgroup['n'], 'subgroup detected')
            interval(subgroup['interval95'])
            if kind == 'snr':
                require(subgroup['n'] == counts['injections']//len(SNRS), 'SNR subgroup is not the planned balanced count')
        require(sum(levels['snr', value]['detected'] for value in SNRS) == row['detected'],
                'SNR subgroup detection counts do not match main row')
        strata[regime, target, label] = levels
    for regime in regimes:
        for target in targets:
            require({key: value['n'] for key, value in strata[regime, target, 'tls'].items()} ==
                    {key: value['n'] for key, value in strata[regime, target, 'bls'].items()},
                    'TLS/BLS subgroup input counts differ')
    contrasts = unique(recovery['contrasts'], ('regime', 'target_fpr'))
    require(set(contrasts) == {(r, f) for r in regimes for f in targets}, 'Missing/extra planned paired contrasts')
    for (regime, target), row in contrasts.items():
        for outcome, planned, endpoint in (('recovery', 'injections', 'detected'), ('fpr', 'nulls', 'false_positives')):
            for suffix in ('', '_simultaneous'):
                value = row['tls_minus_bls_' + outcome + suffix]
                require(value['n'] == counts[planned], 'Paired contrast denominator differs from plan')
                for key in ('first_only', 'second_only'):
                    count(value[key], value['n'], key)
                require(value['first_only'] + value['second_only'] <= value['n'], 'Invalid paired discordant counts')
                difference_count = methods[regime, target, 'tls'][endpoint] - methods[regime, target, 'bls'][endpoint]
                require(value['first_only'] - value['second_only'] == difference_count and
                        math.isclose(value['difference'], difference_count/value['n'], abs_tol=1e-12),
                        'Paired contrast disagrees with main counts')
                interval(value['interval'], -1., 1.)
                require(0 < value['confidence'] < 1, 'Invalid stored contrast confidence')
    receipts = unique(recovery['receipts'], ('path',))
    require(len(receipts) == 2*seal['execution_shards'], 'Missing/extra scientific execution receipts')
    require(Counter(row['split'] for row in receipts.values()) ==
            Counter({key: seal['execution_shards'] for key in ('injections', 'nulls')}), 'Wrong scientific receipt splits')
    require(all(row['production_sources'] == seal['production_sources'] for row in receipts.values()),
            'Scientific numerical source identity differs from seal')
    original_receipts = unique(exactness['identity']['candidate_receipts'], ('path',))
    require({key: value['sha256'] for key, value in receipts.items()} ==
            {key: value['sha256'] for key, value in original_receipts.items()},
            'Exactness did not compare the original scientific receipts')
    cases = unique(exactness['cases'], ('split', 'name'))
    planned_cases = {(regime, split): counts[split] for regime in regimes for split in ('injections', 'nulls')}
    require(Counter((row['regime'], row['split']) for row in cases.values()) == Counter(planned_cases),
            'Missing/extra planned baseline exactness cases')
    for row in cases.values():
        original = row['original_candidate']
        require((original['regime'], original['method'], original['name'], original['input_sha256']) ==
                (row['regime'], 'tls', row['name'], row['input_sha256']), 'Exactness original candidate input identity differs')
        comparison = row['comparison']
        require(comparison['exact'] == (not comparison['differences']), 'Exactness flag disagrees with recorded differences')
        require(not comparison['exact'] or (original['valid'] and row['baseline']['valid']),
                'Invalid execution cannot establish exactness')
        for side in ('original_candidate_decisions', 'baseline_decisions'):
            require(set(comparison[side]) == {'thresholds', 'secondary_thresholds'}, 'Missing frozen-threshold decisions')
            for key, target in zip(('thresholds', 'secondary_thresholds'), targets):
                decision = comparison[side][key]
                require(decision['target_fpr'] == target and decision['threshold'] == methods[row['regime'], target, 'tls']['threshold'],
                        'Exactness used another TLS operating point')
    mismatch_count = sum(not row['comparison']['exact'] for row in cases.values())
    require(exactness['completed_cases'] == len(cases) and exactness['mismatches'] == mismatch_count and
            exactness['exactness_qualified'] == (mismatch_count == 0), 'Exactness summary disagrees with original outcomes')
    require(exactness['incomplete_repeat_diagnostics'] == sum(row['repeat_status'] == 'pending' for row in cases.values()),
            'Exactness repeat-diagnostic summary differs')
    for regime in regimes:
        for split, endpoint in (('injections', 'detected'), ('nulls', 'false_positives')):
            for key, target in zip(('thresholds', 'secondary_thresholds'), targets):
                failures = sum(not row['original_candidate']['valid'] for row in cases.values()
                               if (row['regime'], row['split']) == (regime, split))
                require(failures == methods[regime, target, 'tls']['failed_'+split],
                        'Original exactness candidate failures disagree with the recovery report')
                detected = sum(row['comparison']['original_candidate_decisions'][key]['detected']
                               for row in cases.values() if (row['regime'], row['split']) == (regime, split))
                require(detected == methods[regime, target, 'tls'][endpoint],
                        'Original exactness candidate decisions disagree with the recovery report')
        injections = [row for row in cases.values() if (row['regime'], row['split']) == (regime, 'injections')]
        for key, target in zip(('thresholds', 'secondary_thresholds'), targets):
            for kind, levels in (('snr', SNRS), ('sampling', SAMPLING)):
                for level in levels:
                    selected = []
                    for row in injections:
                        original = row['original_candidate']
                        include = (original['white_oracle_snr'] == level if kind == 'snr' else
                            not original.get('grid_reachable', True) if level == 'grid_unreachable' else
                            original['in_transit_observations'] == 0 if level == 'unsampled' else
                            original['observed_events'] == 1 if level == 'one_event' else
                            original['observed_events'] == 2 if level == 'two_events' else
                            original['observed_events'] >= 3 if level == 'three_plus_events' else
                            0 < original['in_transit_observations'] < 5)
                        if include:
                            selected.append(row)
                    subgroup = strata[regime, target, 'tls'].get((kind, level))
                    require((subgroup['n'] if subgroup else 0) == len(selected),
                            'Reported subgroup size differs from original input membership')
                    require((subgroup['detected'] if subgroup else 0) == sum(
                        row['comparison']['original_candidate_decisions'][key]['detected'] for row in selected),
                        'Reported TLS subgroup detections differ from original outcomes')
    return regimes, targets, methods, strata, contrasts


def snr_distributions(snr, recovery, seal, exactness, seal_hash, synthetic):
    """Descriptive observed distributions, with groups from original TLS decisions."""
    require(bool(snr.get('synthetic_fixture')) == synthetic, 'SNR fixture marker differs')
    require(snr['status'] == 'complete' and snr['split'] == 'injections' and snr['seal_sha256'] == seal_hash,
            'Incomplete or foreign held-out SNR diagnostics')
    manifests = {row['manifest_sha256'] for row in recovery['receipts'] if row['split'] == 'injections'}
    require(manifests == {snr['manifest_sha256']}, 'SNR and recovery injection manifest identities differ')
    cases = {row['name']: row for row in exactness['cases'] if row['split'] == 'injections'}
    rows = {key[0]: value for key, value in unique(snr['rows'], ('name',)).items()}
    require(set(rows) == set(cases), 'Missing/extra held-out SNR case membership')
    for name, row in rows.items():
        require((row['regime'], row['input_sha256']) == (cases[name]['regime'], cases[name]['input_sha256']),
                'SNR case input identity differs from original search')
        for key in SNR_METRICS:
            require(row[key] is None or math.isfinite(row[key]), 'Nonfinite measured SNR diagnostic')
    distributions, joined = [], []
    for name, row in rows.items():
        decisions = cases[name]['comparison']['original_candidate_decisions']
        joined.append(dict(name=name, regime=row['regime'], input_sha256=row['input_sha256'],
            original_tls_valid=cases[name]['original_candidate']['valid'],
            primary_tls_detected=decisions['thresholds']['detected'],
            secondary_tls_detected=decisions['secondary_thresholds']['detected'],
            **{key: row[key] for key in SNR_METRICS}))
    for regime in seal['regimes']:
        population = [row for row in joined if row['regime'] == regime]
        groups = [('all', None, population)]
        for column, target in (('primary_tls_detected', seal['target_fpr']),
                               ('secondary_tls_detected', seal['secondary_target_fpr'])):
            for detected in (True, False):
                groups.append(('tls_detected' if detected else 'tls_missed_including_failures', target,
                               [row for row in population if row[column] == detected]))
        for group, target, selected in groups:
            for metric in SNR_METRICS:
                values = [row[metric] for row in selected if row[metric] is not None]
                distributions.append(dict(regime=regime, target_fpr=target, group=group, metric=metric,
                    group_n=len(selected), finite_n=len(values),
                    observed_median=median(values) if values else None,
                    observed_minimum=min(values) if values else None,
                    observed_maximum=max(values) if values else None))
    return distributions, joined


def csv_file(path, rows, fieldnames=None):
    fieldnames = list(rows[0]) if rows else fieldnames
    require(bool(fieldnames), 'Empty CSV needs its declared schema')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(value).replace('|', '\\|') for value in row) + ' |' for row in rows])


def rate_cell(k, n, bounds):
    return f'{k}/{n} ({100*k/n:.2f}%; {100*bounds[0]:.2f}–{100*bounds[1]:.2f}%)' if n else '0/0 — unrepresented'


def contrast_cell(value):
    return f"{100*value['difference']:+.2f} [{100*value['interval'][0]:+.2f}, {100*value['interval'][1]:+.2f}]"


def render(args):
    inputs = {key: Path(getattr(args, key)) for key in ('recovery', 'seal', 'exactness')}
    recovery, seal, exactness = (json.loads(inputs[key].read_text()) for key in inputs)
    regimes, targets, methods, strata, contrasts = validate(recovery, seal, exactness, sha(inputs['seal']), args.synthetic)
    snr_rows = None
    if getattr(args, 'snr', None):
        inputs['snr'] = Path(args.snr)
        snr = json.loads(inputs['snr'].read_text())
        snr_rows, snr_cases = snr_distributions(snr, recovery, seal, exactness, sha(inputs['seal']), args.synthetic)
    output = Path(args.output)
    source_records = {key: dict(path=str(path.resolve()), sha256=sha(path)) for key,path in inputs.items()}
    if (output/'provenance.json').exists():
        previous = json.loads((output/'provenance.json').read_text())
        require(previous['sources'] == source_records and previous['synthetic_fixture'] == args.synthetic,
                'Output directory already belongs to different source inputs or fixture mode')
    output.mkdir(parents=True, exist_ok=True)
    main, thresholds, subgroups, paired = [], [], [], []
    for regime in regimes:
        for target in targets:
            for label in ('tls', 'bls'):
                row = methods[regime, target, label]
                main.append({key: row[key] for key in ('regime', 'target_fpr', 'method', 'configuration', 'ranker',
                    'detected', 'n_injections', 'recovery', 'false_positives', 'n_nulls', 'fpr',
                    'failed_injections', 'failed_nulls', 'aliases_including_fundamental')})
                for endpoint in ('recovery', 'fpr'):
                    main[-1].update({endpoint+'_interval95_lower': row[endpoint+'_interval95'][0],
                                     endpoint+'_interval95_upper': row[endpoint+'_interval95'][1]})
                thresholds.append(dict(regime=regime, method=label, configuration=row['configuration'],
                                       ranker=row['ranker'], **row['calibration']))
                for kind, levels in (('snr', SNRS), ('sampling', SAMPLING)):
                    for level in levels:
                        value = strata[regime, target, label].get((kind, level))
                        subgroups.append(dict(regime=regime, target_fpr=target, method=label, kind=kind, level=level,
                            status='reported' if value else 'unrepresented', n=value['n'] if value else 0,
                            detected=value['detected'] if value else 0,
                            interval95_lower=value['interval95'][0] if value else '',
                            interval95_upper=value['interval95'][1] if value else ''))
            for endpoint in ('recovery', 'fpr'):
                for bound, suffix in (('marginal', ''), ('simultaneous_family', '_simultaneous')):
                    value = contrasts[regime, target]['tls_minus_bls_'+endpoint+suffix]
                    paired.append(dict(regime=regime, target_fpr=target, endpoint=endpoint, bound=bound,
                        n=value['n'], tls_only=value['first_only'], bls_only=value['second_only'],
                        difference=value['difference'], interval_lower=value['interval'][0], interval_upper=value['interval'][1],
                        stored_individual_confidence=value['confidence'], construction=value['construction']))
    exact_counts, mismatches = [], []
    for regime in regimes:
        for split in ('injections', 'nulls'):
            rows = [row for row in exactness['cases'] if row['regime'] == regime and row['split'] == split]
            exact_counts.append(dict(regime=regime, split=split, planned=seal['counts'][split], compared=len(rows),
                exact=sum(row['comparison']['exact'] for row in rows),
                mismatches=sum(not row['comparison']['exact'] for row in rows),
                candidate_invalid=sum(not row['original_candidate']['valid'] for row in rows),
                baseline_invalid=sum(not row['baseline']['valid'] for row in rows),
                pending_repeat_diagnostics=sum(row['repeat_status'] == 'pending' for row in rows)))
            for row in rows:
                if not row['comparison']['exact']:
                    mismatches.append(dict(regime=regime, split=split, name=row['name'], input_sha256=row['input_sha256'],
                        differences='; '.join(row['comparison']['differences']),
                        candidate_valid=row['original_candidate']['valid'], baseline_valid=row['baseline']['valid'],
                        candidate_error=row['original_candidate'].get('error'), baseline_error=row['baseline'].get('error'),
                        repeat_status=row['repeat_status']))
    for name, rows in (('recovery_fpr.csv', main), ('thresholds.csv', thresholds), ('subgroups.csv', subgroups),
                       ('paired_contrasts.csv', paired), ('exactness.csv', exact_counts), ('exactness_mismatches.csv', mismatches)):
        csv_file(output/name, rows, fieldnames=('regime','split','name','input_sha256','differences',
            'candidate_valid','baseline_valid','candidate_error','baseline_error','repeat_status'))
    if snr_rows is not None:
        csv_file(output/'snr_descriptive.csv', snr_rows)
        csv_file(output/'snr_cases.csv', snr_cases)
    title = 'SYNTHETIC FIXTURE — NOT A SCIENTIFIC RESULT' if args.synthetic else 'Survey recovery and implementation qualification'
    text = [f'# {title}',
        'These tables format the existing sealed analysis. Rates remain separate by regime; interval bounds are copied from the source JSON. '
        'No new inferential statistics or pooled detection rates are calculated.',
        'Native TLS versus the selected native GPU BLS measures blind detection at separately calibrated operating points. '
        'Baseline versus optimized TLS exactness is a separate comparison of the original held-out executions. '
        'Package SDE, BLS power and expected matched-filter SNR are not interchangeable.',
        recovery['limitation'],
        '## Frozen BLS control',
        table(['Regime', 'Selected configuration', 'Ranker'],
              [(r, seal['bls_selected'][r]['method'], seal['bls_selected'][r]['ranker']) for r in regimes])]
    for target in targets:
        title = 'Primary' if target == targets[0] else 'Secondary'
        text += [f'## {title} operating point: {100*target:g}% target FPR',
            'Recovery and observed FPR cells show successes/denominator, rate, and the existing 95% marginal interval. '
            'Failed executions remain in each planned denominator; a failure is not a detection.',
            table(['Regime', 'TLS recovery', 'BLS recovery', 'TLS observed FPR', 'BLS observed FPR'],
                [[regime] + [rate_cell(methods[regime,target,label][k], methods[regime,target,label][n],
                                     methods[regime,target,label][ci])
                    for k,n,ci in (('detected','n_injections','recovery_interval95'),('false_positives','n_nulls','fpr_interval95'))
                    for label in ('tls','bls')] for regime in regimes]),
            'Paired differences below are TLS minus BLS in percentage points. Both marginal and the existing simultaneous-family bounds are shown; '
            'an interval crossing zero does not establish an advantage. These intervals do not establish sub-percentage equivalence.',
            table(['Regime', 'Recovery: marginal', 'Recovery: simultaneous', 'FPR: marginal', 'FPR: simultaneous'],
                  [[r] + [contrast_cell(contrasts[r,target]['tls_minus_bls_'+key]) for key in
                           ('recovery','recovery_simultaneous','fpr','fpr_simultaneous')] for r in regimes])]
        calibration = [methods[r,target,m]['calibration'] for r in regimes for m in ('tls','bls')]
        bounds = ', '.join(f'{100*v:.4f}%' for v in sorted({c['attainable_marginal_fpr'] for c in calibration}))
        text += [f"Independent calibration used {seal['counts']['calibration']} nulls per method and regime, with strict threshold exceedance. "
            f'Stored attainable marginal FPR bound(s): {bounds}. This discrete bound is marginal over calibration sets, not certainty about '
            'the conditional FPR of the realized threshold. Ties can make the operating point more conservative. '
            'All scores, ranks, exceedance counts, tie counts and zero-score counts are in [thresholds.csv](thresholds.csv).',
            table(['Regime', 'Method', 'Failed injections', 'Failed nulls', 'Ties at threshold', 'Extra tie conservatism'],
                  [[r, m, methods[r,target,m]['failed_injections'], methods[r,target,m]['failed_nulls'],
                    methods[r,target,m]['calibration']['calibration_scores_at_threshold'],
                    methods[r,target,m]['calibration']['extra_conservatism_from_ties']] for r in regimes for m in ('tls','bls')])]
        for kind, levels in (('snr', SNRS), ('sampling', SAMPLING)):
            text += [f'### {title} {"target white-noise oracle SNR" if kind == "snr" else "sampling"} subgroups',
                     'Sampling groups overlap; their counts must not be added. An unrepresented group has no estimated recovery interval.'
                     if kind == 'sampling' else 'These are preassigned latent target SNR levels. Unsampled signals can have realized SNR zero and remain in their assigned groups. The held-out diagnostic computes the realized centered signal norm; none of these quantities is a package-reported detection score.',
                     table(['Regime', 'Level', 'TLS recovery', 'BLS recovery'],
                         [[r, level] + [rate_cell(value['detected'], value['n'], value['interval95']) if value else '0/0 — unrepresented'
                          for value in (strata[r,target,m].get((kind,level)) for m in ('tls','bls'))]
                          for r in regimes for level in levels])]
    text += ['## Comparable expected-SNR diagnostics']
    if snr_rows is None:
        text.append('No held-out expected-SNR artifact was supplied for this rendering.')
    else:
        text += ['The native family and ideal box are evaluated at the known period with the same sampled signal, weights, '
                 'and fitted constant. Templates are selected by the white diagonal-error matched-filter objective; '
                 'their white responses are the enumerated family ceilings. OU values evaluate those same white-selected '
                 'filters using the actual OU covariance variance, not an independently OU-optimized family maximum. '
                 'The white native-family optimum is an optimistic ceiling: the actual blind search and native depth/ranking need not attain it. '
                 'These are descriptive diagnostics, not package SNR/SDE values or a measured blind-search advantage.',
                 'Cells show the observed median relative native-family/ideal-box advantage and observed minimum–maximum, in percent; '
                 'these ranges are not confidence intervals. Finite/total counts expose undefined ratios, including zero-signal cases. '
                 'Detected/missed groups use original TLS decisions; misses include invalid executions and do not isolate a causal effect. '
                 'No new tests, approximation allowances, or inferential intervals are calculated.']
        lookup = {(row['regime'],row['target_fpr'],row['group'],row['metric']): row for row in snr_rows}
        def snr_cell(row):
            if not row['finite_n']:
                return f"0/{row['group_n']} finite — unavailable"
            return (f"{row['finite_n']}/{row['group_n']} finite; {100*row['observed_median']:+.3f}% "
                    f"[{100*row['observed_minimum']:+.3f}, {100*row['observed_maximum']:+.3f}]")
        for target, groups in ((None, ('all',)), (targets[0], ('tls_detected','tls_missed_including_failures')),
                              (targets[1], ('tls_detected','tls_missed_including_failures'))):
            text += [('All held-out injections.' if target is None else f'Original TLS decisions at {100*target:g}% target FPR.'),
                     table(['Regime', 'Group', 'White-noise family/box advantage', 'OU-noise family/box advantage'],
                           [[regime, group] + [snr_cell(lookup[regime,target,group,metric]) for metric in
                            ('native_white_advantage','native_ou_advantage')] for regime in regimes for group in groups])]
        text.append('Full native/box SNR distributions are in [snr_descriptive.csv](snr_descriptive.csv); '
                    'the measured case values and original decision join are in [snr_cases.csv](snr_cases.csv).')
    text += ['## Baseline versus optimized TLS: finite implementation qualification',
        ('Every planned original held-out comparison met the exactness gate.' if exactness['exactness_qualified'] else
         '**Aggregate exactness is withheld. Original mismatches or unavailable valid executions remain failures, regardless of diagnostic repeats.**'),
        'This checks the full available period/chi-squared/mask hashes, selected period/SDE, recovery and both frozen-threshold decisions. '
        'It does not establish universal numerical or physical equivalence. BLS is absent from this comparison.',
        table(['Regime', 'Split', 'Planned', 'Compared', 'Exact', 'Mismatches', 'Candidate invalid', 'Baseline invalid', 'Pending repeats'],
              [list(row.values()) for row in exact_counts]),
        'Individual implementation failures are retained in [exactness_mismatches.csv](exactness_mismatches.csv); the original source JSON '
        'retains every diagnostic repeat and any full-array mismatch artifacts.',
        '## Machine-readable tables and provenance',
        '[Recovery/FPR](recovery_fpr.csv), [paired contrasts](paired_contrasts.csv), [all subgroups](subgroups.csv), '
        '[thresholds](thresholds.csv), [per-regime exactness](exactness.csv), [provenance](provenance.json).',
        'All interval bounds in the CSVs preserve the original JSON values. Displayed percentages are rounded only for readability.',
        table(['Source', 'SHA256'], [(key, sha(path)) for key,path in inputs.items()] + [('renderer', sha(__file__))])]
    (output/'RECOVERY.md').write_text('\n\n'.join(text)+'\n')
    provenance = dict(purpose='Format-only rendering of existing sealed inference and exactness receipts',
        synthetic_fixture=args.synthetic, renderer_sha256=sha(__file__),
        sources=source_records,
        validated=dict(regimes=regimes, target_fprs=targets, method_rows=len(methods), contrast_rows=len(contrasts),
                       exactness_cases=len(exactness['cases']), counts_per_regime=seal['counts']),
        outputs={name:sha(output/name) for name in
                 ('RECOVERY.md','recovery_fpr.csv','thresholds.csv','subgroups.csv','paired_contrasts.csv','exactness.csv',
                  'exactness_mismatches.csv') + (('snr_descriptive.csv','snr_cases.csv') if snr_rows is not None else ())})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='rendered', output=str(output.resolve()), **provenance['validated'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovery', 'seal', 'exactness', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--snr', type=Path, help='Optional complete held-out white/OU expected-SNR diagnostic JSON')
    parser.add_argument('--synthetic', action='store_true', help='Require fixture markers and visibly watermark every Markdown report')
    render(parser.parse_args())


if __name__ == '__main__':
    main()
