#!/usr/bin/env python3
"""Render all predeclared panels, including visibly unavailable competitors."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCOPES = ('tess_solar', 'tess_gap_long', 'ztf_solar', 'varied')
BACKENDS = ('baseline', 'candidate', 'gtls', 'bls')


def heldout_qualification(campaign, exactness_path, seal_path):
    """Bind the finite held-out qualification independently of timing eligibility."""
    seal_bytes = Path(seal_path).read_bytes()
    seal = json.loads(seal_bytes)
    seal_hash = hashlib.sha256(seal_bytes).hexdigest()
    exactness_bytes = Path(exactness_path).read_bytes()
    exactness = json.loads(exactness_bytes)
    if (campaign.get('science_seal_sha256') != seal_hash or
            exactness['identity']['seal_sha256'] != seal_hash):
        raise ValueError('Timing/held-out qualification belongs to a different science seal')
    regimes = seal['regimes']
    if not regimes or len(regimes) != len(set(regimes)):
        raise ValueError('Invalid planned held-out regimes')
    planned = {(regime, split): seal['counts'][split] for regime in regimes for split in ('injections','nulls')}
    if any(type(value) is not int or value <= 0 for value in planned.values()):
        raise ValueError('Invalid planned held-out counts')
    rows = exactness['cases']
    expected = sum(planned.values())
    if (exactness['status'] != 'complete' or exactness['completed_cases'] != expected or
            len(rows) != expected or Counter((row['regime'],row['split']) for row in rows) != Counter(planned) or
            len({(row['split'],row['name']) for row in rows}) != expected):
        raise ValueError('Held-out qualification does not cover every planned regime/input')
    for row in rows:
        comparison = row['comparison']
        if (type(comparison['exact']) is not bool or comparison['exact'] != (not comparison['differences']) or
                (comparison['exact'] and (not row['original_candidate']['valid'] or not row['baseline']['valid']))):
            raise ValueError('Held-out exactness flag contradicts the original outcome')
    mismatches = sum(not row['comparison']['exact'] for row in rows)
    if (exactness['mismatches'] != mismatches or
            exactness['exactness_qualified'] != (mismatches == 0)):
        raise ValueError('Held-out exactness summary contradicts the planned case outcomes')
    return dict(path=str(Path(exactness_path).resolve()),
        sha256=hashlib.sha256(exactness_bytes).hexdigest(),
        science_seal_path=str(Path(seal_path).resolve()), science_seal_sha256=seal_hash,
        auxiliary_plan_sha256=exactness['identity']['plan_sha256'],
        planned_cases=expected, compared_cases=len(rows), exact_cases=expected-mismatches,
        mismatches=mismatches, aggregate_exactness_qualified=bool(exactness['exactness_qualified']),
        interpretation='Original finite held-out qualification. Timing-cohort ratios do not establish global sensitivity preservation.')


def figure_csv(path, data, missing, paired, heldout):
    rows = []
    for scope in SCOPES:
        for backend in BACKENDS:
            record = data.get((backend,scope))
            rates = [row['lightcurves_per_second'] for row in record['repetitions']] if record else []
            ratio = None
            if backend == 'candidate' and record and paired.get(scope) and ('baseline',scope) in data:
                baseline = [row['lightcurves_per_second'] for row in data['baseline',scope]['repetitions']]
                ratio = float(np.median(rates)/np.median(baseline))
            rows.append(dict(scope=scope, backend=backend, timing_eligible=record is not None,
                failure_reason=missing.get((backend,scope)),
                median_lightcurves_per_second=float(np.median(rates)) if rates else None,
                minimum_lightcurves_per_second=min(rates) if rates else None,
                maximum_lightcurves_per_second=max(rates) if rates else None,
                timing_cohort_optimized_vs_baseline=ratio,
                heldout_exact_cases=heldout['exact_cases'], heldout_planned_cases=heldout['planned_cases'],
                heldout_aggregate_exactness_qualified=heldout['aggregate_exactness_qualified'],
                heldout_exactness_sha256=heldout['sha256'],
                science_seal_sha256=heldout['science_seal_sha256'],
                auxiliary_plan_sha256=heldout['auxiliary_plan_sha256']))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_campaign(path):
    campaign = json.loads(path.read_text())
    if campaign['stage'] != 'measure' or campaign.get('status') != 'complete':
        raise ValueError('Only completed independent sustained measurements supply the figure')
    data, missing = {}, {}
    for row in campaign.get('unavailable', []):
        missing[(row['backend'], row['scope'])] = row['reason']
    for row in campaign['configs']:
        key = (row['backend'], row['scope'])
        if row['scope'] not in SCOPES or row['backend'] not in BACKENDS:
            continue
        record_path = path.parent/row['result']
        if not row['eligible']:
            missing[key] = row.get('failure_reason', 'Required-output or ownership qualification failed')
            continue
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != row['result_sha256']:
            raise ValueError('Measurement record differs from campaign identity')
        record = json.loads(record_path.read_text())
        if (record['status'] != 'ok' or not record['gpu_ownership']['passed'] or
                len(record['qualification']) != 2 or
                not all(item['gate']['passed'] for item in record['qualification'])):
            raise ValueError('Campaign eligibility disagrees with numerical/ownership receipts')
        if len(record['repetitions']) < 3 or not all(item['status'] == 'ok' for item in record['repetitions']):
            raise ValueError('Three successful sustained repetitions required')
        data[key] = record
    paired = {}
    for check in campaign.get('baseline_candidate_spectra', {}).get('checks', []):
        paired[check['scope']] = paired.get(check['scope'], True) and check['exact']
    for scope, passed in paired.items():
        if not passed:
            data.pop(('candidate', scope), None)
            missing[('candidate', scope)] = 'Paired baseline/optimized complete-spectrum qualification failed'
    for backend in BACKENDS:
        for scope in SCOPES:
            if (backend, scope) not in data:
                missing.setdefault((backend, scope), 'No qualifying result in the predeclared campaign')
    if not data:
        raise ValueError('No qualified timing exists; retain the failure report without a performance figure')
    allocations = {(r['environment']['nvidia_smi'], r['environment']['cpu_quota_cores'],
                    r['environment']['host_memory_limit_bytes']) for r in data.values()}
    if len(allocations) != 1 or any(value is None for value in next(iter(allocations))):
        raise ValueError('The figure requires one verified GPU/CPU/memory allocation')
    return campaign, data, missing, paired, next(iter(allocations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--exactness', type=Path, required=True, help='Complete original held-out exactness receipt')
    parser.add_argument('--science-seal', type=Path, required=True, help='Original science seal with planned population counts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign, data, missing, paired, allocation = read_campaign(args.campaign)
    heldout = heldout_qualification(campaign,args.exactness,args.science_seal)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ('TESS\ndense sector', 'TESS\nseparated sectors', 'ZTF\ng/r', 'Varied sizes\n96-source queue')
    display = {'baseline': 'Baseline TLS', 'candidate': 'Optimized TLS',
               'gtls': 'Public GTLS', 'bls': 'Selected BLS'}
    colors = {'baseline': '#8b98a7', 'candidate': '#147d92', 'gtls': '#cb7950', 'bls': '#7759a0'}
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.8), layout='constrained')
    for ax, scope, title in zip(axes, SCOPES, labels):
        medians, observed = {}, []
        for offset, backend in enumerate(BACKENDS):
            record = data.get((backend, scope))
            if record is None:
                ax.text(offset, .03, 'No qualifying\nresult', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=7, rotation=90, color='#754646')
                continue
            rates = np.asarray([row['lightcurves_per_second'] for row in record['repetitions']])
            if np.any(rates <= 0) or not np.all(np.isfinite(rates)):
                raise ValueError('Nonpositive/nonfinite eligible throughput')
            median = float(np.median(rates))
            observed.extend(rates.tolist())
            medians[backend] = median
            ax.bar(offset, median, .7, color=colors[backend],
                   yerr=[[median-float(rates.min())], [float(rates.max())-median]], capsize=3,
                   error_kw=dict(elinewidth=1, ecolor='#303c46'))
            ax.text(offset, float(rates.max())*1.13, f'{median:.2f}', ha='center', fontsize=8)
        ax.set_xticks(range(4), ['Baseline', 'Optimized', 'GTLS', 'BLS'], rotation=35)
        ax.set_title(title, fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(min(observed)/2 if observed else .01, max(observed)*3 if observed else 1)
        ax.set_xlim(-.6, 3.6)
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis='y', which='major', alpha=.15)
        ax.set_axisbelow(True)
        if paired.get(scope) and all(b in medians for b in ('baseline', 'candidate')):
            message = f"Optimized / baseline: {medians['candidate']/medians['baseline']:.2f}×"
        else:
            message = 'No qualified baseline / optimized ratio'
        ax.text(.5, .98, message, transform=ax.transAxes, ha='center', va='top', fontsize=8,
                color=colors['candidate'])
    axes[0].set_ylabel('Completed light curves per second · log scale')
    status = ('finite tested scope passed' if heldout['aggregate_exactness_qualified'] else 'AGGREGATE EXACTNESS WITHHELD')
    fixture = 'SYNTHETIC FIXTURE · ' if campaign.get('synthetic_fixture') else ''
    fig.suptitle(fixture+'Sustained full transit search · '+allocation[0].split(',')[0]+'\n'
        f"Full held-out TLS exactness: {heldout['exact_cases']}/{heldout['planned_cases']} · {status}",
        fontsize=14, weight='bold', color='#253b46' if heldout['aggregate_exactness_qualified'] else '#9b2424')
    settings = []
    for backend in BACKENDS:
        chosen = campaign['selected'].get(backend)
        settings.append(f"{display[backend]}: {chosen['workers']} workers, batch {chosen['batch_size']}"
                        if chosen else f'{display[backend]}: no qualifying development setting')
    fig.supxlabel('Median of 3 queues; whiskers: observed repeat range. Each queue: ≥96 curves AND ≥120 s.\n'
                  'Logarithmic, separate y scales. Independently selected pool/batch settings; full search and transfers included.\n'
                  'BLS uses science-selected settings/ranking; throughput does not imply equal detection sensitivity.\n'
                  'Ratios qualify their timing cohorts; they do not establish global sensitivity preservation.\n'
                  + '; '.join(settings[:2])+'\n'+'; '.join(settings[2:]), fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output.with_suffix('.'+extension), dpi=180)
    figure_csv(args.output.with_suffix('.csv'),data,missing,paired,heldout)
    args.output.with_suffix('.data.json').write_text(json.dumps(dict(
        campaign=str(args.campaign), campaign_sha256=hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        heldout_exactness=heldout,
        csv_sha256=hashlib.sha256(args.output.with_suffix('.csv').read_bytes()).hexdigest(),
        outputs={args.output.with_suffix('.'+extension).name:
                 hashlib.sha256(args.output.with_suffix('.'+extension).read_bytes()).hexdigest()
                 for extension in ('png','pdf','svg','csv')},
        measurement_records=[dict(path=str((args.campaign.parent/row['result']).resolve()),sha256=row['result_sha256'])
            for row in campaign['configs'] if (row['backend'],row['scope']) in data],
        allocation=allocation, source_records={f'{a}/{b}': r['summary'] for (a,b),r in data.items()},
        numerical_contracts=dict(tls='Exact complete spectra, masks and selected endpoints; paired baseline/optimized gate',
            bls='Exact period arrays, finite masks and selected endpoints; nonwinning powers and unused rankers diagnostic'),
        bls_native_repeat_diagnostics={scope: [dict(phase=item['phase'],
            **{key: item['native_repeat_diagnostics'][key] for key in (
                'changed_power_comparisons', 'changed_selected_endpoints', 'max_absolute_power_difference')})
            for item in record['qualification'] if 'native_repeat_diagnostics' in item]
            for (backend, scope), record in data.items() if backend == 'bls'},
        missing={f'{a}/{b}': reason for (a,b),reason in missing.items()}, paired_qualification=paired),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
