#!/usr/bin/env python3
"""Present strict TLS timings beside separately measured native BLS execution.

The original qualified figure is left intact. Native BLS execution measurements
never acquire exact-repeatability qualification in this presentation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.tls_survey.plot_throughput import (
    BACKENDS, SCOPES, heldout_qualification, read_campaign,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def allocation(record):
    environment = record['environment']
    values = tuple(environment.get(key) for key in (
        'nvidia_smi', 'cpu_quota_cores', 'host_memory_limit_bytes'))
    if any(value is None for value in values):
        raise ValueError('Unverified resource allocation')
    return values


def cohort(record):
    rows = record['cohort']
    identities = [tuple(row[key] for key in
        ('name', 'regime', 'nobs', 'nperiods', 'input_sha256')) for row in rows]
    if not identities or len({row[0] for row in identities}) != len(identities):
        raise ValueError('Empty or duplicated timing cohort')
    return sorted(identities)


def execution_rates(record):
    """Validate accounting, without introducing a numerical passing tolerance."""
    if (record.get('status') != 'complete' or record.get('execution_rates_valid') is not True or
            record.get('gpu_ownership', {}).get('passed') is not True):
        raise ValueError('Execution or ownership validity failed')
    if record['numerical'].get('original_qualification_passed') is not False:
        raise ValueError('Native BLS must retain its failed original qualification')
    repetitions = record['repetitions']
    if len(repetitions) != 3:
        raise ValueError('Exactly three planned sustained repetitions required')
    rates = []
    for row in repetitions:
        if row.get('status') != 'completed_queue':
            raise ValueError('Interrupted or instrument-invalid queue')
        counts = [row[key] for key in ('attempted_count', 'successful_count', 'failed_count')]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('Invalid execution counts')
        attempted, successful, failed = counts
        seconds = row['elapsed_seconds']
        if (attempted != successful + failed or attempted < 96 or
                not math.isfinite(seconds) or seconds < 120):
            raise ValueError('Incomplete sustained queue or inconsistent accounting')
        rate = successful / seconds
        reported = row['successful_lightcurves_per_second']
        if not math.isfinite(reported) or not math.isclose(rate, reported, rel_tol=1e-12, abs_tol=0):
            raise ValueError('Reported execution rate omits failed work or elapsed time')
        rates.append(rate)
    return rates


IDENTITY_KEYS = ('science_seal_sha256', 'auxiliary_plan_sha256', 'supplement_seal_sha256',
                 'supplement_binding_sha256', 'primary_tuning_sha256', 'primary_measurement_sha256')


def read_native(path, primary_path, primary_campaign, expected_allocation, supplement_seal, supplement_binding,
                native_tuning):
    path, primary_path, supplement_seal, supplement_binding, native_tuning = map(
        Path, (path, primary_path, supplement_seal, supplement_binding, native_tuning))
    campaign = json.loads(path.read_text())
    seal = json.loads(supplement_seal.read_text())
    binding = json.loads(supplement_binding.read_text())
    if (seal.get('schema') != 1 or seal.get('kind') != 'native_bls_execution_supplement' or
            binding.get('schema') != 1):
        raise ValueError('Unexpected supplementary design or binding schema')
    if campaign.get('stage') != 'measure' or campaign.get('status') != 'complete':
        raise ValueError('Native comparison requires completed planned measurements')
    if (campaign.get('original_numerical_qualification_passed') is not False or
            campaign.get('science_seal_sha256') != primary_campaign['science_seal_sha256'] or
            campaign.get('primary_measurement_sha256') != sha(primary_path) or
            campaign.get('supplement_seal_sha256') != sha(supplement_seal) or
            campaign.get('supplement_binding_sha256') != sha(supplement_binding) or
            binding.get('supplement_seal_sha256') != sha(supplement_seal) or
            seal.get('science_seal_sha256') != primary_campaign['science_seal_sha256']):
        raise ValueError('Native execution study has changed identity or qualification')
    for key in ('science_seal_sha256', 'auxiliary_plan_sha256'):
        if campaign[key] != seal[key] or binding[key] != seal[key]:
            raise ValueError('Supplement changes the reviewed scientific or auxiliary identity')
    for key in ('primary_tuning_sha256', 'primary_measurement_sha256'):
        if campaign[key] != binding[key]:
            raise ValueError('Supplement changes the mechanically bound primary artifacts')
    tuning_path = seal['binding_rule']['primary_tuning_path']
    if seal['remote_files'].get(tuning_path) != binding['primary_tuning_sha256']:
        raise ValueError('Primary tuning was not frozen in the prospective supplement seal')
    for key in ('manifest_sha256', 'varied_manifest_sha256'):
        if primary_campaign[key] != binding['primary_measurement_'+key]:
            raise ValueError('Supplement changes the mechanically bound timing manifests')
    originals = {}
    expected_configs = []
    for row in primary_campaign['configs']:
        source = primary_path.parent / row['result']
        if sha(source) != row['result_sha256']:
            raise ValueError('Primary timing receipt changed')
        record = json.loads(source.read_text())
        expected_configs.append(dict(scope=row['scope'], result=row['result'], result_sha256=row['result_sha256'],
            cohort_sha256=canonical_sha(record['cohort']), environment_sha256=canonical_sha(record['environment'])))
        if row['scope'] not in SCOPES:
            continue
        if record.get('cohort'):
            value = cohort(record)
            if row['scope'] in originals and originals[row['scope']] != value:
                raise ValueError('Primary competitors used different timing cohorts')
            originals[row['scope']] = value
    if binding['primary_configs'] != expected_configs:
        raise ValueError('Supplement changes the mechanically bound primary cohorts or resources')
    if sha(native_tuning) != campaign['tuning_seal_sha256']:
        raise ValueError('Separate development tuning seal changed')
    tuning_seal = json.loads(native_tuning.read_text())
    if Path(tuning_seal['campaign_path']).name != 'campaign.json':
        raise ValueError('Unexpected development tuning campaign name')
    tuning_path = native_tuning.parent/'campaign.json'
    if sha(tuning_path) != tuning_seal['campaign_sha256']:
        raise ValueError('Separate development tuning campaign changed')
    tuning = json.loads(tuning_path.read_text())
    if (tuning.get('status') != 'complete' or tuning.get('stage') != 'tune' or
            tuning_seal.get('original_qualification_passed') is not False):
        raise ValueError('Separate development tuning is incomplete or requalified')
    for key in (*IDENTITY_KEYS, 'source_identity', 'allocation', 'deadline_epoch', 'selected',
                'original_bls_exclusion_sha256', 'hourly_usd'):
        if tuning_seal[key] != tuning[key] or tuning_seal[key] != campaign[key]:
            raise ValueError('Native measurement differs from its development selection: '+key)
    if tuning_seal['manifest_sha256'] != tuning['manifest_sha256']:
        raise ValueError('Separate development tuning manifest changed')
    for name, digest in tuning['artifact_sha256'].items():
        if sha(tuning_path.parent/name) != digest:
            raise ValueError('Separate development tuning artifact changed')
    records, missing, seen = {}, {}, set()
    for row in campaign.get('unavailable', []):
        missing[row['scope']] = row['reason']
    for row in campaign['configs']:
        scope = row['scope']
        if scope not in SCOPES:
            raise ValueError('Unplanned native timing scope')
        source = path.parent / row['result']
        if sha(source) != row['result_sha256']:
            raise ValueError('Native timing receipt changed')
        record = json.loads(source.read_text())
        if record.get('scope') != scope or record.get('backend') != 'native_bls_execution':
            raise ValueError('Native result scope or backend changed')
        for key in (*IDENTITY_KEYS, 'source_identity'):
            if record[key] != campaign[key]:
                raise ValueError('Native result belongs to another sealed execution: '+key)
        if row.get('reference_only'):
            if row['workers'] != 1 or row['batch_size'] != 1:
                raise ValueError('Invalid native reference diagnostic setting')
            continue
        if scope in seen:
            raise ValueError('Repeated native timing scope')
        seen.add(scope)
        if not row['execution_rates_valid']:
            missing[scope] = row.get('failure_reason') or 'Execution accounting or resource check failed'
            continue
        if scope in missing:
            raise ValueError('Contradictory native availability')
        for key in ('workers', 'batch_size'):
            if (row[key] != campaign['selected'][key] or
                    record[key] != campaign['selected'][key]):
                raise ValueError('Native setting changed after development selection')
        execution_rates(record)
        if allocation(record) != expected_allocation:
            raise ValueError('Native BLS did not use the same GPU/CPU/memory allocation')
        if scope not in originals or cohort(record) != originals[scope]:
            raise ValueError('Native BLS did not use the same timing light curves and grids')
        records[scope] = record
    for scope in SCOPES:
        if scope not in records:
            missing.setdefault(scope, 'No valid native execution measurement')
    return campaign, records, missing


def table_rows(primary, native, missing, native_missing, paired, heldout):
    rows = []
    for scope in SCOPES:
        for backend in BACKENDS:
            is_native = backend == 'bls'
            record = native.get(scope) if is_native else primary.get((backend, scope))
            rates = (execution_rates(record) if is_native else
                     [r['lightcurves_per_second'] for r in record['repetitions']]) if record else []
            repetitions = record['repetitions'] if record else []
            numerical = record.get('numerical', {}) if record else {}
            attempts = sum(r['attempted_count'] for r in repetitions) if is_native and record else None
            successes = sum(r['successful_count'] for r in repetitions) if is_native and record else None
            summary = record['summary'] if record else {}
            rows.append(dict(scope=scope, backend=backend,
                workers=record.get('workers') if record else None,
                batch_size=record.get('batch_size') if record else None,
                rate_contract='native_execution_only' if is_native else 'original_qualified_timing',
                original_numerical_qualification_passed=False if is_native else bool(record),
                rate_available=bool(record),
                missing_reason=(native_missing.get(scope) if is_native else missing.get((backend, scope)))
                    if not record else None,
                median_lightcurves_per_second=statistics.median(rates) if rates else None,
                minimum_lightcurves_per_second=min(rates) if rates else None,
                maximum_lightcurves_per_second=max(rates) if rates else None,
                attempted_count=attempts, successful_count=successes,
                failed_count=sum(r['failed_count'] for r in repetitions) if is_native and record else None,
                completion_fraction=successes/attempts if attempts else None,
                selected_mismatch_count=numerical.get('selected_mismatch_count'),
                complete_output_mismatch_count=numerical.get('complete_output_mismatch_count'),
                cold_first_cohort_including_startup_seconds=record['summary'].get(
                    'cold_first_cohort_including_startup_seconds') if record else None,
                sampled_gpu_peak_bytes=record.get('memory', {}).get('gpu_used_bytes') if record else None,
                sampled_worker_rss_peak_bytes=record.get('memory', {}).get('host_pool_rss_bytes') if record else None,
                total_measured_compute_usd=record['summary'].get('total_measured_compute_usd') if record else None,
                usd_per_million_successful_steady=summary.get('usd_per_million_successful' if is_native else
                                                            'usd_per_million_steady'),
                usd_per_million_successful_cold_amortized=(
                    (summary['total_measured_compute_usd']+summary['cold_preparation_compute_usd'])*1e6/successes
                    if is_native and successes and 'cold_preparation_compute_usd' in summary else
                    summary.get('usd_per_million_cold_amortized') if not is_native else None),
                cold_amortized_successful_lightcurves_per_second=summary.get(
                    'cold_amortized_successful_lightcurves_per_second' if is_native else
                    'cold_amortized_lightcurves_per_second'),
                heldout_exact_cases=heldout['exact_cases'], heldout_planned_cases=heldout['planned_cases'],
                heldout_aggregate_exactness_qualified=heldout['aggregate_exactness_qualified'],
                timing_cohort_paired_tls_qualification=bool(paired.get(scope)),
                science_seal_sha256=heldout['science_seal_sha256']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--native-bls', type=Path, required=True)
    parser.add_argument('--supplement-seal', type=Path, required=True)
    parser.add_argument('--supplement-binding', type=Path, required=True)
    parser.add_argument('--native-tuning', type=Path, required=True, help='Separate development tuning-seal.json')
    parser.add_argument('--science-seal', type=Path, required=True)
    parser.add_argument('--exactness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary_campaign, data, missing, paired, resources = read_campaign(args.primary)
    heldout = heldout_qualification(primary_campaign, args.exactness, args.science_seal)
    native_campaign, native, native_missing = read_native(
        args.native_bls, args.primary, primary_campaign, resources, args.supplement_seal, args.supplement_binding,
        args.native_tuning)
    if native_campaign['auxiliary_plan_sha256'] != heldout['auxiliary_plan_sha256']:
        raise ValueError('Native supplement and held-out exactness use different auxiliary plans')
    rows = table_rows(data, native, missing, native_missing, paired, heldout)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline': '#8898a6', 'candidate': '#087d92', 'gtls': '#cb7950', 'bls': '#7759a0'}
    titles = ('TESS dense sector', 'TESS separated sectors', 'ZTF g/r', 'Varied sizes: 96 sources')
    fig, axes = plt.subplots(1, 4, figsize=(13, 6.2), layout='constrained')
    for ax, scope, title in zip(axes, SCOPES, titles):
        positive = []
        for index, backend in enumerate(BACKENDS):
            row = next(r for r in rows if r['scope'] == scope and r['backend'] == backend)
            value = row['median_lightcurves_per_second']
            if value is None or value == 0:
                label = 'No valid\nmeasurement' if value is None else 'Zero successful\ncompletions'
                ax.text(index, .03, label, transform=ax.get_xaxis_transform(), rotation=90,
                        ha='center', va='bottom', fontsize=7, color='#884343')
                continue
            low, high = row['minimum_lightcurves_per_second'], row['maximum_lightcurves_per_second']
            positive.extend(v for v in (low, value, high) if v > 0)
            native_bar = backend == 'bls'
            ax.bar(index, value, .7, facecolor='white' if native_bar else colors[backend],
                   edgecolor=colors[backend], hatch='///' if native_bar else None,
                   yerr=[[value-low], [high-value]], capsize=3,
                   error_kw=dict(elinewidth=1, ecolor='#303c46'))
            label = f'{value:.2f}'
            if native_bar:
                fraction = row['completion_fraction']
                percent = f'{fraction:.3%}'
                if fraction < 1 and percent == '100.000%':
                    percent = '<100%'
                label += f"\n{percent} completed\n{row['failed_count']} API errors"
            ax.text(index, high*1.13, label, ha='center', fontsize=7 if native_bar else 8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(4), ['Baseline', 'Optimized', 'GTLS', 'Native BLS'], rotation=35)
        ax.set_yscale('log')
        ax.set_ylim(min(positive)/2 if positive else .01, max(positive)*4 if positive else 1)
        ax.set_xlim(-.6, 3.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', which='major', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Successful light curves per second · log scale')
    status = 'finite tested scope passed' if heldout['aggregate_exactness_qualified'] else 'AGGREGATE EXACTNESS WITHHELD'
    fixture = 'SYNTHETIC FIXTURE · ' if primary_campaign.get('synthetic_fixture') else ''
    fig.suptitle(fixture+'Full transit-search throughput · '+resources[0].split(',')[0]+'\n'
        f"Held-out TLS exactness: {heldout['exact_cases']}/{heldout['planned_cases']} · {status}",
        fontsize=13, weight='bold', color='#253b46' if heldout['aggregate_exactness_qualified'] else '#9b2424')
    fig.supxlabel('Hatched BLS bars: native execution only; original exact-repeatability qualification failed.\n'
        'Median of 3 queues; whiskers: observed repeat range. Each queue: ≥96 attempts AND ≥120 s.\n'
        'Same light curves, grids and GPU/CPU/memory allocation; separate development tuning. Full search and transfers included.\n'
        'Native BLS API failures consume elapsed time and reduce successful throughput; score discrepancies remain recorded.\n'
        'Native BLS also includes per-attempt journaling and comparison overhead; that extra cost is retained.\n'
        'Separate logarithmic y scales. Throughput does not establish equal detection sensitivity or global TLS equivalence.', fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output.with_suffix('.'+extension), dpi=180)
    plt.close(fig)
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = dict(primary_campaign_sha256=sha(args.primary), native_campaign_sha256=sha(args.native_bls),
        science_seal_sha256=sha(args.science_seal), supplement_seal_sha256=sha(args.supplement_seal),
        supplement_binding_sha256=sha(args.supplement_binding),
        native_tuning_seal_sha256=sha(args.native_tuning),
        renderer_sha256=sha(__file__), original_renderer_sha256=sha(Path(__file__).with_name('plot_throughput.py')),
        heldout_exactness=heldout, allocation=resources, original_bls_numerical_qualification_passed=False,
        native_instrumentation_note='Per-attempt journaling and exact comparison overhead is included beyond primary instrumentation; no subtraction.',
        primary_missing=primary_campaign.get('unavailable'), native_missing=native_missing,
        native_selected=native_campaign.get('selected'), outputs={
            args.output.with_suffix('.'+extension).name: sha(args.output.with_suffix('.'+extension))
            for extension in ('png', 'pdf', 'svg', 'csv')})
    args.output.with_suffix('.data.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
