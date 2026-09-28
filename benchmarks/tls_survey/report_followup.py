#!/usr/bin/env python3
"""Report verified new-allocation timings without changing original qualifications."""
import argparse
from collections import Counter
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
from benchmarks.tls_survey.plot_throughput import SCOPES, heldout_qualification
from benchmarks.tls_survey.plot_native_bls_comparison import allocation, cohort, execution_rates

BACKENDS = ('baseline', 'candidate', 'gtls', 'bls')
LABELS = dict(baseline='TLS baseline', candidate='TLS experimental', gtls='GTLS', bls='BLS execution')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_comparison(data, expected_allocation, seal_sha):
    thread_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
    for (backend, scope), record in data.items():
        if allocation(record) != tuple(expected_allocation) or record['science_seal_sha256'] != seal_sha:
            raise ValueError('A comparison changes GPU/CPU/RAM allocation or scientific identity')
        threads = record['environment'].get('cpu_math_thread_environment', {})
        if any(threads.get(name) != '1' for name in thread_names):
            raise ValueError('The six numerical thread limits were not preserved')
        expected = {name: 32 for name in SCOPES[:-1]} if scope == 'varied' else {scope: 16}
        if Counter(row['regime'] for row in record['cohort']) != Counter(expected):
            raise ValueError('Timing panel changed its predeclared population')
    for scope in SCOPES:
        identities = [cohort(record) for (backend, panel), record in data.items() if panel == scope]
        if any(value != identities[0] for value in identities[1:]):
            raise ValueError('Competitors used different input bytes: '+scope)


def read_results(work):
    evidence = work/'collected'
    verification = json.loads((work/'collection-verification.json').read_text())
    if verification['status'] != 'archive_and_all_members_verified':
        raise ValueError('Verified collection required')
    inventory = json.loads((evidence/'completion/inventory.json').read_text())

    def checked(relative, expected=None):
        path = evidence/relative
        recorded = inventory[relative]['sha256']
        if sha(path) != recorded or (expected is not None and expected != recorded):
            raise ValueError('Collected receipt identity changed: '+relative)
        return json.loads(path.read_text())

    state = checked('campaign-state.json')
    design = checked('campaign-design.json', state['design_sha256'])
    checked('evidence/science-seal.json')
    seal_sha = sha(evidence/'evidence/science-seal.json')
    exactness = heldout_qualification(dict(science_seal_sha256=seal_sha),
        ROOT/'benchmarks/results/tls_survey_2026-09-10/final-science/exactness-final.json',
        evidence/'evidence/science-seal.json')
    data, missing, paired = {}, {}, {}
    strict_name = 'strict-measure/campaign.json'
    if strict_name in inventory:
        strict = checked(strict_name)
        if strict.get('science_seal_sha256') != seal_sha:
            raise ValueError('Strict timing science seal changed')
        for entry in strict.get('unavailable', []):
            missing[entry['backend'], entry['scope']] = entry['reason']
        for entry in strict.get('configs', []):
            if entry['scope'] not in SCOPES:
                continue
            key = entry['backend'], entry['scope']
            if not entry['result_sha256']:
                missing[key] = entry.get('failure_reason', 'No timing receipt')
                continue
            record = checked('strict-measure/'+entry['result'], entry['result_sha256'])
            if not entry['eligible']:
                missing[key] = entry.get('failure_reason', 'Strict qualification failed')
                continue
            if (record['status'] != 'ok' or record['gpu_ownership']['passed'] is not True or
                    len(record['qualification']) != 2 or
                    not all(q['gate']['passed'] for q in record['qualification']) or
                    len(record['repetitions']) != 3):
                raise ValueError('Strict eligibility contradicts its evidence')
            for row in record['repetitions']:
                if (row['status'] != 'ok' or row['source_count'] < 96 or row['elapsed_seconds'] < 120 or
                        not math.isclose(row['lightcurves_per_second'],
                            row['source_count']/row['elapsed_seconds'], rel_tol=1e-12)):
                    raise ValueError('Incomplete or misreported sustained queue')
            data[key] = record
        for check in strict.get('baseline_candidate_spectra', {}).get('checks', []):
            paired[check['scope']] = paired.get(check['scope'], True) and check['exact']
        for scope, passed in paired.items():
            if not passed:
                data.pop(('candidate', scope), None)
                missing['candidate', scope] = 'Paired baseline/experimental complete-spectrum gate failed'
    bls_name = 'bls-execution/campaign.json'
    if bls_name in inventory:
        bls = checked(bls_name)
        plan = checked('bls-execution/plan.json', bls['plan_sha256'])
        if plan['science_seal_sha256'] != seal_sha or plan['original_numerical_qualification_passed'] is not False:
            raise ValueError('BLS supplement identity or original failure changed')
        for entry in bls.get('configs', []):
            if not entry['label'].startswith('measure-'):
                continue
            scope = entry['label'][len('measure-'):]
            if scope not in SCOPES:
                raise ValueError('Undeclared BLS measurement cohort')
            record = checked('bls-execution/'+entry['label']+'/result.json', entry['result_sha256'])
            if not entry['execution_rates_valid']:
                missing['bls', scope] = record.get('error', record['status'])
                continue
            execution_rates(record)
            data['bls', scope] = record
    expected_allocation = tuple(state['environment'][key] for key in (
        'nvidia_smi', 'cpu_quota_cores', 'host_memory_limit_bytes'))
    validate_comparison(data, expected_allocation, seal_sha)
    rows = []
    for scope in SCOPES:
        for backend in BACKENDS:
            record = data.get((backend, scope))
            row = dict(scope=scope, backend=backend, available=record is not None,
                qualification='execution only; original exact gate failed' if backend == 'bls' else 'strict timing gates',
                workers=None, batch_size=None,
                median_lightcurves_per_second=None, minimum_rate=None, maximum_rate=None,
                attempted=None, successful=None, api_failures=None, selected_discrepancies=None,
                selected_comparisons=None, complete_diagnostic_discrepancies=None,
                cold_preparation_seconds=None, sampled_gpu_peak_bytes=None,
                usd_per_million_successful=None, paired_experimental_speed_ratio=None,
                failure_reason=None if record else missing.get((backend, scope), 'No completed qualifying panel'))
            if record:
                native = backend == 'bls'
                rates = execution_rates(record) if native else [r['lightcurves_per_second'] for r in record['repetitions']]
                attempted = sum(r['attempted_count' if native else 'source_count'] for r in record['repetitions'])
                success = sum(r['successful_count' if native else 'source_count'] for r in record['repetitions'])
                median = statistics.median(rates)
                row.update(workers=record['workers'], batch_size=record['batch_size'],
                    median_lightcurves_per_second=median, minimum_rate=min(rates), maximum_rate=max(rates),
                    attempted=attempted, successful=success, api_failures=attempted-success,
                    selected_discrepancies=record['numerical']['selected_mismatch_count'] if native else 0,
                    selected_comparisons=record['numerical']['comparison_count'] if native else None,
                    complete_diagnostic_discrepancies=record['numerical']['complete_output_mismatch_count'] if native else 0,
                    cold_preparation_seconds=record['summary']['cold_first_cohort_including_startup_seconds'],
                    sampled_gpu_peak_bytes=record['memory']['gpu_used_bytes'],
                    usd_per_million_successful=design['hourly_usd']*1e6/(3600*median) if median else None)
                if backend == 'candidate' and paired.get(scope) and ('baseline', scope) in data:
                    baseline = statistics.median(r['lightcurves_per_second'] for r in data['baseline', scope]['repetitions'])
                    row['paired_experimental_speed_ratio'] = median/baseline
            rows.append(row)
    return dict(rows=rows, original_exactness=exactness, allocation=list(expected_allocation),
                campaign_status=state['status'], design_sha256=state['design_sha256'],
                verified_archive_sha256=verification['archive_sha256'], hourly_usd=design['hourly_usd'],
                reporter_sha256=sha(__file__), validator_sha256={name: sha(Path(__file__).with_name(name))
                    for name in ('plot_throughput.py', 'plot_native_bls_comparison.py')})


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (output/'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), layout='constrained')
    colors = ['#4477aa', '#228833', '#aa3377', '#ccbb44']
    available = [r['median_lightcurves_per_second'] for r in result['rows'] if r['available'] and r['median_lightcurves_per_second']]
    floor = min(available)/3 if available else .01
    ceiling = max(available)*3 if available else 1
    for axis, scope in zip(axes, SCOPES):
        rows = [r for r in result['rows'] if r['scope'] == scope]
        for index, row in enumerate(rows):
            if row['available'] and row['median_lightcurves_per_second']:
                median = row['median_lightcurves_per_second']
                axis.bar(index, median, color=colors[index], hatch='///' if row['backend'] == 'bls' else None,
                         edgecolor='#333333', linewidth=.6)
                axis.errorbar(index, median, yerr=[[median-row['minimum_rate']], [row['maximum_rate']-median]],
                              color='#222222', capsize=3)
            else:
                label = '0 completions' if row['available'] else 'unavailable'
                axis.text(index, floor*1.1, label, rotation=90, ha='center', va='bottom', fontsize=8)
        axis.set(title=scope.replace('_', ' '), yscale='log', ylim=(floor, ceiling), xlim=(-.6, 3.6))
        axis.set_xticks(range(4), [LABELS[b] for b in BACKENDS], rotation=40, ha='right')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Successful lightcurves / second')
    fig.suptitle('Same-allocation sustained throughput: median and observed range\n'
                 'Hatched BLS = execution only; original exact-repeatability qualification failed', fontsize=12)
    exact = result['original_exactness']
    qualification = 'passed' if exact['aggregate_exactness_qualified'] else 'failed'
    fig.supxlabel(f"Experimental study exactness: {exact['exact_cases']:,}/{exact['planned_cases']:,}; "
                  f"aggregate gate {qualification}. Timing results do not requalify sensitivity.", fontsize=9)
    fig.savefig(output/'throughput.png', dpi=180)
    fig.savefig(output/'throughput.svg')
    plt.close(fig)
    lines = ['# September 24 throughput follow-up', '',
        'These are repeated original timing workloads on one new allocation, with unchanged numerical '
        'sources and full grids. Each available panel has three complete queues of at least 96 attempts '
        'and 120 seconds, in whole cohort cycles.', '',
        f"The original experimental exactness outcome remains {exact['exact_cases']:,}/{exact['planned_cases']:,}; "
        f"its {exact['mismatches']} mismatches still fail the aggregate gate. These timing repetitions do not requalify sensitivity.", '',
        '![Sustained throughput](throughput.png)', '',
        '| Workload | Method | Median / second | Observed range | API failures / attempts | Selected discrepancies |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        if row['available']:
            values = (f"{row['median_lightcurves_per_second']:.5g}",
                      f"{row['minimum_rate']:.5g}–{row['maximum_rate']:.5g}",
                      f"{row['api_failures']}/{row['attempted']}", str(row['selected_discrepancies']))
        else:
            values = ('unavailable', '—', '—', '—')
        lines.append('| '+' | '.join([row['scope'], LABELS[row['backend']], *values])+' |')
    lines.extend(['', 'BLS rates count successful native completions and include failed-call elapsed time and '
        'per-attempt journal overhead. BLS selected discrepancies include the pre/post diagnostic comparisons '
        'and measured queues; they introduce no tolerance or numerical passing label. Its original exact '
        'qualification remains failed. TLS/GTLS rates require the unchanged strict timing gates.', '',
        'The CSV retains the selected worker/batch settings, comparison counts, cold preparation, sampled GPU memory, '
        'unavailable reasons and cost projections. BLS batches group serial native calls within a worker. '
        'Projected costs use the median successful rate at the recorded hourly price; they exclude acquisition, '
        'preprocessing and vetting and do not describe an actual million-source run.', '',
        f"Verified evidence archive SHA256: `{result['verified_archive_sha256']}`.",
        f"Frozen follow-up design SHA256: `{result['design_sha256']}`.", ''])
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(read_results(args.work.resolve()), args.output.resolve())


if __name__ == '__main__':
    main()
