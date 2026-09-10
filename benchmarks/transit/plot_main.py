#!/usr/bin/env python3
"""Render the public timing figure from verified benchmark analysis records."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

PROFILES = ['tess_200s', 'tess_gap', 'ztf']
TITLES = {
    'tess_200s': 'TESS: one dense sector',
    'tess_gap': 'TESS: two separated sectors',
    'ztf': 'ZTF: sparse g/r',
}
SUBTITLES = {
    'tess_200s': '200 s cadence · up to 9,736 samples · 26 days',
    'tess_gap': '30 / 10 min cadence · up to 4,295 samples · 735 days',
    'ztf': 'Up to 1,317 samples · 2,744 days',
}
COLORS = {
    'bls_v1': '#008566', 'tls_v1': '#008566', 'bls_pypi': '#2466aa',
    'bls_cpu': '#b55b12', 'bls_gpu': '#8957a5', 'gtls': '#8957a5',
}


def time_label(seconds):
    if seconds < .1:
        return f'{float(f"{seconds * 1000:.2g}"):g} ms'
    return f'{float(f"{seconds:.2g}"):g} s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to the benchmark result directory.')
    parser.add_argument('--tls-study', type=Path,
                        help='Use the independent follow-up TLS timing_analysis.json and supported settings.')
    parser.add_argument('--tls-reference', type=Path,
                        help='Use validated observation-level TLS timing_analysis.json.')
    args = parser.parse_args()
    if args.tls_study and args.tls_reference:
        parser.error('Choose one TLS measurement campaign.')
    recovery = json.loads((args.root / 'recovery_analysis.json').read_text())
    timing = json.loads((args.root / 'timing_analysis.json').read_text())
    for record in (recovery, timing):
        assert record['verification']['complete']
        assert record['verification']['arrays_verified']
    methods = {(r['profile'], r['method']): r for r in recovery['methods']}
    times = {(r['profile'], r['method'], r['mode']): r for r in timing['timings']}
    tls_selection = {}
    reference = None
    reference_profiles = {}
    partial_campaign = False
    tls_modes = ('single', 'batch16')
    if args.tls_reference:
        timing_path = args.tls_reference / 'timing_analysis.json'
        reference = json.loads(timing_path.read_text())
        partial_campaign = reference.get('campaign_pass') is False
        if partial_campaign:
            assessment = json.loads(
                (args.tls_reference / 'reporting_acceptance.json').read_text())
            scope = 'post_hoc_complete_configurations_after_optional_native_warmup_oom'
            original_path = args.tls_reference / 'timing' / 'acceptance.json'
            original = json.loads(original_path.read_text())
            if (assessment.get('reporting_gate', {}).get('pass') is not True or
                    assessment.get('campaign_pass') is not False or
                    reference.get('reporting_scope') != scope or
                    assessment.get('timing_analysis_sha256') !=
                    hashlib.sha256(timing_path.read_bytes()).hexdigest() or
                    assessment.get('original_campaign_acceptance', {}).get('passed') is not False or
                    assessment.get('original_campaign_acceptance', {}).get('sha256') !=
                    hashlib.sha256(original_path.read_bytes()).hexdigest() or
                    original.get('publication_gate', {}).get('pass') is not False):
                raise ValueError('Partial campaign requires its separate hash-bound reporting assessment.')
        for gate in ('complete', 'numerical_validation_complete', 'exclusive_processes'):
            if reference['verification'].get(gate) is not True:
                raise ValueError(f'TLS measurement gate did not pass: {gate}')
        reference_profiles = {r['profile']: r for r in reference['profiles']}
        scope = reference.get('measurement_scope', 'single_and_batch')
        if scope not in ('single', 'single_and_batch'):
            raise ValueError('Unknown TLS timing scope.')
        if scope == 'single':
            tls_modes = ('single',)
        new_times = {(r['profile'], r['method'], r['mode']): r
                     for r in reference['timings']}
        for profile in PROFILES:
            for method in ('tls_v1', 'gtls'):
                for mode in tls_modes:
                    record = new_times[profile, method, 'single' if mode == 'single' else 'batch']
                    if record['boundary'] != 'warm_public_api':
                        raise ValueError('The topline figure requires public-call timings.')
                    expected_n = 1 if mode == 'single' else reference_profiles[profile]['batch_size']
                    if record['n'] != expected_n:
                        raise ValueError('The timing count differs from the displayed workload.')
                    times[profile, method, mode] = dict(
                        seconds_per_source=record['seconds_per_source'], n=1,
                        min_total_s=record['min_seconds_per_source'],
                        max_total_s=record['max_seconds_per_source'], workers=record['workers'])
    if args.tls_study:
        followup = json.loads((args.tls_study / 'timing_analysis.json').read_text())
        assert followup['verification']['complete']
        assert followup['verification']['exclusive_processes']
        tls_selection = {r['profile']: r for r in followup['selected']}
        followup_times = {(r['profile'], r['method'], r['mode']): r for r in followup['timings']}
        for profile in PROFILES:
            for displayed, measured in [('tls_v1', tls_selection[profile]['method']), ('gtls', f'gtls_{profile}')]:
                for mode in ('single', 'batch16'):
                    row = followup_times[profile, measured, mode]
                    # The plot consumes per-source values; n=1 keeps its range
                    # conversion consistent without pretending these are raw calls.
                    times[profile, displayed, mode] = dict(
                        seconds_per_source=row['seconds_per_source'], n=1,
                        min_total_s=row['min_seconds_per_source'],
                        max_total_s=row['max_seconds_per_source'])
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.edgecolor': '#c7cfd5',
        'xtick.color': '#526270', 'ytick.color': '#263c4c',
    })
    fig = plt.figure(figsize=(15, 10.5), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.155, right=.925, top=.815, bottom=.13,
                           hspace=.88, wspace=.79)
    fig.text(.035, .96, 'Faster transit searches across TESS and ZTF cadences',
             fontsize=23, weight='bold', color='#172a3a')
    fig.text(.035, .925, 'Search time per lightcurve · lower is faster',
             fontsize=15, color='#526270')
    batch_sizes = {p.get('batch_size') for p in reference_profiles.values()}
    if tls_modes == ('single',):
        batch_label = 'BLS batch of 16: time per lightcurve'
    else:
        batch_label = ('Batch: time per lightcurve' if reference and batch_sizes != {16}
                       else 'Batch of 16: time per lightcurve')
    fig.legend(handles=[
        Line2D([], [], marker='o', color='#334a5e', markerfacecolor='white',
               linestyle='none', markersize=8, label='One lightcurve'),
        Line2D([], [], marker='o', color='#334a5e', linestyle='none',
               markersize=8, label=batch_label),
    ], loc='upper left', bbox_to_anchor=(.028, .904), ncol=2,
        frameon=False, fontsize=12)

    for row, profile in enumerate(PROFILES):
        for col, (family, entries, v1) in enumerate([
            ('BLS', ['bls_v1', 'bls_pypi', 'bls_cpu', 'bls_gpu'], 'bls_v1'),
            ('TLS', ['tls_v1', 'gtls'], 'tls_v1'),
        ]):
            ax = fig.add_subplot(grid[row, col])
            values_on_axis, labels = [], []
            modes = tls_modes if family == 'TLS' else ('single', 'batch16')
            baseline = times[profile, v1, modes[-1]]['seconds_per_source']
            for index, method in enumerate(entries):
                values = [times[profile, method, mode]['seconds_per_source']
                          for mode in modes]
                values_on_axis.extend(values)
                color = COLORS[method]
                if len(values) > 1:
                    ax.plot(values, [index, index], color=color, lw=2, alpha=.6)
                upper = []
                for mode, value in zip(modes, values):
                    record = times[profile, method, mode]
                    low = record['min_total_s'] / record['n']
                    high = record['max_total_s'] / record['n']
                    upper.append(high)
                    values_on_axis.extend((low, high))
                    ax.errorbar(value, index,
                                xerr=[[max(0, value-low)], [max(0, high-value)]],
                                fmt='none', ecolor=color, capsize=2, alpha=.6)
                ax.scatter(values[0], index, s=65, edgecolors=color,
                           facecolors='white', linewidths=1.8, zorder=4)
                if len(values) > 1:
                    ax.scatter(values[1], index, s=52, color=color, zorder=5)
                if method == v1:
                    label = 'cuvarbase v1'
                    if family == 'TLS' and reference:
                        label += '\n1 worker'
                    if family == 'TLS' and tls_selection:
                        choice = tls_selection[profile]
                        if choice['method'] == 'v1_fine':
                            label += '\nfine grid'
                        elif choice['method'] == 'v1_resolved':
                            label += '\nintermediate grid'
                        if not choice['recovery_supported']:
                            label += ' *'
                elif method == 'bls_pypi':
                    label = 'cuvarbase 0.2.5'
                elif method == 'bls_cpu':
                    backend = methods[profile, method]['config']['backend']
                    label = 'CPU: ' + ('Astropy' if backend == 'astropy' else 'periodfind')
                elif method == 'bls_gpu':
                    label = 'GPU: periodfind'
                else:
                    label = 'GPU: GTLS'
                    if reference and len(modes) > 1:
                        workers = times[profile, method, modes[-1]]['workers']
                        label += f'\nbatch: {workers} worker' + ('s' if workers != 1 else '')
                labels.append(label)
                annotation = time_label(values[-1])
                if method != v1:
                    annotation += f'  ·  {values[-1]/baseline:.1f}×'
                ax.annotate(annotation, (max(upper), index), xytext=(8, 0),
                            textcoords='offset points', va='center', fontsize=12,
                            color=color, weight='bold' if method == v1 else 'normal',
                            annotation_clip=False)
            ax.set_xscale('log')
            ax.set_xlim(min(values_on_axis)/1.7, max(values_on_axis)*25)
            ax.set_ylim(len(entries)-.5, -.6)
            ax.set_yticks(range(len(entries)), labels)
            ax.tick_params(axis='y', length=0, pad=10)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: time_label(x)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis='x', alpha=.18)
            ax.set_axisbelow(True)
            family_label = 'TLS (one source)' if family == 'TLS' and len(modes) == 1 else family
            ax.set_title(f'{family_label}  /  {TITLES[profile]}', loc='left',
                         fontsize=14, weight='bold', color='#172a3a', pad=33)
            subtitle = SUBTITLES[profile]
            if family == 'TLS' and reference:
                metadata = reference_profiles[profile]
                subtitle = (f"{metadata['n_samples']:,} samples · "
                            f"{metadata['baseline_days']:,.0f} days · "
                            f"{metadata['n_periods']:,} trial periods")
            ax.text(0, 1.10, subtitle, transform=ax.transAxes,
                    fontsize=10.5, color='#526270')
    repetitions = ('BLS: 5 single / 3 batch repetitions; TLS: 5 per mode.' if tls_selection
                   else 'Medians of 5 single / 3 batch calls;')
    labels_note = 'Labels give batch time and the ratio to v1.'
    if tls_modes == ('single',):
        labels_note = 'Labels: BLS batch time; TLS single-source time. Ratios are relative to v1.'
        repetitions = 'Medians; whiskers span repetitions;'
    fig.text(.035, .090 if partial_campaign else .069,
             (f'{labels_note} {repetitions} logarithmic axes.' if tls_modes == ('single',)
              else f'{labels_note} {repetitions} Whiskers span repetitions; logarithmic axes.'),
             fontsize=11, color='#394d5d')
    environment = 'A40 + 7.65 CPU-equivalent allocation. Warm searches from prepared arrays; grid construction and preprocessing excluded.'
    if reference:
        gpu_models = {line.split(',')[0].strip() for line in
                      reference['environment']['nvidia_smi'].splitlines() if line.strip()}
        if len(gpu_models) != 1:
            raise ValueError('TLS figure requires one recorded GPU model.')
        tls_gpu = gpu_models.pop().removeprefix('NVIDIA ')
        environment = (f'BLS: A40; TLS: {tls_gpu}. Warm APIs; input loading and grid construction excluded. '
                       'CPU allocations: see report.')
    fig.text(.035, .068 if partial_campaign else .047, environment,
             fontsize=11, color='#526270')
    qualification = 'Recovery qualifications are in the benchmark report. Equivalent TLS detection sensitivity is not established.'
    if tls_selection:
        passed = sum(s['recovery_supported'] for s in tls_selection.values())
        qualification = (f'TLS: {passed}/3 cadences meet the recovery / false-positive matching criterion. '
                         + ('* Matching inconclusive. ' if passed < 3 else '')
                         + 'BLS qualifications: see report.')
    if reference:
        qualification = ('GTLS batch: fastest eligible 1/2/4-worker pool. '
                         'Recovery qualifications and search/diagnostic timings: see report.')
        if tls_modes == ('single',):
            qualification = ('TLS: five calls per method on one noise-only curve per cadence; batch throughput unmeasured. '
                             'Numerical/recovery checks: see report.')
    fig.text(.035, .046 if partial_campaign else .025, qualification,
             fontsize=11, color='#394d5d')
    if partial_campaign:
        failed_labels = {'tess_solar': 'dense TESS', 'tess_gap': 'separated TESS',
                         'ztf_solar': 'ZTF'}
        excluded = reference.get('excluded_configurations', [])
        if not excluded or any(not value.endswith('/gtls_graph_4worker') for value in excluded):
            raise ValueError('Figure failure note does not cover these excluded configurations.')
        failures = ', '.join(failed_labels[value.split('/')[0]] for value in excluded)
        fig.text(.035, .024,
                 'Post hoc report of complete configurations: original campaign gate failed '
                 f'after 4-worker GTLS ran out of memory on {failures}.',
                 fontsize=11, color='#394d5d')
    output = args.output_dir or args.root
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'benchmark_story.{extension}', dpi=160,
                    facecolor='white', metadata={'Creator': 'cuvarbase benchmark tools'})
    plt.close(fig)
    print('Wrote benchmark_story.png / .pdf / .svg')


if __name__ == '__main__':
    main()
