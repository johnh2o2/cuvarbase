#!/usr/bin/env python3
"""Make publication-exportable figures solely from archived, validated records."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#3769a0', '#d18528', '#838a91', '#168579', '#80b8a5']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'analysis/benchmark-audit-20260906')
    args = ap.parse_args()
    root = args.root
    out = root/'figures'
    out.mkdir(exist_ok=True)
    validation = json.loads((root/'validation.json').read_text())
    files = {p.stem: json.loads(p.read_text()) for p in (root/'results').glob('*.json')
             if p.stem.startswith(('ls_', 'tls_', 'ensemble_'))}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white',
                         'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    chosen = []

    def eligible(name, kind='ls'):
        return validation[kind].get(name+'.json', {}).get('eligible', False)

    def select(names, kind='ls'):
        candidates = [(name, files[name]) for name in names if name in files and eligible(name, kind)]
        return min(candidates, key=lambda v: v[1]['seconds_per_lc']) if candidates else None

    def values(pair):
        if pair is None:
            return None
        name, d = pair
        n = d['args']['n_lcs']
        samples = np.asarray(d['timing']['times_s']) * 1000 / n
        return np.median(samples), np.min(samples), np.max(samples)

    def save(fig, name):
        for suffix in ['png', 'svg', 'pdf']:
            fig.savefig(out/f'{name}.{suffix}', dpi=190, bbox_inches='tight')
        plt.close(fig)

    configs = ['small', 'tess', 'ztf', 'kepler']
    ticks = ['Small\n1k observations · 5k frequencies',
             'TESS-size\n20k observations · 13.5k frequencies',
             'ZTF-size\n150 observations · 365k frequencies',
             'Kepler-size\n65k observations · 730k frequencies']
    labels = ['CPU competitor: best tested', 'GPU competitor: nifty-ls',
              'cuvarbase PyPI 0.2.5', 'cuvarbase v1.0']

    for precision in ['float64', 'default']:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9.4), sharex=True)
        for ax, n in zip(axes, [1, 32]):
            for x, cfg in enumerate(configs):
                prefix = f'ls_{cfg}_{n}_'
                cpu = select([k for k in files if k.startswith(prefix) and
                              ('nifty_cpu' in k or k.endswith('astropy'))])
                gpu_names = [prefix+'nifty_gpu']
                if precision == 'default':
                    gpu_names.append(prefix+'nifty_gpu_float32')
                gpu = select(gpu_names)
                tail = '_double' if precision == 'float64' else ''
                old = select([prefix+'pypi'+tail])
                new = select([prefix+'v1'+tail])
                for j, pair in enumerate([cpu, gpu, old, new]):
                    pos = x+(j-1.5)*.185
                    val = values(pair)
                    if val is None:
                        ax.text(pos, .82, 'N/A', rotation=90, ha='center', va='bottom',
                                transform=ax.get_xaxis_transform(), color=COLORS[j], fontsize=9)
                        continue
                    med, low, high = val
                    ax.bar(pos, med, .168, color=COLORS[j],
                           label=labels[j] if x==0 else None, zorder=3)
                    ax.errorbar(pos, med, yerr=[[med-low], [high-med]],
                                color='#333333', capsize=2, lw=.8, fmt='none', zorder=4)
                    ax.annotate(f'{med:.2f}', (pos, high), xytext=(0, 5),
                                textcoords='offset points', ha='center', fontsize=8)
                    chosen.append(dict(figure='ls_'+precision, config=cfg, n_lcs=n,
                                       role=labels[j], source=pair[0]+'.json',
                                       median_ms_per_lc=med, min_ms_per_lc=low,
                                       max_ms_per_lc=high,
                                       input_sha256=pair[1]['input_sha256']))
            ax.set_yscale('log')
            ax.set_ylabel('Milliseconds per lightcurve · log scale')
            ax.set_title('One lightcurve · warm API latency' if n==1 else
                         '32 distinct lightcurves · measured batch time / 32', loc='left', pad=12)
            ax.grid(axis='y', which='major', alpha=.18, zorder=0)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom/1.3, top*2.2)
        axes[-1].set_xticks(np.arange(4), ticks)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(.075, .914),
                   ncol=4, frameon=False, fontsize=9)
        fig.suptitle('Lomb–Scargle: single calls and survey batches', x=.065, ha='left',
                     fontsize=19, weight='bold', y=.982)
        subtitle = 'Float64 computation across all four roles.' if precision=='float64' else \
            'cuvarbase default float32; CPU float64; GPU competitor uses the fastest precision passing the sampled checks.'
        fig.text(.065, .934, subtitle, fontsize=10, color='#444444')
        fig.text(.065, .015,
                 'One A40 / Xeon Gold 6342 host; CPU quota 7.65 cores. Identical host inputs and full host output spectra.\n'
                 'Warm medians of 5; whiskers = observed range. CPU threads/workers tuned over 1, 4, 8. '
                 'Synthetic survey-size arrays; data loading and detrending excluded.\n'
                 '“Best tested” is restricted to validated completed candidates; see selected_timings.csv and validation.json. '
                 'N/A means no eligible completed measurement.', fontsize=9, color='#444444')
        fig.subplots_adjust(left=.078, right=.99, bottom=.13, top=.823, hspace=.30)
        save(fig, 'ls_comparison_'+precision)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for ax, precision in zip(axes, ['float64', 'default']):
        for x, shared in enumerate([False, True]):
            prefix = 'ls_shared_tess_' if shared else 'ls_tess_32_'
            cpu = select([k for k in files if k.startswith(prefix) and
                          ('nifty_cpu' in k or k.endswith('astropy'))])
            gpu_names = [prefix+'nifty_gpu']
            if precision == 'default':
                gpu_names.append(prefix+'nifty_gpu_float32')
            tail = '_double' if precision == 'float64' else ''
            pairs = [cpu, select(gpu_names), select([prefix+'pypi'+tail]),
                     select([prefix+('cuvarbase' if shared else 'v1')+tail])]
            for j, pair in enumerate(pairs):
                pos = x+(j-1.5)*.185
                val = values(pair)
                if val is None:
                    continue
                med, low, high = val
                ax.bar(pos, med, .168, color=COLORS[j], label=labels[j] if x==0 else None)
                ax.errorbar(pos, med, yerr=[[med-low], [high-med]], capsize=2,
                            fmt='none', color='#333333', lw=.8)
                ax.annotate(f'{med:.2f}', (pos, high), xytext=(0, 5),
                            textcoords='offset points', ha='center', fontsize=9)
                if shared:
                    chosen.append(dict(figure='ls_shared_'+precision, config='tess', n_lcs=32,
                                       role=labels[j], source=pair[0]+'.json',
                                       median_ms_per_lc=med, min_ms_per_lc=low,
                                       max_ms_per_lc=high, input_sha256=pair[1]['input_sha256']))
        ax.set_xticks([0, 1], ['Distinct observation times', 'Shared observation times'])
        ax.set_yscale('log')
        ax.set_ylabel('Milliseconds per lightcurve · log scale')
        ax.set_title('Float64 throughout' if precision=='float64' else 'cuvarbase default precision', loc='left')
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo/1.3, hi*2.5)
        ax.grid(axis='y', alpha=.18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(.075, .91),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle('LS batch structure changes the comparison', x=.07, ha='left',
                 fontsize=18, weight='bold', y=.982)
    fig.text(.07, .02,
             '32 synthetic TESS-size lightcurves: 20k observations × 13.5k frequencies. A40 / Xeon Gold 6342, CPU quota 7.65 cores.\n'
             'Shared-time inputs permit nifty-ls native batching. Full host outputs; medians of 5, observed ranges shown.\n'
             'Default panel: CPU float64, cuvarbase float32, fastest validated GPU competitor precision. '
             'Inputs are identical across methods within each workload.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.08, right=.99, bottom=.2, top=.76, wspace=.26)
    save(fig, 'ls_shared_times')

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1))
    names = ['CPU TLS\nbest tested', 'GPU GTLS\n0.5.1',
             'PyPI 0.2.5\nTLS unavailable', 'v1.0 TLS\nwide window',
             'v1.0 TLS\ndefault window']
    for ax, n in zip(axes, [1, 16]):
        prefix = f'tls_27_{n}_'
        pairs = [select([k for k in files if k.startswith(prefix+'cpu')], 'tls'),
                 select([prefix+'gtls_head'], 'tls'),
                 None, select([prefix+'v1_wide'], 'tls'), select([prefix+'v1_default'], 'tls')]
        for x, pair in enumerate(pairs):
            val = values(pair)
            if val is None:
                ax.text(x, .14, 'Not implemented' if x==2 else 'No valid result', rotation=90,
                        transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='#777777')
                continue
            med, low, high = val
            ax.bar(x, med, .65, color=COLORS[x], hatch='//' if x==4 else None, zorder=3)
            ax.errorbar(x, med, yerr=[[med-low], [high-med]], capsize=3,
                        fmt='none', color='#333333', lw=.9, zorder=4)
            rec = validation['tls'][pair[0]+'.json']
            ax.annotate(f'{med:.2f} ms\n{rec["exact_recoveries"]}/{rec["n_injected"]} recovered',
                        (x, high), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
            chosen.append(dict(figure='tls', config='27 days, 1296 observations, 2455 periods',
                               n_lcs=n, role=names[x].replace('\n',' '), source=pair[0]+'.json',
                               median_ms_per_lc=med, min_ms_per_lc=low, max_ms_per_lc=high,
                               input_sha256=pair[1]['input_sha256']))
        ax.set_yscale('log')
        ax.set_ylim(.5, 22000)
        ax.set_xticks(range(5), names, fontsize=9)
        ax.set_ylabel('Milliseconds per lightcurve · log scale')
        ax.set_title('Single lightcurve' if n==1 else '16-lightcurve workload', loc='left', fontsize=13)
        ax.grid(axis='y', which='major', alpha=.18, zorder=0)
    fig.suptitle('TLS timing with period-recovery checks', x=.065, ha='left',
                 fontsize=19, weight='bold', y=.982)
    fig.text(.065, .915, '27-day synthetic lightcurves · 30-minute exposures · one shared grid of 2,455 periods', fontsize=11)
    fig.text(.065, .02,
             'A40 / Xeon Gold 6342; medians of 3, ranges shown. Full API search, host to host. '
             'GTLS processes stars sequentially; CPU threads/workers tuned over 1, 4, 8.\n'
             'Wide-window bounds and nominal epoch density are comparable, but templates, discrete grids and refinement differ.\n'
             'Default-window v1.0 searches a narrower family. Recovery uses a 0.2% period tolerance; '
             'the timing injections alone do not establish completeness at fixed FPR.', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.08, right=.99, bottom=.2, top=.79, wspace=.23)
    save(fig, 'tls_comparison')

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for suffix, label, color, marker in [
            ('cpu', 'CPU TLS · 4 threads', COLORS[0], 's'),
            ('gtls_pypi', 'GTLS PyPI 0.4.4', '#b09868', '^'),
            ('gtls_head', 'GTLS upstream 0.5.1', COLORS[1], 'o'),
            ('v1_wide', 'v1.0 TLS · wide window', COLORS[3], 'o')]:
        xs, meds, lows, highs = [], [], [], []
        for baseline in [27, 200, 1500]:
            pair = select([f'tls_{baseline}_1_{suffix}'], 'tls')
            val = values(pair)
            if val is None:
                continue
            med, low, high = np.asarray(val)/1000
            xs.append(baseline)
            meds.append(med)
            lows.append(med-low)
            highs.append(high-med)
        ax.errorbar(xs, meds, yerr=[lows, highs], marker=marker, color=color,
                    label=label, capsize=3, lw=1.5)
    ax.set(xscale='log', yscale='log', xlabel='Lightcurve baseline (days) · log scale',
           ylabel='Warm seconds per single lightcurve · log scale')
    ax.set_xticks([27, 200, 1500], ['27', '200', '1,500'])
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc='upper left')
    fig.suptitle('Fresh TLS scaling measurements', x=.08, ha='left',
                 fontsize=18, weight='bold', y=1.02)
    fig.text(.08, -.05,
             'A40 / Xeon Gold 6342. Identical input arrays and period grids; differing templates and effective duration/epoch grids.\n'
             'Median of 3 with observed range. Unmeasured/invalid results omitted: CPU and PyPI GTLS at 1,500 d; invalid PyPI GTLS at 27 d.\n'
             'Every displayed run recovered this injection within 0.2% in period. One injection per baseline does not establish sensitivity parity.',
             fontsize=9, color='#444444')
    fig.tight_layout()
    save(fig, 'tls_baseline_scaling')

    # A separate historical figure, never combined into the fresh ratios.
    hist = list(csv.DictReader((root/'historical_tls.csv').open()))
    x = [float(r['baseline_days']) for r in hist]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    axes[0].plot(x, [float(r['gtls_seconds']) for r in hist], 'o-', color=COLORS[1], label='GTLS: 1 measured repeat')
    axes[0].plot(x, [float(r['cuvarbase_seconds']) for r in hist], 'o-', color=COLORS[3], label='cuvarbase: median of 3')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Warm seconds per single lightcurve · log scale')
    axes[0].legend(frameon=False)
    axes[1].plot(x, [100*float(r['fraction_periods_narrow_edge_underresolved']) for r in hist], 'o-',
                 color='#b95b4b', label='Narrow edge under-resolved by phase bins')
    axes[1].plot(x, [100*float(r['fraction_periods_narrow_edge_epoch_capped']) for r in hist], 's--',
                 color='#714f91', label='Narrow edge reaches epoch-count cap')
    axes[1].set_ylabel('Fraction of trial periods affected (%)')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel('Lightcurve baseline (days)')
        ax.grid(alpha=.2)
    fig.suptitle('Historical TLS claim: timing evidence and an effective-grid mismatch', x=.06,
                 ha='left', fontsize=16, weight='bold', y=1.03)
    fig.text(.06, -.055,
             'July 2026 archive, reported A5000. Ratios: 30–171×. Exact implementation SHAs are missing.\n'
             '“Affected” refers to the narrow-duration edge, not every duration or an observed missed-transit rate. '
             'No extrapolated timings are plotted.', fontsize=9)
    fig.tight_layout()
    save(fig, 'historical_tls_audit')

    if chosen:
        with (root/'selected_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
            writer.writeheader()
            writer.writerows(chosen)
    print('Wrote', out)


if __name__ == '__main__':
    main()
