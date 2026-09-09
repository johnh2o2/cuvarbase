#!/usr/bin/env python3
"""Summarize new measurements and the small sensitivity diagnostic."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'analysis/benchmark-audit-20260906')
    args = ap.parse_args()
    root = args.root
    validation = json.loads((root/'validation.json').read_text())
    selected = list(csv.DictReader((root/'selected_timings.csv').open()))
    text = ['# Fresh measurement results', '',
            'All times below are measured warm API wall times on the same A40 host. '
            'The figure groups and source filenames are recorded in `selected_timings.csv`.', '']
    for kind, title in [('ls_float64', 'Lomb–Scargle, float64 throughout'),
                         ('ls_default', 'Lomb–Scargle, cuvarbase default precision'),
                         ('ls_shared_float64', 'Shared-time LS batch, float64 throughout'),
                         ('ls_shared_default', 'Shared-time LS batch, cuvarbase default precision')]:
        text += ['## '+title, '',
                 '| Workload | LCs | CPU ms/LC | GPU competitor ms/LC | PyPI 0.2.5 ms/LC | v1.0 ms/LC | CPU/v1.0 | PyPI/v1.0 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for cfg in (['tess'] if 'shared' in kind else ['small', 'tess', 'ztf', 'kepler']):
            for n in ([32] if 'shared' in kind else [1, 32]):
                rows = {r['role']: float(r['median_ms_per_lc']) for r in selected
                        if r['figure']==kind and r['config']==cfg and int(r['n_lcs'])==n}
                v = [rows.get(k) for k in ['CPU competitor: best tested',
                     'GPU competitor: nifty-ls', 'cuvarbase PyPI 0.2.5', 'cuvarbase v1.0']]
                numbers = [f'{x:.3f}' if x is not None else 'N/A' for x in v]
                ratios = [f'{v[i]/v[3]:.2f}×' if v[i] is not None and v[3] is not None else 'N/A'
                          for i in [0, 2]]
                text += ['| '+ ' | '.join([cfg, str(n), *numbers, *ratios])+' |']
        text += ['', 'Ratios below one mean v1.0 is slower. “Best tested” selects only '
                 'completed candidates passing the sampled accuracy screen. See `validation.json` '
                 'for precision/error measurements and excluded results.', '']

    text += ['## TLS timing and recovery', '',
             '| Record | ms/LC | Exact period recovery | Native output finite |',
             '|---|---:|---:|---|']
    tls_rows = []
    for p in sorted((root/'results').glob('tls_*.json')):
        d = json.loads(p.read_text())
        if d.get('status') != 'ok':
            text.append(f'| {p.name} | incomplete/error | — | — |')
            continue
        rec = d['recovery']
        exact = sum(r['exact_recovery'] is True for r in rec)
        inj = sum(r['injected'] for r in rec)
        finite = all(r['period'] is not None and r['native_sde'] is not None for r in rec)
        text.append(f'| {p.name} | {1000*d["seconds_per_lc"]:.3f} | {exact}/{inj} | {finite} |')
        tls_rows.append(dict(source=p.name, seconds_per_lc=d['seconds_per_lc'],
                             exact_recovery=exact, n_injected=inj, native_finite=finite))
    text += ['', 'PyPI cuvarbase 0.2.5 has no TLS. The default v1.0 window is narrower '
             'than the wide-window comparison. Neither timing configuration is certified to '
             'have equivalent completeness/FPR to GTLS or CPU TLS.', '']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    summary = {}
    methods = [('cpu', 'CPU TLS', '#3769a0'), ('gtls', 'GTLS 0.5.1', '#d18528'),
               ('cuvarbase', 'v1.0 wide', '#168579'),
               ('cuvarbase_default', 'v1.0 default', '#80b8a5')]
    for index, (key, label, color) in enumerate(methods):
        p = root/'results'/f'ensemble_{key}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if d.get('status') != 'ok':
            summary[key] = dict(status=d.get('status'), error=d.get('error'))
            continue
        if not validation['tls'].get(p.name, {}).get('eligible'):
            summary[key] = dict(status='failed_validation',
                                validation=validation['tls'].get(p.name))
            continue
        rows = d['recovery']
        injected = np.asarray([r['injected'] for r in rows], bool)
        correct = np.asarray([r['exact_recovery'] is True for r in rows], bool)
        score = np.asarray([r['identical_sde']['current'] if r['identical_sde']['current'] is not None
                            else -np.inf for r in rows], float)
        thresholds = np.r_[np.inf, np.sort(np.unique(score[np.isfinite(score)]))[::-1], -np.inf]
        fpr = [np.mean(score[~injected]>=threshold) for threshold in thresholds]
        tpr = [np.mean((score[injected]>=threshold) & correct[injected]) for threshold in thresholds]
        axes[0].step(fpr, tpr, where='post', color=color, label=label)
        count = int(correct[injected].sum())
        axes[1].bar(index, count, color=color, width=.65)
        axes[1].text(index, count+.6, f'{count}/{int(injected.sum())}', ha='center', fontsize=11)
        summary[key] = dict(status='ok', n_injected=int(injected.sum()),
                            n_null=int((~injected).sum()), exact_recoveries=count,
                            alias_inclusive_recoveries=sum(r['alias_recovery'] is True for r in rows),
                            finite_common_scores=int(np.isfinite(score).sum()),
                            min_valid_periods=min(r['periods_finite'] for r in rows),
                            fpr=fpr, correct_recovery_tpr=tpr,
                            native_null_above_7=sum(r['native_sde'] is not None and r['native_sde']>7
                                                    for r in rows if not r['injected']))
    axes[0].set(xlabel='Empirical false-positive fraction on 16 nulls',
                ylabel='Correct-period recovery fraction on 48 injections', xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    axes[1].set_xticks(range(4), [m[1] for m in methods])
    axes[1].set(ylabel='Correct periods before any significance threshold', ylim=(0, 52))
    axes[1].grid(axis='y', alpha=.2)
    fig.suptitle('Sensitivity diagnostic · too small to certify equivalence', x=.07, ha='left',
                 fontsize=17, weight='bold', y=1.03)
    fig.text(.07, -.05,
             '64 synthetic 27-day lightcurves: gaps, finite exposures, varied signal strength/geometry, white and correlated noise.\n'
             'Same current-definition re-scoring; exact period tolerance 0.2%. Null resolution is 1/16 = 6.25 percentage points.\n'
             'This experiment cannot establish a 1% FPR or percent-level completeness parity. '
             'It is not a measured survey population.', fontsize=9)
    fig.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(root/'figures'/f'tls_sensitivity_diagnostic.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)
    write_json(root/'sensitivity_summary.json', summary)
    text += ['## Small sensitivity diagnostic', '',
             '| Method | Exact periods | Including aliases | Nulls with native SDE > 7 |',
             '|---|---:|---:|---:|']
    for key, label, _ in methods:
        row = summary.get(key, {})
        if row.get('status') != 'ok':
            text.append(f'| {label} | no completed result | — | — |')
        else:
            text.append(f'| {label} | {row["exact_recoveries"]}/{row["n_injected"]} | '
                        f'{row["alias_inclusive_recoveries"]}/{row["n_injected"]} | '
                        f'{row["native_null_above_7"]}/{row["n_null"]} |')
    text += ['', 'These are diagnostic counts for this particular injection set, not population '
             'completeness estimates. Sixteen nulls are insufficient to validate low false-alarm rates. '
             'The score curves do not remove the need for larger paired injections and real noise.', '']
    (root/'NEW_RESULTS.md').write_text('\n'.join(text))
    write_json(root/'tls_timing_summary.json', tls_rows)
    print('Wrote NEW_RESULTS.md and sensitivity diagnostic')


if __name__ == '__main__':
    main()
