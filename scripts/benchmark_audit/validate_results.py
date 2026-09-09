#!/usr/bin/env python3
"""Check identical inputs and sampled LS accuracy against direct float64 fits.

No GPU, no benchmark timing. Uses immutable campaign inputs and JSON outputs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from astropy.timeseries import LombScargle

from common import array_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'analysis/benchmark-audit-20260906')
    args = ap.parse_args()
    root = args.root
    rows = {}
    failures = []
    groups = {}
    for p in sorted((root/'results').glob('ls_*.json')):
        d = json.loads(p.read_text())
        if d.get('status') != 'ok':
            rows[p.name] = {'executed': False, 'status': d.get('status')}
            continue
        a = d['args']
        key = (a['config'], a['n_lcs'], a['shared_times'])
        groups.setdefault(key, []).append((p, d))
    for (cfg, n, shared), items in groups.items():
        inp = root/'inputs'/f'ls_{cfg}_{"shared" if shared else "distinct"}.npz'
        with np.load(inp) as data:
            t, y, dy, f = data['t'][:n], data['y'][:n], data['dy'][:n], data['freqs']
        expected = array_hash(f, *[a for lc in zip(t, y, dy) for a in lc])
        indices = np.unique(np.concatenate([d['validation_indices'] for _, d in items]))
        # Reference calculation is direct (Astropy cython), not an FFT-based
        # implementation sharing the same approximation as the competitors.
        ref = np.asarray([LombScargle(t[i], y[i], dy[i], normalization='standard',
                                     fit_mean=True, center_data=True).power(
                                         f[indices], method='cython') for i in range(n)])
        for p, d in items:
            ix = np.searchsorted(indices, d['validation_indices'])
            truth = ref[:, ix]
            actual = np.asarray(d['validation_power'])
            error = np.abs(actual-truth)
            same = expected == d['input_sha256']
            finite = bool(np.isfinite(actual).all())
            # Report metrics rather than silently claiming equivalent accuracy.
            # 1e-3 absolute normalized power is an exploratory quality screen;
            # it is not a weak-signal completeness/FAP guarantee.
            tolerance_pass = bool(np.max(error) <= 1e-3)
            peak = np.asarray(d['peak_frequency'])
            reference_peak = f[indices[np.argmax(ref, axis=1)]]
            peak_bins = np.abs(peak-reference_peak)/(f[1]-f[0])
            good_peak = bool(np.max(peak_bins) <= 1.01)
            rows[p.name] = dict(executed=True, input_identical=same,
                                max_abs_power_error=float(error.max()),
                                median_abs_power_error=float(np.median(error)),
                                p99_abs_power_error=float(np.quantile(error, .99)),
                                max_peak_error_bins=float(np.max(peak_bins)),
                                normalized_power_atol=1e-3,
                                sampled_power_pass=tolerance_pass,
                                sampled_peak_pass=good_peak,
                                finite=finite,
                                eligible=bool(same and finite and tolerance_pass and good_peak))
            if not same:
                failures.append(p.name + ': input bytes differ')
        print(cfg, n, 'shared' if shared else 'distinct', 'validated', len(items), flush=True)
    tls = {}
    for p in sorted([*(root/'results').glob('tls_*.json'),
                     *(root/'results').glob('ensemble_*.json')]):
        d = json.loads(p.read_text())
        if d.get('status') != 'ok':
            tls[p.name] = dict(executed=False, status=d.get('status'))
            continue
        a = d['args']
        input_path = root/'inputs'/Path(a['input']).name
        with np.load(input_path) as data:
            n_periods = len(data['periods'])
            grid_atol = 2*float(np.finfo(np.float32).eps)*float(np.max(data['periods']))
            expected = array_hash(data['periods'], *[
                data[f'{field}_{i}'] for i in range(a['n_lcs']) for field in ['t', 'y', 'dy']])
        same = expected == d['input_sha256']
        rec = d['recovery']
        native = all(r['period'] is not None and r['native_sde'] is not None for r in rec)
        grid_matches = all(r['periods_returned']==n_periods and
                           r['period_grid_max_error'] is not None and
                           r['period_grid_max_error'] <= grid_atol for r in rec)
        tls[p.name] = dict(executed=True, input_identical=same, native_outputs_finite=native,
                           returned_grid_matches=grid_matches, period_grid_atol=grid_atol,
                           exact_recoveries=sum(r['exact_recovery'] is True for r in rec),
                           alias_inclusive_recoveries=sum(r['alias_recovery'] is True for r in rec),
                           n_injected=sum(r['injected'] for r in rec),
                           min_valid_periods=min(r['periods_finite'] for r in rec),
                           eligible=bool(same and native and grid_matches))
        if not same:
            failures.append(p.name + ': input bytes differ')
    write_json(root/'validation.json', dict(
        reference='Astropy direct cython GLS, float64, floating mean, standard normalization',
        limitations='Sampled periodogram checks plus peak checks; no complete error bound or FAP calibration',
        ls=rows, tls=tls, input_failures=failures))
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
