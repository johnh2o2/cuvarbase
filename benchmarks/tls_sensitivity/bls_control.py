#!/usr/bin/env python3
"""Secondary BLS comparison on the TLS study's identical, period-restricted inputs.

These descriptive comparisons are not part of the nine primary TLS/GTLS
noninferiority decisions. The box search also uses a different score/ranker,
so the experiment compares complete searches rather than isolating shape.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import PROFILES, V1, calibrate, load, paired, score, valid, wilson, write


CONTROL = 'v1_bls_control'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--thresholds', type=Path, required=True)
    ap.add_argument('--primary-thresholds', type=Path, required=True)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    splits = ('calibration',) if a.calibrate else ('injections', 'nulls')
    groups, signatures = load(a.root, a.configs, splits, (*V1, CONTROL))
    frozen = json.loads(a.primary_thresholds.read_text())
    if signatures != frozen['source_signatures']:
        raise ValueError('Secondary control does not use the frozen numerical sources')
    if a.calibrate:
        only_control = {key: cases for key, cases in groups.items() if key[1] == CONTROL}
        write(a.thresholds, dict(thresholds=calibrate(only_control), source_signatures=signatures,
                                quantile=.95, method='higher', decision='strict exceedance'))
        return
    if a.out is None:
        ap.error('--out is required for analysis')
    thresholds = {**frozen['thresholds'], **json.loads(a.thresholds.read_text())['thresholds']}
    vectors, methods, contrasts = {}, [], []
    for profile in PROFILES:
        for method in (*V1, CONTROL, f'gtls_{profile}'):
            cut = thresholds[f'{profile}/{method}']['threshold']
            injections, nulls = (groups[profile, method, s] for s in ('injections', 'nulls'))
            detected = np.array([valid(c) and c['recovered'] and score(c) > cut for c in injections], bool)
            fp = np.array([score(c) > cut for c in nulls], bool)
            vectors[profile, method] = detected, fp
            by_snr = []
            for snr in (6., 8., 10., 14.):
                take = np.array([c['target_white_oracle_snr'] == snr for c in injections])
                n, k = int(take.sum()), int(detected[take].sum())
                by_snr.append(dict(snr=snr, n=n, detected=k, recall=k/n,
                                   marginal_wilson_95=wilson(k,n)))
            methods.append(dict(profile=profile, method=method, threshold=cut,
                                detected=int(detected.sum()), n_injections=len(injections),
                                false_positives=int(fp.sum()), n_nulls=len(nulls), by_snr=by_snr,
                                invalid_injections=sum(not valid(c) for c in injections),
                                invalid_nulls=sum(not valid(c) for c in nulls)))
        bd, bf = vectors[profile, CONTROL]
        for method in (*V1, f'gtls_{profile}'):
            td, tf = vectors[profile, method]
            contrasts.append(dict(profile=profile, method=method, reference=CONTROL,
                                  recall_nominal_one_sided_bounds=paired(td, bd),
                                  fpr_nominal_one_sided_bounds=paired(tf, bf)))
    write(a.out, dict(secondary=True, methods=methods, contrasts=contrasts,
                      limitation='Different native ranking statistics and numerical grids beyond the identical period grid; this does not isolate template shape. Intervals are nominal and not adjusted for multiple comparisons.'))


if __name__ == '__main__':
    main()
