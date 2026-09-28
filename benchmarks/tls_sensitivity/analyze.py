#!/usr/bin/env python3
"""Calibrate, then analyze the independent TLS sensitivity experiment.

The primary simultaneous family contains 27 one-sided bounds: nine recall
lower bounds and both false-positive bounds for nine comparisons. Marginal
Wilson intervals and diagnostic contrasts are labeled separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from archive import records as shard_records


PROFILES = ('tess_200s', 'tess_gap', 'ztf')
V1 = ('v1_original', 'v1_resolved', 'v1_fine')
COUNTS = dict(calibration=4096, injections=2048, nulls=4096)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing to overwrite a different frozen analysis: {path}')
    path.write_text(text)


def valid(c):
    return c.get('api_result_valid', False) and not c.get('error') and c.get('finite_fraction', 0) > 0


def score(c):
    return float(c['score']) if valid(c) and c.get('score') is not None and np.isfinite(c['score']) else -np.inf


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(max(0, center-half)), float(min(1, center+half))]


def paired(a, b, alpha=.05):
    """Each bound separately covers the difference with >=1-alpha probability.

    D = Pr(A only) - Pr(B only). Bound the two discordant multinomial cells
    with one-sided Clopper-Pearson bounds at alpha/2, then subtract. Their
    Bonferroni coverage does not assume independence between the cells.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or not len(a):
        raise ValueError('Paired vectors must have equal nonzero length')
    n = len(a)
    wins, losses = int(np.sum(a & ~b)), int(np.sum(~a & b))
    def lo(k):
        return float(beta.ppf(alpha/2, k, n-k+1)) if k else 0.
    def hi(k):
        return float(beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.
    return dict(n=n, a_only=wins, b_only=losses, difference=float(a.mean()-b.mean()),
                lower=lo(wins)-hi(losses), upper=hi(wins)-lo(losses), alpha_per_bound=alpha)


def load(root, configs, splits, v1_methods=V1):
    groups = {}
    signatures = {}
    for profile in PROFILES:
        for method in (*v1_methods, f'gtls_{profile}'):
            wanted = json.loads((configs / f'{method}.json').read_text())
            for split in splits:
                records = []
                for path, record in shard_records(root, profile, split, method):
                    if record['config'] != wanted:
                        raise ValueError(f'Wrong execution configuration: {path}')
                    if record['status'] != 'ok' or record['count'] != 128 or len(record['cases']) != 128:
                        raise ValueError(f'Incomplete shard: {path}')
                    module = 'gputls' if method.startswith('gtls') else 'cuvarbase'
                    signature = json.dumps(record['installed_sources'][module], sort_keys=True)
                    if module in signatures and signatures[module] != signature:
                        raise ValueError(f'Installed {module} source bytes differ: {path}')
                    signatures[module] = signature
                    records.extend(record['cases'])
                records.sort(key=lambda c: c['global_index'])
                if [c['global_index'] for c in records] != list(range(COUNTS[split])):
                    raise ValueError(f'Missing or duplicated cases: {profile}/{method}/{split} ({len(records)})')
                groups[profile, method, split] = records
        # Every method must have received the same prepared observations.
        for split in splits:
            base = groups[profile, v1_methods[0], split]
            for method in (*v1_methods[1:], f'gtls_{profile}'):
                for a, b in zip(base, groups[profile, method, split]):
                    for key in ('seed', 'array_sha256', 'period', 'duration', 'injected'):
                        if a[key] != b[key]:
                            raise ValueError(f'Input mismatch: {profile}/{method}/{split}/{a["global_index"]}/{key}')
    return groups, {k: hashlib.sha256(v.encode()).hexdigest() for k, v in signatures.items()}


def calibrate(groups):
    thresholds = {}
    for (profile, method, split), cases in groups.items():
        if split != 'calibration' or any(c['injected'] for c in cases):
            raise ValueError('Calibration requires exclusively independent nulls')
        threshold = float(np.quantile([score(c) for c in cases], .95, method='higher'))
        if not np.isfinite(threshold):
            raise ValueError('Nonfinite calibration threshold')
        digest = hashlib.sha256(json.dumps([(c['global_index'], c.get('score'), valid(c), c['array_sha256'])
                                           for c in cases], sort_keys=True).encode()).hexdigest()
        thresholds[f'{profile}/{method}'] = dict(threshold=threshold, n=len(cases),
                                                invalid=sum(not valid(c) for c in cases),
                                                calibration_digest=digest)
    return thresholds


def analyze(groups, thresholds):
    methods, comparisons, diagnostics = [], [], []
    vectors = {}
    for profile in PROFILES:
        for method in (*V1, f'gtls_{profile}'):
            threshold = thresholds[f'{profile}/{method}']['threshold']
            injections, nulls = (groups[profile, method, s] for s in ('injections', 'nulls'))
            recovered = np.array([valid(c) and c['recovered'] for c in injections], bool)
            detection = recovered & np.array([score(c) > threshold for c in injections])
            alias_detection = np.array([valid(c) and c['alias_recovered'] and score(c) > threshold
                                        for c in injections], bool)
            fp = np.array([score(c) > threshold for c in nulls])
            vectors[profile, method] = detection, fp
            snr_rows = []
            for snr in (6., 8., 10., 14.):
                take = np.array([c['target_white_oracle_snr'] == snr for c in injections])
                n, k = int(take.sum()), int(detection[take].sum())
                assert n == 512
                snr_rows.append(dict(snr=snr, n=n, detected=k, recall=k/n, marginal_wilson_95=wilson(k,n)))
            methods.append(dict(profile=profile, method=method, threshold=threshold,
                                n_injections=len(injections), n_nulls=len(nulls),
                                detected=int(detection.sum()), period_recovered=int(recovered.sum()),
                                alias_detected=int(alias_detection.sum()),
                                false_positives=int(fp.sum()), recall=float(detection.mean()), fpr=float(fp.mean()),
                                recall_marginal_wilson_95=wilson(int(detection.sum()),len(injections)),
                                fpr_marginal_wilson_95=wilson(int(fp.sum()),len(nulls)), by_snr=snr_rows,
                                invalid_injections=sum(not valid(c) for c in injections),
                                invalid_nulls=sum(not valid(c) for c in nulls),
                                partial_injections=sum(valid(c) and c['finite_fraction'] < 1 for c in injections),
                                partial_nulls=sum(valid(c) and c['finite_fraction'] < 1 for c in nulls)))
        gtls_d, gtls_fp = vectors[profile, f'gtls_{profile}']
        for method in V1:
            detection, fp = vectors[profile, method]
            recall_bound = paired(detection, gtls_d, .05/27)
            fpr_bounds = paired(fp, gtls_fp, .05/27)
            comparisons.append(dict(profile=profile, method=method,
                                    recall_difference=recall_bound['difference'],
                                    recall_lower_simultaneous_95=recall_bound['lower'],
                                    fpr_difference=fpr_bounds['difference'],
                                    fpr_lower_simultaneous_95=fpr_bounds['lower'],
                                    fpr_upper_simultaneous_95=fpr_bounds['upper'],
                                    recall_nominal_one_sided_bounds=paired(detection, gtls_d),
                                    fpr_nominal_one_sided_bounds=paired(fp, gtls_fp),
                                    supported=recall_bound['lower'] > -.05 and fpr_bounds['lower'] > -.02 and fpr_bounds['upper'] < .02))
        fine_d, fine_fp = vectors[profile, 'v1_fine']
        for method in V1[:2]:
            detection, fp = vectors[profile, method]
            diagnostics.append(dict(profile=profile, method=method, reference='v1_fine',
                                    recall_nominal_one_sided_bounds=paired(detection, fine_d),
                                    fpr_nominal_one_sided_bounds=paired(fp, fine_fp)))
    return dict(methods=methods, comparisons=comparisons, resolution_diagnostics=diagnostics)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--thresholds', type=Path, required=True)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.calibrate:
        groups, signatures = load(a.root, a.configs, ('calibration',))
        write(a.thresholds, dict(thresholds=calibrate(groups), source_signatures=signatures,
                                quantile=.95, method='higher', decision='strict exceedance'))
    else:
        if a.out is None:
            ap.error('--out is required for held-out analysis')
        frozen = json.loads(a.thresholds.read_text())
        groups, signatures = load(a.root, a.configs, ('injections', 'nulls'))
        if signatures != frozen['source_signatures']:
            raise ValueError('Installed sources changed after calibration')
        write(a.out, analyze(groups, frozen['thresholds']))
