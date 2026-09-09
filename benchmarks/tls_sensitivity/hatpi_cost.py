#!/usr/bin/env python3
"""Price a synthetic HATPI-like season; this is not observed HATPI evidence.

102 clear eight-hour nights are drawn within 196 days. Native 30-second
measurements and weighted five-minute averages come from the same four
synthetic lightcurves. Their common 0.6--12-day period grid is prepared before
timing. This pilot estimates search cost, not equivalent recovery.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import batman
import numpy as np

from run import Backend, dump, sha


CONFIGS = {
    'v1_bls': dict(backend='v1_bls_batch', noverlap=4, qmin_fac=.5),
    'v1_tls_original': dict(backend='v1_tls', epoch_os=4, durations=16),
    'v1_tls_intermediate': dict(backend='v1_tls', epoch_os=8, durations=16, nbins=4096),
    'v1_tls_fine': dict(backend='v1_tls', epoch_os=16, durations=32, nbins=8192),
    'gtls': dict(backend='gtls', density=True, fast=True, workers=1, margin=.125,
                 release_cache=True),
}


def prepare(root):
    from cuvarbase.tls_grids import period_grid_ofir
    rng = np.random.default_rng(2026090931)
    nights = np.sort(rng.choice(np.arange(196), size=102, replace=False))
    t = np.concatenate([n + .25 + np.arange(960) * 30 / 86400 for n in nights])
    periods = period_grid_ofir(t, period_min=.6, period_max=12., oversampling_factor=3)
    q = np.arcsin((1 / (periods * 8.6307))**(2/3)) / np.pi
    common = dict(tls_periods=periods, freqs=1/periods, q=q)
    profiles = {mode: dict(common) for mode in ('native30s', 'binned300s')}
    truth = []
    for i, period in enumerate((2.3, 5.7, 2.3, 5.7)):
        pm = batman.TransitParams()
        pm.t0, pm.per, pm.rp = .37 + i * .21, period, .05
        pm.a = (6.6743e-11 * 1.9884e30 * (period * 86400)**2 / (4*np.pi**2))**(1/3) / 6.957e8
        pm.inc = float(np.degrees(np.arccos(.5 / pm.a)))
        pm.ecc, pm.w, pm.u, pm.limb_dark = 0., 90., [.4804, .1867], 'quadratic'
        signal = 1 - batman.TransitModel(pm, t, supersample_factor=7,
                                        exp_time=30/86400).light_curve(pm)
        sigma = np.sqrt(np.sum((signal - signal.mean())**2)) / 10
        dy = sigma * (.8 + .4 * rng.random(len(t)))
        y = 1 - (signal if i < 2 else 0) + rng.normal(size=len(t)) * dy
        weights = dy.reshape(-1, 10)**-2
        total = weights.sum(axis=1)
        averaged = (
            np.sum(weights * t.reshape(-1, 10), axis=1) / total,
            np.sum(weights * y.reshape(-1, 10), axis=1) / total,
            1 / np.sqrt(total))
        for mode, lc in [('native30s', (t, y, dy)), ('binned300s', averaged)]:
            profiles[mode].update({f'{k}_{i}': v for k, v in zip(('t', 'y', 'dy'), lc)})
        truth.append(dict(index=i, injected=i < 2, period=period, epoch=pm.t0))
    for mode, arrays in profiles.items():
        arrays['metadata'] = np.array(json.dumps(dict(
            profile=mode, split='cost_pilot', pmin=.6, pmax=12., baseline=float(np.ptp(t)),
            cases=truth, seed=2026090931, n_clear_nights=102, season_days=196,
            provenance='Entirely synthetic HATPI-like observing pattern. No HATPI lightcurve was downloaded.',
            time_binning='Ten consecutive original samples per average within each night, inverse-variance weighted.')))
        np.savez_compressed(root / f'{mode}.npz', **arrays)


def measure(args):
    mode, method = args.job.split('/')
    path = args.out / f'{mode}.npz'
    data = np.load(path)
    lcs = [tuple(data[f'{k}_{i}'] for k in ('t', 'y', 'dy')) for i in range(4)]
    result = dict(status='running', mode=mode, method=method, config=CONFIGS[method],
                  input_sha256=sha(path), n_points=[len(lc[0]) for lc in lcs],
                  n_periods=len(data['tls_periods']), repetitions=[])
    output = args.out / mode / method / 'summary.json'
    dump(output, result)
    begin = time.perf_counter()
    b = Backend(CONFIGS[method], data, max(len(lc[0]) for lc in lcs))
    b.sync()
    result['initialization_s'] = time.perf_counter() - begin
    # The serial GTLS API has no cross-source batch amortization. Use three
    # distinct warm single-source calls to keep the pricing pilot bounded.
    workload = ([('first_api', lcs[:1]), *[(f'rep{i}', lcs[i+1:i+2]) for i in range(3)]]
                if method == 'gtls' else
                [('first_api', lcs[:1]), ('batch_warmup', lcs), *[(f'rep{i}', lcs) for i in range(3)]])
    for label, selected in workload:
        b.sync()
        begin = time.perf_counter()
        outputs = b.search(selected)
        b.sync()
        elapsed = time.perf_counter() - begin
        candidates = [o['candidate'] for o in outputs]
        valid = len(outputs) == len(selected) and all(
            not o.get('error') and o['candidate'].get('api_result_valid', True)
            and o['candidate']['finite_fraction'] == 1 for o in outputs)
        result['repetitions'].append(dict(label=label, elapsed_s=elapsed,
                                           n_sources=len(selected), valid=valid, candidates=candidates,
                                           errors=[o.get('error') for o in outputs if o.get('error')]))
        dump(output, result)
        if not valid:
            raise RuntimeError('Invalid cost-pilot API result; evidence retained')
    result.update(status='ok', seconds_per_source=float(np.median(
        [r['elapsed_s'] / r['n_sources'] for r in result['repetitions'] if r['label'].startswith('rep')])))
    dump(output, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=600, help='Maximum seconds per configuration process')
    ap.add_argument('--job', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.job:
        measure(a)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    prepare(a.out)
    jobs = [f'{p}/{m}' for p in ('native30s', 'binned300s') for m in CONFIGS]
    np.random.default_rng(2026090932).shuffle(jobs)
    statuses = []
    for job in jobs:
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(a.out),
                            '--job', job], timeout=a.timeout, check=True)
            statuses.append(dict(job=job, status='ok'))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            statuses.append(dict(job=job, status=type(e).__name__))
        dump(a.out / 'status.json', statuses)
        print(statuses[-1], flush=True)


if __name__ == '__main__':
    main()
