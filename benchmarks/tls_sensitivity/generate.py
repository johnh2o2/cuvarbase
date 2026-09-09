#!/usr/bin/env python3
"""Independent, reproducible transit cohorts on frozen observed cadences.

Cadence archives contain t, relative_error, band, exposure_days, freqs, q,
tls_periods, and JSON metadata. Each case has an independent SeedSequence;
changing the shard size or running on another machine does not change it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import batman
import numpy as np
from numba import njit


PROFILES = ('tess_200s', 'tess_gap', 'ztf')
SPLITS = ('calibration', 'injections', 'nulls')
SEED = 2026090917


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@njit(cache=True)
def ou_noise(t, z, tau):
    red = np.empty(len(t))
    red[0] = z[0]
    for j in range(1, len(t)):
        a = np.exp(-(t[j] - t[j - 1]) / tau)
        red[j] = a * red[j - 1] + np.sqrt(1 - a * a) * z[j]
    return red


def make_case(cadence, profile, split, index):
    seed = [SEED, PROFILES.index(profile), SPLITS.index(split), index]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    raw = [cadence[k] for k in ('t', 'relative_error', 'band', 'exposure_days')]
    keep = rng.random(len(raw[0])) > rng.uniform(0, .03)
    t, relative, band, exposure = [v[keep] for v in raw]
    cm = json.loads(str(cadence['metadata']))
    target = (6., 8., 10., 14.)[index % 4]
    injected = split == 'injections'
    for attempt in range(1, 1001):
        period = float(np.exp(rng.uniform(np.log(.8), np.log(min(12., .8 * cm['pmax'])))))
        epoch = float(rng.uniform(0, period))
        rp = float(rng.choice([.025, .05, .10]))
        impact = float(rng.uniform(0, .85))
        a = (6.6743e-11 * 1.9884e30 * (period * 86400)**2 / (4 * np.pi**2))**(1/3) / 6.957e8
        pm = batman.TransitParams()
        pm.t0, pm.per, pm.rp, pm.a = epoch, period, rp, a
        pm.inc, pm.ecc, pm.w = float(np.degrees(np.arccos(impact / a))), 0., 90.
        pm.u, pm.limb_dark = [.4804, .1867], 'quadratic'
        model = np.ones(len(t))
        for exp in np.unique(exposure):
            take = exposure == exp
            model[take] = batman.TransitModel(pm, t[take], supersample_factor=7,
                                             exp_time=float(exp)).light_curve(pm)
        inside = model < 1 - 1e-9
        events = np.unique(np.rint((t[inside] - epoch) / period).astype(int))
        if inside.sum() >= 5 and len(events) >= 2:
            break
    else:
        raise RuntimeError('Observable-injection sampling exhausted')
    w = relative**-2
    signal = model - np.dot(w, model) / w.sum()
    white_scale = np.sqrt(np.dot(w, signal * signal)) / target
    dy = white_scale * relative
    red = ou_noise(t, rng.normal(size=len(t)), .15 if profile.startswith('tess') else 1.)
    y = (model if injected else np.ones(len(t))) + rng.normal(size=len(t)) * dy + .25 * np.median(dy) * red
    duration = period / np.pi * np.arcsin(np.sqrt((1 + rp)**2 - impact**2) / np.sqrt(a*a - impact**2))
    data = dict(t=t, y=y, dy=dy, band=band)
    truth = dict(global_index=index, seed=seed, injected=injected,
                 target_white_oracle_snr=target if injected else None,
                 period=period, epoch=epoch, rp=rp, impact=impact,
                 duration=float(duration), ndata=len(t), n_in_transit=int(inside.sum()),
                 observed_transit_events=len(events), ephemeris_draws=attempt,
                 noise_scale=float(white_scale),
                 white_oracle_snr=float(np.sqrt(np.sum((signal / dy)**2))) if injected else None,
                 array_sha256={k: array_hash(v) for k, v in data.items()})
    return data, truth


def generate(cadence_path, profile, split, start, count, out):
    with np.load(cadence_path) as cadence:
        metadata = json.loads(str(cadence['metadata']))
        arrays = {k: cadence[k] for k in ('freqs', 'q', 'tls_periods')}
        rows = []
        for local_index, index in enumerate(range(start, start + count)):
            data, truth = make_case(cadence, profile, split, index)
            truth['index'] = local_index
            rows.append(truth)
            arrays.update({f'{k}_{local_index}': v for k, v in data.items()})
    metadata.update(profile=profile, split=split, start=start, count=count, cases=rows,
                    generator_sha256=sha(__file__), cadence_sha256=sha(cadence_path))
    arrays['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return metadata


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cadence', type=Path, required=True)
    ap.add_argument('--profile', choices=PROFILES, required=True)
    ap.add_argument('--split', choices=SPLITS, required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--count', type=int, default=128)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    generate(a.cadence, a.profile, a.split, a.start, a.count, a.out)
