#!/usr/bin/env python3
"""Write immutable transit inputs, independent of the benchmark environments."""
import argparse
import importlib.util
import json
from pathlib import Path

import batman
import numpy as np

from common import array_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=float, default=27)
    ap.add_argument('--n-lcs', type=int, default=16)
    ap.add_argument('--cadence-min', type=float, default=30)
    ap.add_argument('--ensemble', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    module_path = Path(__file__).resolve().parents[2] / 'cuvarbase/tls_grids.py'
    spec = importlib.util.spec_from_file_location('tls_grids', module_path)
    grids = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grids)
    cadence = args.cadence_min / (24 * 60)
    t_regular = np.arange(round(args.baseline / cadence)) * cadence
    periods = grids.period_grid_ofir(t_regular, period_min=0.6,
                                    oversampling_factor=3, R_star=1, M_star=1)
    ps = periods * 86400
    qmin = np.minimum(695508000 * .05 * (4 * ps / (20848 * 1e15))**(1/3) / ps, .15)
    qmax = np.minimum((695508000 * 4 + 69911000 * 2) *
                     (4 * ps / (416970 * 1e15))**(1/3) / ps, .15)
    qmin = np.clip(qmin, 1e-5, .15 * .999)
    qmax = np.clip(qmax, qmin * 1.0001, .999)
    data = dict(periods=periods, qmin=qmin, qmax=qmax)
    meta = dict(args=vars(args), n_periods=len(periods), cases=[],
                note='Wide GTLS-style duration window, nominal epoch density 8; '
                     'templates and discrete duration/epoch grids still differ.')
    for i in range(args.n_lcs):
        r = np.random.RandomState(29000 + i)
        t = t_regular.copy()
        if args.ensemble:
            # Missing observations, several durations/periods/impact parameters,
            # white and correlated noise; every fourth case is a null.
            t = t[r.uniform(size=len(t)) > .1]
            p = float(r.uniform(1.0, min(args.baseline / 3, 12)))
            impact = float([0, .5, .85][i % 3])
            depth = float([0, .00035, .0007, .0014][i % 4])
            noise = .001
        else:
            p, impact, depth, noise = 8.13, 0.0, .004, .004
        dy = np.full(len(t), noise)
        if args.ensemble:
            dy *= r.uniform(.8, 1.2, len(t))
        y = 1.0 + r.normal(size=len(t)) * dy
        red = bool(args.ensemble and i % 8 >= 4)
        if red:
            z = r.normal(size=len(t))
            for j in range(1, len(z)):
                z[j] = .8 * z[j - 1] + .6 * z[j]
            y += noise * .5 * z
        a = (6.67430e-11 * 1.9884e30 * (p * 86400)**2 /
             (4 * np.pi**2))**(1/3) / 6.957e8
        epoch = float(r.uniform(0, p)) if args.ensemble else .35 * p
        if depth:
            pm = batman.TransitParams()
            pm.t0, pm.per, pm.rp, pm.a = epoch, p, float(np.sqrt(depth)), float(a)
            pm.inc = float(np.degrees(np.arccos(impact / a)))
            pm.ecc, pm.w, pm.u, pm.limb_dark = 0, 90, [.4804, .1867], 'quadratic'
            # Finite exposure integration is appropriate for 30-minute data.
            y += batman.TransitModel(pm, t, supersample_factor=7,
                                    exp_time=cadence).light_curve(pm) - 1
        data.update({f't_{i}': t, f'y_{i}': y, f'dy_{i}': dy})
        meta['cases'].append(dict(index=i, period=p, epoch=epoch, impact=impact,
                                  depth=depth, noise=noise, red_noise=red,
                                  ndata=len(t), injected=depth > 0,
                                  sha256=array_hash(t, y, dy)))
    meta['frequency_sha256'] = array_hash(periods)
    meta['input_sha256'] = array_hash(*data.values())
    data['metadata'] = np.array(json.dumps(meta))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    write_json(Path(args.out).with_suffix('.json'), meta)
    print(args.out, len(periods), 'periods', args.n_lcs, 'lightcurves')


if __name__ == '__main__':
    main()
