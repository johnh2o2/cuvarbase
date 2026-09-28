#!/usr/bin/env python3
"""Isolate phase-bin compression at known transit parameters on old inputs.

This is a white-noise matched-filter diagnostic, NOT a search/recovery test.
It holds period, epoch, duration, and the cuvarbase template fixed. Expected
signal-to-noise is computed from each filter's actual weighted variance, not
from either package's SDE or the coarse kernel's bin-averaged T-squared norm.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import batman
import numpy as np


def load_tables(source):
    spec = importlib.util.spec_from_file_location('frozen_tls_models', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BATMAN_AVAILABLE
    return module.generate_template_tables(n_table=1024, u=[.4804, .1867])


def bin_filter(t, period, epoch, duration, nbins, integral):
    # Same per-lightcurve phase origin as the public fast API; float64 here
    # deliberately excludes folding roundoff from the compression diagnostic.
    origin = np.floor(t.min())
    phase = ((t - origin) / period) % 1
    phase_center = ((epoch - origin) / period) % 1
    bins = np.floor(phase * nbins).astype(int)
    mid = (((bins + .5) / nbins - phase_center + .5) % 1) - .5
    c0 = (mid - .5 / nbins) / (.5 * duration / period)
    c1 = (mid + .5 / nbins) / (.5 * duration / period)
    knots = np.linspace(-1, 1, len(integral))
    return (np.interp(c1, knots, integral) - np.interp(c0, knots, integral)) / (c1 - c0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-inputs', type=Path, required=True)
    ap.add_argument('--template-source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    template, integral, squared_integral = load_tables(a.template_source)
    knots = np.linspace(-1, 1, len(template))
    rows = []
    for profile in ('tess_200s', 'tess_gap', 'ztf'):
        with np.load(a.old_inputs / f'{profile}_heldout.npz') as d:
            metadata = json.loads(str(d['metadata']))
            for truth in metadata['cases']:
                if not truth['injected']:
                    continue
                i = truth['index']
                t, dy, bands = (d[f'{k}_{i}'] for k in ('t', 'dy', 'band'))
                p, epoch, duration = (truth[k] for k in ('period', 'epoch', 'duration'))
                pm = batman.TransitParams()
                pm.t0, pm.per, pm.rp = epoch, p, truth['rp']
                pm.a = (6.6743e-11 * 1.9884e30 * (p * 86400)**2 / (4 * np.pi**2))**(1/3) / 6.957e8
                pm.inc = float(np.degrees(np.arccos(truth['impact'] / pm.a)))
                pm.ecc, pm.w, pm.u, pm.limb_dark = 0., 90., [.4804, .1867], 'quadratic'
                signal = np.empty(len(t))
                for band in np.unique(bands):
                    take = bands == band
                    seconds = {1: 1800, 27: 600, 67: 200}[int(band)] if profile.startswith('tess') else 30
                    signal[take] = 1 - batman.TransitModel(pm, t[take], supersample_factor=7,
                                                          exp_time=seconds / 86400).light_curve(pm)
                # Match the fast kernel's weight regularizer, while isolating
                # white noise. No OU recovery inference is made from this test.
                w = 1 / (dy*dy + 1e-10)
                def snr(filt):
                    return float(np.dot(w, signal * filt) / np.sqrt(np.dot(w*w*dy*dy, filt*filt)))
                phase = ((t - epoch + .5 * p) % p) - .5 * p
                direct = np.interp(phase / (.5 * duration), knots, template, left=0., right=0.)
                direct_snr = snr(direct)
                oracle_snr = float(np.sqrt(np.dot(1 / (dy*dy), signal * signal)))
                # This box has oracle center and duration. It is only a shape
                # control, not a timing or sensitivity comparison with BLS.
                box_snr = snr((np.abs(phase) <= .5 * duration).astype(float))
                qmin = .5 * np.arcsin((1 / (p * 8.6307))**(2/3)) / np.pi
                automatic = int(2**np.ceil(np.log2(max(256, 4 / qmin))))
                t23 = p / np.pi * np.arcsin(np.sqrt((1 - pm.rp)**2 - truth['impact']**2) /
                                             np.sqrt(pm.a**2 - truth['impact']**2))
                ingress = .5 * (duration - t23)
                for label, bins in [('original_auto', automatic), ('1024', 1024), ('4096', 4096), ('8192', 8192)]:
                    averaged = bin_filter(t, p, epoch, duration, bins, integral)
                    averaged_squared = bin_filter(t, p, epoch, duration, bins, squared_integral)
                    binned_snr = snr(averaged)
                    rows.append(dict(profile=profile, index=i, config=label, phase_bins=bins,
                                     period_days=p, duration_days=duration, ingress_days=float(ingress),
                                     bins_across_transit=duration / p * bins, bins_across_ingress=ingress / p * bins,
                                     direct_snr=direct_snr, binned_snr=binned_snr, oracle_snr=oracle_snr,
                                     box_snr=box_snr, binned_over_direct_snr=binned_snr / direct_snr,
                                     coarse_norm_over_actual_noise=float(np.sqrt(np.dot(w, averaged_squared) /
                                                                                np.dot(w*w*dy*dy, averaged*averaged)))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for profile in ('tess_200s', 'tess_gap', 'ztf'):
        for config in ('original_auto', '4096', '8192'):
            subset = [r for r in rows if r['profile'] == profile and r['config'] == config]
            loss = np.array([1 - r['binned_over_direct_snr'] for r in subset])
            print(profile, config, 'SNR loss quantiles 0/50/95/100 %:',
                  np.round(100 * np.quantile(loss, [0, .5, .95, 1]), 3).tolist())


if __name__ == '__main__':
    main()
