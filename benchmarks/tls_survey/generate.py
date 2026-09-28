#!/usr/bin/env python3
"""Generate development, calibration, or sealed independent recovery populations."""
import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import numpy as np
from common import (ROOT, CADENCES, REGIMES, SNRS, module, now, sha, source_identity, write)


def physical_modules():
    return (module(ROOT/'benchmarks/tls_accuracy/diagnose.py', 'survey_physics'),
            module(ROOT/'cuvarbase/tls_reference_math.py', 'survey_reference'),
            module(ROOT/'benchmarks/tls_reference/cases.py', 'survey_cases'))


def hatpi_cadence():
    """Synthetic 30-s samples, eight-hour nights; missing nights and nightly gaps."""
    times = np.concatenate([night+np.arange(0.,8/24,30/86400) for night in range(30)
                            if night % 7 not in (3,4)])
    times = times[(times % 1 < .12) | (times % 1 > .15)]
    errors = 1.+.5*np.sin(np.pi*(times%1)/(8/24))**2
    return times, errors, np.full(len(times),30/86400), np.zeros(len(times),int)


def make_case(regime, split, index, diag, ref, shared, exposure_nodes=64):
    name = '%s_%s_%04d' % (regime,split,index)
    settings = REGIMES[regime]
    rng = shared.deterministic_rng('survey-v1-20260910:'+split,name)
    if settings['cadence'] == 'hatpi':
        t, errors, exposure, band = hatpi_cadence()
        provenance = 'Synthetic HATpi-like 30-second nightly sampling; no real HATpi data'
    else:
        with np.load(CADENCES/(settings['cadence']+'.npz')) as d:
            t,errors,exposure,band = [np.asarray(d[k]).copy() for k in
                                     ('t','relative_error','exposure_days','band')]
        provenance = 'Observed archived survey timestamps; synthetic flux and noise'
    stride = settings.get('stride',1)
    t,errors,exposure,band = (v[::stride] for v in (t,errors,exposure,band))
    if 'exposure_seconds' in settings:
        exposure[:] = settings['exposure_seconds']/86400
    errors = errors/np.median(errors)
    # Independent occasional poor measurements, including ~10x error range.
    errors *= np.exp(rng.uniform(-.5,.5,len(t)))
    radius,mass = (settings.get(k,1.) for k in ('radius','mass'))
    period = rng.uniform(*settings.get('period',(2.,6.)))
    impact = rng.uniform(*settings['impact'])
    eccentricity = rng.uniform(*settings.get('eccentricity',(0.,0.)))
    physical = diag.Regime(name,period,radius=radius,mass=mass,rp=.00916/radius,
                           impact=impact,eccentricity=eccentricity)
    duration,full,a = diag.durations(physical)
    epoch = rng.random()*period
    signal = diag.physical_signal(physical,t,exposure,epoch=epoch,exposure_nodes=exposure_nodes)
    inside = signal > signal.max()*1e-8 if signal.max() > 0 else np.zeros(len(t),bool)
    events = np.unique(np.rint((t[inside]-epoch)/period)).size
    snr = float(SNRS[index%len(SNRS)] if split in ('development','injections') else rng.choice(SNRS))
    norm = shared.weighted_signal_norm(signal,errors)
    # No observability rejection: unsampled and one-event cases remain planned outcomes.
    scale = norm/snr if norm > 0 else 1e-4
    dy = scale*errors
    tau = 1. if settings['cadence']=='ztf' else .15
    amplitude = .25*np.median(dy)
    noise = dy*rng.normal(size=len(t))+shared.ou_noise(t,amplitude,tau,rng)
    injected = split in ('development','injections')
    y = 1.+noise-(signal if injected else 0.)
    bounds = (.6, 12.878375495285127 if settings['cadence']=='tess_200s' else
              10. if settings['cadence']=='ztf' else 5. if settings['cadence']=='hatpi' else 27.457888046800917)
    periods = np.sort(ref.period_grid(float(np.ptp(t)),R_star=radius,M_star=mass,
                                     period_min=bounds[0],period_max=bounds[1],
                                     oversampling_factor=settings.get("grid_oversampling",3)))
    metadata = dict(name=name,regime=regime,split=split,cohort=split,null=not injected,
        purpose='recovery_full_grid',physical=asdict(physical),truth_period=period,truth_epoch=epoch+1.,
        duration_days=duration,full_duration_days=full,ingress_days=.5*(duration-full),
        stellar_density_solar=mass/radius**3,semimajor_stellar_radii=a,
        periastron_stellar_radii=a*(1-eccentricity),baseline_days=float(np.ptp(t)),ndata=len(t),
        cadence=settings['cadence'],cadence_construction=provenance,
        fractional_duration=duration/period,in_transit_observations=int(inside.sum()),observed_events=int(events),
        conditional_population=False,observable=bool(inside.sum()>=5 and events>=2),
        latent_white_oracle_snr=snr,realized_latent_white_oracle_snr=norm/scale,
        exposure_quadrature_nodes=exposure_nodes,truth_inserted_in_grid=False,period_count=len(periods),
        noise=dict(ou_amplitude=float(amplitude),ou_tau_days=tau,oracle_snr_excludes_ou=True),
        period_bounds=list(bounds),
        grid_kwargs=dict(R_star=radius,M_star=mass,period_min=bounds[0],period_max=bounds[1],
                         oversampling_factor=settings.get('grid_oversampling',3),n_transits_min=2),
        nearest_grid_drift_over_half_duration=float(np.min(np.abs(periods/period-1))*np.ptp(t)/(.5*duration)),
        search_kwargs=dict(R_star=radius,M_star=mass,oversampling_factor=3))
    return dict(t=t+1.,y=y,dy=dy,periods=periods,signal=signal,exposure_days=exposure,band=band), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=('development','development_nulls','calibration','injections','nulls'),required=True)
    parser.add_argument('--count',type=int,required=True,help='Per regime; injections balanced over SNR 6/8/10/12')
    parser.add_argument('--regimes',default=','.join(REGIMES))
    parser.add_argument('--seal',type=Path)
    parser.add_argument('--exposure-nodes',type=int,default=64)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Refuse to overwrite an input directory')
    if args.count<1 or args.exposure_nodes<16:
        parser.error('Positive count and at least 16 exposure quadrature nodes required')
    identity=source_identity()
    regimes=args.regimes.split(',')
    if args.split in ('calibration','injections','nulls'):
        if args.seal is None:
            parser.error('Independent generation requires a frozen development seal')
        seal=json.loads(args.seal.read_text())
        if seal['source_identity'] != identity:
            parser.error('Generator/analysis sources changed after development freeze')
        if regimes != seal['regimes'] or args.count != seal['counts'][args.split]:
            parser.error('Requested population differs from frozen plan')
        if args.exposure_nodes != seal['exposure_nodes']:
            parser.error('Exposure quadrature differs from frozen plan')
    # cases.py imports validate from its own directory.
    import sys
    sys.path.insert(0,str(ROOT/'benchmarks/tls_reference'))
    diag,ref,shared=physical_modules()
    args.out.mkdir(parents=True)
    manifest=dict(status='running',suite='survey-'+args.split,created_utc=now(),split=args.split,source_identity=identity,
                  seal_sha256=sha(args.seal) if args.seal else None,
                  count_per_regime=args.count,regimes=regimes,environment=dict(python=platform.python_version(),numpy=np.__version__),cases=[])
    write(args.out/'manifest.json',manifest)
    for regime in regimes:
        for index in range(args.count):
            arrays,metadata=make_case(regime,args.split,index,diag,ref,shared,args.exposure_nodes)
            filename=metadata['name']+'.npz'
            np.savez_compressed(args.out/filename,**arrays,metadata=json.dumps(metadata,sort_keys=True))
            manifest['cases'].append(dict(file=filename,sha256=sha(args.out/filename),metadata=metadata,
                arrays={key:shared.array_hash(value) for key,value in arrays.items()}))
        write(args.out/'manifest.json',manifest)
        print(json.dumps(dict(regime=regime,completed=len(manifest['cases']))),flush=True)
    manifest['status']='complete'
    write(args.out/'manifest.json',manifest)


if __name__=='__main__':
    main()
