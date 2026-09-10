#!/usr/bin/env python3
"""CPU-only physical inputs for numerical parity and frozen recovery checks.

Selected-period stress cases deliberately include truth and aliases. They are
mathematical differential fixtures, never recovery or performance evidence.
Fresh held-out generation requires a matching seal of sources and the plan.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

from validate import array_hash, production_sources, sha, write


REGIMES = {
    'tess_solar': dict(cadence='tess_200s', impact=(.2, .7)),
    'tess_highimpact': dict(cadence='tess_200s', impact=(.94, .96)),
    'tess_eccentric': dict(cadence='tess_200s', impact=(.2, .7), period=(6., 12.), eccentricity=(.7, .8)),
    'tess_mdwarf': dict(cadence='tess_200s', impact=(.2, .7), radius=.1, mass=.1),
    'ztf_solar': dict(cadence='ztf', impact=(.2, .7)),
    'ztf_highimpact': dict(cadence='ztf', impact=(.94, .96)),
    'ztf_mdwarf': dict(cadence='ztf', impact=(.2, .7), radius=.1, mass=.1),
    'tess_gap': dict(cadence='tess_gap', impact=(.2, .7)),
}
SNRS = (6., 8., 10., 12.)

STRESS = [
    dict(name='solar_ordinary', regime='tess_solar'),
    dict(name='solar_highimpact', regime='tess_highimpact'),
    dict(name='solar_eccentric', regime='tess_eccentric'),
    dict(name='dense_m_ordinary', regime='tess_mdwarf'),
    dict(name='dense_m_highimpact', regime='tess_mdwarf', impact=.95),
    dict(name='solar_grazing', regime='tess_solar', impact=1.),
    dict(name='sparse_ordinary', regime='ztf_solar'),
    dict(name='sparse_highimpact', regime='ztf_highimpact'),
    dict(name='sparse_dense_m', regime='ztf_mdwarf'),
    dict(name='gapped_ordinary', regime='tess_gap'),
    dict(name='phase_wrap', regime='tess_solar', phase=1.-2**-23),
    dict(name='phase_ties', regime='tess_solar', duplicate=True),
    dict(name='heteroskedastic', regime='tess_solar', heteroskedastic=True),
    dict(name='flat', regime='tess_solar', flat=True),
    dict(name='noise_only', regime='tess_solar', null=True),
    dict(name='large_absolute_epoch', regime='tess_highimpact', absolute_epoch=2457000.123456789),
    dict(name='cleaning_edges', regime='tess_solar', cleaning_edges=True),
    dict(name='dense_long_solar', regime='tess_solar', period=365.25, impact=.8, phase=5./365.25, repeated_campaigns=8, campaign_spacing_days=180.),
    dict(name='dense_long_mdwarf', regime='tess_mdwarf', period=365.25, impact=.9, phase=5./365.25, repeated_campaigns=8, campaign_spacing_days=180.),
    dict(name='long_solar', regime='ztf_solar', period=365.25, impact=.8),
    dict(name='long_dense_m', regime='ztf_mdwarf', period=365.25, impact=.9),
    dict(name='compact_star_native_extension', regime='ztf_mdwarf', period=10., radius=.012, mass=.6,
         expected_native_cache_failure=True),
    dict(name='sparse_long_native_extension', regime='ztf_solar', period=365.25, thin_points=200,
         expected_native_cache_failure=True),
]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def deterministic_rng(stream, name):
    digest = hashlib.sha256((stream+'\0'+name).encode()).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype='<u4'))


def selected_periods(period, duration, baseline, count=96):
    if count < 60:
        raise ValueError('Native GTLS can choose a zero memory-group size with fewer than 60 periods')
    lower, upper = min(.6, period/4), max(12., period*1.6)
    unrelated = np.geomspace(lower, upper, count-9)
    step = duration*period/max(baseline, period)/4
    special = np.array([period/2, period/3, period*2/3, period*1.5, period*2,
                        period-2*step, period-step, period, period+step, period+2*step])
    result = np.unique(np.r_[unrelated, special])
    assert len(result) >= 60
    return result.astype(np.float64)


def ou_noise(times, amplitude, tau, rng):
    result = np.empty(len(times), dtype=np.float64)
    result[0] = amplitude*rng.normal()
    for i in range(1, len(times)):
        rho = np.exp(-(times[i]-times[i-1])/tau)
        result[i] = rho*result[i-1]+amplitude*np.sqrt(max(0., 1-rho*rho))*rng.normal()
    return result


def weighted_signal_norm(signal, relative_error):
    w = relative_error**-2
    center = np.dot(w, signal)/w.sum()
    return float(np.sqrt(np.dot(w, (signal-center)**2)))


def source_identity(repo_root):
    import batman
    import scipy
    cadence_root = repo_root/'benchmarks/results/tls_sensitivity_2026-09-09/cadences'
    return dict(generator_sha256=sha(__file__), harness_sha256=sha(Path(__file__).with_name('validate.py')),
                generation_environment=dict(python=platform.python_version(), platform=platform.platform(),
                    machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__, batman=batman.__version__),
                validation_sources={p.name: sha(p) for p in sorted(Path(__file__).parent.iterdir())
                                    if p.suffix in ('.py', '.md') and p.is_file()},
                diagnostic_sha256=sha(repo_root/'benchmarks/tls_accuracy/diagnose.py'),
                production_sources=production_sources(repo_root),
                cadences={p.name: sha(p) for p in sorted(cadence_root.glob('*.npz'))})


def build_case(name, regime_name, snr, null, stream, diag, ref, cadence_root,
               full_grid, stress=None, exposure_nodes=64, conditional=True):
    stress = stress or {}
    settings = dict(REGIMES[regime_name])
    rng = deterministic_rng(stream, name)
    if null and full_grid:
        # I.i.d. nulls from the predeclared equal mixture, rather than pooling
        # two fixed-size populations and silently treating them as binomial.
        snr = float(rng.choice(SNRS))
    path = cadence_root/(settings['cadence']+'.npz')
    with np.load(path, allow_pickle=False) as data:
        times = data['t'].astype(np.float64)
        errors = data['relative_error'].astype(np.float64)
        exposures = data['exposure_days'].astype(np.float64)
        band = data['band'].copy()
    if stress.get('repeated_campaigns'):
        count = int(stress['repeated_campaigns'])
        spacing = float(stress['campaign_spacing_days'])
        times = np.concatenate([times+i*spacing for i in range(count)])
        errors, exposures, band = (np.tile(v, count) for v in (errors, exposures, band))
    if 'thin_points' in stress:
        index = np.unique(np.round(np.linspace(0, len(times)-1, stress['thin_points'])).astype(int))
        times, errors, exposures, band = (x[index] for x in (times, errors, exposures, band))
    if stress.get('duplicate'):
        index = np.sort(np.r_[np.arange(len(times)), np.arange(0, len(times), 251)])
        times, errors, exposures, band = (x[index] for x in (times, errors, exposures, band))
    if stress.get('heteroskedastic'):
        errors *= np.exp(rng.uniform(-2., 2., len(times)))
    errors /= np.median(errors)
    radius, mass = (stress.get(key, settings.get(key, 1.)) for key in ('radius', 'mass'))
    for attempt in range(1, 10001):
        period = stress.get('period', rng.uniform(*settings.get('period', (2., 6.))))
        impact = stress.get('impact', rng.uniform(*settings['impact']))
        eccentricity = rng.uniform(*settings.get('eccentricity', (0., 0.)))
        physical = diag.Regime(name, period, radius=radius, mass=mass, rp=.00916/radius,
                              impact=impact, eccentricity=eccentricity)
        duration, full_duration, semimajor = diag.durations(physical)
        epoch = stress.get('phase', rng.uniform())*period
        signal = diag.physical_signal(physical, times, exposures, epoch=epoch,
                                      exposure_nodes=exposure_nodes)
        in_transit = signal > np.max(signal)*1e-8 if np.max(signal) > 0 else np.zeros(len(times), bool)
        events = np.unique(np.rint((times[in_transit]-epoch)/period).astype(int)).size
        signal_norm = weighted_signal_norm(signal, errors)
        observable = in_transit.sum() >= 5 and events >= 2 and signal_norm > 0
        if observable or not conditional:
            break
    else:
        raise RuntimeError('Predeclared observability conditioning failed after 10000 proposals: '+name)
    scale = signal_norm/snr if signal_norm > 0 else 1e-4
    dy = scale*errors
    white = dy*rng.normal(size=len(times))
    ou_amplitude = .25*np.median(dy)
    ou_tau = 1. if settings['cadence'] == 'ztf' else .15
    correlated = ou_noise(times, ou_amplitude, ou_tau, rng)
    y = 1.+white+correlated-(0 if null or stress.get('null') else signal)
    if stress.get('flat'):
        y = np.ones_like(y)
    # The public default uses a positive relative origin; supplying both
    # methods this same input makes that dispatch idempotent. Development
    # scouts preserve large epochs to stress literal native preprocessing.
    absolute_epoch = stress.get('absolute_epoch', 1. if stream.startswith('heldout:') else 2457000.)
    t = times+absolute_epoch
    epoch += absolute_epoch
    if stress.get('cleaning_edges'):
        # Deliberate independently cleaned invalid rows; preserve valid inputs.
        t = np.r_[0., np.nan, -1., t, t[-1]+1., t[-1]+2.]
        y = np.r_[1., 1., 1., y, np.nan, 1.]
        dy = np.r_[dy[0], dy[0], dy[0], dy, dy[0], 0.]
        signal, exposures = (np.pad(x, (3, 2)) for x in (signal, exposures))
        band = np.pad(band, (3, 2))
    period_bounds = (.6, 12.878375495285127 if settings['cadence'] == 'tess_200s' else
                     10. if settings['cadence'] == 'ztf' else 27.457888046800917)
    if full_grid:
        periods = np.sort(ref.period_grid(float(np.ptp(times)), R_star=radius, M_star=mass,
                          period_min=period_bounds[0], period_max=period_bounds[1]))
    else:
        periods = selected_periods(period, duration, float(np.ptp(times)))
    if len(periods) < 60:
        raise ValueError('Native GTLS period grid too short for valid reference grouping')
    solar = diag.Regime('solar_comparison', period)
    solar_duration = diag.durations(solar)[0]
    metadata = dict(name=name, regime=regime_name, cohort='heldout' if stream.startswith('heldout:') else 'development',
        purpose='recovery_full_grid' if full_grid else 'mathematical_differential', stream=stream,
        null=bool(null or stress.get('null') or stress.get('flat')), physical=asdict(physical),
        truth_period=period, truth_epoch=epoch, duration_days=duration, full_duration_days=full_duration,
        semimajor_stellar_radii=semimajor, periastron_stellar_radii=semimajor*(1-eccentricity),
        latent_white_oracle_snr=float(snr), realized_latent_white_oracle_snr=signal_norm/scale,
        conditional_population=conditional, accepted_proposal=attempt, observable=bool(observable),
        in_transit_observations=int(in_transit.sum()), observed_events=int(events),
        noise=dict(white='independent Gaussian with supplied dy', ou_amplitude=float(ou_amplitude),
                   ou_tau_days=ou_tau, oracle_snr_excludes_ou=True),
        duration_over_solar_central=duration/solar_duration,
        below_approx_native_duration_envelope=duration/solar_duration < .135,
        expected_native_cache_failure=stress.get('expected_native_cache_failure', False),
        cadence=settings['cadence'], cadence_sha256=sha(path), ndata=len(t), baseline_days=float(np.ptp(times)),
        cadence_construction='Synthetic repeated observed TESS campaign blocks' if stress.get('repeated_campaigns') else 'Observed cadence',
        fractional_duration=duration/period, duration_times_ndata=duration/period*len(t),
        input_time_offset_from_cadence_days=absolute_epoch,
        input_time_origin_policy='Common positive relative origin for both APIs' if stream.startswith('heldout:') else
                                 'Large absolute epoch for literal native preprocessing development stress',
        period_count=len(periods), period_bounds=period_bounds if full_grid else [float(periods[0]), float(periods[-1])],
        truth_inserted_in_grid=not full_grid,
        grid_execution='Explicit identical period array supplied to both engines; generated on CPU from pinned GTLS grid arithmetic.' if full_grid else
                       'Explicit selected-period mathematical stress grid including truth and aliases.',
        auto_grid_api_exercised=False,
        exposure_quadrature_nodes=exposure_nodes,
        search_kwargs=dict(R_star=radius, M_star=mass, oversampling_factor=3,
                           period_min=period_bounds[0], period_max=period_bounds[1]),
        stress_settings=stress)
    return dict(t=t, y=y, dy=dy, periods=periods, signal=signal, exposure_days=exposures, band=band), metadata


def case_specs(suite, plan):
    if suite == 'stress':
        return [(s['name'], s['regime'], 10., bool(s.get('null')), False, s) for s in STRESS]
    specs = []
    for regime in REGIMES:
        counts = plan.get(regime, {}) if suite == 'heldout' else {'8': 1, '10': 1, 'null': 1}
        for snr in SNRS:
            for i in range(int(counts.get(str(int(snr)), 0))):
                specs.append(('%s_snr%d_%04d' % (regime, snr, i), regime, snr, False, True, {}))
        for i in range(int(counts.get('null', 0))):
            specs.append(('%s_null_%04d' % (regime, i), regime, SNRS[i % len(SNRS)], True, True, {}))
    return specs


def verify_seal(seal, identity, plan, stream):
    if not stream.startswith('heldout:'):
        raise ValueError('Independent stream must explicitly start heldout:')
    if seal.get('source_identity') != identity or seal.get('plan') != plan or seal.get('stream') != stream:
        raise ValueError('Seal does not match current sources, frozen plan and independent stream')
    required = ('gates', 'modes', 'thresholds', 'reference_commit', 'freeze_timestamp_utc', 'hardware',
                'engine_kind', 'chunk_policy', 'auto_grid', 'options', 'reference_package_sources')
    if any(key not in seal for key in required):
        raise ValueError('Seal lacks predeclared numeric gates, mode, thresholds, reference, date or hardware')


def replay_manifest(repo_root, manifest_path, output):
    """Recreate existing numerical inputs, with no new independent-data claim."""
    manifest = json.loads(manifest_path.read_text())
    if output.exists():
        raise ValueError('Refuse to overwrite input directory')
    diagnostic = module(repo_root/'benchmarks/tls_accuracy/diagnose.py', 'physical_diagnostic')
    reference = module(repo_root/'cuvarbase/tls_reference_math.py', 'fixture_reference_math')
    cadence_root = repo_root/'benchmarks/results/tls_sensitivity_2026-09-09/cadences'
    frozen_sources = manifest.get('source_identity', {})
    for name, expected in frozen_sources.get('cadences', {}).items():
        if sha(cadence_root/name) != expected:
            raise ValueError('Cadence source hash changed: '+name)
    if frozen_sources.get('diagnostic_sha256') != sha(repo_root/'benchmarks/tls_accuracy/diagnose.py'):
        raise ValueError('Physical signal generator differs from the published study')
    output.mkdir(parents=True)
    receipts = []
    for case in manifest['cases']:
        metadata = case['metadata']
        arrays, generated = build_case(metadata['name'], metadata['regime'], metadata['latent_white_oracle_snr'],
            metadata['null'], metadata['stream'], diagnostic, reference, cadence_root,
            metadata['purpose'] == 'recovery_full_grid', stress=metadata.get('stress_settings'),
            exposure_nodes=metadata['exposure_quadrature_nodes'], conditional=metadata['conditional_population'])
        actual = {key: array_hash(value) for key, value in arrays.items()}
        if actual != case['arrays']:
            changed = [key for key in set(actual)|set(case['arrays']) if actual.get(key) != case['arrays'].get(key)]
            raise ValueError('Numerical input differs for %s: %s. Use the published generation dependencies.' % (metadata['name'], changed))
        path = output/case['file']
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, sort_keys=True))
        receipts.append(dict(name=metadata['name'], numerical_arrays_equal=True,
            original_npz_sha256=case['sha256'], regenerated_npz_sha256=sha(path)))
        write(output/'reproduction.json', dict(original_manifest_sha256=sha(manifest_path),
            generator_sha256=sha(__file__), cases=receipts,
            interpretation='Reproduced frozen numerical inputs, not newly independent observations'))
    # Preserve the original seal/manifest; compression versions can change
    # container bytes even when all numerical values match exactly.
    (output/'original_manifest.json').write_bytes(manifest_path.read_bytes())
    reproduced = dict(manifest, suite='reproduction', original_manifest_sha256=sha(manifest_path))
    reproduced['cases'] = [dict(case, original_npz_sha256=case['sha256'],
        sha256=sha(output/case['file'])) for case in manifest['cases']]
    write(output/'manifest.json', reproduced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay-manifest', type=Path, help='Regenerate published inputs using their frozen stream and metadata, verifying every numerical array hash.')
    parser.add_argument('--suite', choices=('stress', 'full-dev', 'heldout'), default='stress')
    parser.add_argument('--select', action='append', help='Optional explicit development case names; never filters heldout.')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--seal', type=Path)
    parser.add_argument('--stream', default='development:20260910-v1')
    args = parser.parse_args()
    if args.replay_manifest:
        replay_manifest(args.repo_root, args.replay_manifest, args.out)
        return
    identity = source_identity(args.repo_root)
    plan = json.loads(args.plan.read_text()) if args.plan else {}
    if args.suite == 'heldout':
        if not args.seal or not args.plan or args.select:
            parser.error('Heldout requires --seal, --plan, and no case filtering')
        verify_seal(json.loads(args.seal.read_text()), identity, plan, args.stream)
    elif args.stream.startswith('heldout:'):
        parser.error('A heldout stream may only be used with the sealed heldout suite')
    if args.out.exists():
        parser.error('Output already exists; generation is immutable')
    specs = case_specs(args.suite, plan)
    if args.select:
        selected = set(args.select)
        missing = selected-{s[0] for s in specs}
        if missing:
            parser.error('Unknown selected cases: '+str(sorted(missing)))
        specs = [s for s in specs if s[0] in selected]
    diag = module(args.repo_root/'benchmarks/tls_accuracy/diagnose.py', 'physical_diagnostic')
    ref = module(args.repo_root/'cuvarbase/tls_reference_math.py', 'fixture_reference_math')
    cadence_root = args.repo_root/'benchmarks/results/tls_sensitivity_2026-09-09/cadences'
    args.out.mkdir(parents=True)
    manifest = dict(suite=args.suite, stream=args.stream, source_identity=identity,
                    seal_sha256=sha(args.seal) if args.seal else None, cases=[])
    for name, regime, snr, null, full_grid, stress in specs:
        arrays, metadata = build_case(name, regime, snr, null, args.stream, diag, ref,
                                     cadence_root, full_grid, stress=stress, conditional=full_grid)
        metadata['seal_sha256'] = sha(args.seal) if args.seal else None
        path = args.out/(name+'.npz')
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, sort_keys=True))
        manifest['cases'].append(dict(file=path.name, sha256=sha(path), metadata=metadata,
                                     arrays={key: array_hash(value) for key, value in arrays.items()}))
        write(args.out/'manifest.json', manifest)
        print(json.dumps(dict(case=name, ndata=metadata['ndata'], nperiods=metadata['period_count'],
                              observable=metadata['observable'], purpose=metadata['purpose'])), flush=True)
    if identity != source_identity(args.repo_root):
        manifest['status'] = 'source_changed_during_generation'
        manifest['source_identity_after'] = source_identity(args.repo_root)
        write(args.out/'manifest.json', manifest)
        if args.suite == 'heldout':
            raise RuntimeError('Source changed during heldout generation; seal invalid')
        print('Development source tree changed during generation; recorded before/after hashes.', file=sys.stderr)
        return
    manifest['status'] = 'complete'
    write(args.out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
