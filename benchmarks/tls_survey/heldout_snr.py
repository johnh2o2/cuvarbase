#!/usr/bin/env python3
"""Apply the frozen development filter diagnostic to every held-out injection.

This operational wrapper preserves the development-only guard in development.py.
It imports that module's existing numerical definitions without modifying them.
"""
import argparse
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

from common import ROOT, load_case, module, now, sha, source_identity, write
from development import optimal_box, optimal_native_family, weighted_snr, ou_filter_snr


def diagnostic_row(a, m, cache):
    """Identical row definitions to the frozen development.py main loop."""
    phase = (a['t'] - m['truth_epoch'] + .5 * m['truth_period']) % m['truth_period'] - .5 * m['truth_period']
    bs, bg = optimal_box(phase, a['signal'], a['dy'])
    ts, tg, winner = optimal_native_family(a['t'], m['truth_period'], a['signal'], a['dy'], cache)
    oracle = weighted_snr(a['signal'], a['signal'], a['dy'])
    red_kwargs = dict(amplitude=m['noise']['ou_amplitude'], tau=m['noise']['ou_tau_days'])
    br = ou_filter_snr(a['t'], a['signal'], bg, a['dy'], **red_kwargs)
    tr = ou_filter_snr(a['t'], a['signal'], tg, a['dy'], **red_kwargs)
    return dict(name=m['name'], regime=m['regime'], ndata=m['ndata'], period=m['truth_period'],
        q=m['fractional_duration'], impact=m['physical']['impact'], eccentricity=m['physical']['eccentricity'],
        stellar_density_solar=m['stellar_density_solar'], observed_events=m['observed_events'],
        in_transit_observations=m['in_transit_observations'], white_oracle_snr=oracle,
        native_family_white_snr=ts, ideal_box_white_snr=bs,
        native_family_ou_snr=tr, ideal_box_ou_snr=br,
        native_white_advantage=ts/bs-1 if bs > 0 else None,
        native_ou_advantage=tr/br-1 if br > 0 else None,
        native_cache_winner=[int(v) for v in winner] if winner else None,
        template_filter_white_check=weighted_snr(a['signal'], tg, a['dy']),
        box_filter_white_check=weighted_snr(a['signal'], bg, a['dy']))


def execute(args):
    begin = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Require a complete original input manifest')
    def guard():
        if args.command == 'run':
            seal = json.loads(args.seal.read_text()); plan = json.loads(args.plan.read_text())
            if (sha(args.plan) != args.plan_sha256 or plan['seal_sha256'] != sha(args.seal) or
                    manifest['seal_sha256'] != sha(args.seal) or manifest['split'] != 'injections' or
                    source_identity() != seal['source_identity'] or
                    manifest['source_identity'] != seal['source_identity'] or
                    sha(__file__) != plan['heldout_snr_protocol_sha256']):
                raise ValueError('Held-out diagnostic differs from the reviewed sources/plan/inputs')
            if args.manifest.resolve() != (Path(plan['planned_campaign_root']) / 'inputs-injections/manifest.json').resolve():
                raise ValueError('Diagnostic inputs differ from the pre-input planned campaign')
            if manifest['regimes'] != seal['regimes'] or manifest['count_per_regime'] != seal['counts']['injections']:
                raise ValueError('Diagnostic population differs from the science seal')
            for regime in seal['regimes']:
                if sum(entry['metadata']['regime'] == regime for entry in manifest['cases']) != seal['counts']['injections']:
                    raise ValueError('Missing planned diagnostic regime')
        elif manifest['split'] != 'development':
            raise ValueError('Wrapper validation may only consume development inputs')
    guard()
    identity = dict(manifest_sha256=sha(args.manifest), wrapper_sha256=sha(__file__),
                    scientific_diagnostic_sha256=sha(ROOT / 'benchmarks/tls_survey/development.py'),
                    cache_math_sha256=sha(ROOT / 'cuvarbase/tls_reference_math.py'))
    if args.command == 'run':
        identity.update(seal_sha256=sha(args.seal), auxiliary_plan_sha256=sha(args.plan))
    state = json.loads(args.out.read_text()) if args.out.exists() else dict(
        identity=identity, manifest_sha256=identity['manifest_sha256'], split=manifest['split'],
        started_utc=now(), rows=[], attempts=[], interpretation='Frozen known-period GTLS sample-index family and ideal-box filter ceilings, with common fitted-constant white/OU expected SNR; no package-SNR/SDE equivalence, no endpoint or setting selection.')
    if state['identity'] != identity:
        raise ValueError('Cannot resume with changed diagnostic definitions or input identity')
    if args.command == 'run':
        state.update(seal_sha256=identity['seal_sha256'], auxiliary_plan_sha256=identity['auxiliary_plan_sha256'])
    planned = {entry['metadata']['name']: entry['sha256'] for entry in manifest['cases']}
    done = {row['name']: row['input_sha256'] for row in state['rows']}
    if len(planned) != len(manifest['cases']) or len(done) != len(state['rows']) or any(planned.get(k) != v for k, v in done.items()):
        raise ValueError('Duplicate or foreign diagnostic input/output')
    ref = module(ROOT / 'cuvarbase/tls_reference_math.py', 'heldout_snr_cache_math')
    caches = {}
    state.update(status='running', planned_cases=len(planned))
    state['attempts'].append(dict(started_utc=now(), numerical_threads=1))
    write(args.out, state)
    for entry in manifest['cases']:
        if entry['metadata']['name'] in done:
            continue
        started = time.perf_counter()
        a, m = load_case(args.manifest.parent, entry)
        key = (len(a['t']), tuple(a['periods'][[0, -1]]))
        if key not in caches:
            caches[key] = ref.build_cache(a['periods'], len(a['t']))
        row = diagnostic_row(a, m, caches[key])
        row.update(input_sha256=entry['sha256'], elapsed_s=time.perf_counter()-started)
        state['rows'].append(row)
        state['completed_cases'] = len(state['rows'])
        write(args.out, state)
        print(json.dumps(dict(completed=len(state['rows']), count=len(planned))), flush=True)
    if sha(args.manifest) != identity['manifest_sha256']:
        raise ValueError('Input manifest changed during the diagnostic')
    guard()
    state['attempts'][-1].update(completed_utc=now(), elapsed_s=time.perf_counter()-begin)
    state.update(status='complete', completed_utc=now(), total_attempt_seconds=sum(
        attempt.get('elapsed_s', 0.) for attempt in state['attempts']))
    write(args.out, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validation = sub.add_parser('validate')
    run = sub.add_parser('run'); run.add_argument('--seal', type=Path, required=True)
    run.add_argument('--plan', type=Path, required=True); run.add_argument('--plan-sha256', required=True)
    for command in (validation, run):
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
