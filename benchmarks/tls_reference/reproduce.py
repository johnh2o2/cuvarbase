#!/usr/bin/env python3
"""Reproduce every published TLS comparison with an elapsed-time ceiling."""
import argparse
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import time

import validate as harness
import summarize
from comparison import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seal', type=Path, required=True, help='Original published seal supplies options and gates; it is never rewritten.')
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=8400.)
    args = parser.parse_args()
    if args.out.exists() or args.max_seconds <= 0:
        parser.error('Require a new output directory and a positive elapsed-time limit')
    seal = json.loads(args.seal.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('suite') != 'reproduction' or manifest.get('seal_sha256') != harness.sha(args.seal):
        parser.error('Use cases.py --replay-manifest to regenerate the published numerical inputs first')
    if seal['modes'] != ['full'] or seal['engine_kind'] != 'public' or seal['chunk_policy'] != 'default':
        parser.error('This frozen cohort requires the actual public full default')
    args.out.mkdir(parents=True)
    options_path, gates_path = args.out/'options.json', args.out/'gates.json'
    harness.write(options_path, seal['options'])
    harness.write(gates_path, seal['gates'])
    started = time.perf_counter()
    state = dict(seal_sha256=harness.sha(args.seal), manifest_sha256=harness.sha(args.manifest),
                 started_epoch=time.time(), elapsed_limit_seconds=args.max_seconds, planned=len(manifest['cases']),
                 cases=[], retention='Keep every record and both comparisons; retain full arrays for all nonpassing literal or corrected pairs '
                     'and the first passing injection/null in each regime. Matching discarded arrays retain original hashes.')
    retained_signatures = set()

    def timeout(signum, frame):
        raise TimeoutError('Frozen cohort elapsed-time ceiling reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        for case in manifest['cases']:
            if time.perf_counter()-started > args.max_seconds-180:
                state['status'] = 'incomplete_elapsed_time_ceiling'
                break
            path = args.manifest.parent/case['file']
            if harness.sha(path) != case['sha256']:
                raise ValueError('Frozen input bytes changed: '+str(path))
            name = case['metadata']['name']
            case_root = args.out/name
            row = dict(name=name, statuses={})
            state['cases'].append(row)
            harness.write(args.out/'progress.json', state)
            outcome = run_case(path, args.repo_root, case_root, seal=args.seal,
                options=options_path, gates=gates_path, thresholds=seal['thresholds'], replay=True)
            row.update(outcome)
            signature = (case['metadata']['regime'], case['metadata']['null'])
            keep = not row.get('passed') or not row.get('literal_passed') or signature not in retained_signatures
            if keep and row.get('passed') and row.get('literal_passed'):
                retained_signatures.add(signature)
            elif not keep:
                removed = []
                for backend in ('gtls', 'gtls_corrected', 'candidate'):
                    record = json.loads((case_root/backend/'record.json').read_text())
                    artifact = case_root/backend/record['arrays_file']
                    if harness.sha(artifact) != record['arrays_sha256']:
                        raise ValueError('Array bytes changed before planned retention step')
                    removed.append(dict(file=str(artifact.relative_to(case_root)), sha256=record['arrays_sha256']))
                    artifact.unlink()
                harness.write(case_root/'retention.json', dict(all_checks_passed=True, arrays_removed=removed,
                    reason='Predeclared compact retention; complete output arrays compared before removal and hashes retained'))
            row['full_arrays_retained'] = keep
            row['elapsed_since_start_seconds'] = time.perf_counter()-started
            harness.write(args.out/'progress.json', state)
            print(json.dumps(row), flush=True)
            if row.get('passed') is False and row.get('comparison_status') == 'compared':
                state.update(status='stopped_numerical_or_public_contract_disagreement', first_failure=name)
                break
        else:
            state['status'] = 'complete'
    except TimeoutError as error:
        state.update(status='incomplete_elapsed_time_ceiling', error=str(error))
    except Exception as error:
        state.update(status='stopped_execution_or_protocol_error', error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['elapsed_seconds'] = time.perf_counter()-started
        harness.write(args.out/'progress.json', state)
    rows, tables = [], []
    for reference_backend, comparison_file in (('gtls_corrected', 'compare.json'), ('gtls', 'literal_compare.json')):
        for threshold in seal['thresholds']:
            selected = [summarize.read_case(case, args.out, threshold, reference_backend, comparison_file)
                        for case in manifest['cases']]
            rows.extend(selected)
            tables.extend([dict(value, threshold=threshold) for value in
                           summarize.strata(selected, alpha=.05/(2*len(seal['thresholds'])))])
    harness.write(args.out/'summary.json', dict(state=state, strata=tables, cases=rows,
        classification='Reproduction, not newly independent validation', primary_endpoint='Every complete spectrum, refinement and selected period against the corrected native host; '
                         'actual public API contract; literal native outcomes retained separately.',
        warning='Small per-regime recovery samples are a cross-check, not a tight population-level sensitivity guarantee. '
                'The masked-candidate host correction can change literal GTLS thresholds and outcomes.'))
    accounted = len(state['cases']) == len(manifest['cases']) and all(
        set(r['statuses']) == {'gtls', 'gtls_corrected', 'candidate'} for r in state['cases'])
    unresolved = [r['name'] for r in state['cases'] if
        r.get('comparison_status') == 'compared' and r.get('passed') is not True]
    candidate_regressions = [r['name'] for r in state['cases'] if
        r['statuses'].get('gtls_corrected') == 'ok' and r['statuses'].get('candidate') != 'ok']
    paired = sum(r.get('passed') is True for r in state['cases'])
    failures = [{k:v for k,v in r.items() if k in ('name','statuses','comparison_status')}
                for r in state['cases'] if any(value != 'ok' for value in r['statuses'].values())]
    gate = (accounted and state.get('status') == 'complete' and not unresolved and
            not candidate_regressions and paired == len(manifest['cases']))
    harness.write(args.out/'acceptance.json', dict(schema_version=1,
        reproduction_gate=dict(pass_=gate), inputs_manifest_sha256=harness.sha(args.manifest),
        seal_sha256=harness.sha(args.seal), original_source_identity=seal['source_identity'],
        reproduction_sources=dict(production=harness.production_sources(args.repo_root),
            tools={p.name:harness.sha(p) for p in Path(__file__).parent.glob('*.py')}),
        reference_package_sources=seal['reference_package_sources'],
        counts=dict(planned=len(manifest['cases']), accounted=len(state['cases']), numerical_pairs_passed=paired,
                    literal_exact_pairs=sum(r.get('literal_passed') is True for r in state['cases']),
                    corrected_reference_reused=sum(r.get('corrected_reference_reused') is True for r in state['cases'])),
        all_planned_accounted=accounted, unresolved_numerical_cases=unresolved,
        candidate_regressions=candidate_regressions, execution_failures=failures,
        limits=['Finite tested input population, not all possible inputs',
                'Exact equality refers to the separately disclosed native host correction',
                'The small recovery/null sample does not establish a 2 percentage-point population margin',
                'SDE8 is descriptive, not a calibrated fixed false-positive rate',
                'Native unsupported cases are explicit and do not count as numerical parity successes']))
    acceptance = json.loads((args.out/'acceptance.json').read_text())
    acceptance['reproduction_gate']['pass'] = acceptance['reproduction_gate'].pop('pass_')
    harness.write(args.out/'acceptance.json', acceptance)



if __name__ == '__main__':
    main()
