#!/usr/bin/env python3
"""Audit complete timing receipts before calculating any speed ratio.

The fastest tested native pool is eligible only when every declared repetition
returns the original one-worker period/power/chi2/primary/SDE hashes. Failed
or missing measurements remain visible and never become speed denominators.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

if __package__:
    from .benchmark import consistency, ownership_valid
    from .common import BATCH_REPETITIONS, SINGLE_REPETITIONS, write
else:
    from benchmark import consistency, ownership_valid
    from common import BATCH_REPETITIONS, SINGLE_REPETITIONS, write


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    if not values or any(value is None or not math.isfinite(value) or value <= 0 for value in values):
        return None
    return dict(repetitions=len(values), raw_seconds=values,
                median_seconds=statistics.median(values),
                minimum_seconds=min(values), maximum_seconds=max(values))


def gate(record, kind, names, reference=None, field='strict'):
    if record is None:
        return dict(eligible=False, problems=[dict(reason='Missing configuration')], reference={})
    result = consistency(record.get(kind, []), reference=reference, expected_names=names,
                         expected_repetitions=(SINGLE_REPETITIONS if kind == 'single'
                                               else BATCH_REPETITIONS), field=field)
    if record.get('status') != 'ok':
        result['eligible'] = False
        result['problems'].append(dict(reason='Configuration did not complete'))
    if not ownership_valid(record):
        result['eligible'] = False
        result['problems'].append(dict(reason='GPU process birth, call ownership, or exit proof failed'))
    return result


def public_times(record, kind):
    return distribution([rep.get('denominator_seconds') for rep in record.get(kind, [])])


def ratio(native, candidate):
    return (None if native is None or candidate is None else
            native['median_seconds'] / candidate['median_seconds'])


def compare_public(records, names, single_name, widths, frozen=None):
    native_one = records.get('gtls_graph_1worker')
    candidate = records.get('candidate_graph_1worker')
    one_gate = gate(native_one, 'batch', names)
    candidate_gate = gate(candidate, 'batch', names)
    common_reference = gate(native_one, 'batch', names, field='common')
    common_gate = gate(candidate, 'batch', names,
                       reference=common_reference['reference'], field='common')
    candidate_times = public_times(candidate, 'batch') if candidate is not None else None
    frozen_gates = {}
    if frozen is not None:
        for backend, record in (('gtls', native_one), ('candidate', candidate)):
            reference = {name: value['strict'] for name, value in frozen.get(backend, {}).items()}
            frozen_gates[backend] = gate(record, 'batch', names, reference=reference)
    # Scientific equivalence is established by the separate validation study.
    # A documented fix for undefined native trials can legitimately change
    # native-vs-candidate hashes. Timing still requires each implementation's
    # frozen expected outputs, and every pool must reproduce literal native1.
    required = ((one_gate, candidate_gate, *frozen_gates.values()) if frozen is not None else
                (one_gate, candidate_gate, common_reference, common_gate))
    base_eligible = all(item['eligible'] for item in required)
    pools = {}
    for width in widths:
        record = records.get(f'gtls_graph_{width}worker')
        pool_gate = gate(record, 'batch', names, reference=one_gate['reference'])
        measured = public_times(record, 'batch') if record is not None else None
        pools[str(width)] = dict(eligible=base_eligible and pool_gate['eligible'] and measured is not None,
                                output_gate=pool_gate, elapsed=measured,
                                startup_seconds=None if record is None else record.get('pool_startup_seconds'))
    eligible = [(int(width), item) for width, item in pools.items() if item['eligible']]
    fastest = min(eligible, key=lambda pair: (pair[1]['elapsed']['median_seconds'], pair[0])) if eligible else None
    batch = dict(source_count=len(names), candidate_output_gate=candidate_gate,
                 common_output_gate=common_gate, native_one_worker_gate=one_gate,
                 frozen_output_gates=frozen_gates,
                 native_pool_configurations=pools,
                 candidate=candidate_times,
                 strongest_tested_native_workers=None if fastest is None else fastest[0],
                 strongest_tested_native=None if fastest is None else fastest[1]['elapsed'],
                 speedup=None if fastest is None else ratio(fastest[1]['elapsed'], candidate_times),
                 eligible=fastest is not None and candidate_times is not None)

    single = dict(eligible=False, speedup=None, case=single_name)
    if single_name is not None:
        native_single_gate = gate(native_one, 'single', [single_name])
        candidate_single_gate = gate(candidate, 'single', [single_name])
        native_common = gate(native_one, 'single', [single_name], field='common')
        single_common = gate(candidate, 'single', [single_name], reference=native_common['reference'], field='common')
        # The same source must retain its full search when dispatched in a batch.
        native_cross = gate(native_one, 'single', [single_name], reference={single_name: one_gate['reference'].get(single_name)})
        candidate_cross = gate(candidate, 'single', [single_name], reference={single_name: candidate_gate['reference'].get(single_name)})
        native_time = public_times(native_one, 'single') if native_one is not None else None
        candidate_time = public_times(candidate, 'single') if candidate is not None else None
        single_frozen = {}
        if frozen is not None:
            for backend, record in (('gtls', native_one), ('candidate', candidate)):
                reference = {single_name: frozen.get(backend, {}).get(single_name, {}).get('strict')}
                single_frozen[backend] = gate(record, 'single', [single_name], reference=reference)
        required = ((native_single_gate, candidate_single_gate, native_cross, candidate_cross,
                     *single_frozen.values()) if frozen is not None else
                    (native_single_gate, candidate_single_gate, native_common, single_common,
                     native_cross, candidate_cross))
        valid = all(item['eligible'] for item in required)
        single.update(eligible=valid and native_time is not None and candidate_time is not None,
                      native=native_time, candidate=candidate_time,
                      native_output_gate=native_single_gate, candidate_output_gate=candidate_single_gate,
                      common_output_gate=single_common,
                      frozen_output_gates=single_frozen,
                      native_single_batch_gate=native_cross, candidate_single_batch_gate=candidate_cross,
                      speedup=ratio(native_time, candidate_time) if valid else None)

    row = records.get('candidate_row_1worker')
    ab = None
    if row is not None and single_name is not None:
        graph_gate = gate(candidate, 'single', [single_name], field='full_digest')
        row_gate = gate(row, 'single', [single_name], reference=graph_gate['reference'], field='full_digest')
        row_time = public_times(row, 'single')
        graph_time = public_times(candidate, 'single') if candidate is not None else None
        valid = graph_gate['eligible'] and row_gate['eligible'] and row_time is not None and graph_time is not None
        ab = dict(eligible=valid, output_gate=row_gate, graph=graph_time, original_row_prefix=row_time,
                  public_call_speedup=ratio(row_time, graph_time) if valid else None,
                  scope='Separate same-code full-public-call attribution; only prefix dispatch changed at runtime')
    corrected = None
    corrected_record = records.get('gtls_corrected_graph_1worker')
    if corrected_record is not None or (frozen is not None and 'gtls_corrected' in frozen):
        corrected = dict(scope='Conditional corrected-native cross-check on the same cohort; excluded from literal strongest-pool selection')
        expected = {name: value['strict'] for name, value in (frozen or {}).get('gtls_corrected', {}).items()}
        for kind, selected in (('batch', names), ('single', [single_name])):
            if kind == 'single' and single_name is None:
                corrected[kind] = dict(eligible=False, speedup=None, reason='No paired single case')
                continue
            expected_kind = {name: expected.get(name) for name in selected}
            own = gate(corrected_record, kind, selected, reference=expected_kind)
            candidate_own = gate(candidate, kind, selected, reference={
                name: (frozen or {}).get('candidate', {}).get(name, {}).get('strict') for name in selected})
            corrected_common = gate(corrected_record, kind, selected, field='common')
            common = gate(candidate, kind, selected, reference=corrected_common['reference'], field='common')
            elapsed = public_times(corrected_record, kind) if corrected_record is not None else None
            other = public_times(candidate, kind) if candidate is not None else None
            valid = all(value['eligible'] for value in (own, candidate_own, corrected_common, common))
            corrected[kind] = dict(eligible=valid and elapsed is not None and other is not None,
                source_count=len(selected), own_frozen_output_gate=own,
                candidate_frozen_output_gate=candidate_own, common_output_gate=common,
                corrected_native=elapsed, candidate=other,
                speedup=ratio(elapsed, other) if valid else None)
    return dict(public_single=single, public_batch=batch, prefix_ab=ab,
                corrected_native_crosscheck=corrected)


def compare_single_public(records, single_name, frozen):
    """Five full public calls; no batch denominator or worker-pool selection."""
    if single_name is None:
        raise ValueError('Single-source timing has no paired successful input')
    names = [single_name]
    native = records.get('gtls_graph_1worker')
    candidate = records.get('candidate_graph_1worker')
    checks, measured, full_gates = {}, {}, {}
    for backend, record in (('gtls', native), ('candidate', candidate)):
        expected = {single_name: frozen.get(backend, {}).get(single_name, {}).get('strict')}
        checks[backend] = gate(record, 'single', names, reference=expected)
        if record is not None and (record.get('batch') or record.get('pool_width') != 1):
            checks[backend]['eligible'] = False
            checks[backend]['problems'].append(dict(reason='Single scope cannot contain batch or multiworker measurements'))
        measured[backend] = public_times(record, 'single') if record is not None else None
        full_gates[backend] = gate(record, 'single', names, field='full_digest')
    native_common = gate(native, 'single', names, field='common')
    common = gate(candidate, 'single', names, reference=native_common['reference'], field='common')
    valid = all(value['eligible'] for value in (*checks.values(), *full_gates.values())) and all(measured.values())
    single = dict(eligible=bool(valid), case=single_name, source_count=1,
        native=measured['gtls'], candidate=measured['candidate'],
        frozen_output_gates=checks, full_repeatability_gates=full_gates, common_output_gate=common,
        speedup=ratio(measured['gtls'], measured['candidate']) if valid else None,
        scope='Five full public calls in one persistent worker per method; no TLS batch throughput or strongest-pool comparison')
    corrected = None
    if 'gtls_corrected' in frozen or 'gtls_corrected_graph_1worker' in records:
        record = records.get('gtls_corrected_graph_1worker')
        own = gate(record, 'single', names, reference={single_name:
            frozen.get('gtls_corrected', {}).get(single_name, {}).get('strict')})
        if record is not None and (record.get('batch') or record.get('pool_width') != 1):
            own['eligible'] = False
            own['problems'].append(dict(reason='Single corrected scope contains batch or multiworker measurements'))
        corrected_common = gate(record, 'single', names, field='common')
        paired = gate(candidate, 'single', names, reference=corrected_common['reference'], field='common')
        elapsed = public_times(record, 'single') if record is not None else None
        corrected_full = gate(record, 'single', names, field='full_digest')
        ok = all(value['eligible'] for value in (own, checks['candidate'], full_gates['candidate'],
                                                corrected_common, corrected_full, paired))
        corrected = dict(scope='Conditional corrected-native single-source cross-check; literal GTLS remains separately reported',
            single=dict(eligible=bool(ok and elapsed and measured['candidate']), source_count=1,
                corrected_native=elapsed, candidate=measured['candidate'],
                own_frozen_output_gate=own, full_repeatability_gate=corrected_full, common_output_gate=paired,
                speedup=ratio(elapsed, measured['candidate']) if ok else None),
            batch=dict(eligible=False, status='not_measured'))
    return dict(public_single=single,
        public_batch=dict(eligible=False, status='not_measured', source_count=0,
                          scope='Batch protocol deferred before execution'),
        prefix_ab=None, corrected_native_crosscheck=corrected)


def compare_components(native, candidate, single_name, frozen_required=False):
    problems = []
    for backend, record in (('gtls', native), ('candidate', candidate)):
        if record is None or record.get('case', {}).get('name') != single_name:
            problems.append(dict(backend=backend, reason='Missing or different component input'))
            continue
        reps = record.get('repetitions', [])
        if not ownership_valid(record):
            problems.append(dict(backend=backend, reason='Component GPU ownership lifecycle failed'))
        if (record.get('status') != 'ok' or len(reps) != SINGLE_REPETITIONS or
                any(not rep.get('denominator_eligible') or not rep.get('endpoint_valid') or
                    not rep.get('output_identical_to_literal') or
                    (rep.get('outputs') or {}).get('full_digest') != (record.get('literal_outputs') or {}).get('full_digest')
                    for rep in reps)):
            problems.append(dict(backend=backend, reason='Incomplete or non-identical component repetitions'))
        if frozen_required and (not record.get('literal_matches_frozen') or
                (record.get('literal_outputs') or {}).get('strict') != (record.get('frozen_outputs') or {}).get('strict')):
            problems.append(dict(backend=backend, reason='Component literal output differs from its own frozen study result'))
    if not problems:
        if not frozen_required and native['literal_outputs']['common'] != candidate['literal_outputs']['common']:
            problems.append(dict(reason='Component runs have different complete common search outputs'))
        if native['case'] != candidate['case']:
            problems.append(dict(reason='Component execution arrays or options differ'))
    if problems:
        return dict(eligible=False, speedup=None, problems=problems)
    stats = {}
    for backend, record in (('gtls', native), ('candidate', candidate)):
        reps = record['repetitions']
        stages = sorted({name for rep in reps for name in rep['inclusive_stage_seconds']})
        stats[backend] = dict(
            common_search=distribution([rep['common_search_seconds'] for rep in reps]),
            public_instrumented=distribution([rep['public_instrumented_seconds'] for rep in reps]),
            after_common_search=distribution([rep['after_common_search_seconds'] for rep in reps]),
            inclusive_stages={name: distribution([rep['inclusive_stage_seconds'].get(name, 0.) for rep in reps])
                              for name in stages})
    valid = all(stats[name]['common_search'] is not None for name in stats)
    return dict(eligible=valid, measurements=stats,
                common_outputs_identical=native['literal_outputs']['common'] == candidate['literal_outputs']['common'],
                speedup=ratio(stats['gtls']['common_search'], stats['candidate']['common_search']) if valid else None,
                scope='Separate instrumented single-source search through final window selection; candidate includes compact winner transfers, while native stops after final GPU argmin and excludes subsequent physical/SNR diagnostics',
                overlap_note='Inclusive stage durations overlap and must not be added together.')


def summarize(root, native_components=None, candidate_components=None, corrected_components=None):
    root = Path(root)
    plan = read(root/'plan.json')
    measurement_scope = plan.get('measurement_scope', 'full')
    if measurement_scope not in ('full', 'single'):
        raise ValueError('Unknown timing measurement scope')
    if measurement_scope == 'single' and (plan['native_pool_widths'] != [1] or
            plan['batch_repetitions'] != 0 or plan['single_repetitions'] != SINGLE_REPETITIONS):
        raise ValueError('Single-source plan contains batch measurements or changed repetitions')
    output = dict(status='audited', measurement_scope=measurement_scope, environment=plan['environment'],
                  scope=plan['scope'], native_extras=plan['native_extras'],
                  cohort_selection=plan['cohort_selection'], regimes={},
                  scientific_scope='Timings require each method to reproduce its own frozen validation output. Numerical-search equivalence and any declared native bug correction must be assessed from the independent validation study; timing does not establish that scientific claim.')
    for regime in plan['regimes']:
        selection = plan['cohort_selection'][regime]
        records = {path.parent.name: read(path) for path in (root/regime).glob('*/record.json')}
        result = (compare_single_public(records, selection['single_case'], plan['frozen_outputs'][regime])
                  if measurement_scope == 'single' else
                  compare_public(records, selection['selected_cases'], selection['single_case'],
                                 plan['native_pool_widths'], frozen=plan['frozen_outputs'][regime]))
        if native_components is not None and candidate_components is not None:
            paths = [Path(folder)/(regime+'.json') for folder in (native_components, candidate_components)]
            component_records = [read(path) if path.exists() else None for path in paths]
            result['common_search_components'] = compare_components(*component_records, selection['single_case'],
                                                                     frozen_required=True)
        if corrected_components is not None and candidate_components is not None and selection['correction_timing']['required']:
            paths = [Path(folder)/(regime+'.json') for folder in (corrected_components, candidate_components)]
            component_records = [read(path) if path.exists() else None for path in paths]
            result['corrected_common_search_components'] = compare_components(*component_records,
                selection['single_case'], frozen_required=True)
            result['corrected_common_search_components']['native_kind'] = 'gtls_corrected'
        output['regimes'][regime] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--components-gtls', type=Path)
    parser.add_argument('--components-candidate', type=Path)
    parser.add_argument('--components-corrected', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.components_gtls) != bool(args.components_candidate):
        parser.error('Provide both component directories or neither')
    if args.components_corrected is not None and args.components_candidate is None:
        parser.error('Corrected component comparison requires candidate components')
    write(args.output, summarize(args.root, args.components_gtls, args.components_candidate, args.components_corrected))


if __name__ == '__main__':
    main()
