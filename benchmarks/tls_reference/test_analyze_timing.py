"""Accounting gates for the public figure, using fictitious timing receipts."""
import copy

import pytest

from analyze_timing import PROFILES, analyze, timing_row


def measured(values):
    return dict(raw_seconds=values, median_seconds=sorted(values)[len(values)//2])


def receipts():
    manifest = dict(source_identity=dict(production_sources={'engine': 'frozen'}), cases=[])
    checks = dict(regimes={}, cohort_selection={}, environment={}, native_extras=[],
                  scientific_scope='Synthetic unit-test receipt; no benchmark measurement.')
    for regime in PROFILES:
        names = [f'{regime}_{i}.npz' for i in range(16)]
        for name in names:
            manifest['cases'].append(dict(file=name, sha256=name+'hash', arrays={'periods': 'gridhash'},
                metadata=dict(regime=regime, ndata=100, baseline_days=20, period_count=200,
                              search_kwargs={'full': True})))
        checks['cohort_selection'][regime] = dict(selected_cases=names, single_case=names[0],
            actual_batch_size=16, expected_candidate_sources={'engine': 'frozen'},
            examined=[dict(case=name, input_sha256=name+'hash') for name in names])
        checks['regimes'][regime] = dict(
            public_single=dict(eligible=True, case=names[0], speedup=2,
                               native=measured([2]*5), candidate=measured([1]*5)),
            public_batch=dict(eligible=True, source_count=16, speedup=2,
                strongest_tested_native_workers=2, candidate=measured([8]*3),
                strongest_tested_native=measured([16]*3),
                native_pool_configurations={
                    '1': dict(eligible=True, elapsed=measured([32]*3)),
                    '2': dict(eligible=True, elapsed=measured([16]*3)),
                    '4': dict(eligible=False, elapsed=measured([4]*3))}),
            common_search_components=dict(eligible=True))
    acceptance = dict(publication_gate={'pass': True}, inputs_manifest_sha256='manifesthash')
    return checks, manifest, acceptance


def test_figure_uses_all_sources_and_fastest_valid_pool():
    result = analyze(*receipts(), 'manifesthash')
    row = next(row for row in result['timings']
               if row['method'] == 'gtls' and row['mode'] == 'batch')
    assert row['workers'] == 2  # Four workers are faster but changed results.
    assert row['seconds_per_source'] == 1
    assert row['n'] == 16


def single_receipts():
    checks, manifest, acceptance = receipts()
    checks['measurement_scope'] = 'single'
    for result in checks['regimes'].values():
        result['public_batch'] = dict(eligible=False, status='not_measured', source_count=0)
    return checks, manifest, acceptance


def test_single_normalization_emits_only_measured_latency():
    result = analyze(*single_receipts(), 'manifesthash')
    assert result['measurement_scope'] == 'single'
    assert len(result['timings']) == 6
    assert all(row['mode'] == 'single' and row['n'] == 1 and row['workers'] == 1 for row in result['timings'])
    assert all(row['batch_size'] is None for row in result['profiles'])
    assert all(row['batch'] is None and row['gtls_batch_workers'] is None for row in result['speedups'])


def merged_receipts():
    checks, manifest, acceptance = single_receipts()
    manifest['studies'] = {'main': dict(manifest_sha256='original', seal_sha256='seal')}
    origin = dict(manifest_sha256='original', seal_sha256='seal', numerical_validation_passed=True,
                  evidence_kind='independent', production_sources={'engine': 'frozen'})
    merged = dict(numerical_validation_passed=True, merged_manifest_sha256='manifesthash',
        accepted_studies={'main': origin}, evidence_kind='independent', publication_gate_passed=True)
    acceptance = dict(merged_origin_checks=merged)
    for selection in checks['cohort_selection'].values():
        selection['accepted_study'] = copy.deepcopy(merged)
    return checks, manifest, acceptance


def test_merged_single_evidence_retains_each_accepted_origin():
    result = analyze(*merged_receipts(), 'manifesthash')
    assert result['verification']['numerical_evidence_kind'] == 'independent'
    assert len(result['timings']) == 6


@pytest.mark.parametrize('change', ('missing_origin', 'unaccepted', 'changed_source', 'stale_selection'))
def test_merged_single_evidence_cannot_bypass_source_gates(change):
    checks, manifest, acceptance = merged_receipts()
    merged = acceptance['merged_origin_checks']
    if change == 'missing_origin':
        merged['accepted_studies'] = {}
    elif change == 'unaccepted':
        merged['numerical_validation_passed'] = False
    elif change == 'changed_source':
        merged['accepted_studies']['main']['production_sources'] = {'engine': 'changed'}
    else:
        checks['cohort_selection']['tess_solar']['accepted_study']['merged_manifest_sha256'] = 'other'
    with pytest.raises(ValueError):
        analyze(checks, manifest, acceptance, 'manifesthash')


@pytest.mark.parametrize('change', ('batch_claim', 'missing_single', 'wrong_source', 'missing_component', 'missing_correction'))
def test_single_normalization_rejects_missing_or_invented_measurements(change):
    checks, manifest, acceptance = single_receipts()
    result = checks['regimes']['tess_solar']
    if change == 'batch_claim':
        result['public_batch']['eligible'] = True
    elif change == 'missing_single':
        result['public_single']['native']['raw_seconds'].pop()
    elif change == 'wrong_source':
        result['public_single']['case'] = 'another'
    elif change == 'missing_component':
        result['common_search_components']['eligible'] = False
    else:
        checks['cohort_selection']['tess_solar']['correction_timing'] = dict(required=True)
    with pytest.raises(ValueError):
        analyze(checks, manifest, acceptance, 'manifesthash')


@pytest.mark.parametrize('change', ['input', 'sources', 'count', 'invalid_pool', 'slower_pool', 'component'])
def test_incomplete_or_mismatched_receipts_cannot_publish(change):
    checks, manifest, acceptance = receipts()
    selection = checks['cohort_selection']['tess_solar']
    result = checks['regimes']['tess_solar']
    if change == 'input':
        selection['examined'][0]['input_sha256'] = 'different'
    elif change == 'sources':
        selection['expected_candidate_sources']['engine'] = 'different'
    elif change == 'count':
        result['public_batch']['source_count'] = 15
    elif change == 'invalid_pool':
        result['public_batch']['strongest_tested_native_workers'] = 4
    elif change == 'slower_pool':
        result['public_batch']['strongest_tested_native_workers'] = 1
    else:
        result['common_search_components']['eligible'] = False
    with pytest.raises(ValueError):
        analyze(checks, manifest, acceptance, 'manifesthash')


def test_scientific_acceptance_is_separate_from_timing_success():
    checks, manifest, acceptance = receipts()
    rejected = copy.deepcopy(acceptance)
    rejected['publication_gate']['pass'] = False
    with pytest.raises(ValueError, match='validation gate'):
        analyze(checks, manifest, rejected, 'manifesthash')
    with pytest.raises(ValueError, match='different input manifest'):
        analyze(checks, manifest, acceptance, 'anothermanifest')


def reproduced_receipts():
    checks, manifest, acceptance = receipts()
    acceptance.pop('publication_gate')
    acceptance.update(reproduction_gate={'pass': True},
        original_source_identity=copy.deepcopy(manifest['source_identity']),
        reproduction_sources=dict(production=copy.deepcopy(manifest['source_identity']['production_sources'])))
    for selection in checks['cohort_selection'].values():
        selection['accepted_study'] = dict(evidence_kind='reproduction',
            numerical_validation_passed=True, publication_gate_passed=False)
    return checks, manifest, acceptance


def test_reproduced_timings_keep_their_evidence_label():
    result = analyze(*reproduced_receipts(), 'manifesthash')
    assert result['verification']['numerical_evidence_kind'] == 'reproduction'
    assert result['verification']['complete']
    assert len(result['timings']) == 12
    assert analyze(*receipts(), 'manifesthash')['verification']['numerical_evidence_kind'] == 'independent'


@pytest.mark.parametrize('change', ('failed', 'both_gates', 'source', 'selection', 'promotion'))
def test_reproduced_evidence_cannot_bypass_or_relabel_gates(change):
    checks, manifest, acceptance = reproduced_receipts()
    if change == 'failed':
        acceptance['reproduction_gate']['pass'] = False
    elif change == 'both_gates':
        acceptance['publication_gate'] = {'pass': True}
    elif change == 'source':
        acceptance['reproduction_sources']['production'] = {'engine': 'changed'}
    elif change == 'selection':
        checks['cohort_selection']['tess_solar']['accepted_study']['publication_gate_passed'] = True
    else:
        acceptance['publication_gate'] = acceptance.pop('reproduction_gate')
    with pytest.raises(ValueError):
        analyze(checks, manifest, acceptance, 'manifesthash')


@pytest.mark.parametrize('values', [[], [1]*4, [0]*5, [float('nan')]*5, [float('inf')]*5])
def test_failed_or_incomplete_repetitions_are_not_speed_denominators(values):
    with pytest.raises(ValueError):
        timing_row('tess_200s', 'gtls', 'single', dict(raw_seconds=values), 1, 1)
