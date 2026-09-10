"""CPU gates against accidental parity/heldout overclaims."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cases as fixtures
import validate as harness
import summarize


def test_nonfinite_pattern_and_signed_zero_are_reported():
    a = np.array([np.nan, np.inf, -np.inf, 0., 2.], np.float32)
    b = a.copy()
    b[3] = -0.
    result = harness.compare_array(a, b, 0., 0.)
    assert result['passed'] and not result['bitwise']
    b[1] = -np.inf
    assert not harness.compare_array(a, b, 100., 100.)['passed']


def test_native_chunk_local_winner_is_decoded_per_group():
    arrays = {'p_winning_local_flat_index': np.array([3, 11, 9]),
              'p_chunk_width_masks': np.array([[True, False, True], [False, True, False]]),
              'p_widths': np.array([2, 3, 5])}
    harness.normalize_native_winners(arrays, 'p_', 10, 2, np.array([1., 2., 3.]))
    np.testing.assert_array_equal(arrays['p_winning_start'], [3, 1, 9])
    np.testing.assert_array_equal(arrays['p_winning_width'], [2, 5, 3])
    arrays['p_winning_local_flat_index'][0] = 30
    with pytest.raises(ValueError, match='invalid logical winner'):
        harness.normalize_native_winners(arrays, 'p_', 10, 2, np.ones(3))


def fast_record(tmp_path, label, power, index):
    path = tmp_path/label
    path.mkdir()
    values = {'periods': np.array([1., 2., 3.]), 'periods_mask': np.zeros(3, bool),
        'prepared_t': np.array([1., 2., 3.]), 'prepared_y': np.ones(3), 'prepared_dy': np.ones(3),
        'cache_widths': np.array([1], np.int32), 'cache_template_deficits': np.array([[.5]], np.float32),
        'cache_overshoot': np.array([1.], np.float32), 'cache_signal_lengths': np.array([1], np.int32),
        'chi2': np.array([1., 2., 3.], np.float32), 'chi2_mask': np.zeros(3, bool),
        'coarse_chi2': np.array([1., 2., 3.], np.float32), 'coarse_chi2_mask': np.zeros(3, bool),
        'power': np.array(power), 'power_mask': np.zeros(3, bool),
        'stage0_minimum_width': np.ones(3, np.int32), 'stage0_maximum_width': np.ones(3, np.int32),
        'stage0_chunk_width_masks': np.ones((1, 1), bool), 'stage0_winning_start': np.zeros(3, np.int32),
        'stage0_winning_width': np.ones(3, np.int32)}
    for field in ('chi2', 'chi2_mask', 'power', 'power_mask'):
        values['stage0_'+field] = values[field].copy()
    np.savez_compressed(path/'arrays.npz', **values)
    harness.write(path/'record.json', dict(status='ok', input_sha256='same', input_metadata={}, mode='fast',
        arrays_file='arrays.npz', arrays_sha256=harness.sha(path/'arrays.npz'),
        result=dict(primary_index=index, period=float(index+1), score=float(max(power)), stages=[dict(index=0)])))
    return path/'record.json'


def test_numeric_tolerance_does_not_waive_changed_recovery_or_threshold(tmp_path):
    r = fast_record(tmp_path, 'r', [8., 7.999, 1.], 0)
    c = fast_record(tmp_path, 'c', [7.999, 8.0001, 1.], 1)
    gates = tmp_path/'gates.json'
    harness.write(gates, {'power': {'atol': .01}})
    out = tmp_path/'compared.json'
    harness.compare(SimpleNamespace(reference=r, candidate=c, gates=gates, threshold=[8.00005], out=out))
    result = json.loads(out.read_text())
    assert result['numeric_passed']
    assert not result['decisions_passed']
    assert not result['passed']
    assert result['ranking']['changed_within_error_band']
    assert not result['thresholds'][0]['same_decision']


def test_missing_stage_cannot_pass(tmp_path):
    r = fast_record(tmp_path, 'r', [8., 2., 1.], 0)
    c = fast_record(tmp_path, 'c', [8., 2., 1.], 0)
    record = json.loads(r.read_text())
    record['result']['stages'].append(dict(index=1))
    harness.write(r, record)
    out = tmp_path/'compared.json'
    harness.compare(SimpleNamespace(reference=r, candidate=c, gates=None, threshold=[], out=out))
    result = json.loads(out.read_text())
    assert not result['passed']
    assert result['checks']['stage1_chi2']['missing_candidate']


def test_selected_grids_are_large_enough_and_include_truth():
    for period in (2., 10., 365.25):
        periods = fixtures.selected_periods(period, .05, 1500.)
        assert len(periods) >= 96 and np.all(np.diff(periods) > 0)
        assert period in periods and period/2 in periods
    with pytest.raises(ValueError, match='fewer than 60'):
        fixtures.selected_periods(10., .05, 1000., count=21)


def test_stream_separation_and_fixed_case_seed():
    x = fixtures.deterministic_rng('development:v1', 'case').normal(size=20)
    np.testing.assert_array_equal(x, fixtures.deterministic_rng('development:v1', 'case').normal(size=20))
    assert not np.array_equal(x, fixtures.deterministic_rng('heldout:v1', 'case').normal(size=20))


def test_white_snr_normalization_and_duplicate_time_ou():
    signal = np.array([0., .002, .004, .001, 0.])
    errors = np.array([1., 2., 3., 1., 1.])
    scale = fixtures.weighted_signal_norm(signal, errors)/8.
    assert fixtures.weighted_signal_norm(signal, scale*errors) == pytest.approx(8., rel=1e-14)
    times = np.array([0., 0., 1., 1., 1.1])
    noise = fixtures.ou_noise(times, 1., .15, fixtures.deterministic_rng('dev', 'ou'))
    assert noise[0] == noise[1] and noise[2] == noise[3]


def test_heldout_requires_matching_complete_seal():
    identity, plan = {'generator_sha256': 'abc'}, {'tess_solar': {'8': 2}}
    seal = dict(source_identity=identity, plan=plan, stream='heldout:frozen',
                gates={}, modes=['full'], thresholds=[8.], reference_commit='pinned',
                freeze_timestamp_utc='2026-09-10T00:00:00Z', hardware={'gpu': 'A40'},
                engine_kind='public', chunk_policy='default', auto_grid=False, options={}, reference_package_sources={})
    fixtures.verify_seal(seal, identity, plan, 'heldout:frozen')
    for changed in ({'generator_sha256': 'new'}, {}):
        with pytest.raises(ValueError, match='does not match'):
            fixtures.verify_seal(seal, changed, plan, 'heldout:frozen')
    with pytest.raises(ValueError, match='start heldout:'):
        fixtures.verify_seal(seal, identity, plan, 'development:frozen')
    del seal['gates']
    with pytest.raises(ValueError, match='lacks'):
        fixtures.verify_seal(seal, identity, plan, 'heldout:frozen')


def test_full_plans_keep_every_predeclared_case():
    specs = fixtures.case_specs('heldout', {'tess_highimpact': {'8': 3, '10': 2, 'null': 3}})
    assert len(specs) == 8
    assert sum(s[3] for s in specs) == 3
    assert all(s[4] for s in specs)


def test_simultaneous_zero_discordance_bound_is_not_a_tight_margin():
    upper = summarize.discordance_upper(0, 64, .05/24)
    assert upper == pytest.approx(1-(.05/24)**(1/64), abs=1e-14)
    assert upper > .09
    assert summarize.discordance_upper(0, 0, .05) is None
    assert summarize.binomial_interval(0, 10)[0] == 0
    assert summarize.binomial_interval(10, 10)[1] == 1


def test_period_match_uses_baseline_drift_and_alias_separation():
    meta = {'truth_period': 10., 'baseline_days': 1000., 'duration_days': .1}
    assert summarize.recovered(10.0004, meta)
    assert not summarize.recovered(10.0006, meta)
    assert not summarize.recovered(5., meta)
    assert summarize.recovered(5., meta, alias=.5)


def test_public_chi2_must_be_in_original_error_units(tmp_path):
    r = fast_record(tmp_path, 'r', [8., 2., 1.], 0)
    c = fast_record(tmp_path, 'c', [8., 2., 1.], 0)
    for path, public in ((r, False), (c, True)):
        record = json.loads(path.read_text())
        record['result']['error_scale'] = 2.
        with np.load(path.parent/'arrays.npz') as source:
            values = {key: source[key] for key in source.files}
        values['stage0_SR'], values['stage0_SR_mask'] = np.array([1., .5, 1/3]), np.zeros(3, bool)
        if public:
            record['engine_kind'] = 'public'
            record['result']['public_contract'] = dict(period=1., SDE=8., search_configuration=dict(
                method='reference', phase_binning=False, samples_used=3, input_count=3, time_origin=0.))
            for key in ('periods', 'chi2', 'power'):
                values['public_'+key] = values[key].copy()
            values['public_SR'] = values['stage0_SR'].copy()
            values['public_valid_periods'] = np.ones(3, bool)
        np.savez_compressed(path.parent/'arrays.npz', **values)
        record['arrays_sha256'] = harness.sha(path.parent/'arrays.npz')
        harness.write(path, record)
    out = tmp_path/'compared.json'
    harness.compare(SimpleNamespace(reference=r, candidate=c, gates=None, threshold=[], out=out))
    result = json.loads(out.read_text())
    assert not result['passed']
    assert not result['public_checks']['chi2']['passed']
    assert result['public_checks']['standard_engine']['passed']


def test_native_correction_changes_only_filter_site():
    import corrected_reference
    source = 'def search_multi_periods():\n    combined = list(enumerate(zip(periods, -power)))\n    period = periods[periodIndex]\n'
    result = corrected_reference.corrected_source(source)
    assert result.count('period = periods[periodIndex]') == 1
    assert result.replace(result.splitlines()[2]+'\n', '') == source
    with pytest.raises(ValueError, match='audited correction sites'):
        corrected_reference.corrected_source(source+source)


def test_noop_reuse_requires_complete_unchanged_selection_trace():
    import corrected_reference
    periods = np.linspace(.6, 12., 301)
    powers = np.arange(301, dtype=float)[::-1]
    arrays = dict(periods=periods, stage0_chi2_mask=np.zeros(301,bool),
                  stage0_power=powers,stage0_power_mask=np.zeros(301,bool))
    indices=corrected_reference.finite_candidate_indices(periods,powers)
    arrays.update(refinement_indices=indices,refinement0_periods=periods[indices],
                  refinement1_periods=np.array([1.,2.,4.]))
    result=dict(period=2.,stages=[{}, {},dict(preceding_selected_period_masked=False,
                                           preceding_selected_period_finite=True)])
    assert corrected_reference.no_op_receipt(arrays,result)['proved_no_op']
    arrays['stage0_chi2_mask'][0]=True
    assert not corrected_reference.no_op_receipt(arrays,result)['proved_no_op']
    arrays['stage0_chi2_mask'][0]=False
    arrays['refinement0_periods'][0]=np.nan
    assert not corrected_reference.no_op_receipt(arrays,result)['proved_no_op']


def test_finite_reference_selection_preserves_ties_and_period_cut():
    import corrected_reference
    periods=np.linspace(.6,12.,400)
    powers=np.zeros(400)
    pm=np.zeros(400,bool);sm=np.zeros(400,bool)
    pm[0:10]=True;sm[50:60]=True;powers[80]=np.nan
    p=np.ma.array(periods,mask=pm);s=np.ma.array(powers,mask=sm)
    result=corrected_reference.finite_candidate_indices(p,s)
    valid=np.flatnonzero(~(pm|sm)&np.isfinite(powers))
    np.testing.assert_array_equal(result[:100],valid[:100])
    expected=[i for i in valid[100:] if periods[i]>1][:100]
    np.testing.assert_array_equal(result[100:],expected)
