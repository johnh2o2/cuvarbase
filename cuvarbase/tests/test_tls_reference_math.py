"""CPU regressions for the independently verified, unbinned TLS host math.

The compact goldens come from GTLS at the recorded source commit. They check
scientific search and score conventions without a CUDA device or importing the
GTLS package. They are numerical regression fixtures, not a recovery study.
"""

import hashlib
import warnings

import numpy as np
import pytest

from cuvarbase import tls_reference_math as ref
from cuvarbase.tests._tls_reference_goldens import GOLDEN, PROVENANCE


def test_reference_pin_is_the_independently_evaluated_source():
    assert ref.GTLS_COMMIT == PROVENANCE['gtls_commit']


def test_cleaning_preserves_alignment_and_original_error_units():
    t = np.array([-1., 0., 1., 2., 3., 4., 5., 6., 7.])
    y = np.array([1., 1., 1., 1.001, 0., .999, 1., np.inf, 1.002])
    dy = np.array([1., 1., 0., .001, .002, .003, .002, .003, .004])
    result = ref.preprocess_inputs(t, y, dy)
    np.testing.assert_array_equal(result['kept_indices'], [3, 5, 6, 8])
    np.testing.assert_array_equal(result['t'], t[[3, 5, 6, 8]])
    np.testing.assert_array_equal(result['y'], y[[3, 5, 6, 8]])
    np.testing.assert_allclose(result['dy'], [.4, 1.2, .8, 1.6], rtol=1e-15)
    assert result['error_scale'] == pytest.approx(.0025)
    assert result['input_count'] == len(t)
    # Native missing-error behavior is distinct from its supplied-error branch.
    missing = ref.preprocess_inputs([1, 2, 3], [.999, 1, 1.001])
    np.testing.assert_array_equal(missing['dy'], np.full(3, np.std([.999, 1, 1.001])))
    assert missing['error_scale'] == 1.


@pytest.mark.parametrize('case', GOLDEN['periods'])
def test_ofir_period_grid_matches_native_outputs(case):
    span, radius, mass, low, high = case['args']
    periods = ref.period_grid(span, radius, mass, low, np.inf if high is None else high)
    assert len(periods) == case['count']
    np.testing.assert_allclose(periods[case['index']], case['values'], rtol=2e-14)
    assert np.all(np.diff(periods) < 0)
    assert np.all(periods > low)
    if high is not None:
        assert np.all(periods <= high)


def test_narrow_period_request_is_preserved_and_native_reset_is_opt_in():
    narrow = ref.period_grid(25.75, period_min=5., period_max=5.01)
    assert 0 < len(narrow) < 100
    assert np.all((narrow > 5.) & (narrow <= 5.01))
    with pytest.warns(UserWarning, match='resets short grids'):
        compatible = ref.period_grid(25.75, period_min=5., period_max=5.01, native_fallback=True)
    expected = GOLDEN['periods'][0]
    assert len(compatible) == expected['count']
    np.testing.assert_allclose(compatible[expected['index']], expected['values'], rtol=2e-14)
    with pytest.raises(ValueError, match='provide explicit periods'):
        ref.period_grid(25.75, R_star=.005)


def test_duration_grid_matches_native_global_envelope():
    # The reference's host cache envelope differs from its CUDA period mask.
    np.testing.assert_allclose(ref.duration_grid([.6, 12.8]), GOLDEN['durations'], rtol=2e-14)
    assert GOLDEN['durations'][-1] == .12


def test_automatic_duration_grid_rejects_unbounded_density_before_allocating():
    with pytest.raises(ValueError, match='coarser duration_grid_step'):
        ref.duration_grid([.6, 12.8], duration_grid_step=1.000000000001)
    # The bounded explicit override is an available sample-resolved alternative.
    bounded = ref.augment_duration_grid([.6, 12.8], 1000, .001, .08,
                                        duration_grid_step=1.000000000001)
    assert len(bounded['fractional_durations']) <= 1001


@pytest.mark.parametrize('template', ['default', 'grazing', 'box'])
def test_template_cache_matches_native_widths_shapes_and_depth_scale(template):
    pytest.importorskip('batman')
    cache = ref.build_cache([.6, 12.8], 1000, transit_template=template)
    expected = GOLDEN['caches'][template]
    np.testing.assert_array_equal(cache['widths'], expected['widths'])
    np.testing.assert_array_equal(cache['unique_indices'], expected['unique_indices'])
    np.testing.assert_array_equal(cache['signal_lengths'], expected['lengths'])
    np.testing.assert_allclose(cache['overshoot'], expected['overshoot'], rtol=3e-6, atol=2e-7)
    for row in expected['samples']:
        np.testing.assert_allclose(cache['template_deficits'][row['row'], row['index']],
                                   row['deficit'], rtol=3e-6, atol=2e-7)
    assert not cache['omitted_rows']
    assert cache['template_deficits'].dtype == np.float32
    assert cache['template_deficits'].flags.c_contiguous
    # GTLS pads the FLUX with zero. Deficits in the padded tail are therefore
    # one; zero-padding the deficits would silently change the objective.
    for row, length in enumerate(cache['signal_lengths']):
        np.testing.assert_array_equal(cache['template_deficits'][row, length:], 1.)


def test_sparse_cache_omits_only_unrepresentable_rows_and_records_them():
    pytest.importorskip('batman')
    cache = ref.build_cache([.6, 365.25], 800)
    assert cache['omitted_rows']
    assert set(cache['overview_source_indices']).isdisjoint(row['index'] for row in cache['omitted_rows'])
    assert len(cache['overview_source_indices']) + len(cache['omitted_rows']) == len(cache['duration_grid'])
    np.testing.assert_array_equal(cache['overview']['duration'], cache['duration_grid'][cache['overview_source_indices']])
    assert np.all(cache['widths'] > 0)
    assert np.all(cache['signal_lengths'] > 0)
    assert np.all(np.isfinite(cache['template_deficits']))
    assert np.all(np.isfinite(cache['overshoot']))
    with pytest.raises(ValueError, match='zero-sample duration'):
        ref.build_cache([.6, 365.25], 800, strict=True)


def test_explicit_duration_bounds_are_per_period_despite_a_shared_cache():
    qmin = np.array([.0011, .0501, .01011])
    qmax = np.array([.0039, .0909, .01019])
    grid = ref.augment_duration_grid([1., 2., 3.], 1000, qmin, qmax, n_durations=5)
    np.testing.assert_array_equal(grid['width_minima'], [1, 50, 10])
    np.testing.assert_array_equal(grid['width_maxima'], [3, 90, 10])
    np.testing.assert_array_equal(grid['requested_qmin'], qmin)
    np.testing.assert_array_equal(grid['requested_qmax'], qmax)
    assert len(grid['fractional_durations']) <= 1000
    assert np.all(np.diff(grid['widths']) > 0)
    for period_index in range(3):
        allowed = ((grid['widths'] >= grid['width_minima'][period_index]) &
                   (grid['widths'] <= grid['width_maxima'][period_index]))
        representative = np.maximum(grid['representative_durations'][allowed], qmin[period_index])
        assert np.all(representative >= qmin[period_index])
        assert np.all(representative <= qmax[period_index])
        mapped = ((representative/grid['maximum_fractional_duration'])*grid['reference_maxwidth']).astype(int)
        np.testing.assert_array_equal(mapped, grid['widths'][allowed])
    # This period admits a single width even though the shared cache has
    # numerous rows. A native logical-group union must not widen its bounds.
    allowed = (grid['widths'] >= 10) & (grid['widths'] <= 10)
    np.testing.assert_array_equal(grid['widths'][allowed], [10])


def test_explicit_duration_grid_includes_requested_geometric_resolution():
    grid = ref.augment_duration_grid([1., 2.], 1000, [.0071, .0301], [.0319, .1609], n_durations=[3, 5])
    for low, high, count in zip([.0071, .0301], [.0319, .1609], [3, 5]):
        requested = np.geomspace(low, high, count)
        expected_widths = ((requested/grid['maximum_fractional_duration'])*grid['reference_maxwidth']).astype(int)
        assert set(expected_widths).issubset(set(grid['widths']))
    assert grid['maximum_fractional_duration'] == .1609
    assert grid['metadata']['saturated_period_count'] == 0


@pytest.mark.parametrize('ndata, upper', [(1000, .1609), (973, .1321), (801, .129837712472)])
def test_augmented_grid_keeps_build_cache_normalization_and_actual_widths(ndata, upper):
    pytest.importorskip('batman')
    grid = ref.augment_duration_grid([.6, 12.8], ndata, [.0043, .0601], [.0329, upper], n_durations=31)
    cache = ref.build_cache([.6, 12.8], ndata, fractional_durations=grid['fractional_durations'])
    assert np.max(cache['duration_grid']) == grid['maximum_fractional_duration']
    assert cache['reference_maxwidth'] == grid['reference_maxwidth']
    omitted_widths = {row['width_in_samples'] for row in cache['omitted_rows']}
    np.testing.assert_array_equal(cache['widths'], [w for w in grid['widths'] if w not in omitted_widths])
    for i in range(2):
        allowed = ((cache['widths'] >= grid['width_minima'][i]) &
                   (cache['widths'] <= grid['width_maxima'][i]))
        q = np.maximum(cache['overview']['duration'][cache['unique_indices']][allowed], grid['requested_qmin'][i])
        assert np.all(q >= grid['requested_qmin'][i])
        assert np.all(q <= grid['requested_qmax'][i])
        mapped = ((q/np.max(cache['duration_grid']))*cache['reference_maxwidth']).astype(int)
        np.testing.assert_array_equal(mapped, cache['widths'][allowed])


def test_duration_override_saturates_at_sample_resolution_without_large_grid():
    grid = ref.augment_duration_grid([1., 2.], 1000, [.0011, .0501], [.0309, .0809], n_durations=10**12)
    assert grid['metadata']['saturated_period_count'] == 2
    assert len(grid['fractional_durations']) <= 1000
    for low, high in [(1, 30), (50, 80)]:
        assert set(range(low, high+1)).issubset(set(grid['widths']))
    tiny = ref.augment_duration_grid([1.], 1000, 1e-300, .0809, duration_grid_step=1.000000000001)
    assert tiny['metadata']['saturated_period_count'] == 1
    assert np.all(np.isfinite(tiny['fractional_durations']))


@pytest.mark.parametrize('kwargs', [
    dict(qmin=0., qmax=.1), dict(qmin=.1, qmax=1.),
    dict(qmin=[.01, .02], qmax=.1), dict(qmin=.1, qmax=.01),
    dict(qmin=.01, qmax=.1, n_durations=2.5),
])
def test_invalid_explicit_duration_requests_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        ref.augment_duration_grid([1.], 1000, **kwargs)


def _spectrum_input(size, dtype):
    i = np.arange(size)
    raw = (1 + .3*np.sin(i*.025) + .04*np.cos(i*.8)).astype(dtype)
    if size > 200:
        raw[35] -= .10
        raw[100] -= .13
        raw[[5, 177]] = 1e6
    return raw


@pytest.mark.parametrize('case', list(GOLDEN['spectra']))
def test_native_spectrum_masks_detrending_and_primary_rank(case):
    size, dtype = case.split('/')
    raw = _spectrum_input(int(size), dtype)
    result = ref.native_spectra(raw)
    expected = GOLDEN['spectra'][case]
    np.testing.assert_array_equal(np.flatnonzero(np.ma.getmaskarray(result['chi2'])), expected['masked'])
    tolerance = 2e-6 if dtype == 'float32' else 2e-13
    for key in ('SR', 'power_raw', 'power'):
        target = np.array(expected[key], dtype=float)  # None means masked/NaN.
        actual = np.ma.filled(result[key], np.nan)[expected['index']]
        np.testing.assert_allclose(actual, target, rtol=tolerance, atol=tolerance, equal_nan=True)
    for key in ('SDE', 'SDE_raw'):
        assert float(result[key]) == pytest.approx(expected[key], rel=tolerance)
    assert result['primary_index'] == expected['primary']
    if int(size) > 200:
        # A lower raw chi-squared is not necessarily the primary detection
        # after the reference's running-median normalization.
        assert expected['primary'] != expected['minimum_chi2']


def test_spectrum_recalculation_preserves_its_supplied_mask():
    raw = _spectrum_input(217, 'float64')
    first = ref.native_spectra(raw)
    refined = first['chi2'].copy()
    refined.data[5] = .1
    result = ref.native_spectra(refined, mask_outliers=False)
    assert np.ma.getmaskarray(result['chi2'])[5]
    assert result['primary_index'] != 5


def test_explicit_spectrum_window_is_odd_and_changes_only_requested_statistic():
    raw = _spectrum_input(217, 'float64')
    even = ref.native_spectra(raw, kernel_size=30)
    odd = ref.native_spectra(raw, kernel_size=31)
    np.testing.assert_array_equal(even['power'], odd['power'])
    native = ref.native_spectra(raw)
    np.testing.assert_array_equal(even['SR'], native['SR'])
    np.testing.assert_array_equal(even['power_raw'], native['power_raw'])
    assert not np.array_equal(even['power'], native['power'])
    with pytest.raises(ValueError, match='positive integer'):
        ref.native_spectra(raw, kernel_size=0)


def test_fractional_default_spectrum_window_requires_explicit_integer_window():
    raw = 1 + .1*np.sin(np.arange(217))
    with pytest.raises(ValueError, match='sde_kernel_size'):
        ref.native_spectra(raw, oversampling_factor=3.01)
    assert ref.spectrum_kernel_size(3.) == 91
    assert ref.spectrum_kernel_size(2.5) == 75
    assert ref.spectrum_kernel_size(3.01, 90) == 91
    explicit = ref.native_spectra(raw, oversampling_factor=3.01, kernel_size=90)
    native = ref.native_spectra(raw)
    np.testing.assert_array_equal(explicit['power'], native['power'])


@pytest.mark.parametrize('window_chunk', [1, 4, 8192])
def test_running_median_chunking_preserves_full_masks_and_edge_padding(window_chunk):
    values = ((np.arange(31)*7) % 17).astype(float)
    mask = np.zeros(31, bool)
    mask[6:13] = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        actual = ref._running_median(np.ma.array(values, mask=mask), 5, window_chunk)
    np.testing.assert_allclose(np.ma.getdata(actual), np.array(GOLDEN['median']['values'], dtype=float), equal_nan=True)
    np.testing.assert_array_equal(np.flatnonzero(np.ma.getmaskarray(actual)), GOLDEN['median']['masked'])


def test_epoch_stride_covers_thin_windows_and_full_mode_visits_every_start():
    widths = np.array([1, 7, 8, 9, 15, 16, 31, 64, 128])
    np.testing.assert_array_equal(ref.epoch_strides(widths), [1, 1, 1, 1, 1, 2, 3, 8, 16])
    np.testing.assert_array_equal(ref.epoch_strides(widths, T0_fit_margin=0), np.ones(len(widths)))
    np.testing.assert_array_equal(ref.epoch_strides(widths, full=True), np.ones(len(widths)))


def test_native_group_width_union_is_explicit():
    masks = ref.chunk_width_masks([2, 4, 8, 16], [1, 6, 12], [5, 9, 20], chunk_size=2)
    np.testing.assert_array_equal(masks, [[True, True, True, False], [False, False, False, True]])


def test_unmasked_full_candidate_rank_order_matches_frozen_native_selection():
    periods = np.linspace(.05, 5, 250)
    power = ((np.arange(250)*37) % 251)/251.
    actual = ref.refinement_candidate_indices(periods, power).astype('<i8')
    expected = GOLDEN['candidates']['False']
    assert len(actual) == expected['count']
    np.testing.assert_array_equal(actual[:12], expected['first'])
    np.testing.assert_array_equal(actual[-12:], expected['last'])
    assert hashlib.sha256(actual.tobytes()).hexdigest() == expected['sha256']


def test_full_candidates_exclude_masked_and_nonfinite_values_before_ranking():
    periods = np.ma.array([.3, 1.5, 2., 3., np.nan, 4., 5., 6., 7., 8., .5, np.inf],
                          mask=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    power = np.ma.array([100., 4., 8., 200., 500., np.nan, np.inf, -np.inf, 6., 6., -2., 9.],
                        mask=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Five eligible rows have scores 8, 6, 6, 4, -2. Their original order
    # resolves the equal scores. A period below one day is valid in top100.
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        actual = ref.refinement_candidate_indices(periods, power)
    np.testing.assert_array_equal(actual, [2, 8, 9, 1, 10])


def test_masked_prefix_does_not_consume_finite_candidate_slots():
    periods = 2. + np.arange(320)*.01
    mask = np.arange(320) < 70
    power = np.ma.array(np.arange(320, dtype=float), mask=mask)
    # The 200 highest finite scores are rows 319 through 120. Native sorting
    # of masked scalar keys previously let leading masked rows consume slots.
    actual = ref.refinement_candidate_indices(np.ma.array(periods, mask=mask), power)
    np.testing.assert_array_equal(actual, np.arange(319, 119, -1))


def test_candidate_ties_keep_input_order_and_second_quota_requires_period_above_one():
    periods = np.r_[np.full(110, .75), np.ones(10), np.full(120, 2.)]
    power = np.ones(240)
    # Equal scores keep rows 0:100 in the first quota. Rows 100:120 fail the
    # strict P>1 condition; rows 120:220 fill the second quota in input order.
    actual = ref.refinement_candidate_indices(periods, power)
    np.testing.assert_array_equal(actual, np.r_[np.arange(100), np.arange(120, 220)])


@pytest.mark.parametrize('periods,power', [
    ([], []),
    (np.ma.masked_all(3), [1., 2., 3.]),
    ([1., 2., 3.], np.ma.masked_all(3)),
    ([np.nan, np.inf, -np.inf], [1., 2., 3.]),
    ([1., 2., 3.], [np.nan, np.inf, -np.inf]),
])
def test_full_candidate_selection_returns_empty_when_no_finite_unmasked_trial_exists(periods, power):
    actual = ref.refinement_candidate_indices(periods, power)
    assert actual.shape == (0,)
    assert np.issubdtype(actual.dtype, np.integer)


def test_harmonic_selection_preserves_native_order_and_mask_behavior():
    periods = np.ma.array([.2, .9, 1., 1.5, 2., 2.7, 3.8], mask=[0, 0, 1, 0, 0, 0, 0])
    # Native find_nearest_indices drops the period mask before nearest lookup.
    np.testing.assert_array_equal(ref.harmonic_candidate_indices(periods, 1.), [0, 2, 4, 1, 3])
    np.testing.assert_array_equal(ref.harmonic_candidate_indices([1., 2.], 1.), [0, 0, 1, 0, 0])


@pytest.mark.parametrize('copy_score_mask', [False, True])
def test_valid_harmonic_refinement_can_rehabilitate_a_masked_coarse_period(copy_score_mask):
    periods = np.ma.array([.2, .9, 1., 1.5, 2., 2.7, 3.8], mask=[0, 0, 1, 0, 0, 0, 0])
    chi2 = np.ma.array([4., 4., 1e6, 4., 4., 4., 4.],
                       mask=np.ma.getmaskarray(periods), copy=copy_score_mask)
    indices = ref.harmonic_candidate_indices(periods, 2.)
    np.testing.assert_array_equal(indices, [2, 4, 6, 3, 5])
    # These stand for successful, finite full-window evaluations, including
    # P=1d, which was masked at the coarse stage. NumPy's indexed assignment
    # deliberately clears its chi2 mask. Native shares that mask with periods;
    # production keeps a separate copy of the original period mask.
    chi2[indices] = [1., 3., 4., 4., 4.]
    spectrum = ref.native_spectra(chi2, mask_outliers=False)
    assert spectrum['primary_index'] == 2
    assert not np.ma.getmaskarray(spectrum['chi2'])[2]
    # The actual grid value remains the correct finite input for this newly
    # valid harmonic in either case, regardless of mask ownership.
    assert np.ma.getdata(periods)[spectrum['primary_index']] == 1.
    assert np.ma.is_masked(periods[spectrum['primary_index']]) == copy_score_mask


@pytest.mark.parametrize('wrap', [False, True])
def test_final_parameters_match_native_sample_window_and_preserve_snr_units(wrap):
    pytest.importorskip('batman')
    cache = ref.build_cache([.6, 12.8], 1000)
    i = np.arange(1000)
    t = 1 + i*.026 + ((i*7) % 11)*.00001
    y = 1 + .0002*((i*11) % 23-11)/11
    expected = GOLDEN['final'][str(wrap)]
    period, width_index = 3.7, 12
    rank = np.argsort((t % period)/period)
    window = (expected['epoch_index'] + np.arange(expected['width'])) % len(t)
    y[rank[window]] -= .001
    result = ref.final_parameters(t, y, np.ones(len(t)), period, cache, width_index,
                                  expected['epoch_index'], exposure_days=200/86400, error_scale=.0002)
    for key in ('T0', 'depth', 'duration', 'native_gtls_snr', 'transit_times', 'per_transit_count'):
        np.testing.assert_allclose(result[key], expected[key], rtol=2e-12, atol=1e-14)
    assert result['SNR'] == pytest.approx(np.sqrt(max(0, result['chi2_null'] - result['chi2_min']))/.0002)
    assert result['exposure']['integrated_in_search'] is False
    assert result['exposure']['median_days'] == pytest.approx(200/86400)
    assert result['width_in_samples'] == expected['width']
    assert result['n_transits'] == len(expected['transit_times'])
