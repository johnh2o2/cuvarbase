"""CPU API-contract tests using a mocked observation-level TLS engine.

These tests cover routing, validation, units, ordering and null-search settings.
They do not evaluate a transit search or make physical-sensitivity claims.
"""

import builtins
import sys
import types

import numpy as np
import pytest

import cuvarbase
from cuvarbase import base, tls, tls_reference_frontend as frontend


@pytest.fixture
def lightcurve():
    return (np.array([0., 1.5, 3., 5.]),
            np.array([1., .999, 1.0002, .9997]),
            np.array([.001, .002, .003, .002]))


@pytest.fixture
def mock_engine(monkeypatch):
    engine = types.ModuleType('cuvarbase.tls_reference')
    engine.calls, engine.final_calls, engine.fit_calls = [], [], []
    engine.context_calls = 0
    engine.null = False
    engine.invalid_final = False
    engine.invalid_raw = False
    engine.mask_first = False

    def context():
        engine.context_calls += 1

    def stage(count, scale, first_time):
        return dict(chi2=(3 + np.arange(count, dtype=float))*scale**2,
                    start=np.zeros(count, dtype=np.int64),
                    width_index=np.zeros(count, dtype=np.int64),
                    width=np.ones(count, dtype=np.int64),
                    depth=np.full(count, .001), start_time=np.full(count, first_time))

    def run(full, t, y, dy, periods, **options):
        engine.calls.append(dict(full=full, t=t.copy(), y=y.copy(), dy=dy.copy(),
                                 periods=periods.copy(), options=options))
        prepared = frontend.reference.preprocess_inputs(t, y, dy)
        engine.last_prepared = prepared
        scale = prepared['error_scale']
        count = len(periods)
        primary = min(1, count-1)
        raw = stage(count, scale, t.min() + .1)
        raw['chi2'][primary] = 2*scale**2
        raw['group_size'] = 1
        if engine.invalid_raw:
            raw['width_index'][0] = -1
            raw['depth'][0] = 0
        power = np.arange(count, dtype=float) - 1
        power[primary] = 5
        spectra = dict(chi2=np.ma.array(raw['chi2']),
                       power=np.ma.array(power), SR=np.ma.array(np.full(count, .8)),
                       primary_index=None if engine.null else primary, SDE=5., SDE_raw=6.)
        if engine.mask_first:
            spectra['chi2'][0] = np.ma.masked
            spectra['power'][0] = np.ma.masked
            spectra['SR'][0] = np.ma.masked
        cache = dict(overview=np.array([(.02, 1, 1.2), (.05, 2, 1.3)],
                                      dtype=frontend.reference.OVERVIEW_DTYPE),
                     unique_indices=np.array([0, 1]), widths=np.array([1, 2]),
                     omitted_rows=[])
        result = dict(prepared=prepared, cache=cache, spectra=spectra, raw=raw,
                      primary_index=primary, period=None if engine.null else periods[primary],
                      duration_selection=None)
        if full and not engine.null:
            selected = stage(1, scale, t.min() + .1)
            final = stage(1, scale, t.min() + .1)
            final['chi2'][0] = scale**2
            if engine.invalid_final:
                final['width_index'][0] = -1
            result.update(candidates=np.array([primary]), refined=selected,
                          harmonics=np.array([primary]), harmonic_results=selected, final=final)
        return result

    def raw_search(periods, t, y, dy, cache, **options):
        engine.final_calls.append(dict(periods=periods.copy(), options=options))
        result = stage(1, engine.last_prepared['error_scale'], t.min() + .1)
        result['chi2'][0] = engine.last_prepared['error_scale']**2
        return result

    def final_parameters(t, y, dy, period, cache, width_index, epoch_index, **options):
        engine.fit_calls.append(dict(t=t.copy(), period=period, options=options))
        scale = options['error_scale']
        T0 = float(t.min() + .4)
        return dict(period=float(period), T0=T0, t0_phase=.123, duration=.05, depth=.001,
                    chi2_min=float(options['fit_chi2']), chi2_null=20*scale**2,
                    chi2_cpu_model=1.01*scale**2, delta_chi2=19., SNR=np.sqrt(19.),
                    n_transits=2, transit_times=np.array([T0, T0+period]))

    engine.search_full = lambda *args, **kwargs: run(True, *args, **kwargs)
    engine.search_fast = lambda *args, **kwargs: run(False, *args, **kwargs)
    engine.raw_search = raw_search
    engine._select_durations = lambda selection, indices: None
    monkeypatch.setitem(sys.modules, 'cuvarbase.tls_reference', engine)
    monkeypatch.setattr(cuvarbase, 'tls_reference', engine, raising=False)
    monkeypatch.setattr(base, 'ensure_context', context)
    monkeypatch.setattr(frontend.reference, 'final_parameters', final_parameters)
    return engine


def test_public_default_routes_to_broad_full_observation_search(lightcurve, mock_engine):
    result = tls.tls_search_gpu(*lightcurve, periods=[1., 2., 3.])
    call = mock_engine.calls[0]
    assert call['full'] is True
    assert call['options']['qmin'] is None
    assert call['options']['qmax'] is None
    assert call['options']['n_durations'] is None
    assert call['options']['refine_top_k'] is None
    assert result['search_configuration']['method'] == 'reference'
    assert result['search_configuration']['phase_binning'] is False
    assert result['search_configuration']['duration_policy'] == 'reference'


def test_fractional_sde_window_is_rejected_before_gpu_work(lightcurve, mock_engine):
    with pytest.raises(ValueError, match='sde_kernel_size'):
        frontend.search(*lightcurve, periods=[1., 2., 3.], oversampling_factor=3.01)
    assert mock_engine.context_calls == 0
    assert mock_engine.calls == []
    result = frontend.search(*lightcurve, periods=[1., 2., 3.],
                             oversampling_factor=3.01, sde_kernel_size=91)
    assert result['period'] == 2.
    assert mock_engine.calls[0]['options']['sde_kernel_size'] == 91


def test_automatic_late_m_dwarf_grid_uses_reference_stellar_range(lightcurve, mock_engine):
    t, y, dy = lightcurve
    expected = np.sort(frontend.reference.period_grid(np.ptp(t), R_star=.1, M_star=.1))
    result = frontend.search(t, y, dy, R_star=.1, M_star=.1, return_arrays=False)
    np.testing.assert_array_equal(mock_engine.calls[0]['periods'], expected)
    assert result['R_star'] == result['M_star'] == .1
    assert mock_engine.calls[0]['full'] is True
    assert mock_engine.calls[0]['options']['qmin'] is None


@pytest.mark.parametrize('origin', [-5.25, 0., 2457000.25])
def test_time_origin_keeps_every_sample_and_restores_absolute_epoch(origin, lightcurve, mock_engine):
    offset, y, dy = lightcurve
    t = offset + origin
    result = frontend.search(t, y, dy, periods=[1., 2., 3.])
    shifted = mock_engine.calls[0]['t']
    expected_origin = np.floor(t.min()) - 1
    np.testing.assert_array_equal(shifted, t-expected_origin)
    assert np.min(shifted) > 0
    assert result['search_configuration']['samples_used'] == len(t)
    assert result['T0'] == pytest.approx(t.min()+.4, abs=1e-9)
    np.testing.assert_allclose(result['transit_times'], [t.min()+.4, t.min()+2.4], atol=1e-9)
    expected_phase = ((result['T0']-np.floor(t.min()))/result['period']) % 1
    assert result['t0_phase'] == pytest.approx(expected_phase)


def test_result_chi_squared_uses_original_uncertainties(lightcurve, mock_engine):
    result = frontend.search(*lightcurve, periods=[1., 2., 3.])
    assert result['chi2_min'] == pytest.approx(1.)
    assert result['chi2_null'] == pytest.approx(20.)
    assert result['chi2_cpu_model'] == pytest.approx(1.01)
    assert result['SNR'] == pytest.approx(np.sqrt(19.))
    np.testing.assert_allclose(result['chi2'], [3., 2., 5.])
    assert mock_engine.fit_calls[0]['options']['error_scale'] == pytest.approx(.002)


def test_sorted_explicit_q_and_float64_periods_return_in_caller_order(lightcurve, mock_engine):
    periods = np.array([3., 1.0000000001234, 2.])
    qmin, qmax = np.array([.03, .01, .02]), np.array([.06, .04, .05])
    result = frontend.search(*lightcurve, periods=periods, qmin=qmin, qmax=qmax, n_durations=7)
    call = mock_engine.calls[0]
    np.testing.assert_array_equal(call['periods'], periods[[1, 2, 0]])
    assert call['periods'].dtype == np.float64
    np.testing.assert_array_equal(call['options']['qmin'], qmin[[1, 2, 0]])
    np.testing.assert_array_equal(call['options']['qmax'], qmax[[1, 2, 0]])
    np.testing.assert_array_equal(result['periods'], periods)
    np.testing.assert_allclose(result['chi2'], [5., 3., 2.])
    assert result['search_configuration']['duration_policy'] == 'explicit'


def test_fast_is_explicit_and_still_gets_the_native_final_noskip_fit(lightcurve, mock_engine):
    result = frontend.search(*lightcurve, periods=[1., 2., 3.], full=False, t0_oversample=4)
    assert mock_engine.calls[0]['full'] is False
    assert mock_engine.calls[0]['options']['T0_fit_margin'] == .125
    assert len(mock_engine.final_calls) == 1
    assert mock_engine.final_calls[0]['options']['full'] is True
    np.testing.assert_array_equal(mock_engine.final_calls[0]['periods'], [2.])
    assert result['search_configuration']['full'] is False


@pytest.mark.parametrize('change, kwargs', [
    ('dy_none', {}), ('nonfinite_time', {}), ('nonpositive_flux', {}),
    ('none', {'periods': [1., np.nan]}), ('none', {'R_star': 0.}),
    ('none', {'R_star': np.nan}), ('none', {'R_star': np.inf}),
    ('none', {'M_star': -1.}), ('none', {'M_star': np.nan}),
    ('none', {'qmin': .01}), ('none', {'qmin': [0.01, .02], 'qmax': .1}),
    ('none', {'nbins': 8192}), ('none', {'sde_kernel_size': 0}),
    ('none', {'n_durations': 1}), ('none', {'work_chunk': 0}),
    ('none', {'transit_template': 'misspelled'}),
])
def test_invalid_request_fails_before_engine_import_or_context(change, kwargs, lightcurve, mock_engine, monkeypatch):
    t, y, dy = (array.copy() for array in lightcurve)
    if change == 'dy_none':
        dy = None
    elif change == 'nonfinite_time':
        t[1] = np.nan
    elif change == 'nonpositive_flux':
        y[1] = 0
    options = dict(periods=[1., 2., 3.])
    options.update(kwargs)
    original_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if 'tls_reference' in fromlist:
            pytest.fail('Invalid input reached the GPU engine import')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(ValueError):
        frontend.search(t, y, dy, **options)
    assert mock_engine.context_calls == 0
    assert not mock_engine.calls


def test_degenerate_spectrum_returns_existing_null_contract(lightcurve, mock_engine):
    mock_engine.null = True
    with pytest.warns(UserWarning, match='null result'):
        result = frontend.search(*lightcurve, periods=[3., 1., 2.])
    assert np.isnan(result['period'])
    assert np.isnan(result['T0'])
    assert np.isnan(result['duration'])
    assert result['SDE'] == result['SNR'] == 0
    assert result['depth'] == 0
    assert not np.any(result['valid_periods'])
    np.testing.assert_array_equal(result['periods'], [3., 1., 2.])
    assert not mock_engine.fit_calls


def test_finite_sentinel_without_a_fitted_window_is_a_null_result(lightcurve, mock_engine):
    mock_engine.invalid_final = True
    with pytest.warns(UserWarning, match='no fitted transit'):
        result = frontend.search(*lightcurve, periods=[1., 2., 3.])
    assert np.isnan(result['period'])
    assert result['SDE'] == result['SNR'] == 0
    assert not mock_engine.fit_calls


def test_native_masking_is_not_reported_as_a_failed_trial(lightcurve, mock_engine):
    mock_engine.mask_first = True
    result = frontend.search(*lightcurve, periods=[1., 2., 3.])
    assert result['n_failed_periods'] == 0
    assert result['n_masked_periods'] == 1
    assert not result['valid_periods'][0]


def test_finite_unfitted_trial_retains_score_but_has_no_physical_parameters(lightcurve, mock_engine):
    mock_engine.invalid_raw = True
    result = frontend.search(*lightcurve, periods=[1., 2., 3.])
    assert result['valid_periods'][0]
    assert not result['parameter_valid_periods'][0]
    assert np.isfinite(result['chi2'][0])
    assert np.isnan(result['best_duration_per_period'][0])
    assert np.isnan(result['best_t0_per_period'][0])


def test_sparse_parameter_estimation_error_preserves_spectrum(lightcurve, mock_engine, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError('Native final duration estimate is nonpositive')

    monkeypatch.setattr(frontend.reference, 'final_parameters', unavailable)
    with pytest.warns(UserWarning, match='parameters are unavailable'):
        result = frontend.search(*lightcurve, periods=[1., 2., 3.])
    assert result['period'] == 2.
    assert result['SDE'] == 5.
    assert np.isnan(result['T0'])
    assert np.isnan(result['duration'])
    assert result['chi2_min'] == pytest.approx(1.)
    assert 'nonpositive' in result['parameter_error']


def test_batch_null_searches_keep_full_refinement_and_all_search_settings(lightcurve, mock_engine):
    t, y, dy = lightcurve
    settings = dict(periods=[3., 1., 2.], qmin=.01, qmax=.09, n_durations=7,
                    sde_kernel_size=31, transit_depth_min=2e-5, work_chunk=19)
    results = frontend.search_batch([(t, y, dy)], fap_null_draws=3, fap_seed=123, **settings)
    assert len(mock_engine.calls) == 4
    for call in mock_engine.calls:
        assert call['full'] is True
        assert call['options']['n_durations'] == 7
        assert call['options']['sde_kernel_size'] == 31
        assert call['options']['transit_depth_min'] == 2e-5
        assert call['options']['work_chunk'] == 19
        np.testing.assert_array_equal(call['options']['qmin'], [.01]*3)
        np.testing.assert_array_equal(call['options']['qmax'], [.09]*3)
        assert sorted(zip(call['y'], call['dy'])) == sorted(zip(y, dy))
    np.testing.assert_array_equal(results[0]['SDE_null'], [5., 5., 5.])
    assert results[0]['FAP'] == 1.
    assert 'periods' not in results[0]


def test_batch_uses_one_longest_baseline_grid(lightcurve, mock_engine, monkeypatch):
    t, y, dy = lightcurve
    grid_calls = []

    def grid(span, **kwargs):
        grid_calls.append((span, kwargs))
        return np.array([3., 2., 1.])

    monkeypatch.setattr(frontend.reference, 'period_grid', grid)
    frontend.search_batch([(t, y, dy), (t*2, y, dy)])
    assert len(grid_calls) == 1
    assert grid_calls[0][0] == np.ptp(t*2)
    for call in mock_engine.calls:
        np.testing.assert_array_equal(call['periods'], [1., 2., 3.])


def test_invalid_later_batch_input_is_rejected_before_any_gpu_work(lightcurve, mock_engine):
    t, y, dy = lightcurve
    with pytest.raises(ValueError):
        frontend.search_batch([lightcurve, (t, y, dy[:-1])], periods=[1., 2., 3.])
    assert not mock_engine.calls
    assert mock_engine.context_calls == 0
