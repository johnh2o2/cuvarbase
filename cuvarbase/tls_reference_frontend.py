"""Public input/result contract for the observation-level TLS engine.

GPU dependencies are imported only after input validation. Native numerical
details are isolated in tls_reference and tls_reference_math.
"""
import operator
import warnings

import numpy as np

from . import tls_reference_math as reference
from . import tls_grids, tls_models, tls_stats


def _positive_integer(value, name, minimum=1):
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError('%s must be an integer >= %d' % (name, minimum))
    if value < minimum:
        raise ValueError('%s must be an integer >= %d' % (name, minimum))
    return value


def _grid(t, periods, R_star, M_star, period_min, period_max,
          oversampling_factor, n_transits_min):
    from .tls import _validate_periods
    if periods is None:
        periods = reference.period_grid(
            np.ptp(t), R_star=R_star, M_star=M_star,
            period_min=0. if period_min is None else period_min,
            period_max=np.inf if period_max is None else period_max,
            oversampling_factor=oversampling_factor,
            n_transits_min=n_transits_min)
    return np.asarray(_validate_periods(periods), dtype=np.float64)


def _check_inputs(t, y, dy, name):
    from .tls import _check_tls_lightcurve
    _check_tls_lightcurve(t, y, dy, name)
    t, y, dy = (np.asarray(v, dtype=np.float64) for v in (t, y, dy))
    if len(t) < 3:
        raise ValueError('%s requires at least three observations' % name)
    if np.any(y <= 0):
        raise ValueError('%s requires positive flux normalized to a baseline of 1' % name)
    return t, y, dy


def search(t, y, dy, periods=None, *, R_star=1., M_star=1.,
           period_min=None, period_max=None, n_transits_min=2,
           oversampling_factor=3, duration_grid_step=1.1,
           qmin=None, qmax=None, qmin_fac=None, qmax_fac=None,
           duration_window=None, R_planet=1., n_durations=None,
           limb_dark='quadratic', u=None, transit_template='default',
           template_parameters=None, full=True, T0_fit_margin=.125,
           transit_depth_min=1e-5, work_chunk=256, return_arrays=True,
           t0_oversample=None, refine_top_k=None, refine_oversample=None,
           nbins=None, block_size=None, sde_kernel_size=None):
    """Run the standard full GTLS-compatible numerical search.

    Omitted duration controls select the broad native duration domain. Explicit
    q bounds replace that domain; they are never widened by workspace grouping.
    """
    from .tls import (_sort_period_grid, _to_caller_order, _null_result,
                      _validate_n_durations)
    t, y, dy = _check_inputs(t, y, dy, 'tls_search_gpu')
    for name, value in (('R_star', R_star), ('M_star', M_star)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError('%s must be finite and positive' % name)
    if u is None:
        u = [.4804, .1867]
    tls_models.validate_limb_darkening_coeffs(u, limb_dark)
    reference.resolve_template(transit_template, u, limb_dark, template_parameters)
    if nbins is not None or block_size is not None or refine_oversample is not None:
        raise ValueError('nbins, block_size and refine_oversample configure the '
                         "approximate engine; select method='binned' to use them")
    work_chunk = _positive_integer(work_chunk, 'work_chunk')
    n_transits_min = _positive_integer(n_transits_min, 'n_transits_min')
    if not np.isfinite(oversampling_factor) or oversampling_factor <= 0:
        raise ValueError('oversampling_factor must be finite and positive')
    if not np.isfinite(duration_grid_step) or duration_grid_step <= 1:
        raise ValueError('duration_grid_step must be finite and > 1')
    if not np.isfinite(T0_fit_margin) or T0_fit_margin < 0:
        raise ValueError('T0_fit_margin must be finite and nonnegative')
    if not np.isfinite(transit_depth_min) or transit_depth_min < 0:
        raise ValueError('transit_depth_min must be finite and nonnegative')
    if t0_oversample is not None:
        if not np.isfinite(t0_oversample) or t0_oversample <= 0:
            raise ValueError('t0_oversample must be finite and positive')
        T0_fit_margin = 1. / max(8., t0_oversample)
    if refine_top_k is not None:
        refine_top_k = _positive_integer(refine_top_k, 'refine_top_k', 0)
        if refine_top_k == 0:
            full = False
    if sde_kernel_size is not None:
        sde_kernel_size = _positive_integer(sde_kernel_size, 'sde_kernel_size')
    reference.spectrum_kernel_size(oversampling_factor, sde_kernel_size)
    if n_durations is not None:
        n_durations = _validate_n_durations(n_durations)
    if (qmin is None) != (qmax is None):
        raise ValueError('provide both qmin and qmax, or neither')
    if duration_window not in (None, 'reference', 'keplerian', 'fixed'):
        raise ValueError("duration_window must be 'reference', 'keplerian' or 'fixed'")
    periods_in = _grid(t, periods, R_star, M_star, period_min, period_max,
                      oversampling_factor, n_transits_min)
    periods, order = _sort_period_grid(periods_in)
    explicit_q = qmin is not None
    if explicit_q and (qmin_fac is not None or qmax_fac is not None or
                       duration_window not in (None, 'reference')):
        raise ValueError('qmin/qmax cannot be combined with a duration_window or q factors')
    if not explicit_q and (duration_window in ('keplerian', 'fixed') or
                           qmin_fac is not None or qmax_fac is not None):
        qmin, qmax = tls_grids.duration_window(
            periods_in, R_star=R_star, M_star=M_star, R_planet=R_planet,
            qmin_fac=.5 if qmin_fac is None else qmin_fac,
            qmax_fac=2. if qmax_fac is None else qmax_fac,
            window='fixed' if duration_window == 'fixed' else 'keplerian')
        explicit_q = True
    if explicit_q:
        bounds = []
        for value in (qmin, qmax):
            value = np.asarray(value, dtype=np.float64)
            if value.ndim == 0:
                value = np.full(len(periods), value)
            if value.shape != periods.shape or np.any(~np.isfinite(value)):
                raise ValueError('qmin and qmax must be finite scalars or aligned with periods')
            bounds.append(value if order is None else value[order])
        qmin, qmax = bounds
        if np.any((qmin <= 0) | (qmax >= 1) | (qmin > qmax)):
            raise ValueError('require 0 < qmin <= qmax < 1 at every period')
    elif n_durations is not None:
        raise ValueError('n_durations requires explicit qmin/qmax or a duration_window; '
                         'the default reference grid uses duration_grid_step')

    # Keep every observation, including legitimate zero/negative timestamps,
    # and avoid losing phase precision when callers use absolute BJD times.
    # Comparisons with public GTLS must give it this same positive-origin data.
    epoch = float(np.floor(np.min(t)) - 1.)
    shifted_t = t - epoch
    try:
        from . import tls_reference as engine
    except ImportError as exc:
        raise ImportError('The standard TLS engine requires CuPy and batman-package. '
                          'Install cuvarbase[tls] for CUDA 12, or install the CuPy '
                          'wheel matching your CUDA runtime plus batman-package.') from exc
    from .base import ensure_context
    ensure_context()
    runner = engine.search_full if full else engine.search_fast
    options = dict(work_chunk=work_chunk, T0_fit_margin=T0_fit_margin,
                   duration_grid_step=duration_grid_step,
                   oversampling_factor=oversampling_factor, u=u,
                   limb_dark=limb_dark, transit_template=transit_template,
                   template_parameters=template_parameters,
                   transit_depth_min=transit_depth_min,
                   sde_kernel_size=sde_kernel_size, qmin=qmin, qmax=qmax,
                   n_durations=n_durations)
    if full:
        options['refine_top_k'] = refine_top_k
    result = runner(shifted_t, y, dy, periods, **options)
    prepared, cache, spectra = result['prepared'], result['cache'], result['spectra']
    metadata = dict(method='reference', full=bool(full), phase_binning=False,
                    candidate_policy='finite_unmasked_before_ranking',
                    time_origin=epoch, input_count=len(t),
                    samples_used=len(prepared['t']),
                    duration_policy='explicit' if explicit_q else 'reference',
                    logical_group_size=result['raw']['group_size'],
                    work_chunk=work_chunk,
                    omitted_unrepresentable_durations=cache['omitted_rows'],
                    per_period_parameters='nominal sample-window diagnostics; '
                                          'winner parameters use native final postprocessing')
    null_chi2 = float(np.sum(((1. - y) / dy)**2))
    if result['period'] is None:
        message = 'TLS search has no finite detection spectrum; returning a null result (SDE = 0)'
        warnings.warn(message)
        public = _null_result(len(periods), null_chi2, message,
                              periods=periods_in, arrays=return_arrays)
        public.update(search_configuration=metadata, R_star=R_star, M_star=M_star)
        return public

    primary = result.get('primary_index', spectra['primary_index'])
    winning = result.get('final')
    if winning is None:
        # Supply cuvarbase's fitted-parameter result contract even in fast
        # mode. Public GTLS fast=True returns only its coarse periodogram;
        # this extra no-skip winner fit does not rerank that spectrum.
        winning = engine.raw_search(
            periods[primary:primary + 1], prepared['t'], prepared['y'],
            prepared['dy'], cache, group_size=1, work_chunk=1, full=True,
            transit_depth_min=transit_depth_min,
            duration_selection=engine._select_durations(
                result.get('duration_selection'), slice(primary, primary + 1)))
    scale = prepared['error_scale']
    if (winning['width_index'][0] < 0 or winning['start'][0] < 0 or
            not np.isfinite(winning['chi2'][0]) or winning['depth'][0] <= 0):
        message = 'TLS winning sample window has no fitted transit; returning a null result (SDE = 0)'
        warnings.warn(message)
        public = _null_result(len(periods), null_chi2, message,
                              periods=periods_in, arrays=return_arrays)
        public.update(search_configuration=metadata, R_star=R_star, M_star=M_star)
        return public
    fit_cache = cache
    selection = result.get('duration_selection')
    if selection is not None and winning['width_index'][0] >= 0:
        fit_cache = dict(cache, overview=cache['overview'].copy())
        row = cache['unique_indices'][winning['width_index'][0]]
        fit_cache['overview']['duration'][row] = max(
            fit_cache['overview']['duration'][row], selection['requested_qmin'][primary])
    try:
        fitted = reference.final_parameters(
            prepared['t'], prepared['y'], prepared['dy'], periods[primary],
            fit_cache, winning['width_index'][0], winning['start'][0],
            fit_chi2=winning['chi2'][0], error_scale=scale)
    except ValueError as exc:
        # A numerical spectrum is useful even when a sparse winning sample
        # window cannot support the native physical-duration estimator.
        fitted = dict(period=float(periods[primary]), T0=np.nan, t0_phase=np.nan,
                      duration=np.nan, depth=float(winning['depth'][0]),
                      chi2_min=float(winning['chi2'][0]),
                      SNR=float(np.sqrt(max(0., null_chi2 - winning['chi2'][0] / scale**2))),
                      n_transits=0, parameter_error=str(exc))
        warnings.warn('TLS period detected, but final transit parameters are unavailable: %s' % exc)
    fitted['T0'] += epoch
    if np.isfinite(fitted['T0']):
        fitted['T0'] = float(np.min(t) + ((fitted['T0'] - np.min(t)) % fitted['period']))
        fitted['t0_phase'] = float(((fitted['T0'] - np.floor(np.min(t))) / fitted['period']) % 1.)
    if 'transit_times' in fitted:
        fitted['transit_times'] = fitted['transit_times'] + epoch
    for key in ('chi2_min', 'chi2_null', 'chi2_cpu_model'):
        if key in fitted:
            fitted[key] /= scale**2
    chi2 = np.ma.filled(spectra['chi2'], np.nan).astype(np.float64) / scale**2
    valid = np.isfinite(chi2)
    public = dict(fitted, SDE=float(spectra['SDE']), SDE_raw=float(spectra['SDE_raw']),
                  period_uncertainty=tls_stats.compute_period_uncertainty(periods, chi2, primary),
                  n_failed_periods=int(np.sum(~np.isfinite(result['raw']['chi2']))),
                  n_masked_periods=int(np.sum(~valid)), R_star=R_star, M_star=M_star,
                  search_configuration=metadata)
    if return_arrays:
        raw = {key: np.array(result['raw'][key], copy=True)
               for key in ('start', 'width_index', 'width', 'depth', 'start_time')}
        if full:
            for indices, stage in ((result['candidates'], result['refined']),
                                   (result['harmonics'], result['harmonic_results'])):
                for key in raw:
                    raw[key][indices] = stage[key]
        safe_width = np.maximum(raw['width_index'], 0)
        nominal_q = cache['overview']['duration'][cache['unique_indices']][safe_width]
        if selection is not None:
            nominal_q = np.maximum(nominal_q, selection['requested_qmin'])
        duration = nominal_q * periods
        phase = ((raw['start_time'] - 1.) / periods + nominal_q / 2.) % 1.
        parameter_valid = valid & (raw['width_index'] >= 0) & (raw['depth'] > 0)
        def scatter(values):
            return _to_caller_order(values, order)
        public.update(periods=periods_in, chi2=scatter(chi2),
                      power=scatter(np.ma.filled(spectra['power'], np.nan)),
                      SR=scatter(np.ma.filled(spectra['SR'], np.nan)),
                      valid_periods=scatter(valid),
                      parameter_valid_periods=scatter(parameter_valid),
                      best_t0_per_period=scatter(np.where(parameter_valid, phase, np.nan)),
                      best_duration_per_period=scatter(np.where(parameter_valid, duration, np.nan)),
                      best_depth_per_period=scatter(np.where(parameter_valid, raw['depth'], np.nan)),
                      best_start_index_per_period=scatter(raw['start']),
                      best_width_samples_per_period=scatter(raw['width']))
    return public


def search_batch(lightcurves, *, return_arrays=False, fap_null_draws=0,
                 fap_seed=None, **kwargs):
    """Process a survey with the same full search and one shared period grid."""
    lightcurves = list(lightcurves)
    if not lightcurves:
        return []
    n_null = _positive_integer(fap_null_draws, 'fap_null_draws', 0)
    checked = []
    for i, lc in enumerate(lightcurves):
        if len(lc) != 3:
            raise ValueError('lightcurve %d must be a (t, y, dy) tuple' % i)
        checked.append(_check_inputs(*lc, name='tls_search_batch lightcurve %d' % i))
    if kwargs.get('periods') is None:
        t = max(checked, key=lambda lc: np.ptp(lc[0]))[0]
        kwargs['periods'] = _grid(
            t, None, kwargs.get('R_star', 1.), kwargs.get('M_star', 1.),
            kwargs.get('period_min'), kwargs.get('period_max'),
            kwargs.get('oversampling_factor', 3), kwargs.get('n_transits_min', 2))
    results = [search(*lc, return_arrays=return_arrays, **kwargs) for lc in checked]
    if n_null:
        # Refine nulls with exactly the observed search settings. The full
        # engine's SDE includes refinement, so a coarse-only null is invalid.
        rng = np.random.RandomState(fap_seed)
        for result, (t, y, dy) in zip(results, checked):
            null_sde = []
            for _ in range(n_null):
                permutation = rng.permutation(len(t))
                null_sde.append(search(t, y[permutation], dy[permutation],
                                       return_arrays=False, **kwargs)['SDE'])
            null_sde = np.asarray(null_sde)
            result.update(SDE_null=null_sde,
                          FAP=float((1. + np.sum(null_sde >= result['SDE'])) / (1. + n_null)))
    return results
