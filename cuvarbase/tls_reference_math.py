"""CPU preparation and scoring for the unbinned reference TLS search.

Reference: Farthing-0/GTLS at 74e449c325792a763dde4fbffab98039c5e8c111.
This module deliberately retains the reference template, index-based widths,
zero-padding convention, unit flux baseline and native spectrum normalization.
The functions do not initialize a GPU. Template construction imports the
optional ``batman`` dependency only when a cache is requested.

Adapted portions: grid.py, transit.py, validate.py, stats.py and helpers.py.
MIT License
Copyright (c) 2018 Michael Hippke 2023 Quanquan Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings

import numpy as np


GTLS_COMMIT = '74e449c325792a763dde4fbffab98039c5e8c111'
G = 6.673e-11
R_SUN = 695508000.
R_JUP = 69911000.
# Preserve the pinned expression: it differs by one binary64 ULP from 1.989e30.
M_SUN = 1.989 * 10**30
SECONDS_PER_DAY = 86400.
DEFAULT_TEMPLATE = dict(per=12.9, rp=.03, a=23.1, inc=89.21, ecc=0., w=90.,
                        u=[.4804, .1867], limb_dark='quadratic')
OVERVIEW_DTYPE = [('duration', 'f8'), ('width_in_samples', 'i8'), ('overshoot', 'f8')]


def preprocess_inputs(t, y, dy=None):
    """Apply native GTLS cleaning and error rescaling, preserving row indices.

    In particular, GTLS drops t <= 0. A caller adopting a different time-origin
    policy must do so explicitly before this function; that policy is not exact
    preprocessing parity on arbitrary inputs. Flux is checked, not normalized.
    Supplied dy is divided by its arithmetic mean. Omitted dy uses std(flux).
    """
    t, y = np.asarray(t), np.asarray(y)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError('t and y must be one-dimensional arrays of equal length')
    if dy is not None:
        dy = np.asarray(dy)
        if dy.ndim != 1 or len(dy) != len(y):
            raise ValueError('dy must have the same shape as y')

    def valid(value):
        return value is not None and not np.isnan(value) and value > 0 and value < np.inf

    kept = [i for i in range(len(y)) if valid(t[i]) and valid(y[i]) and
            (dy is None or valid(dy[i]))]
    index = np.array(kept, dtype=np.int64)
    clean_t = np.array(t[index], dtype=float)
    clean_y = np.array(y[index], dtype=float)
    if len(clean_y) < 3 or np.ptp(clean_t) <= 0:
        raise ValueError('At least three finite positive samples and a positive time span are required')
    if np.mean(clean_y) > 1.01 or np.mean(clean_y) < .99:
        warnings.warn('The mean flux should be normalized to 1; input mean is %s' % np.mean(clean_y))
    if dy is None:
        clean_dy = np.full(len(clean_y), np.std(clean_y))
        dy_scale = 1.
    else:
        clean_dy = np.array(dy[index], dtype=float)
        dy_scale = float(np.mean(clean_dy))
        clean_dy = clean_dy / np.mean(clean_dy)
    return dict(t=clean_t, y=clean_y, dy=clean_dy, kept_indices=index,
                input_count=len(t), error_scale=dy_scale)


def period_grid(time_span, R_star=1., M_star=1., period_min=0., period_max=np.inf,
                oversampling_factor=3., n_transits_min=2, native_fallback=False):
    """Ofir grid using pinned GTLS constants; returns native descending order.

    Normal mode respects requested stellar/period bounds, including grids with
    fewer than 100 periods. It rejects unsupported stellar values explicitly.
    native_fallback=True reproduces GTLS's historical clamps and its <100-point
    fallback, which discards requested bounds. That option is for fixtures only.
    The search caller must sort the result ascending, as GTLS.main.power does.
    """
    R_star, M_star, time_span = float(R_star), float(M_star), float(time_span)
    if not np.isfinite(time_span) or time_span <= 0 or not np.isfinite(oversampling_factor) or oversampling_factor <= 0:
        raise ValueError('time_span and oversampling_factor must be finite and positive')
    if not isinstance(n_transits_min, (int, np.integer)) or n_transits_min < 1:
        raise ValueError('n_transits_min must be a positive integer')
    if period_min < 0 or period_max <= period_min or np.isnan(period_max):
        raise ValueError('Require 0 <= period_min < period_max')
    if not np.isfinite(R_star) or not np.isfinite(M_star):
        raise ValueError('Stellar radius and mass must be finite')
    if native_fallback:
        if R_star < .01:
            warnings.warn('Native GTLS radius clamp sets R_star=0.1 below 0.01')
            R_star = .1
        if R_star > 10000:
            warnings.warn('Native GTLS radius clamp sets R_star=10000')
            R_star = 10000.
        if M_star < .01:
            warnings.warn('Native GTLS mass clamp sets M_star=0.01')
            M_star = .01
        if M_star > 1000:
            warnings.warn('Native GTLS mass clamp sets M_star=1000')
            M_star = 1000.
    elif not (.01 <= R_star <= 10000 and .01 <= M_star <= 1000):
        raise ValueError('Reference period grid supports 0.01 <= R_star <= 10000 and 0.01 <= M_star <= 1000; provide explicit periods for other ranges')

    radius, mass, span = R_star * R_SUN, M_star * M_SUN, time_span * SECONDS_PER_DAY
    f_min = n_transits_min / span
    f_max = 1. / (2 * np.pi) * np.sqrt(G * mass / (3 * radius)**3)
    A = ((2 * np.pi)**(2. / 3) / np.pi * radius / (G * mass)**(1. / 3) /
         (span * oversampling_factor))
    C = f_min**(1. / 3) - A / 3.
    count = (f_max**(1. / 3) - f_min**(1. / 3) + A / 3) * 3 / A
    X = np.arange(count) + 1
    frequencies = (A / 3 * X + C)**3
    periods = (1 / frequencies) / SECONDS_PER_DAY
    periods = periods[(periods > period_min) & (periods <= period_max)]
    if len(periods) > 10**6:
        warnings.warn('Reference period grid contains more than one million periods')
    if len(periods) < 100 and native_fallback:
        warnings.warn('Native GTLS resets short grids to solar parameters and default period limits')
        return period_grid(max(time_span, 5.), native_fallback=True)
    return np.asarray(periods, dtype=np.float64)


def _t14(R_s, M_s, period, small=False, upper_limit=.12):
    seconds = period * SECONDS_PER_DAY
    radius = R_SUN * R_s
    mass = M_SUN * M_s
    if not small:
        radius += 2 * R_JUP
    duration = radius * ((4 * seconds) / (np.pi * G * mass))**(1. / 3)
    return min(duration / seconds, upper_limit)


def duration_grid(periods, duration_grid_step=1.1):
    """Pinned global duration grid; native accepted stellar bounds are unused.

    Host constants are R=[0.13,3.5], M=[0.1,1], cap=0.12. The separate CUDA
    admissibility envelope uses different literal constants. These are sample
    index widths once cached; they are not an exact phase-domain duration fit.
    """
    periods = np.asarray(periods, dtype=np.float64)
    if periods.ndim != 1 or len(periods) == 0 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError('periods must be a nonempty positive finite 1D array')
    if not np.isfinite(duration_grid_step) or duration_grid_step <= 1:
        raise ValueError('duration_grid_step must be greater than one')
    maximum = _t14(3.5, 1., min(periods), small=False)
    minimum = _t14(.13, .1, max(periods), small=True)
    estimated_rows = max(2., np.ceil((np.log(maximum) - np.log(minimum)) /
                                    np.log(duration_grid_step)) + 1.)
    if not np.isfinite(estimated_rows) or estimated_rows > 1_000_000:
        raise ValueError('duration_grid_step would create more than 1,000,000 '
                         'duration rows; choose a coarser duration_grid_step '
                         'or explicit sample-resolved qmin/qmax bounds')
    widths = [minimum]
    current = minimum
    while current * duration_grid_step < maximum:
        # Repeated multiplication is retained for exact native default rows.
        # Guard its possible accumulated last-bit error at the estimate limit.
        if len(widths) >= 999_999:
            raise ValueError('duration_grid_step would create more than 1,000,000 '
                             'duration rows; choose a coarser duration_grid_step '
                             'or explicit sample-resolved qmin/qmax bounds')
        current *= duration_grid_step
        widths.append(current)
    widths.append(maximum)
    return np.array(widths, dtype=np.float64)


def resolve_template(transit_template='default', u=None, limb_dark='quadratic',
                     template_parameters=None):
    """Mirror GTLS.validate_args template resolution, including default resets."""
    result = dict(DEFAULT_TEMPLATE)
    result['u'] = list(DEFAULT_TEMPLATE['u'] if u is None else u)
    result['limb_dark'] = limb_dark
    supplied = {} if template_parameters is None else dict(template_parameters)
    unsupported = set(supplied) - set(DEFAULT_TEMPLATE) - {'b'}
    if unsupported:
        raise ValueError('Unknown template parameters: ' + ', '.join(sorted(unsupported)))
    result.update({k: v for k, v in supplied.items() if k != 'b'})
    if 'b' in supplied:
        result['inc'] = np.degrees(np.arccos(supplied['b'] / result['a']))
    if transit_template == 'default':
        for key in ('per', 'rp', 'a', 'inc'):
            result[key] = DEFAULT_TEMPLATE[key]
    elif transit_template == 'grazing':
        result['inc'] = np.degrees(np.arccos(.99 / result['a']))
    elif transit_template == 'box':
        result.update(per=29., rp=.1, a=26.9, inc=90., u=[0.], limb_dark='linear')
    else:
        raise ValueError('transit_template must be default, grazing, or box')
    result['u'] = list(result['u'])
    return result


def _interp(x_new, x, y):
    """Native linear interpolation arithmetic, without a Numba dependency."""
    x, y, x_new = np.asarray(x), np.asarray(y), np.asarray(x_new)
    if len(x) < 2:
        raise ValueError('Reference interpolation needs at least two samples')
    index = np.clip(np.searchsorted(x, x_new, side='right') - 1, 0, len(x) - 2)
    theta = (x_new - x[index]) / (x[index + 1] - x[index])
    return (1 - theta) * y[index] + theta * y[index + 1]


def _reference_transit(samples, parameters):
    import batman
    t = np.linspace(-.5, .5, 10000)
    params = batman.TransitParams()
    params.t0 = 0
    for name, value in parameters.items():
        setattr(params, name, value)
    flux = batman.TransitModel(params, t).light_curve(params)
    first = np.argmax(flux < 1)
    interior_flux = flux[first:-first + 1]
    interior_time = t[first:-first + 1]
    x_new = np.linspace(t[first], t[-first - 1], samples)
    sampled = _interp(x_new, interior_time, interior_flux)
    return (np.min(sampled) - sampled) / (np.min(sampled) - 1)


def build_cache(periods, ndata, transit_template='default', duration_grid_step=1.1,
                u=None, limb_dark='quadratic', template_parameters=None,
                fractional_durations=None, strict=False):
    """Return exact host cache rows plus contiguous float32 GPU input buffers.

    fractional_durations is an explicit caller-owned search override. The
    default preserves the pinned global duration grid. template_deficits uses
    GTLS's literal *zero-flux* padding, so padded deficits are one. signal_lengths
    records the trimmed cache lengths separately; the GPU still scans widths.
    Normal mode omits rows with no representable in-transit sample and records
    them in omitted_rows. This keeps all usable reference rows on sparse input
    where GTLS otherwise fails constructing the entire cache. strict=True
    retains that failure for native compatibility fixtures.
    """
    if not isinstance(ndata, (int, np.integer)) or ndata < 3:
        raise ValueError('ndata must be an integer >= 3')
    durations = (duration_grid(periods, duration_grid_step) if fractional_durations is None
                 else np.asarray(fractional_durations, dtype=np.float64))
    if durations.ndim != 1 or len(durations) == 0 or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError('fractional durations must be a nonempty positive finite 1D array')
    maxwidth = int(np.max(durations) * ndata)
    if maxwidth % 2:
        maxwidth += 1
    if maxwidth < 2:
        raise ValueError('Reference template cache is undersampled (maximum width < 2 samples)')
    params = resolve_template(transit_template, u, limb_dark, template_parameters)
    reference = _reference_transit(maxwidth, params)
    overview = np.zeros(len(durations), dtype=OVERVIEW_DTYPE)
    curves = [None] * len(durations)
    usable = []
    omitted = []
    reference_time = np.linspace(-.5, .5, maxwidth)
    for row, duration in enumerate(durations):
        used = int((duration / np.max(durations)) * maxwidth)
        if used < 1:
            if strict:
                raise ValueError('Reference cache contains a zero-sample duration')
            omitted.append(dict(index=row, duration=float(duration), width_in_samples=used,
                                reason='zero-sample duration'))
            continue
        sampled = _interp(np.linspace(-.5, .5, used), reference_time, reference)
        missing = maxwidth - used
        empty = np.ones(int(missing * .5))
        scaled = np.append(np.append(empty, sampled), empty)
        if len(scaled) < maxwidth:
            scaled = np.append(scaled, np.ones(1))
        scaled = 1 - ((1 - scaled) * .5)
        inside = np.where(scaled < (1 - .01e-6))[0]
        if len(inside) == 0:
            if strict:
                raise ValueError('Reference cache contains a template with no in-transit samples')
            omitted.append(dict(index=row, duration=float(duration), width_in_samples=used,
                                reason='no in-transit template samples'))
            continue
        signal = scaled[int(np.min(inside)):int(np.max(inside)) + 1]
        overshoot = np.mean(signal) / np.min(signal)
        overview[row] = duration, used, 1 / (2 - overshoot)
        curves[row] = signal
        usable.append(row)
    if not usable:
        raise ValueError('No reference template is representable at this sample count')
    overview = overview[usable]
    curves = [curves[i] for i in usable]
    widths, indices = np.unique(overview['width_in_samples'], return_index=True)
    curves_unique = [curves[i] for i in indices]
    template_deficits = 1 - np.array([np.pad(curve, (0, int(np.max(widths)) - len(curve)), 'constant')
                                     for curve in curves_unique])
    return dict(duration_grid=durations, overview=overview, unique_indices=indices,
                widths=np.asarray(widths, dtype=np.int32),
                template_deficits=np.ascontiguousarray(template_deficits, dtype=np.float32),
                signal_lengths=np.array([len(v) for v in curves_unique], dtype=np.int32),
                overshoot=np.ascontiguousarray(overview['overshoot'][indices], dtype=np.float32),
                template_parameters=params, reference_flux=reference,
                signal_flux=curves_unique, reference_maxwidth=maxwidth,
                padded_data_width=int(np.max(widths)) + int(np.max(widths) % 2),
                overview_source_indices=np.array(usable, dtype=np.int64),
                omitted_rows=omitted, strict_native_cache=bool(strict))


def augment_duration_grid(periods, ndata, qmin, qmax, duration_grid_step=1.1,
                          n_durations=None):
    """Build a bounded-memory cache grid for explicit fractional durations.

    ``qmin`` and ``qmax`` are positive scalars or arrays aligned with periods;
    they refer to the native cache's nominal duration/period, not width/N.
    Requested boundaries and geometric rows augment the native global grid.
    ``n_durations`` is an optional integer or aligned integer array >= 2 and
    specifies a minimum density: other periods can contribute extra rows.

    Rows with equal integer sample widths have identical native templates, so
    only their lowest nominal q is retained, plus a maximum-q sentinel that
    keeps build_cache's template normalization unchanged. Both boundaries are
    inserted before deduplication. Since q -> width is monotonic, a width has
    at least one contributing row inside a period's bounds exactly when it is
    between that period's returned width_minima and width_maxima. Apply those
    limits per period; a logical-group union would broaden an explicit search.

    For winner metadata, max(representative_q, requested_qmin[period_index])
    gives its lowest admissible contributing nominal duration. The cache's
    shared representative can lie below a particular period's lower bound.

    Working storage is O(Nperiod + Ndata), with at most Ndata cache rows. When
    a requested geometric grid has more rows than representable sample widths,
    all admissible integer widths are included, giving at least that density.
    Empty/zero-sample templates are still recorded and omitted by build_cache.
    """
    if not isinstance(ndata, (int, np.integer)) or ndata < 3:
        raise ValueError('ndata must be an integer >= 3')
    periods = np.asarray(periods, dtype=np.float64)
    if periods.ndim != 1 or len(periods) == 0 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError('periods must be a nonempty positive finite 1D array')
    if not np.isfinite(duration_grid_step) or duration_grid_step <= 1:
        raise ValueError('duration_grid_step must be greater than one')

    def aligned(value, name, dtype):
        result = np.asarray(value)
        if result.ndim > 1 or (result.ndim == 1 and result.shape != periods.shape):
            raise ValueError(name + ' must be a scalar or an array aligned with periods')
        return np.full(len(periods), result, dtype=dtype) if result.ndim == 0 else result.astype(dtype, copy=True)

    lower, upper = aligned(qmin, 'qmin', float), aligned(qmax, 'qmax', float)
    if (np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or
            np.any(lower <= 0) or np.any(upper >= 1) or np.any(lower > upper)):
        raise ValueError('Require finite 0 < qmin <= qmax < 1 for every period')
    original_lower = _t14(.13, .1, np.max(periods), small=True)
    original_upper = _t14(3.5, 1., np.min(periods), small=False)
    maximum_q = max(float(original_upper), float(np.max(upper)))
    maximum_width = int(maximum_q * ndata)
    maximum_width += maximum_width % 2
    if maximum_width < 2:
        raise ValueError('Reference template cache is undersampled (maximum width < 2 samples)')
    representative = np.full(maximum_width + 1, np.inf)

    def widths(values):
        return ((np.asarray(values) / maximum_q) * maximum_width).astype(np.int64)

    def add(values):
        values = np.asarray(values, dtype=np.float64)
        np.minimum.at(representative, widths(values), values)

    original_count = max(2, int(np.ceil((np.log(original_upper) - np.log(original_lower)) /
                                       np.log(duration_grid_step))) + 1)
    original_saturated = original_count > maximum_width + 1
    if original_saturated:
        add([original_lower, original_upper])
    else:
        original = duration_grid(periods, duration_grid_step)
        original_count = len(original)
        add(original)
    add(lower)
    add(upper)
    minima, maxima = widths(lower), widths(upper)
    log_lower = np.log(lower)
    log_range = np.log(upper) - log_lower
    if n_durations is None:
        log_step = np.full(len(periods), np.log(duration_grid_step))
        # Cap before conversion to integer: arbitrarily tiny grid steps should
        # saturate sample resolution instead of overflowing an integer count.
        count = np.minimum(np.ceil(log_range / log_step) + 1, maximum_width + 2).astype(np.int64)
        count = np.maximum(count, 2)
    else:
        supplied = np.asarray(n_durations)
        if supplied.dtype.kind not in 'iu' or np.any(supplied < 2):
            raise ValueError('n_durations must contain integers >= 2')
        count = aligned(n_durations, 'n_durations', np.int64)
        if np.any(count < 2):
            raise ValueError('n_durations is outside the supported integer range')
        log_step = log_range / (count - 1)
    saturated = count > maximum_width + 1
    ordinary = ~saturated
    for level in range(1, int(np.max(count[ordinary])) - 1 if np.any(ordinary) else 1):
        active = ordinary & (level < count - 1)
        if np.any(active):
            values = np.exp(log_lower[active] + level * log_step[active])
            # Endpoints were inserted exactly; constrain transcendental last
            # bits so generated interior rows never exceed their own bounds.
            add(np.clip(values, lower[active], upper[active]))
    if np.any(saturated) or original_saturated:
        # Partial first widths are already represented by exact qmin. Union
        # the interior integer-width intervals with a difference array.
        change = np.zeros(maximum_width + 2, dtype=np.int64)
        np.add.at(change, minima[saturated] + 1, 1)
        np.add.at(change, maxima[saturated] + 1, -1)
        if original_saturated:
            first, last = widths([original_lower, original_upper])
            change[first + 1] += 1
            change[last + 1] -= 1
        target = np.flatnonzero(np.cumsum(change)[:maximum_width + 1] > 0)
        values = target / maximum_width * maximum_q
        rounded_down = widths(values) < target
        while np.any(rounded_down):
            values[rounded_down] = np.nextafter(values[rounded_down], np.inf)
            rounded_down = widths(values) < target
        add(values)
    usable = np.flatnonzero(np.isfinite(representative) & (np.arange(len(representative)) > 0))
    fractions = representative[usable]
    # A smaller nominal duration can round to the same final width. Dropping
    # the actual maximum would change build_cache's normalization and every
    # template. Retain it as a duplicate-width row; build_cache dedups later.
    if fractions[-1] != maximum_q:
        fractions = np.append(fractions, maximum_q)
    return dict(fractional_durations=fractions, widths=usable.astype(np.int32),
                representative_durations=representative[usable],
                width_minima=minima.astype(np.int32), width_maxima=maxima.astype(np.int32),
                requested_qmin=lower, requested_qmax=upper,
                reference_maxwidth=maximum_width, maximum_fractional_duration=maximum_q,
                metadata=dict(duration_semantics='nominal cached duration/period',
                              eligibility='per-period OR of contributing nominal rows; no group widening',
                              n_durations_semantics='minimum geometric density, with extra admissible widths allowed',
                              saturated_period_count=int(np.sum(saturated)),
                              original_grid_saturated=bool(original_saturated),
                              grid_row_count=len(fractions), unique_width_count=len(usable),
                              original_grid_row_count=original_count))


def nominal_width_bounds(periods, ndata, time_span):
    """Host translation of pinned CUDA duration literals.

    CUDA computes pow in double and receives a float32 time span. Boundary
    rounding must be checked against the device for exact integer-grid parity.
    Returned bounds are nominal; the native scan uses their union per chunk.
    """
    periods = np.asarray(periods, dtype=np.float64)
    seconds = periods * 86400
    qmin = np.minimum(.15, (695508000 * .05) * ((4 * seconds) / (20848 * 1e15))**(1. / 3) / seconds)
    qmax = np.minimum(.15, (695508000 * 4 + 2 * 69911000) * ((4 * seconds) / (416970 * 1e15))**(1. / 3) / seconds)
    transits = float(np.float32(time_span)) / periods
    correction = (transits + 1.) / transits
    return np.floor(qmin * ndata).astype(np.int32), np.ceil(qmax * ndata * correction).astype(np.int32)


def chunk_width_masks(widths, minima, maxima, chunk_size):
    """Native union of admissible integer widths over each period chunk."""
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    widths, minima, maxima = np.asarray(widths), np.asarray(minima), np.asarray(maxima)
    if minima.shape != maxima.shape or minima.ndim != 1:
        raise ValueError('minima and maxima must be aligned 1D arrays')
    admissible = (widths[None, :] >= minima[:, None]) & (widths[None, :] <= maxima[:, None])
    return np.array([np.any(admissible[start:start + chunk_size], axis=0)
                     for start in range(0, len(minima), chunk_size)], dtype=bool)


def epoch_strides(widths, T0_fit_margin=.125, full=False):
    """Native coarse window-start stride; full candidate stages use every row."""
    widths = np.asarray(widths, dtype=np.int32)
    if full or T0_fit_margin <= 0:
        return np.ones_like(widths)
    margin = min(float(T0_fit_margin), .125)
    skip_point = int(1 / margin)
    return np.where(widths > skip_point, widths // skip_point, 1).astype(np.int32)


def _running_median(data, kernel, window_chunk=8192):
    """Native per-window masked medians with bounded temporary allocations.

    Concatenating masked chunk results with ma.concatenate preserves masks
    until the exact native np.append edge-padding operations. Using np.append
    for the intermediate join would prematurely discard masks on some NumPy
    versions and change fully masked-window behavior.
    """
    if not isinstance(window_chunk, (int, np.integer)) or window_chunk < 1:
        raise ValueError('window_chunk must be a positive integer')
    window_count = len(data) - kernel + 1
    offsets = np.arange(kernel)
    pieces = []
    for start in range(0, int(np.ceil(window_count)), window_chunk):
        stops = min(start + window_chunk, window_count)
        index = offsets + np.arange(start, stops)[:, None]
        pieces.append(np.ma.median(data[index.astype(int)], axis=1))
    med = np.ma.concatenate(pieces)
    missing = len(data) - len(med)
    front = int(missing * .5)
    return np.append(np.append(np.full(front, med[0]), med), np.full(missing - front, med[-1]))


def spectrum_kernel_size(oversampling_factor=3, kernel_size=None):
    """Validate the native median-window policy before any GPU execution."""
    if kernel_size is None:
        width = float(oversampling_factor) * 30
        if not np.isfinite(width) or width < 1 or width != np.floor(width):
            raise ValueError('30 * oversampling_factor must be a positive integer '
                             'for the native SDE window; provide an integer '
                             'sde_kernel_size for another oversampling factor')
        kernel_size = int(width)
    elif not isinstance(kernel_size, (int, np.integer)) or kernel_size < 1:
        raise ValueError('kernel_size must be a positive integer or None')
    return kernel_size + (kernel_size % 2 == 0)


def native_spectra(chi2, oversampling_factor=3, mask_outliers=True, kernel_size=None):
    """Pinned GTLS spectrum arithmetic and max-detrended-power primary rank.

    Preserve input dtype (the GPU reference returns float32 residuals) and the
    masked-array arithmetic. No replacement of unsupported/degenerate scores
    by invented detections is performed. Full-mode stages preserve the current
    mask and pass mask_outliers=False when recomputing this spectrum.
    An explicit positive integer kernel_size changes the detection statistic;
    even values are increased by one. None preserves the native default.
    """
    kernel = spectrum_kernel_size(oversampling_factor, kernel_size)
    raw_input = np.asanyarray(chi2).copy()
    chi2 = np.ma.array(raw_input, copy=False)
    if mask_outliers:
        mask = raw_input > (100 * np.median(raw_input))
        chi2 = np.ma.array(raw_input, mask=np.ma.getmaskarray(raw_input) | np.ma.filled(mask, True))
    with np.errstate(divide='ignore', invalid='ignore'):
        SR = np.min(chi2) / chi2
        SDE_raw = (1 - np.mean(SR)) / np.std(SR)
        power_raw = SR - np.mean(SR)
        power_raw = power_raw * (SDE_raw / np.max(power_raw))
        if len(power_raw) > 2 * kernel:
            power = power_raw - _running_median(power_raw, kernel)
            power = power - np.mean(power)
            SDE = np.max(power / np.std(power))
            power = power * (SDE / np.max(power))
        else:
            power, SDE = power_raw, SDE_raw
    finite_power = np.ma.filled(power, np.nan)
    primary = int(np.nanargmax(finite_power)) if np.any(np.isfinite(finite_power)) else None
    return dict(chi2=chi2, SR=SR, power_raw=power_raw, power=power,
                SDE_raw=SDE_raw, SDE=SDE, primary_index=primary)


def refinement_candidate_indices(periods, power):
    """Rank valid candidates: top100, then next100 at P>1d.

    Filtering before the stable sort fixes a native GTLS host-mask defect.
    Its masked scalars do not form a total ordering and can enter the top100
    period list as NaN, leading to undefined GPU integer conversions. Only
    finite, unmasked scores and periods represent physical first-stage trials.
    The native rank policy and tie order are unchanged on valid entries.
    """
    periods, power = np.ma.asarray(periods), np.ma.asarray(power)
    valid = (~np.ma.getmaskarray(periods) & ~np.ma.getmaskarray(power) &
             np.isfinite(np.ma.getdata(periods)) &
             np.isfinite(np.ma.getdata(power)))
    combined = [(i, (periods.data[i], -power.data[i]))
                for i in np.flatnonzero(valid)]
    ranked = sorted(combined, key=lambda item: item[1][1])
    top = [item[0] for item in ranked[:100]]
    remaining = [item for item in combined if item[0] not in top and item[1][0] > 1]
    next_best = sorted(remaining, key=lambda item: item[1][1])[:100]
    return np.array(top + [item[0] for item in next_best], dtype=np.int64)


def harmonic_candidate_indices(periods, primary_period):
    """Nearest existing periods to [0.5,1,2,2/3,3/2] times the chosen period."""
    # Native find_nearest_indices explicitly converts masked periods to an
    # ordinary ndarray before argmin, so masked grid rows can be selected here.
    periods = np.array(periods)
    return np.array([np.argmin(np.abs(periods - primary_period * scale))
                     for scale in (.5, 1., 2., 2./3, 3./2)], dtype=np.int64)


def _native_transit_times(T0, t, period):
    times = [T0 + period] if T0 < min(t) else [T0]
    previous = times[0]
    while True:
        following = previous + period
        if following < (np.min(t) + (np.max(t) - np.min(t))):
            times.append(following)
            previous = following
        else:
            return np.array(times)


def _native_duration_days(t, period, start_epoch, raw_duration):
    shifted = start_epoch + period / 2
    phases = (t - shifted) / period - np.floor((t - shifted) / period)
    sorted_phases = phases[np.argsort(phases)]
    first = np.argmin(np.abs(sorted_phases - .5))
    last = first + int(np.array(raw_duration) * len(t))
    if not 0 <= last < len(t):
        raise ValueError('Native final duration estimate exceeds the sorted phase array')
    duration = (sorted_phases[last] - .5) * period
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Native final duration estimate is nonpositive')
    return duration


def final_parameters(t, y, dy, period, cache, width_index, epoch_index,
                     exposure_days=None, fit_chi2=None, error_scale=1.):
    """CPU postprocessing for the winning no-skip GTLS sample window.

    t/y/dy must already be preprocessed. width_index addresses cache['widths'];
    epoch_index is the start row in the phase-sorted sample array. Native GTLS
    recomputes its depth in float64 after GPU selection; this helper does too.

    duration/T0/depth follow native sample-window conventions. SNR preserves
    cuvarbase's sqrt(delta chi-squared) definition in original error units,
    using error_scale from preprocess_inputs. The historically inconsistent
    native GTLS SNR mask is reproduced only as native_gtls_snr, never as SNR.
    Exposure information is diagnostic; neither cache nor fit integrates it.
    """
    t, y, dy = [np.asarray(value, dtype=np.float64) for value in (t, y, dy)]
    if t.ndim != 1 or y.shape != t.shape or dy.shape != t.shape or len(t) < 3:
        raise ValueError('t, y and dy must be aligned 1D arrays')
    if not np.isfinite(period) or period <= 0 or not np.isfinite(error_scale) or error_scale <= 0:
        raise ValueError('period and error_scale must be finite and positive')
    if not 0 <= int(width_index) < len(cache['widths']) or not 0 <= int(epoch_index) < len(t):
        raise ValueError('Winning width/epoch index is out of range')
    width_index, epoch_index = int(width_index), int(epoch_index)
    width = int(cache['widths'][width_index])
    if width > len(t):
        raise ValueError('Winning sample window exceeds one folded light curve')
    row = cache['overview'][int(cache['unique_indices'][width_index])]
    raw_duration = float(row['duration'])
    # Match core.foldCPU, which deliberately recomputes the phase order here.
    rank = np.argsort((t % period) / period)
    sorted_time, sorted_flux = t[rank], y[rank]
    window_rows = (np.arange(width) + epoch_index) % len(t)
    window_flux = sorted_flux[window_rows]
    mean = window_flux.mean()
    depth = ((1 - mean) * row['overshoot']).item()
    first_time = sorted_time[epoch_index]
    start_epoch = first_time - int((first_time - min(t)) / period) * period - period
    duration = float(_native_duration_days(t, period, start_epoch, raw_duration))
    predicted_times = _native_transit_times(start_epoch, t, period) + duration / 2
    T0 = start_epoch + duration / 2
    if T0 < min(t):
        T0 += period

    unit_model = np.ones(len(t))
    deficit = np.asarray(cache['template_deficits'][width_index, :width], dtype=np.float64)
    unit_model[rank[window_rows]] -= deficit * (depth * 2.)
    chi2_null = float(np.sum(((1 - y) / dy)**2))
    chi2_cpu_model = float(np.sum(((unit_model - y) / dy)**2))
    chi2_fit = chi2_cpu_model if fit_chi2 is None else float(fit_chi2)
    delta_chi2 = (chi2_null - chi2_fit) / float(error_scale)**2
    snr = np.sqrt(max(0., delta_chi2)) if np.isfinite(delta_chi2) else np.nan

    odd, even, per_event = [], [], []
    for i, epoch in enumerate(predicted_times):
        inside = (t > epoch - duration / 2) & (t < epoch + duration / 2)
        per_event.append(int(np.sum(inside)))
        (even if i % 2 == 0 else odd).extend(y[inside])
    all_intransit = np.concatenate((np.array(odd), np.array(even)))
    # Preserve the pinned expression only under an explicit native diagnostic:
    # core passes fractional raw_duration to a mask accepting duration in days.
    native_mask = np.abs((t - T0 + .5 * period) % period - .5 * period) < raw_duration
    native_ootr = y[~native_mask]
    if len(all_intransit) and len(native_ootr) and np.std(native_ootr) > 0:
        native_snr = ((1 - np.mean(all_intransit)) / np.std(native_ootr)) * len(all_intransit)**.5
    else:
        native_snr = np.nan

    exposure = dict(integrated_in_search=False, supplied=exposure_days is not None)
    if exposure_days is not None:
        values = np.asarray(exposure_days, dtype=np.float64)
        if values.ndim > 1 or (values.ndim == 1 and len(values) != len(t)) or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('exposure_days must be a finite nonnegative scalar or aligned array')
        exposure.update(minimum_days=float(np.min(values)), maximum_days=float(np.max(values)),
                        median_days=float(np.median(values)))
    return dict(period=float(period), T0=float(T0),
                t0_phase=float(((T0 - np.floor(np.min(t))) / period) % 1.),
                duration=duration, depth=float(depth), fractional_duration=raw_duration,
                width_in_samples=width, width_index=width_index, epoch_index=epoch_index,
                chi2_min=chi2_fit, chi2_null=chi2_null, chi2_cpu_model=chi2_cpu_model,
                chi2_error_scale=float(error_scale), delta_chi2=delta_chi2, SNR=float(snr),
                native_gtls_snr=float(native_snr), n_transits=len(predicted_times),
                observed_transits=sum(v > 0 for v in per_event), transit_times=predicted_times,
                per_transit_count=np.array(per_event, dtype=np.int64), exposure=exposure)
