"""
Frequency grid utilities for BLS transit searches.

Provides Keplerian-aware frequency grids that exploit the physical
relationship between orbital period and transit duration to minimize the
number of trial frequencies while maintaining sensitivity.

The transit-duration/period relation is Seager & Mallen-Ornelas (2003),
ApJ 585, 1038, "A Unique Solution of Planet and Star Parameters from an
Extrasolar Planet Transit Light Curve" (eq. 3). The duty-cycle-based
frequency spacing is Ofir (2014), A&A 561, A138, "Optimizing the search
for transiting planets in long time series" (eq. 4; arXiv:1307.7330).
Consistent with :func:`cuvarbase.bls.transit_autofreq`.
"""
import numpy as np


def _q_transit(freq, rho=1.0):
    """
    Keplerian transit duration fraction q = T_dur / P.

    For a central transit (impact parameter 0) of a planet on a circular
    orbit, Seager & Mallen-Ornelas (2003) eq. (3) reduces to::

        q = arcsin((f / f_max0)^(2/3)) / pi

    where ``f_max0 = sqrt(G rho_star / 3pi)`` is the surface-orbit
    frequency (``8.6307 * sqrt(rho)`` cycles/day, a derived constant --
    see :func:`cuvarbase.bls.fmax_transit0`).

    Parameters
    ----------
    freq : float or array_like
        Orbital frequency (1/days).
    rho : float
        Mean stellar density in solar units.

    Returns
    -------
    q : float or array_like
        Transit duration fraction.
    """
    fmax0 = 8.6307 * np.sqrt(rho)
    f23 = np.minimum(1.0, np.power(freq / fmax0, 2.0 / 3.0))
    return np.arcsin(f23) / np.pi


_GRID_METHODS = ('vectorized', 'recursion')


def _validate_grid_method(method):
    if method not in _GRID_METHODS:
        raise ValueError("grid method must be one of %r, got %r"
                         % (list(_GRID_METHODS), method))


def _dq_transit(freq, fmax0):
    r"""``d q / d f`` for :func:`_q_transit`, zero where ``q`` saturates.

    With :math:`x = (f/f_0)^{2/3}` and :math:`q = \arcsin(x)/\pi`,
    :math:`dq/df = 2x / (3 \pi f \sqrt{1 - x^2})`.
    """
    freq = np.asarray(freq, dtype=np.float64)
    x = np.power(freq / fmax0, 2.0 / 3.0)
    capped = ~(x < 1.0)
    x = np.minimum(1.0, x)
    with np.errstate(divide='ignore', invalid='ignore'):
        dq = (2.0 / (3.0 * np.pi * freq)) * x / np.sqrt(1.0 - x * x)
    return np.where(capped | ~np.isfinite(dq), 0.0, dq)


def _euler_transit_grid(fmin, fmax, num_fac, denom, fmax0, rho=1.0,
                        q_floor=0.0, tol=1e-15, max_iter=8,
                        seed_points=4096):
    r"""The Ofir (2014) duty-cycle frequency recursion, vectorized.

    Returns the frequencies of

    .. math:: f_{n+1} = f_n + a\,\max(q(f_n),\,q_{\rm floor}) / b

    (``a = num_fac``, ``b = denom``) from ``f_0 = fmin`` up to and
    including the first point at or above ``fmax`` -- exactly what the
    scalar ``while`` loop this replaces builds, but without a Python
    iteration per frequency (0.3-1.0 s at survey grid sizes, dwarfing
    the GPU search itself; Sep 2026 audit, ids 4 and 44).

    Method: seed with the continuum solution of ``df/dn = a q(f)/b``
    (cumulative trapezoid of its reciprocal in the variable
    :math:`u = f^{1/3}`, where the integrand is smooth at both ends,
    inverted with ``np.interp``), then apply defect correction. Writing
    the residual of the recursion as
    ``d_n = f_n + a q(f_n)/b - f_{n+1}``, the error obeys the linear
    recursion ``e_{n+1} = (1 + a q'(f_n)/b) e_n + d_n``, whose solution
    is a ``cumprod``/``cumsum`` pair -- so each correction pass is O(N)
    numpy work, and two to four passes drive the residual to float64
    rounding.

    This converges to a fixed point of the same recursion, not to an
    approximation of it: measured against the scalar loop over
    ZTF/HAT/TESS/Kepler baselines and ``rho`` in [0.05, 5], it
    reproduces the grid length exactly and every frequency to <= 4e-15
    relative -- and bitwise once cast to the float32
    :func:`keplerian_freq_grid` returns.
    """
    fmin = float(fmin)
    fmax = float(fmax)
    num_fac = float(num_fac)
    denom = float(denom)
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin <= 0:
        raise ValueError("frequency grid needs finite bounds with "
                         "fmin > 0; got fmin=%r fmax=%r" % (fmin, fmax))
    if not np.isfinite(denom) or denom <= 0 or num_fac <= 0:
        raise ValueError("frequency grid step must be positive; got "
                         "num_fac=%r denom=%r" % (num_fac, denom))
    if fmin >= fmax:
        return np.array([fmin], dtype=np.float64)

    step_scale = num_fac / denom

    def _q(f):
        q = _q_transit(f, rho=rho)
        return np.maximum(q, q_floor) if q_floor > 0 else q

    def _step(f):
        # exactly the scalar loop's expression, elementwise
        return (num_fac * _q(f)) / denom

    def _dstep(f):
        dq = _dq_transit(f, fmax0)
        if q_floor > 0:
            dq = np.where(_q_transit(f, rho=rho) < q_floor, 0.0, dq)
        return step_scale * dq

    # --- seed: invert n(f) = int df / step(f) ---
    top = fmax
    for _ in range(64):
        u = np.linspace(fmin ** (1. / 3.), top ** (1. / 3.),
                        int(seed_points))
        fa = u ** 3
        g = 3.0 * u * u / np.maximum(_step(fa), 1e-300)
        nn = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1])
                                              * np.diff(u))))
        ntot = int(np.floor(np.interp(fmax, fa, nn))) + 8
        f = np.interp(np.arange(ntot + 1, dtype=np.float64), nn, fa)
        f[0] = fmin
        if f[-1] > fmax:
            break
        top = top + max(top - fmin, 1e-12)
    else:  # pragma: no cover - unreachable for finite, positive bounds
        raise RuntimeError("could not bracket the frequency grid "
                           "(fmin=%r fmax=%r)" % (fmin, fmax))

    # --- defect correction ---
    scale = max(abs(fmax), abs(fmin))
    for _ in range(int(max_iter)):
        d = f[:-1] + _step(f[:-1]) - f[1:]
        logp = np.concatenate(([0.0], np.cumsum(np.log1p(_dstep(f[:-1])))))
        p = np.exp(logp)
        e = np.concatenate(([0.0], p[1:] * np.cumsum(d / p[1:])))
        f = f + e
        f[0] = fmin
        if np.max(np.abs(e)) <= tol * scale:
            break

    idx = int(np.searchsorted(f, fmax, side='left'))
    return f[:idx + 1]


def _recursion_transit_grid(fmin, fmax, num_fac, denom, rho=1.0,
                            q_floor=0.0):
    """The same recursion as :func:`_euler_transit_grid`, run as the
    scalar Python loop: the reference implementation, and what
    ``method='recursion'`` selects."""
    freqs = [float(fmin)]
    while freqs[-1] < fmax:
        q = float(_q_transit(freqs[-1], rho=rho))
        if q_floor > 0:
            q = max(q, q_floor)
        freqs.append(freqs[-1] + (num_fac * q) / denom)
    return np.array(freqs, dtype=np.float64)


def keplerian_freq_grid(period_min, period_max, baseline, *,
                        R_star=1.0, M_star=1.0, oversampling=2,
                        return_qvals=False, method='vectorized'):
    """
    Generate a non-uniform frequency grid optimized for transit detection.

    Transit duration scales as T_dur ~ P^(1/3) (Kepler's third law),
    so the required frequency resolution scales as df ~ q(f) / (T * oversampling)
    where q(f) is the transit duration fraction at frequency f. This gives
    fewer frequencies at low frequencies (long periods) where transits are
    longer and the resolution requirement is coarser.

    This is the duty-cycle-based spacing of Ofir (2014), A&A 561, A138,
    eq. (4) (``df = q(f) / (oversampling * T)``), consistent with
    :func:`cuvarbase.bls.transit_autofreq`.

    Parameters
    ----------
    period_min : float
        Minimum period to search (days).
    period_max : float
        Maximum period to search (days).
    baseline : float
        Total observation baseline (days).
    R_star : float, optional (default: 1.0)
        Stellar radius in solar radii. Used to compute stellar density.
    M_star : float, optional (default: 1.0)
        Stellar mass in solar masses. Used to compute stellar density.
    oversampling : float, optional (default: 2)
        Oversampling factor. Higher values give denser grids.
    return_qvals : bool, optional (default: False)
        Also return the Keplerian transit duration fraction q at each
        frequency. Pass e.g. ``qmin=0.5 * qvals, qmax=2.0 * qvals`` to
        :func:`cuvarbase.bls.eebls_gpu_batch` for a duration-
        constrained search (the batch kernel supports per-frequency
        q bounds).
    method : str, optional (default: ``'vectorized'``)
        How to evaluate the spacing recursion. ``'vectorized'`` solves
        it with numpy (10-30x faster; agrees with the loop to <= 4e-15
        relative in float64 and bitwise in the float32 returned here).
        ``'recursion'`` runs the original scalar Python loop, one
        ``q`` evaluation per frequency.

        .. versionadded:: 1.0

    Returns
    -------
    freqs : ndarray, float32
        Non-uniform frequency array (1/days), sorted ascending.
    qvals : ndarray, float32
        Keplerian q at each frequency (only if ``return_qvals=True``).
    """
    # Mean stellar density in solar units
    rho = M_star / (R_star ** 3)

    f_min = 1.0 / period_max
    f_max = 1.0 / period_min

    T = baseline

    _validate_grid_method(method)
    # ``q`` is floored at 1e-6 to avoid a zero step at f -> 0.
    if method == 'recursion':
        freqs = _recursion_transit_grid(f_min, f_max, 1.0,
                                        oversampling * T, rho=rho,
                                        q_floor=1e-6)
    else:
        freqs = _euler_transit_grid(f_min, f_max, 1.0, oversampling * T,
                                    8.6307 * np.sqrt(rho), rho=rho,
                                    q_floor=1e-6)

    freqs = np.array(freqs, dtype=np.float32)

    # Trim to exact range
    freqs = freqs[freqs <= f_max * 1.001]

    if return_qvals:
        qvals = _q_transit(freqs.astype(np.float64),
                           rho=rho).astype(np.float32)
        return freqs, qvals

    return freqs


def uniform_freq_grid(period_min, period_max, baseline, *, oversampling=2,
                      R_star=1.0, M_star=1.0):
    """
    Generate a uniform frequency grid matched to Keplerian sensitivity.

    Uses the finest resolution needed by the Keplerian grid (at the lowest
    frequency / longest period) as the uniform spacing. This gives a fair
    comparison: both grids detect the same transits, but the uniform grid
    wastes resolution at high frequencies where coarser spacing would suffice.

    Parameters
    ----------
    period_min : float
        Minimum period (days).
    period_max : float
        Maximum period (days).
    baseline : float
        Total observation baseline (days).
    oversampling : float, optional (default: 2)
        Oversampling factor.
    R_star : float, optional (default: 1.0)
        Stellar radius in solar radii.
    M_star : float, optional (default: 1.0)
        Stellar mass in solar masses.

    Returns
    -------
    freqs : ndarray, float32
        Uniform frequency array (1/days).
    """
    rho = M_star / (R_star ** 3)
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min

    # Use the finest resolution needed (at lowest frequency)
    q_min_freq = float(_q_transit(f_min, rho=rho))
    q_min_freq = max(q_min_freq, 1e-6)
    df = q_min_freq / (oversampling * baseline)

    nf = int(np.ceil((f_max - f_min) / df))
    return np.linspace(f_min, f_max, max(nf, 1)).astype(np.float32)


def freq_grid_stats(freqs, baseline):
    """
    Compute summary statistics for a frequency grid.

    Parameters
    ----------
    freqs : ndarray
        Frequency array.
    baseline : float
        Observation baseline (days).

    Returns
    -------
    stats : dict
        Dictionary with grid statistics.
    """
    nf = len(freqs)
    df = np.diff(freqs)
    periods = 1.0 / freqs

    # Sensitivity-matched uniform grid: use finest df in this grid
    df_min = float(df.min())
    uniform_nf = int(np.ceil((freqs[-1] - freqs[0]) / df_min))

    return {
        'nfreq': nf,
        'f_min': float(freqs[0]),
        'f_max': float(freqs[-1]),
        'period_min': float(periods[-1]),
        'period_max': float(periods[0]),
        'df_min': df_min,
        'df_max': float(df.max()),
        'df_ratio': float(df.max() / df.min()),
        'uniform_nfreq': uniform_nf,
        'reduction_factor': uniform_nf / nf if nf > 0 else 0,
    }
