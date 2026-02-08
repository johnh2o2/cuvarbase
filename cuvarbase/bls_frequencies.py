"""
Frequency grid utilities for BLS transit searches.

Provides Keplerian-aware frequency grids (Ofir 2014) that exploit the
physical relationship between orbital period and transit duration to
minimize the number of trial frequencies while maintaining sensitivity.
"""
import numpy as np


def _q_transit(freq, rho=1.0):
    """
    Keplerian transit duration fraction q = T_dur / P.

    For a central transit of a planet on a circular orbit:
        q = arcsin((f / f_max0)^(2/3)) / pi

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


def keplerian_freq_grid(period_min, period_max, baseline,
                        R_star=1.0, M_star=1.0, oversampling=2):
    """
    Generate a non-uniform frequency grid optimized for transit detection.

    Transit duration scales as T_dur ~ P^(1/3) (Kepler's third law),
    so the required frequency resolution scales as df ~ q(f) / (T * oversampling)
    where q(f) is the transit duration fraction at frequency f. This gives
    fewer frequencies at low frequencies (long periods) where transits are
    longer and the resolution requirement is coarser.

    Based on the frequency spacing in Ofir (2014) and consistent with
    cuvarbase.bls.transit_autofreq.

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

    Returns
    -------
    freqs : ndarray, float32
        Non-uniform frequency array (1/days), sorted ascending.
    """
    # Mean stellar density in solar units
    rho = M_star / (R_star ** 3)

    f_min = 1.0 / period_max
    f_max = 1.0 / period_min

    T = baseline

    freqs = [f_min]
    while freqs[-1] < f_max:
        q = float(_q_transit(freqs[-1], rho=rho))
        # Minimum q to avoid zero step
        q = max(q, 1e-6)
        df = q / (oversampling * T)
        freqs.append(freqs[-1] + df)

    freqs = np.array(freqs, dtype=np.float32)

    # Trim to exact range
    freqs = freqs[freqs <= f_max * 1.001]

    return freqs


def uniform_freq_grid(period_min, period_max, baseline, oversampling=2):
    """
    Generate a uniform frequency grid for BLS.

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

    Returns
    -------
    freqs : ndarray, float32
        Uniform frequency array (1/days).
    """
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    df = 1.0 / (baseline * oversampling)
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

    uniform_nf = int(np.ceil((freqs[-1] - freqs[0]) * baseline * 2))

    return {
        'nfreq': nf,
        'f_min': float(freqs[0]),
        'f_max': float(freqs[-1]),
        'period_min': float(periods[-1]),
        'period_max': float(periods[0]),
        'df_min': float(df.min()),
        'df_max': float(df.max()),
        'df_ratio': float(df.max() / df.min()),
        'uniform_nfreq': uniform_nf,
        'reduction_factor': uniform_nf / nf if nf > 0 else 0,
    }
