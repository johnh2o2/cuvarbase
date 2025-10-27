"""
Period and duration grid generation for Transit Least Squares.

Implements the Ofir (2014) optimal frequency sampling algorithm and
logarithmically-spaced duration grids based on stellar parameters.

References
----------
.. [1] Ofir (2014), "Algorithmic Considerations for the Search for
       Continuous Gravitational Waves", A&A 561, A138
.. [2] Hippke & Heller (2019), "Transit Least Squares", A&A 623, A39
"""

import numpy as np


# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
R_sun = 6.95700e8  # Solar radius (m)
M_sun = 1.98840e30  # Solar mass (kg)
R_earth = 6.371e6  # Earth radius (m)


def transit_duration_max(period, R_star=1.0, M_star=1.0, R_planet=1.0):
    """
    Calculate maximum transit duration for circular orbit.

    Parameters
    ----------
    period : float or array_like
        Orbital period in days
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)
    R_planet : float, optional
        Planet radius in Earth radii (default: 1.0)

    Returns
    -------
    duration : float or array_like
        Maximum transit duration in days (for edge-on circular orbit)

    Notes
    -----
    Formula: T_14 = (R_star + R_planet) * (4 * P / (π * G * M_star))^(1/3)

    Assumes:
    - Circular orbit (e = 0)
    - Edge-on configuration (i = 90°)
    - Planet + stellar radii contribute to transit chord
    """
    period_sec = period * 86400.0  # Convert to seconds
    R_total = R_star * R_sun + R_planet * R_earth  # Total radius in meters
    M_star_kg = M_star * M_sun  # Mass in kg

    # Duration in seconds
    duration_sec = R_total * (4.0 * period_sec / (np.pi * G * M_star_kg))**(1.0/3.0)

    # Convert to days
    duration_days = duration_sec / 86400.0

    return duration_days


def period_grid_ofir(t, R_star=1.0, M_star=1.0, oversampling_factor=3,
                     period_min=None, period_max=None, n_transits_min=2):
    """
    Generate optimal period grid using Ofir (2014) algorithm.

    This creates a non-uniform period grid that optimally samples the
    period space, with denser sampling at shorter periods where transit
    durations are shorter.

    Parameters
    ----------
    t : array_like
        Observation times (days)
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)
    oversampling_factor : float, optional
        Oversampling factor for period grid (default: 3)
        Higher values give denser grids
    period_min : float, optional
        Minimum period to search (days). If None, calculated from
        Roche limit and minimum transits
    period_max : float, optional
        Maximum period to search (days). If None, set to half the
        total observation span
    n_transits_min : int, optional
        Minimum number of transits required (default: 2)

    Returns
    -------
    periods : ndarray
        Array of trial periods (days)

    Notes
    -----
    Uses the Ofir (2014) frequency-to-cubic transformation:

    f_x = (A/3 * x + C)^3

    where A = (2π)^(2/3) / π * R_star / (G * M_star)^(1/3) * 1/(S * OS)

    This ensures optimal statistical sampling across the period space.
    """
    t = np.asarray(t)
    T_span = np.max(t) - np.min(t)  # Total observation span

    # Store user's requested limits (for filtering later)
    user_period_min = period_min
    user_period_max = period_max

    # Physical boundary conditions (following Ofir 2014 and CPU TLS)
    # f_min: require n_transits_min transits over baseline
    f_min = n_transits_min / (T_span * 86400.0)  # 1/seconds

    # f_max: Roche limit (maximum possible frequency)
    # P_roche = 2π * sqrt(a^3 / (G*M)) where a = 3*R at Roche limit
    R_star_m = R_star * R_sun
    M_star_kg = M_star * M_sun
    f_max = 1.0 / (2.0 * np.pi) * np.sqrt(G * M_star_kg / (3.0 * R_star_m)**3)

    # Ofir (2014) parameters - equations (5), (6), (7)
    T_span_sec = T_span * 86400.0  # Convert to seconds

    # Equation (5): optimal frequency sampling parameter
    A = ((2.0 * np.pi)**(2.0/3.0) / np.pi * R_star_m /
         (G * M_star_kg)**(1.0/3.0) / (T_span_sec * oversampling_factor))

    # Equation (6): offset parameter
    C = f_min**(1.0/3.0) - A / 3.0

    # Equation (7): optimal number of frequency samples
    n_freq = int(np.ceil((f_max**(1.0/3.0) - f_min**(1.0/3.0) + A / 3.0) * 3.0 / A))

    # Ensure we have at least some frequencies
    if n_freq < 10:
        n_freq = 10

    # Linear grid in cubic-root frequency space
    x = np.arange(n_freq) + 1  # 1-indexed like CPU TLS

    # Transform to frequency space (Hz)
    freqs = (A / 3.0 * x + C)**3

    # Convert to periods (days)
    periods = 1.0 / freqs / 86400.0

    # Apply user-requested period limits
    if user_period_min is not None or user_period_max is not None:
        if user_period_min is None:
            user_period_min = 0.0
        if user_period_max is None:
            user_period_max = np.inf

        periods = periods[(periods > user_period_min) & (periods <= user_period_max)]

    # If we somehow got no periods, use simple linear grid
    if len(periods) == 0:
        if user_period_min is None:
            user_period_min = T_span / 20.0
        if user_period_max is None:
            user_period_max = T_span / 2.0
        periods = np.linspace(user_period_min, user_period_max, 100)

    # Sort in increasing order (standard convention)
    periods = np.sort(periods)

    return periods


def duration_grid(periods, R_star=1.0, M_star=1.0, R_planet_min=0.5,
                  R_planet_max=5.0, duration_grid_step=1.1):
    """
    Generate logarithmically-spaced duration grid for each period.

    Parameters
    ----------
    periods : array_like
        Trial periods (days)
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)
    R_planet_min : float, optional
        Minimum planet radius to consider in Earth radii (default: 0.5)
    R_planet_max : float, optional
        Maximum planet radius to consider in Earth radii (default: 5.0)
    duration_grid_step : float, optional
        Multiplicative step for duration grid (default: 1.1)
        1.1 means each duration is 10% larger than previous

    Returns
    -------
    durations : list of ndarray
        List where durations[i] is array of durations for periods[i]
    duration_counts : ndarray
        Number of durations for each period

    Notes
    -----
    Durations are sampled logarithmically from the minimum transit time
    (small planet) to maximum transit time (large planet) for each period.

    The grid spacing ensures we don't miss any transit duration while
    avoiding excessive oversampling.
    """
    periods = np.asarray(periods)

    # Calculate duration bounds for each period
    T_min = transit_duration_max(periods, R_star, M_star, R_planet_min)
    T_max = transit_duration_max(periods, R_star, M_star, R_planet_max)

    durations = []
    duration_counts = np.zeros(len(periods), dtype=np.int32)

    for i, (period, t_min, t_max) in enumerate(zip(periods, T_min, T_max)):
        # Generate logarithmically-spaced durations
        dur = []
        t = t_min
        while t <= t_max:
            dur.append(t)
            t *= duration_grid_step

        # Ensure we include the maximum duration
        if dur[-1] < t_max:
            dur.append(t_max)

        durations.append(np.array(dur, dtype=np.float32))
        duration_counts[i] = len(dur)

    return durations, duration_counts


def t0_grid(period, duration, n_transits=None, oversampling=5):
    """
    Generate grid of T0 (mid-transit time) positions to test.

    Parameters
    ----------
    period : float
        Orbital period (days)
    duration : float
        Transit duration (days)
    n_transits : int, optional
        Number of transits in observation span. If None, assumes
        you want to sample one full period cycle.
    oversampling : int, optional
        Number of T0 positions to test per transit duration (default: 5)

    Returns
    -------
    t0_values : ndarray
        Array of T0 positions (in phase, 0 to 1)

    Notes
    -----
    This creates a grid of phase offsets to test. The spacing is
    determined by the transit duration and oversampling factor.

    For computational efficiency, we typically use stride sampling
    (not every possible phase offset).
    """
    # Phase-space duration
    q = duration / period

    # Step size in phase
    step = q / oversampling

    # Number of steps to cover one full period
    if n_transits is not None:
        n_steps = int(np.ceil(1.0 / (step * n_transits)))
    else:
        n_steps = int(np.ceil(1.0 / step))

    # Grid from 0 to 1 (phase)
    t0_values = np.linspace(0, 1 - step, n_steps, dtype=np.float32)

    return t0_values


def validate_stellar_parameters(R_star=1.0, M_star=1.0,
                                R_star_min=0.13, R_star_max=3.5,
                                M_star_min=0.1, M_star_max=1.0):
    """
    Validate stellar parameters are within reasonable bounds.

    Parameters
    ----------
    R_star : float
        Stellar radius in solar radii
    M_star : float
        Stellar mass in solar masses
    R_star_min, R_star_max : float
        Allowed range for stellar radius
    M_star_min, M_star_max : float
        Allowed range for stellar mass

    Raises
    ------
    ValueError
        If parameters are outside allowed ranges
    """
    if not (R_star_min <= R_star <= R_star_max):
        raise ValueError(f"R_star={R_star} outside allowed range "
                        f"[{R_star_min}, {R_star_max}] solar radii")

    if not (M_star_min <= M_star <= M_star_max):
        raise ValueError(f"M_star={M_star} outside allowed range "
                        f"[{M_star_min}, {M_star_max}] solar masses")


def estimate_n_evaluations(periods, durations, t0_oversampling=5):
    """
    Estimate total number of chi-squared evaluations.

    Parameters
    ----------
    periods : array_like
        Trial periods
    durations : list of array_like
        Duration grids for each period
    t0_oversampling : int
        T0 grid oversampling factor

    Returns
    -------
    n_total : int
        Total number of evaluations (P × D × T0)
    """
    n_total = 0
    for i, period in enumerate(periods):
        n_durations = len(durations[i])
        for duration in durations[i]:
            t0_vals = t0_grid(period, duration, oversampling=t0_oversampling)
            n_total += len(t0_vals)

    return n_total
