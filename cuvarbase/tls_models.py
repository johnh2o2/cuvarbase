"""
Transit model generation for TLS.

This module handles creation of physically realistic transit light curves
using the Batman package for limb-darkened transits.

References
----------
.. [1] Kreidberg (2015), "batman: BAsic Transit Model cAlculatioN in Python",
       PASP 127, 1161
.. [2] Mandel & Agol (2002), "Analytic Light Curves for Planetary Transit
       Searches", ApJ 580, L171
"""

import numpy as np
try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    import warnings
    warnings.warn("batman package not available. Install with: pip install batman-package")


def create_reference_transit(n_samples=1000, limb_dark='quadratic',
                             u=[0.4804, 0.1867]):
    """
    Create a reference transit model normalized to Earth-like transit.

    This generates a high-resolution transit template that can be scaled
    and interpolated for different durations and depths.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples in the model (default: 1000)
    limb_dark : str, optional
        Limb darkening law (default: 'quadratic')
        Options: 'uniform', 'linear', 'quadratic', 'nonlinear'
    u : list, optional
        Limb darkening coefficients (default: [0.4804, 0.1867])
        Default values are for Sun-like star in Kepler bandpass

    Returns
    -------
    phases : ndarray
        Phase values (0 to 1)
    flux : ndarray
        Normalized flux (1.0 = out of transit, <1.0 = in transit)

    Notes
    -----
    The reference model assumes:
    - Period = 1.0 (arbitrary units, we work in phase)
    - Semi-major axis = 1.0 (normalized)
    - Planet-to-star radius ratio scaled to produce unit depth
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman package required for transit models. "
                         "Install with: pip install batman-package")

    # Batman parameters for reference transit
    params = batman.TransitParams()

    # Fixed parameters (Earth-like)
    params.t0 = 0.0                   # Mid-transit time
    params.per = 1.0                  # Period (arbitrary, we use phase)
    params.rp = 0.1                   # Planet-to-star radius ratio (will normalize)
    params.a = 15.0                   # Semi-major axis in stellar radii (typical)
    params.inc = 90.0                 # Inclination (degrees) - edge-on
    params.ecc = 0.0                  # Eccentricity - circular
    params.w = 90.0                   # Longitude of periastron
    params.limb_dark = limb_dark      # Limb darkening model
    params.u = u                      # Limb darkening coefficients

    # Create time array spanning the transit
    # For a = 15, duration is approximately 0.05 in phase units
    # We'll create a grid from -0.1 to 0.1 (well beyond transit)
    t = np.linspace(-0.15, 0.15, n_samples)

    # Generate model
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    # Normalize: shift so out-of-transit = 1.0, in-transit depth = 1.0 at center
    flux_oot = flux[0]  # Out of transit flux
    depth = flux_oot - np.min(flux)  # Transit depth

    if depth < 1e-10:
        raise ValueError("Transit depth too small - check parameters")

    flux_normalized = (flux - flux_oot) / depth + 1.0

    # Convert time to phase (0 to 1)
    phases = (t - t[0]) / (t[-1] - t[0])

    return phases, flux_normalized


def create_transit_model_cache(durations, period=1.0, n_samples=1000,
                               limb_dark='quadratic', u=[0.4804, 0.1867],
                               R_star=1.0, M_star=1.0):
    """
    Create cache of transit models for different durations.

    Parameters
    ----------
    durations : array_like
        Array of transit durations (days) to cache
    period : float, optional
        Reference period (days) - used for scaling (default: 1.0)
    n_samples : int, optional
        Number of samples per model (default: 1000)
    limb_dark : str, optional
        Limb darkening law (default: 'quadratic')
    u : list, optional
        Limb darkening coefficients (default: [0.4804, 0.1867])
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)

    Returns
    -------
    models : list of ndarray
        List of flux arrays for each duration
    phases : ndarray
        Phase array (same for all models)

    Notes
    -----
    This creates models at different durations by adjusting the semi-major
    axis in the batman model to produce the desired transit duration.
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman package required for transit models")

    durations = np.asarray(durations)
    models = []

    for duration in durations:
        # Create batman parameters
        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = period
        params.rp = 0.1  # Will be scaled later
        params.inc = 90.0
        params.ecc = 0.0
        params.w = 90.0
        params.limb_dark = limb_dark
        params.u = u

        # Calculate semi-major axis to produce desired duration
        # T_14 ≈ (P/π) * arcsin(R_star/a) for edge-on transit
        # Approximation: a ≈ R_star * P / (π * duration)
        a = R_star * period / (np.pi * duration)
        params.a = max(a, 1.5)  # Ensure a > R_star + R_planet

        # Create time array
        t = np.linspace(-0.15, 0.15, n_samples)

        # Generate model
        m = batman.TransitModel(params, t)
        flux = m.light_curve(params)

        # Normalize
        flux_oot = flux[0]
        depth = flux_oot - np.min(flux)

        if depth < 1e-10:
            # If depth is too small, use reference model
            phases, flux_normalized = create_reference_transit(
                n_samples, limb_dark, u)
        else:
            flux_normalized = (flux - flux_oot) / depth + 1.0
            phases = (t - t[0]) / (t[-1] - t[0])

        models.append(flux_normalized.astype(np.float32))

    return models, phases.astype(np.float32)


def simple_trapezoid_transit(phases, duration_phase, depth=1.0,
                             ingress_duration=0.1):
    """
    Create a simple trapezoidal transit model (fast, no Batman needed).

    This is a simplified model for testing or when Batman is not available.

    Parameters
    ----------
    phases : array_like
        Phase values (0 to 1)
    duration_phase : float
        Total transit duration in phase units
    depth : float, optional
        Transit depth (default: 1.0)
    ingress_duration : float, optional
        Ingress/egress duration as fraction of total duration (default: 0.1)

    Returns
    -------
    flux : ndarray
        Flux values (1.0 = out of transit)

    Notes
    -----
    This creates a trapezoid with linear ingress/egress. It's much faster
    than Batman but less physically accurate (no limb darkening).
    """
    phases = np.asarray(phases)
    flux = np.ones_like(phases, dtype=np.float32)

    # Calculate ingress/egress duration
    t_ingress = duration_phase * ingress_duration
    t_flat = duration_phase * (1.0 - 2.0 * ingress_duration)

    # Transit centered at phase = 0.5
    t1 = 0.5 - duration_phase / 2.0  # Start of ingress
    t2 = t1 + t_ingress               # Start of flat bottom
    t3 = t2 + t_flat                  # Start of egress
    t4 = t3 + t_ingress               # End of transit

    # Ingress
    mask_ingress = (phases >= t1) & (phases < t2)
    flux[mask_ingress] = 1.0 - depth * (phases[mask_ingress] - t1) / t_ingress

    # Flat bottom
    mask_flat = (phases >= t2) & (phases < t3)
    flux[mask_flat] = 1.0 - depth

    # Egress
    mask_egress = (phases >= t3) & (phases < t4)
    flux[mask_egress] = 1.0 - depth * (t4 - phases[mask_egress]) / t_ingress

    return flux


def interpolate_transit_model(model_phases, model_flux, target_phases,
                              target_depth=1.0):
    """
    Interpolate a transit model to new phase grid and scale depth.

    Parameters
    ----------
    model_phases : array_like
        Phase values of the template model
    model_flux : array_like
        Flux values of the template model
    target_phases : array_like
        Desired phase values for interpolation
    target_depth : float, optional
        Desired transit depth (default: 1.0)

    Returns
    -------
    flux : ndarray
        Interpolated and scaled flux values

    Notes
    -----
    Uses linear interpolation. For GPU implementation, texture memory
    with hardware interpolation would be faster.
    """
    # Interpolate to target phases
    flux_interp = np.interp(target_phases, model_phases, model_flux)

    # Scale depth: current depth is (1.0 - min(model_flux))
    current_depth = 1.0 - np.min(model_flux)

    if current_depth < 1e-10:
        return flux_interp

    # Scale: flux = 1 - target_depth * (1 - flux_normalized)
    flux_scaled = 1.0 - target_depth * (1.0 - flux_interp)

    return flux_scaled.astype(np.float32)


def get_default_limb_darkening(filter='Kepler', T_eff=5500):
    """
    Get default limb darkening coefficients for common filters and T_eff.

    Parameters
    ----------
    filter : str, optional
        Filter name: 'Kepler', 'TESS', 'Johnson_V', etc. (default: 'Kepler')
    T_eff : float, optional
        Effective temperature (K) (default: 5500)

    Returns
    -------
    u : list
        Quadratic limb darkening coefficients [u1, u2]

    Notes
    -----
    These are approximate values. For precise work, calculate coefficients
    for your specific stellar parameters using packages like ldtk.

    Values from Claret & Bloemen (2011), A&A 529, A75
    """
    # Simple lookup table for common cases
    # Format: {filter: {T_eff_range: [u1, u2]}}

    if filter == 'Kepler':
        if T_eff < 4500:
            return [0.7, 0.1]  # Cool stars
        elif T_eff < 6000:
            return [0.4804, 0.1867]  # Solar-type
        else:
            return [0.3, 0.2]  # Hot stars

    elif filter == 'TESS':
        if T_eff < 4500:
            return [0.5, 0.2]
        elif T_eff < 6000:
            return [0.3, 0.3]
        else:
            return [0.2, 0.3]

    else:
        # Default to Solar-type in Kepler
        return [0.4804, 0.1867]


def validate_limb_darkening_coeffs(u, limb_dark='quadratic'):
    """
    Validate limb darkening coefficients are physically reasonable.

    Parameters
    ----------
    u : list
        Limb darkening coefficients
    limb_dark : str
        Limb darkening law

    Raises
    ------
    ValueError
        If coefficients are unphysical
    """
    u = np.asarray(u)

    if limb_dark == 'quadratic':
        if len(u) != 2:
            raise ValueError("Quadratic limb darkening requires 2 coefficients")
        # Physical constraints: 0 < u1 + u2 < 1, u1 > 0, u1 + 2*u2 > 0
        if not (0 < u[0] + u[1] < 1):
            raise ValueError(f"u1 + u2 = {u[0] + u[1]} must be in (0, 1)")
        if not (u[0] > 0):
            raise ValueError(f"u1 = {u[0]} must be > 0")
        if not (u[0] + 2*u[1] > 0):
            raise ValueError(f"u1 + 2*u2 = {u[0] + 2*u[1]} must be > 0")

    elif limb_dark == 'linear':
        if len(u) != 1:
            raise ValueError("Linear limb darkening requires 1 coefficient")
        if not (0 < u[0] < 1):
            raise ValueError(f"u = {u[0]} must be in (0, 1)")
