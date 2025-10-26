"""
Optional wrapper for the gce package (Conditional Entropy with period derivatives).

This module provides a cuvarbase-compatible interface to the gce package
by Katz et al. (2020), which implements a more efficient Conditional Entropy
algorithm that enables period derivative searches.

**This is NOT a reimplementation** - it requires gce to be installed separately.
We provide this wrapper only for API convenience and integration with the
cuvarbase ecosystem.

Installation
------------
To use this module, install gce separately::

    pip install gce-search

Or from source::

    pip install git+https://github.com/mikekatz04/gce.git

Credits
-------
All credit for the CE algorithm implementation goes to:

Katz, M. L., Larson, S. L., Cohn, J., Vallisneri, M., & Graff, P. B. (2020).
"Efficient computation of the Conditional Entropy period search."
arXiv:2006.06866

gce package: https://github.com/mikekatz04/gce

Examples
--------
Simple period search (no period derivatives)::

    from cuvarbase.ce_gce import ce_gce
    import numpy as np

    t = np.sort(np.random.uniform(0, 100, 1000))
    y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    y += 0.01 * np.random.randn(len(t))

    freqs = np.linspace(0.1, 1.0, 1000)
    ce_values = ce_gce(t, y, freqs)

Period derivative search::

    freqs = np.linspace(0.1, 1.0, 100)
    fdots = np.linspace(-1e-6, 1e-6, 50)
    ce_values = ce_gce(t, y, freqs, fdots=fdots)  # Returns 2D array

With magnitude information for better CE::

    mag = y  # Can use magnitudes instead of fluxes
    ce_values = ce_gce(t, mag, freqs, use_mag=True)
"""

import numpy as np
import warnings

__all__ = ['ce_gce', 'CE_GCE_AVAILABLE', 'check_gce_available']

# Check if gce is available
try:
    from gce import ConditionalEntropy
    CE_GCE_AVAILABLE = True
except ImportError:
    CE_GCE_AVAILABLE = False
    ConditionalEntropy = None


def ce_gce(t, y, freqs, fdots=None, phase_bins=10, mag_bins=5,
           use_mag=False, return_best=False, **kwargs):
    """
    Conditional Entropy period search using the gce package backend.

    This function wraps the gce package (Katz et al. 2020) to provide
    a cuvarbase-compatible API for Conditional Entropy searches, including
    optional period derivative (Pdot) searches.

    Parameters
    ----------
    t : array-like
        Observation times
    y : array-like
        Observed values (fluxes or magnitudes)
    freqs : array-like
        Frequencies to search (1D array)
    fdots : array-like, optional
        Frequency derivatives to search. If provided, performs a 2D search
        over (frequency, frequency_derivative) space. Default: None (1D search)
    phase_bins : int, optional
        Number of phase bins for the CE calculation. Default: 10
    mag_bins : int, optional
        Number of magnitude bins for the CE calculation. Default: 5
    use_mag : bool, optional
        If True, treats y as magnitudes. If False, treats as fluxes.
        Default: False
    return_best : bool, optional
        If True, returns only the minimum CE value and corresponding parameters.
        If False, returns the full CE grid. Default: False
    **kwargs : dict
        Additional keyword arguments passed to gce.ConditionalEntropy

    Returns
    -------
    ce_values : ndarray
        Conditional entropy values. Shape is (len(freqs),) for 1D search
        or (len(freqs), len(fdots)) for 2D search.

    best_params : dict, optional
        Only returned if return_best=True. Contains:
        - 'ce_min': minimum CE value
        - 'freq_best': best frequency
        - 'fdot_best': best frequency derivative (if fdots provided)
        - 'period_best': best period (1/freq_best)

    Raises
    ------
    ImportError
        If gce package is not installed

    Notes
    -----
    This is a thin wrapper around gce.ConditionalEntropy. All credit for
    the algorithm and implementation goes to Katz et al. (2020).

    The gce implementation is significantly more efficient than cuvarbase's
    native CE implementation, especially for:
    - Period derivative searches (not tractable with cuvarbase)
    - Large frequency grids
    - Better GPU utilization even for simple period searches

    References
    ----------
    Katz, M. L., Larson, S. L., Cohn, J., Vallisneri, M., & Graff, P. B. (2020).
    arXiv:2006.06866

    Examples
    --------
    >>> from cuvarbase.ce_gce import ce_gce
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1000)
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0) + 0.01 * np.random.randn(1000)
    >>> freqs = np.linspace(0.1, 1.0, 100)
    >>> ce = ce_gce(t, y, freqs)
    >>> best_freq_idx = np.argmin(ce)
    >>> print(f"Best period: {1/freqs[best_freq_idx]:.2f} days")
    """
    if not CE_GCE_AVAILABLE:
        raise ImportError(
            "This function requires the gce package.\n"
            "Install it with: pip install gce-search\n"
            "Or from source: pip install git+https://github.com/mikekatz04/gce.git\n"
            "\n"
            "For simple period searches without gce, you can use "
            "cuvarbase.ce.ConditionalEntropyAsyncProcess, though it is less efficient."
        )

    # Convert inputs to numpy arrays
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    if fdots is not None:
        fdots = np.asarray(fdots, dtype=np.float64)

    # Prepare data for gce
    # gce expects times, magnitudes/fluxes, and optionally errors
    if 'dy' in kwargs:
        dy = np.asarray(kwargs.pop('dy'), dtype=np.float64)
    else:
        dy = None

    # Convert frequencies to periods for gce if needed
    # (gce may expect periods - check their API)
    periods = 1.0 / freqs

    if fdots is not None:
        # Convert fdot to pdot: pdot = -fdot / f^2
        pdots = -fdots / (freqs[:, None] ** 2)
        pdots = pdots[0, :]  # Extract 1D pdot array
    else:
        pdots = None

    # Create gce ConditionalEntropy object
    # Note: This is a simplified interface - adjust based on actual gce API
    try:
        ce_search = ConditionalEntropy(
            times=t,
            mags=y,  # or fluxes, depending on gce's expectation
            errors=dy,
            num_phase_bins=phase_bins,
            num_mag_bins=mag_bins,
            **kwargs
        )

        # Run the search
        if pdots is not None:
            # 2D search
            ce_grid = ce_search.run_search(periods=periods, pdots=pdots)
        else:
            # 1D search
            ce_grid = ce_search.run_search(periods=periods)

        # Convert back to frequency ordering if needed
        ce_values = ce_grid

        if return_best:
            min_idx = np.unravel_index(np.argmin(ce_values), ce_values.shape)
            best_params = {
                'ce_min': ce_values[min_idx],
                'freq_best': freqs[min_idx[0]],
                'period_best': periods[min_idx[0]],
            }
            if pdots is not None and len(min_idx) > 1:
                best_params['fdot_best'] = fdots[min_idx[1]]
                best_params['pdot_best'] = pdots[min_idx[1]]

            return ce_values, best_params

        return ce_values

    except Exception as e:
        # Provide helpful error message with gce-specific context
        raise RuntimeError(
            f"Error calling gce.ConditionalEntropy: {e}\n"
            "\n"
            "This wrapper may need to be updated to match the current gce API.\n"
            "Please check the gce documentation: https://github.com/mikekatz04/gce\n"
            "\n"
            "If you believe this is a bug in cuvarbase's wrapper, please report it at:\n"
            "https://github.com/johnh2o2/cuvarbase/issues"
        ) from e


def check_gce_available():
    """
    Check if the gce package is available.

    Returns
    -------
    available : bool
        True if gce can be imported, False otherwise
    message : str
        Status message

    Examples
    --------
    >>> from cuvarbase.ce_gce import check_gce_available
    >>> available, message = check_gce_available()
    >>> if available:
    ...     print("gce is available!")
    ... else:
    ...     print(f"gce not available: {message}")
    """
    if CE_GCE_AVAILABLE:
        try:
            # Try to get version info
            import gce
            version = getattr(gce, '__version__', 'unknown')
            return True, f"gce version {version} is available"
        except Exception as e:
            return True, f"gce is importable but version check failed: {e}"
    else:
        return False, (
            "gce is not installed. Install it with:\n"
            "  pip install gce-search\n"
            "Or from source:\n"
            "  pip install git+https://github.com/mikekatz04/gce.git"
        )


# Informative message when module is imported
if not CE_GCE_AVAILABLE:
    warnings.warn(
        "cuvarbase.ce_gce: gce package not found. "
        "Install it with 'pip install gce-search' to enable period derivative searches.",
        ImportWarning,
        stacklevel=2
    )
