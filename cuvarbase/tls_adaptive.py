"""
Adaptive mode selection for transit search.

Automatically selects between sparse BLS, standard BLS, and TLS
based on dataset characteristics.

References
----------
.. [1] Hippke & Heller (2019), A&A 623, A39
.. [2] Panahi & Zucker (2021), arXiv:2103.06193 (sparse BLS)
"""

import numpy as np


def estimate_computational_cost(ndata, nperiods, method='tls'):
    """
    Estimate computational cost for a given method.

    Parameters
    ----------
    ndata : int
        Number of data points
    nperiods : int
        Number of trial periods
    method : str
        Method: 'sparse_bls', 'bls', or 'tls'

    Returns
    -------
    cost : float
        Relative computational cost (arbitrary units)

    Notes
    -----
    Sparse BLS: O(ndata² × nperiods)
    Standard BLS: O(ndata × nbins × nperiods)
    TLS: O(ndata log ndata × ndurations × nt0 × nperiods)
    """
    if method == 'sparse_bls':
        # Sparse BLS: tests all pairs of observations
        cost = ndata**2 * nperiods / 1e6
    elif method == 'bls':
        # Standard BLS: binning + search
        nbins = min(ndata, 200)  # Typical bin count
        cost = ndata * nbins * nperiods / 1e7
    elif method == 'tls':
        # TLS: sorting + search over durations and T0
        ndurations = 15
        nt0 = 30
        cost = ndata * np.log2(ndata + 1) * ndurations * nt0 * nperiods / 1e8
    else:
        cost = 0.0

    return cost


def select_optimal_method(t, nperiods=None, period_range=None,
                         sparse_threshold=500, tls_threshold=100,
                         prefer_accuracy=False):
    """
    Automatically select optimal transit search method.

    Parameters
    ----------
    t : array_like
        Observation times
    nperiods : int, optional
        Number of trial periods (estimated if None)
    period_range : tuple, optional
        (period_min, period_max) in days
    sparse_threshold : int, optional
        Use sparse BLS if ndata < this (default: 500)
    tls_threshold : int, optional
        Use TLS if ndata > this (default: 100)
    prefer_accuracy : bool, optional
        Prefer TLS even for small datasets (default: False)

    Returns
    -------
    method : str
        Recommended method: 'sparse_bls', 'bls', or 'tls'
    reason : str
        Explanation for the choice

    Notes
    -----
    Decision tree:
    1. Very few data points (< 100): Always sparse BLS
    2. Few data points (100-500): Sparse BLS unless prefer_accuracy
    3. Medium (500-2000): BLS or TLS depending on period range
    4. Many points (> 2000): TLS preferred

    Special cases:
    - Very short observation span: Sparse BLS (few transits anyway)
    - Very long period range: TLS (needs fine period sampling)
    """
    t = np.asarray(t)
    ndata = len(t)
    T_span = np.max(t) - np.min(t)

    # Estimate number of periods if not provided
    if nperiods is None:
        if period_range is not None:
            period_min, period_max = period_range
        else:
            period_min = T_span / 20  # At least 20 transits
            period_max = T_span / 2   # At least 2 transits

        # Rough estimate based on Ofir sampling
        nperiods = int(100 * (period_max / period_min)**(1/3))

    # Decision logic
    if ndata < tls_threshold:
        # Very few data points - sparse BLS is optimal
        if prefer_accuracy:
            method = 'tls'
            reason = "Few data points, but accuracy preferred → TLS"
        else:
            method = 'sparse_bls'
            reason = f"Few data points ({ndata} < {tls_threshold}) → Sparse BLS optimal"

    elif ndata < sparse_threshold:
        # Small to medium dataset
        # Compare computational costs
        cost_sparse = estimate_computational_cost(ndata, nperiods, 'sparse_bls')
        cost_bls = estimate_computational_cost(ndata, nperiods, 'bls')
        cost_tls = estimate_computational_cost(ndata, nperiods, 'tls')

        if prefer_accuracy:
            method = 'tls'
            reason = f"Medium dataset ({ndata}), accuracy preferred → TLS"
        elif cost_sparse < min(cost_bls, cost_tls):
            method = 'sparse_bls'
            reason = f"Sparse BLS fastest for {ndata} points, {nperiods} periods"
        elif cost_bls < cost_tls:
            method = 'bls'
            reason = f"Standard BLS optimal for {ndata} points"
        else:
            method = 'tls'
            reason = f"TLS preferred for best accuracy with {ndata} points"

    else:
        # Large dataset - TLS is best
        method = 'tls'
        reason = f"Large dataset ({ndata} > {sparse_threshold}) → TLS optimal"

    # Override for special cases
    if T_span < 10:
        # Very short observation span
        method = 'sparse_bls'
        reason += f" (overridden: short span {T_span:.1f} days → Sparse BLS)"

    if nperiods > 10000:
        # Very fine period sampling needed
        if ndata > sparse_threshold:
            method = 'tls'
            reason += f" (confirmed: {nperiods} periods needs efficient method)"

    return method, reason


def adaptive_transit_search(t, y, dy, **kwargs):
    """
    Adaptive transit search that automatically selects optimal method.

    Parameters
    ----------
    t, y, dy : array_like
        Time series data
    **kwargs
        Passed to the selected search method
        Special parameters:
        - force_method : str, force use of specific method
        - prefer_accuracy : bool, prefer accuracy over speed
        - sparse_threshold : int, threshold for sparse BLS
        - tls_threshold : int, threshold for TLS

    Returns
    -------
    results : dict
        Search results with added 'method_used' field

    Examples
    --------
    >>> results = adaptive_transit_search(t, y, dy)
    >>> print(f"Used method: {results['method_used']}")
    >>> print(f"Best period: {results['period']:.4f} days")
    """
    # Extract adaptive parameters
    force_method = kwargs.pop('force_method', None)
    prefer_accuracy = kwargs.pop('prefer_accuracy', False)
    sparse_threshold = kwargs.pop('sparse_threshold', 500)
    tls_threshold = kwargs.pop('tls_threshold', 100)

    # Get period range if specified
    period_range = None
    if 'period_min' in kwargs and 'period_max' in kwargs:
        period_range = (kwargs['period_min'], kwargs['period_max'])
    elif 'periods' in kwargs and kwargs['periods'] is not None:
        periods = kwargs['periods']
        period_range = (np.min(periods), np.max(periods))

    # Select method
    if force_method:
        method = force_method
        reason = "Forced by user"
    else:
        method, reason = select_optimal_method(
            t,
            period_range=period_range,
            sparse_threshold=sparse_threshold,
            tls_threshold=tls_threshold,
            prefer_accuracy=prefer_accuracy
        )

    print(f"Adaptive mode: Using {method.upper()}")
    print(f"Reason: {reason}")

    # Run selected method
    if method == 'sparse_bls':
        try:
            from . import bls
            # Use sparse BLS from cuvarbase
            freqs, powers, solutions = bls.eebls_transit(
                t, y, dy,
                use_sparse=True,
                use_gpu=True,
                **kwargs
            )

            # Convert to TLS-like results format
            results = {
                'periods': 1.0 / freqs,
                'power': powers,
                'method_used': 'sparse_bls',
                'method_reason': reason,
            }

            # Find best
            best_idx = np.argmax(powers)
            results['period'] = results['periods'][best_idx]
            results['q'], results['phi'] = solutions[best_idx]

        except ImportError:
            print("Warning: BLS module not available, falling back to TLS")
            method = 'tls'

    if method == 'bls':
        try:
            from . import bls
            # Use standard BLS
            freqs, powers = bls.eebls_transit(
                t, y, dy,
                use_sparse=False,
                use_fast=True,
                **kwargs
            )

            results = {
                'periods': 1.0 / freqs,
                'power': powers,
                'method_used': 'bls',
                'method_reason': reason,
            }

            best_idx = np.argmax(powers)
            results['period'] = results['periods'][best_idx]

        except ImportError:
            print("Warning: BLS module not available, falling back to TLS")
            method = 'tls'

    if method == 'tls':
        from . import tls
        # Use TLS
        results = tls.tls_search_gpu(t, y, dy, **kwargs)
        results['method_used'] = 'tls'
        results['method_reason'] = reason

    return results


def compare_methods(t, y, dy, periods=None, **kwargs):
    """
    Run all three methods and compare results.

    Useful for testing and validation.

    Parameters
    ----------
    t, y, dy : array_like
        Time series data
    periods : array_like, optional
        Trial periods for all methods
    **kwargs
        Passed to search methods

    Returns
    -------
    comparison : dict
        Results from each method with timing information

    Examples
    --------
    >>> comp = compare_methods(t, y, dy)
    >>> for method, res in comp.items():
    ...     print(f"{method}: Period={res['period']:.4f}, Time={res['time']:.3f}s")
    """
    import time

    comparison = {}

    # Common parameters
    if periods is not None:
        kwargs['periods'] = periods

    # Test sparse BLS
    print("Testing Sparse BLS...")
    try:
        t0 = time.time()
        results = adaptive_transit_search(
            t, y, dy, force_method='sparse_bls', **kwargs
        )
        t1 = time.time()
        results['time'] = t1 - t0
        comparison['sparse_bls'] = results
        print(f"  ✓ Completed in {results['time']:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test standard BLS
    print("Testing Standard BLS...")
    try:
        t0 = time.time()
        results = adaptive_transit_search(
            t, y, dy, force_method='bls', **kwargs
        )
        t1 = time.time()
        results['time'] = t1 - t0
        comparison['bls'] = results
        print(f"  ✓ Completed in {results['time']:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test TLS
    print("Testing TLS...")
    try:
        t0 = time.time()
        results = adaptive_transit_search(
            t, y, dy, force_method='tls', **kwargs
        )
        t1 = time.time()
        results['time'] = t1 - t0
        comparison['tls'] = results
        print(f"  ✓ Completed in {results['time']:.3f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    return comparison
