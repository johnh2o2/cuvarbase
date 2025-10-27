"""
GPU-accelerated Transit Least Squares (TLS) periodogram.

This module implements a fast GPU version of the Transit Least Squares
algorithm for detecting planetary transits in photometric time series.

References
----------
.. [1] Hippke & Heller (2019), "Transit Least Squares",  A&A 623, A39
.. [2] Kovács et al. (2002), "Box Least Squares", A&A 391, 369
"""

import sys
import threading
from collections import OrderedDict
import resource

import pycuda.autoprimaryctx
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np

from .utils import find_kernel, _module_reader
from . import tls_grids
from . import tls_models
from . import tls_stats

_default_block_size = 128  # Smaller default than BLS (TLS has more shared memory needs)
_KERNEL_CACHE_MAX_SIZE = 10
_kernel_cache = OrderedDict()
_kernel_cache_lock = threading.Lock()


def _choose_block_size(ndata):
    """
    Choose optimal block size for TLS kernel based on data size.

    Parameters
    ----------
    ndata : int
        Number of data points

    Returns
    -------
    block_size : int
        Optimal CUDA block size (32, 64, or 128)

    Notes
    -----
    TLS uses more shared memory than BLS, so we use smaller block sizes
    to avoid shared memory limits.
    """
    if ndata <= 32:
        return 32
    elif ndata <= 64:
        return 64
    else:
        return 128  # Max for TLS (vs 256 for BLS)


def _get_cached_kernels(block_size):
    """
    Get compiled TLS kernel from cache.

    Parameters
    ----------
    block_size : int
        CUDA block size

    Returns
    -------
    kernel : PyCUDA function
        Compiled kernel function
    """
    key = block_size

    with _kernel_cache_lock:
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]

        # Compile kernel
        compiled = compile_tls(block_size=block_size)

        # Add to cache
        _kernel_cache[key] = compiled
        _kernel_cache.move_to_end(key)

        # Evict oldest if needed
        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)

        return compiled


def compile_tls(block_size=_default_block_size):
    """
    Compile TLS CUDA kernels.

    Parameters
    ----------
    block_size : int, optional
        CUDA block size (default: 128)

    Returns
    -------
    kernels : dict
        Dictionary with 'standard' and 'keplerian' kernel functions

    Notes
    -----
    The kernels use insertion sort for phase sorting, which is efficient
    for nearly-sorted data (common after phase folding sorted time series).
    Works well for datasets up to ~5000 points.

    The 'keplerian' kernel variant accepts per-period qmin/qmax arrays
    to focus the duration search on physically plausible values.
    """
    cppd = dict(BLOCK_SIZE=block_size)

    kernel_name = 'tls'
    kernel_txt = _module_reader(find_kernel(kernel_name), cpp_defs=cppd)

    # Compile with fast math
    # no_extern_c=True needed for proper extern "C" handling
    module = SourceModule(kernel_txt, options=['--use_fast_math'], no_extern_c=True)

    # Get both kernel functions
    kernels = {
        'standard': module.get_function('tls_search_kernel'),
        'keplerian': module.get_function('tls_search_kernel_keplerian')
    }

    return kernels


class TLSMemory:
    """
    Memory management for TLS GPU computations.

    This class handles allocation and transfer of data between CPU and GPU
    for TLS periodogram calculations.

    Parameters
    ----------
    max_ndata : int
        Maximum number of data points
    max_nperiods : int
        Maximum number of trial periods
    stream : pycuda.driver.Stream, optional
        CUDA stream for async operations

    Attributes
    ----------
    t, y, dy : ndarray
        Pinned CPU arrays for time, flux, uncertainties
    t_g, y_g, dy_g : gpuarray
        GPU arrays for data
    periods_g, chi2_g : gpuarray
        GPU arrays for periods and chi-squared values
    best_t0_g, best_duration_g, best_depth_g : gpuarray
        GPU arrays for best-fit parameters
    """

    def __init__(self, max_ndata, max_nperiods, stream=None, **kwargs):
        self.max_ndata = max_ndata
        self.max_nperiods = max_nperiods
        self.stream = stream
        self.rtype = np.float32

        # CPU pinned memory for fast transfers
        self.t = None
        self.y = None
        self.dy = None

        # GPU memory
        self.t_g = None
        self.y_g = None
        self.dy_g = None
        self.periods_g = None
        self.qmin_g = None  # Keplerian duration constraints
        self.qmax_g = None  # Keplerian duration constraints
        self.chi2_g = None
        self.best_t0_g = None
        self.best_duration_g = None
        self.best_depth_g = None

        self.allocate_pinned_arrays()

    def allocate_pinned_arrays(self):
        """Allocate page-aligned pinned memory on CPU for fast transfers."""
        pagesize = resource.getpagesize()

        self.t = cuda.aligned_zeros(shape=(self.max_ndata,),
                                    dtype=self.rtype,
                                    alignment=pagesize)

        self.y = cuda.aligned_zeros(shape=(self.max_ndata,),
                                    dtype=self.rtype,
                                    alignment=pagesize)

        self.dy = cuda.aligned_zeros(shape=(self.max_ndata,),
                                     dtype=self.rtype,
                                     alignment=pagesize)

        self.periods = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                         dtype=self.rtype,
                                         alignment=pagesize)

        self.chi2 = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                      dtype=self.rtype,
                                      alignment=pagesize)

        self.best_t0 = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                         dtype=self.rtype,
                                         alignment=pagesize)

        self.best_duration = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                               dtype=self.rtype,
                                               alignment=pagesize)

        self.best_depth = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                            dtype=self.rtype,
                                            alignment=pagesize)

        # Keplerian duration constraints
        self.qmin = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                      dtype=self.rtype,
                                      alignment=pagesize)

        self.qmax = cuda.aligned_zeros(shape=(self.max_nperiods,),
                                      dtype=self.rtype,
                                      alignment=pagesize)

    def allocate_gpu_arrays(self, ndata=None, nperiods=None):
        """Allocate GPU memory."""
        if ndata is None:
            ndata = self.max_ndata
        if nperiods is None:
            nperiods = self.max_nperiods

        self.t_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.y_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.dy_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.periods_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.qmin_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.qmax_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.chi2_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.best_t0_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.best_duration_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        self.best_depth_g = gpuarray.zeros(nperiods, dtype=self.rtype)

    def setdata(self, t, y, dy, periods=None, qmin=None, qmax=None, transfer=True):
        """
        Set data for TLS computation.

        Parameters
        ----------
        t : array_like
            Observation times
        y : array_like
            Flux measurements
        dy : array_like
            Flux uncertainties
        periods : array_like, optional
            Trial periods
        qmin : array_like, optional
            Minimum fractional duration per period (for Keplerian search)
        qmax : array_like, optional
            Maximum fractional duration per period (for Keplerian search)
        transfer : bool, optional
            Transfer to GPU immediately (default: True)
        """
        ndata = len(t)

        # Copy to pinned memory
        self.t[:ndata] = np.asarray(t).astype(self.rtype)
        self.y[:ndata] = np.asarray(y).astype(self.rtype)
        self.dy[:ndata] = np.asarray(dy).astype(self.rtype)

        if periods is not None:
            nperiods = len(periods)
            self.periods[:nperiods] = np.asarray(periods).astype(self.rtype)

        if qmin is not None:
            nperiods = len(qmin)
            self.qmin[:nperiods] = np.asarray(qmin).astype(self.rtype)

        if qmax is not None:
            nperiods = len(qmax)
            self.qmax[:nperiods] = np.asarray(qmax).astype(self.rtype)

        # Allocate GPU memory if needed
        if self.t_g is None or len(self.t_g) < ndata:
            self.allocate_gpu_arrays(ndata, len(periods) if periods is not None else self.max_nperiods)

        # Transfer to GPU
        if transfer:
            self.transfer_to_gpu(ndata, len(periods) if periods is not None else None,
                               qmin is not None, qmax is not None)

    def transfer_to_gpu(self, ndata, nperiods=None, has_qmin=False, has_qmax=False):
        """Transfer data from CPU to GPU."""
        if self.stream is None:
            self.t_g.set(self.t[:ndata])
            self.y_g.set(self.y[:ndata])
            self.dy_g.set(self.dy[:ndata])
            if nperiods is not None:
                self.periods_g.set(self.periods[:nperiods])
            if has_qmin:
                self.qmin_g.set(self.qmin[:nperiods])
            if has_qmax:
                self.qmax_g.set(self.qmax[:nperiods])
        else:
            self.t_g.set_async(self.t[:ndata], stream=self.stream)
            self.y_g.set_async(self.y[:ndata], stream=self.stream)
            self.dy_g.set_async(self.dy[:ndata], stream=self.stream)
            if nperiods is not None:
                self.periods_g.set_async(self.periods[:nperiods], stream=self.stream)
            if has_qmin:
                self.qmin_g.set_async(self.qmin[:nperiods], stream=self.stream)
            if has_qmax:
                self.qmax_g.set_async(self.qmax[:nperiods], stream=self.stream)

    def transfer_from_gpu(self, nperiods):
        """Transfer results from GPU to CPU."""
        if self.stream is None:
            self.chi2[:nperiods] = self.chi2_g.get()[:nperiods]
            self.best_t0[:nperiods] = self.best_t0_g.get()[:nperiods]
            self.best_duration[:nperiods] = self.best_duration_g.get()[:nperiods]
            self.best_depth[:nperiods] = self.best_depth_g.get()[:nperiods]
        else:
            self.chi2_g.get_async(ary=self.chi2, stream=self.stream)
            self.best_t0_g.get_async(ary=self.best_t0, stream=self.stream)
            self.best_duration_g.get_async(ary=self.best_duration, stream=self.stream)
            self.best_depth_g.get_async(ary=self.best_depth, stream=self.stream)

    @classmethod
    def fromdata(cls, t, y, dy, periods=None, **kwargs):
        """
        Create TLSMemory instance from data.

        Parameters
        ----------
        t, y, dy : array_like
            Time series data
        periods : array_like, optional
            Trial periods
        **kwargs
            Passed to __init__

        Returns
        -------
        memory : TLSMemory
            Initialized memory object
        """
        max_ndata = kwargs.get('max_ndata', len(t))
        max_nperiods = kwargs.get('max_nperiods',
                                  len(periods) if periods is not None else 10000)

        mem = cls(max_ndata, max_nperiods, **kwargs)
        mem.setdata(t, y, dy, periods=periods, transfer=kwargs.get('transfer', True))

        return mem


def tls_search_gpu(t, y, dy, periods=None, durations=None,
                   qmin=None, qmax=None, n_durations=15,
                   R_star=1.0, M_star=1.0,
                   period_min=None, period_max=None, n_transits_min=2,
                   oversampling_factor=3, duration_grid_step=1.1,
                   R_planet_min=0.5, R_planet_max=5.0,
                   limb_dark='quadratic', u=[0.4804, 0.1867],
                   block_size=None,
                   kernel=None, memory=None, stream=None,
                   transfer_to_device=True, transfer_to_host=True,
                   **kwargs):
    """
    Run Transit Least Squares search on GPU.

    Parameters
    ----------
    t : array_like
        Observation times (days)
    y : array_like
        Flux measurements (arbitrary units, will be normalized)
    dy : array_like
        Flux uncertainties
    periods : array_like, optional
        Custom period grid. If None, generated automatically.
    qmin : array_like, optional
        Minimum fractional duration per period (for Keplerian search).
        If provided, enables Keplerian mode.
    qmax : array_like, optional
        Maximum fractional duration per period (for Keplerian search).
        If provided, enables Keplerian mode.
    n_durations : int, optional
        Number of duration samples per period (default: 15).
        Only used in Keplerian mode.
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)
    period_min, period_max : float, optional
        Period search range (days). Auto-computed if None.
    n_transits_min : int, optional
        Minimum number of transits required (default: 2)
    oversampling_factor : float, optional
        Period grid oversampling (default: 3)
    duration_grid_step : float, optional
        Duration grid spacing factor (default: 1.1)
    R_planet_min, R_planet_max : float, optional
        Planet radius range in Earth radii (default: 0.5 to 5.0)
    limb_dark : str, optional
        Limb darkening law (default: 'quadratic')
    u : list, optional
        Limb darkening coefficients (default: [0.4804, 0.1867])
    block_size : int, optional
        CUDA block size (auto-selected if None)
    kernel : PyCUDA function, optional
        Pre-compiled kernel
    memory : TLSMemory, optional
        Pre-allocated memory object
    stream : cuda.Stream, optional
        CUDA stream for async execution
    transfer_to_device : bool, optional
        Transfer data to GPU (default: True)
    transfer_to_host : bool, optional
        Transfer results to CPU (default: True)

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'periods': Trial periods
        - 'chi2': Chi-squared values
        - 'best_t0': Best mid-transit times
        - 'best_duration': Best durations
        - 'best_depth': Best depths
        - 'SDE': Signal Detection Efficiency (if computed)

    Notes
    -----
    This is the main GPU TLS function. For the first implementation,
    it provides a basic version that will be optimized in Phase 2.
    """
    # Validate stellar parameters
    tls_grids.validate_stellar_parameters(R_star, M_star)

    # Validate limb darkening
    tls_models.validate_limb_darkening_coeffs(u, limb_dark)

    # Generate period grid if not provided
    if periods is None:
        periods = tls_grids.period_grid_ofir(
            t, R_star=R_star, M_star=M_star,
            oversampling_factor=oversampling_factor,
            period_min=period_min, period_max=period_max,
            n_transits_min=n_transits_min
        )

    # Convert to numpy arrays
    t = np.asarray(t, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    periods = np.asarray(periods, dtype=np.float32)

    ndata = len(t)
    nperiods = len(periods)

    # Choose block size
    if block_size is None:
        block_size = _choose_block_size(ndata)

    # Determine if using Keplerian mode
    use_keplerian = (qmin is not None and qmax is not None)

    # Get or compile kernels
    if kernel is None:
        kernels = _get_cached_kernels(block_size)
        kernel = kernels['keplerian'] if use_keplerian else kernels['standard']

    # Allocate or use existing memory
    if memory is None:
        memory = TLSMemory.fromdata(t, y, dy, periods=periods,
                                    stream=stream,
                                    transfer=transfer_to_device)
    elif transfer_to_device:
        memory.setdata(t, y, dy, periods=periods, transfer=True)

    # Set qmin/qmax if using Keplerian mode
    if use_keplerian:
        qmin = np.asarray(qmin, dtype=np.float32)
        qmax = np.asarray(qmax, dtype=np.float32)
        if len(qmin) != nperiods or len(qmax) != nperiods:
            raise ValueError(f"qmin and qmax must have same length as periods ({nperiods})")
        memory.setdata(t, y, dy, periods=periods, qmin=qmin, qmax=qmax, transfer=transfer_to_device)

    # Calculate shared memory requirements
    # Simple/basic kernels: phases, y_sorted, dy_sorted, + 4 thread arrays
    # = ndata * 3 + block_size * 4 (for chi2, t0, duration, depth)
    shared_mem_size = (3 * ndata + 4 * block_size) * 4  # 4 bytes per float

    # Additional for config index tracking (int)
    shared_mem_size += block_size * 4  # int32

    # Launch kernel
    grid = (nperiods, 1, 1)
    block = (block_size, 1, 1)

    if use_keplerian:
        # Keplerian kernel with qmin/qmax arrays
        if stream is None:
            kernel(
                memory.t_g, memory.y_g, memory.dy_g,
                memory.periods_g, memory.qmin_g, memory.qmax_g,
                np.int32(ndata), np.int32(nperiods), np.int32(n_durations),
                memory.chi2_g, memory.best_t0_g,
                memory.best_duration_g, memory.best_depth_g,
                block=block, grid=grid,
                shared=shared_mem_size
            )
        else:
            kernel(
                memory.t_g, memory.y_g, memory.dy_g,
                memory.periods_g, memory.qmin_g, memory.qmax_g,
                np.int32(ndata), np.int32(nperiods), np.int32(n_durations),
                memory.chi2_g, memory.best_t0_g,
                memory.best_duration_g, memory.best_depth_g,
                block=block, grid=grid,
                shared=shared_mem_size,
                stream=stream
            )
    else:
        # Standard kernel with fixed duration range
        if stream is None:
            kernel(
                memory.t_g, memory.y_g, memory.dy_g,
                memory.periods_g,
                np.int32(ndata), np.int32(nperiods),
                memory.chi2_g, memory.best_t0_g,
                memory.best_duration_g, memory.best_depth_g,
                block=block, grid=grid,
                shared=shared_mem_size
            )
        else:
            kernel(
                memory.t_g, memory.y_g, memory.dy_g,
                memory.periods_g,
                np.int32(ndata), np.int32(nperiods),
                memory.chi2_g, memory.best_t0_g,
                memory.best_duration_g, memory.best_depth_g,
                block=block, grid=grid,
                shared=shared_mem_size,
                stream=stream
            )

    # Transfer results if requested
    if transfer_to_host:
        if stream is not None:
            stream.synchronize()
        memory.transfer_from_gpu(nperiods)

        chi2_vals = memory.chi2[:nperiods].copy()
        best_t0_vals = memory.best_t0[:nperiods].copy()
        best_duration_vals = memory.best_duration[:nperiods].copy()
        best_depth_vals = memory.best_depth[:nperiods].copy()

        # Find best period
        best_idx = np.argmin(chi2_vals)
        best_period = periods[best_idx]
        best_chi2 = chi2_vals[best_idx]
        best_t0 = best_t0_vals[best_idx]
        best_duration = best_duration_vals[best_idx]
        best_depth = best_depth_vals[best_idx]

        # Estimate number of transits
        T_span = np.max(t) - np.min(t)
        n_transits = int(T_span / best_period)

        # Compute statistics
        stats = tls_stats.compute_all_statistics(
            chi2_vals, periods, best_idx,
            best_depth, best_duration, n_transits
        )

        # Period uncertainty
        period_uncertainty = tls_stats.compute_period_uncertainty(
            periods, chi2_vals, best_idx
        )

        results = {
            # Raw outputs
            'periods': periods,
            'chi2': chi2_vals,
            'best_t0_per_period': best_t0_vals,
            'best_duration_per_period': best_duration_vals,
            'best_depth_per_period': best_depth_vals,

            # Best-fit parameters
            'period': best_period,
            'period_uncertainty': period_uncertainty,
            'T0': best_t0,
            'duration': best_duration,
            'depth': best_depth,
            'chi2_min': best_chi2,

            # Statistics
            'SDE': stats['SDE'],
            'SDE_raw': stats['SDE_raw'],
            'SNR': stats['SNR'],
            'FAP': stats['FAP'],
            'power': stats['power'],
            'SR': stats['SR'],

            # Metadata
            'n_transits': n_transits,
            'R_star': R_star,
            'M_star': M_star,
        }
    else:
        # Just return periods if not transferring
        results = {
            'periods': periods,
            'chi2': None,
            'best_t0_per_period': None,
            'best_duration_per_period': None,
            'best_depth_per_period': None,
        }

    return results


def tls_search(t, y, dy, **kwargs):
    """
    High-level TLS search function.

    This is the main user-facing function for TLS searches.

    Parameters
    ----------
    t, y, dy : array_like
        Time series data
    **kwargs
        Passed to tls_search_gpu

    Returns
    -------
    results : dict
        Search results

    See Also
    --------
    tls_search_gpu : Lower-level GPU function
    tls_transit : Keplerian-aware search wrapper
    """
    return tls_search_gpu(t, y, dy, **kwargs)


def tls_transit(t, y, dy, R_star=1.0, M_star=1.0, R_planet=1.0,
                qmin_fac=0.5, qmax_fac=2.0, n_durations=15,
                period_min=None, period_max=None, n_transits_min=2,
                oversampling_factor=3, **kwargs):
    """
    Transit Least Squares search with Keplerian duration constraints.

    This is the TLS analog of BLS's eebls_transit() function. It uses stellar
    parameters to focus the duration search on physically plausible values,
    providing ~7-8× efficiency improvement over fixed duration ranges.

    Parameters
    ----------
    t : array_like
        Observation times (days)
    y : array_like
        Flux measurements (arbitrary units)
    dy : array_like
        Flux uncertainties
    R_star : float, optional
        Stellar radius in solar radii (default: 1.0)
    M_star : float, optional
        Stellar mass in solar masses (default: 1.0)
    R_planet : float, optional
        Fiducial planet radius in Earth radii (default: 1.0)
        Sets the central duration value around which to search
    qmin_fac : float, optional
        Minimum duration factor (default: 0.5)
        Searches down to qmin_fac × q_keplerian
    qmax_fac : float, optional
        Maximum duration factor (default: 2.0)
        Searches up to qmax_fac × q_keplerian
    n_durations : int, optional
        Number of duration samples per period (default: 15)
    period_min, period_max : float, optional
        Period search range (days). Auto-computed if None.
    n_transits_min : int, optional
        Minimum number of transits required (default: 2)
    oversampling_factor : float, optional
        Period grid oversampling (default: 3)
    **kwargs
        Additional parameters passed to tls_search_gpu

    Returns
    -------
    results : dict
        Search results with keys:
        - 'period': Best-fit period
        - 'T0': Best mid-transit time
        - 'duration': Best transit duration
        - 'depth': Best transit depth
        - 'SDE': Signal Detection Efficiency
        - 'periods': Trial periods
        - 'chi2': Chi-squared values per period
        ... (see tls_search_gpu for full list)

    Notes
    -----
    This function automatically generates:
    1. Optimal period grid using Ofir (2014) algorithm
    2. Per-period duration ranges based on Keplerian physics
    3. Qmin/qmax arrays for focused duration search

    The duration search at each period focuses on physically plausible values:
    - For short periods: searches shorter durations
    - For long periods: searches longer durations
    - Scales with stellar density (M_star, R_star)

    This is much more efficient than searching a fixed fractional duration
    range (0.5%-15%) at all periods.

    Examples
    --------
    >>> from cuvarbase import tls
    >>> results = tls.tls_transit(t, y, dy,
    ...                            R_star=1.0, M_star=1.0,
    ...                            period_min=5.0, period_max=20.0)
    >>> print(f"Best period: {results['period']:.4f} days")
    >>> print(f"Transit depth: {results['depth']:.4f}")

    See Also
    --------
    tls_search_gpu : Lower-level GPU function
    tls_grids.duration_grid_keplerian : Generate Keplerian duration grids
    tls_grids.q_transit : Calculate Keplerian fractional duration
    """
    # Generate period grid
    periods = tls_grids.period_grid_ofir(
        t, R_star=R_star, M_star=M_star,
        oversampling_factor=oversampling_factor,
        period_min=period_min, period_max=period_max,
        n_transits_min=n_transits_min
    )

    # Generate Keplerian duration constraints
    durations, dur_counts, q_values = tls_grids.duration_grid_keplerian(
        periods, R_star=R_star, M_star=M_star, R_planet=R_planet,
        qmin_fac=qmin_fac, qmax_fac=qmax_fac, n_durations=n_durations
    )

    # Calculate qmin and qmax arrays
    qmin = q_values * qmin_fac
    qmax = q_values * qmax_fac

    # Run TLS search with Keplerian constraints
    results = tls_search_gpu(
        t, y, dy,
        periods=periods,
        qmin=qmin,
        qmax=qmax,
        n_durations=n_durations,
        R_star=R_star,
        M_star=M_star,
        **kwargs
    )

    return results
