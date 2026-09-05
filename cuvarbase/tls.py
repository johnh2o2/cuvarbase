"""
GPU-accelerated Transit Least Squares (TLS) periodogram.

This module implements a fast GPU version of the Transit Least Squares
algorithm for detecting planetary transits in photometric time series.

References
----------
- Hippke & Heller (2019), "Transit Least Squares", A&A 623, A39
- Kovács et al. (2002), "Box Least Squares", A&A 391, 369
"""

import sys
import threading
import warnings
import operator
from collections import OrderedDict

import pycuda.driver as cuda  # noqa: E402
import pycuda.gpuarray as gpuarray  # noqa: E402
from pycuda.compiler import SourceModule  # noqa: E402

import numpy as np

from .base import ensure_context  # noqa: E402
from .memory._host import host_array  # noqa: E402
from .utils import (find_kernel, _module_reader,
                    check_lightcurve)
from . import tls_grids
from . import tls_models
from . import tls_stats

_default_block_size = 128  # Smaller default than BLS (TLS has more shared memory needs)
_KERNEL_CACHE_MAX_SIZE = 10
_kernel_cache = OrderedDict()
_kernel_cache_lock = threading.Lock()

# Default CUDA limit for dynamic shared memory per block; exceeding it
# fails at kernel launch, so we guard at the Python layer instead.
_SHARED_MEM_LIMIT = 48 * 1024

# The kernels initialize each period's chi2 to this sentinel and only
# overwrite it when a valid solution is found.
TLS_CHI2_SENTINEL = np.float32(1e30)


_NO_SOLUTION_MSG = (
    "TLS kernel returned no valid solution for any of the %d trial "
    "periods (a flat or noiseless light curve gives zero depth at every "
    "trial, which the kernels reject)")


# Minimum number of observations any TLS entry point accepts. The
# transit model is fitted against the constant-baseline chi2 of the
# same light curve, which is identically zero for a single point (the
# reported chi2 ratio came back 0/0), and the automatic Ofir period
# grid needs a non-zero baseline.
_TLS_MIN_NDATA = 2


def _mask_failed_periods(chi2_vals):
    """Return a boolean mask of trial periods with a valid solution.

    Failed periods keep the kernel's 1e30 chi2 initializer; left
    unmasked they corrupt the best-fit argmin and collapse the SDE
    statistics. Warns when any period failed. When EVERY period failed
    (e.g. a flat/noiseless light curve) the mask is all-False and a
    warning says so; the search wrappers then return a null result
    (SDE = 0, NaN best-fit parameters) instead of raising, like the
    reference ``transitleastsquares`` package.
    """
    chi2_vals = np.asarray(chi2_vals)
    valid = np.isfinite(chi2_vals) & (chi2_vals < 0.1 * TLS_CHI2_SENTINEL)
    n_failed = int(chi2_vals.size - valid.sum())
    if n_failed == chi2_vals.size:
        warnings.warn(_NO_SOLUTION_MSG % chi2_vals.size
                      + "; returning a null result (SDE = 0)")
    elif n_failed:
        warnings.warn(
            "%d of %d trial periods returned no valid TLS solution "
            "(chi2 sentinel); they are excluded from the best-fit "
            "search and the SDE statistics and appear as NaN in "
            "the returned arrays" % (n_failed, chi2_vals.size))
    return valid


def _null_result(nperiods, chi2_0, message, periods=None, arrays=False):
    """Result dict for a light curve with no valid trial period: SDE = 0,
    NaN best-fit parameters, and the failure message under 'error'."""
    res = {
        'period': np.nan,
        'period_uncertainty': np.nan,
        't0_phase': np.nan,
        'T0': np.nan,
        'duration': np.nan,
        'depth': 0.0,
        'chi2_min': float(chi2_0),
        'SDE': 0.0,
        'SDE_raw': 0.0,
        'SNR': 0.0,
        'n_transits': 0,
        'n_failed_periods': int(nperiods),
        'error': message,
    }
    if arrays:
        def _nan():
            return np.full(nperiods, np.nan)
        res.update({
            'periods': periods,
            'chi2': _nan(),
            'best_t0_per_period': _nan(),
            'best_duration_per_period': _nan(),
            'best_depth_per_period': _nan(),
            'valid_periods': np.zeros(nperiods, dtype=bool),
            'power': _nan(),
            'SR': _nan(),
        })
    return res


def _validate_periods(periods):
    """Common checks on a trial-period grid (any order)."""
    periods = np.asarray(periods)
    if periods.ndim != 1:
        raise ValueError("periods must be a 1-d array")
    if periods.size == 0:
        raise ValueError("periods must be non-empty")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods must be finite and > 0")
    return periods


def _sort_period_grid(periods):
    """Return (periods_ascending, order): the SDE running-median detrend
    and the period-uncertainty neighbour walk assume period-ordered
    neighbours, so user grids are sorted on entry. ``order`` is None
    when the grid is already ascending, else the argsort that maps
    the caller's order to ascending (see :func:`_to_caller_order`)."""
    periods = np.asarray(periods)
    if periods.size > 1 and np.any(np.diff(periods) < 0):
        order = np.argsort(periods, kind='stable')
        return periods[order], order
    return periods, None


def _to_caller_order(values, order):
    """Scatter a per-period array from ascending order back to the
    caller's grid order (identity when ``order`` is None)."""
    if order is None:
        return values
    out = np.empty_like(values)
    out[order] = values
    return out


def _validate_q_window(qmin, qmax, periods=None):
    bad = (np.asarray(qmin) <= 0) | (np.asarray(qmax) < np.asarray(qmin)) \
        | (np.asarray(qmax) >= 1)
    if not np.any(bad):
        return
    where = ""
    if periods is not None and np.ndim(bad) and np.any(bad):
        pbad = np.asarray(periods, dtype=float)[np.asarray(bad)]
        where = (" at P = %.4g .. %.4g d" % (pbad.min(), pbad.max()))
    raise ValueError(
        "need 0 < qmin <= qmax < 1 at every period%s (the transit "
        "duration must be shorter than the period; the binned scan "
        "would double-count phase bins for q >= 1). The Keplerian "
        "duration window reaches q >= 1 at sub-Roche periods: shrink "
        "qmax_fac, pass explicit qmin/qmax, raise the shortest trial "
        "period, or opt into the constant window with "
        "duration_window='fixed'." % where)


def _first_transit_at_or_after(t_mid, period, tmin):
    """Shift a mid-transit time by whole periods into [tmin, tmin +
    period): the 'T0' convention of every TLS result (same as the
    reference package's ``T0``)."""
    return tmin + ((t_mid - tmin) % period)


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


def _get_cached_kernels(block_size, t0_oversample=3.0):
    """
    Get compiled TLS kernel from cache.

    Parameters
    ----------
    block_size : int
        CUDA block size
    t0_oversample : float, optional (default: 3.0)
        Transit-epoch oversampling baked into the kernel's
        ``T0_OVERSAMPLE`` define; part of the cache key, so distinct
        values compile (and cache) distinct kernels.

    Returns
    -------
    kernel : PyCUDA function
        Compiled kernel function
    """
    key = (block_size, float(t0_oversample))

    with _kernel_cache_lock:
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]

        # Compile kernel
        compiled = compile_tls(block_size=block_size,
                               t0_oversample=t0_oversample)

        # Add to cache
        _kernel_cache[key] = compiled
        _kernel_cache.move_to_end(key)

        # Evict oldest if needed
        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)

        return compiled


def compile_tls(block_size=_default_block_size, t0_oversample=3.0):
    """
    Compile TLS CUDA kernels.

    Parameters
    ----------
    block_size : int, optional
        CUDA block size (default: 128)
    t0_oversample : float, optional (default: 3.0)
        Transit-epoch (t0) oversampling: the on-device epoch stride is
        ``duration_phase / t0_oversample``, so larger values test a finer
        grid of transit times -- more sensitive to the exact epoch (and
        to narrow transits) at a roughly linear cost in kernel time.
        This compiles the kernel's ``T0_OVERSAMPLE`` ``#define`` and
        mirrors :func:`cuvarbase.tls_grids.t0_grid_size`'s ``oversample``.
        The reference ``transitleastsquares`` package steps t0 about
        100x finer than a duration (every cadence for dense data); the
        default of 3 trades fidelity for speed -- see
        :func:`tls_search_gpu` for the measured cost.

    Returns
    -------
    kernels : dict
        Dictionary with 'standard' and 'keplerian' kernel functions

    Notes
    -----
    The kernels stage the data and a limb-darkened transit template in
    shared memory for physically realistic fitting (the depth/chi2
    accumulations are order-independent, so no phase sort is needed).
    The shared-memory layout caps datasets at ~3,500 points; see
    tls_search_gpu, which raises ValueError above the budget.

    The 'keplerian' kernel accepts per-period qmin/qmax arrays and is
    the one every legacy-path search launches since 1.0 (the default
    duration window is Keplerian, and the fixed opt-in window is passed
    as constant arrays). The 'standard' kernel hard-codes the pre-1.0
    constant window [0.005, 0.15] and is retained only for API
    compatibility of this dict; no wrapper launches it.
    """
    # Compiling a kernel needs an active CUDA context (lazily created).
    ensure_context()

    cppd = dict(BLOCK_SIZE=block_size,
                T0_OVERSAMPLE=float(t0_oversample))

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
        Pinned CPU arrays for time, flux, uncertainties. ``t`` holds
        the times MINUS ``epoch`` (see below), cast to float32 after
        the subtraction so that BJD-scale inputs keep their phase
        precision.
    epoch : float
        ``floor(min(t))`` of the last ``setdata`` call (0.0 before any
        data is set); the legacy kernel folds relative to it, so its
        per-period ``best_t0`` phases are relative to ``epoch``.
    t_g, y_g, dy_g : gpuarray
        GPU arrays for data
    periods_g, chi2_g : gpuarray
        GPU arrays for periods and chi-squared values
    best_t0_g, best_duration_g, best_depth_g : gpuarray
        GPU arrays for best-fit parameters
    """

    def __init__(self, max_ndata, max_nperiods, stream=None, **kwargs):
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import).
        ensure_context()
        self.max_ndata = max_ndata
        self.max_nperiods = max_nperiods
        self.stream = stream
        self.rtype = np.float32
        # Pinned (page-locked) host buffers by default for async overlap;
        # graceful fallback to page-aligned if pinning fails.
        self.pinned = kwargs.get('pinned', True)
        # floor(min(t)) subtracted from the times in setdata
        self.epoch = 0.0

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
        self.template_g = None

        self.allocate_host_arrays()

    def allocate_host_arrays(self):
        """Allocate host transfer buffers (page-locked by default, with a
        page-aligned fallback if pinning fails)."""
        p = self.pinned
        nd, npd = (self.max_ndata,), (self.max_nperiods,)

        self.t = host_array(nd, self.rtype, pinned=p)
        self.y = host_array(nd, self.rtype, pinned=p)
        self.dy = host_array(nd, self.rtype, pinned=p)

        self.periods = host_array(npd, self.rtype, pinned=p)
        self.chi2 = host_array(npd, self.rtype, pinned=p)
        self.best_t0 = host_array(npd, self.rtype, pinned=p)
        self.best_duration = host_array(npd, self.rtype, pinned=p)
        self.best_depth = host_array(npd, self.rtype, pinned=p)

        # Keplerian duration constraints
        self.qmin = host_array(npd, self.rtype, pinned=p)
        self.qmax = host_array(npd, self.rtype, pinned=p)

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

    def set_template(self, template):
        """Transfer transit template to GPU.

        Parameters
        ----------
        template : ndarray
            Float32 template array from generate_transit_template()
        """
        template = np.asarray(template, dtype=self.rtype)
        self.template_g = gpuarray.to_gpu(template)

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

        # Subtract the epoch floor(min t) in float64 BEFORE the float32
        # cast: folding raw BJD-scale float32 times loses the phase
        # entirely (float32 resolves 0.25 d at 2.45e6), and the fold
        # origin must be the same floor(min t) the fast path uses so
        # that 't0_phase' means the same thing on both paths.
        t64 = np.asarray(t, dtype=np.float64)
        self.epoch = float(np.floor(t64.min())) if ndata else 0.0
        self.t[:ndata] = (t64 - self.epoch).astype(self.rtype)
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

    def set_duration_bounds(self, qmin, qmax):
        """Stage and transfer per-period duration bounds only (used when
        the caller manages the data transfer itself with
        ``transfer_to_device=False``)."""
        nperiods = len(qmin)
        self.qmin[:nperiods] = np.asarray(qmin).astype(self.rtype)
        self.qmax[:nperiods] = np.asarray(qmax).astype(self.rtype)
        if self.qmin_g is None or len(self.qmin_g) < nperiods:
            self.qmin_g = gpuarray.zeros(nperiods, dtype=self.rtype)
            self.qmax_g = gpuarray.zeros(nperiods, dtype=self.rtype)
        if self.stream is None:
            self.qmin_g.set(self.qmin[:nperiods])
            self.qmax_g.set(self.qmax[:nperiods])
        else:
            self.qmin_g.set_async(self.qmin[:nperiods], stream=self.stream)
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
            # The host buffers are page-locked, so these copies are
            # genuinely asynchronous; callers read them immediately after
            # this returns, so sync here (matches BLSBatchMemory).
            self.stream.synchronize()

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


def tls_search_gpu(t, y, dy, periods=None,
                   qmin=None, qmax=None, n_durations=15,
                   R_star=1.0, M_star=1.0,
                   period_min=None, period_max=None, n_transits_min=2,
                   oversampling_factor=3, duration_grid_step=1.1,
                   R_planet_min=0.5, R_planet_max=5.0,
                   limb_dark='quadratic', u=[0.4804, 0.1867],
                   block_size=None, t0_oversample=3.0,
                   kernel=None, memory=None, stream=None,
                   transfer_to_device=True, transfer_to_host=True,
                   use_fast=True, refine_top_k=50,
                   refine_oversample=33.0, nbins=None,
                   R_planet=1.0, qmin_fac=0.5, qmax_fac=2.0,
                   duration_window='keplerian', sde_kernel_size=None,
                   **kwargs):
    """
    Run Transit Least Squares search on GPU.

    Parameters
    ----------
    t : array_like
        Observation times (days). Absolute BJD-scale times are safe on
        both paths: the epoch ``floor(min(t))`` is subtracted in float64
        before any float32 cast (the fast path then folds with a
        float-float pair; the legacy path folds the shifted float32
        times, so its phase precision degrades with the baseline, about
        1e-4 d at 1400 d).
    y : array_like
        Fluxes, normalized so the out-of-transit baseline is ~1.0. The
        transit model is ``1 - depth * T`` with a FIXED baseline of 1:
        no TLS path rescales the input or fits a free out-of-transit
        level (see Notes), so unnormalized flux (e.g. raw counts) gives
        meaningless depths.
    dy : array_like
        Flux uncertainties
    periods : array_like, optional
        Custom period grid (any order; sorted internally, and every
        per-period output array is returned in the caller's order). If
        None, generated automatically (Ofir 2014 grid).
    qmin, qmax : array_like, optional
        Explicit per-period fractional duration bounds (aligned with
        ``periods``; give both or neither). When omitted the window is
        built by :func:`cuvarbase.tls_grids.duration_window` from the
        stellar parameters (see ``duration_window``).
    n_durations : int, optional
        Number of log-spaced trial durations per period (default: 15).
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
    t0_oversample : float, optional (default: 3.0)
        Transit-epoch (t0) trial positions tested per transit duration.
        The on-device epoch stride is ``duration_phase / t0_oversample``;
        larger values resolve the transit time more finely (and recover
        narrower transits) at a roughly linear increase in kernel time.
        The reference ``transitleastsquares`` steps ~100x finer (every
        cadence for dense data). Measured cost of the default 3: the
        SDE of a P = 7.3 d, q = 0.021 transit varies by 17% (19.8-23.4)
        with the injected epoch relative to the coarse grid (6.5% at
        33); for a narrow transit (M dwarf, 3.4 cadences of 30 min)
        SDE 29.1 at 3 vs 32.9 at 10 and 32.7 at 33 (-11%). Raise it to
        10 (matches 33 within 1% in those runs) for sensitivity-critical
        searches. Distinct values compile and cache distinct kernels.
        See :func:`cuvarbase.tls_grids.t0_grid_size` for the resulting
        grid size.
    kernel : PyCUDA function, optional
        Pre-compiled kernel (legacy path). Must be the ``'keplerian'``
        kernel of :func:`compile_tls` (per-period duration bounds);
        the ``'standard'`` kernel has a different signature and is no
        longer launched by any wrapper.
    memory : TLSMemory, optional
        Pre-allocated memory object (legacy path)
    stream : cuda.Stream, optional
        CUDA stream for async execution (legacy path)
    transfer_to_device : bool, optional
        Transfer data to GPU (default: True). With False the caller must
        have staged ``t``, ``y``, ``dy`` and an ASCENDING ``periods``
        grid through ``memory.setdata`` (which epoch-subtracts the
        times); a non-ascending grid raises ValueError. The per-period
        duration bounds are uploaded here regardless.
    transfer_to_host : bool, optional
        Transfer results to CPU (default: True)
    use_fast : bool, optional (default: True)
        Use the phase-binned batch engine with exact top-K refinement
        (no ndata cap; see :func:`tls_search_batch`). Ignored with a
        warning when a pre-compiled kernel, external memory/stream, or
        transfer control is supplied — those fall back to the legacy
        per-point kernel.
    refine_top_k : int, optional (default: 50)
        Fast path only: number of best candidate periods per lightcurve
        re-fit exactly on a finer local (duration, t0) grid (0
        disables).
    refine_oversample : float, optional (default: 33.0)
        Fast path only: refinement epoch stride = duration / this.
    nbins : int, optional
        Fast path only: phase-bin override (power of two). By default
        the period grid is banded into per-band bin counts
        automatically.
    R_planet : float, optional
        Fiducial planet radius (Earth radii) of the Keplerian duration
        window (default: 1.0)
    qmin_fac, qmax_fac : float, optional
        Keplerian duration window factors: search ``[qmin_fac,
        qmax_fac] * q_kep(P)`` at each period (default 0.5, 2.0)
    duration_window : {'keplerian', 'fixed'}, optional
        Duration window used when ``qmin``/``qmax`` are omitted.
        'keplerian' (default) derives per-period bounds from
        ``R_star``/``M_star``/``R_planet`` (the same window
        :func:`tls_transit` and :func:`tls_search_batch` use). 'fixed'
        is the pre-1.0 constant window [0.005, 0.15] at every period,
        kept as an opt-in that warns when the Keplerian duration falls
        outside it: beyond P ~ 60 d (Sun-like) no trial duration is
        physical there and a transit is returned at an alias period
        with a biased depth (measured: P = 365 d on a 1400-d baseline
        came back at 182.5 d with half the depth).
    sde_kernel_size : int, optional
        Running-median window of the SDE detrend (see
        :func:`cuvarbase.tls_stats.signal_detection_efficiency`).

    Returns
    -------
    results : dict
        Dictionary with keys:

        - 'periods': trial periods (the caller's grid and order)
        - 'chi2': chi-squared per trial period (NaN where no valid
          solution)
        - 'best_t0_per_period', 'best_duration_per_period',
          'best_depth_per_period', 'valid_periods', 'n_failed_periods'
        - 'period', 'period_uncertainty': best-fit period (days)
        - 'T0': absolute mid-transit time (days, same scale as ``t``)
          of the first transit at or after ``min(t)``, i.e.
          ``min(t) <= T0 < min(t) + period`` -- the convention of the
          reference package. Fold with ``((t - T0) / period) % 1`` to
          put the transit at phase 0.
        - 't0_phase': the same epoch as a fold phase in [0, 1) relative
          to ``floor(min(t))``: ``T0 = floor(min(t)) + t0_phase *
          period`` shifted by whole periods into the range above.
        - 'duration' (days), 'depth' (fractional), 'chi2_min'
        - 'SDE', 'SDE_raw': signal detection efficiency of the
          per-period spectrum, ``SR = chi2_min / chi2`` (the reference
          definition; see :mod:`cuvarbase.tls_stats`)
        - 'SNR': ``sqrt(chi2_0 - chi2_min)``, the delta-chi-squared
          significance of the best fit over the constant model
        - 'power', 'SR': detrended / raw signal-residue spectra
        - 'n_transits', 'R_star', 'M_star'

        There is NO 'FAP' key: the pre-1.0 value was an uncalibrated
        function of the SDE (23% of pure-noise light curves got
        FAP < 0.01). Use ``tls_search_batch(fap_null_draws=N)`` for an
        empirical, per-configuration false-alarm probability.

        A light curve with no valid solution at any trial period (flat
        or noiseless flux) returns SDE = 0, NaN best-fit parameters and
        the message under 'error' (with a warning) instead of raising.

    Notes
    -----
    The default fast path binds the data once per period into phase
    bins and refines the best candidates exactly; the legacy path
    (``use_fast=False``) is the original per-point kernel, capped at
    ~3,500 points by its shared-memory layout.

    No free baseline term. The model is ``1 - depth * T(phase)`` with
    the out-of-transit level fixed at exactly 1 (shared with the
    reference package). A flux-normalization offset of a fraction of
    the per-point scatter changes the SDE materially and asymmetrically
    (measured, P = 7.3 d, sigma = 1e-3: +5e-4 raised the SDE from 21.5
    to 29.3 with the depth 20% low; -5e-4 halved it to 10.1; -1e-3 gave
    the wrong period). Normalize to a median (not mean) out-of-transit
    level of 1 to ~0.1 sigma per point before searching.
    """
    # Validate the light curve before anything else: the automatic
    # period grid is built from t, and a NaN sample or dy = 0 used to
    # travel all the way to the kernel (chi2 off by a factor ~1e3 on
    # the fast path; Sep 2026 audit, defect 23).
    check_lightcurve(t, y, dy, min_n=_TLS_MIN_NDATA, name='tls_search_gpu')

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

    # The fast path keeps t in float64 for epoch subtraction; only the
    # legacy path (below) downcasts inputs to float32 up front.
    periods = np.asarray(_validate_periods(periods), dtype=np.float32)
    nperiods = len(periods)

    # ---- Per-period duration window (caller's grid order) ----
    if (qmin is None) != (qmax is None):
        raise ValueError("provide both qmin and qmax, or neither")
    if qmin is not None:
        if duration_window != 'keplerian':
            raise ValueError("duration_window applies only when qmin/qmax "
                             "are not given")
        qmin_arr = np.asarray(qmin, dtype=np.float64)
        qmax_arr = np.asarray(qmax, dtype=np.float64)
        if len(qmin_arr) != nperiods or len(qmax_arr) != nperiods:
            raise ValueError(
                "qmin and qmax must have same length as periods "
                "(%d)" % nperiods)
    else:
        qmin_arr, qmax_arr = tls_grids.duration_window(
            periods.astype(np.float64), R_star=R_star, M_star=M_star,
            R_planet=R_planet, qmin_fac=qmin_fac, qmax_fac=qmax_fac,
            window=duration_window)
    _validate_q_window(qmin_arr, qmax_arr, periods=periods)

    # Fast path: phase-binned batch engine with exact top-K refinement.
    # Falls through to the legacy per-point kernel when the caller uses
    # the low-level plumbing (pre-compiled kernel, external memory or
    # stream, or transfer control), which the batch engine does not
    # expose.
    fast_gate = (kernel is None and memory is None and stream is None
                 and transfer_to_device and transfer_to_host)
    if use_fast and not fast_gate:
        warnings.warn(
            "use_fast=True is ignored because a pre-compiled kernel, "
            "external memory/stream, or transfer control was supplied; "
            "falling back to the legacy per-point kernel (which caps "
            "ndata at ~3,500 points)")
    if use_fast and fast_gate:
        r = tls_search_batch(
            [(t, y, dy)],
            periods=periods, qmin=qmin_arr, qmax=qmax_arr,
            n_durations=n_durations, t0_oversample=t0_oversample,
            refine_top_k=refine_top_k,
            refine_oversample=refine_oversample,
            block_size=block_size, nbins=nbins,
            limb_dark=limb_dark, u=u,
            R_star=R_star, M_star=M_star,
            return_arrays=True, sde_kernel_size=sde_kernel_size,
            _warn_failed=True)[0]

        results = {
            'periods': periods,
            'chi2': r['chi2'],
            'best_t0_per_period': r['best_t0_per_period'],
            'best_duration_per_period': r['best_duration_per_period'],
            'best_depth_per_period': r['best_depth_per_period'],
            'valid_periods': r['valid_periods'],
            'n_failed_periods': r['n_failed_periods'],
            'period': r['period'],
            'period_uncertainty': r['period_uncertainty'],
            'T0': r['T0'],
            't0_phase': r['t0_phase'],
            'duration': r['duration'],
            'depth': r['depth'],
            'chi2_min': r['chi2_min'],
            'SDE': r['SDE'],
            'SDE_raw': r['SDE_raw'],
            'SNR': r['SNR'],
            'power': r['power'],
            'SR': r['SR'],
            'n_transits': r['n_transits'],
            'R_star': R_star,
            'M_star': M_star,
        }
        if 'error' in r:
            results['error'] = r['error']
        return results

    # ---- Legacy per-point kernel path ----

    # float64 copies for the epoch, span and chi2_0; the kernel inputs
    # are cast to float32 by TLSMemory.setdata (after epoch subtraction)
    t64 = np.asarray(t, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    dy64 = np.asarray(dy, dtype=np.float64)
    ndata = len(t64)
    if len(y64) != ndata or len(dy64) != ndata:
        raise ValueError("t, y, dy lengths differ (%d, %d, %d)"
                         % (ndata, len(y64), len(dy64)))

    # Ascending trial grid for the statistics; the duration window is
    # aligned with the caller's order, so reorder it the same way.
    periods_sorted, order = _sort_period_grid(periods)
    if order is not None:
        if memory is not None and not transfer_to_device:
            raise ValueError(
                "transfer_to_device=False requires an ascending period "
                "grid: the periods staged on the device through "
                "memory.setdata must match the sorted grid the "
                "statistics assume")
        qmin_arr = qmin_arr[order]
        qmax_arr = qmax_arr[order]
    qmin32 = np.ascontiguousarray(qmin_arr, dtype=np.float32)
    qmax32 = np.ascontiguousarray(qmax_arr, dtype=np.float32)

    # Choose block size
    if block_size is None:
        block_size = _choose_block_size(ndata)

    # Shared-memory budget check BEFORE compiling kernels or touching
    # the GPU. Layout: phases[ndata] + y_sorted[ndata] +
    # dy_sorted[ndata] + template[n_template] + 4 thread arrays of
    # block_size floats, 4 bytes each. The default CUDA cap of 48 KB
    # per block bounds ndata at ~3,500 points for the default
    # template/block sizes.
    n_template = kwargs.get('n_template', 1000)
    shared_mem_size = (3 * ndata + n_template + 4 * block_size) * 4
    if shared_mem_size > _SHARED_MEM_LIMIT:
        max_ndata = (_SHARED_MEM_LIMIT // 4
                     - n_template - 4 * block_size) // 3
        raise ValueError(
            "ndata=%d requires %d bytes of shared memory per block but "
            "the kernel limit is %d: the TLS kernels support at most "
            "~%d points with n_template=%d and block_size=%d. Bin or "
            "split the light curve." % (ndata, shared_mem_size,
                                        _SHARED_MEM_LIMIT, max_ndata,
                                        n_template, block_size))

    # Get or compile kernels. Every legacy search runs the 'keplerian'
    # kernel (per-period duration bounds); the 'standard' kernel with
    # its hard-coded [0.005, 0.15] window is no longer launched.
    if kernel is None:
        kernels = _get_cached_kernels(block_size, t0_oversample=t0_oversample)
        kernel = kernels['keplerian']

    # Allocate or use existing memory (setdata epoch-subtracts t)
    if memory is None:
        memory = TLSMemory(ndata, nperiods, stream=stream)
        memory.setdata(t64, y64, dy64, periods=periods_sorted,
                       qmin=qmin32, qmax=qmax32,
                       transfer=transfer_to_device)
    elif transfer_to_device:
        memory.setdata(t64, y64, dy64, periods=periods_sorted,
                       qmin=qmin32, qmax=qmax32, transfer=True)
    else:
        # the caller staged t/y/dy/periods; the duration bounds are ours
        memory.set_duration_bounds(qmin32, qmax32)

    # Generate and transfer transit template (n_template and
    # shared_mem_size were computed with the guard above)
    if memory.template_g is None:
        template = tls_models.generate_transit_template(
            n_template=n_template, limb_dark=limb_dark, u=u
        )
        memory.set_template(template)

    # Launch kernel
    grid = (nperiods, 1, 1)
    block = (block_size, 1, 1)

    kernel_args = [
        memory.t_g, memory.y_g, memory.dy_g,
        memory.periods_g, memory.qmin_g, memory.qmax_g,
        memory.template_g,
        np.int32(ndata), np.int32(nperiods), np.int32(n_durations),
        np.int32(n_template),
        memory.chi2_g, memory.best_t0_g,
        memory.best_duration_g, memory.best_depth_g,
    ]

    kernel_kwargs = dict(block=block, grid=grid, shared=shared_mem_size)
    if stream is not None:
        kernel_kwargs['stream'] = stream

    kernel(*kernel_args, **kernel_kwargs)

    # Transfer results if requested
    if transfer_to_host:
        if stream is not None:
            stream.synchronize()
        memory.transfer_from_gpu(nperiods)

        chi2_vals = memory.chi2[:nperiods].copy()
        best_t0_vals = memory.best_t0[:nperiods].copy()
        best_duration_vals = memory.best_duration[:nperiods].copy()
        best_depth_vals = memory.best_depth[:nperiods].copy()

        # constant-model chi2 (float64) for the SNR; the kernel uses the
        # same sigma^2 + 1e-10 regularizer
        chi2_0 = float(np.sum((1.0 - y64) ** 2 / (dy64 ** 2 + 1e-10)))
        tmin = float(t64.min())
        epoch = getattr(memory, 'epoch', None)
        if epoch is None:
            epoch = float(np.floor(tmin))

        # Mask failed periods (1e30 sentinel) before any statistics:
        # unmasked they collapse SDE to ~0
        valid = _mask_failed_periods(chi2_vals)
        if not valid.any():
            results = _null_result(nperiods, chi2_0,
                                   _NO_SOLUTION_MSG % nperiods,
                                   periods=periods, arrays=True)
            results.update({'R_star': R_star, 'M_star': M_star})
            return results
        chi2_valid = chi2_vals[valid]
        periods_valid = periods_sorted[valid]

        # Find best period among the valid ones
        best_valid_idx = int(np.argmin(chi2_valid))
        best_idx = int(np.flatnonzero(valid)[best_valid_idx])
        best_period = float(periods_sorted[best_idx])
        best_chi2 = float(chi2_vals[best_idx])
        best_t0 = float(best_t0_vals[best_idx])
        best_duration = float(best_duration_vals[best_idx])
        best_depth = float(best_depth_vals[best_idx])

        # Estimate number of transits
        T_span = float(t64.max() - tmin)
        n_transits = int(T_span / best_period)

        # Compute statistics on the valid periods only
        stats = tls_stats.compute_all_statistics(
            chi2_valid, periods_valid, best_valid_idx,
            best_depth, best_duration, n_transits,
            kernel_size=sde_kernel_size,
            chi2_null=chi2_0, chi2_best=best_chi2)

        # Period uncertainty
        period_uncertainty = tls_stats.compute_period_uncertainty(
            periods_valid, chi2_valid, best_valid_idx
        )

        # Absolute mid-transit time: the kernel's phase is relative to
        # the epoch floor(min t); report the first transit >= min(t)
        T0 = _first_transit_at_or_after(epoch + best_t0 * best_period,
                                        best_period, tmin)

        # Failed periods appear as NaN in the returned spectra; every
        # per-period array goes back to the caller's grid order
        def _expand(values):
            full = np.full(nperiods, np.nan)
            full[valid] = values
            return _to_caller_order(full, order)

        results = {
            # Raw outputs (NaN at failed periods)
            'periods': periods,
            'chi2': _to_caller_order(np.where(valid, chi2_vals, np.nan),
                                     order),
            'best_t0_per_period': _to_caller_order(best_t0_vals, order),
            'best_duration_per_period': _to_caller_order(
                best_duration_vals, order),
            'best_depth_per_period': _to_caller_order(best_depth_vals,
                                                      order),
            'valid_periods': _to_caller_order(valid, order),
            'n_failed_periods': int(nperiods - valid.sum()),

            # Best-fit parameters
            'period': best_period,
            'period_uncertainty': period_uncertainty,
            'T0': T0,
            't0_phase': best_t0,
            'duration': best_duration,
            'depth': best_depth,
            'chi2_min': best_chi2,

            # Statistics (computed on valid periods, expanded to the
            # full grid with NaN at failed periods)
            'SDE': stats['SDE'],
            'SDE_raw': stats['SDE_raw'],
            'SNR': stats['SNR'],
            'power': _expand(stats['power']),
            'SR': _expand(stats['SR']),

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
    check_lightcurve(t, y, dy, min_n=_TLS_MIN_NDATA, name='tls_search')
    return tls_search_gpu(t, y, dy, **kwargs)


def tls_transit(t, y, dy, R_star=1.0, M_star=1.0, R_planet=1.0,
                qmin_fac=0.5, qmax_fac=2.0, n_durations=15,
                period_min=None, period_max=None, n_transits_min=2,
                oversampling_factor=3, **kwargs):
    """
    Transit Least Squares search with Keplerian duration constraints.

    This is the TLS analog of BLS's eebls_transit() function. It uses stellar
    parameters to focus the duration search on physically plausible values.
    Since 1.0 :func:`tls_search_gpu` builds the same Keplerian window by
    default, so this wrapper is equivalent to ``tls_search_gpu(t, y, dy,
    R_star=..., M_star=..., R_planet=..., qmin_fac=..., qmax_fac=...)``
    and is kept for its explicit name and the explicit qmin/qmax it
    passes.

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
        - 'T0': absolute mid-transit time (days, same scale as ``t``)
          of the first transit at or after min(t); 't0_phase' is the
          fold phase relative to floor(min(t))
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
    range (0.5%-15%) at all periods -- and, unlike that fixed window,
    stays physical at long periods (the fixed window excludes the
    Keplerian duration beyond P ~ 60 d for a Sun-like star).

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
    tls_grids.duration_window : Per-period duration bounds (used here)
    tls_grids.q_transit : Calculate Keplerian fractional duration
    """
    check_lightcurve(t, y, dy, min_n=_TLS_MIN_NDATA, name='tls_transit')

    # Generate period grid
    periods = tls_grids.period_grid_ofir(
        t, R_star=R_star, M_star=M_star,
        oversampling_factor=oversampling_factor,
        period_min=period_min, period_max=period_max,
        n_transits_min=n_transits_min
    )

    # Per-period Keplerian duration bounds. These are the same bounds
    # duration_grid_keplerian returns as ``q_values * (qmin_fac,
    # qmax_fac)`` -- tls_grids.duration_window is the shared window
    # helper every other TLS entry point uses -- but without building
    # the (nperiods x n_durations) duration table, which nothing
    # downstream reads: tls_search_gpu takes only qmin/qmax and
    # n_durations. Measured on an A40 (shared), old and new bodies
    # interleaved in one process: tls_transit 4.51 -> 3.71 ms at 2,486
    # trial periods, 43.01 -> 24.70 at 42,001, 219.49 -> 159.75 at
    # 171,688 (the table alone costs 0.80 / 13.92 / 58.95 ms).
    qmin, qmax = tls_grids.duration_window(
        periods, R_star=R_star, M_star=M_star, R_planet=R_planet,
        qmin_fac=qmin_fac, qmax_fac=qmax_fac
    )

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


# =====================================================================
# Fast batch TLS engine (phase-binned scan + exact top-K refinement)
# =====================================================================
#
# One kernel launch searches a whole batch of lightcurves over a shared
# period grid: grid = (nperiods, n_lightcurves), one block per
# (lightcurve, period). Each block folds its lightcurve once into
# shared-memory phase bins and scans every (duration, t0) trial against
# the bins, so trial cost is independent of ndata and there is no
# shared-memory cap on the lightcurve length. A second, exact kernel
# then re-fits the best `refine_top_k` candidate periods per lightcurve
# with per-point template evaluation on a finer local (duration, t0)
# grid. See kernels/tls_fast.cu for the algorithm notes.

_TLS_FAST_NTEMPLATE = 1024
_TLS_FAST_MAX_DURATIONS = 64
_TLS_FAST_MAX_NBINS = 8192
_TLS_FAST_DEFAULT_BLOCK = 256

# Chunking budgets (per kernel launch)
_TLS_FAST_MAX_OUT_FLOATS = 32 * 1024 * 1024   # per output array
_TLS_FAST_MAX_POINTS = 16 * 1024 * 1024       # concatenated data points
_TLS_FAST_MAX_GRID_Y = 65535


def _device_max_shared():
    """Max opt-in dynamic shared memory per block on the current device."""
    ensure_context()
    dev = cuda.Context.get_device()
    try:
        return dev.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    except Exception:
        return dev.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)


def _tls_fast_shared_size(block_size, nbins):
    """Dynamic shared memory (bytes) for tls_fast_search_kernel."""
    nt = _TLS_FAST_NTEMPLATE
    md = _TLS_FAST_MAX_DURATIONS
    n_floats = 2 * nbins + 2 * (nt + 1) + 4 * block_size + md
    n_ints = md + 1
    return 4 * (n_floats + n_ints)


def _tls_refine_shared_size(block_size):
    """Dynamic shared memory (bytes) for tls_refine_kernel."""
    return 4 * ((_TLS_FAST_NTEMPLATE + 1) + 4 * (block_size // 32))


def compile_tls_fast(block_size=_TLS_FAST_DEFAULT_BLOCK, nbins=2048,
                     t0_oversample=3.0, refine_nd=3):
    """
    Compile the fast (batched, phase-binned) TLS kernels.

    Parameters
    ----------
    block_size : int
        CUDA block size (multiple of 32).
    nbins : int
        Number of phase bins (power of two).
    t0_oversample : float
        Epoch oversampling: t0 stride = duration / t0_oversample in the
        coarse scan (same convention as the legacy kernels).
    refine_nd : int
        Number of local durations in the refinement kernel (odd;
        default 3 spans one coarse duration-grid step each way).

    Returns
    -------
    kernels : dict
        {'search': ..., 'refine': ...} PyCUDA functions.
    """
    ensure_context()
    if block_size < 32 or (block_size & (block_size - 1)):
        # the block max-reduction assumes a power-of-two blockDim
        raise ValueError("block_size must be a power of two >= 32")
    if nbins & (nbins - 1):
        raise ValueError("nbins must be a power of two")
    if int(refine_nd) != refine_nd or refine_nd < 2:
        raise ValueError("refine_nd must be an integer >= 2 "
                         "(odd recommended so the coarse duration sits "
                         "on the refinement grid)")

    cppd = dict(BLOCK_SIZE=block_size,
                NBINS=nbins,
                NTEMPLATE=_TLS_FAST_NTEMPLATE,
                MAX_DURATIONS=_TLS_FAST_MAX_DURATIONS,
                T0_OVERSAMPLE=float(t0_oversample),
                REFINE_ND=refine_nd)
    kernel_txt = _module_reader(find_kernel('tls_fast'), cpp_defs=cppd)
    module = SourceModule(kernel_txt, options=['--use_fast_math'],
                          no_extern_c=True)
    search = module.get_function('tls_fast_search_kernel')
    refine = module.get_function('tls_refine_kernel')

    smem = _tls_fast_shared_size(block_size, nbins)
    if smem > _SHARED_MEM_LIMIT:
        max_shared = _device_max_shared()
        if smem > max_shared:
            raise ValueError(
                "TLS fast kernel wants %d bytes of shared memory per "
                "block but the device caps at %d; reduce nbins (or "
                "block_size)" % (smem, max_shared))
        # opt in to >48KB dynamic shared memory (sm_70+)
        search.set_attribute(
            cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)

    return {'search': search, 'refine': refine}


def _get_cached_fast_kernels(block_size, nbins, t0_oversample,
                             refine_nd=3):
    key = ('fast', block_size, nbins, float(t0_oversample), refine_nd)
    with _kernel_cache_lock:
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]
        compiled = compile_tls_fast(block_size=block_size, nbins=nbins,
                                    t0_oversample=t0_oversample,
                                    refine_nd=refine_nd)
        _kernel_cache[key] = compiled
        _kernel_cache.move_to_end(key)
        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)
        return compiled


def _preprocess_batch(lightcurves):
    """Epoch-subtract, weight, and concatenate lightcurves (float64
    accumulation; times stored as a float-float hi/lo pair so the
    kernels can fold at ~float64 precision with pure FP32 math).

    Returns (t_hi, t_lo, a_c, b_c, offs, lens, chi2_0, epochs, spans);
    chi2_0 stays float64 for cancellation-free chi2 reconstruction.
    """
    n_lc = len(lightcurves)
    lens = np.array([len(lc[0]) for lc in lightcurves], dtype=np.int64)
    for i, (lc, n) in enumerate(zip(lightcurves, lens)):
        if n > np.iinfo(np.int32).max:
            raise ValueError(
                "lightcurve %d has %d points; the TLS kernels index "
                "points within a chunk with int32" % (i, n))
        # equal lengths, finite t/y/dy, dy > 0 (dy = 0 gave a chi2
        # 1.3e3 times too large on the fast path; Sep 2026 audit,
        # defect 23)
        check_lightcurve(lc[0], lc[1], lc[2], min_n=_TLS_MIN_NDATA,
                         name='lightcurve %d' % i)
    # batch-wide offsets in int64 (a large survey can exceed 2^31
    # total points); per-chunk offsets are rebased and cast to int32
    # at upload, where the chunk-size cap keeps them small
    offs = np.zeros(n_lc, dtype=np.int64)
    if n_lc > 1:
        offs[1:] = np.cumsum(lens)[:-1]
    total = int(lens.sum())

    t_hi = np.empty(total, dtype=np.float32)
    t_lo = np.empty(total, dtype=np.float32)
    a_c = np.empty(total, dtype=np.float32)
    b_c = np.empty(total, dtype=np.float32)
    chi2_0 = np.empty(n_lc, dtype=np.float64)
    epochs = np.empty(n_lc, dtype=np.float64)
    spans = np.empty(n_lc, dtype=np.float64)

    for i, (t, y, dy) in enumerate(lightcurves):
        t64 = np.asarray(t, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        dy64 = np.asarray(dy, dtype=np.float64)
        epoch = np.floor(t64.min())
        # sigma^2 regularizer matches the legacy kernel (float32 dy)
        s2 = dy64 * dy64 + 1e-10
        o, n = int(offs[i]), int(lens[i])
        tshift = t64 - epoch
        hi = tshift.astype(np.float32)
        t_hi[o:o + n] = hi
        t_lo[o:o + n] = (tshift - hi.astype(np.float64)).astype(np.float32)
        resid = 1.0 - y64
        a_c[o:o + n] = resid / s2
        b_c[o:o + n] = 1.0 / s2
        chi2_0[i] = np.sum(resid * resid / s2)
        epochs[i] = epoch
        spans[i] = t64.max() - t64.min()

    return t_hi, t_lo, a_c, b_c, offs, lens, chi2_0, epochs, spans


def tls_search_batch(lightcurves, R_star=1.0, M_star=1.0, R_planet=1.0,
                     periods=None, qmin=None, qmax=None,
                     period_min=None, period_max=None,
                     n_transits_min=2, oversampling_factor=3,
                     qmin_fac=0.5, qmax_fac=2.0, n_durations=15,
                     t0_oversample=3.0,
                     refine_top_k=50, refine_oversample=33.0,
                     block_size=None, nbins=None,
                     limb_dark='quadratic', u=[0.4804, 0.1867],
                     return_arrays=False, sde_kernel_size=None,
                     fap_null_draws=0, fap_seed=None,
                     _warn_failed=False):
    """
    Survey-scale Transit Least Squares search over a batch of
    lightcurves sharing one trial-period grid.

    This is the fast path for N >> 1 lightcurves: a single kernel
    launch (per chunk) searches every (lightcurve, period) pair with a
    phase-binned scan, then an exact per-point refinement kernel
    re-fits the ``refine_top_k`` best candidate periods per lightcurve
    on a finer local (duration, t0) grid. There is no cap on ndata.

    Parameters
    ----------
    lightcurves : list of (t, y, dy) tuples
        Times (days), fluxes (normalized to a baseline of 1.0), and
        flux uncertainties. Each lightcurve's epoch floor(min(t)) is
        subtracted internally (float64), so BJD-scale times are safe.
    R_star, M_star : float
        Stellar radius/mass in solar units; set the period grid and the
        Keplerian duration window (shared by all lightcurves).
    R_planet : float
        Fiducial planet radius (Earth radii) for the duration window.
    periods, qmin, qmax : array_like, optional
        Explicit trial grid: periods (days, any order -- sorted
        internally, per-period output arrays come back in the caller's
        order) and per-period fractional duration bounds aligned with
        ``periods``. Auto-generated (Ofir 2014 grid + Keplerian
        durations from :func:`cuvarbase.tls_grids.duration_window`)
        when omitted.
    period_min, period_max : float, optional
        Period search range for the auto grid.
    n_transits_min, oversampling_factor : optional
        Auto period-grid parameters (see tls_grids.period_grid_ofir).
    qmin_fac, qmax_fac : float
        Keplerian duration window factors (search [qmin_fac*q,
        qmax_fac*q] at each period).
    n_durations : int
        Trial durations per period (log-spaced), max 64.
    t0_oversample : float
        Coarse epoch oversampling; t0 stride = duration / t0_oversample.
    refine_top_k : int
        Number of best candidate periods per lightcurve re-fit exactly
        (default 50; 0 disables refinement).
    refine_oversample : float
        Refinement epoch stride = duration / refine_oversample (the
        reference transitleastsquares package uses ~100).
    block_size : int, optional
        CUDA block size override (power of two). By default each
        bin-count band picks its own (256, or 512 for bands with 4096+
        bins, shrunk to fit the device's shared-memory cap).
    nbins : int, optional
        Phase bins (power of two). Auto-sized so a bin is no wider than
        the narrowest trial duration / t0_oversample, within the
        device's shared-memory limit.
    limb_dark, u : optional
        Limb-darkening law/coefficients for the transit template.
    return_arrays : bool
        Also return the per-period chi2/t0/duration/depth arrays and
        derived spectra for each lightcurve (adds D2H transfer time).
    sde_kernel_size : int, optional
        Median-detrend window for the SDE statistic (see tls_stats).
    fap_null_draws : int, optional (default: 0)
        Opt-in empirical false-alarm probability. For each lightcurve,
        ``fap_null_draws`` null realizations are built by randomly
        permuting the (y, dy) pairs over the observation times (a
        white-noise null that keeps the sampling, the point count and
        the noise distribution but destroys any coherent signal and
        any red noise), searched on the identical trial grid and
        settings (coarse scan only; the SDE never uses the
        refinement), and the result gets ``'FAP' = (1 + n_exceed) /
        (fap_null_draws + 1)`` where ``n_exceed`` counts null SDEs
        >= the observed SDE, plus the null SDEs under ``'SDE_null'``.
        Cost: ``fap_null_draws`` extra searches per lightcurve
        (measured 400 pure-noise searches of 2880 points x 6157
        periods in 1.6 s on an A40). The smallest resolvable FAP is
        ``1 / (fap_null_draws + 1)``. No 'FAP' key is returned
        otherwise: the pre-1.0 value was an uncalibrated function of
        the SDE.
    fap_seed : int or None, optional
        Seed of the ``numpy.random.RandomState`` used for the null
        permutations (None: fresh entropy).

    Returns
    -------
    results : list of dict
        One dict per lightcurve:
        'period', 'period_uncertainty', 't0_phase' (fold phase of the
        mid-transit relative to floor(min t)), 'T0' (absolute
        mid-transit time of the first transit at or after min(t), so
        ``min(t) <= T0 < min(t) + period``; fold with
        ``((t - T0) / period) % 1``), 'duration', 'depth', 'chi2_min',
        'SDE', 'SDE_raw' (``SR = chi2_min / chi2`` statistic, see
        :mod:`cuvarbase.tls_stats`), 'SNR' (``sqrt(chi2_0 -
        chi2_min)``), 'n_transits', 'n_failed_periods'; plus the
        per-period arrays (in the caller's period order) when
        ``return_arrays`` is set, and 'FAP'/'SDE_null' when
        ``fap_null_draws`` > 0.

        A lightcurve with no valid solution at any trial period (flat
        or noiseless flux) gets the same keys with SDE = 0, NaN best-fit
        parameters and the message under 'error' (a warning is raised).

        The best-fit parameters (including 'chi2_min') come from the
        exact refinement pass, so 'chi2_min' is generally slightly
        below the minimum of the returned coarse 'chi2' spectrum; the
        SDE statistics are computed from the uniform coarse spectrum
        only, keeping the detection statistic's scale consistent
        across periods.
    """
    tls_grids.validate_stellar_parameters(R_star, M_star)
    tls_models.validate_limb_darkening_coeffs(u, limb_dark)

    if len(lightcurves) == 0:
        return []
    # Validate every light curve up front: the automatic period grid is
    # built from the longest baseline, and the kernels are compiled and
    # the trial grids uploaded well before _preprocess_batch runs.
    for i, lc in enumerate(lightcurves):
        if len(lc) != 3:
            raise ValueError("tls_search_batch: lightcurve %d must be a "
                             "(t, y, dy) tuple; got %d elements"
                             % (i, len(lc)))
        check_lightcurve(lc[0], lc[1], lc[2], min_n=_TLS_MIN_NDATA,
                         name='tls_search_batch lightcurve %d' % i)
    if n_durations < 2 or n_durations > _TLS_FAST_MAX_DURATIONS:
        raise ValueError("n_durations must be in [2, %d]" %
                         _TLS_FAST_MAX_DURATIONS)
    if refine_top_k is not None and refine_top_k < 0:
        raise ValueError("refine_top_k must be >= 0 (got %r)"
                         % (refine_top_k,))
    if refine_top_k and not refine_oversample > 0:
        raise ValueError("refine_oversample must be > 0 (got %r)"
                         % (refine_oversample,))

    # ---- Trial grid (shared across the batch) ----
    if periods is None:
        # build the grid from the longest lightcurve baseline
        spans_probe = [np.max(lc[0]) - np.min(lc[0]) for lc in lightcurves]
        t_ref = lightcurves[int(np.argmax(spans_probe))][0]
        periods = tls_grids.period_grid_ofir(
            t_ref, R_star=R_star, M_star=M_star,
            oversampling_factor=oversampling_factor,
            period_min=period_min, period_max=period_max,
            n_transits_min=n_transits_min)
    periods_in = np.asarray(_validate_periods(periods), dtype=np.float32)
    nperiods = len(periods_in)

    if (qmin is None) != (qmax is None):
        raise ValueError("provide both qmin and qmax, or neither")
    if qmin is None:
        # only the q bounds are needed here; skip building the
        # (nperiods x n_durations) duration table
        qmin, qmax = tls_grids.duration_window(
            periods_in.astype(np.float64), R_star=R_star, M_star=M_star,
            R_planet=R_planet, qmin_fac=qmin_fac, qmax_fac=qmax_fac)
    qmin = np.ascontiguousarray(qmin, dtype=np.float32)
    qmax = np.ascontiguousarray(qmax, dtype=np.float32)
    if len(qmin) != nperiods or len(qmax) != nperiods:
        raise ValueError("qmin and qmax must have same length as periods "
                         "(%d)" % nperiods)
    _validate_q_window(qmin, qmax, periods=periods_in)

    # The statistics (running-median detrend, period uncertainty)
    # assume an ascending grid: sort here, scatter outputs back to
    # the caller's order at the end.
    periods, order = _sort_period_grid(periods_in)
    if order is not None:
        qmin = np.ascontiguousarray(qmin[order])
        qmax = np.ascontiguousarray(qmax[order])

    # ---- Kernel configuration: band the grid by required bin count.
    # The trial-scan cost is proportional to NBINS, while the bin count
    # a period actually needs scales with 1/qmin at that period, so
    # running the whole grid at the finest band's NBINS overpays by 2x+
    # on long-baseline searches. Each band compiles (and caches) its
    # own NBINS variant and scatters results through period_map. ----
    qmin_global = float(np.min(qmin))
    max_dev_shared = _device_max_shared()
    ensure_context()
    cc_major = cuda.Context.get_device().compute_capability()[0]

    def _band_block_size(nb):
        if block_size is not None:
            return block_size
        # Swept on RTX A5000 (sm_86), RTX 4000 Ada (sm_89) and Tesla
        # V100 (sm_70), kepler-4yr config with the float-float fold:
        # 256 beats 128 everywhere; 512 wins on the big-bin bands on
        # Ampere/Ada from 4096 bins up, while Volta prefers 256 until
        # shared memory forces one block per SM (8192 bins).
        # On devices with a hard 48KB cap (no opt-in; Pascal and
        # earlier) prefer shrinking the block over losing phase bins.
        big_bin_threshold = 4096 if cc_major >= 8 else 8192
        bs = 512 if nb >= big_bin_threshold else 256
        while bs > 64 and _tls_fast_shared_size(bs, nb) > max_dev_shared:
            bs //= 2
        return bs

    need = t0_oversample / np.maximum(qmin.astype(np.float64), 1e-6)
    if nbins is None:
        nbins_per = np.power(
            2, np.ceil(np.log2(np.clip(need, 256, None)))).astype(np.int64)
        nbins_per = np.minimum(nbins_per, _TLS_FAST_MAX_NBINS)
        # shared-memory cap for this device
        while _tls_fast_shared_size(
                _band_block_size(int(nbins_per.max())),
                int(nbins_per.max())) > max_dev_shared:
            cap = int(nbins_per.max()) // 2
            nbins_per = np.minimum(nbins_per, cap)
            if cap <= 256:
                break
        short = need > nbins_per
        if np.any(short):
            warnings.warn(
                "TLS fast path: %d of %d trial periods have their "
                "narrowest durations under-resolved by the phase bins "
                "(device shared-memory cap); their coarse scan is "
                "smeared and recovery there relies on the exact "
                "refinement pass." % (int(short.sum()), nperiods))
        bands = [(int(nb), np.flatnonzero(nbins_per == nb).astype(np.int32))
                 for nb in np.unique(nbins_per)]
        smear = float(np.max(need / nbins_per))
    else:
        bands = [(int(nbins), np.arange(nperiods, dtype=np.int32))]
        smear = float(np.max(need / nbins))

    # When the coarse bins under-resolve a duration (smear > 1), the
    # coarse best duration is biased wide by the bin convolution;
    # widen the refinement's duration window accordingly and use more
    # local durations so the true value stays inside it.
    smear = max(1.0, smear)
    refine_nd = 3 if smear <= 1.3 else 5

    band_launches = []   # (kernels, block_size, smem, n, per_g, qmn_g, qmx_g, map_g)
    for nb, idx in bands:
        bs = _band_block_size(nb)
        kern = _get_cached_fast_kernels(bs, nb, t0_oversample,
                                        refine_nd=refine_nd)
        band_launches.append((
            kern, bs, _tls_fast_shared_size(bs, nb), len(idx),
            gpuarray.to_gpu(periods[idx]),
            gpuarray.to_gpu(qmin[idx]),
            gpuarray.to_gpu(qmax[idx]),
            gpuarray.to_gpu(idx)))

    # refinement runs at the first band's block size (any variant works)
    refine_bs = _band_block_size(bands[0][0])
    refine_kern = band_launches[0][0]
    refine_smem = _tls_refine_shared_size(refine_bs)

    # refinement trial-grid shape (see kernels/tls_fast.cu). The t0
    # halfwidth must cover the worst coarse quantization, which lives
    # in the FINEST band if the device cap clamped it below its need.
    dur_ratio = float(np.median(qmax / qmin))
    dur_span = dur_ratio ** (1.0 / (2.0 * max(n_durations - 1, 1)))
    dur_span *= min(smear, 4.0)
    nbins_finest = bands[-1][0]
    t0_halfwidth = min(3.0, max(0.5, 1.5 / (nbins_finest * qmin_global)))

    # ---- Template tables ----
    T_tab, S1_tab, S2_tab = tls_models.generate_template_tables(
        n_table=_TLS_FAST_NTEMPLATE, limb_dark=limb_dark, u=u)

    # ---- Host preprocessing ----
    t_hi_c, t_lo_c, a_c, b_c, offs, lens, chi2_0, epochs, spans = \
        _preprocess_batch(lightcurves)
    n_lc = len(lightcurves)
    tmins = np.array([np.min(np.asarray(lc[0], dtype=np.float64))
                      for lc in lightcurves], dtype=np.float64)

    # ---- Static GPU arrays ----
    periods_g = gpuarray.to_gpu(periods)
    T_g = gpuarray.to_gpu(T_tab)
    S1_g = gpuarray.to_gpu(S1_tab)
    S2_g = gpuarray.to_gpu(S2_tab)

    # ---- Chunk plan: bound output size, data size, and grid.y ----
    max_lcs_by_out = max(1, _TLS_FAST_MAX_OUT_FLOATS // max(nperiods, 1))
    chunks = []          # list of (i0, i1)
    i0 = 0
    while i0 < n_lc:
        i1 = i0 + 1
        pts = int(lens[i0])
        while (i1 < n_lc
               and i1 - i0 < max_lcs_by_out
               and i1 - i0 < _TLS_FAST_MAX_GRID_Y
               and pts + int(lens[i1]) <= _TLS_FAST_MAX_POINTS):
            pts += int(lens[i1])
            i1 += 1
        chunks.append((i0, i1))
        i0 = i1

    max_chunk_lcs = max(i1 - i0 for i0, i1 in chunks)
    max_chunk_pts = max(int(lens[i0:i1].sum()) for i0, i1 in chunks)

    # reusable per-chunk GPU buffers
    thi_g = gpuarray.empty(max_chunk_pts, np.float32)
    tlo_g = gpuarray.empty(max_chunk_pts, np.float32)
    a_g = gpuarray.empty(max_chunk_pts, np.float32)
    b_g = gpuarray.empty(max_chunk_pts, np.float32)
    off_g = gpuarray.empty(max_chunk_lcs, np.int32)
    len_g = gpuarray.empty(max_chunk_lcs, np.int32)
    out_n = max_chunk_lcs * nperiods
    score_g = gpuarray.empty(out_n, np.float32)
    t0_g = gpuarray.empty(out_n, np.float32)
    dur_g = gpuarray.empty(out_n, np.float32)
    depth_g = gpuarray.empty(out_n, np.float32)

    # Refinement targets the peak region only: capping K at ~10% of the
    # grid keeps the SDE background dominated by uniformly-treated
    # (coarse) periods, so the refined peak stands out the same way it
    # would in a full-fidelity spectrum.
    K = int(min(refine_top_k, max(16, nperiods // 10),
                nperiods)) if refine_top_k else 0
    if K:
        cand_g = gpuarray.empty(max_chunk_lcs * K, np.int32)
        # compact refined outputs, one slot per candidate; the coarse
        # spectrum is never overwritten (SDE needs uniform fidelity)
        rscore_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
        rt0_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
        rdur_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
        rdepth_g = gpuarray.empty(max_chunk_lcs * K, np.float32)

    results = [None] * n_lc

    for (i0, i1) in chunks:
        nc = i1 - i0
        p0 = int(offs[i0])
        pts = int(lens[i0:i1].sum())

        # H2D (chunk-relative offsets are bounded by the points cap,
        # so the int32 cast is safe)
        thi_g[:pts].set(t_hi_c[p0:p0 + pts])
        tlo_g[:pts].set(t_lo_c[p0:p0 + pts])
        a_g[:pts].set(a_c[p0:p0 + pts])
        b_g[:pts].set(b_c[p0:p0 + pts])
        off_g[:nc].set((offs[i0:i1] - p0).astype(np.int32))
        len_g[:nc].set(lens[i0:i1].astype(np.int32))

        # coarse binned scan, one launch per bin-count band
        for kern, bs, smem, band_n, per_g, qmn_g, qmx_g, map_g \
                in band_launches:
            kern['search'](
                thi_g, tlo_g, a_g, b_g, off_g, len_g,
                per_g, qmn_g, qmx_g, map_g, S1_g, S2_g,
                np.int32(band_n), np.int32(nperiods),
                np.int32(n_durations),
                score_g, t0_g, dur_g, depth_g,
                block=(bs, 1, 1), grid=(band_n, nc, 1),
                shared=smem)

        # score = chi2_0 - chi2 (cancellation-free); <= 0 marks failure
        score_h = score_g[:nc * nperiods].get().reshape(nc, nperiods)

        # exact refinement of the best K candidate periods per LC
        # (parameters only; the coarse spectrum feeds the statistics)
        rscore_h = rt0_h = rdur_h = rdepth_h = cand = None
        if K:
            cand = np.empty((nc, K), dtype=np.int32)
            for j in range(nc):
                if K < nperiods:
                    # K largest scores = K smallest chi2; failed
                    # periods (score < 0) sort last automatically
                    cand[j] = np.argpartition(-score_h[j], K)[:K]
                else:
                    cand[j] = np.arange(nperiods)
            cand_g[:nc * K].set(cand.ravel())
            refine_kern['refine'](
                thi_g, tlo_g, a_g, b_g, off_g, len_g,
                periods_g, cand_g, T_g,
                np.int32(nperiods), np.int32(K),
                np.float32(dur_span), np.float32(t0_halfwidth),
                np.float32(refine_oversample),
                t0_g, dur_g,
                rscore_g, rt0_g, rdur_g, rdepth_g,
                block=(refine_bs, 1, 1), grid=(K, nc, 1),
                shared=refine_smem)
            rscore_h = rscore_g[:nc * K].get().reshape(nc, K)
            rt0_h = rt0_g[:nc * K].get().reshape(nc, K)
            rdur_h = rdur_g[:nc * K].get().reshape(nc, K)
            rdepth_h = rdepth_g[:nc * K].get().reshape(nc, K)

        # Coarse per-period best-fit params. Needed when return_arrays is set,
        # when there is no refinement (K == 0), AND as the fallback in
        # _finish_lc when a light curve's top-K exact refinements all return
        # the sentinel (the else-branch below reads t0_h/dur_h/depth_h). Fetch
        # only when actually needed so the common default path pays no extra
        # D2H. (`rscore_h` is only touched when K > 0, where it is bound.)
        if (return_arrays or not K
                or bool((rscore_h.max(axis=1) <= 0.0).any())):
            t0_h = t0_g[:nc * nperiods].get().reshape(nc, nperiods)
            dur_h = dur_g[:nc * nperiods].get().reshape(nc, nperiods)
            depth_h = depth_g[:nc * nperiods].get().reshape(nc, nperiods)

        # ---- Per-LC statistics (pure CPU, one light curve at a
        # time). This used to run on a ThreadPoolExecutor on the
        # assumption that scipy released the GIL in the running-median
        # detrend; it does not, and the pool made the work slower and
        # the warning order nondeterministic. Measured on an A40
        # (shared), the pooled and the sequential module interleaved
        # in one process: 64 tess-ffi light curves 166.2 -> 77.0 ms
        # (2.16x), 16 tess-yr light curves 322.2 -> 269.4 ms (1.20x);
        # the statistics alone are 21.7 ms sequential vs 40.3 ms on 8
        # threads. ----
        def _finish_lc(j):
            lc_idx = i0 + j
            srow = score_h[j]
            valid = srow > 0.0
            n_failed = int(nperiods - valid.sum())
            if n_failed == nperiods:
                msg = _NO_SOLUTION_MSG % nperiods
                warnings.warn("lightcurve %d: %s; returning a null "
                              "result (SDE = 0)" % (lc_idx, msg))
                return lc_idx, _null_result(
                    nperiods, chi2_0[lc_idx], msg, periods=periods_in,
                    arrays=return_arrays)
            if n_failed and _warn_failed:
                warnings.warn(
                    "%d of %d trial periods returned no valid TLS "
                    "solution (chi2 sentinel); they are excluded from "
                    "the best-fit search and the SDE statistics and "
                    "appear as NaN in the returned arrays"
                    % (n_failed, nperiods))

            # chi2 reconstructed in float64 against the float64 chi2_0
            row = chi2_0[lc_idx] - srow.astype(np.float64)
            chi2_valid = row[valid]
            periods_valid = periods[valid]

            # Best-fit parameters come from the exact refinement pass
            # when available; the coarse spectrum (row) is what feeds
            # the SDE statistics either way.
            slot = int(np.argmax(rscore_h[j])) if K else 0
            if K and rscore_h[j, slot] > 0.0:
                best_idx = int(cand[j, slot])
                best_t0 = float(rt0_h[j, slot])
                best_duration = float(rdur_h[j, slot])
                best_depth = float(rdepth_h[j, slot])
                chi2_min = float(chi2_0[lc_idx] - rscore_h[j, slot])
                best_valid_idx = int(np.searchsorted(
                    np.flatnonzero(valid), best_idx))
            else:
                best_valid_idx = int(np.argmin(chi2_valid))
                best_idx = int(np.flatnonzero(valid)[best_valid_idx])
                chi2_min = float(row[best_idx])
                best_t0 = float(t0_h[j, best_idx])
                best_duration = float(dur_h[j, best_idx])
                best_depth = float(depth_h[j, best_idx])

            best_period = float(periods[best_idx])
            n_transits = int(spans[lc_idx] / best_period)

            stats = tls_stats.compute_all_statistics(
                chi2_valid, periods_valid, best_valid_idx,
                best_depth, best_duration, n_transits,
                kernel_size=sde_kernel_size,
                chi2_null=float(chi2_0[lc_idx]), chi2_best=chi2_min)
            period_uncertainty = tls_stats.compute_period_uncertainty(
                periods_valid, chi2_valid, best_valid_idx)

            # Absolute mid-transit time: the kernel phase is relative
            # to the epoch floor(min t); report the first transit at
            # or after the first observation
            T0 = _first_transit_at_or_after(
                epochs[lc_idx] + best_t0 * best_period, best_period,
                tmins[lc_idx])

            res = {
                'period': best_period,
                'period_uncertainty': period_uncertainty,
                't0_phase': best_t0,
                'T0': float(T0),
                'duration': best_duration,
                'depth': best_depth,
                'chi2_min': chi2_min,
                'SDE': stats['SDE'],
                'SDE_raw': stats['SDE_raw'],
                'SNR': stats['SNR'],
                'n_transits': n_transits,
                'n_failed_periods': n_failed,
            }
            if return_arrays:
                def _expand(values):
                    full = np.full(nperiods, np.nan)
                    full[valid] = values
                    return _to_caller_order(full, order)
                res.update({
                    'periods': periods_in,
                    'chi2': _to_caller_order(
                        np.where(valid, row, np.nan), order),
                    'best_t0_per_period': _to_caller_order(
                        t0_h[j].copy(), order),
                    'best_duration_per_period': _to_caller_order(
                        dur_h[j].copy(), order),
                    'best_depth_per_period': _to_caller_order(
                        depth_h[j].copy(), order),
                    'valid_periods': _to_caller_order(valid, order),
                    'power': _expand(stats['power']),
                    'SR': _expand(stats['SR']),
                })
            return lc_idx, res

        for j in range(nc):
            lc_idx, res = _finish_lc(j)
            results[lc_idx] = res

    if fap_null_draws:
        try:
            n_null_draws = operator.index(fap_null_draws)
        except TypeError:
            raise ValueError(
                "fap_null_draws must be an integer >= 1 (got %r)"
                % (fap_null_draws,))
        _attach_null_fap(
            results, lightcurves, n_null_draws, fap_seed,
            dict(periods=periods, qmin=qmin, qmax=qmax,
                 n_durations=n_durations, t0_oversample=t0_oversample,
                 refine_top_k=0, block_size=block_size, nbins=nbins,
                 limb_dark=limb_dark, u=u, R_star=R_star, M_star=M_star,
                 sde_kernel_size=sde_kernel_size))

    return results


def _attach_null_fap(results, lightcurves, n_draws, seed, search_kwargs):
    """Empirical FAP by flux permutation (see tls_search_batch,
    ``fap_null_draws``): each lightcurve's (y, dy) pairs are permuted
    over its times ``n_draws`` times, searched with the identical trial
    grid and settings, and the exceedance of the observed SDE is
    recorded under 'FAP' (add-one estimator) with the null SDEs under
    'SDE_null'."""
    if n_draws < 1:
        raise ValueError(
            "fap_null_draws must be an integer >= 1 (got %r)" % (n_draws,))
    rng = np.random.RandomState(seed)
    n_lc = len(lightcurves)
    lens = [len(lc[0]) for lc in lightcurves]
    i0 = 0
    while i0 < n_lc:
        # group lightcurves so one null batch stays within the
        # per-launch point budget of the search
        i1 = i0 + 1
        pts = lens[i0] * n_draws
        while (i1 < n_lc
               and pts + lens[i1] * n_draws <= _TLS_FAST_MAX_POINTS):
            pts += lens[i1] * n_draws
            i1 += 1
        null_lcs = []
        for i in range(i0, i1):
            t, y, dy = lightcurves[i]
            y = np.asarray(y)
            dy = np.asarray(dy)
            for _ in range(n_draws):
                perm = rng.permutation(len(y))
                null_lcs.append((t, y[perm], dy[perm]))
        null_res = tls_search_batch(null_lcs, **search_kwargs)
        for k, i in enumerate(range(i0, i1)):
            sde_null = np.array(
                [r['SDE'] for r in null_res[k * n_draws:(k + 1) * n_draws]],
                dtype=np.float64)
            res = results[i]
            n_exceed = int(np.sum(sde_null >= res['SDE']))
            res['FAP'] = (n_exceed + 1.0) / (n_draws + 1.0)
            res['SDE_null'] = sde_null
        i0 = i1
