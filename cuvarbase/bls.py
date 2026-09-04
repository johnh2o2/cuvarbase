"""
Implementation of the box-least squares periodogram [K2002]_
and variants.

The Keplerian transit-search helpers (:func:`q_transit`,
:func:`freq_transit`, :func:`transit_autofreq`, :func:`eebls_transit`)
assume the transiting body orbits at the host star's mean density. That
assumption fixes the transit-duration/period relation [SM03]_ and, with
it, the optimal frequency-grid spacing for a transit search [O2014]_.

.. [K2002] `Kovacs et al. 2002, A&A 391, 369 <https://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_

"""
import functools
import threading
import warnings
from collections import OrderedDict

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import ensure_context
from .utils import (find_kernel, _module_reader, subtract_epoch,
                    conflict_scatter_perm, check_lightcurve, check_freqs)
from .memory.bls_memory import BLSBatchMemory
from .memory._host import host_array

import numpy as np

_default_block_size = 256

# Minimum number of observations any BLS path accepts. Every BLS
# statistic is normalized by the weighted variance of y, which is
# identically zero for a single point (the periodogram came back as
# 0/0 = NaN); two points is the smallest input for which the null
# model is defined. The Keplerian entry points need more than this --
# ``fmin_transit`` needs ``min_obs_per_transit`` (default 5) or the
# duty cycle q = min_obs_per_transit / N exceeds 1 and the grid comes
# back all-NaN -- and raise from there.
_BLS_MIN_NDATA = 2
_all_function_names = ['full_bls_no_sol',
                       'full_bls_no_sol_optimized',
                       'full_bls_no_sol_fused',
                       'bin_and_phase_fold_custom',
                       'reduction_max',
                       'store_best_sols',
                       'store_best_sols_custom',
                       'bin_and_phase_fold_bst_multifreq',
                       'binned_bls_bst']

# Kernel cache: (block_size, use_optimized, function_names) -> compiled functions
# LRU cache with max 20 entries to prevent unbounded memory growth
# Each entry is ~1-5 MB (compiled CUDA kernels)
# Expected max memory: ~100 MB for full cache
_KERNEL_CACHE_MAX_SIZE = 20
_kernel_cache = OrderedDict()
_kernel_cache_lock = threading.Lock()


def _choose_block_size(ndata):
    """
    Choose a CUDA block size based on data size.

    Parameters
    ----------
    ndata : int
        Number of data points

    Returns
    -------
    block_size : int
        CUDA block size (32, 64, 128, or 256)

    Notes
    -----
    The heuristic considers only ``ndata``; occupancy effects driven
    by the number of phase bins (i.e. small ``qmin``) are ignored, so
    the choice may be suboptimal for unusual ``ndata``/``nbins``
    combinations. The v1.0 re-benchmark (warm kernel cache) measures ~1.0-1.3x over
    fixed blocks on Keplerian-style grids; benchmark ``block_size``
    yourself if it matters for your workload.
    """
    if ndata <= 32:
        return 32   # Single warp
    elif ndata <= 64:
        return 64   # Two warps
    elif ndata <= 128:
        return 128  # Four warps
    else:
        return 256  # Default (8 warps)


# Frequency-chunk size for occupancy-aware launches (see
# _shmem_limits_occupancy): 8192 measured best on an RTX A5000 Kepler
# grid (131K freqs; 8192 -> 114.8 ms vs 151.2 ms unchunked, 16384
# within 2%), and small enough that launch overhead stays negligible
# for any grid where chunking triggers at all.
_OCCUPANCY_FREQ_CHUNK = 8192


def _shmem_limits_occupancy(mem_req, block_size):
    """True when a launch needing ``mem_req`` bytes of shared memory
    per block caps resident blocks/SM below the thread-count limit --
    i.e. shared memory, not threads, is the occupancy limiter and
    frequency-chunked launches (which size shared memory per chunk)
    can win occupancy back."""
    dev = ensure_context().device
    try:
        smem_sm = dev.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        thr_sm = dev.get_attribute(
            cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
    except Exception:
        return False
    blocks_by_threads = max(1, thr_sm // block_size)
    blocks_by_shmem = max(1, smem_sm // max(1, int(mem_req)))
    return blocks_by_shmem < blocks_by_threads


def _get_cached_kernels(block_size, use_optimized=False, function_names=None):
    """
    Get compiled kernels from cache, or compile and cache if not present.

    Thread-safe LRU cache implementation. When cache exceeds max size,
    least recently used entries are evicted.

    Parameters
    ----------
    block_size : int
        CUDA block size
    use_optimized : bool
        Use optimized kernel
    function_names : list, optional
        Function names to compile

    Returns
    -------
    functions : dict
        Compiled kernel functions

    Notes
    -----
    Cache size is limited to _KERNEL_CACHE_MAX_SIZE entries (~100 MB max).
    Each compiled kernel is approximately 1-5 MB in memory.
    Thread-safe for concurrent access from multiple threads.
    """
    if function_names is None:
        function_names = _all_function_names

    # Ensure a CUDA context exists before returning kernels, even on a
    # cache hit (compile_bls only runs on a miss): callers go straight on
    # to launches/memory allocation, so kernel acquisition must
    # self-guarantee the context rather than rely on a warm-cache having
    # been compiled in this process. Idempotent/cached after first call.
    ensure_context()

    # Create cache key from block size, optimization flag, and function names
    key = (block_size, use_optimized, tuple(sorted(function_names)))

    with _kernel_cache_lock:
        # Check if key exists and move to end (most recently used)
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]

        # Compile kernel (done inside lock to prevent duplicate compilation)
        compiled_functions = compile_bls(block_size=block_size,
                                         use_optimized=use_optimized,
                                         function_names=function_names)

        # Add to cache
        _kernel_cache[key] = compiled_functions
        _kernel_cache.move_to_end(key)

        # Evict oldest entry if cache is full
        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)  # Remove oldest (FIFO = LRU)

        return compiled_functions


_function_signatures = {
    'full_bls_no_sol': [np.intp, np.intp, np.intp,
                        np.intp, np.intp, np.intp,
                        np.intp, np.uint32, np.uint32,
                        np.uint32, np.uint32, np.uint32,
                        np.float32, np.float32, np.uint32],
    'full_bls_no_sol_optimized': [np.intp, np.intp, np.intp,
                        np.intp, np.intp, np.intp,
                        np.intp, np.uint32, np.uint32,
                        np.uint32, np.uint32, np.uint32,
                        np.float32, np.float32, np.uint32],
    # fused-noverlap variant (bls_common.cuh, present in both modules);
    # identical argument list, hist_size = noverlap * max_nbins
    'full_bls_no_sol_fused': [np.intp, np.intp, np.intp,
                        np.intp, np.intp, np.intp,
                        np.intp, np.uint32, np.uint32,
                        np.uint32, np.uint32, np.uint32,
                        np.float32, np.float32, np.uint32],
    'bin_and_phase_fold_custom': [np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.float64,
                                  np.uint32, np.uint32, np.uint32,
                                  np.uint32, np.uint32],
    'reduction_max': [np.intp, np.intp, np.uint32, np.uint32, np.uint32,
                      np.intp, np.intp, np.uint32, np.uint32],
    # (argmaxes, best_phi, best_q, nbins0_arr, nbinsf_arr, noverlap,
    #  dlogq, nfreq, freq_offset): per-frequency bin-count arrays
    'store_best_sols': [np.intp, np.intp, np.intp, np.intp, np.intp,
                        np.uint32, np.float32, np.uint32, np.uint32],
    'store_best_sols_custom': [np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.uint32,
                               np.uint32, np.uint32, np.uint32],
    # (t, yw, w, yw_bin, w_bin, freqs, nbins0_arr, nbinsf_arr, ndata,
    #  nfreq, freq_offset, noverlap, dlogq, nbins_tot)
    'bin_and_phase_fold_bst_multifreq':
        [np.intp, np.intp, np.intp, np.intp,
         np.intp, np.intp, np.intp, np.intp,
         np.uint32, np.uint32, np.uint32, np.uint32,
         np.float32, np.uint32],
    'binned_bls_bst': [np.intp, np.intp, np.intp, np.uint32, np.uint32]
}


def _reduction_max(max_func, arr, arr_args, nfreq, nbins,
                   stream, final_arr, final_argmax_arr,
                   final_index, block_size):
    # The reduction kernels require the compiled power-of-two block
    # size; a mismatched block_size silently corrupts the tree
    # reduction. (The old `assert(block_size - 2*(block_size/2) == 0)`
    # was always true under Python 3 division.)
    _validate_block_size(block_size)

    block = (block_size, 1, 1)
    grid_size = int(np.ceil(float(nbins) / block_size)) * nfreq
    grid = (grid_size, 1)
    nbins0 = nbins

    init = np.uint32(1)
    while (grid_size > nfreq):

        max_func.prepared_async_call(grid, block, stream,
                                     arr.ptr, arr_args.ptr,
                                     np.uint32(nfreq), np.uint32(nbins0),
                                     np.uint32(nbins),
                                     arr.ptr, arr_args.ptr, np.uint32(0), init)
        init = np.uint32(0)

        nbins0 = grid_size // nfreq
        grid_size = int(np.ceil(float(nbins0) / block_size)) * nfreq
        grid = (grid_size, 1)

    max_func.prepared_async_call(grid, block, stream,
                                 arr.ptr,  arr_args.ptr,
                                 np.uint32(nfreq), np.uint32(nbins0),
                                 np.uint32(nbins),
                                 final_arr.ptr, final_argmax_arr.ptr,
                                 np.uint32(final_index), init)


def fmin_transit(t, rho=1., min_obs_per_transit=5, **kwargs):
    """Minimum search frequency for a Keplerian transit grid.

    The larger of (a) the frequency whose Keplerian duration holds at
    least ``min_obs_per_transit`` samples and (b) ``2 / T`` (two cycles
    over the baseline ``T``), the latter being the long-period limit of
    Ofir (2014), Sect. 3.1 [O2014]_.
    """
    t = np.asarray(t)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("fmin_transit: t must be a non-empty array of "
                         "finite observation times")
    if t.size < int(min_obs_per_transit):
        # q = min_obs_per_transit / N > 1 below this, and
        # freq_transit(q) = fmax0 * sin(pi q)**1.5 is NaN for q > 1:
        # transit_autofreq used to return freqs = [nan], q = [nan],
        # which reached the kernels as a NaN uint32 bin count and
        # crashed the device (Sep 2026 audit, defect 23).
        raise ValueError(
            "fmin_transit: %d observations cannot hold %d samples in a "
            "single transit (the Keplerian duty cycle would exceed 1); "
            "pass an explicit fmin/freqs, or lower "
            "min_obs_per_transit" % (t.size, int(min_obs_per_transit)))
    qmin = float(min_obs_per_transit) / len(t)

    fmin1 = freq_transit(qmin, rho=rho)
    fmin2 = 2./(np.max(t) - np.min(t))
    return max([fmin1, fmin2])


def fmax_transit0(rho=1., **kwargs):
    """Orbital frequency of a body grazing the stellar surface.

    This is the natural high-frequency cutoff for a transit search: the
    Keplerian frequency of a circular orbit at the stellar radius,
    :math:`f_{\\max,0} = \\sqrt{G \\rho_\\star / 3\\pi}` (period =
    free-fall/orbit time at the surface). For ``rho = 1`` (solar mean
    density) this evaluates to ``8.6307`` cycles/day -- a *derived*
    constant, not a literature value. (Ofir 2014 [O2014]_ instead caps
    at the Roche-limit frequency ``fmax0 / 3**1.5``.)
    """
    return 8.6307 * np.sqrt(rho)


def q_transit(freq, rho=1., **kwargs):
    """Keplerian transit-duration fraction ``q`` at a given frequency.

    Assuming a central transit (inclination 90 deg, impact parameter 0)
    of a body orbiting at the host's mean density, the fractional transit
    duration is :math:`q = \\arcsin[(f / f_{\\max,0})^{2/3}] / \\pi`.
    This is Seager & Mallen-Ornelas (2003) eq. (3) reduced to ``b = 0``
    [SM03]_, with ``fmax0`` from :func:`fmax_transit0`.
    """
    fmax0 = fmax_transit0(rho=rho)

    f23 = np.power(freq / fmax0, 2./3.)
    f23 = np.minimum(1., f23)
    return np.arcsin(f23) / np.pi


def freq_transit(q, rho=1., **kwargs):
    """Frequency at which the Keplerian transit fraction equals ``q``.

    Inverse of :func:`q_transit`:
    :math:`f = f_{\\max,0}\\,\\sin(\\pi q)^{3/2}` [SM03]_.
    """
    fmax0 = fmax_transit0(rho=rho)
    return fmax0 * (np.sin(np.pi * q) ** 1.5)


def fmax_transit(rho=1., qmax=0.5, **kwargs):
    """Maximum search frequency, capped by the surface-orbit cutoff.

    The smaller of :func:`fmax_transit0` and the frequency whose
    Keplerian duration reaches ``qmax`` [SM03]_.
    """
    fmax0 = fmax_transit0(rho=rho)
    return min([fmax0, freq_transit(qmax, rho=rho, **kwargs)])


def transit_autofreq(t, fmin=None, fmax=None, samples_per_peak=2,
                     rho=1., qmin_fac=0.2, qmax_fac=None, **kwargs):
    """
    Produce list of frequencies for a given frequency range
    suitable for performing Keplerian BLS.

    Parameters
    ----------
    t: array_like, float
        Observation times.
    fmin: float, optional (default: ``None``)
        Minimum frequency. By default this is determined by ``fmin_transit``.
    fmax: float, optional (default: ``None``)
        Maximum frequency. By default this is determined by ``fmax_transit``.
    samples_per_peak: float, optional (default: 2)
        Oversampling factor. Frequency spacing is multiplied by
        ``1/samples_per_peak``.
    rho: float, optional (default: 1)
        Mean stellar density of host star in solar units
        :math:`\\rho=\\rho_{\\star} / \\rho_{\\odot}`, where
        :math:`\\rho_{\\odot}`
        is the mean density of the sun
    qmin_fac: float, optional (default: 0.2)
        The minimum :math:`q` value to search in units of the Keplerian
        :math:`q` value
    qmax_fac: float, optional (default: None)
        The maximum :math:`q` value to search in units of the Keplerian
        :math:`q` value. If ``None``, this defaults to ``1/qmin_fac``.
    **kwargs:
        passed to `fmin_transit`

    Returns
    -------
    freqs: array_like
        The frequency grid
    q0vals: array_like
        The list of Keplerian :math:`q` values.

    Notes
    -----
    The grid is spaced by :math:`\\Delta f = q(f) / (\\mathrm{OS}\\,T)`,
    Ofir (2014) eq. (4) [O2014]_ (with ``OS = samples_per_peak``): the
    local frequency resolution is set by the transit duty cycle ``q(f)``,
    so the grid is denser at high frequencies. This is far coarser than a
    uniform grid while still Nyquist-sampling every trial transit.

    """
    if qmax_fac is None:
        qmax_fac = 1./qmin_fac

    t = np.asarray(t)
    if t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("transit_autofreq: t must be a non-empty array "
                         "of finite observation times")

    if fmin is None:
        fmin = fmin_transit(t, rho=rho, **kwargs)
    if fmax is None:
        fmax = fmax_transit(rho=rho, qmax=0.5 / qmax_fac, **kwargs)

    T = np.max(t) - np.min(t)
    freqs = [fmin]
    while freqs[-1] < fmax:
        df = qmin_fac * q_transit(freqs[-1], rho=rho) / (samples_per_peak * T)
        freqs.append(freqs[-1] + df)
    freqs = np.array(freqs)
    q0vals = q_transit(freqs, rho=rho)
    return freqs, q0vals


def _validate_block_size(block_size):
    """Validate CUDA block size for the BLS kernels.

    The tree reductions and warp-shuffle stages assume a power-of-two
    block of at least one full warp; anything else silently produces
    wrong results or undefined behavior, so fail loudly here instead.
    """
    if not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be an integer, got %r"
                         % (block_size,))
    if block_size < 32 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a power of 2 and >= 32 "
                         "(one warp); got %d" % block_size)


def compile_bls(block_size=_default_block_size,
                function_names=_all_function_names,
                prepare=True,
                use_optimized=False,
                **kwargs):
    """
    Compile BLS kernel

    Parameters
    ----------
    block_size: int, optional (default: _default_block_size)
        CUDA threads per CUDA block.
    function_names: list, optional (default: _all_function_names)
        Function names to load and prepare
    prepare: bool, optional (default: True)
        Whether or not to prepare functions (for slightly faster
        kernel launching)
    use_optimized: bool, optional (default: False)
        Use optimized kernel with bank conflict fixes and warp shuffles

    Returns
    -------
    functions: dict
        Dictionary of (function name, PyCUDA function object) pairs

    """
    _validate_block_size(block_size)

    # Compiling a kernel needs an active CUDA context (lazily created).
    ensure_context()

    # Read kernel
    cppd = dict(BLOCK_SIZE=block_size)
    kernel_name = 'bls_optimized' if use_optimized else 'bls'
    kernel_txt = _module_reader(find_kernel(kernel_name),
                                cpp_defs=cppd)

    # Filter function names based on kernel variant:
    # bls_optimized.cu has full_bls_no_sol_optimized but not full_bls_no_sol
    # bls.cu has full_bls_no_sol but not full_bls_no_sol_optimized
    requested = list(function_names)
    if use_optimized:
        function_names = [n for n in function_names
                          if n != 'full_bls_no_sol']
    else:
        function_names = [n for n in function_names
                          if n != 'full_bls_no_sol_optimized']

    if len(function_names) == 0:
        raise ValueError(
            "compile_bls: no loadable functions remain from %r with "
            "use_optimized=%r (the %s kernel provides %r)"
            % (requested, use_optimized,
               'optimized' if use_optimized else 'standard',
               'full_bls_no_sol_optimized' if use_optimized
               else 'full_bls_no_sol'))

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    functions = {name: module.get_function(name) for name in function_names}

    # prepare functions
    if prepare:
        for name in functions.keys():
            sig = _function_signatures[name]
            functions[name] = functions[name].prepare(sig)

    return functions


class BLSMemory:
    def __init__(self, max_ndata, max_nfreqs, stream=None, **kwargs):
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import).
        ensure_context()
        self.max_ndata = max_ndata
        self.max_nfreqs = max_nfreqs
        self.t = None
        self.yw = None
        self.w = None

        self.t_g = None
        self.yw_g = None
        self.w_g = None

        self.freqs = None
        self.freqs_g = None

        self.qmin = None
        self.nbins0_g = None
        self.qmax = None
        self.nbinsf_g = None

        self.chi2_0 = None

        self.bls = None
        self.bls_g = None

        self.rtype = np.float32

        # floor(min(t)) subtracted from the times before the float32 cast
        # (phases are measured relative to it)
        self.epoch = None

        self.stream = stream

        # Pinned (page-locked) host buffers by default for true async
        # transfer overlap; falls back to page-aligned if pinning fails.
        self.pinned = kwargs.get('pinned', True)

        self.allocate_host_arrays(nfreqs=max_nfreqs, ndata=max_ndata)

    def allocate_pinned_arrays(self, nfreqs=None, ndata=None):
        """Deprecated alias for :meth:`allocate_host_arrays`."""
        warnings.warn("allocate_pinned_arrays is deprecated; use "
                      "allocate_host_arrays", DeprecationWarning)
        return self.allocate_host_arrays(nfreqs=nfreqs, ndata=ndata)

    def allocate_host_arrays(self, nfreqs=None, ndata=None):
        """Allocate host arrays for transfers.

        By default (``pinned=True``) these are page-locked so
        ``set_async``/``get_async`` transfers overlap with computation;
        if pinning fails they fall back to page-aligned memory (see
        :func:`cuvarbase.memory._host.host_array`).
        """
        if nfreqs is None:
            nfreqs = int(self.max_nfreqs)
        if ndata is None:
            ndata = int(self.max_ndata)

        self.bls = host_array((nfreqs,), self.rtype, pinned=self.pinned)
        self.nbins0 = host_array((nfreqs,), np.int32, pinned=self.pinned)
        self.nbinsf = host_array((nfreqs,), np.int32, pinned=self.pinned)
        self.t = host_array((ndata,), self.rtype, pinned=self.pinned)
        self.yw = host_array((ndata,), self.rtype, pinned=self.pinned)
        self.w = host_array((ndata,), self.rtype, pinned=self.pinned)

    def allocate_freqs(self, nfreqs=None):
        if nfreqs is None:
            nfreqs = self.max_nfreqs

        self.freqs_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
        self.bls_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
        self.nbins0_g = gpuarray.zeros(nfreqs, dtype=np.uint32)
        self.nbinsf_g = gpuarray.zeros(nfreqs, dtype=np.uint32)

    def allocate_data(self, ndata=None):
        if ndata is None:
            ndata = len(self.t)
        self.t_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.yw_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.w_g = gpuarray.zeros(ndata, dtype=self.rtype)

    def transfer_data_to_gpu(self, transfer_freqs=True):
        self.t_g.set_async(self.t, stream=self.stream)
        self.yw_g.set_async(self.yw, stream=self.stream)
        self.w_g.set_async(self.w, stream=self.stream)

        if transfer_freqs:
            self.freqs_g.set_async(self.freqs, stream=self.stream)
            self.nbins0_g.set_async(self.nbins0, stream=self.stream)
            self.nbinsf_g.set_async(self.nbinsf, stream=self.stream)

    def transfer_data_to_cpu(self):
        # self.bls_g.get_async(ary=self.bls, stream=self.stream)
        if self.stream is None:
            self.bls = self.bls_g.get() / self.yy

        else:
            self.bls_g.get_async(ary=self.bls, stream=self.stream)
            # self.bls is page-locked, so the copy above is genuinely
            # asynchronous: sync before the host-side normalization or
            # the divide races the DMA and gets overwritten by it.
            self.stream.synchronize()
            self.bls /= self.yy

        # return self.bls

    def setdata(self, t, y, dy, qmin=None, qmax=None,
                freqs=None, nf=None, transfer=True,
                **kwargs):

        # The weights below are dy**-2 and the periodogram is divided
        # by the weighted variance of y: a non-finite sample or
        # dy = 0 used to travel to the device unnoticed.
        check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA,
                         name='BLSMemory.setdata')

        if freqs is not None:
            self.freqs = np.asarray(freqs).astype(self.rtype)
            self.nbins0, self.nbinsf = _fast_path_nbins(self.freqs,
                                                        qmin, qmax)

        # Epoch-subtract in float64 before the float32 cast: absolute
        # timestamps (e.g. BJD) would otherwise destroy the phase fold.
        t, self.epoch = subtract_epoch(t)

        w = np.power(dy, -2)
        w /= np.sum(w)

        self.ybar = np.sum(y * w)
        # einsum, not np.dot: BLAS ddot spawns a full threadpool for
        # large vectors, and on CPU-quota-limited containers (RunPod,
        # K8s) the burst trips CFS throttling and freezes the process
        # ~90 ms per 100 ms period (measured 8x end-to-end slowdown at
        # TESS scale). einsum stays in numpy core, single-threaded.
        self.yy = float(np.einsum('i,i->', w,
                                  np.power(y - self.ybar, 2)))
        # chi2 of the constant model for the data actually loaded here;
        # convert_bls_power scalings must use this rather than whatever
        # y/dy a later (memory-reuse) call happens to pass.
        self.chi2_0 = _chi2_null(y, dy)

        u = (y - self.ybar) * w

        # Store in conflict-scattered order: time-sorted input puts
        # warp-adjacent samples into the same phase bin at nearly every
        # trial frequency, serializing the kernels' shared-memory
        # atomics (3.1x on a TESS-like cadence). Binning is a sum, so
        # the order is semantically free. See
        # utils.conflict_scatter_perm.
        perm = conflict_scatter_perm(len(t))
        if perm is None:
            self.t[:len(t)] = t.astype(self.rtype)[:]
            self.w[:len(t)] = np.asarray(w).astype(self.rtype)[:]
            self.yw[:len(t)] = np.asarray(u).astype(self.rtype)[:]
        else:
            self.t[:len(t)] = t.astype(self.rtype)[perm]
            self.w[:len(t)] = np.asarray(w).astype(self.rtype)[perm]
            self.yw[:len(t)] = np.asarray(u).astype(self.rtype)[perm]

        if any([x is None for x in [self.t_g, self.yw_g, self.w_g]]):
            self.allocate_data()

        if self.freqs_g is None:
            if nf is None:
                nf = len(freqs)
            self.allocate_freqs(nfreqs=nf)
        elif freqs is not None and len(self.freqs) != len(self.freqs_g):
            # the device grid arrays keep their first size; a silent
            # pycuda "ary and self must be the same size" used to
            # surface from set_async
            raise ValueError(
                "BLSMemory: this memory's device frequency arrays hold "
                "%d frequencies (sized by the first setdata call) but "
                "%d were given; reuse a BLSMemory with the same "
                "len(freqs) or construct a new one"
                % (len(self.freqs_g), len(self.freqs)))

        if transfer:
            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))

        return self

    @classmethod
    def fromdata(cls, t, y, dy, qmin=None, qmax=None,
                 freqs=None, nf=None, transfer=True,
                 **kwargs):
        """Construct a :class:`BLSMemory` sized for ``t``/``freqs`` and
        load the data. ``max_ndata`` / ``max_nfreqs`` may be given as
        keywords to over-allocate the host arrays (they used to be
        passed on to ``__init__`` a second time and raise ``TypeError``;
        Sep 2026 audit, id 67). Note the device frequency arrays are
        sized by the first ``setdata`` call: reuse requires the same
        ``len(freqs)``."""
        # pop, not get: __init__ takes them positionally
        max_ndata = kwargs.pop('max_ndata', len(t))
        max_nfreqs = kwargs.pop('max_nfreqs', nf if freqs is None
                                else len(freqs))
        c = cls(max_ndata, max_nfreqs, **kwargs)

        return c.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                         freqs=freqs, nf=nf, transfer=transfer,
                         **kwargs)


def _fast_path_nbins(freqs32, qmin, qmax):
    """Per-frequency bin counts of the fast (shared-memory) kernels:
    ``nbinsf = floor(1/qmin)`` fine bins and ``nbins0 = floor(1/qmax)``
    (box widths ``m / nbinsf`` for ``m`` up to and including
    ``floor(nbinsf / nbins0)`` -- see :func:`_fast_box_widths`),
    exactly as :meth:`BLSMemory.setdata` uploads them.
    ``freqs32`` is the float32 frequency array (only its length and
    dtype matter); ``qmin``/``qmax`` scalar or per-frequency.

    The bounds are validated here because this is where they become
    ``uint32``: ``(1 / np.array([nan, 0.01, 5, inf])).astype(uint32)``
    is ``[0, 100, 0, 0]``, and a zero bin count makes the kernels
    divide by zero and ``atomicAdd`` outside the histogram -- an
    illegal memory access that kills the process's CUDA context (Sep
    2026 audit, defect 23).

    The division is deliberately left in the input dtype: float32 and
    float64 truncate to different bin counts for some bounds (e.g.
    ``qmin = 1/7`` gives 6 in float32 and 7 in float64), so promoting
    it here would change every existing periodogram.
    """
    _validate_fast_q_bounds(len(freqs32), qmin, qmax)
    nbinsf = (np.ones_like(freqs32) / qmin).astype(np.uint32)
    nbins0 = (np.ones_like(freqs32) / qmax).astype(np.uint32)
    return nbins0, nbinsf


def _validate_fast_q_bounds(nfreqs, qmin, qmax):
    """Validate transit-duration bounds for the binned (fast) kernels.

    ``_validate_q_bounds`` (finite, qmin >= 0, qmax > 0, qmin <= qmax)
    plus the two conditions the *binned* kernels add: ``qmin > 0`` and
    ``qmax <= 1`` (see :func:`_check_q_bounds_for_bins`). ``None``
    bounds fall through to the caller's default so this never changes
    which exception an unsupported call raises.
    """
    if qmin is None or qmax is None:
        return
    qmins = _broadcast_q_bound(qmin, nfreqs, 1e-2, 'qmin')
    qmaxes = _broadcast_q_bound(qmax, nfreqs, 0.5, 'qmax')
    _validate_q_bounds(qmins, qmaxes)
    _check_q_bounds_for_bins(qmins, qmaxes)


def _validate_noverlap(noverlap):
    """noverlap must be a positive integer (number of phase-shifted
    passes on the fast BLS paths)."""
    if not isinstance(noverlap, (int, np.integer)) or noverlap < 1:
        raise ValueError("noverlap must be a positive integer, got %r"
                         % (noverlap,))


def _eebls_gpu_fast_impl(t, y, dy, freqs, fname, use_optimized,
                         qmin=1e-2, qmax=0.5,
                         ignore_negative_delta_sols=False,
                         functions=None, stream=None, dlogq=0.3,
                         memory=None, noverlap=2, max_nblocks=5000,
                         force_nblocks=None, dphi=0.0,
                         shmem_lim=None, freq_batch_size=None,
                         transfer_to_device=True,
                         transfer_to_host=True,
                         convention='chi2ratio', **kwargs):
    """Shared implementation behind :func:`eebls_gpu_fast` and
    :func:`eebls_gpu_fast_optimized`; see their docstrings for the
    parameter descriptions."""
    # Validate before ANY device work (kernel compile included): a NaN
    # in t used to give a finite periodogram with a wrong argmax, and a
    # NaN or out-of-range q bound crashed the kernel and killed the
    # process's CUDA context (Sep 2026 audit, defect 23).
    _name = ('eebls_gpu_fast_optimized' if use_optimized
             else 'eebls_gpu_fast')
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name=_name)
    check_freqs(freqs, name=_name)
    _validate_fast_q_bounds(len(freqs), qmin, qmax)
    _validate_noverlap(noverlap)
    _validate_convention(convention)
    if convention != 'chi2ratio' and not transfer_to_host:
        raise ValueError("convention=%r requires transfer_to_host=True "
                         "(the device-side periodogram is always "
                         "'chi2ratio')" % (convention,))

    if functions is None:
        # Use the thread-safe LRU kernel cache (compilation costs ~150 ms
        # per call otherwise). Fall back to a direct compile only for
        # non-default compile options that aren't part of the cache key.
        # The fused-noverlap kernel ships in the same module, so request
        # it alongside (same single compilation).
        if kwargs.get('prepare', True):
            functions = _get_cached_kernels(
                kwargs.get('block_size', _default_block_size),
                use_optimized, [fname, 'full_bls_no_sol_fused'])
        else:
            ckw = dict(kwargs)
            ckw.setdefault('use_optimized', use_optimized)
            functions = compile_bls(
                function_names=[fname, 'full_bls_no_sol_fused'], **ckw)

    func = functions[fname]

    # Fused-noverlap fast path: for power-of-two noverlap with no base
    # phase offset, one launch histograms at noverlap-times finer phase
    # resolution and derives every pass's box sums from it -- the
    # noverlap-x fold + histogram (and per-frequency fixed costs) are
    # paid once. Bin assignment is bit-identical to the multi-pass loop
    # there (see full_bls_no_sol_fused in bls_common.cuh); any other
    # (noverlap, dphi) combination keeps the host-side loop, as do
    # caller-provided ``functions`` dicts without the fused kernel.
    fused_func = None
    try:
        fused_func = functions.get('full_bls_no_sol_fused')
    except AttributeError:
        fused_func = None
    noverlap_int = int(noverlap)
    use_fused = (fused_func is not None
                 and noverlap_int >= 2
                 and float(dphi) == 0.0
                 and (noverlap_int & (noverlap_int - 1)) == 0)

    if shmem_lim is None:
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = ensure_context().device.get_attribute(att)

    if memory is None:
        memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax,
                                    freqs=freqs, stream=stream,
                                    transfer=True,
                                    **kwargs)
    elif transfer_to_device:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                       freqs=freqs, transfer=True,
                       **kwargs)

    float_size = np.float32(1).nbytes
    block_size = kwargs.get('block_size', _default_block_size)

    auto_freq_batch = freq_batch_size is None
    if freq_batch_size is None:
        freq_batch_size = len(freqs)

    block = (block_size, 1, 1)

    # minimum q value that we can handle with the shared memory limit
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)

    # The fused kernel needs (block_size + 2*noverlap*max_nbins) floats
    # of shared memory; fall back to the multi-pass loop when that
    # exceeds the device limit (the loop only needs the 1x histogram).
    if use_fused:
        global_max_nbins = int(np.max(memory.nbinsf[:len(freqs)]))
        fused_req = (block_size
                     + 2 * noverlap_int * global_max_nbins) * float_size
        if fused_req > shmem_lim:
            use_fused = False
        elif auto_freq_batch and _shmem_limits_occupancy(fused_req,
                                                         block_size):
            # Occupancy-aware chunking: launches size shared memory by
            # the max bin count of the frequencies they cover, and
            # ascending grids have monotonically decreasing bin counts
            # -- chunked launches let everything past the first chunks
            # run at full occupancy (measured +32% on the Kepler
            # config; only triggers when shared memory is the
            # occupancy limiter).
            freq_batch_size = _OCCUPANCY_FREQ_CHUNK

    # Phase oversampling: the kernel's box start positions step one
    # fine phase bin, so a single pass undersamples boxes whose width
    # is near the finest bin. Fused path: one launch builds the
    # noverlap-times finer histogram and evaluates all shifted grids.
    # Fallback: run ``noverlap`` passes with the bin grid shifted by
    # 1/noverlap of a bin each time and keep the elementwise max --
    # equivalent to the manual dphi re-run procedure this replaces.
    best_bls_g = None
    n_passes = 1 if use_fused else noverlap
    for i_pass in range(n_passes):
        dphi_pass = dphi + float(i_pass) / noverlap

        i_freq = 0
        while (i_freq < len(freqs)):
            j_freq = min([i_freq + freq_batch_size, len(freqs)])
            nfreqs = j_freq - i_freq

            # np.max, not builtin max(): iterating a 300K-element numpy
            # array through Python scalars cost 10+ ms per call at
            # HAT-Net/Kepler grid sizes.
            max_nbins = int(np.max(memory.nbinsf[i_freq:j_freq]))

            if use_fused:
                hist_size = noverlap_int * int(max_nbins)
            else:
                hist_size = int(max_nbins)
            mem_req = (block_size + 2 * hist_size) * float_size

            if mem_req > shmem_lim:
                s = "qmin = %.2e requires too much shared memory." \
                    % (1. / max_nbins)
                s += " Either try a larger value of qmin (> %e)" % (qmin_min)
                s += " or avoid using %s." % (
                    'eebls_gpu_fast_optimized' if use_optimized
                    else 'eebls_gpu_fast')
                raise ValueError(s)
            nblocks = min([nfreqs, max_nblocks])
            if force_nblocks is not None:
                nblocks = force_nblocks

            grid = (nblocks, 1)
            args = (grid, block)
            if stream is not None:
                args += (stream,)
            args += (memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr)
            args += (memory.bls_g.ptr, memory.freqs_g.ptr)
            args += (memory.nbins0_g.ptr, memory.nbinsf_g.ptr)
            args += (np.uint32(len(t)), np.uint32(nfreqs),
                     np.uint32(i_freq))
            if use_fused:
                # hist_size is the fine histogram size; the fused
                # kernel consumes the real noverlap and the base dphi.
                args += (np.uint32(hist_size), np.uint32(noverlap_int))
                args += (np.float32(dlogq), np.float32(dphi))
            else:
                # The kernel's own noverlap argument is a no-op in the
                # compiled (linear bin spacing) branch; phase
                # oversampling is implemented by the dphi-shifted
                # passes above.
                args += (np.uint32(max_nbins), np.uint32(1))
                args += (np.float32(dlogq), np.float32(dphi_pass))
            args += (np.uint32(ignore_negative_delta_sols),)

            launch_func = fused_func if use_fused else func
            if stream is not None:
                launch_func.prepared_async_call(*args,
                                                shared_size=int(mem_req))
            else:
                launch_func.prepared_call(*args, shared_size=int(mem_req))

            i_freq = j_freq

        if not use_fused and noverlap > 1:
            if best_bls_g is None:
                best_bls_g = memory.bls_g.copy()
            else:
                gpuarray.maximum(memory.bls_g, best_bls_g,
                                 out=best_bls_g, stream=stream)

    if best_bls_g is not None:
        cuda.memcpy_dtod(memory.bls_g.gpudata, best_bls_g.gpudata,
                         best_bls_g.nbytes)

    if transfer_to_host:
        memory.transfer_data_to_cpu()
        if stream is not None:
            stream.synchronize()
        # Use the chi2_0 of the data actually loaded in the memory: on
        # the memory-reuse path (memory= given, transfer_to_device=False)
        # the y/dy arguments may not be the data that produced this
        # periodogram, and 'snr'/'loglik' would be scaled by the wrong
        # null model.
        chi2_0 = getattr(memory, 'chi2_0', None)
        if chi2_0 is None:
            chi2_0 = _chi2_null(y, dy)
        return _convert_bls_power_from_chi2_0(memory.bls, chi2_0,
                                              convention)

    return memory.bls


def eebls_gpu_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   ignore_negative_delta_sols=False,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, noverlap=2, max_nblocks=5000,
                   force_nblocks=None, dphi=0.0,
                   shmem_lim=None, freq_batch_size=None,
                   transfer_to_device=True,
                   transfer_to_host=True, **kwargs):
    """
    Box-Least Squares with PyCUDA but about 2-3 orders of magnitude
    faster than eebls_gpu. Uses shared memory for the binned data,
    which means that there is a lower limit on the q values that
    this function can handle.

    To save memory and improve speed, the best solution is not
    kept. To get the best solution, run ``eebls_gpu`` at the
    optimal frequency.

    .. warning::

        If you are running on a single-GPU machine, there may be a
        kernel time limit set by your OS. If running this function
        produces a timeout error, try setting ``freq_batch_size`` to a
        reasonable number (~10). That will split up the computations by
        frequency.

    .. note::

        No extra global memory is needed, meaning you likely do *not* need
        to use ``large_run`` with this function.

    .. warning::

        BLS weights each observation by ``1/dy**2`` (normalized). A
        point with a near-zero reported uncertainty concentrates
        essentially all of the statistical weight in one phase bin and
        deterministically produces spurious power of ~0.99 in pure
        noise, at nearly every trial frequency. Symptoms:
        ``max(dy**-2) / sum(dy**-2)`` close to 1, and suspiciously
        high, nearly flat power on noise-like data. Guard with a
        percentile-based error floor before calling::

            dy_floor = np.percentile(dy, 10)
            dy = np.clip(dy, dy_floor, None)

        See the "Data hygiene: near-zero uncertainties" section of the
        BLS documentation for details.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency; scalar or one
        value per frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency; scalar or one
        value per frequency.

        .. note::

            The shared-memory kernels do not search a continuum of
            ``q``. Phase is binned into ``nbinsf = floor(1/qmin)``
            bins and a box is ``m`` of those bins, so the searched
            widths are ``q = m / nbinsf`` for ``m = 1, 1 + dnbins(1),
            ...`` up to ``floor(nbinsf / floor(1/qmax))`` -- the
            widest box with ``q <= 1/floor(1/qmax)``. The widest box
            is included (before 1.0 the loop stopped one level short
            and never tested ``qmax`` itself), but the geometric
            ``dlogq`` step can still skip it: with the defaults
            (``qmin=0.01``, ``qmax=0.5``, ``dlogq=0.3``) the widest
            tested width is ``q = 0.48``. Box start phases step one
            fine bin divided by ``noverlap``, so a box of ``m`` bins
            can be misaligned by up to ``1 / (2 m noverlap)`` of its
            width, which costs power: the Sep 2026 audit measured
            49-90 % of the exact float64 box power for boxes at or
            near ``qmin`` (``m`` of order 1). Raise ``noverlap``
            (nearly free on the fused path) or lower ``qmin`` if you
            need to compare fast-path power with an exact (e.g.
            astropy) box fit at face value; :func:`eebls_gpu` uses a
            finer q ladder.
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    noverlap: int, optional (default: 2)
        Phase-offset oversampling: the periodogram is the elementwise
        maximum over ``noverlap`` passes, with the phase-bin grid
        shifted by ``1/noverlap`` of the finest bin width between
        passes. This recovers box solutions whose phase offset falls
        between bin boundaries (important when the best ``q`` is close
        to ``qmin``); runtime scales linearly with ``noverlap``.
        ``noverlap=1`` is a single unshifted pass.
    convention: str, optional (default: 'chi2ratio')
        Power-spectrum convention for the returned periodogram
        ('chi2ratio', 'snr' or 'loglik'); see
        :func:`convert_bls_power`. Requires ``transfer_to_host=True``
        for non-default values.
    dphi: float, optional (default: 0.)
        Base phase-bin offset in units of the finest grid spacing;
        pass ``i_pass`` adds ``i_pass / noverlap`` to it.
    dlogq: float
        The logarithmic spacing of the q values to use. If negative,
        the q values increase by ``dq = qmin``.
    functions: dict
        Dictionary of compiled functions (see :func:`compile_bls`)
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; if
        ``None`` this will run a single batch for all frequencies
        simultaneously
    shmem_lim: int, optional (default: None)
        Maximum amount of shared memory to use per block in bytes.
        This is GPU-dependent but usually around 48KB. If ``None``,
        uses device information provided by PyCUDA (recommended).
    max_nblocks: int, optional (default: 5000)
        Maximum grid size to use
    force_nblocks: int, optional (default: None)
        If this is set the gridsize is forced to be this value
    memory: :class:`BLSMemory` instance, optional (default: None)
        See :class:`BLSMemory`.
    transfer_to_host: bool, optional (default: True)
        Transfer BLS back to CPU.
    transfer_to_device: bool, optional (default: True)
        Transfer data to GPU
    **kwargs:
        passed to `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to
        :math:`1 - \\chi_2(\\omega) / \\chi_2(constant)`

    Notes
    -----
    The phase fold is float32, so it resolves ``ulp(T * max(freqs))``:
    keep ``qmin / noverlap`` well above it or narrow boxes lose power
    (``q = 0.01`` boxes recover 3-15 % less than the exact float64 box
    at ``T * f > 7000``, 22 % less over a 10-year baseline at 20 c/d),
    and binned power moves by up to ~10 % with the fractional part of
    ``min(t)``. Because the kernels accumulate through float32 atomics,
    two identical calls differ by ~1e-8 to 1e-7. See "Precision and
    reproducibility" in the BLS documentation.

    """
    return _eebls_gpu_fast_impl(
        t, y, dy, freqs, 'full_bls_no_sol',
        kwargs.pop('use_optimized', False),
        qmin=qmin, qmax=qmax,
        ignore_negative_delta_sols=ignore_negative_delta_sols,
        functions=functions, stream=stream, dlogq=dlogq,
        memory=memory, noverlap=noverlap, max_nblocks=max_nblocks,
        force_nblocks=force_nblocks, dphi=dphi,
        shmem_lim=shmem_lim, freq_batch_size=freq_batch_size,
        transfer_to_device=transfer_to_device,
        transfer_to_host=transfer_to_host, **kwargs)


def eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   ignore_negative_delta_sols=False,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, noverlap=2, max_nblocks=5000,
                   force_nblocks=None, dphi=0.0,
                   shmem_lim=None, freq_batch_size=None,
                   transfer_to_device=True,
                   transfer_to_host=True, **kwargs):
    """
    Variant of eebls_gpu_fast built from the bls_optimized.cu module.

    Its multi-pass kernel (``full_bls_no_sol_optimized``) uses separate
    yw/w shared arrays (no bank conflicts) and a warp-shuffle finish
    for the block reduction. At the default power-of-two ``noverlap``
    with ``dphi=0`` both entry points launch the SAME fused kernel
    (``full_bls_no_sol_fused``, shared through bls_common.cuh), so they
    perform identically; only the multi-pass fallback (other
    ``noverlap`` values, ``dphi != 0``) differs, where the v1.0
    re-benchmark measured parity (~1.0x) rather than the 20-30 % once
    claimed here.

    All parameters are identical to eebls_gpu_fast.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency; scalar or one
        value per frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency; scalar or one
        value per frequency.

        .. note::

            The shared-memory kernels do not search a continuum of
            ``q``. Phase is binned into ``nbinsf = floor(1/qmin)``
            bins and a box is ``m`` of those bins, so the searched
            widths are ``q = m / nbinsf`` for ``m = 1, 1 + dnbins(1),
            ...`` up to ``floor(nbinsf / floor(1/qmax))`` -- the
            widest box with ``q <= 1/floor(1/qmax)``. The widest box
            is included (before 1.0 the loop stopped one level short
            and never tested ``qmax`` itself), but the geometric
            ``dlogq`` step can still skip it: with the defaults
            (``qmin=0.01``, ``qmax=0.5``, ``dlogq=0.3``) the widest
            tested width is ``q = 0.48``. Box start phases step one
            fine bin divided by ``noverlap``, so a box of ``m`` bins
            can be misaligned by up to ``1 / (2 m noverlap)`` of its
            width, which costs power: the Sep 2026 audit measured
            49-90 % of the exact float64 box power for boxes at or
            near ``qmin`` (``m`` of order 1). Raise ``noverlap``
            (nearly free on the fused path) or lower ``qmin`` if you
            need to compare fast-path power with an exact (e.g.
            astropy) box fit at face value; :func:`eebls_gpu` uses a
            finer q ladder.
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    noverlap: int, optional (default: 2)
        Phase-offset oversampling (elementwise max over ``noverlap``
        bin-grid-shifted passes); see :func:`eebls_gpu_fast`.
    dphi: float, optional (default: 0.)
        Base phase-bin offset (in units of the finest grid spacing)
    dlogq: float
        The logarithmic spacing of the q values to use
    functions: dict
        Dictionary of compiled functions (see :func:`compile_bls`)
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch
    shmem_lim: int, optional (default: None)
        Maximum amount of shared memory to use per block in bytes
    max_nblocks: int, optional (default: 5000)
        Maximum grid size to use
    force_nblocks: int, optional (default: None)
        If this is set the gridsize is forced to be this value
    memory: :class:`BLSMemory` instance, optional (default: None)
        See :class:`BLSMemory`.
    transfer_to_host: bool, optional (default: True)
        Transfer BLS back to CPU.
    transfer_to_device: bool, optional (default: True)
        Transfer data to GPU
    **kwargs:
        passed to `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to
        :math:`1 - \\chi_2(\\omega) / \\chi_2(constant)`

    """
    kwargs.pop('use_optimized', None)
    return _eebls_gpu_fast_impl(
        t, y, dy, freqs, 'full_bls_no_sol_optimized', True,
        qmin=qmin, qmax=qmax,
        ignore_negative_delta_sols=ignore_negative_delta_sols,
        functions=functions, stream=stream, dlogq=dlogq,
        memory=memory, noverlap=noverlap, max_nblocks=max_nblocks,
        force_nblocks=force_nblocks, dphi=dphi,
        shmem_lim=shmem_lim, freq_batch_size=freq_batch_size,
        transfer_to_device=transfer_to_device,
        transfer_to_host=transfer_to_host, **kwargs)


def eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   ignore_negative_delta_sols=False,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, noverlap=2, max_nblocks=5000,
                   force_nblocks=None, dphi=0.0,
                   shmem_lim=None, freq_batch_size=None,
                   transfer_to_device=True,
                   transfer_to_host=True,
                   use_optimized=True,
                   **kwargs):
    """
    Adaptive BLS with dynamic block sizing for optimal performance.

    Automatically selects optimal block size based on ndata:
    - ndata <= 32: 32 threads (single warp)
    - ndata <= 64: 64 threads (two warps)
    - ndata <= 128: 128 threads (four warps)
    - ndata > 128: 256 threads (eight warps)

    Smaller blocks reduce idle-thread overhead for small datasets.
    Measured benefit is modest: the v1.0 re-benchmark (warm kernel
    cache) puts the block-size effect at ~1.0-1.3x vs the fixed
    256-thread default (earlier 1.4-5.3x figures were dominated by
    per-call kernel handling that the kernel cache now amortizes; see
    ``benchmark_results_by_gpu/block_size_a5000.json``).

    All other parameters identical to eebls_gpu_fast.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency; scalar or one
        value per frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency; scalar or one
        value per frequency.

        .. note::

            The shared-memory kernels do not search a continuum of
            ``q``. Phase is binned into ``nbinsf = floor(1/qmin)``
            bins and a box is ``m`` of those bins, so the searched
            widths are ``q = m / nbinsf`` for ``m = 1, 1 + dnbins(1),
            ...`` up to ``floor(nbinsf / floor(1/qmax))`` -- the
            widest box with ``q <= 1/floor(1/qmax)``. The widest box
            is included (before 1.0 the loop stopped one level short
            and never tested ``qmax`` itself), but the geometric
            ``dlogq`` step can still skip it: with the defaults
            (``qmin=0.01``, ``qmax=0.5``, ``dlogq=0.3``) the widest
            tested width is ``q = 0.48``. Box start phases step one
            fine bin divided by ``noverlap``, so a box of ``m`` bins
            can be misaligned by up to ``1 / (2 m noverlap)`` of its
            width, which costs power: the Sep 2026 audit measured
            49-90 % of the exact float64 box power for boxes at or
            near ``qmin`` (``m`` of order 1). Raise ``noverlap``
            (nearly free on the fused path) or lower ``qmin`` if you
            need to compare fast-path power with an exact (e.g.
            astropy) box fit at face value; :func:`eebls_gpu` uses a
            finer q ladder.
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta
    use_optimized: bool, optional (default: True)
        Use optimized kernel with bank conflict fixes and warp shuffles
    **kwargs:
        All other parameters passed to underlying implementation

    Returns
    -------
    bls: array_like, float
        BLS periodogram

    See Also
    --------
    eebls_gpu_fast : Standard implementation with fixed block size
    eebls_gpu_fast_optimized : Optimized implementation
    """
    # Validated here as well as in the shared implementation: this
    # wrapper compiles a kernel (GPU work) before it delegates.
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA,
                     name='eebls_gpu_fast_adaptive')
    check_freqs(freqs, name='eebls_gpu_fast_adaptive')
    _validate_fast_q_bounds(len(freqs), qmin, qmax)

    ndata = len(t)

    # Choose optimal block size
    block_size = _choose_block_size(ndata)

    # Override any user-provided block_size
    kwargs['block_size'] = block_size

    # Get cached kernels for this block size
    if functions is None:
        fname = 'full_bls_no_sol_optimized' if use_optimized else 'full_bls_no_sol'
        functions = _get_cached_kernels(block_size, use_optimized, [fname])

    # Use optimized implementation
    if use_optimized:
        return eebls_gpu_fast_optimized(
            t, y, dy, freqs, qmin=qmin, qmax=qmax,
            ignore_negative_delta_sols=ignore_negative_delta_sols,
            functions=functions, stream=stream, dlogq=dlogq,
            memory=memory, noverlap=noverlap, max_nblocks=max_nblocks,
            force_nblocks=force_nblocks, dphi=dphi,
            shmem_lim=shmem_lim, freq_batch_size=freq_batch_size,
            transfer_to_device=transfer_to_device,
            transfer_to_host=transfer_to_host,
            **kwargs)
    else:
        return eebls_gpu_fast(
            t, y, dy, freqs, qmin=qmin, qmax=qmax,
            ignore_negative_delta_sols=ignore_negative_delta_sols,
            functions=functions, stream=stream, dlogq=dlogq,
            memory=memory, noverlap=noverlap, max_nblocks=max_nblocks,
            force_nblocks=force_nblocks, dphi=dphi,
            shmem_lim=shmem_lim, freq_batch_size=freq_batch_size,
            transfer_to_device=transfer_to_device,
            transfer_to_host=transfer_to_host,
            **kwargs)


def eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                     ignore_negative_delta_sols=False,
                     freq_batch_size=None, nstreams=5, max_memory=None,
                     functions=None, convention='chi2ratio', **kwargs):
    """
    Box-Least Squares, with custom q and phi values. Useful
    if you're honing the initial solution or testing between
    a relatively small number of possible solutions.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    q_values: array_like
        Set of q values to search at each trial frequency
    phi_values: float or array_like
        Set of transit start phases to search at each trial frequency.
        These are ABSOLUTE phases, ``(t * f) mod 1`` in the original
        input timescale (the same convention as the ``phi`` returned
        by :func:`eebls_gpu` and accepted by :func:`single_bls`); they
        are re-referenced internally to the subtracted epoch in float64.
        Consequently the same coarse ``phi_values`` grid samples
        different absolute phases for ``t`` and ``t + 2457000.5``, and
        low-power frequencies can differ between the two (a fine grid,
        or :func:`hone_solution`, makes this negligible).
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determined
        automatically from ``max_memory`` when ``None``; capped at
        ``len(freqs)`` and at ``(2**31 - 1) // len(t)`` either way.
    max_memory: float, optional (default: None)
        Memory budget in bytes for the device scratch buffers. Ignored
        if ``freq_batch_size`` is specified; ``None`` budgets half of
        the free memory reported by ``pycuda.driver.mem_get_info()``.
    functions: tuple of CUDA functions
        Dictionary of prepared functions from :func:`compile_bls`.
    **kwargs:
        passed to :func:`compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    qphi_sols: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """
    # Validate before any GPU work: an unknown convention would
    # otherwise only raise at the return statement, after the whole
    # multi-stream grid search has run.
    _validate_convention(convention)
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA,
                     name='eebls_gpu_custom')
    check_freqs(freqs, name='eebls_gpu_custom')

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    block_size = kwargs.get('block_size', _default_block_size)
    ndata = len(t)
    nfreq = len(freqs)

    # default budget: half of the free device memory, bounded below by
    # what the grid needs (see _DEFAULT_MEMORY_FRACTION)
    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(_DEFAULT_MEMORY_FRACTION * free)

    if freq_batch_size is None:
        # compute memory
        real_type_size = 4

        # data
        mem0 = ndata * 3 * real_type_size

        nq = len(q_values)
        nphi = len(phi_values)

        # q_values (float32) and phi_values (float64)
        mem0 += nq * real_type_size + nphi * 2 * real_type_size

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += nfreq * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nq * nphi * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size <= 0:
            raise RuntimeError("Not enough memory (freq_batch_size = 0)")

    # cap at len(freqs) and at (2^31 - 1) // ndata (fold launch geometry;
    # see _cap_freq_batch_size)
    freq_batch_size = _cap_freq_batch_size(freq_batch_size, ndata, nfreq)

    nbtot = len(q_values) * len(phi_values) * freq_batch_size

    # move data to GPU
    w = np.power(dy, -2)
    w /= np.sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(np.array(y) - ybar, 2))
    yw = (np.array(y) - ybar) * np.array(w)

    t, epoch = subtract_epoch(t)
    t_g = gpuarray.to_gpu(t.astype(np.float32))
    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float64))

    nbatches = int(np.ceil(float(nfreq) / freq_batch_size))

    # One scratch set per stream, but never more streams than batches
    # (a single-batch grid does not need nstreams x 4 zero-filled
    # buffers).
    nsets = max(1, min(int(nstreams), nbatches))
    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nsets):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(nbtot, dtype=np.uint32))

    bls_g = gpuarray.zeros(nfreq, dtype=np.float32)
    bls_sol_g = gpuarray.zeros(nfreq, dtype=np.uint32)

    bls_best_phi = gpuarray.zeros(nfreq, dtype=np.float32)
    bls_best_q = gpuarray.zeros(nfreq, dtype=np.float32)

    q_values_g = gpuarray.to_gpu(np.asarray(q_values).astype(np.float32))
    # phi values stay float64: the kernel re-references them to the
    # subtracted epoch as (phi - epoch*freq) % 1 in double precision
    # (epoch*freq can be ~1e6 cycles), matching single_bls bit for bit
    phi_values_g = gpuarray.to_gpu(np.asarray(phi_values).astype(np.float64))

    block = (block_size, 1, 1)

    bin_func = functions['bin_and_phase_fold_custom']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols_custom']

    for batch in range(nbatches):
        imin = freq_batch_size * batch
        imax = min([nfreq, freq_batch_size * (batch + 1)])

        nf = imax - imin
        j = batch % nsets
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)
        bls_tmp_g.fill(np.float32(0), stream=stream)
        bls_tmp_sol_g.fill(np.int32(0), stream=stream)

        bin_grid = (int(np.ceil(float(len(t) * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr, np.float64(epoch))
        args += (np.uint32(len(q_values)), np.uint32(len(phi_values)))
        args += (np.uint32(len(t)), np.uint32(nf))
        args += (np.uint32(freq_batch_size * batch),)
        bin_func.prepared_async_call(*args)

        nb = len(q_values) * len(phi_values)

        bls_grid = (int(np.ceil(float(nf * nb) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.uint32(nf * nb))
        args += (np.uint32(ignore_negative_delta_sols),)
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nb, stream, bls_g, bls_sol_g)
        args += (batch * freq_batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr)
        args += (np.uint32(len(q_values)), np.uint32(len(phi_values)))
        args += (np.uint32(nf), np.uint32(batch * freq_batch_size))
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))

    return (convert_bls_power(bls_g.get() / YY, y, dy,
                              convention=convention),
            qphi_sols)


def dnbins(nbins, dlogq):
    if (dlogq < 0):
        return 1

    n = int(np.floor(dlogq * nbins))

    return n if n > 0 else 1


def nbins_iter(i, nb0, dlogq):
    nb = nb0
    for j in range(i):
        nb += dnbins(nb, dlogq)

    return nb


def count_tot_nbins(nbins0, nbinsf, dlogq):
    ntot = 0

    i = 0
    while nbins_iter(i, nbins0, dlogq) <= nbinsf:
        ntot += nbins_iter(i, nbins0, dlogq)
        i += 1
    return ntot


@functools.lru_cache(maxsize=65536)
def _count_tot_nbins_cached(nbins0, nbinsf, dlogq):
    """``count_tot_nbins`` memoized on its (small) set of distinct
    arguments: the batch table below evaluates it once per batch (or
    once per frequency on the per-frequency path), and Keplerian grids
    have only a few hundred distinct ``(nbins0, nbinsf)`` pairs."""
    return count_tot_nbins(int(nbins0), int(nbinsf), float(dlogq))


# Fraction of the free device memory that eebls_gpu / eebls_gpu_custom
# budget by default. The allocation is further bounded by what the
# frequency grid actually needs (freq_batch_size is capped at
# len(freqs)), so small grids allocate only a few MB; large grids
# leave half the device to other processes instead of taking ~90% of
# it for a transient zero-filled scratch buffer (Sep 2026 audit,
# finding 135 / plan item BLS-2).
_DEFAULT_MEMORY_FRACTION = 0.5

# Largest ndata * (frequencies per batch) product one fold launch may
# cover: bin_and_phase_fold_bst_multifreq / bin_and_phase_fold_custom
# run one thread per (observation, frequency) pair and the grid size
# must stay a sane 32-bit block count. The kernels index in 64 bits, so
# this cap is a launch-geometry bound, not a correctness requirement.
_MAX_FOLD_THREADS = 2 ** 31 - 1


def _cap_freq_batch_size(freq_batch_size, ndata, nfreq):
    """Bound a (user-supplied or auto-sized) ``freq_batch_size``.

    The batch never exceeds the frequency grid (a 300-frequency grid
    used to allocate scratch space for the ~100K-frequency batch the
    memory budget allowed) and ``ndata * freq_batch_size`` never
    exceeds ``_MAX_FOLD_THREADS`` (the fold kernels used to be launched
    with a 32-bit ``ndata * nfreq`` bound that wrapped at 2^32 --
    defect 1 of the Sep 2026 audit). Always >= 1.
    """
    cap = max(1, _MAX_FOLD_THREADS // max(1, int(ndata)))
    return int(max(1, min(int(freq_batch_size), cap, int(nfreq))))


def _q_bounds_to_nbins(qmins, qmaxes):
    """Per-frequency bin counts for the binned (eebls_gpu) kernels:
    ``nbins0 = floor(1/qmax)`` (coarsest) and ``nbinsf = ceil(1/qmin)``
    (finest), as int64 arrays. The bounds must already have passed
    ``_validate_q_bounds``; the binned kernels additionally need
    ``qmin > 0`` (a finite finest bin count) and ``qmax <= 1``
    (``nbins0 >= 1``; ``nbins0 = 0`` divides by zero on the device)."""
    qmins = np.asarray(qmins, dtype=np.float64)
    qmaxes = np.asarray(qmaxes, dtype=np.float64)
    _check_q_bounds_for_bins(qmins, qmaxes)
    nbins0 = np.floor(1. / qmaxes).astype(np.int64)
    nbinsf = np.ceil(1. / qmins).astype(np.int64)
    return nbins0, nbinsf


def _per_freq_nbins_tot(nbins0, nbinsf, dlogq):
    """``count_tot_nbins(nbins0[i], nbinsf[i], dlogq)`` for every
    frequency, evaluated once per distinct ``(nbins0, nbinsf)`` pair (a
    Keplerian grid of 10^5 frequencies has only a few hundred)."""
    nbins0 = np.asarray(nbins0, dtype=np.int64)
    nbinsf = np.asarray(nbinsf, dtype=np.int64)
    pairs = np.stack([nbins0, nbinsf], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    counts = np.array([_count_tot_nbins_cached(int(a), int(b), dlogq)
                       for a, b in uniq], dtype=np.int64)
    return counts[np.asarray(inv).ravel()]


def _max_nbins_tot(nbins0, nbinsf, dlogq):
    """Largest number of (phase bin, q level) cells any single frequency
    needs -- the bound used to budget memory before batching.

    Note ``count_tot_nbins(nb0, nbf, dlogq)`` is non-decreasing in
    ``nbf`` but NOT monotone in ``nb0``: at ``nbf = 359`` and
    ``dlogq = 0.2`` it is 1875, 1939 and 1704 for ``nb0`` = 28, 29, 30.
    The pre-1.0 sizing used the value at the grid-wide ``(min nb0, max
    nbf)`` collapse, which a batch starting at a larger ``nb0`` could
    exceed (the 44 MB overrun of the audit's 70,000-point ``fmin=0.02,
    fmax=0.5`` case). The kernels now work per frequency, so the exact
    per-frequency maximum is the bound.
    """
    return int(np.max(_per_freq_nbins_tot(nbins0, nbinsf, dlogq)))


def _bls_batch_table(nbins0, nbinsf, freq_batch_size, dlogq):
    """Batch table for :func:`eebls_gpu`, built BEFORE the device
    scratch buffers are allocated so they can be sized from the actual
    maximum over batches.

    Returns a list of ``(imin, imax, nbins_tot_b)`` per batch of
    ``freq_batch_size`` frequencies, where ``nbins_tot_b`` is the
    batch's row stride: the largest per-frequency
    ``count_tot_nbins(nbins0[i], nbinsf[i], dlogq)`` among its
    frequencies (each frequency writes only its own cells; the rest of
    its row stays zero). Batch boundaries never change which boxes a
    frequency searches.
    """
    nbins_tot_f = _per_freq_nbins_tot(nbins0, nbinsf, dlogq)
    nfreq = len(nbins_tot_f)
    freq_batch_size = int(freq_batch_size)
    if freq_batch_size < 1:
        raise ValueError("freq_batch_size must be >= 1")
    table = []
    for imin in range(0, nfreq, freq_batch_size):
        imax = min(nfreq, imin + freq_batch_size)
        table.append((imin, imax, int(np.max(nbins_tot_f[imin:imax]))))
    return table


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              ignore_negative_delta_sols=False,
              nstreams=5, noverlap=3, dlogq=0.2, max_memory=None,
              freq_batch_size=None, functions=None,
              convention='chi2ratio', **kwargs):

    """
    Box-Least Squares, accelerated with PyCUDA

    .. warning::

        BLS weights each observation by ``1/dy**2`` (normalized). A
        point with a near-zero reported uncertainty concentrates
        essentially all of the statistical weight in one phase bin and
        deterministically produces spurious power of ~0.99 in pure
        noise, at nearly every trial frequency. Symptoms:
        ``max(dy**-2) / sum(dy**-2)`` close to 1, and suspiciously
        high, nearly flat power on noise-like data. Guard with a
        percentile-based error floor before calling::

            dy_floor = np.percentile(dy, 10)
            dy = np.clip(dy, dy_floor, None)

        See the "Data hygiene: near-zero uncertainties" section of the
        BLS documentation for details.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like
        Minimum q value(s) to test for each frequency. A scalar applies
        to every frequency; an array (same length as ``freqs``) gives a
        per-frequency bound. The finest phase bin at frequency ``i`` is
        ``1 / ceil(1 / qmin[i])``.
    qmax: float or array_like
        Maximum q value(s) to test for each frequency (scalar or
        per-frequency array). The coarsest bin is
        ``1 / floor(1 / qmax[i])``. Per-frequency bounds are honoured
        exactly per frequency (each frequency searches only its own
        q levels), so results do not depend on ``freq_batch_size`` or on
        the free device memory.
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 3)
        Phase-offset oversampling: each q level is evaluated on
        ``noverlap`` phase-bin grids shifted by ``1/noverlap`` of a bin
        (``phi = q * (j + s / noverlap)``), not extra q levels.
    dlogq: float, optional, (default: 0.2)
        logarithmic spacing of :math:`q` values, where :math:`d\\log q = dq / q`
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determined
        automatically from ``max_memory`` when ``None``. Whether given
        or automatic, it is capped at ``len(freqs)`` and at
        ``(2**31 - 1) // len(t)`` (one fold thread per (observation,
        frequency) pair per launch).
    max_memory: float, optional (default: None)
        Memory budget in bytes for the device scratch buffers (four
        arrays per stream, sized by the frequency batch). Ignored if
        ``freq_batch_size`` is specified. ``None`` budgets half of the
        free memory reported by ``pycuda.driver.mem_get_info``; the
        allocation never exceeds what ``len(freqs)`` frequencies need.
    functions: tuple of CUDA functions
        returned by ``compile_bls``
    convention: str, optional (default: 'chi2ratio')
        Power-spectrum convention for the returned periodogram
        ('chi2ratio', 'snr' or 'loglik'); see
        :func:`convert_bls_power`.

    Returns
    -------
    bls: array_like, float
        BLS periodogram; in the default convention, normalized to
        :math:`1 - \\chi^2(f) / \\chi^2_0`
    qphi_sols: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

    """

    _validate_convention(convention)
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name='eebls_gpu')
    check_freqs(freqs, name='eebls_gpu')

    block_size = kwargs.get('block_size', _default_block_size)
    ndata = len(t)
    nfreq = len(freqs)

    # Per-frequency bin counts (scalar bounds broadcast). Validated
    # before any device work (including the kernel compile): qmin >
    # qmax used to surface as a ZeroDivisionError from count_tot_nbins,
    # qmax > 1 as a device divide-by-zero.
    qmins = _broadcast_q_bound(qmin, nfreq, 1e-2, 'qmin')
    qmaxes = _broadcast_q_bound(qmax, nfreq, 0.5, 'qmax')
    _validate_q_bounds(qmins, qmaxes)
    nbins0_f, nbinsf_f = _q_bounds_to_nbins(qmins, qmaxes)

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(_DEFAULT_MEMORY_FRACTION * free)

    real_type_size = np.float32(1).nbytes

    if freq_batch_size is None:
        # data
        mem0 = ndata * 3 * real_type_size

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += nfreq * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32), sized
        # by the largest per-frequency cell count (the batch stride can
        # never exceed it; see _max_nbins_tot)
        nbins_tot_bound = _max_nbins_tot(nbins0_f, nbinsf_f, dlogq)
        mem_per_f = 4 * nstreams * nbins_tot_bound * noverlap * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size <= 0:
            raise RuntimeError("Not enough memory (freq_batch_size = 0)")

    # Cap user-supplied and automatic batch sizes alike: at len(freqs)
    # (allocate only what the grid needs) and at (2^31 - 1) // ndata.
    freq_batch_size = _cap_freq_batch_size(freq_batch_size, ndata, nfreq)

    # The batch table is built BEFORE allocating so the scratch buffers
    # are sized from the actual maximum over batches. The old code
    # sized them from count_tot_nbins(grid-wide min nbins0, grid-wide
    # max nbinsf), which is not an upper bound (non-monotone in
    # nbins0) -- a batch could need more cells than were allocated and
    # the fold kernel's atomics ran off the end of the buffer (illegal
    # memory access on the default eebls_transit path; audit defect 1).
    batches = _bls_batch_table(nbins0_f, nbinsf_f, freq_batch_size, dlogq)
    nbatches = len(batches)
    gs = max((imax - imin) * nbins_tot for
             (imin, imax, nbins_tot) in batches) * noverlap

    # move data to GPU
    w = np.power(dy, -2)
    w /= np.sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(np.array(y) - ybar, 2))
    yw = (np.array(y) - ybar) * np.array(w)

    t, epoch = subtract_epoch(t)
    t_g = gpuarray.to_gpu(t.astype(np.float32))
    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))

    # Per-frequency bin counts, read by the fold and store kernels at
    # index (i_freq + freq_offset): every frequency searches exactly its
    # own q levels. Before 1.0 the kernels took one scalar pair per
    # launch, so array bounds collapsed to the batch-wide (min nbins0,
    # max nbinsf) window and the result depended on freq_batch_size /
    # free memory (Sep 2026 audit defect 7).
    nbins0_g = gpuarray.to_gpu(nbins0_f.astype(np.uint32))
    nbinsf_g = gpuarray.to_gpu(nbinsf_f.astype(np.uint32))

    # One scratch set per stream, but never more streams than batches
    # (a 3-batch grid does not need 5 x 4 zero-filled buffers).
    nsets = max(1, min(int(nstreams), nbatches))
    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nsets):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(gs, dtype=np.int32))

    bls_g = gpuarray.zeros(nfreq, dtype=np.float32)
    bls_sol_g = gpuarray.zeros(nfreq, dtype=np.int32)

    bls_best_phi = gpuarray.zeros(nfreq, dtype=np.float32)
    bls_best_q = gpuarray.zeros(nfreq, dtype=np.float32)

    block = (block_size, 1, 1)

    bin_func = functions['bin_and_phase_fold_bst_multifreq']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols']

    for batch, (imin, imax, nbins_tot) in enumerate(batches):

        nf = imax - imin
        all_bins = nf * nbins_tot * noverlap
        if all_bins > gs:
            # cannot happen with the table-derived gs above; guard the
            # device against ever overrunning its buffers again
            raise ValueError(
                "eebls_gpu: batch %d needs %d bin cells but only %d were "
                "allocated (nbins_tot=%d, noverlap=%d)"
                % (batch, all_bins, gs, nbins_tot, noverlap))

        j = batch % nsets
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)
        bls_tmp_g.fill(np.float32(0), stream=stream)
        bls_tmp_sol_g.fill(np.int32(0), stream=stream)

        bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (nbins0_g.ptr, nbinsf_g.ptr)
        args += (np.uint32(ndata), np.uint32(nf))
        args += (np.uint32(imin), np.uint32(noverlap))
        args += (np.float32(dlogq), np.uint32(nbins_tot))
        bin_func.prepared_async_call(*args)

        bls_grid = (int(np.ceil(float(all_bins) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(all_bins))
        args += (np.uint32(ignore_negative_delta_sols),)
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (imin, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (nbins0_g.ptr, nbinsf_g.ptr, np.uint32(noverlap))
        args += (np.float32(dlogq), np.uint32(nf))
        args += (np.uint32(imin),)
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))
    # Adjust phases to original timescale
    qphi_sols = [(q, (phi + (epoch * freq)) % 1.0) for (q, phi), freq in zip(qphi_sols, freqs)]

    return (convert_bls_power(bls_g.get() / YY, y, dy,
                              convention=convention),
            qphi_sols)


def single_bls(t, y, dy, freq, q, phi0, ignore_negative_delta_sols=False):
    """
    Evaluate BLS power for a single set of (freq, q, phi0)
    parameters.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freq: float
        Frequency of the signal
    q: float
        Transit duration in phase
    phi0: float
        Phase offset of transit, in the ORIGINAL input timescale
        (internally re-referenced to the subtracted epoch, consistent
        with the phases reported by the GPU functions in this module)
    ignore_negative_delta_sols:
        Whether or not to ignore solutions with negative delta (inverted dips)

    Returns
    -------
    bls: float
        BLS power for this set of parameters
    """
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name='single_bls')
    if not (np.isfinite(freq) and np.isfinite(q) and np.isfinite(phi0)):
        raise ValueError("single_bls: freq, q and phi0 must be finite; "
                         "got freq=%r, q=%r, phi0=%r" % (freq, q, phi0))
    if freq <= 0:
        raise ValueError("single_bls: freq must be > 0; got %r" % (freq,))

    # Epoch-subtract before the float32 cast
    t, epoch = subtract_epoch(t)

    # Adjust phase offset to subtracted timescale
    phi0 = (phi0 - (epoch * freq)) % 1.0

    phi = t.astype(np.float32) * np.float32(freq)
    # Wrap into [0, 1) BEFORE subtracting the phase offset, exactly like
    # the GPU kernels' mod1(t * f) (verified bit-identical to the
    # compiled kernels' fold on hardware; nvcc does not FMA-contract the
    # mod1 expression). Subtracting phi0 first -- the old order --
    # happens at magnitude ~t*f, where float32 resolution is only
    # ulp(t*f)/2 ~ 1.5e-5 phase for a 1-year baseline (2.4e-4 for 10
    # years), so points within that fuzz of a box edge acquired the
    # wrong membership relative to the kernels' full-resolution [0, 1)
    # fold. Wrapping first shrinks the CPU-vs-GPU edge-disagreement
    # window by ~2 orders of magnitude, to the float32 rounding of the
    # kernels' bin-index arithmetic (~1e-7).
    phi -= np.floor(phi)
    phi -= np.float32(phi0)
    phi -= np.floor(phi)

    mask = phi < np.float32(q)

    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))

    # Centre in float64 before the float32 cast (defect 8 of the Sep
    # 2026 audit: float32 sums of raw mag-12 fluxes minus ybar * W lost
    # 1e-3..1e-2 of the power). ybar of the centred float32 flux is
    # residual roundoff (~1e-8), kept for parity with the kernels.
    yc, _ = _center_flux_float64(y, dy)
    ybar = np.dot(w, yc)
    YY = np.dot(w, np.power(yc - ybar, 2))

    W = np.sum(w[mask])
    YW = np.dot(w[mask], yc[mask]) - ybar * W

    if YW > 0 and ignore_negative_delta_sols:
        return 0
    # Upper bound mirrors the GPU kernels' bls_value: a box holding
    # (nearly) all the statistical weight has no out-of-transit baseline
    # and its power is roundoff-divided-by-roundoff (this function sums
    # float32-cast quantities like the kernels do).
    if W < 1e-9 or W > 1 - 1e-4:
        return 0
    return (YW ** 2) / (W * (1 - W)) / YY


_BLS_POWER_CONVENTIONS = ('chi2ratio', 'snr', 'loglik')


def _chi2_null(y, dy):
    """Weighted chi-squared of the constant (weighted-mean) model,
    computed in float64; the normalization connecting the BLS power
    conventions."""
    y = np.asarray(y, dtype=np.float64)
    w = np.power(np.asarray(dy, dtype=np.float64), -2)
    # einsum, not np.dot: keep the per-LC path off BLAS threadpools
    # (CFS-throttling cliff on CPU-quota-limited hosts; see
    # BLSMemory.setdata).
    ybar = float(np.einsum('i,i->', w, y)) / np.sum(w)
    return float(np.einsum('i,i->', w, np.power(y - ybar, 2)))


def _validate_convention(convention):
    if convention not in _BLS_POWER_CONVENTIONS:
        raise ValueError("convention must be one of %s, got %r"
                         % (_BLS_POWER_CONVENTIONS, convention))


def convert_bls_power(power, y, dy, convention='chi2ratio'):
    """
    Convert the native BLS power to another power-spectrum convention.

    The native convention ('chi2ratio') is

    .. math::

        P = 1 - \\chi^2 / \\chi^2_0

    where :math:`\\chi^2` is the weighted sum of squared residuals of
    the best-fit box and :math:`\\chi^2_0` that of the constant
    (weighted-mean) model.

    Parameters
    ----------
    power: array_like or float
        BLS power(s) in the native 'chi2ratio' convention.
    y: array_like, float
        Observations (used to compute :math:`\\chi^2_0`).
    dy: array_like, float
        Observation uncertainties.
    convention: str, optional (default: 'chi2ratio')
        One of:

        * ``'chi2ratio'``: the native power, returned unchanged.
        * ``'snr'``: :math:`\\sqrt{\\chi^2_0 P}` -- the unsigned
          signal-to-noise of the best-fit transit depth,
          :math:`|\\hat{\\delta}| / \\sigma_{\\hat{\\delta}}`. Equals
          ``astropy.timeseries.BoxLeastSquares`` power with
          ``objective='snr'`` at the same (period, duration, phase)
          (astropy reports it signed and keeps dips only).
        * ``'loglik'``: :math:`\\chi^2_0 P / 2` -- the improvement in
          Gaussian log-likelihood of the best two-level (in/out of
          transit) model over the constant weighted-mean model.
          Note: astropy's ``objective='likelihood'`` power uses the
          out-of-transit level as its reference instead, so it equals
          this value divided by :math:`(1 - r)`, with :math:`r` the
          in-transit fraction of the total statistical weight; the
          two agree in the transit limit :math:`q \\ll 1`.

    Returns
    -------
    power: array_like or float
        Power(s) in the requested convention.
    """
    _validate_convention(convention)
    return _convert_bls_power_from_chi2_0(power, None, convention,
                                          y=y, dy=dy)


def _convert_bls_power_from_chi2_0(power, chi2_0, convention,
                                   y=None, dy=None):
    """``convert_bls_power`` with a precomputed :math:`\\chi^2_0`
    (falls back to computing it from ``y``/``dy`` when ``None``)."""
    _validate_convention(convention)
    if convention == 'chi2ratio':
        return power
    if chi2_0 is None:
        chi2_0 = _chi2_null(y, dy)
    if convention == 'snr':
        return np.sqrt(chi2_0 * np.asarray(power))
    return 0.5 * chi2_0 * np.asarray(power)  # 'loglik'


def _broadcast_q_bound(value, nfreqs, default, name):
    """Broadcast a transit-duration bound (scalar or per-frequency
    array; ``None`` means ``default``) to a float array of length
    ``nfreqs``."""
    if value is None:
        value = default
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if len(arr) == 1:
        arr = np.full(nfreqs, arr[0])
    elif len(arr) != nfreqs:
        raise ValueError("%s must be a scalar or have the same length "
                         "as freqs (%d); got length %d"
                         % (name, nfreqs, len(arr)))
    return arr


def _center_flux_float64(y, dy):
    """Weighted-mean-subtract ``y`` in float64 and return the centred
    flux as float32 (plus the float64 normalized weights).

    The sparse kernels and :func:`single_bls` accumulate float32 sums
    of ``w * y``; with raw fluxes of magnitude ~12 (or normalized flux
    ~1) those partial sums carry the mean, and subtracting
    ``ybar * W`` afterwards cancels catastrophically: 1e-3..1e-2
    relative power errors on mag-12 data and powers > 1 when one point
    is ~1e3x more precise than the rest (Sep 2026 audit, defect 8).
    Centring in float64 BEFORE the float32 cast (as the binned path
    always did) leaves ~1e-6.
    """
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.power(np.asarray(dy, dtype=np.float64), -2)
    w64 /= np.sum(w64)
    ybar = float(np.einsum('i,i->', w64, y64))
    return (y64 - ybar).astype(np.float32), w64


def _validate_q_bounds(qmins, qmaxes):
    """Reject transit-duration bounds that would silently produce an
    all-zero periodogram (every candidate box rejected)."""
    if not (np.all(np.isfinite(qmins)) and np.all(np.isfinite(qmaxes))):
        raise ValueError("qmin/qmax must be finite")
    if np.any(qmins < 0):
        raise ValueError("qmin must be >= 0 (0 disables the lower bound)")
    if np.any(qmaxes <= 0):
        raise ValueError("qmax must be > 0")
    if np.any(qmins > qmaxes):
        raise ValueError("qmin > qmax for %d frequencies; every candidate "
                         "transit would be rejected"
                         % int(np.sum(qmins > qmaxes)))


def _check_q_bounds_for_bins(qmins, qmaxes):
    """The two extra conditions the *binned* BLS kernels impose on top
    of :func:`_validate_q_bounds`.

    ``qmin > 0``: the finest phase bin is ``1/qmin`` wide, so ``qmin =
    0`` asks for infinitely many bins (and casts to a bin count of 0).
    ``qmax <= 1``: the coarsest bin count is ``1/qmax``, and ``nbins0 =
    0`` divides by zero inside the kernel and lets its ``atomicAdd``
    run outside the shared-memory histogram.
    """
    qmins = np.asarray(qmins, dtype=np.float64)
    qmaxes = np.asarray(qmaxes, dtype=np.float64)
    if np.any(qmins <= 0):
        raise ValueError("qmin must be > 0 for the binned BLS kernels "
                         "(the finest phase bin is 1/qmin wide); got "
                         "min(qmin) = %g" % float(np.min(qmins)))
    if np.any(qmaxes > 1):
        raise ValueError("qmax must be <= 1; got max(qmax) = %g"
                         % float(np.max(qmaxes)))


def sparse_bls_cpu(t, y, dy, freqs, *, qmin=None, qmax=None,
                   ignore_negative_delta_sols=False,
                   convention='chi2ratio'):
    """
    Sparse BLS implementation for CPU (no binning, tests all pairs of observations).

    This is more efficient than traditional BLS when the number of observations
    is small, as it avoids redundant grid searching over finely-grained parameter
    grids. Based on https://arxiv.org/abs/2103.06193

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies to test
    qmin: float or array_like, optional (default: None)
        Minimum transit duration (in phase) to consider. A scalar
        applies to all frequencies; an array gives a per-frequency
        bound (same length as ``freqs``, e.g. Keplerian
        ``q_transit(freqs) * qmin_fac``). ``None`` means no lower
        bound (all ``q > 0``).
    qmax: float or array_like, optional (default: None)
        Maximum transit duration (in phase), scalar or per-frequency.
        ``None`` means the algorithm's standard upper cutoff of 0.5.
    ignore_negative_delta_sols: bool, optional (default: False)
        Whether or not to ignore solutions with negative delta (inverted dips)
    convention: str, optional (default: 'chi2ratio')
        Power-spectrum convention for the returned powers ('chi2ratio',
        'snr' or 'loglik'); see :func:`convert_bls_power`.

    Returns
    -------
    bls: array_like, float
        BLS power at each frequency
    solutions: list of (q, phi0) tuples
        Best (q, phi0) solution at each frequency
    """
    _validate_convention(convention)
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name='sparse_bls_cpu')
    check_freqs(freqs, name='sparse_bls_cpu')

    # Original flux kept for convert_bls_power's chi2_0
    y_orig, dy_orig = y, dy

    t, epoch = subtract_epoch(t)
    t = t.astype(np.float32)
    # Centre in float64 BEFORE the float32 cast (see
    # _center_flux_float64): the float32 pair scan below otherwise
    # loses 1e-3..1e-2 of the power on mag-scale fluxes.
    y, _ = _center_flux_float64(y, dy)
    dy = np.asarray(dy).astype(np.float32)
    # Keep a float64 copy for the phase re-referencing below: the
    # original-timescale conversion (phi + epoch*freq) % 1 must use the
    # same float64 frequency the caller will use to convert back (e.g.
    # in single_bls); with the float32-cast frequency the phases would
    # be off by epoch * |f64 - f32|, which reaches ~0.07 cycles for
    # BJD-scale epochs (~2.45e6 days).
    freqs64 = np.asarray(freqs, dtype=np.float64)
    freqs = freqs64.astype(np.float32)

    ndata = len(t)
    nfreqs = len(freqs)

    qmins = _broadcast_q_bound(qmin, nfreqs, 0.0, 'qmin')
    qmaxes = _broadcast_q_bound(qmax, nfreqs, 0.5, 'qmax')
    _validate_q_bounds(qmins, qmaxes)

    # Precompute weights (constant across all frequencies)
    w = np.power(dy, -2).astype(np.float32)
    w /= np.sum(w)

    bls_powers = np.zeros(nfreqs, dtype=np.float32)
    best_q = np.zeros(nfreqs, dtype=np.float32)
    best_phi = np.zeros(nfreqs, dtype=np.float32)

    # residual float32 mean of the centred flux (~1e-8); kept so the
    # scan is exactly the kernel's arithmetic
    ybar = float(np.dot(w, y))
    YY = float(np.dot(w, np.power(y - ybar, 2)))

    # Vectorized pair scan. Transit candidates are exactly the
    # contiguous runs of phase-sorted observations (plus wrap-around
    # runs); prefix sums turn each candidate's (W, YW) into two array
    # lookups, so the scan is O(N^2) numpy work with O(N^2)
    # temporaries. The previous pure-Python loop recomputed each
    # slice sum, costing O(N^3) time (minutes per frequency at the
    # ndata=500 sparse threshold).
    i_idx = np.arange(ndata)

    for i_freq, freq in enumerate(freqs):
        qmin_f = qmins[i_freq]
        qmax_f = qmaxes[i_freq]

        # Compute phases and sort
        phi = (t * freq) % 1.0
        order = np.argsort(phi)
        phi_s = phi[order].astype(np.float64)
        y_s = y[order].astype(np.float64)
        w_s = w[order].astype(np.float64)

        # Prefix sums: cw[k] = sum(w_s[:k]), cyw[k] = sum((w*y)_s[:k])
        cw = np.concatenate(([0.0], np.cumsum(w_s)))
        cyw = np.concatenate(([0.0], np.cumsum(w_s * y_s)))

        # mid[j]: upper transit boundary when the last in-transit
        # observation is j-1 (midpoint to the first excluded
        # observation; epsilon past the final phase when nothing
        # is excluded)
        mid = np.empty(ndata + 1)
        mid[0] = 0.0  # unused
        mid[1:ndata] = 0.5 * (phi_s[1:] + phi_s[:-1])
        mid[ndata] = phi_s[ndata - 1] + 1e-7

        # ---- Non-wrapped transits: obs i..j-1, 0 <= i < j <= ndata.
        # Matrices indexed [i, j-1].
        W_nw = cw[None, 1:] - cw[i_idx, None]
        YW_nw = cyw[None, 1:] - cyw[i_idx, None] - ybar * W_nw
        q_nw = mid[None, 1:] - phi_s[:, None]
        valid_nw = np.triu(np.ones((ndata, ndata), dtype=bool))

        # ---- Wrapped transits: obs i..end plus 0..k-1, 0 <= k < i.
        # Matrices indexed [i, k]; the head boundary for k=0 is an
        # epsilon past phase 1 (only the tail is in transit).
        head_q = np.empty(ndata)
        head_q[0] = 1e-7
        head_q[1:] = mid[1:ndata]
        W_w = (cw[ndata] - cw[:ndata, None]) + cw[None, :ndata]
        YW_w = ((cyw[ndata] - cyw[:ndata, None]) + cyw[None, :ndata]
                - ybar * W_w)
        q_w = (1.0 - phi_s[:, None]) + head_q[None, :]
        valid_w = i_idx[None, :] < i_idx[:, None]

        powers = []
        for W, YW, q, valid in ((W_nw, YW_nw, q_nw, valid_nw),
                                (W_w, YW_w, q_w, valid_w)):
            # W bounds mirror sparse_bls.cu's MIN_W/MAX_W_COMPLEMENT:
            # the complement must exceed float32 roundoff so the GPU
            # kernel and this reference exclude the same degenerate
            # all-weight boxes (parity tests compare them directly)
            valid = (valid & (q > 0) & (q >= qmin_f) & (q <= qmax_f)
                     & (W > 1e-9) & (W < 1.0 - 1e-4))
            if ignore_negative_delta_sols:
                valid &= (YW <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.where(valid,
                             (YW * YW) / (W * (1.0 - W)) / YY, 0.0)
            powers.append(p)

        all_powers = np.concatenate([p.ravel() for p in powers])
        imax = int(np.argmax(all_powers))
        if all_powers[imax] > 0:
            n_nw = ndata * ndata
            if imax < n_nw:
                ii, jj = divmod(imax, ndata)
                q_best = q_nw[ii, jj]
            else:
                ii, kk = divmod(imax - n_nw, ndata)
                q_best = q_w[ii, kk]
            bls_powers[i_freq] = all_powers[imax]
            best_q[i_freq] = q_best
            best_phi[i_freq] = phi_s[ii]

    solutions = list(zip(best_q, best_phi))
    # Adjust phases to original timescale (float64 frequencies: the
    # inverse conversion in single_bls uses the caller's float64 freq)
    solutions = [(q, (phi + (epoch * freq)) % 1.0)
                 for (q, phi), freq in zip(solutions, freqs64)]

    return (convert_bls_power(bls_powers, y_orig, dy_orig,
                              convention=convention),
            solutions)


def _sparse_shared_mem_bytes(ndata, block_size):
    """Dynamic shared memory ``sparse_bls_kernel`` needs per block for
    ``ndata`` points: three arrays padded to the next power of two (for
    the bitonic sort), two prefix-sum arrays and three per-thread
    scratch values, all float32."""
    n_pow2 = 1
    while n_pow2 < ndata:
        n_pow2 *= 2
    return (3 * n_pow2 + 2 * int(ndata) + 3 * int(block_size)) * 4


def _sparse_max_ndata(shmem_lim, block_size):
    """Largest ``ndata`` whose :func:`_sparse_shared_mem_bytes` fits in
    ``shmem_lim`` bytes."""
    best = 0
    n_pow2 = 1
    while (3 * n_pow2 + 3 * block_size) * 4 <= shmem_lim:
        n = min(n_pow2, (shmem_lim // 4 - 3 * n_pow2 - 3 * block_size) // 2)
        best = max(best, int(n))
        n_pow2 *= 2
    return best


def _reject_use_simple(kwargs, where):
    """The bubble-sort ``sparse_bls_simple.cu`` kernel was removed in
    1.0 (it still carried the pre-PR#65 ``MAX_W_COMPLEMENT 1E-9`` bound
    and returned powers up to 4.6 in pure noise on single-site data;
    Sep 2026 audit, defect 20). Refuse the old switch loudly instead
    of silently running the full kernel."""
    if 'use_simple' in kwargs:
        raise TypeError("%s: the 'use_simple' sparse kernel was removed in "
                        "cuvarbase 1.0; drop the argument (the bitonic "
                        "sort + prefix-sum kernel is the only sparse "
                        "kernel)" % where)


def compile_sparse_bls(block_size=_default_block_size, **kwargs):
    """
    Compile sparse BLS GPU kernel (bitonic sort + prefix sums for O(1)
    range queries).

    Parameters
    ----------
    block_size: int, optional (default: _default_block_size)
        CUDA threads per CUDA block.

    Returns
    -------
    kernel: PyCUDA function
        The compiled sparse_bls_kernel function
    """
    _reject_use_simple(kwargs, 'compile_sparse_bls')

    # Compiling a kernel needs an active CUDA context (lazily created).
    ensure_context()

    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel('sparse_bls'),
                                cpp_defs=cppd)

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    kernel = module.get_function('sparse_bls_kernel')

    # Don't use prepare() - it causes issues with large shared memory
    return kernel


def sparse_bls_gpu(t, y, dy, freqs, *, qmin=None, qmax=None,
                   ignore_negative_delta_sols=False,
                   block_size=64, max_ndata=None,
                   stream=None, kernel=None,
                   convention='chi2ratio'):
    """
    GPU-accelerated sparse BLS implementation.

    Uses a CUDA kernel to test all pairs of observations as potential
    transit boundaries. More efficient than CPU implementation for datasets
    with ~100-1000 observations.

    Based on https://arxiv.org/abs/2103.06193

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies to test
    qmin: float or array_like, optional (default: None)
        Minimum transit duration (in phase) to consider; scalar or
        per-frequency array (same length as ``freqs``). ``None`` means
        no lower bound (all ``q > 0``).
    qmax: float or array_like, optional (default: None)
        Maximum transit duration (in phase), scalar or per-frequency.
        ``None`` means the algorithm's standard upper cutoff of 0.5.
    ignore_negative_delta_sols: bool, optional (default: False)
        Whether or not to ignore solutions with negative delta (inverted dips)
    block_size: int, optional (default: 64)
        CUDA threads per CUDA block (use 32-128 for best performance)
    max_ndata: int, optional (default: None)
        Maximum number of data points (for shared memory allocation).
        If None, uses len(t)
    stream: pycuda.driver.Stream, optional (default: None)
        CUDA stream for async execution
    kernel: PyCUDA function, optional (default: None)
        Pre-compiled kernel. If None, compiles kernel automatically.
    convention: str, optional (default: 'chi2ratio')
        Power-spectrum convention for the returned powers ('chi2ratio',
        'snr' or 'loglik'); see :func:`convert_bls_power`.

    Returns
    -------
    bls_powers: array_like, float
        BLS power at each frequency
    solutions: list of (q, phi0) tuples
        Best (q, phi0) solution at each frequency
    """
    _validate_convention(convention)
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name='sparse_bls_gpu')
    check_freqs(freqs, name='sparse_bls_gpu')

    # Original flux kept for convert_bls_power's chi2_0
    y_orig, dy_orig = y, dy

    # Convert to numpy arrays (epoch-subtract before the float32 cast)
    t, epoch = subtract_epoch(t)
    t = t.astype(np.float32)
    # Centre in float64 BEFORE the float32 cast: the kernel's float32
    # prefix sums of w*y otherwise carry the mean flux and the
    # `YW -= ybar * W` correction cancels catastrophically (Sep 2026
    # audit, defect 8; the in-kernel ybar is now ~1e-8 and harmless).
    y, _ = _center_flux_float64(y, dy)
    dy = np.asarray(dy).astype(np.float32)
    # float64 copy for the phase re-referencing below (see
    # sparse_bls_cpu: the float32-cast frequency would put the
    # original-timescale phases off by epoch * |f64 - f32|)
    freqs64 = np.asarray(freqs, dtype=np.float64)
    freqs = freqs64.astype(np.float32)

    ndata = len(t)
    nfreqs = len(freqs)

    qmins = _broadcast_q_bound(qmin, nfreqs, 0.0,
                               'qmin').astype(np.float32)
    qmaxes = _broadcast_q_bound(qmax, nfreqs, 0.5,
                                'qmax').astype(np.float32)
    _validate_q_bounds(qmins, qmaxes)

    if max_ndata is None:
        max_ndata = ndata

    # Block size must be a power of 2 for tree reductions
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")

    # Compile kernel if not provided
    if kernel is None:
        kernel = compile_sparse_bls(block_size=block_size)

    # Shared memory per block:
    #   sh_phi[n_pow2] + sh_y[n_pow2] + sh_w[n_pow2]
    #   + sh_cumsum_w[N] + sh_cumsum_yw[N] + 3*blockDim.x
    shared_mem_size = _sparse_shared_mem_bytes(max_ndata, block_size)

    # The kernel keeps the whole light curve in shared memory, so it is
    # limited to ~2000 points on a 48 KB device; the launch used to fail
    # with a bare "cuLaunchKernel failed: invalid argument" (Sep 2026
    # audit, ids 77/126). Check before any allocation or launch.
    att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    shmem_lim = int(ensure_context().device.get_attribute(att))
    if shared_mem_size > shmem_lim:
        raise ValueError(
            "sparse_bls_gpu: %d points need %d bytes of shared memory "
            "per block, above this device's %d-byte limit (the sparse "
            "kernel handles at most %d points here with block_size=%d). "
            "Use the binned kernels for larger light curves: "
            "eebls_transit(use_sparse=False) / eebls_gpu_fast / eebls_gpu."
            % (max_ndata, shared_mem_size, shmem_lim,
               _sparse_max_ndata(shmem_lim, block_size), block_size))

    # Allocate GPU memory
    t_g = gpuarray.to_gpu(t)
    y_g = gpuarray.to_gpu(y)
    dy_g = gpuarray.to_gpu(dy)
    freqs_g = gpuarray.to_gpu(freqs)
    qmin_g = gpuarray.to_gpu(qmins)
    qmax_g = gpuarray.to_gpu(qmaxes)

    bls_powers_g = gpuarray.zeros(nfreqs, dtype=np.float32)
    best_q_g = gpuarray.zeros(nfreqs, dtype=np.float32)
    best_phi_g = gpuarray.zeros(nfreqs, dtype=np.float32)

    # Launch kernel
    # Grid: one block per frequency (or fewer if limited by hardware)
    max_blocks = 65535  # CUDA maximum
    grid = (min(nfreqs, max_blocks), 1)
    block = (block_size, 1, 1)

    if stream is None:
        stream = cuda.Stream()

    # Call kernel without prepare() to avoid resource issues
    kernel(
        t_g, y_g, dy_g, freqs_g, qmin_g, qmax_g,
        np.uint32(ndata), np.uint32(nfreqs),
        np.uint32(ignore_negative_delta_sols),
        bls_powers_g, best_q_g, best_phi_g,
        block=block, grid=grid, stream=stream,
        shared=shared_mem_size
    )

    # Copy results back
    stream.synchronize()
    bls_powers = bls_powers_g.get()
    best_q = best_q_g.get()
    best_phi = best_phi_g.get()

    solutions = list(zip(best_q, best_phi))
    # Adjust phases to original timescale (float64 frequencies: the
    # inverse conversion in single_bls uses the caller's float64 freq)
    solutions = [(q, (phi + (epoch * freq)) % 1.0)
                 for (q, phi), freq in zip(solutions, freqs64)]

    return (convert_bls_power(bls_powers, y_orig, dy_orig,
                              convention=convention),
            solutions)


def _fast_box_widths(nbinsf, nbins0, dlogq):
    """Box widths, in fine phase bins, that the fast (shared-memory)
    kernels iterate at one frequency: ``m = 1, 1 + dnbins(1, dlogq),
    ...`` up to and including ``max_bin_width = nbinsf // nbins0``.

    The searched durations are ``q = m / nbinsf``, so the widest one is
    ``(nbinsf // nbins0) / nbinsf <= 1 / nbins0``, i.e. the discretized
    ``qmax`` (``nbins0 = floor(1/qmax)``). Before 1.0 the kernels wrote
    ``max_bin_width = divrndup(nbinsf, nbins0)`` and looped ``m <
    max_bin_width``: the same set whenever ``nbins0`` does not divide
    ``nbinsf``, but one level short when it does -- ``qmin=0.025,
    qmax=0.1`` searched only ``q <= 0.075`` (Sep 2026 audit, id 64).

    Note the geometric step can still overshoot the last level: with
    ``qmin=0.01, qmax=0.5, dlogq=0.3`` the ladder is ``..., 37, 48``
    and 62 > 50, so ``q = 0.48`` remains the widest box tested.
    """
    nbf = int(nbinsf)
    nb0 = max(1, int(nbins0))
    max_bin_width = nbf // nb0
    widths = []
    m = 1
    while m <= max_bin_width:
        widths.append(m)
        m += dnbins(m, dlogq)
    return widths


def _fast_bls_box_scan(t32, yw32, w32, freq, nbins0, nbinsf, dlogq,
                       noverlap, dphi=0.0,
                       ignore_negative_delta_sols=False):
    """CPU replica of the box grid the fast kernels
    (``full_bls_no_sol`` / ``_optimized`` / ``_fused``) search at ONE
    frequency: fold in float32, histogram into ``nbinsf`` phase bins on
    ``noverlap`` grids shifted by ``1/noverlap`` of a bin (plus the base
    offset ``dphi``), and scan every box of ``m`` bins for ``m = 1,
    1 + dnbins(1), ...`` up to and including ``nbinsf // nbins0`` (the
    widest box with ``q = m / nbinsf <= 1 / nbins0``).

    ``t32`` are the epoch-subtracted float32 times, ``yw32 = w * (y -
    ybar)`` and ``w32`` the normalized weights, all as
    :meth:`BLSMemory.setdata` uploads them (order is irrelevant).

    Returns ``(value, q, phi0)`` with ``value = YW^2 / (W (1 - W))``
    (divide by ``YY`` for the 'chi2ratio' power), ``q = m / nbinsf`` and
    the box start phase ``phi0 = (n + dphi_pass) / nbinsf`` (mod 1)
    relative to the epoch of ``t32``; ``(0, 0, 0)`` when no box passes
    the weight guards.
    """
    nbf = int(nbinsf)
    # q levels, exactly as the kernels iterate them
    ms = _fast_box_widths(nbf, nbins0, dlogq)

    phi = np.asarray(t32, dtype=np.float32) * np.float32(freq)
    phi = phi - np.floor(phi)
    w64 = np.asarray(w32, dtype=np.float64)
    yw64 = np.asarray(yw32, dtype=np.float64)
    n = np.arange(nbf)

    best_val, best_q, best_phi = 0.0, 0.0, 0.0
    for s_pass in range(int(noverlap)):
        dphi_pass = np.float32(float(dphi) + float(s_pass) / noverlap)
        b = np.floor(np.float32(nbf) * phi - dphi_pass)
        b = b.astype(np.int64) % nbf
        hw = np.bincount(b, weights=w64, minlength=nbf)
        hyw = np.bincount(b, weights=yw64, minlength=nbf)
        # circular prefix sums: box (n, m) = bins n .. n + m - 1 mod nbf
        cw = np.concatenate(([0.0], np.cumsum(np.concatenate([hw, hw]))))
        cyw = np.concatenate(([0.0],
                              np.cumsum(np.concatenate([hyw, hyw]))))
        for m in ms:
            W = cw[n + m] - cw[n]
            YW = cyw[n + m] - cyw[n]
            # same guards as bls_value in bls_common.cuh
            ok = (W > 1e-10) & (W < 1.0 - 1e-4)
            if ignore_negative_delta_sols:
                ok &= (YW <= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.where(ok, YW * YW / (W * (1.0 - W)), 0.0)
            k = int(np.argmax(val))
            if val[k] > best_val:
                best_val = float(val[k])
                best_q = m / float(nbf)
                best_phi = ((n[k] + float(dphi_pass)) / float(nbf)) % 1.0
    return best_val, best_q, best_phi


def _fast_bls_solutions(t, y, dy, freqs, powers, qmin, qmax, n_solutions,
                        dlogq=0.3, noverlap=2, dphi=0.0,
                        ignore_negative_delta_sols=False):
    """Best-fit ``(q, phi)`` at the ``n_solutions`` highest-power
    frequencies of a fast-kernel periodogram (``eebls_gpu_fast`` and
    friends do not track solutions).

    Each selected frequency's box grid is re-scanned on the CPU with
    :func:`_fast_bls_box_scan` -- the same q levels, phase-bin grids
    and float32 fold the kernel used -- so the returned ``(q, phi)`` is
    the box that produced ``powers[k]`` (up to float32 accumulation
    order). ``phi`` is the transit start phase in the ORIGINAL input
    timescale (the convention of :func:`eebls_gpu` /
    :func:`single_bls`).

    Returns a list of length ``len(freqs)``: ``(q, phi)`` tuples at the
    selected frequencies (skipping those with zero power) and ``None``
    elsewhere.
    """
    nfreq = len(freqs)
    sols = [None] * nfreq
    n_sel = int(min(max(0, int(n_solutions)), nfreq))
    if n_sel == 0:
        return sols

    powers = np.asarray(powers, dtype=np.float64)
    order = np.argsort(-powers, kind='stable')[:n_sel]

    t64, epoch = subtract_epoch(np.asarray(t, dtype=np.float64))
    y64 = np.asarray(y, dtype=np.float64)
    w = np.power(np.asarray(dy, dtype=np.float64), -2)
    w /= np.sum(w)
    ybar = float(np.einsum('i,i->', w, y64))
    t32 = t64.astype(np.float32)
    w32 = w.astype(np.float32)
    yw32 = ((y64 - ybar) * w).astype(np.float32)

    freqs64 = np.asarray(freqs, dtype=np.float64)
    freqs32 = freqs64.astype(np.float32)
    qmins = _broadcast_q_bound(qmin, nfreq, 1e-2, 'qmin')
    qmaxes = _broadcast_q_bound(qmax, nfreq, 0.5, 'qmax')
    nbins0, nbinsf = _fast_path_nbins(freqs32, qmins, qmaxes)

    for k in order:
        k = int(k)
        if not powers[k] > 0:
            continue
        val, q, phi = _fast_bls_box_scan(
            t32, yw32, w32, freqs32[k], nbins0[k], nbinsf[k], dlogq,
            noverlap, dphi=dphi,
            ignore_negative_delta_sols=ignore_negative_delta_sols)
        if val <= 0:
            continue
        # back to the original timescale (float64 frequency, as eebls_gpu)
        sols[k] = (float(q), float((phi + epoch * freqs64[k]) % 1.0))
    return sols


def eebls_transit(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                  qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                  fmax=None, freqs=None, qvals=None,
                  use_fast=False,  use_optimized=False,
                  use_sparse=None, sparse_threshold=500,
                  use_gpu=True,
                  ignore_negative_delta_sols=False,
                  n_solutions=10,
                  **kwargs):
    """
    Keplerian BLS transit search, automatically selecting the
    implementation from the dataset size.

    For small datasets (``ndata < sparse_threshold``) the sparse BLS
    algorithm (Panahi & Zucker 2021) tests every pair of observations
    as transit boundaries (no binning; chosen for its detection
    properties on sparse data, not for speed). For larger datasets the
    periodogram is computed by the fast shared-memory GPU kernel
    (:func:`eebls_gpu_fast`, fused phase-oversampling) and the best-fit
    ``(q, phi)`` is recovered at the ``n_solutions`` highest peaks.
    Both paths honour the same per-frequency Keplerian duration bounds
    ``[qmin_fac, qmax_fac] * q_transit(f)``.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    fmax_frac: float, optional (default: 1.0)
        Maximum frequency is `fmax_frac * fmax`, where
        `fmax` is automatically selected by `fmax_transit`.
    fmin_frac: float, optional (default: 1.0)
        Minimum frequency is `fmin_frac * fmin`, where
        `fmin` is automatically selected by `fmin_transit`.
    fmin: float, optional (default: None)
        Overrides automatic frequency minimum with this value
    fmax: float, optional (default: None)
        Overrides automatic frequency maximum with this value
    qmin_fac: float, optional (default: 0.5)
        Fraction of the fiducial q value to search
        at each frequency (minimum)
    qmax_fac: float, optional (default: 2.0)
        Fraction of the fiducial q value to search
        at each frequency (maximum)
    freqs: array_like, optional (default: None)
        Overrides the auto-generated frequency grid
    qvals: array_like, optional (default: None)
        Overrides the keplerian q values
    use_fast: bool, optional (default: False)
        Periodogram only: skip the ``(q, phi)`` recovery pass and return
        ``solutions=None``. The periodogram itself is the same
        :func:`eebls_gpu_fast` result the default path returns (kept
        for backward compatibility; before 1.0 the default path ran the
        slower binned :func:`eebls_gpu` search).
    use_optimized: bool, optional (default: False)
        Use the optimized GPU kernel (:func:`eebls_gpu_fast_optimized`;
        periodogram only, ``solutions=None``).

        Unless an explicit ``block_size`` is passed (which is always
        respected), this automatically selects a block size based on
        ndata:

        - ndata <= 32: 32 threads (single warp)
        - ndata <= 64: 64 threads (two warps)
        - ndata <= 128: 128 threads (four warps)
        - ndata > 128: 256 threads (eight warps)

        Smaller blocks reduce idle-thread overhead for small datasets
        (measured effect with a warm kernel cache is ~1.0-1.3x; see
        eebls_gpu_fast_adaptive).
    use_sparse: bool, optional (default: None)
        If True, use sparse BLS. If False, use standard BLS. If None (default),
        automatically select based on dataset size (sparse_threshold).
    sparse_threshold: int, optional (default: 500)
        Threshold for automatically selecting sparse BLS. If ndata < threshold
        and use_sparse is None, sparse BLS is used.
    use_gpu: bool, optional (default: True)
        Use GPU implementation. If True, uses GPU for both sparse and standard BLS.
        If False, uses CPU for sparse BLS. The use_gpu parameter only affects sparse BLS; standard BLS always uses GPU.
    ignore_negative_delta_sols: bool, optional (default: False)
        Whether or not to ignore inverted dips
    n_solutions: int, optional (default: 10)
        Standard (non-sparse) path: number of highest-power frequencies
        at which the best-fit ``(q, phi)`` is recovered (a CPU re-scan
        of the kernel's box grid at those frequencies; see
        :func:`_fast_bls_solutions`). The remaining entries of
        ``solutions`` are ``None``. ``0`` returns a list of ``None``.
        For a solution at every frequency use :func:`eebls_transit_gpu`
        or :func:`eebls_gpu` (the full binned search; much slower).
    **kwargs:
        passed to `eebls_gpu_fast` (``dlogq``, ``noverlap``, ``dphi``,
        ``freq_batch_size``, ``functions``, ``block_size``, ...; the
        fast-kernel defaults ``dlogq=0.3``, ``noverlap=2`` apply),
        `compile_bls`, `fmax_transit`, `fmin_transit`, and
        `transit_autofreq`. The :func:`eebls_gpu`-only kwargs
        ``nstreams`` and ``max_memory`` are ignored (with a
        ``UserWarning``; use :func:`eebls_transit_gpu` or
        :func:`eebls_gpu` if you need them). On the sparse
        path, only the kwargs that `sparse_bls_gpu` accepts
        (``block_size``, ``max_ndata``, ``stream``, ``kernel``,
        ``convention``) are forwarded to it. A ``convention=`` kwarg
        ('chi2ratio', 'snr' or 'loglik'; see
        :func:`convert_bls_power`) selects the power-spectrum
        convention on every path.

        .. note::

            Both paths honour the per-frequency Keplerian
            ``qmin_fac``/``qmax_fac`` duration bounds exactly per
            frequency (the pre-1.0 standard path collapsed them to one
            batch-wide window, so its results depended on
            ``freq_batch_size`` and on the free device memory), so
            results are comparable across the ``sparse_threshold``
            boundary up to the two algorithms' different candidate
            sets (binned box grid vs observation pairs). Pass
            ``use_sparse=False`` to force the standard path.

    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \\chi^2(f) / \\chi^2_0`
    solutions: list of ``(q, phi)`` tuples, or None
        Best ``(q, phi)`` solution per frequency; ``phi`` is the transit
        start phase in the original input timescale. Sparse path: a
        solution at every frequency. Standard path: solutions at the
        ``n_solutions`` highest peaks (always including the argmax),
        ``None`` elsewhere. ``None`` altogether when ``use_fast=True``
        or ``use_optimized=True``.

    """
    # Validate before anything else -- including the Keplerian grid
    # builder, which turns a NaN timestamp into a NaN frequency grid
    # and (with use_fast=True) a device crash that kills the CUDA
    # context (Sep 2026 audit, defect 23). The Keplerian grid needs
    # min_obs_per_transit (default 5) points; fmin_transit raises for
    # shorter light curves, so only the universal floor is applied
    # here (an explicit ``freqs=`` grid does not need the extra
    # points).
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name='eebls_transit')
    if freqs is not None:
        check_freqs(freqs, name='eebls_transit')

    ndata = len(t)
    _reject_use_simple(kwargs, 'eebls_transit')

    # Determine whether to use sparse BLS
    if use_sparse is None:
        use_sparse = ndata < sparse_threshold

    # Generate frequency grid if not provided
    if freqs is None:
        if qvals is not None:
            raise ValueError("qvals must be None if freqs is None")
        if fmin is None:
            fmin = fmin_transit(t, **kwargs) * fmin_frac
        if fmax is None:
            fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
        freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                        qmin_fac=qmin_fac, **kwargs)
    if qvals is None:
        qvals = q_transit(freqs, **kwargs)

    qmins = np.asarray(qvals) * qmin_fac
    qmaxes = np.asarray(qvals) * qmax_fac

    # Use sparse BLS for small datasets
    if use_sparse:
        # The sparse path honors the same per-frequency Keplerian
        # q bounds as the standard path; use_fast only selects
        # between the standard implementations.
        if use_gpu:
            # Forward only the kwargs sparse_bls_gpu accepts; the rest
            # (rho, samples_per_peak, dlogq, ...) belong to the frequency
            # grid helpers or standard-BLS layers above.
            sparse_keys = ('block_size', 'max_ndata', 'stream', 'kernel',
                           'convention')
            sparse_kwargs = {k: v for k, v in kwargs.items()
                             if k in sparse_keys}
            powers, sols = sparse_bls_gpu(t, y, dy, freqs,
                                          qmin=qmins, qmax=qmaxes,
                                          ignore_negative_delta_sols=ignore_negative_delta_sols,
                                          **sparse_kwargs)
        else:
            # Use CPU sparse BLS (fallback)
            powers, sols = sparse_bls_cpu(
                t, y, dy, freqs, qmin=qmins, qmax=qmaxes,
                ignore_negative_delta_sols=ignore_negative_delta_sols,
                convention=kwargs.get('convention', 'chi2ratio'))
        return freqs, powers, sols

    # Standard (binned) GPU path for larger datasets: the periodogram
    # comes from the fast shared-memory kernel, which honours the
    # per-frequency Keplerian bounds (before 1.0 this path ran
    # eebls_gpu, whose kernels collapsed array bounds to one batch-wide
    # window -- Sep 2026 audit defect 7); the best (q, phi) is recovered
    # at the top n_solutions peaks afterwards.
    for key in ('nstreams', 'max_memory'):   # eebls_gpu-only
        if kwargs.pop(key, None) is not None:
            warnings.warn(
                "eebls_transit ignores %s: the default path runs "
                "eebls_gpu_fast, which uses one stream and sizes its "
                "own shared-memory batches. Call eebls_transit_gpu "
                "(Keplerian bounds, solution at every frequency) or "
                "eebls_gpu directly if you need %s." % (key, key),
                UserWarning, stacklevel=2)
    dlogq = kwargs.setdefault('dlogq', 0.3)
    noverlap = kwargs.setdefault('noverlap', 2)
    dphi = kwargs.get('dphi', 0.0)

    if use_optimized:
        # Choose a block size from ndata unless the caller asked for a
        # specific one (an explicit block_size must never be silently
        # overridden -- the compiled BLOCK_SIZE and the launch
        # configuration have to agree with what the caller expects)
        block_size = kwargs.get('block_size')
        if block_size is None:
            block_size = _choose_block_size(ndata)
        kwargs['block_size'] = block_size

        # Get cached kernels for this block size
        fname = 'full_bls_no_sol_optimized'
        functions = _get_cached_kernels(block_size, use_optimized, [fname])

        powers = eebls_gpu_fast_optimized(t, y, dy, freqs,
                                          qmin=qmins, qmax=qmaxes,
                                          ignore_negative_delta_sols=ignore_negative_delta_sols,
                                          functions=functions,
                                          **kwargs)
        return freqs, powers, None

    powers = eebls_gpu_fast(t, y, dy, freqs,
                            qmin=qmins, qmax=qmaxes,
                            ignore_negative_delta_sols=ignore_negative_delta_sols,
                            **kwargs)
    if use_fast:
        return freqs, powers, None

    sols = _fast_bls_solutions(
        t, y, dy, freqs, powers, qmins, qmaxes, n_solutions,
        dlogq=dlogq, noverlap=noverlap, dphi=dphi,
        ignore_negative_delta_sols=ignore_negative_delta_sols)
    return freqs, powers, sols


_batch_function_signature = {
    'full_bls_batch': [
        np.intp, np.intp, np.intp,       # t_all, yw_all, w_all
        np.intp, np.intp,                 # bls_all, freqs
        np.intp, np.intp,                 # nbins0, nbinsf
        np.intp,                          # ndata_per_lc
        np.uint32, np.uint32, np.uint32,  # max_ndata, nfreq, freq_offset
        np.uint32, np.uint32,             # hist_size, noverlap
        np.float32, np.float32,           # dlogq, dphi
        np.uint32, np.uint32,             # ignore_neg, n_lcs
        np.uint32,                        # bls_stride (output row pitch)
    ],
    # fused-noverlap variant: identical argument list; hist_size is the
    # FINE histogram size (noverlap * max_nbins)
    'full_bls_batch_fused': [
        np.intp, np.intp, np.intp,
        np.intp, np.intp,
        np.intp, np.intp,
        np.intp,
        np.uint32, np.uint32, np.uint32,
        np.uint32, np.uint32,
        np.float32, np.float32,
        np.uint32, np.uint32,
        np.uint32,
    ],
}


def compile_bls_batch(block_size=_default_block_size, **kwargs):
    """
    Compile the multi-LC batch BLS kernel.

    Parameters
    ----------
    block_size : int, optional (default: _default_block_size)
        CUDA threads per block.

    Returns
    -------
    functions : dict
        Dictionary of compiled kernel functions.
    """
    _validate_block_size(block_size)

    # Compiling a kernel needs an active CUDA context (lazily created).
    ensure_context()

    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel('bls_batch'), cpp_defs=cppd)
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    functions = {}
    for name, sig in _batch_function_signature.items():
        func = module.get_function(name)
        functions[name] = func.prepare(sig)

    return functions


def _get_cached_batch_kernels(block_size):
    """``compile_bls_batch`` through the same thread-safe LRU cache the
    single-LC paths use. Without this every ``eebls_gpu_batch`` call
    recompiled the kernel (~0.6-0.9 s on an A5000) -- which dwarfed the
    2-10 ms of actual kernel work and was the entire "batch is ~12x
    slower at TESS scale" regression (E1)."""
    ensure_context()
    key = (block_size, 'batch')
    with _kernel_cache_lock:
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]
        compiled = compile_bls_batch(block_size=block_size)
        _kernel_cache[key] = compiled
        _kernel_cache.move_to_end(key)
        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)
        return compiled


def eebls_gpu_batch(lightcurves, freqs, qmin=1e-2, qmax=0.5,
                    noverlap=2, dlogq=0.3, dphi=0.0,
                    ignore_negative_delta_sols=False,
                    max_batch_lcs=256, block_size=None,
                    functions=None, convention='chi2ratio',
                    memory=None, freq_batch_size=None, **kwargs):
    """
    Process multiple lightcurves in batched GPU operations.

    Launches a single kernel with grid=(nfreq_blocks, n_lcs), where each
    CUDA block handles one (frequency, lightcurve) pair. This eliminates
    per-lightcurve Python loop overhead and kernel launch costs.

    Parameters
    ----------
    lightcurves : list of (t, y, dy) tuples
        List of lightcurves to process.
    freqs : array_like
        Frequency grid (shared across all lightcurves).
    qmin : float or array_like, optional (default: 1e-2)
        Minimum fractional transit duration. An array gives a
        per-frequency bound (e.g. from
        ``bls_frequencies.keplerian_freq_grid(..., return_qvals=True)``
        scaled by a qmin factor); must have the same length as
        ``freqs``.
    qmax : float or array_like, optional (default: 0.5)
        Maximum fractional transit duration (scalar or per-frequency,
        as for ``qmin``).
    noverlap : int, optional (default: 2)
        Phase-bin oversampling: the periodogram is the elementwise max
        over ``noverlap`` kernel passes with the phase-bin grid shifted
        by ``1/noverlap`` of the finest bin between passes (same
        semantics as ``eebls_gpu_fast``). Runtime scales linearly;
        ``noverlap=1`` gives a single unshifted pass. Must be a
        positive integer (``noverlap=0`` used to return an all-zero
        periodogram instead of raising).
    dlogq : float, optional (default: 0.3)
        Logarithmic spacing of q values.
    dphi : float, optional (default: 0.0)
        Phase offset (in units of the finest phase bin).
    ignore_negative_delta_sols : bool, optional (default: False)
        Ignore solutions with positive residuals (inverted dips).
    max_batch_lcs : int, optional (default: 256)
        Maximum lightcurves per kernel launch.
    block_size : int, optional
        CUDA threads per block. If None, auto-selects based on max ndata.
    functions : dict, optional
        Pre-compiled batch kernel functions.
    memory : :class:`cuvarbase.memory.bls_memory.BLSBatchMemory`, optional
        Reusable staging/device memory. Streaming many chunks of
        lightcurves through repeated ``eebls_gpu_batch`` calls pays
        several ms of pinned-host + device allocation per call
        otherwise; construct one ``BLSBatchMemory(max_ndata,
        min(max_batch_lcs, n_lcs), nfreq, stream=Stream())`` sized for
        the largest chunk and pass it to every call. Must satisfy
        ``max_ndata >= max(len(t))``, ``n_lcs >= min(max_batch_lcs,
        len(lightcurves))`` and ``nfreqs >= len(freqs)``.
    freq_batch_size : int, optional
        Frequencies per kernel launch. ``None`` (default) launches the
        whole grid at once unless shared memory would limit occupancy
        (large bin counts from small Keplerian ``qmin``), in which
        case an occupancy-aware chunk size is used automatically.

    Returns
    -------
    bls_results : list of ndarray
        BLS power array for each lightcurve, each shape (nfreq,).

    Notes
    -----
    What batching buys is the removal of per-call host overhead
    (pinned-host and device allocation, transfers, launches): the
    kernel throughput per light curve is the same as the single-LC
    fused kernel once one light curve fills the GPU (Sep 2026 audit,
    id 139: 0.16-0.20 ms/LC batched vs 0.20 ms single at ZTF/TESS
    scale, 8.5-9.3 vs 8.3-8.9 ms/LC at HAT scale). The ~5-10x measured
    against a naive per-call ``eebls_gpu_fast`` loop (RTX A5000, Jul
    2026; fresh ``BLSMemory`` per call) is that overhead; against a
    single-LC loop that reuses its ``BLSMemory`` the whole-call cost
    per light curve is about the same (~0.4 ms/LC either way at ZTF
    scale). Pass ``memory=`` to keep the batch path itself from
    re-allocating per chunk. The earlier "~12x slower at TESS scale"
    regression was per-call kernel compilation (now LRU-cached like
    the single-LC paths); see
    ``analysis/v1.0-gpu-batch3-jul2026/E1_E2_DIAGNOSIS.md``.
    """
    _validate_convention(convention)
    # Validate every light curve, the shared grid and the q bounds
    # before any device work: one NaN sample used to give a finite
    # periodogram with a wrong argmax, and a NaN or out-of-range q
    # bound crashed the kernel and killed the process's CUDA context
    # (Sep 2026 audit, defect 23).
    check_freqs(freqs, name='eebls_gpu_batch')
    for i, lc in enumerate(lightcurves):
        if len(lc) != 3:
            raise ValueError("eebls_gpu_batch: lightcurve %d must be a "
                             "(t, y, dy) tuple; got %d elements"
                             % (i, len(lc)))
        check_lightcurve(lc[0], lc[1], lc[2], min_n=_BLS_MIN_NDATA,
                         name='eebls_gpu_batch lightcurve %d' % i)
    _validate_fast_q_bounds(len(freqs), qmin, qmax)

    freqs = np.asarray(freqs).astype(np.float32)
    nfreq = len(freqs)
    n_total = len(lightcurves)
    # noverlap=0 used to launch nothing and return the untouched (zero,
    # or stale on memory reuse) periodogram (Sep 2026 audit, id 75)
    _validate_noverlap(noverlap)

    # Group LCs by similar ndata to minimize padding
    lc_indices = list(range(n_total))
    lc_ndatas = [len(lc[0]) for lc in lightcurves]

    # Sort by ndata for efficient grouping
    sorted_indices = sorted(lc_indices, key=lambda i: lc_ndatas[i])

    # Auto-select block size
    max_ndata_all = max(lc_ndatas)
    if block_size is None:
        block_size = _choose_block_size(max_ndata_all)

    # Compile kernel if needed (LRU-cached; per-call compilation was
    # the dominant cost of this function -- see _get_cached_batch_kernels)
    if functions is None:
        functions = _get_cached_batch_kernels(block_size)

    func = functions['full_bls_batch']

    # Fused-noverlap path (mirrors _eebls_gpu_fast_impl): one launch
    # with a noverlap-times finer histogram replaces the dphi-shifted
    # multi-pass loop for power-of-two noverlap with dphi == 0.
    fused_func = functions.get('full_bls_batch_fused')
    noverlap_int = int(noverlap)
    use_fused = (fused_func is not None
                 and noverlap_int >= 2
                 and float(dphi) == 0.0
                 and (noverlap_int & (noverlap_int - 1)) == 0)

    # Process in batches
    all_results = [None] * n_total  # indexed by original order

    shmem_lim = kwargs.get('shmem_lim', None)
    if shmem_lim is None:
        dev = ensure_context().device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = dev.get_attribute(att)

    float_size = np.float32(1).nbytes

    # One BLSBatchMemory serves every chunk (and, via ``memory=``,
    # every future call with compatible sizes): allocating the pinned
    # staging buffers + device arrays per chunk cost multiple ms per
    # call (cuMemHostAlloc dominates at survey nfreq).
    batch_cap = min(max_batch_lcs, n_total)
    if memory is not None:
        mem = memory
        if (mem.max_ndata < max_ndata_all or mem.n_lcs < batch_cap
                or mem.nfreqs < nfreq):
            raise ValueError(
                "eebls_gpu_batch: provided memory is too small "
                f"(max_ndata {mem.max_ndata} < {max_ndata_all}, "
                f"n_lcs {mem.n_lcs} < {batch_cap}, or nfreqs "
                f"{mem.nfreqs} < {nfreq})")
        stream = mem.stream
    else:
        stream = cuda.Stream()
        mem = BLSBatchMemory(max_ndata_all, batch_cap, nfreq,
                             stream=stream)

    # Set frequency grid once for all chunks
    max_nbins = mem.set_freqs(freqs, qmin=qmin, qmax=qmax)

    # Check shared memory (qmin may be a per-frequency array)
    mem_req = (block_size + 2 * max_nbins) * float_size
    if mem_req > shmem_lim:
        qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)
        raise ValueError(
            f"qmin={float(np.min(qmin)):.2e} requires too much "
            f"shared memory ({mem_req} > {shmem_lim}). "
            f"Try qmin > {qmin_min:.2e}."
        )

    # Fused path needs the noverlap-times finer histogram to fit;
    # otherwise fall back to the multi-pass loop.
    batch_use_fused = use_fused
    if batch_use_fused:
        fused_req = (block_size
                     + 2 * noverlap_int * max_nbins) * float_size
        if fused_req > shmem_lim:
            batch_use_fused = False
        else:
            mem_req = fused_req

    # Occupancy-aware frequency chunking: each launch sizes its shared
    # memory by the max bin count of the frequencies it covers, and
    # ascending frequency grids have monotonically decreasing bin
    # counts -- so when the global max would cap resident blocks below
    # the thread limit (Kepler-scale qmin), chunked launches let all
    # but the first chunks run at full occupancy (measured +32% on
    # Kepler, neutral elsewhere; only triggers when shared memory is
    # the occupancy limiter).
    if freq_batch_size is None:
        freq_batch_size = nfreq
        if _shmem_limits_occupancy(mem_req, block_size):
            freq_batch_size = _OCCUPANCY_FREQ_CHUNK

    freqs_uploaded = False
    i = 0
    while i < len(sorted_indices):
        # Take up to max_batch_lcs from sorted order
        batch_end = min(i + max_batch_lcs, len(sorted_indices))
        batch_indices = sorted_indices[i:batch_end]
        batch_n = len(batch_indices)

        # Set lightcurve data
        for j, orig_idx in enumerate(batch_indices):
            t, y, dy = lightcurves[orig_idx]
            mem.set_lightcurve(j, t, y, dy)

        # Transfer to GPU (frequency grid only once; only the
        # populated LC slots)
        mem.transfer_to_gpu(n_lcs_active=batch_n,
                            transfer_freqs=not freqs_uploaded)
        freqs_uploaded = True

        # Launch kernel(s)
        block = (block_size, 1, 1)

        # Phase oversampling, mirroring _eebls_gpu_fast_impl (A2).
        # Fused path: a single launch of full_bls_batch_fused evaluates
        # all noverlap bin grids from one finer histogram. Fallback
        # (non-power-of-two noverlap, dphi != 0, or fused histogram over
        # the shared-memory limit): run ``noverlap`` passes with the bin
        # grid shifted by 1/noverlap of a fine bin and keep the
        # elementwise max. Without multi-passing the batch path was
        # single-pass while the fast/adaptive reference multi-passes --
        # the small-ndata periodogram divergence flagged in the Jun GPU
        # batch (E1). Frequency chunks size their shared memory by the
        # chunk's own max bin count (occupancy; see freq_batch_size
        # above).
        best_bls_g = None
        n_passes = 1 if batch_use_fused else noverlap
        for i_pass in range(n_passes):
            dphi_pass = dphi + float(i_pass) / noverlap

            i_freq = 0
            while i_freq < nfreq:
                j_freq = min(i_freq + freq_batch_size, nfreq)
                nf_chunk = j_freq - i_freq
                chunk_nbins = int(np.max(mem.nbinsf[i_freq:j_freq]))

                if batch_use_fused:
                    hist_size = noverlap_int * chunk_nbins
                else:
                    hist_size = chunk_nbins
                chunk_req = (block_size + 2 * hist_size) * float_size

                grid = (min(nf_chunk, 5000), batch_n)
                args = (grid, block, stream)
                args += (mem.t_g.ptr, mem.yw_g.ptr, mem.w_g.ptr)
                args += (mem.bls_g.ptr, mem.freqs_g.ptr)
                args += (mem.nbins0_g.ptr, mem.nbinsf_g.ptr)
                args += (mem.ndata_per_lc_g.ptr,)
                # per-LC stride of the padded data layout = the
                # memory's allocation stride (constant across chunks
                # on reuse)
                args += (np.uint32(mem.max_ndata),)
                args += (np.uint32(nf_chunk), np.uint32(i_freq))
                if batch_use_fused:
                    args += (np.uint32(hist_size),
                             np.uint32(noverlap_int))
                    args += (np.float32(dlogq), np.float32(dphi))
                else:
                    args += (np.uint32(hist_size), np.uint32(1))
                    args += (np.float32(dlogq), np.float32(dphi_pass))
                args += (np.uint32(int(ignore_negative_delta_sols)),)
                args += (np.uint32(batch_n),)
                # output row pitch = the memory's frequency allocation
                # (may exceed len(freqs) on reuse)
                args += (np.uint32(mem.nfreqs),)

                launch_func = fused_func if batch_use_fused else func
                launch_func.prepared_async_call(*args,
                                                shared_size=int(chunk_req))
                i_freq = j_freq

            if not batch_use_fused and noverlap > 1:
                if best_bls_g is None:
                    best_bls_g = mem.bls_g.copy()
                else:
                    gpuarray.maximum(mem.bls_g, best_bls_g,
                                     out=best_bls_g, stream=stream)

        if best_bls_g is not None:
            cuda.memcpy_dtod(mem.bls_g.gpudata, best_bls_g.gpudata,
                             best_bls_g.nbytes)

        # Transfer results back (only the populated rows)
        mem.transfer_to_cpu(n_lcs_active=batch_n)
        batch_results = mem.get_results(n_lcs_active=batch_n,
                                        nfreq_active=nfreq)

        # Store results in original order
        for j, orig_idx in enumerate(batch_indices):
            _, y_j, dy_j = lightcurves[orig_idx]
            all_results[orig_idx] = convert_bls_power(
                batch_results[j], y_j, dy_j, convention=convention)

        i = batch_end

    return all_results


def hone_solution(t, y, dy, f0, df0, q0, dlogq0, phi0, stop=1e-5,
                  samples_per_peak=5, max_iter=50, noverlap=3, **kwargs):
    """
    Experimental!
    """
    p0 = single_bls(t, y, dy, f0, q0, phi0)
    pn = None

    df = df0
    dlogq = dlogq0
    q = q0
    phi = phi0
    f = f0
    nol = noverlap

    baseline = np.max(t) - np.min(t)

    functions = compile_bls(**kwargs)
    i = 0
    while pn is None or i < 5 or ((pn - p0) / p0 > stop and i < max_iter):

        if pn is not None:
            p0 = pn

        fmin, fmax = f - 25 * df, f + 25 * df
        qmin = q / (1 + 5 * dlogq)
        qmax = q * (1 + 5 * dlogq)
        df *= 0.1
        dlogq *= 0.25

        nq = int(np.ceil(np.log(qmax/qmin)/dlogq))
        q_values = np.logspace(np.log(qmin), np.log(qmax), num=nq, base=np.e)
        dphi = 2 * ((fmax - fmin) * baseline * samples_per_peak + dlogq * q)

        phimin = 0.
        phimax = 1.
        if dphi < 0.25:
            phimin = phi - dphi
            phimax = phi + dphi

        nphi = max([10, int(np.ceil(10 * min([2 * dphi, 1.]) / qmin))])
        phi_values = np.linspace(phimin, phimax, nphi)
        nf = int((fmax - fmin)/df)
        freqs = np.linspace(fmin, fmax + df, nf)

        powers, sols = eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                                        freq_batch_size=5, nstreams=5,
                                        functions=functions, **kwargs)

        ibest = np.argmax(powers)
        f = freqs[ibest]
        q, phi = sols[ibest]
        pn = powers[ibest]
        i += 1
    return f, pn, i, (q, phi)


def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                      qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                      fmax=None, freqs=None, qvals=None,
                      use_fast=False, use_optimized=False,
                      ignore_negative_delta_sols=False,
                      **kwargs):
    """
    Compute BLS for timeseries assuming edge-on keplerian
    orbit of a planet with Mp/Ms << 1, Rp/Rs < 1, Lp/Ls << 1 and
    negligible eccentricity.

    .. warning::

        BLS weights each observation by ``1/dy**2`` (normalized). A
        point with a near-zero reported uncertainty concentrates
        essentially all of the statistical weight in one phase bin and
        deterministically produces spurious power of ~0.99 in pure
        noise, at nearly every trial frequency. Symptoms:
        ``max(dy**-2) / sum(dy**-2)`` close to 1, and suspiciously
        high, nearly flat power on noise-like data. Guard with a
        percentile-based error floor before calling::

            dy_floor = np.percentile(dy, 10)
            dy = np.clip(dy, dy_floor, None)

        See the "Data hygiene: near-zero uncertainties" section of the
        BLS documentation for details.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    fmax_frac: float, optional (default: 1.0)
        Maximum frequency is `fmax_frac * fmax`, where
        `fmax` is automatically selected by `fmax_transit`.
    fmin_frac: float, optional (default: 1.0)
        Minimum frequency is `fmin_frac * fmin`, where
        `fmin` is automatically selected by `fmin_transit`.
    fmin: float, optional (default: None)
        Overrides automatic frequency minimum with this value
    fmax: float, optional (default: None)
        Overrides automatic frequency maximum with this value
    qmin_fac: float, optional (default: 0.5)
        Fraction of the fiducial q value to search
        at each frequency (minimum)
    qmax_fac: float, optional (default: 2.0)
        Fraction of the fiducial q value to search
        at each frequency (maximum)
    freqs: array_like, optional (default: None)
        Overrides the auto-generated frequency grid
    qvals: array_like, optional (default: None)
        Overrides the keplerian q values
    functions: tuple, optional (default=None)
        result of ``compile_bls(**kwargs)``.
    use_fast: bool, optional (default: False)
        Use fast GPU implementation.
    use_optimized: bool, optional (default: False)
        Use optimized GPU implementation (if not using fast).

    ignore_negative_delta_sols: bool
        Whether or not to ignore inverted dips
    **kwargs:
        passed to `eebls_gpu`, `compile_bls`, `fmax_transit`,
        `fmin_transit`, and `transit_autofreq`


    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \\chi^2(f) / \\chi^2_0`
    solutions: list of ``(q, phi)`` tuples, or None
        Best ``(q, phi)`` solution at each frequency; ``phi`` is in the
        original input timescale. ``None`` when ``use_fast=True`` or
        ``use_optimized=True`` (those kernels do not track solutions).
        The return is always a 3-tuple, matching :func:`eebls_transit`.

    """
    # See eebls_transit: validate before the Keplerian grid builder.
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA,
                     name='eebls_transit_gpu')
    if freqs is not None:
        check_freqs(freqs, name='eebls_transit_gpu')

    if freqs is None:
        if qvals is not None:
            raise ValueError("qvals must be None if freqs is None")
        if fmin is None:
            fmin = fmin_transit(t, **kwargs) * fmin_frac
        if fmax is None:
            fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
        freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                        qmin_fac=qmin_fac, **kwargs)
    if qvals is None:
        qvals = q_transit(freqs, **kwargs)

    qmins = qvals * qmin_fac
    qmaxes = qvals * qmax_fac

    if use_fast:
        powers = eebls_gpu_fast(t, y, dy, freqs,
                                qmin=qmins, qmax=qmaxes,
                                ignore_negative_delta_sols=ignore_negative_delta_sols,
                                **kwargs)

        return freqs, powers, None
    elif use_optimized:
        powers = eebls_gpu_fast_optimized(t, y, dy, freqs,
                                          qmin=qmins, qmax=qmaxes,
                                          ignore_negative_delta_sols=ignore_negative_delta_sols,
                                          **kwargs)

        return freqs, powers, None

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
    return freqs, powers, sols

