"""
Implementation of the box-least squares periodogram [K2002]_
and variants.

The Keplerian transit-search helpers (:func:`q_transit`,
:func:`freq_transit`, :func:`transit_autofreq`, :func:`eebls_transit`)
assume the transiting body orbits at the host star's mean density. That
assumption fixes the transit-duration/period relation [SM03]_ and, with
it, the optimal frequency-grid spacing for a transit search [O2014]_.

.. [K2002] `Kovacs et al. 2002, A&A 391, 369 <https://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_
.. [SM03] `Seager & Mallen-Ornelas 2003, ApJ 585, 1038 <https://ui.adsabs.harvard.edu/abs/2003ApJ...585.1038S>`_, "A Unique Solution of Planet and Star Parameters from an Extrasolar Planet Transit Light Curve" (eq. 3-4)
.. [O2014] `Ofir 2014, A&A 561, A138 <https://ui.adsabs.harvard.edu/abs/2014A%26A...561A.138O>`_, "Optimizing the search for transiting planets in long time series" (arXiv:1307.7330; corrigendum A&A 597, C2)

"""
import threading
import warnings
from collections import OrderedDict

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import ensure_context
from .utils import find_kernel, _module_reader, subtract_epoch
from .memory.bls_memory import BLSBatchMemory
from .memory._host import host_array

import numpy as np

_default_block_size = 256
_all_function_names = ['full_bls_no_sol',
                       'full_bls_no_sol_optimized',
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
    'bin_and_phase_fold_custom': [np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.float64,
                                  np.uint32, np.uint32, np.uint32,
                                  np.uint32, np.uint32],
    'reduction_max': [np.intp, np.intp, np.uint32, np.uint32, np.uint32,
                      np.intp, np.intp, np.uint32, np.uint32],
    'store_best_sols': [np.intp, np.intp, np.intp, np.uint32,
                        np.uint32, np.uint32, np.float32, np.uint32,
                        np.uint32],
    'store_best_sols_custom': [np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.uint32,
                               np.uint32, np.uint32, np.uint32],
    'bin_and_phase_fold_bst_multifreq':
        [np.intp, np.intp, np.intp, np.intp,
         np.intp, np.intp, np.uint32, np.uint32,
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

        if freqs is not None:
            self.freqs = np.asarray(freqs).astype(self.rtype)
            self.nbinsf = (np.ones_like(self.freqs)/qmin).astype(np.uint32)
            self.nbins0 = (np.ones_like(self.freqs)/qmax).astype(np.uint32)

        # Epoch-subtract in float64 before the float32 cast: absolute
        # timestamps (e.g. BJD) would otherwise destroy the phase fold.
        t, self.epoch = subtract_epoch(t)
        self.t[:len(t)] = t.astype(self.rtype)[:]

        w = np.power(dy, -2)
        w /= np.sum(w)
        self.w[:len(t)] = np.asarray(w).astype(self.rtype)[:]

        self.ybar = np.sum(y * w)
        self.yy = np.dot(w, np.power(y - self.ybar, 2))
        # chi2 of the constant model for the data actually loaded here;
        # convert_bls_power scalings must use this rather than whatever
        # y/dy a later (memory-reuse) call happens to pass.
        self.chi2_0 = _chi2_null(y, dy)

        u = (y - self.ybar) * w
        self.yw[:len(t)] = np.asarray(u).astype(self.rtype)[:]

        if any([x is None for x in [self.t_g, self.yw_g, self.w_g]]):
            self.allocate_data()

        if self.freqs_g is None:
            if nf is None:
                nf = len(freqs)
            self.allocate_freqs(nfreqs=nf)

        if transfer:
            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))

        return self

    @classmethod
    def fromdata(cls, t, y, dy, qmin=None, qmax=None,
                 freqs=None, nf=None, transfer=True,
                 **kwargs):

        max_ndata = kwargs.get('max_ndata', len(t))
        max_nfreqs = kwargs.get('max_nfreqs', nf if freqs is None
                                else len(freqs))
        c = cls(max_ndata, max_nfreqs, **kwargs)

        return c.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                         freqs=freqs, nf=nf, transfer=transfer,
                         **kwargs)


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
        if kwargs.get('prepare', True):
            functions = _get_cached_kernels(
                kwargs.get('block_size', _default_block_size),
                use_optimized, [fname])
        else:
            ckw = dict(kwargs)
            ckw.setdefault('use_optimized', use_optimized)
            functions = compile_bls(function_names=[fname], **ckw)

    func = functions[fname]

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

    if freq_batch_size is None:
        freq_batch_size = len(freqs)

    block = (block_size, 1, 1)

    # minimum q value that we can handle with the shared memory limit
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)

    # Phase oversampling: the kernel's box start positions step one
    # fine phase bin, so a single pass undersamples boxes whose width
    # is near the finest bin. Run ``noverlap`` passes with the bin
    # grid shifted by 1/noverlap of a bin each time and keep the
    # elementwise max -- equivalent to the manual dphi re-run
    # procedure this replaces.
    best_bls_g = None
    for i_pass in range(noverlap):
        dphi_pass = dphi + float(i_pass) / noverlap

        i_freq = 0
        while (i_freq < len(freqs)):
            j_freq = min([i_freq + freq_batch_size, len(freqs)])
            nfreqs = j_freq - i_freq

            max_nbins = max(memory.nbinsf[i_freq:j_freq])

            mem_req = (block_size + 2 * max_nbins) * float_size

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
            # The kernel's own noverlap argument is a no-op in the
            # compiled (linear bin spacing) branch; phase oversampling
            # is implemented by the dphi-shifted passes above.
            args += (np.uint32(max_nbins), np.uint32(1))
            args += (np.float32(dlogq), np.float32(dphi_pass))
            args += (np.uint32(ignore_negative_delta_sols),)

            if stream is not None:
                func.prepared_async_call(*args, shared_size=int(mem_req))
            else:
                func.prepared_call(*args, shared_size=int(mem_req))

            i_freq = j_freq

        if noverlap > 1:
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
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
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
    max_nblocks: int, optional (default: 200)
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
        :math:`1 - \chi_2(\omega) / \chi_2(constant)`

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
    Optimized version of eebls_gpu_fast with improved CUDA kernel.

    This uses an optimized kernel with:
    - Fixed bank conflicts (separate yw/w arrays)
    - Fast math intrinsics (floorf)
    - Warp shuffle reduction (eliminates 4 __syncthreads calls)

    Expected speedup: 20-30% over standard version

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
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
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
        :math:`1 - \chi_2(\omega) / \chi_2(constant)`

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
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
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
        Set of phi values to search at each trial frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically by default based on ``max_memory``
    max_memory: float, optional (default: None)
        Maximum memory to use in bytes. Will ignore this if
        ``freq_batch_size`` is specified. If ``None``, will use the
        free memory given by ``pycuda.driver.mem_get_info()``
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

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    block_size = kwargs.get('block_size', _default_block_size)
    ndata = len(t)

    # read max_memory as total free memory available from driver
    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(0.9 * free)

    if freq_batch_size is None:
        # compute memory
        real_type_size = 4

        # data
        mem0 = ndata * 3 * real_type_size

        nq = len(q_values)
        nphi = len(phi_values)

        # q_values and phi_values
        mem0 += nq + nphi

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += len(freqs) * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nq * nphi * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size == 0:
            raise RuntimeError("Not enough memory (freq_batch_size = 0)")

    nbtot = len(q_values) * len(phi_values) * freq_batch_size

    grid_size = int(np.ceil(float(nbtot) / block_size))

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

    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nstreams):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(nbtot, dtype=np.uint32))

    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.uint32)

    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)

    q_values_g = gpuarray.to_gpu(np.asarray(q_values).astype(np.float32))
    phi_values_g = gpuarray.to_gpu(np.asarray(phi_values).astype(np.float32))

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / freq_batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_custom']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols_custom']

    for batch in range(nbatches):
        imin = freq_batch_size * batch
        imax = min([len(freqs), freq_batch_size * (batch + 1)])

        nf = imax - imin
        j = batch % nstreams
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


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              ignore_negative_delta_sols=False,
              nstreams=5, noverlap=3, dlogq=0.2, max_memory=None,
              freq_batch_size=None, functions=None,
              convention='chi2ratio', **kwargs):

    """
    Box-Least Squares, accelerated with PyCUDA

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
        Minimum q value(s) to test for each frequency
    qmax: float or array_like
        Maximum q value(s) to test for each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 3)
        Number of overlapping q bins to use
    dlogq: float, optional, (default: 0.5)
        logarithmic spacing of :math:`q` values, where :math:`d\log q = dq / q`
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically based on ``max_memory``
    max_memory: float, optional (default: None)
        Maximum memory to use in bytes. Will ignore this if
        ``freq_batch_size`` is specified, and will use the total free memory
        as returned by ``pycuda.driver.mem_get_info`` if this is ``None``.
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
        :math:`1 - \chi^2(f) / \chi^2_0`
    qphi_sols: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

    """

    def locext(ext, arr, imin=None, imax=None):
        if isinstance(arr, float) or isinstance(arr, int):
            return arr
        return ext(arr[slice(imin, imax)])

    _validate_convention(convention)

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(0.9 * free)

    # smallest and largest number of bins
    nbins0_max = 1
    nbinsf_max = 1
    block_size = kwargs.get('block_size', _default_block_size)

    max_q_vals = locext(max, qmax)
    min_q_vals = locext(min, qmin)

    nbins0_max = int(np.floor(1./max_q_vals))
    nbinsf_max = int(np.ceil(1./min_q_vals))

    ndata = len(t)

    nbins_tot_max = count_tot_nbins(nbins0_max, nbinsf_max, dlogq)

    if freq_batch_size is None:
        # compute memory
        real_type_size = np.float32(1).nbytes

        # data
        mem0 = ndata * 3 * real_type_size

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += len(freqs) * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nbins_tot_max * noverlap * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size == 0:
            raise RuntimeError("Not enough memory (freq_batch_size = 0)")

    gs = freq_batch_size * nbins_tot_max * noverlap

    grid_size = int(np.ceil(float(gs) / block_size))

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

    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nstreams):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(gs, dtype=np.int32))

    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.int32)

    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / freq_batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_bst_multifreq']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols']

    for batch in range(nbatches):

        imin = freq_batch_size * batch
        imax = min([len(freqs), freq_batch_size * (batch + 1)])

        minq = locext(min, qmin, imin, imax)
        maxq = locext(max, qmax, imin, imax)

        nbins0 = int(np.floor(1./maxq))
        nbinsf = int(np.ceil(1./minq))

        nbins_tot = count_tot_nbins(nbins0, nbinsf, dlogq)

        nf = imax - imin
        j = batch % nstreams
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]
        # stream.synchronize()

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)
        bls_tmp_g.fill(np.float32(0), stream=stream)
        bls_tmp_sol_g.fill(np.int32(0), stream=stream)

        bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (np.int32(ndata), np.int32(nf))
        args += (np.int32(nbins0), np.int32(nbinsf))
        args += (np.int32(freq_batch_size * batch), np.int32(noverlap))
        args += (np.float32(dlogq), np.int32(nbins_tot))
        bin_func.prepared_async_call(*args)

        all_bins = nf * nbins_tot * noverlap

        bls_grid = (int(np.ceil(float(all_bins) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(all_bins))
        args += (np.uint32(ignore_negative_delta_sols),)
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * freq_batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (np.uint32(nbins0), np.uint32(nbinsf), np.uint32(noverlap))
        args += (np.float32(dlogq), np.uint32(nf))
        args += (np.uint32(batch * freq_batch_size),)
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
        Phase offset of transit, relative to ``floor(min(t))`` (times are
        epoch-subtracted before folding, consistent with the GPU
        functions in this module)
    ignore_negative_delta_sols:
        Whether or not to ignore solutions with negative delta (inverted dips)

    Returns
    -------
    bls: float
        BLS power for this set of parameters
    """

    phi = subtract_epoch(t)[0].astype(np.float32) * np.float32(freq)
    phi -= np.float32(phi0)
    phi -= np.floor(phi)

    mask = phi < np.float32(q)

    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))

    ybar = np.dot(w, np.asarray(y).astype(np.float32))
    YY = np.dot(w, np.power(np.asarray(y).astype(np.float32) - ybar, 2))

    W = np.sum(w[mask])
    YW = np.dot(w[mask], np.asarray(y).astype(np.float32)[mask]) - ybar * W

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
    ybar = np.dot(w, y) / np.sum(w)
    return float(np.dot(w, np.power(y - ybar, 2)))


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
        Best (q, phi0) solution at each frequency; ``phi0`` is measured
        relative to ``floor(min(t))``
    """
    _validate_convention(convention)

    t = subtract_epoch(t)[0].astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    dy = np.asarray(dy).astype(np.float32)
    freqs = np.asarray(freqs).astype(np.float32)

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
    return (convert_bls_power(bls_powers, y, dy, convention=convention),
            solutions)


def compile_sparse_bls(block_size=_default_block_size, use_simple=False, **kwargs):
    """
    Compile sparse BLS GPU kernel

    Parameters
    ----------
    block_size: int, optional (default: _default_block_size)
        CUDA threads per CUDA block.
    use_simple: bool, optional (default: False)
        Use simplified kernel (bubble sort + parallel pairs).
        Full kernel uses bitonic sort + prefix sums for O(1) range queries.

    Returns
    -------
    kernel: PyCUDA function
        The compiled sparse_bls_kernel function
    """
    # Compiling a kernel needs an active CUDA context (lazily created).
    ensure_context()

    kernel_name = 'sparse_bls_simple' if use_simple else 'sparse_bls'
    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel(kernel_name),
                                cpp_defs=cppd)

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    func_name = 'sparse_bls_kernel_simple' if use_simple else 'sparse_bls_kernel'
    kernel = module.get_function(func_name)

    # Don't use prepare() - it causes issues with large shared memory
    return kernel


def sparse_bls_gpu(t, y, dy, freqs, *, qmin=None, qmax=None,
                   ignore_negative_delta_sols=False,
                   block_size=64, max_ndata=None,
                   stream=None, kernel=None, use_simple=False,
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
    use_simple: bool, optional (default: False)
        Use simple kernel (bubble sort). Passed to compile_sparse_bls.
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

    # Convert to numpy arrays (epoch-subtract before the float32 cast)
    t, epoch = subtract_epoch(t)
    t = t.astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    dy = np.asarray(dy).astype(np.float32)
    freqs = np.asarray(freqs).astype(np.float32)

    ndata = len(t)
    nfreqs = len(freqs)

    qmins = _broadcast_q_bound(qmin, nfreqs, 0.0,
                               'qmin').astype(np.float32)
    qmaxes = _broadcast_q_bound(qmax, nfreqs, 0.5,
                                'qmax').astype(np.float32)
    _validate_q_bounds(qmins, qmaxes)

    if max_ndata is None:
        max_ndata = ndata

    # Compile kernel if not provided
    if kernel is None:
        kernel = compile_sparse_bls(block_size=block_size,
                                    use_simple=use_simple)

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

    # Block size must be a power of 2 for tree reductions
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")

    # Calculate shared memory size
    if use_simple:
        # Simple kernel: sh_phi[N] + sh_y[N] + sh_w[N] + 3*blockDim.x
        shared_mem_size = (3 * max_ndata + 3 * block_size) * 4
    else:
        # Full kernel: sh_phi[n_pow2] + sh_y[n_pow2] + sh_w[n_pow2]
        #            + sh_cumsum_w[N] + sh_cumsum_yw[N] + 3*blockDim.x
        n_pow2 = 1
        while n_pow2 < max_ndata:
            n_pow2 *= 2
        shared_mem_size = (3 * n_pow2 + 2 * max_ndata + 3 * block_size) * 4

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
    # Adjust phases to original timescale
    solutions = [(q, (phi + (epoch * freq)) % 1.0) for (q, phi), freq in zip(solutions, freqs)]

    return (convert_bls_power(bls_powers, y, dy, convention=convention),
            solutions)


def eebls_transit(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                  qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                  fmax=None, freqs=None, qvals=None,
                  use_fast=False,  use_optimized=False,
                  use_sparse=None, sparse_threshold=500,
                  use_gpu=True,
                  ignore_negative_delta_sols=False,
                  **kwargs):
    """
    Compute BLS for timeseries, automatically selecting between GPU and
    CPU implementations based on dataset size.

    For small datasets (ndata < sparse_threshold), uses the sparse BLS
    algorithm (Panahi & Zucker 2021) which avoids binning and grid searching.
    For larger datasets, uses the standard GPU-accelerated BLS.

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
        Use fast GPU implementation (if not using sparse or optimized)
    use_optimized: bool, optional (default: False)
        Use optimized GPU implementation (if not using sparse).

        This automatically selects optimal block size based on ndata:
        - ndata <= 32: 32 threads (single warp)
        - ndata <= 64: 64 threads (two warps)
        - ndata <= 128: 128 threads (four warps)
        - ndata > 128: 256 threads (eight warps)

        This provides significant speedups for small datasets by reducing
        idle thread overhead and kernel launch costs.
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
    **kwargs:
        passed to `eebls_gpu`, `eebls_gpu_fast`, `compile_bls`,
        `fmax_transit`, `fmin_transit`, and `transit_autofreq`. On the
        sparse path, only the kwargs that `sparse_bls_gpu` accepts
        (``block_size``, ``max_ndata``, ``stream``, ``kernel``,
        ``use_simple``, ``convention``) are forwarded to it. A
        ``convention=`` kwarg ('chi2ratio', 'snr' or 'loglik'; see
        :func:`convert_bls_power`) selects the power-spectrum
        convention on every path.

        .. note::

            The sparse-BLS path (default for ``ndata <
            sparse_threshold``) honors the same per-frequency Keplerian
            ``qmin_fac``/``qmax_fac`` duration bounds as the standard
            path, so results are comparable across the
            ``sparse_threshold`` boundary. ``use_fast`` only selects
            between the standard (non-sparse) implementations; pass
            ``use_sparse=False`` to force a standard grid search.

    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    solutions: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

        .. note::

            Only returned when ``use_fast=False``.

    """
    ndata = len(t)

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
                           'use_simple', 'convention')
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

    # Use GPU BLS for larger datasets

    if use_optimized:
        # Choose optimal block size
        block_size = _choose_block_size(ndata)

        # Override any user-provided block_size
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
    elif use_fast:
        powers = eebls_gpu_fast(t, y, dy, freqs,
                                qmin=qmins, qmax=qmaxes,
                                ignore_negative_delta_sols=ignore_negative_delta_sols,
                                **kwargs)
        return freqs, powers, None

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
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
                    functions=None, convention='chi2ratio', **kwargs):
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
        ``noverlap=1`` gives a single unshifted pass.
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

    Returns
    -------
    bls_results : list of ndarray
        BLS power array for each lightcurve, each shape (nfreq,).

    Notes
    -----
    With the kernel cache warm, batch mode beats a single-LC
    ``eebls_gpu_fast`` loop at every measured scale (RTX A5000,
    Jul 2026): ~10x at ndata=200, ~6x at 2,000, ~5x at 20,000
    (10 LCs, nfreq ~1800-5000). The earlier "~12x slower at TESS
    scale" regression was per-call kernel compilation (now LRU-cached
    like the single-LC paths) and its warning has been retired; see
    ``analysis/v1.0-gpu-batch3-jul2026/E1_E2_DIAGNOSIS.md``.
    """
    freqs = np.asarray(freqs).astype(np.float32)
    nfreq = len(freqs)
    n_total = len(lightcurves)
    _validate_convention(convention)

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

    # Process in batches
    all_results = [None] * n_total  # indexed by original order

    shmem_lim = kwargs.get('shmem_lim', None)
    if shmem_lim is None:
        dev = ensure_context().device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = dev.get_attribute(att)

    float_size = np.float32(1).nbytes

    i = 0
    while i < len(sorted_indices):
        # Take up to max_batch_lcs from sorted order
        batch_end = min(i + max_batch_lcs, len(sorted_indices))
        batch_indices = sorted_indices[i:batch_end]
        batch_n = len(batch_indices)

        # Max ndata in this batch
        max_ndata_batch = max(lc_ndatas[idx] for idx in batch_indices)

        # Allocate batch memory
        stream = cuda.Stream()
        mem = BLSBatchMemory(max_ndata_batch, batch_n, nfreq, stream=stream)

        # Set frequency grid
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

        # Set lightcurve data
        for j, orig_idx in enumerate(batch_indices):
            t, y, dy = lightcurves[orig_idx]
            mem.set_lightcurve(j, t, y, dy)

        # Transfer to GPU
        mem.transfer_to_gpu()

        # Launch kernel
        max_nblocks = min(nfreq, 5000)
        grid = (max_nblocks, batch_n)
        block = (block_size, 1, 1)

        # Phase oversampling, mirroring _eebls_gpu_fast_impl (A2): the
        # kernel's own noverlap argument is a no-op in its box scan, so
        # run ``noverlap`` passes with the bin grid shifted by
        # 1/noverlap of a fine bin and keep the elementwise max.
        # Without this the batch path was single-pass while the
        # fast/adaptive reference multi-passes -- the small-ndata
        # periodogram divergence flagged in the Jun GPU batch (E1).
        best_bls_g = None
        for i_pass in range(noverlap):
            dphi_pass = dphi + float(i_pass) / noverlap

            args = (grid, block, stream)
            args += (mem.t_g.ptr, mem.yw_g.ptr, mem.w_g.ptr)
            args += (mem.bls_g.ptr, mem.freqs_g.ptr)
            args += (mem.nbins0_g.ptr, mem.nbinsf_g.ptr)
            args += (mem.ndata_per_lc_g.ptr,)
            args += (np.uint32(max_ndata_batch),)
            args += (np.uint32(nfreq), np.uint32(0))
            args += (np.uint32(max_nbins), np.uint32(1))
            args += (np.float32(dlogq), np.float32(dphi_pass))
            args += (np.uint32(int(ignore_negative_delta_sols)),)
            args += (np.uint32(batch_n),)

            func.prepared_async_call(*args, shared_size=int(mem_req))

            if noverlap > 1:
                if best_bls_g is None:
                    best_bls_g = mem.bls_g.copy()
                else:
                    gpuarray.maximum(mem.bls_g, best_bls_g,
                                     out=best_bls_g, stream=stream)

        if best_bls_g is not None:
            cuda.memcpy_dtod(mem.bls_g.gpudata, best_bls_g.gpudata,
                             best_bls_g.nbytes)

        # Transfer results back
        mem.transfer_to_cpu()
        batch_results = mem.get_results()

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
    fmin_frac: float, optional (default: 1.5)
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
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    solutions: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

        .. note::

            Only returned when ``use_fast=False``.

    """

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

        return freqs, powers
    elif use_optimized:
        powers = eebls_gpu_fast_optimized(t, y, dy, freqs,
                                          qmin=qmins, qmax=qmaxes,
                                          ignore_negative_delta_sols=ignore_negative_delta_sols,
                                          **kwargs)

        return freqs, powers

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
    return freqs, powers, sols

