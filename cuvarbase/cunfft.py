#!/usr/bin/env python
"""
NFFT (Non-equispaced Fast Fourier Transform) implementation.

This module provides GPU-accelerated NFFT functionality for periodogram computation.
"""
import sys
import resource
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
# import pycuda.autoinit

from . import _cufft as cufft

from .base import GPUAsyncProcess
from .utils import find_kernel, _module_reader, check_lightcurve
from .memory import NFFTMemory


__all__ = [
    'nfft_adjoint_async',
    'NFFTAsyncProcess',
]


def nfft_adjoint_async(memory, functions,
                       minimum_frequency=0., block_size=256,
                       just_return_gridded_data=False, use_grid=None,
                       fast_grid=True, transfer_to_device=True,
                       transfer_to_host=True, precomp_psi=True,
                       samples_per_peak=1, **kwargs):
    """
    Asynchronous NFFT adjoint operation.

    Use the ``NFFTAsyncProcess`` class and related subroutines when possible.

    Parameters
    ----------
    memory: ``NFFTMemory``
        Allocated memory, must have data already set (see, e.g.,
        ``NFFTAsyncProcess.allocate()``, which validates the light
        curve with :func:`cuvarbase.utils.check_lightcurve`; this
        low-level entry point cannot re-check data it does not see)
    functions: tuple, length 5
        Tuple of compiled functions from `SourceModule`. Must be prepared with
        their appropriate dtype.
    minimum_frequency: float, optional (default: 0)
        First frequency of transform
    block_size: int, optional
        Number of CUDA threads per block
    just_return_gridded_data: bool, optional
        If True, returns grid via `grid_g.get()` after gridding
    use_grid: ``GPUArray``, optional
        If specified, will skip gridding procedure and use the `GPUArray`
        provided
    fast_grid: bool, optional, default: True
        Whether or not to use the "fast" gridding procedure
    transfer_to_device: bool, optional, (default: True)
        If the data is already on the gpu, set as False
    transfer_to_host: bool, optional, (default: True)
        If False, will not transfer the resulting nfft to CPU memory.
        If True, the stream is synchronized before returning, so the
        returned host buffer is complete (before Sep 2026 the pinned
        buffer was returned while the device-to-host copy was still in
        flight: immediate reads were stale on reused memory).
    precomp_psi: bool, optional, (default: True)
        Only relevant if ``fast`` is True. Will precompute values for the
        fast gridding procedure.
    samples_per_peak: float, optional (default: 1)
        Frequency spacing is reduced by this factor, but number of frequencies
        is kept the same

    Returns
    -------
    ghat_cpu: ``np.array``
        The resulting NFFT (``memory.ghat_c``, the memory's pinned host
        buffer -- copy it out before reusing the memory)

    Notes
    -----
    The gridding kernels accumulate with atomic adds, so ``memory.ghat_g``
    is zeroed here on every call; a memory object can be reused across
    calls (before Sep 2026 a second call on the same memory summed onto
    the previous grid). With ``transfer_to_host=False`` nothing is
    synchronized: call ``memory.stream.synchronize()`` (or
    ``NFFTAsyncProcess.finish()``) before reading ``ghat_g``.
    """

    # The light curve behind ``memory`` was validated where it was
    # loaded (NFFTAsyncProcess.allocate / LombScargleMemory.setdata);
    # only the transform's own scalars can be checked here. A
    # non-finite minimum_frequency poisons every mode's phase factor
    # and a non-positive samples_per_peak collapses the grid.
    # ``minimum_frequency`` may be negative: the adjoint transform is
    # defined over modes -nf/2 .. nf/2 and the tests exercise
    # ``minimum_frequency = -nf // 2``.
    if not np.isfinite(minimum_frequency):
        raise ValueError("nfft_adjoint_async: minimum_frequency must be "
                         "finite; got %r" % (minimum_frequency,))
    if not (np.isfinite(samples_per_peak) and samples_per_peak > 0):
        raise ValueError("nfft_adjoint_async: samples_per_peak must be "
                         "finite and > 0; got %r" % (samples_per_peak,))

    precompute_psi, fast_gaussian_grid, slow_gaussian_grid, \
        nfft_shift, normalize = functions

    stream = memory.stream

    block = (block_size, 1, 1)

    batch_size = 1

    def grid_size(nthreads):
        return int(np.ceil(float(nthreads) / block_size))

    minimum_frequency = memory.real_type(minimum_frequency)

    # transfer data -> gpu
    if transfer_to_device:
        memory.transfer_data_to_gpu()

    # The gridding kernels accumulate into ghat_g with atomic adds: zero
    # it on every call so reused memory does not sum onto the previous
    # transform (only fresh gpuarray.zeros buffers were ever clean).
    if use_grid is None:
        memory.ghat_g.fill(memory.complex_type(0), stream=stream)

    # smooth data onto uniform grid
    if fast_grid:
        if memory.precomp_psi:
            grid = (grid_size(memory.n0 + 2 * memory.m + 1), 1)
            args = (grid, block, stream)
            args += (memory.t_g.ptr,)
            args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
            args += (np.int32(memory.n0), np.int32(memory.n),
                     np.int32(memory.m), memory.real_type(memory.b))
            args += (memory.real_type(memory.tmin),
                     memory.real_type(memory.tmax),
                     memory.real_type(samples_per_peak))
            precompute_psi.prepared_async_call(*args)

        grid = (grid_size(memory.n0), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
        args += (np.int32(memory.n0), np.int32(memory.n),
                 np.int32(batch_size), np.int32(memory.m))
        args += (memory.real_type(memory.tmin),
                 memory.real_type(memory.tmax),
                 memory.real_type(samples_per_peak))
        fast_gaussian_grid.prepared_async_call(*args)

    else:
        grid = (grid_size(memory.n), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (np.int32(memory.n0), np.int32(memory.n),
                 np.int32(batch_size), np.int32(memory.m),
                 memory.real_type(memory.b))
        args += (memory.real_type(memory.tmin),
                 memory.real_type(memory.tmax),
                 memory.real_type(samples_per_peak))
        slow_gaussian_grid.prepared_async_call(*args)

    # Stop if user wants the grid
    if just_return_gridded_data:
        stream.synchronize()
        return np.real(memory.ghat_g.get())

    # Set the grid manually if the user wants to
    # (only for debugging)
    if use_grid is not None:
        memory.ghat_g.set(use_grid)

    # for a non-zero minimum frequency, do a shift
    if abs(minimum_frequency) > 1E-9:
        grid = (grid_size(memory.n), 1)
        args = (grid, block, stream)
        args += (memory.ghat_g.ptr, memory.ghat_g.ptr)
        args += (np.int32(memory.n), np.int32(batch_size))
        args += (memory.real_type(memory.tmin),
                 memory.real_type(memory.tmax),
                 memory.real_type(samples_per_peak),
                 memory.real_type(minimum_frequency))
        nfft_shift.prepared_async_call(*args)

    # Run IFFT on grid
    cufft.ifft(memory.ghat_g, memory.ghat_g, memory.cu_plan)

    # Normalize result (deconvolve smoothing kernel)
    grid = (grid_size(memory.nf), 1)
    args = (grid, block, stream)
    args += (memory.ghat_g.ptr, memory.ghat_g.ptr)
    args += (np.int32(memory.n),
             np.int32(memory.nf),
             np.int32(batch_size),
             memory.real_type(memory.b))
    args += (memory.real_type(memory.tmin),
             memory.real_type(memory.tmax),
             memory.real_type(samples_per_peak),
             memory.real_type(minimum_frequency))
    normalize.prepared_async_call(*args)

    # Transfer result and wait for it: the caller gets the pinned host
    # buffer, which is only valid once the async D2H copy has landed.
    if transfer_to_host:
        memory.transfer_nfft_to_cpu()
        if stream is not None:
            stream.synchronize()
        else:
            cuda.Context.synchronize()

    return memory.ghat_c


class NFFTAsyncProcess(GPUAsyncProcess):
    """
    `GPUAsyncProcess` for the adjoint NFFT.

    Parameters
    ----------
    sigma: float, optional (default: 4)
        Size of NFFT grid will be NFFT_SIZE * sigma. The transform
        returns the one-sided modes ``k = 0..nf-1`` on a grid of
        ``sigma * nf`` points, so the effective oversampling at the top
        of the band is ``sigma / 2``: ``sigma >= 4`` is required for
        full-band accuracy in this layout (with ``sigma = 2`` the modes
        ``k >= nf/2`` are aliased at O(1), in double precision too).
    m: int, optional (default: 8)
        Maximum radius for grid contributions, used when
        ``autoset_m`` is False.
    autoset_m: bool, optional (default: False)
        Automatically set the ``m`` parameter based on the
        error tolerance given by the ``tol`` parameter (see
        :meth:`estimate_m`)
    tol: float, optional (default: 1E-8)
        Error tolerance for the NFFT (used to auto set ``m``)
    block_size: int, optional (default: 256)
        CUDA block size.
    use_double: bool, optional (default: False)
        Use double precision. On non-Tesla cards this will
        make things ~24 times slower.
    use_fast_math: bool, optional (default: True)
        Compile kernel with the ``--use_fast_math`` option
        supplied to ``nvcc``.

    Example
    -------

    >>> import numpy as np
    >>> t = np.random.rand(100)
    >>> y = np.cos(10 * t - 0.4) + 0.1 * np.random.randn(len(t))
    >>> proc = NFFTAsyncProcess()
    >>> data = [(t, y, 2 * len(t))]
    >>> nfft_adjoint = proc.run(data)

    """

    def __init__(self, *args, **kwargs):
        super(NFFTAsyncProcess, self).__init__(*args, **kwargs)

        self.sigma = kwargs.get('sigma', 4)
        self.m = kwargs.get('m', 8)
        self.autoset_m = kwargs.get('autoset_m', False)
        self.block_size = kwargs.get('block_size', 256)
        self.use_double = kwargs.get('use_double', False)
        self.m_tol = kwargs.get('tol', 1E-8)
        self.module_options = []
        if kwargs.get('use_fast_math', True):
            self.module_options.append('--use_fast_math')

        self.real_type = np.float64 if self.use_double \
            else np.float32
        self.complex_type = np.complex128 if self.use_double \
            else np.complex64

        self._cpp_defs = dict(BLOCK_SIZE=self.block_size)
        if self.use_double:
            self._cpp_defs['DOUBLE_PRECISION'] = None

        self.function_names = ['precompute_psi',
                               'fast_gaussian_grid',
                               'slow_gaussian_grid', 'nfft_shift',
                               'normalize']

        self.allocated_memory = []

    def m_from_C(self, C, sigma):
        """ 
        Returns an estimate for what ``m`` value to use from ``C``,
        where ``C`` is something like ``err_tolerance/N_freq``.

        Pulled from <https://github.com/jakevdp/nfft>_
        """
        D = (np.pi * (1. - 1. / (2. * sigma - 1.)))
        return int(np.ceil(-np.log(0.25 * C) / D))

    def estimate_m(self, N=None, y=None):
        """
        Choose the filter radius ``m`` to meet the error tolerance
        ``self.m_tol``.

        Parameters
        ----------
        N: int, optional
            Size of the NFFT. Required when ``y`` is not given
            (heuristic fallback below).
        y: array_like, optional
            The input coefficients of the adjoint NFFT (the
            observations). When given, ``m`` is chosen from the
            rigorous L1-norm error bound below.

        Returns
        -------
        m: int
            Maximum grid radius

        Notes
        -----
        The approximation error of the (adjoint) NFFT with a Gaussian
        window satisfies (NFFT3 guide, p. 11, eq. (5.9); Steidl 1998)

        .. math::

            \\max_k |E_k| \\le 4 e^{-m \\pi (1 - 1/(2\\sigma - 1))}
            \\, \\|y\\|_1

        so given the data ``y``, ``m`` is set to the smallest integer
        with :math:`4 e^{-m \\pi (1 - 1/(2\\sigma-1))} \\|y\\|_1 \\le`
        ``tol``.

        In double precision (``use_double=True``) the realized error
        tracks this bound down to the ``~1e-10`` absolute level
        (A5000-validated, Jul 2026: max error is *below* the bound for
        every ``m <= 14`` on the reference configuration, bottoming out
        near ``1e-11`` from FFT roundoff amplified by the Gaussian
        deconvolution). That figure assumes the gridding kernel rounds
        the grid coordinate in double: while ``cunfft.cu`` used
        ``floorf()`` on that coordinate (the case before the Sep-2026
        NFFT fixes) the double-precision error floor was ~1e-2 for
        times far from the origin, and the ``~1e-10`` level was reached
        only when the coordinates were exactly representable in
        float32. An earlier revision of this docstring described
        a ``~1e-3``, m-independent error floor as inherent; that floor
        was a kernel defect -- a float32 ``PI`` literal in the phase
        factors of ``nfft_shift``/``normalize`` (error
        ``~2.8e-8 * 2*pi*|k0|* ||y||_1``, amplified with ``m`` by the
        deconvolution) -- fixed in the same pass. In single precision a
        genuine floor of roughly ``1e-3`` absolute (``1e-5`` relative)
        remains: it comes from float32 trig on large un-reduced phase
        arguments and float32 grid/FFT roundoff, and very large ``m``
        *increases* it (the wider Gaussian amplifies grid noise).
        Requesting ``tol`` below that floor at single precision will not
        be honored -- use ``use_double=True`` for tolerances below
        ``~1e-2``.

        When ``y`` is unavailable, this falls back to the historical
        heuristic (from `jakevdp/nfft
        <https://github.com/jakevdp/nfft>`_) that substitutes ``N``
        for :math:`\\|y\\|_1`, which guarantees the tolerance only
        when ``max|y| <= 1``.
        """
        if y is not None:
            l1 = float(np.sum(np.absolute(y)))
            if l1 <= 0:
                # zero input: the transform is exactly zero for any m
                return 1
            return max(1, self.m_from_C(self.m_tol / l1, self.sigma))

        if N is None:
            raise ValueError("estimate_m requires N when y is not given")
        # Clamp like the y-path above: pathological tolerances
        # (m_tol > 4N) would give m <= 0, i.e. a negative Gaussian
        # shape parameter b and garbage gridding.
        return max(1, self.m_from_C(self.m_tol / N, self.sigma))

    def get_m(self, N=None, y=None):
        """
        Returns the ``m`` value for ``N`` frequencies.

        Parameters
        ----------
        N: int
            Number of frequencies, only needed if ``autoset_m`` is ``True``
            and ``y`` is not given.
        y: array_like, optional
            Adjoint-NFFT input coefficients; when given (and
            ``autoset_m`` is ``True``), ``m`` comes from the rigorous
            L1-norm bound in :func:`estimate_m`. Callers that size
            shared buffers before seeing the data (e.g. the
            Lomb-Scargle memory layouts) use the ``N`` fallback.

        Returns
        -------
        m: int
            The filter radius (in grid points)
        """
        if self.autoset_m:
            return self.estimate_m(N=N, y=y)
        else:
            return self.m

    def _compile_and_prepare_functions(self, **kwargs):
        module_txt = _module_reader(find_kernel('cunfft'), self._cpp_defs)

        self.module = SourceModule(module_txt, options=self.module_options)

        self.dtypes = dict(
            precompute_psi=[np.intp, np.intp, np.intp, np.intp, np.int32,
                            np.int32, np.int32, self.real_type,
                            self.real_type, self.real_type, self.real_type],

            fast_gaussian_grid=[np.intp, np.intp, np.intp, np.intp,
                                np.intp, np.intp, np.int32, np.int32,
                                np.int32, np.int32, self.real_type,
                                self.real_type, self.real_type],

            slow_gaussian_grid=[np.intp, np.intp, np.intp, np.int32,
                                np.int32, np.int32, np.int32, self.real_type,
                                self.real_type, self.real_type,
                                self.real_type],

            normalize=[np.intp, np.intp, np.int32, np.int32, np.int32,
                       self.real_type, self.real_type, self.real_type,
                       self.real_type, self.real_type],

            nfft_shift=[np.intp, np.intp, np.int32, np.int32, self.real_type,
                        self.real_type, self.real_type, self.real_type]
        )

        for function, dtype in self.dtypes.items():
            func = self.module.get_function(function)
            self.prepared_functions[function] = func.prepare(dtype)

        self.function_tuple = tuple([self.prepared_functions[f]
                                     for f in self.function_names])

    def allocate(self, data, **kwargs):
        """
        Allocate GPU memory for NFFT-related computations

        Parameters
        ----------
        data: list of (t, y, N) tuples
            List of data, ``[(t_1, y_1, N_1), ...]``
            * ``t``: Observation times.
            * ``y``: Observations.
            * ``nf``: int, FFT size
        **kwargs

        Returns
        -------
        allocated_memory: list of ``NFFTMemory`` objects
            List of allocated memory for each dataset

        """

        # Purge any previously allocated memory
        allocated_memory = []

        for i, d in enumerate(data):
            if len(d) != 3:
                raise ValueError(
                    "NFFTAsyncProcess.allocate: dataset %d must be a "
                    "(t, y, nf) tuple; got %d elements" % (i, len(d)))
            check_lightcurve(d[0], d[1], min_n=2,
                             name='NFFTAsyncProcess.allocate dataset %d' % i)

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        for i, (t, y, nf) in enumerate(data):

            m = self.get_m(nf, y=y)

            mem = NFFTMemory(self.sigma, self.streams[i], m,
                             use_double=self.use_double, **kwargs)

            allocated_memory.append(mem.fromdata(t, y, nf=nf,
                                                 allocate=True,
                                                 **kwargs))

        return allocated_memory

    def run(self, data, memory=None, **kwargs):
        """
        Run the adjoint NFFT on a batch of data

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, w), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``nf``: int, size of NFFT
        memory: list of ``NFFTMemory``, optional
            Preallocated memory (from :meth:`allocate`), one per
            dataset; ``data`` is ignored when given. The memory may be
            reused across calls: the grid is zeroed on every transform.
        **kwargs
            Passed to :func:`nfft_adjoint_async` (``transfer_to_host``,
            ``transfer_to_device``, ``fast_grid``, ...)

        Returns
        -------
        powers: list of np.ndarrays
            List of adjoint NFFTs. Each is the memory's pinned host
            buffer ``ghat_c``; with the default ``transfer_to_host=True``
            the stream has been synchronized and the buffer is complete
            on return (copy it before reusing the memory). With
            ``transfer_to_host=False`` call :meth:`finish` (or
            ``memory.stream.synchronize()``) before reading ``ghat_g``.

        """
        # Validate before any device work (kernel compile included).
        # ``data`` is ignored when ``memory`` is supplied, and the
        # light curve behind a memory object was validated when it was
        # allocated. min_n = 2: NFFTMemory rescales the times to
        # [-1/2, 1/2) by the baseline max(t) - min(t), which is zero
        # for a single sample -- the transform came back all-NaN.
        if memory is None:
            for i, d in enumerate(data):
                if len(d) != 3:
                    raise ValueError(
                        "NFFTAsyncProcess.run: dataset %d must be a "
                        "(t, y, nf) tuple; got %d elements" % (i, len(d)))
                check_lightcurve(d[0], d[1], min_n=2,
                                 name='NFFTAsyncProcess.run dataset %d' % i)
                nf = d[2]
                if not (np.isscalar(nf) and np.isfinite(nf)
                        and nf > 0 and int(nf) == nf):
                    raise ValueError(
                        "NFFTAsyncProcess.run: dataset %d: nf must be a "
                        "positive integer; got %r" % (i, nf))

        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions
                     for func in self.function_names]):
            self._compile_and_prepare_functions(**kwargs)

        if memory is None:
            memory = self.allocate(data, **kwargs)

        nfft_kwargs = dict(block_size=self.block_size)
        nfft_kwargs.update(kwargs)

        results = [nfft_adjoint_async(mem, self.function_tuple,
                                      **nfft_kwargs)
                   for mem in memory]

        return results
