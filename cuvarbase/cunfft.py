#!/usr/bin/env python

import sys
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import skcuda.fft as cufft
from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader
import resource
import numpy as np


class NFFTMemory(object):
    def __init__(self, sigma, stream, m, double_precision=False,
                 precomp_psi=True, **kwargs):

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.double_precision = double_precision
        self.precomp_psi = precomp_psi

        # set datatypes
        self.real_type = np.float32 if not self.double_precision \
            else np.float64
        self.complex_type = np.complex64 if not self.double_precision \
            else np.complex128

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.f0 = kwargs.get('f0', 0.)
        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)
        self.t_g = kwargs.get('t_g', None)
        self.y_g = kwargs.get('y_g', None)
        self.ghat_g = kwargs.get('ghat_g', None)
        self.ghat_c = kwargs.get('ghat_c', None)
        self.q1 = kwargs.get('q1', None)
        self.q2 = kwargs.get('q2', None)
        self.q3 = kwargs.get('q3', None)
        self.cu_plan = kwargs.get('cu_plan', None)

        D = (2 * self.sigma - 1) * np.pi
        self.b = self.real_type(float(2 * self.sigma * self.m) / D)

    def allocate_data(self, n0=None, **kwargs):
        n0 = n0 if n0 is not None else self.n0

        assert(n0 is not None)

        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(n0, dtype=self.real_type)

        return self

    def allocate_precomp_psi(self, n0=None,  **kwargs):
        n0 = n0 if n0 is not None else self.n0

        assert(n0 is not None)

        self.q1 = gpuarray.zeros(n0, dtype=self.real_type)
        self.q2 = gpuarray.zeros(n0, dtype=self.real_type)
        self.q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)

        return self

    def allocate_grid(self, nf=None, **kwargs):
        nf = nf if nf is not None else self.nf

        assert(nf is not None)

        n = int(self.sigma * nf)
        self.ghat_g = gpuarray.zeros(n,
                                     dtype=self.complex_type)
        self.cu_plan = cufft.Plan(n, self.complex_type, self.complex_type,
                                  stream=self.stream)
        return self

    def allocate_pinned_cpu(self, nf=None, **kwargs):
        nf = nf if nf is not None else self.nf

        assert(nf is not None)
        self.ghat_c = cuda.aligned_zeros(shape=(nf,), dtype=self.complex_type,
                                         alignment=resource.getpagesize())
        self.ghat_c = cuda.register_host_memory(self.ghat_c)

        return self

    def is_ready(self):
        assert(self.n0 == len(self.t_g))
        assert(self.n0 == len(self.y_g))
        assert(self.n == len(self.ghat_g))

        if self.ghat_c is not None:
            assert(self.nf == len(self.ghat_c))

        if self.precomp_psi:
            assert(self.n0 == len(self.q1))
            assert(self.n0 == len(self.q2))
            assert(2 * self.m + 1 == len(self.q3))

    def allocate(self, n0, nf, **kwargs):
        self.n0 = n0
        self.nf = nf
        self.n = int(self.sigma * nf)

        self.allocate_data()
        self.allocate_grid()
        self.allocate_pinned_cpu()
        if self.precomp_psi:
            self.allocate_precomp_psi()

        return self

    def transfer_data_to_gpu(self, t, y, **kwargs):
        self.tmin = self.real_type(min(t))
        self.tmax = self.real_type(max(t))

        self.t_g.set_async(np.asarray(t).astype(self.real_type),
                           stream=self.stream)
        self.y_g.set_async(np.asarray(y).astype(self.real_type),
                           stream=self.stream)

    def transfer_nfft_to_cpu(self, **kwargs):
        cuda.memcpy_dtoh_async(self.ghat_cpu, self.ghat_gpu.ptr,
                               stream=self.stream)


def nfft_adjoint_async(memory, functions,
                       min_freq=0., block_size=256, n0=None,
                       just_return_gridded_data=False, use_grid=None,
                       fast_grid=True, transfer_to_device=True,
                       transfer_to_host=True, precomp_psi=True,
                       use_double=False, samples_per_peak=1, **kwargs):
    """
    Asynchronous NFFT adjoint operation.

    Use the `NFFTAsyncProcess` class and related subroutines when possible.

    Parameters
    ----------
    stream: cuda.Stream
        Cuda stream to use.
    data: tuple, length 3
        The tuple contains (`t`, `y`, `N`);
            * `t` contain observation times
            * `y` contain the measured values
            * `N` is the size of the NFFT
    gpu_data: tuple, length 8
        The tuple contains
            * `t_g`, `y_g` : `GPUArray`s of length `len(t)`. Used to transfer
              `t` and `y` to the GPU. Assumed to have a dtype of `np.float32`
            * `q1`, `q2`, `q3` : These are `GPUArray`s that hold precomputed
               values for the fast gridding procedures. The `q1` and `q2`
               `GPUArrays` both have the same length as `t` and the `q3` array
               has length `2 * m + 1`, where `m` is the filter radius. Assumed
               to have a dtype of `np.float32`
            * `grid_g` : `GPUArray` for the grid; length `n * sigma`, dtype
               assumed to be `np.float32`.
            * `ghat_g` : `GPUArray` for the transform: assumed to have length
               `n * sigma` and have a dtype of `np.complex64`
            * `cu_plan` : is the `skcuda.fft.Plan` for the FFT.
    result: `GPUArray`
        Place to transfer (asynchronously) the result on the CPU.
        Must be registered with `cuda.register_host_memory` in order to be
        asynchronous.
    functions: tuple, length 5
        Tuple of compiled functions from `SourceModule`. Must be prepared with
        their appropriate dtype.
    m: int, optional
        The filter "radius" for smoothing data onto grid.
    sigma: int, optional
        The grid size divided by the final FFT size
    block_size: int, optional
        Number of CUDA threads per block
    just_return_gridded_data: bool, optional
        If True, returns grid via `grid_g.get()` after gridding
    use_grid: `GPUArray`, optional
        If specified, will skip gridding procedure and use the `GPUArray`
        provided
    fast_grid: bool, optional, default: True
        Whether or not to use the "fast" gridding procedure
    transfer_to_device: bool, optional, default: True
        If False, does not transfer result from GPU to CPU.
    precomp_psi: bool, optional, default: True
        Only relevant if `fast` is True. If False, will compute the `q1`, `q2`
        and `q3` values.
    use_double : bool, optional (default: False)
        Use double-precision (on GTX cards this will make things ~24 times
        slower)

    Returns
    -------
    `ghat_cpu`: GPUArray
        The resulting complex transform.
    """

    precompute_psi, fast_gaussian_grid, slow_gaussian_grid, \
        nfft_shift, normalize = functions
    
    stream = memory.stream

    block = (block_size, 1, 1)

    batch_size = np.int32(1)

    def grid_size(nthreads):
        return int(np.ceil(float(nthreads) / block_size))

    spp = real_type(samples_per_peak)
    min_freq = real_type(min_freq)

    if fast_grid:
        if memory.precomp_psi:
            grid = (grid_size(memory.n0 + 2 * memory.m + 1), 1)
            args = (grid, block, stream)
            args += (memory.t_g.ptr,)
            args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
            args += (memory.n0, memory.n, memory.m, memory.b)
            args += (memory.tmin, memory.tmax, memory.spp)
            precompute_psi.prepared_async_call(*args)

        grid = (grid_size(n0), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
        args += (memory.n0, memory.n, batch_size, memory.m)
        args += (memory.tmin, memory.tmax, memory.spp)
        fast_gaussian_grid.prepared_async_call(*args)
    else:
        grid = (grid_size(n), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (memory.n0, memory.n, batch_size, memory.m, memory.b)
        args += (memory.tmin, memory.tmax, memory.spp)
        slow_gaussian_grid.prepared_async_call(*args)

    if just_return_gridded_data:
        stream.synchronize()
        return np.real(memory.ghat_g.get())

    if use_grid is not None:
        memory.ghat_g.set(use_grid)

    # Shift the grid in Fourier space
    grid = (grid_size(n), 1)
    args = (grid, block, stream)
    args += (memory.ghat_g.ptr, memory.ghat_g.ptr)
    args += (memory.n, batch_size)
    args += (memory.tmin, memory.tmax, memory.spp, min_freq)
    nfft_shift.prepared_async_call(*args)

    # Run IFFT on centered grid
    cufft.ifft(memory.ghat_g, memory.ghat_g, memory.cu_plan)

    # Normalize
    grid = (grid_size(N), 1)
    args = (grid, block, stream)
    args += (memory.ghat_g.ptr, memory.ghat_g.ptr)
    args += (memory.n, memory.nf, batch_size, memory.b)
    args += (memory.tmin, memory.tmax, memory.spp, min_freq)
    normalize.prepared_async_call(*args)

    # Transfer result!
    if transfer_to_host:
        memory.transfer_nfft_to_cpu()

    return memory.ghat_c


class NFFTAsyncProcess(GPUAsyncProcess):
    """
    `GPUAsyncProcess` for the adjoint NFFT.

    Parameters
    ----------
    sigma: float, optional (default: 2)
        Size of NFFT grid will be NFFT_SIZE * sigma
    m: int, optional (default: 8)
        Maximum radius for grid contributions (by default,
        this value will automatically be set based on a specified
        error tolerance)
    autoset_m: bool, optional (default: True)
        Automatically set the ``m`` parameter based on the
        error tolerance given by the ``m_tol`` parameter
    m_tol: float, optional (default: 1E-8)
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

        self.sigma = kwargs.get('sigma', 2)
        self.m = kwargs.get('m', 8)
        self.autoset_m = kwargs.get('autoset_m', True)
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
        D = (np.pi * (1. - 1. / (2. * sigma - 1.)))
        return int(np.ceil(-np.log(0.25 * C) / D))

    def estimate_m(self, N):
        """
        Automatically set ``m`` based on an error tolerance of ``tol``.


        Parameters
        ----------
        tol: float
            Error tolerance
        N: int
            size of NFFT
        sigma: float
            Grid size is ``sigma * N``

        Returns
        -------
        m: int
            Maximum grid radius

        Notes
        -----
        Pulled from <https://github.com/jakevdp/nfft>_.

        """

        # TODO: this should be computed in terms of the L1-norm of the true
        #   Fourier coefficients... see p. 11 of
        #   https://www-user.tu-chemnitz.de/~potts/nfft/guide/nfft3.pdf
        #   Need to think about how to estimate the value of m more accurately
        return self.m_from_C(self.m_tol / N, self.sigma)

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

        for function, dtype in self.dtypes.iteritems():
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
            * ``N``: int, FFT size
        **kwargs

        """

        # Purge any previously allocated memory
        self.allocated_memory = []

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        for i, (t, y, nf) in enumerate(data):

            m = self.m
            if self.autoset_m:
                m = self.estimate_m(nf)

            mem = NFFTMemory(sigma, self.streams[i], m,
                             double_precision=self.use_double)

            self.allocated_memory.append(mem.allocate(len(t), nf))

    def run(self, data, mem=None, **kwargs):
        """
        Run the adjoint NFFT on a batch of data

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, w), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``N``: size of NFFT
        gpu_data: optional, list of tuples
            List of tuples containing allocated GPU objects for each dataset
        pow_cpus: optional, list of ``np.ndarray``
            List of page-locked (registered) np.ndarrays for asynchronous
            transfers of NFFT to CPU
        **kwargs

        Returns
        -------
        powers: list of np.ndarrays
            List of adjoint NFFTs

        """
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions
                     for func in self.function_names]):
            self._compile_and_prepare_functions(**kwargs)

        if mem is None:
            self.allocate(data, **kwargs)
        else:
            assert(len(mem) == len(data))
            self.allocated_memory = mem

        streams = [s for i, s in enumerate(self.streams) if i < len(data)]

        nfft_kwargs = dict(block_size=self.block_size)

        nfft_kwargs.update(kwargs)

        results = [nfft_adjoint_async(stream, cdat, mem, self.function_tuple,
                                      **nfft_kwargs)
                   for stream, cdat, mem in
                   zip(streams, data, self.allocated_memory)]

        return results
