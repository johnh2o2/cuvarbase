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
    def __init__(self, sigma, stream, m, use_double=False,
                 precomp_psi=True, **kwargs):

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.use_double = use_double
        self.precomp_psi = precomp_psi

        # set datatypes
        self.real_type = np.float32 if not self.use_double \
            else np.float64
        self.complex_type = np.complex64 if not self.use_double \
            else np.complex128

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.t = kwargs.get('t', None)
        self.y = kwargs.get('y', None)
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

    def allocate_data(self, **kwargs):
        self.n0 = kwargs.get('n0', self.n0)
        self.nf = kwargs.get('nf', self.nf)

        assert(self.n0 is not None)
        assert(self.nf is not None)

        self.t_g = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(self.n0, dtype=self.real_type)

        return self

    def allocate_precomp_psi(self,  **kwargs):
        self.n0 = kwargs.get('n0', self.n0)

        assert(self.n0 is not None)

        self.q1 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q2 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)

        return self

    def allocate_grid(self, **kwargs):
        self.nf = kwargs.get('nf', self.nf)

        assert(self.nf is not None)

        self.n = int(self.sigma * self.nf)
        self.ghat_g = gpuarray.zeros(self.n,
                                     dtype=self.complex_type)
        self.cu_plan = cufft.Plan(self.n, self.complex_type, self.complex_type,
                                  stream=self.stream)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        self.nf = kwargs.get('nf', self.nf)

        assert(self.nf is not None)
        self.ghat_c = cuda.aligned_zeros(shape=(self.nf,),
                                         dtype=self.complex_type,
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

    def allocate(self, **kwargs):
        self.n0 = kwargs.get('n0', self.n0)
        self.nf = kwargs.get('nf', self.nf)

        assert(self.n0 is not None)
        assert(self.nf is not None)
        self.n = int(self.sigma * self.nf)

        self.allocate_data(**kwargs)
        self.allocate_grid(**kwargs)
        self.allocate_pinned_cpu(**kwargs)
        if self.precomp_psi:
            self.allocate_precomp_psi(**kwargs)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        t = kwargs.get('t', self.t)
        y = kwargs.get('y', self.y)

        assert(t is not None)
        assert(y is not None)

        self.t_g.set_async(t, stream=self.stream)
        self.y_g.set_async(y, stream=self.stream)

    def transfer_nfft_to_cpu(self, **kwargs):
        cuda.memcpy_dtoh_async(self.ghat_c, self.ghat_g.ptr,
                               stream=self.stream)

    def fromdata(self, t, y, allocate=True, **kwargs):
        self.tmin = self.real_type(min(t))
        self.tmax = self.real_type(max(t))

        self.t = np.asarray(t).astype(self.real_type)
        self.y = np.asarray(y).astype(self.real_type)

        self.n0 = kwargs.get('n0', np.int32(len(t)))
        self.nf = kwargs.get('nf', self.nf)

        if self.nf is not None and allocate:
            self.allocate(**kwargs)

        return self


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
        ``NFFTAsyncProcess.allocate()``)
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
        If False, will not transfer the resulting nfft to CPU memory
    precomp_psi: bool, optional, (default: True)
        Only relevant if ``fast`` is True. Will precompute values for the
        fast gridding procedure.
    samples_per_peak: float, optional (default: 1)
        Frequency spacing is reduced by this factor, but number of frequencies
        is kept the same

    Returns
    -------
    ghat_cpu: ``np.array``
        The resulting NFFT
    """

    precompute_psi, fast_gaussian_grid, slow_gaussian_grid, \
        nfft_shift, normalize = functions

    stream = memory.stream

    block = (block_size, 1, 1)

    batch_size = np.int32(1)

    def grid_size(nthreads):
        return int(np.ceil(float(nthreads) / block_size))

    spp = memory.real_type(samples_per_peak)
    minimum_frequency = memory.real_type(minimum_frequency)

    # transfer data -> gpu
    if transfer_to_device:
        memory.transfer_data_to_gpu()

    def check_arr(arr, name=None):
        r = any(np.isnan(arr))
        if name is None:
            print(r)
        else:
            print(name, r)
    # smooth data onto uniform grid
    if fast_grid:
        if memory.precomp_psi:
            grid = (grid_size(memory.n0 + 2 * memory.m + 1), 1)
            args = (grid, block, stream)
            args += (memory.t_g.ptr,)
            args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
            args += (memory.n0, memory.n, memory.m, memory.b)
            args += (memory.tmin, memory.tmax, spp)
            precompute_psi.prepared_async_call(*args)

        grid = (grid_size(memory.n0), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (memory.q1.ptr, memory.q2.ptr, memory.q3.ptr)
        args += (memory.n0, memory.n, batch_size, memory.m)
        args += (memory.tmin, memory.tmax, spp)
        fast_gaussian_grid.prepared_async_call(*args)

    else:
        grid = (grid_size(memory.n), 1)
        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.ghat_g.ptr)
        args += (memory.n0, memory.n, batch_size, memory.m, memory.b)
        args += (memory.tmin, memory.tmax, spp)
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
        args += (memory.n, batch_size)
        args += (memory.tmin, memory.tmax, spp, minimum_frequency)
        nfft_shift.prepared_async_call(*args)

    # Run IFFT on grid
    cufft.ifft(memory.ghat_g, memory.ghat_g, memory.cu_plan)

    # Normalize result (deconvolve smoothing kernel)
    grid = (grid_size(memory.nf), 1)
    args = (grid, block, stream)
    args += (memory.ghat_g.ptr, memory.ghat_g.ptr)
    args += (memory.n, memory.nf, batch_size, memory.b)
    args += (memory.tmin, memory.tmax, spp, minimum_frequency)
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
        D = (np.pi * (1. - 1. / (2. * sigma - 1.)))
        return int(np.ceil(-np.log(0.25 * C) / D))

    def estimate_m(self, N):
        """
        Estimate ``m`` based on an error tolerance of ``self.tol``.

        Parameters
        ----------
        N: int
            size of NFFT

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

    def get_m(self, N=None):
        if self.autoset_m:
            return self.estimate_m(N)
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
            * ``nf``: int, FFT size
        **kwargs

        Returns
        -------
        allocated_memory: list of ``NFFTMemory`` objects
            List of allocated memory for each dataset

        """

        # Purge any previously allocated memory
        allocated_memory = []

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        for i, (t, y, nf) in enumerate(data):

            m = self.get_m(nf)

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
        memory:
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

        if memory is None:
            memory = self.allocate(data, **kwargs)

        nfft_kwargs = dict(block_size=self.block_size)
        nfft_kwargs.update(kwargs)

        results = [nfft_adjoint_async(mem, self.function_tuple,
                                      **nfft_kwargs)
                   for mem in memory]

        return results
