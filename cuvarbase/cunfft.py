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


def nfft_adjoint_async(stream, data, gpu_data, result, functions,
                       m=8, sigma=2, min_freq=0., block_size=256, n0=None,
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
        Use double-precision (on GTX cards this will make things ~24 times slower)

    Returns
    -------
    `ghat_cpu`: GPUArray
        The resulting complex transform.
    """


    t, y, N = data
    t_g, y_g, q1, q2, q3, grid_g, ghat_g, ghat_g_final, cu_plan = gpu_data
    ghat_cpu = result
    precompute_psi, fast_gaussian_grid, slow_gaussian_grid, \
        nfft_shift, divide_phi_hat = functions

    # types
    real_type = np.float64 if use_double else np.float32
    complex_type = np.complex128 if use_double else np.complex64

    block = (block_size, 1, 1)

    batch_size = np.int32(1)

    def grid_size(nthreads): int(np.ceil(float(nthreads) / block_size))

    # allow for buffered arrays with longer length
    if n0 is None:
        n0 = len(t)

    n0 = np.int32(n0)
    n = np.int32(sigma * N)
    m = np.int32(m)
    b = real_type(float(2 * sigma * m) / ((2 * sigma - 1) * np.pi))
    tmin = real_type(min(t[:n0]))
    tmax = real_type(max(t[:n0]))
    spp = real_type(samples_per_peak)
    min_freq = real_type(min_freq)

    if transfer_to_device:
        t_g.set_async(np.asarray(t).astype(real_type), stream=stream)
        y_g.set_async(np.asarray(y).astype(real_type), stream=stream)

    if fast_grid:
        if precomp_psi:
            grid = (grid_size(n0 + 2 * m + 1), 1)
            precompute_psi.prepared_async_call(grid, block, stream,
                        t_g.ptr, q1.ptr, q2.ptr, q3.ptr, n0, n, m, b,
                        tmin, tmax, spp)

        grid = (grid_size(n0), 1)
        fast_gaussian_grid.prepared_async_call(grid, block, stream,
                                            t_g.ptr, y_g.ptr, grid_g.ptr,
                                            q1.ptr, q2.ptr, q3.ptr,
                                            n0, n, batch_size, m, tmin, tmax,
                                            spp)
    else:
        grid = (grid_size(n), 1)
        slow_gaussian_grid.prepared_async_call(grid, block, stream,
                                t_g.ptr, y_g.ptr, grid_g.ptr, n0, n,
                                batch_size, m, b, tmin, tmax, spp)

    if just_return_gridded_data:
        stream.synchronize()
        return grid_g.get()

    if use_grid is not None:
        grid_g.set(use_grid)

    # Shift the grid in Fourier space
    grid = (grid_size(n), 1)
    nfft_shift.prepared_async_call(grid, block, stream,
                                   grid_g.ptr, ghat_g.ptr, n,
                                   batch_size, tmin, tmax, spp, min_freq)

    # Run IFFT on centered grid
    cufft.ifft(ghat_g, ghat_g, cu_plan)

    # Normalize
    grid = (grid_size(N), 1)
    divide_phi_hat.prepared_async_call(grid, block, stream,
                                       ghat_g.ptr, ghat_g_final.ptr, n, N,
                                       batch_size, b, tmin, tmax, spp,
                                       min_freq)

    # Transfer result!
    if transfer_to_host:
        cuda.memcpy_dtoh_async(ghat_cpu, ghat_g_final.ptr, stream)

    return ghat_cpu


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

        self.function_names = ['precompute_psi_noscale',
                               'fast_gaussian_grid_noscale',
                               'slow_gaussian_grid_noscale', 'nfft_shift',
                               'divide_phi_hat_noscale']

    def m_from_C(self, C, sigma):
        D = (np.pi * (1. - 1. / (2. * sigma - 1.)))
        return int(np.ceil(-np.log(0.25 * C) / D))

    def estimate_m(self, tol, N, sigma):
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
        C = tol / N
        return self.m_from_C(C, sigma)

    def _compile_and_prepare_functions(self, **kwargs):
        module_txt = _module_reader(find_kernel('cunfft'), self._cpp_defs)

        self.module = SourceModule(module_txt, options=self.module_options)

        self.dtypes = dict(
            precompute_psi_noscale = [np.intp, np.intp, np.intp, np.intp, np.int32,
                              np.int32, np.int32, self.real_type, self.real_type,
                              self.real_type, self.real_type],

            fast_gaussian_grid_noscale = [np.intp, np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.int32, np.int32,
                                  np.int32, np.int32, self.real_type,
                                  self.real_type, self.real_type],

            slow_gaussian_grid_noscale = [np.intp, np.intp, np.intp, np.int32,
                                  np.int32, np.int32, np.int32, self.real_type,
                                  self.real_type, self.real_type, self.real_type],

            divide_phi_hat_noscale = [np.intp, np.intp, np.int32, np.int32, np.int32,
                              self.real_type, self.real_type, self.real_type, 
                              self.real_type, self.real_type],

            nfft_shift = [np.intp, np.intp, np.int32, np.int32, self.real_type,
                          self.real_type, self.real_type, self.real_type]
        )

        for function, dtype in self.dtypes.iteritems():
            func = self.module.get_function(function)
            self.prepared_functions[function] = func.prepare(dtype)



        self.function_tuple = tuple([self.prepared_functions[f]
                                     for f in self.function_names])


    def allocate_grid(self, N):
        """
        Allocate GPU memory for NFFT grid & final FFT.

        Allocates a total of (2 * sigma * n + N) * sizeof(float)
        bytes of memory on the GPU.

        Parameters
        ----------
        N: int
            Size of grid.

        Returns
        -------
        grid_g: GPUArray, real
            Grid (GPU)
        ghat_g: GPUArray, complex
            Temporary array (GPU)
        ghat_g_final:

        ghat_cpu: ``np.ndarray`` (aligned)
            Grid (CPU) ``np.ndarray``, aligned with page size to
            allow asynchronous data transfer

        """

        if not N%2 == 0:
            raise Exception("N = %d is not even."%(N))
        n = int(self.sigma * N)

        ghat_g = gpuarray.zeros(n, dtype=self.complex_type)
        ghat_g_final = ghat_g
        grid_g = ghat_g

        ghat_cpu = cuda.aligned_zeros(shape=(N,), dtype=self.complex_type,
                            alignment=resource.getpagesize())
        ghat_cpu = cuda.register_host_memory(ghat_cpu)

        return grid_g, ghat_g, ghat_g_final, ghat_cpu

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

        Returns
        -------
        gpu_data: list of tuples
            List of tuples containing GPU-allocated objects for each
            dataset
            * ``t_g``: ``GPUArray``, real, length = length of data
            * ``y_g``
            * ``q1``
            * ``q2``
            * ``q3``
            * ``grid_g``
            * ``ghat_g``
            * ``ghat_g_final``
            * ``cu_plan``
        pow_cpus: list of ``np.ndarray``s
            List of registered ndarrays for transferring final NFFTs

        """

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, pow_cpus =  [], []

        for i, (t, y, N) in enumerate(data):

            n = int(self.sigma * N)

            n0 = len(t)

            t_g, y_g, q1, q2 = tuple([gpuarray.zeros(n0, dtype=self.real_type)
                                      for j in range(4)])

            q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)
            grid_g, ghat_g, ghat_g_final, ghat_cpu = self.allocate_grid(N)

            cu_plan = cufft.Plan(int(n), self.complex_type, self.complex_type,
                                 batch=1, stream=self.streams[i],
                                 istride=1, ostride=1, idist=n, odist=n)

            gpu_data.append((t_g, y_g, q1, q2, q3, grid_g, ghat_g,
                             ghat_g_final, cu_plan))

            pow_cpus.append(ghat_cpu)

        return gpu_data, pow_cpus

    def run(self, data, gpu_data=None, pow_cpus=None, **kwargs):
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
        if self.autoset_m:
            N = max([d[2] for d in data])
            self.m = self.estimate_m(self.m_tol, N, self.sigma)

        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions
                     for func in self.function_names]):
            self._compile_and_prepare_functions(**kwargs)

        if pow_cpus is None or gpu_data is None:
            gpu_data, pow_cpus = self.allocate(data, **kwargs)

        streams = [s for i, s in enumerate(self.streams) if i < len(data)]

        nfft_kwargs = dict(sigma=self.sigma, m=self.m,
                           block_size=self.block_size,
                           use_double=self.use_double)

        nfft_kwargs.update(kwargs)

        results = [nfft_adjoint_async(stream, cdat, gdat, pcpu,
                                      self.function_tuple,
                                      **nfft_kwargs)
                   for stream, cdat, gdat, pcpu in \
                   zip(streams, data, gpu_data, pow_cpus)]

        return results
