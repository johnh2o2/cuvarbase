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

def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1


def time_shift(t, samples_per_peak=5):
    """
    Shifts and scales a time series to the range [-0.5, 5)
    for an NFFT.

    Parameters
    ----------
    t : array_like
        Observation times
    samples_per_peak : float, optional (default: 5)
        Scaling factor; the frequency spacing is
        `(1/(samples_per_peak * T))`, where `T` is
        `max(t) - min(t)`.

    Returns
    -------
    tshift : np.array
        Shifted observation times
    phi0 :
    """

    t = np.asarray(t)

    T = max(t) - min(t)
    tshift = (t - min(t)) / (T * samples_per_peak) - 0.5

    phi0 = min(t) / (T * samples_per_peak)

    return tshift, phi0

def nfft_adjoint_async(stream, data, gpu_data, result, functions,
                        m=8, sigma=2, block_size=256,
                        just_return_gridded_data=False, use_grid=None,
                        fast_grid=True, phi0=0, transfer_to_device=True,
                        transfer_to_host=True, precomp_psi=True,
                        use_double=False, **kwargs):
    """
    Asynchronous NFFT adjoint operation.

    Use the `NFFTAsyncProcess` class and related subroutines when possible.

    Parameters
    ----------
    stream: cuda.Stream
        Cuda stream to use.
    data: tuple, length 3
        The tuple contains (`t`, `y`, `N`);
            * `t` contain observation times, scaled to be within [-0.5, 0.5)
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
    phi0: float, default: 0
        The initial phase shift due to shifting t to the [-0.5, 0.5) interval
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
       center_fft, divide_phi_hat = functions

     # types
    real_type = np.float64 if use_double else np.float32
    complex_type = np.complex128 if use_double else np.complex64

    block = (block_size, 1, 1)

    batch_size = np.int32(1)

    grid_size = lambda nthreads : int(np.ceil(float(nthreads) / block_size))

    n0 = np.int32(len(t))
    n = np.int32(sigma * N)
    m = np.int32(m)
    b = real_type(float(2 * sigma * m) / ((2 * sigma - 1) * np.pi))
    phi0 = real_type(phi0)

    if transfer_to_device:
        t_g.set_async(np.asarray(t).astype(real_type), stream=stream)
        y_g.set_async(np.asarray(y).astype(real_type), stream=stream)

    if fast_grid:
        if precomp_psi:
            grid = ( grid_size(n0 + 2 * m + 1), 1 )
            precompute_psi.prepared_async_call(grid, block, stream,
                        t_g.ptr, q1.ptr, q2.ptr, q3.ptr, n0, n, m, b)

        """
        assert(not any(np.isnan(q1.get())))
        assert(not any(np.isnan(q2.get())))
        assert(not any(np.isnan(q3.get())))
        print(q1.get(), q2.get(), q3.get())
        import sys
        sys.exit()
        """

        grid = ( grid_size(n0), 1 )
        fast_gaussian_grid.prepared_async_call(grid, block, stream,
                                            t_g.ptr, y_g.ptr, grid_g.ptr,
                                            q1.ptr, q2.ptr, q3.ptr,
                                            n0, n, batch_size, m)
        #assert(not any(np.isnan(grid_g.get())))
        #print("grid? ", any(np.isnan(grid_g.get())))
        #print("grid: ", grid_g.get())
    else:
        grid = (grid_size(n), 1)
        slow_gaussian_grid.prepared_async_call(grid, block, stream,
                                t_g.ptr, y_g.ptr, grid_g.ptr, n0, n,
                                batch_size, m, b)

    if just_return_gridded_data:
        stream.synchronize()
        return grid_g.get()

    if use_grid is not None:
        grid_g.set(use_grid)


    grid = ( grid_size(n), 1 )
    center_fft.prepared_async_call(grid, block, stream,
                                   grid_g.ptr, ghat_g.ptr, n,
                                   batch_size, phi0)


    #print("ghat (after center)? ", any(np.isnan(ghat_g.get())))
    #print("ghat: ", ghat_g.get())

    cufft.ifft(ghat_g, ghat_g, cu_plan)

    #print("ghat (after ifft)? ", any(np.isnan(ghat_g.get())))
    #print("ghat: ", ghat_g.get())

    grid = ( grid_size(N), 1 )
    divide_phi_hat.prepared_async_call(grid, block, stream,
                                       ghat_g.ptr, ghat_g_final.ptr, n, N,
                                       batch_size, b)

    #print("ghat (after normalization)? ", any(np.isnan(ghat_g.get())))
    #print("ghat: ", ghat_g.get())
    if transfer_to_host:
        cuda.memcpy_dtoh_async(ghat_cpu, ghat_g_final.ptr, stream)

    return ghat_cpu


class NFFTAsyncProcess(GPUAsyncProcess):
    """
    `GPUAsyncProcess` for the adjoint NFFT.

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
        if kwargs.get('use_fast_math', False):
            self.module_options.append('--use_fast_math')


        self.real_type = np.float64 if self.use_double \
                         else np.float32
        self.complex_type = np.complex128 if self.use_double \
                            else np.complex64


        self._cpp_defs = dict(BLOCK_SIZE=self.block_size)
        if self.use_double:
            self._cpp_defs['DOUBLE_PRECISION'] = None

    def m_from_C(self, C, sigma):
        return int(np.ceil(-np.log(0.25 * C) / (np.pi * (1. - 1. / (2. * sigma - 1.)))))

    def estimate_m(self, tol, N, sigma):
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
            precompute_psi = [np.intp, np.intp, np.intp, np.intp, np.int32,
                              np.int32, np.int32, self.real_type],

            fast_gaussian_grid = [np.intp, np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.int32, np.int32,
                                  np.int32, np.int32],

            slow_gaussian_grid = [np.intp, np.intp, np.intp, np.int32,
                                  np.int32, np.int32, np.int32, self.real_type],

            divide_phi_hat = [np.intp, np.intp, np.int32, np.int32, np.int32,
                              self.real_type],

            center_fft = [np.intp, np.intp, np.int32, np.int32, self.real_type]
        )

        for function, dtype in self.dtypes.iteritems():
            func = self.module.get_function(function)
            self.prepared_functions[function] = func.prepare(dtype)

        function_names = ['precompute_psi', 'fast_gaussian_grid',
                          'slow_gaussian_grid', 'center_fft',
                          'divide_phi_hat']

        self.function_tuple = tuple([self.prepared_functions[f]
                                     for f in function_names])


    def allocate_grid(self, N):
        if not N%2 == 0:
            raise Exception("N = %d is not even."%(N))
        n = int(self.sigma * N)

        grid_g = gpuarray.zeros(n, dtype=self.real_type)
        ghat_g = gpuarray.zeros(n, dtype=self.complex_type)
        ghat_g_final = gpuarray.zeros(N, dtype=self.complex_type)

        ghat_cpu = cuda.aligned_zeros(shape=(N,), dtype=self.complex_type,
                            alignment=resource.getpagesize())
        ghat_cpu = cuda.register_host_memory(ghat_cpu)

        return grid_g, ghat_g, ghat_g_final, ghat_cpu

    def allocate(self, data, **kwargs):
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

            gpu_data.append((t_g, y_g, q1, q2, q3, grid_g, ghat_g, ghat_g_final, cu_plan))
            pow_cpus.append(ghat_cpu)

        return gpu_data, pow_cpus

    def run(self, data, gpu_data=None, pow_cpus=None, **kwargs):

        if self.autoset_m:
            N = max([d[2] for d in data])
            self.m = self.estimate_m(self.m_tol, N, self.sigma)

        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions
                     for func in ['precompute_psi', 'fast_gaussian_grid',
                                  'slow_gaussian_grid', 'divide_phi_hat',
                                  'center_fft']]):
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

        [s.synchronize() for s in self.streams]

        return results


"""
if __name__ == '__main__':

    ndata = 5000
    year = 365.
    p_min = 0.1  # minimum period (minutes)
    T = 1. * year   # baseline (years)
    oversampling = 5 # df = 1 / (o * T)
    batch_size = 10
    nlcs = 1 * batch_size
    block_size = 160

    # nominal number of frequencies needed
    Nf = int(oversampling * T / p_min)
    #print(Nf)
    #Nf = 10
    sigma = 2
    noise_sigma = 0.1
    m=8

    # nearest power of 2
    n = 2 ** int(np.ceil(np.log2(Nf)))

    rand = np.random.RandomState(100)
    signal_freqs = np.linspace(0.1, 0.4, nlcs)

    random_times = lambda N : shifted(np.sort(rand.rand(N) - 0.5))
    noise = lambda : noise_sigma * rand.randn(len(x))
    omega = lambda freq : 2 * np.pi * freq * len(x)
    phase = lambda : 2 * np.pi * rand.rand()

    random_signal = lambda X, frq : np.cos(omega(frq) * X - phase()) + noise()

    x = random_times(ndata)
    y = [ random_signal(x, freq) for freq in signal_freqs ]
    err = [ noise_sigma * np.ones_like(Y) for Y in y ]


    #test_fast_gridding(x, y[0], n)
    #print("FAST GRIDDING OK!")

    #test_nfft_adjoint_async(x, y[0], n)
   # print("NFFT OK!")
    test_fast_gridding(x, y[0], n)
    test_slow_gridding(x, y[0], n)
    test_post_gridding(x, y[0], n)
    test_nfft_adjoint_async(x, y[0], n)
    #fhats = nfft_adjoint_accelerated(x, y, n, fast=fast, sigma=sigma, batch_size=batch_size,
    #                               m=m, block_size=block_size)

    #dt_batch = time() - t0

    #fhats_nb = []
    #t0 = time()
    #for Y in y:
    #    fhats_nb.extend(nfft_adjoint_accelerated(x, Y, n, fast=fast, sigma=sigma,
            #                m=m, block_size=block_size))
    #dt_nonbatch = time() - t0


    #warp_size = 32
    #timing_info = []
    #for warp_multiple in 1 + np.arange(32):
    #    block_size = warp_multiple * warp_size
    #    t0 = time()
    #    fhats = nfft_adjoint_accelerated(x, y, n, fast=True, sigma=sigma,
    #                                            m=m, block_size=block_size)
    #    dt_fast = time() - t0
    #    timing_info.append((block_size, dt_fast))

    #for b, dt in timing_info:
    #    print(b, dt)

    ncpu = len(signal_freqs)
    t0 = time()
    fhat_cpus = [ nfft_adjoint_cpu(x, Y, n,
                                    sigma=sigma, m=m,
                                    use_fft=True,
                                    truncated=True) \
                    for i, Y in enumerate(y) if i < ncpu ]

    dt_cpu = time() - t0

    print(dt_batch / len(signal_freqs), dt_nonbatch / len(signal_freqs), dt_cpu / ncpu)

    #sys.exit()
    #fhat_cpus = nfft_adjoint_accelerated(x, y, n, m, fast=False)


    for i, (fhat, fhat_cpu) in enumerate(zip(fhats, fhat_cpus)):
        freqs = np.arange(len(fhat)) - len(fhat) / 2
        f, ax = plt.subplots()
        X = np.absolute(fhat_cpu)
        Y = np.absolute(fhat)
        #ax.scatter(freqs, 2 * (Y - X) / np.median(Y + X), marker='.', s=1, alpha=0.5)

        ax.scatter(X, Y, s=1, alpha=0.05)
        #ax.plot(X, color='k')
        #ax.plot(Y, color='r', alpha=0.5)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.set_xlim(1E-1, 1.1 * max([ max(X), max(Y) ]))
        #ax.set_ylim(1E-1, 1.1 * max([ max(X), max(Y) ]))
        #ax.plot(freqs, np.absolute(fhat_cpu), color='b', alpha=0.6 / (i + 1))
        #ax.plot(freqs, np.absolute(fhat) , color='r', alpha=0.6 / (i + 1))
        #ax.axvline( freq * ndata)

        #xmin, xmax = ax.get_xlim()
        #xline = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
        #ax.plot(xline, xline, ls=':', color='k')
        plt.show()
        plt.close(f)
"""
