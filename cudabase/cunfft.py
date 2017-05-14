#!/usr/bin/env python

import sys
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import skcuda.fft as cufft
import resource


def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1

def nfft_adjoint_async(stream, data, gpu_data, result, functions, 
                        m=8, sigma=2, block_size=160, 
                        just_return_gridded_data=False, use_grid=None,
                        fast_grid=True, transfer_to_device=True, 
                        transfer_to_host=True):
    t, y, N = data
    t_g, y_g, q1, q2, q3, grid_g, ghat_g, cu_plan = gpu_data
    ghat_cpu = result
    precompute_psi, fast_gaussian_grid, slow_gaussian_grid, \
                                 center_fft, divide_phi_hat = functions
    block = (block_size, 1, 1)

    batch_size = np.int32(1)

    grid_size = lambda nthreads : int(np.ceil(float(nthreads) / block_size))

    n0 = np.int32(len(t))
    n = np.int32(sigma * N)
    m = np.int32(m)
    b = np.float32(float(2 * sigma * m) / ((2 * sigma - 1) * np.pi))

    if transfer_to_device:
        t_g.set_async(np.asarray(t).astype(np.float32), stream=stream)
        y_g.set_async(np.asarray(y).astype(np.float32), stream=stream)
    
    if fast_grid:
        grid = ( grid_size(n0 + 2 * m + 1), 1 )
        precompute_psi.prepared_async_call(grid, block, stream,
                        t_g.ptr, q1.ptr, q2.ptr, q3.ptr, n0, n, m, b)

        grid = ( grid_size(n0), 1 )
        fast_gaussian_grid.prepared_async_call(grid, block, stream,
                                            t_g.ptr, y_g.ptr, grid_g.ptr, 
                                            q1.ptr, q2.ptr, q3.ptr, 
                                            n0, n, batch_size, m)
    else:
        grid = (grid_size(n), 1)
        slow_gaussian_grid.prepared_async_call(grid, block, stream,
                                t_g.ptr, y_g.ptr, grid_g.ptr, n0, n, 
                                batch_size, m, b)


    if just_return_gridded_data:
        stream.synchronize()
        return grid_g.get()

    if not use_grid is None:
        grid_g.set(use_grid)

    grid = ( grid_size(n), 1 )
    center_fft.prepared_async_call(grid, block, stream,
                            grid_g.ptr, ghat_g.ptr, n, batch_size)

    cufft.ifft(ghat_g, ghat_g, cu_plan)

    grid = ( grid_size(N), 1 )
    divide_phi_hat.prepared_async_call(grid, block, stream,
                            ghat_g.ptr, n, N, batch_size, b)

    if transfer_to_host:
        cuda.memcpy_dtoh_async(ghat_cpu, ghat_g.ptr, stream)
    
    return ghat_cpu


class NFFTAsyncProcess(GPUAsyncProcess):
    def _compile_and_prepare_functions(self, **kwargs):

        self.module = SourceModule(open('../kernels/cunfft.cu', 'r').read(), options=[ '--use_fast_math'])
        self.dtypes = dict(
            precompute_psi = [ np.intp, np.intp, np.intp, np.intp, np.int32, np.int32, np.int32, np.float32 ],
            fast_gaussian_grid = [ np.intp,
                                np.intp, np.intp, np.intp, np.intp, np.intp, np.int32,
                                                        np.int32, np.int32, np.int32],
            slow_gaussian_grid = [ np.intp,
                                np.intp, np.intp, np.int32, np.int32, np.int32, np.int32, np.float32 ],
            divide_phi_hat = [ np.intp, np.int32, np.int32, np.int32, np.float32 ],
            center_fft = [ np.intp, np.intp, np.int32, np.int32 ]
        )

        for function, dtype in self.dtypes.iteritems():
            self.prepared_functions[function] = self.module.get_function(function).prepare(dtype)

    def allocate(self, data, sigma=2, m=8, **kwargs):
        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, pow_cpus =  [], []

        for i, (t, y, N) in enumerate(data):

            n = int(sigma * N)
            n0 = len(t)

            t_g, y_g, q1, q2 = tuple([ gpuarray.zeros(n0, dtype=np.float32) for i in range(4) ])
            q3 = gpuarray.zeros(2 * m + 1, dtype=np.float32)
            grid_g = gpuarray.zeros(n, dtype=np.float32)
            ghat_g = gpuarray.zeros(n, dtype=np.complex64)

            ghat_cpu = cuda.aligned_zeros(shape=(N,), dtype=np.complex64, 
                                alignment=resource.getpagesize())
            ghat_cpu = cuda.register_host_memory(ghat_cpu)
            
            cu_plan = cufft.Plan(n, np.complex64, np.complex64, batch=1, 
                           stream=self.streams[i], istride=1, ostride=1, idist=n, odist=n)    


            gpu_data.append((t_g, y_g, q1, q2, q3, grid_g, ghat_g, cu_plan))
            pow_cpus.append(ghat_cpu)

        return gpu_data, pow_cpus

    def run(self, data, gpu_data=None, pow_cpus=None, **kwargs):
        if not hasattr(self, 'prepared_functions') or \
                   not all([ func in self.prepared_functions for func in \
                                [ 'precompute_psi', 'fast_gaussian_grid', 
                                    'slow_gaussian_grid', 
                                    'divide_phi_hat', 'center_fft']]):
            self._compile_and_prepare_functions(**kwargs)

        if pow_cpus is None or gpu_data is None:
            gpu_data, pow_cpus = self.allocate(data, **kwargs)

        streams = [ s for i, s in enumerate(self.streams) if i < len(data) ]
        results = [ nfft_adjoint_async(stream, cdat, gdat, pcpu, self.prepared_functions, **kwargs)\
                          for stream, cdat, gdat, pcpu in \
                                  zip(streams, data, gpu_data, pow_cpus)]
        
        return results



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
