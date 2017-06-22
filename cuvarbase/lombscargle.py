import numpy as np 
import pycuda.driver as cuda
import skcuda.fft as cufft
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource 
from .core import GPUAsyncProcess, BaseSpectrogram
from .utils import weights, find_kernel
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async, time_shift

def lomb_scargle_async(stream, data_cpu, data_gpu, lsp, functions, block_size=256,
            sigma=2, m=8, **kwargs):
    t, y, w, phi0, nf = data_cpu

    lomb_power, nfft_funcs = functions

    t_g, yw_g, w_g, g_lsp, q1, q2, q3, \
        grid_g_w, grid_g_yw, ghat_g_w, \
        ghat_g_yw, cu_plan_w, cu_plan_yw, reg = data_gpu

    ybar = np.dot(w, y)
    yw = np.multiply(w, y - ybar)
    YY = np.dot(w, np.power(y-ybar, 2))

    data_w  = (t, w, 4 * nf)
    data_yw = (t, yw, 2 * nf)

    t_g.set_async(np.asarray(t).astype(np.float32), stream)
    w_g.set_async(np.asarray(w).astype(np.float32), stream)
    yw_g.set_async(np.asarray(yw).astype(np.float32), stream)

    # NFFT(weights)
    gpu_data_w = (t_g, w_g, q1, q2, q3, grid_g_w, ghat_g_w, cu_plan_w)
    nfft_adjoint_async(stream, data_w, gpu_data_w, None, nfft_funcs, 
                        m=m, sigma=sigma, block_size=block_size, phi0=phi0,
                        transfer_to_host=False, transfer_to_device=False, **kwargs)

    # NFFT(w * (y - ybar))
    gpu_data_yw = (t_g, yw_g, q1, q2, q3, grid_g_yw, ghat_g_yw, cu_plan_yw)
    nfft_adjoint_async(stream, data_yw, gpu_data_yw, None, nfft_funcs, 
                        m=m, sigma=sigma, block_size=block_size, phi0=phi0,
                        transfer_to_host=False, transfer_to_device=False, 
                        precomp_psi=False, **kwargs)

    block = (block_size, 1, 1)
    grid = (int(np.ceil(nf / float(block_size))), 1)
    lomb_power.prepared_async_call(grid, block, stream, ghat_g_w.ptr, 
                                    ghat_g_yw.ptr, g_lsp.ptr, np.int32(nf), 
                                    np.float32(YY), reg.ptr)

    #ghg = ghat_g_yw.get()

    cuda.memcpy_dtoh_async(lsp, g_lsp.ptr, stream)
    #ghg = ghg[len(ghg)/2:]
    #lsp[:] = ghg[:len(lsp)]
    return lsp



class LombScargleAsyncProcess(GPUAsyncProcess):
    def __init__(self, *args, **kwargs):
        super(LombScargleAsyncProcess, self).__init__(*args, **kwargs)

        self.nfft_proc = NFFTAsyncProcess(*args, **kwargs)

        self.sigma = 2 if not 'sigma' in kwargs else kwargs['sigma']
        self.m = 8 if not 'm' in kwargs else kwargs['m']
        self.block_size = 256 if not 'block_size' in kwargs else kwargs['block_size']

    #@property
    #def streams(self):
    #    return self.nfft_proc.streams

    #def _create_streams(self, nstreams):
    #    self.nfft_proc._create_streams(nstreams)

    def _compile_and_prepare_functions(self, **kwargs):
        self.module = SourceModule(open(find_kernel('lomb'), 'r').read(), 
                        options=[ '--use_fast_math'])
        self.dtypes = dict(
            lomb = [ np.intp, np.intp, np.intp, np.int32, np.float32, np.intp ]
        )

        self.nfft_proc._compile_and_prepare_functions( **kwargs)
        for fname, dtype in self.dtypes.iteritems():
            self.prepared_functions[fname] = self.module.get_function(fname).prepare(dtype)
        self.function_tuple = self.prepared_functions['lomb']
        #self.prepared_functions.update(self.nfft_proc.prepared_functions)


    @classmethod
    def scale_data(cls, data, samples_per_peak=5, maximum_frequency=None, 
                        nyquist_factor=5, **kwargs):
        
        scaled_data = []
        freqs = []
        for (t, y, w) in data:
            T = max(t) - min(t)
            tshift, phi0 = time_shift(t, samples_per_peak=samples_per_peak)

            if maximum_frequency is None:
                maximum_frequency = nyquist_factor * 0.5 * len(t) / T

            nf = int(np.ceil(maximum_frequency * T * samples_per_peak))

            scaled_data.append((tshift, y, w, phi0, nf))

            df = 1./(samples_per_peak * T)
            freqs.append(df * (1 + np.arange(nf)))

        return scaled_data, freqs


    def allocate(self, data, regularize=None, **kwargs):
        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))


        gpu_data, lsps =  [], []

        for i, (t, y, w, phi0, nf) in enumerate(data):
            n = int(nf * self.sigma)
            n0 = len(t)

            t_g, yw_g, w_g, q1, q2 = tuple([ gpuarray.zeros(n0, dtype=np.float32) for j in range(5) ])
            q3 = gpuarray.zeros(2 * self.m + 1, dtype=np.float32)
            reg = gpuarray.zeros(3, dtype=np.float32)
            if not regularize is None:
                reg.set(np.array(regularize, dtype=np.float32))

            g_lsp = gpuarray.zeros(nf, dtype=np.float32)

            # Need NFFT that's 2 * length needed, since it's *centered*
            # -N/2, -N/2 + 1, ..., 0, 1, ..., (N-1)/2
            grid_g_w = gpuarray.zeros(4 * n, dtype=np.float32)
            ghat_g_w = gpuarray.zeros(4 * n, dtype=np.complex64)
            

            grid_g_yw = gpuarray.zeros(2 * n, dtype=np.float32)
            ghat_g_yw = gpuarray.zeros(2 * n, dtype=np.complex64)

            cu_plan_w = cufft.Plan(4 * n, np.complex64, np.complex64, 
                           stream=self.streams[i])

            cu_plan_yw = cufft.Plan(2 * n, np.complex64, np.complex64, 
                           stream=self.streams[i])    

            lsp = cuda.aligned_zeros(shape=(nf,), dtype=np.float32, 
                                        alignment=resource.getpagesize())
            lsp = cuda.register_host_memory(lsp)

            gpu_data.append((t_g, yw_g, w_g, g_lsp, q1, q2, q3, 
                                grid_g_w, grid_g_yw, ghat_g_w, 
                                ghat_g_yw, cu_plan_w, cu_plan_yw, reg))
            lsps.append(lsp)

        return gpu_data, lsps

    def run(self, data, freqs=None, regularize=None, gpu_data=None, lsps=None, 
                    sleep_length=0.01, scale=True, **kwargs):

        scaled_data, freqs = data, freqs
        if scale:
            scaled_data, freqs = self.scale_data(data, **kwargs)
        if not hasattr(self, 'prepared_functions') or \
                   not all([ func in self.prepared_functions for func in \
                                [ 'lomb']]):
            self._compile_and_prepare_functions(**kwargs)

        if lsps is None or gpu_data is None:
            gpu_data, lsps = self.allocate(scaled_data, regularize=regularize, **kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        streams = [ s for i, s in enumerate(self.streams) if i < len(data) ]
        results = [ lomb_scargle_async(stream, cdat, gdat, lsp, funcs, 
                            sigma=self.sigma, m=self.m, block_size=self.block_size,
                            **kwargs)\
                          for stream, cdat, gdat, lsp in \
                                  zip(streams, scaled_data, gpu_data, lsps)]
        
        return freqs, results


def lomb_scargle(t, y, dy, **kwargs):
    
    w = np.power(dy, -2)
    w /= sum(w)
    proc = LombScargleAsyncProcess()
    freqs, results = proc.run([(t, y, w)], **kwargs)
    proc.finish()

    return freqs[0], results[0]



class LombScargleSpectrogram(BaseSpectrogram):
    def __init__(self, t, y, w, **kwargs):

        super(LombScargleSpectrogram, self).__init__(t, y, w, **kwargs)
        if self.proc is None:
            self.proc = LombScargleAsyncProcess()

        
    


