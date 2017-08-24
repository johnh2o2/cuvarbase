import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource
from .core import GPUAsyncProcess, BaseSpectrogram
from .utils import weights, find_kernel
from time import time


class Wavelet(object):
    def __init__(self, t, y, dy, pmin=0.05, pmax=15.0, sigma=0.05, 
                    precision=8, samples_per_peak=5, resolution_enhancement=1.,
                    block_size=128):
        self.t = t
        self.y = y
        self.dy = dy

        self.sigma = sigma
        self.precision = precision
        self.samples_per_peak = samples_per_peak
        self.pmin = pmin
        self.pmax = pmax
        self.block_size=block_size
        self.resolution_enhancement = resolution_enhancement

        self.dlnf = 7.32E-3 * (5./samples_per_peak) * (sigma / 0.05) * (8./precision)**0.5

        self.nfreqs = int(np.ceil(np.log(pmax / pmin) / self.dlnf))

        self.freqs = np.logspace(np.log10(1./pmax), np.log10(1./pmin), self.nfreqs)

        print("WAVELET: making taus")
        t0 = time()
        self.taus = []
        for f in self.freqs:
            
            # 0.5 = exp(-(sigma * omega * (width/2))^2)
            dtau = np.sqrt(np.log(2)) / ( np.pi * self.sigma * f * self.resolution_enhancement)

            ntaus = int(np.ceil((max(t) - min(t)) / dtau))
            taus = np.linspace(min(t), max(t), ntaus).tolist()
            taus.append(max(t))
            self.taus.append(taus)

            
        print(time() - t0)
        self.prepared_functions = {}
        print("WAVELET: preparing/compiling functions")
        t0 = time()
        self._compile_and_prepare_functions()
        print(time() - t0)

    def _compile_and_prepare_functions(self, **kwargs):

        self.module = SourceModule(open(find_kernel('wavelet'), 'r').read(), 
                                        options=['--use_fast_math'])

        self.dtypes = dict(
            wavelet_spectrogram =[ np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, 
                                    np.intp, np.int32, np.int32, np.float32, np.float32]
        )

        for function, dtype in self.dtypes.iteritems():
            self.prepared_functions[function] = self.module.get_function(function).prepare(dtype)

#float *t, float *y, float *w, float *spectrogram, 
#float *freqs, float *taus, int *ntaus, int nfreqs, 
#int nobs, float sigma, float prec
    def run(self):
        ntaus = [ len(taus) for taus in self.taus ]
        npxls = sum(ntaus)


        spectrogram_g = gpuarray.zeros(npxls, dtype=np.float32)
        ntaus_g = gpuarray.to_gpu(np.array(ntaus, dtype=np.int32))
        t_g = gpuarray.to_gpu(np.array(self.t, dtype=np.float32))
        y_g = gpuarray.to_gpu(np.array(self.y, dtype=np.float32))

        w = np.power(self.dy, -2)
        w_g = gpuarray.to_gpu(np.array(w / np.median(w), dtype=np.float32))

        sig_g = np.float32(self.sigma)
        prec_g = np.float32(self.precision)
        nobs_g = np.int32(len(self.t))
        nfreqs_g = np.int32(len(self.freqs))

        freqs_g = gpuarray.to_gpu(np.array(self.freqs, dtype=np.float32))

        taus_flat = []
        for taus in self.taus:
            taus_flat.extend(list(taus))
        taus_g = gpuarray.to_gpu(np.array(taus_flat, dtype=np.float32))


        block = (self.block_size, 1, 1)
        grid = (int(np.ceil(float(npxls) / self.block_size)), 1) 
        self.prepared_functions['wavelet_spectrogram'].prepared_call(grid, block,
            t_g.ptr, y_g.ptr, w_g.ptr, spectrogram_g.ptr, freqs_g.ptr, taus_g.ptr, ntaus_g.ptr, nfreqs_g, 
            nobs_g, sig_g, prec_g)

        self.spectrogram = spectrogram_g.get()

        return self.spectrogram



#class Wavelet(object):
#   def __init__(self, t, y, dy, sigma=0.05, samples_per_peak=5, precision=8, )