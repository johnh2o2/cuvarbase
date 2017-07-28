import numpy as np
import pycuda.driver as cuda
import skcuda.fft as cufft
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource
from .core import GPUAsyncProcess, BaseSpectrogram
from .utils import weights, find_kernel, _module_reader
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async, time_shift

def lomb_scargle_direct_sums(t, w, yw, freqs, YY):
    lsp = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        phase = 2 * np.pi * (((t + 0.5) * freq) % 1.0).astype(np.float64)
        phase2 = 2 * np.pi * (((t + 0.5) * 2 * freq) % 1.0).astype(np.float64)

        cc = np.cos(phase)
        ss = np.sin(phase)

        cc2 = np.cos(phase2)
        ss2 = np.sin(phase2)

        C = np.dot(w, cc)
        S = np.dot(w, ss)

        C2 = np.dot(w, cc2)
        S2 = np.dot(w, ss2)

        Ch = np.dot(yw, cc)
        Sh = np.dot(yw, ss)

        tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))

        C2w = 1. / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
        S2w = tan_2omega_tau * C2w

        Cw = np.sqrt(0.5 * (1 + C2w))
        Sw = np.sqrt(0.5 * (1 - C2w))

        if S2w < 0:
            Sw *= -1

        YC = Ch * Cw + Sh * Sw
        YS = Sh * Cw - Ch * Sw
        CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
        SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

        CC -= (C * Cw + S * Sw) * (C * Cw + S * Sw);
        SS -= (S * Cw - C * Sw) * (S * Cw - C * Sw);

        P = (YC * YC / CC + YS * YS / SS) / YY;

        print(P, YC, YS, CC, SS, YY)
        lsp[i] = 0 if (np.isinf(P) or np.isnan(P)) else P

    return lsp

def lomb_scargle_async(stream, data_cpu, data_gpu, lsp, functions, block_size=256,
            sigma=2, m=8, use_cpu_nfft=False, use_double=False,
            use_fft=True, freqs=None, python_dir_sums=False, **kwargs):


    t, y, w, phi0, nf = data_cpu

    # types
    real_type = np.float64 if use_double else np.float32
    complex_type = np.complex128 if use_double else np.complex64

    (lomb, lomb_dirsum), nfft_funcs = functions

    t_g, yw_g, w_g, g_lsp, q1, q2, q3, \
        grid_g_w, grid_g_yw, ghat_g_w, \
        ghat_g_yw, ghat_g_w_f, ghat_g_yw_f, \
        cu_plan_w, cu_plan_yw = data_gpu

    ybar = np.dot(w, y)
    yw = np.multiply(w, y - ybar)
    YY = np.dot(w, np.power(y-ybar, 2))

    data_w  = (t, w, 4 * nf)
    data_yw = (t, yw, 2 * nf)

    t_g.set_async(np.asarray(t).astype(real_type), stream)
    w_g.set_async(np.asarray(w).astype(real_type), stream)
    yw_g.set_async(np.asarray(yw).astype(real_type), stream)

    if python_dir_sums:
        return lomb_scargle_direct_sums(t_g.get(), w_g.get(), yw_g.get(),
                                        freqs.get(), YY)

    block = (block_size, 1, 1)
    grid = (int(np.ceil(nf / float(block_size))), 1)

    if not use_fft:

        print("USING DIRECT SUMS")
        if freqs is None:
            raise Exception("If use_fft is False, freqs must be specified"
                            "and must be a GPUArray instance.")

        if not isinstance(freqs, gpuarray.GPUArray):
            raise Exception("If use_fft is False, freqs must be specified"
                            "and must be a GPUArray instance. (freqs is"
                            "a `{type}` instance".format(type(freqs)))

        lomb_dirsum.prepared_async_call(grid, block, stream,
                                        t_g.ptr, w_g.ptr, yw_g.ptr,
                                        freqs.ptr, g_lsp.ptr, np.int32(nf),
                                        np.int32(len(t)), real_type(YY))
        cuda.memcpy_dtoh_async(lsp, g_lsp.ptr, stream)
        return lsp

    # NFFT
    nfft_kwargs = dict(sigma=sigma, m=m, block_size=block_size,
                       phi0=phi0, transfer_to_host=False,
                       transfer_to_device=False,
                       use_double=use_double)

    nfft_kwargs.update(kwargs)

    if use_cpu_nfft:
        # FOR DEBUGGING
        from nfft import nfft_adjoint as nfft_adjoint_cpu

        gh_yw = np.zeros(len(ghat_g_yw_f), dtype=complex_type)
        gh_w = np.zeros(len(ghat_g_w_f), dtype=complex_type)


        gh_yw = nfft_adjoint_cpu(*data_yw, sigma=sigma,
                                 m=m).astype(complex_type)
        gh_w = nfft_adjoint_cpu(*data_w, sigma=sigma,
                                m=m).astype(complex_type)

        ghat_g_yw_f.set_async(gh_yw, stream)
        ghat_g_w_f.set_async(gh_w, stream)

    else:
        # NFFT(w)
        gpu_data_w = (t_g, w_g, q1, q2, q3, grid_g_w,
                      ghat_g_w, ghat_g_w_f, cu_plan_w)
        nfft_adjoint_async(stream, data_w, gpu_data_w, None,
                           nfft_funcs, precomp_psi=True,
                           **nfft_kwargs)

        # NFFT(w * (y - ybar))
        gpu_data_yw = (t_g, yw_g, q1, q2, q3, grid_g_yw,
                       ghat_g_yw, ghat_g_yw_f, cu_plan_yw)
        nfft_adjoint_async(stream, data_yw, gpu_data_yw, None,
                           nfft_funcs, precomp_psi=False,
                           **nfft_kwargs)


    lomb.prepared_async_call(grid, block, stream,
                             ghat_g_w_f.ptr, ghat_g_yw_f.ptr, g_lsp.ptr,
                             np.int32(nf), real_type(YY))
    cuda.memcpy_dtoh_async(lsp, g_lsp.ptr, stream)

    return lsp




class LombScargleAsyncProcess(GPUAsyncProcess):
    def __init__(self, *args, **kwargs):
        super(LombScargleAsyncProcess, self).__init__(*args, **kwargs)

        self.nfft_proc = NFFTAsyncProcess(*args, **kwargs)
        self._cpp_defs = self.nfft_proc._cpp_defs

        self.real_type = self.nfft_proc.real_type
        self.complex_type = self.nfft_proc.complex_type

        self.block_size = self.nfft_proc.block_size
        self.module_options = self.nfft_proc.module_options

    def _compile_and_prepare_functions(self, **kwargs):

        module_text = _module_reader(find_kernel('lomb'), self._cpp_defs)

        self.module = SourceModule(module_text, options=self.module_options)
        self.dtypes = dict(
            lomb = [ np.intp, np.intp, np.intp, np.int32,
                     self.real_type],
            lomb_dirsum = [ np.intp, np.intp, np.intp, np.intp, np.intp,
                     np.int32, np.int32, self.real_type ]
        )

        self.nfft_proc._compile_and_prepare_functions(**kwargs)
        for fname, dtype in self.dtypes.iteritems():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))
        #self.prepared_functions.update(self.nfft_proc.prepared_functions)


    @classmethod
    def scale_data(cls, data, samples_per_peak=5, maximum_frequency=None,
                        nyquist_factor=5, **kwargs):

        scaled_data = []
        scaled_freqs = []
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
            scaled_freqs.append(1 + np.arange(nf))
        return scaled_data, freqs, scaled_freqs


    def move_freqs_to_gpu(self, freqs, **kwargs):
        return [gpuarray.to_gpu(frq.astype(self.real_type))
                for frq in freqs]

    def allocate(self, data, regularize=None, **kwargs):
        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, lsps =  [], []

        for i, (t, y, w, phi0, nf) in enumerate(data):
            n = int(nf * self.nfft_proc.sigma)
            n0 = len(t)

            t_g, yw_g, w_g, q1, q2 = tuple([gpuarray.zeros(n0, dtype=self.real_type)
                                            for j in range(5) ])
            q3 = gpuarray.zeros(2 * self.nfft_proc.m + 1, dtype=self.real_type)

            g_lsp = gpuarray.zeros(nf, dtype=self.real_type)

            # Need NFFT that's 2 * length needed, since it's *centered*
            # -N/2, -N/2 + 1, ..., 0, 1, ..., (N-1)/2
            grid_g_w = gpuarray.zeros(4 * n, dtype=self.real_type)
            ghat_g_w = gpuarray.zeros(4 * n, dtype=self.complex_type)
            ghat_g_w_f = gpuarray.zeros(4 * nf, dtype=self.complex_type)

            grid_g_yw = gpuarray.zeros(2 * n, dtype=self.real_type)
            ghat_g_yw = gpuarray.zeros(2 * n, dtype=self.complex_type)
            ghat_g_yw_f = gpuarray.zeros(2 * nf, dtype=self.complex_type)

            cu_plan_w = cufft.Plan(4 * n, self.complex_type, self.complex_type,
                           stream=self.streams[i])

            cu_plan_yw = cufft.Plan(2 * n, self.complex_type, self.complex_type,
                           stream=self.streams[i])

            lsp = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
                                        alignment=resource.getpagesize())
            lsp = cuda.register_host_memory(lsp)

            gpu_data.append((t_g, yw_g, w_g, g_lsp, q1, q2, q3,
                                grid_g_w, grid_g_yw, ghat_g_w,
                                ghat_g_yw, ghat_g_w_f, ghat_g_yw_f,
                                cu_plan_w, cu_plan_yw))
            lsps.append(lsp)

        return gpu_data, lsps

    def run(self, data, freqs=None, gpu_data=None, lsps=None,
            scale=True, use_fft=True, **kwargs):

        scaled_data, freqs = data, freqs
        if scale:
            scaled_data, freqs, scaled_freqs = self.scale_data(data, **kwargs)
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in \
                     ['lomb', 'lomb_dirsum']]):
            self._compile_and_prepare_functions(**kwargs)

        if lsps is None or gpu_data is None:
            gpu_data, lsps = self.allocate(scaled_data, **kwargs)

        freqs_gpu = [None]*len(data)
        if not use_fft:
            freqs_gpu = self.move_freqs_to_gpu(scaled_freqs)

        ls_kwargs = dict(sigma=self.nfft_proc.sigma,
                         m=self.nfft_proc.m,
                         block_size=self.block_size,
                         use_double=self.nfft_proc.use_double,
                         use_fft=use_fft)

        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        streams = [s for i, s in enumerate(self.streams) if i < len(data)]
        results = [lomb_scargle_async(stream, cdat, gdat, lsp, funcs, freqs=fgpu,
                                      **ls_kwargs)
                   for stream, cdat, gdat, lsp, fgpu in \
                                  zip(streams, scaled_data, gpu_data,
                                      lsps, freqs_gpu)]

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





