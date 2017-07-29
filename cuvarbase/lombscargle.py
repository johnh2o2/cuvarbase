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
    """
    Super slow lomb scargle (for debugging), uses
    direct summations to compute C, S, ...
    """
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

        if (np.isinf(P) or np.isnan(P) or P < 0 or P > 1):
            print(P, YC, YS, CC, SS, YY)
            P = 0.
        lsp[i] = P

    return lsp

def lomb_scargle_async(stream, data_cpu, data_gpu, lsp, functions, block_size=256,
            sigma=2, m=8, use_cpu_nfft=False, use_double=False, phi0=0.,
            use_fft=True, freqs=None, python_dir_sums=False, **kwargs):


    t, y, w, nf = data_cpu

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

    # Use direct sums (on GPU)
    if not use_fft:
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
        stream.synchronize()
        #print(zip(freqs.get()[:10], g_lsp.get()[:10]))

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
    """
    GPUAsyncProcess for the Lomb Scargle periodogram

    Parameters
    ----------
    **kwargs: passed to ``NFFTAsyncProcess``

    Example
    -------
    >>> proc = LombScargleAsyncProcess()
    >>> Ndata = 1000
    >>> t = np.sort(365 * np.random.rand(N))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    >>> y += 0.01 * np.random.randn(len(t))
    >>> dy = 0.01 * np.ones_like(y)
    >>> freqs, powers = proc.run([(t, y, dy)])
    >>> proc.finish()
    >>> ls_freqs, ls_powers = freqs[0], powers[0]

    """
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
    def scale_data(cls, data, convert_to_weights=True,
                   samples_per_peak=5, maximum_frequency=None,
                   nyquist_factor=5, **kwargs):

        scaled_data = []
        scaled_freqs = []
        freqs = []
        phi0s = []
        for (t, y, err) in data:
            T = max(t) - min(t)
            tshift, phi0 = time_shift(t, samples_per_peak=samples_per_peak)

            if maximum_frequency is None:
                maximum_frequency = nyquist_factor * 0.5 * len(t) / T

            nf = int(np.ceil(maximum_frequency * T * samples_per_peak))

            w = err if not convert_to_weights else weights(err)
            scaled_data.append((tshift, y, w, nf))
            phi0s.append(phi0)

            df = 1./(samples_per_peak * T)
            freqs.append(df * (1 + np.arange(nf)))
            scaled_freqs.append(1 + np.arange(nf))
        return scaled_data, freqs, scaled_freqs, phi0s


    def move_freqs_to_gpu(self, freqs, **kwargs):
        return [gpuarray.to_gpu(frq.astype(self.real_type))
                for frq in freqs]

    def memory_requirement(self, data, **kwargs):
        """ return an approximate GPU memory requirement """
        tot_npts = sum([ len(t) for t, y, err in data ])

        nflts = 5 * tot_npts + 4 * len(data) * (4 * sigma + 3) * nf

        return nflts * (32 if self.real_type == np.float32 else 64)

    def allocate(self, data, **kwargs):

        """
        Allocate GPU memory for Lomb Scargle computations

        Parameters
        ----------
        data: list of (t, y, N) tuples
            List of data, ``[(t_1, y_1, w_1), ...]``
            * ``t``: Observation times
            * ``y``: Observations
            * ``w``: Observation **weights** (sum(w) = 1)
        **kwargs

        Returns
        -------
        gpu_data: list of tuples
            List of tuples containing GPU-allocated objects for each
            dataset
            * ``t_g``: ``GPUArray``, real, length = length of data
            * ``y_g``: ``GPUArray``, real, length = length of data
            * ``w_g``: ``GPUArray``, real, length = length of data
            * ``q1``: ``GPUArray``, real, length = length of data
            * ``q2``: ``GPUArray``, real, length = length of data
            * ``q3``: ``GPUArray``, real, length = 2 * m + 1
            * ``grid_g_w``: ``GPUArray``, real, length = 4 * sigma * nf
            * ``grid_g_yw``: ``GPUArray``, real, length = 4 * sigma * nf
            * ``ghat_g_w``: ``GPUArray``, complex, length = 2 * sigma * nf
            * ``ghat_g_yw``: ``GPUArray``, complex, length = 2 * sigma * nf
            * ``ghat_g_w_f``: ``GPUArray``, complex, length = 4 * nf
            * ``ghat_g_yw_f``: ``GPUArray``, complex, length = 2 * nf
            * ``cu_plan_w``: ``cufft.Plan`` for C2C transform (4*sigma*nf)
            * ``cu_plan_yw``: ``cufft.Plan`` for C2C transform (2*sigma*nf)

        lsps: list of ``np.ndarray``s
            List of registered ndarrays for transferring periodograms to CPU

        """


        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, lsps =  [], []

        for i, (t, y, w, nf) in enumerate(data):
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
            phi0s=None, scale=True, use_fft=True,
            convert_to_weights=True, **kwargs):

        """
        Run Lomb Scargle on a batch of data.

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, w), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``y_err``: observation uncertainties
        convert_to_weights: optional, bool, (default: True)
            If False, it assumes ``y_err`` are weights (i.e. sum(y_err) = 1)
        freqs: optional, list of ``np.ndarray`` [**DO NOT USE**]
            List of custom frequencies (don't use this!! Not working)
        gpu_data: optional, list of tuples
            List of tuples containing allocated GPU objects for each dataset
        lsps: optional, list of ``np.ndarray``
            List of page-locked (registered) np.ndarrays for asynchronously
            transferring results to CPU
        scale: optional, bool (default: True)
            Scale the incoming data and frequencies to [-0.5, 0.5].
            If you're passing regular data (times, mags, mag_errs),
            you want this to be true!
        use_fft: optional, bool (default: True)
            Uses the NFFT, otherwise just does direct summations (which
            are quite slow...)
        phi0s: optional, array_like, (default: None)
            The phase shifts for the scaled observation times
        **kwargs

        Returns
        -------
        results: list of lists
            list of zip(freqs, pows) for each LS periodogram
        powers: list of np.ndarrays
            List of periodogram powers

        """
        scaled_data, freqs = data, freqs

        if scale:
            scaled_data, freqs, scaled_freqs, phi0s \
                = self.scale_data(data, convert_to_weights=convert_to_weights,
                                    **kwargs)

        if phi0s is None:
            phi0s = np.zeros(len(data))
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
        results = [lomb_scargle_async(stream, cdat, gdat, lsp, funcs,
                                      freqs=fgpu, phi0=phi0,
                                      **ls_kwargs)
                   for stream, cdat, gdat, lsp, fgpu, phi0 in \
                                  zip(streams, scaled_data, gpu_data,
                                      lsps, freqs_gpu, phi0s)]
        results = [zip(f, r) for f, r in zip(freqs, results)]
        return results


def lomb_scargle_simple(t, y, dy, **kwargs):
    """
    Simple lomb-scargle interface for testing that
    things work on the GPU. Note: This will be
    substantially slower than working with the
    ``LombScargleAsyncProcess`` interface.
    """

    w = np.power(dy, -2)
    w /= sum(w)
    proc = LombScargleAsyncProcess()
    freqs, results = proc.run([(t, y, w)], **kwargs)
    proc.finish()

    return freqs[0], results[0]


"""
class LombScargleSpectrogram(BaseSpectrogram):
    def __init__(self, t, y, w, **kwargs):

        super(LombScargleSpectrogram, self).__init__(t, y, w, **kwargs)
        if self.proc is None:
            self.proc = LombScargleAsyncProcess()
"""