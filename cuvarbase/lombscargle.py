import numpy as np
import pycuda.driver as cuda
import skcuda.fft as cufft
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource
from .core import GPUAsyncProcess, BaseSpectrogram
from .utils import weights, find_kernel, _module_reader
from .utils import autofrequency as utils_autofreq
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async

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


def lomb_scargle_async(stream, data_cpu, data_gpu, lsp, functions, nf,
                       block_size=256, sigma=2, m=8, use_cpu_nfft=False,
                       use_double=False, n0=None, use_fft=True,
                       minimum_frequency=0., use_dy_as_weights=False,
                       python_dir_sums=False, samples_per_peak=10, **kwargs):

    t, y, dy = data_cpu
    nf = np.int32(nf)

    n0 = n0 if n0 is not None else len(t)

    w = np.zeros_like(dy)
    w[:n0] = dy[:n0] if use_dy_as_weights else np.power(dy[:n0], -2)
    w /= sum(w)

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

    data_w = (t, w, 2 * nf)
    data_yw = (t, yw, nf)

    t_g.set_async(np.asarray(t).astype(real_type), stream)
    w_g.set_async(np.asarray(w).astype(real_type), stream)
    yw_g.set_async(np.asarray(yw).astype(real_type), stream)

    if python_dir_sums:
        df = 1. / (max(t[:n0]) - min(t[:n0])) / samples_per_peak

        return lomb_scargle_direct_sums(t_g.get(), w_g.get(), yw_g.get(),
                                        df * (1 + np.arange(nf)), YY)

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
        df = 1. / (max(t[:n0]) - min(t[:n0])) / samples_per_peak
        df = real_type(df)
        lomb_dirsum.prepared_async_call(grid, block, stream,
                                        t_g.ptr, w_g.ptr, yw_g.ptr,
                                        freqs.ptr, g_lsp.ptr, np.int32(nf),
                                        np.int32(len(t)), real_type(YY),
                                        df, df)
        cuda.memcpy_dtoh_async(lsp, g_lsp.ptr, stream)
        stream.synchronize()
        # print(zip(freqs.get()[:10], g_lsp.get()[:10]))

        return lsp

    # NFFT
    nfft_kwargs = dict(sigma=sigma, m=m, block_size=block_size,
                       transfer_to_host=False,
                       transfer_to_device=False, min_freq=minimum_frequency,
                       use_double=use_double, n0=n0,
                       samples_per_peak=samples_per_peak)

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
            lomb_dirsum = [ np.intp, np.intp, np.intp, np.intp,
                     np.int32, np.int32, self.real_type, self.real_type, self.real_type ]
        )

        self.nfft_proc._compile_and_prepare_functions(**kwargs)
        for fname, dtype in self.dtypes.iteritems():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))
        #self.prepared_functions.update(self.nfft_proc.prepared_functions)


    def move_freqs_to_gpu(self, freqs, **kwargs):
        return [gpuarray.to_gpu(frq.astype(self.real_type))
                for frq in freqs]

    def memory_requirement(self, data, **kwargs):
        """ return an approximate GPU memory requirement in bytes """
        raise NotImplementedError()

        tot_npts = sum([ len(t) for t, y, err in data ])

        nflts = 5 * tot_npts + 4 * len(data) * (4 * sigma + 3) * nf

        return nflts * (32 if self.real_type == np.float32 else 64)

    def allocate_for_single_lc(self, n0, nf, cu_plan_w=None,
                               cu_plan_yw=None, stream=None, **kwargs):

        n = int(nf * self.nfft_proc.sigma)
        t_g, yw_g, w_g, q1, q2 = tuple([gpuarray.zeros(n0, dtype=self.real_type)
                                        for j in range(5) ])
        q3 = gpuarray.zeros(2 * self.nfft_proc.m + 1, dtype=self.real_type)

        g_lsp = gpuarray.zeros(nf, dtype=self.real_type)

        # Need NFFT that's 2 * length needed, since it's *centered*
        # -N/2, -N/2 + 1, ..., 0, 1, ..., (N-1)/2
        # grid_g_w = gpuarray.zeros(4 * n, dtype=self.real_type)
        ghat_g_w = gpuarray.zeros(2 * n, dtype=self.complex_type)
        ghat_g_w_f = ghat_g_w
        grid_g_w = ghat_g_w

        # grid_g_yw = gpuarray.zeros(2 * n, dtype=self.real_type)
        ghat_g_yw = gpuarray.zeros(n, dtype=self.complex_type)
        ghat_g_yw_f = ghat_g_yw
        grid_g_yw = ghat_g_yw

        if cu_plan_w is None:
            cu_plan_w = cufft.Plan(2 * n, self.complex_type, self.complex_type,
                                   stream=stream)
        if cu_plan_yw is None:
            cu_plan_yw = cufft.Plan(n, self.complex_type, self.complex_type,
                                    stream=stream)

        lsp = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
                                    alignment=resource.getpagesize())
        lsp = cuda.register_host_memory(lsp)

        gpu_data = (t_g, yw_g, w_g, g_lsp, q1, q2, q3,
                            grid_g_w, grid_g_yw, ghat_g_w,
                            ghat_g_yw, ghat_g_w_f, ghat_g_yw_f,
                            cu_plan_w, cu_plan_yw)

        return gpu_data, lsp

    def autofrequency(self, *args, **kwargs):
        return utils_autofreq(*args, **kwargs)

    def _nfreqs(self, t, nyquist_factor=5,
                samples_per_peak=5, maximum_frequency=None, **kwargs):

        df = 1. / (max(t) - min(t)) / samples_per_peak

        if maximum_frequency is not None:
            nf = int(np.ceil(maximum_frequency / df - 1))
        else:
            nf = int(0.5 * samples_per_peak * nyquist_factor * len(t))

        return nf

    def allocate(self, data, nfreqs=None, **kwargs):

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

        nfrqs = nfreqs

        if nfrqs is None:
            nfrqs = [self._nfreqs(t, **kwargs) for (t, y, dy) in data]

        elif not hasattr(nfreqs, '__getitem__'):
            nfrqs = nfrqs * np.ones(len(data))

        for i, ((t, y, dy), nf) in enumerate(zip(data, nfrqs)):

            gpu_d, lsp = self.allocate_for_single_lc(len(t), nf, stream=self.streams[i])

            gpu_data.append(gpu_d)
            lsps.append(lsp)
        return gpu_data, lsps

    def run(self, data, freqs=None, gpu_data=None, lsps=None,
            phi0s=None, scale=True, use_fft=True,
            convert_to_weights=True, nyquist_factor=5, samples_per_peak=5,
            minimum_frequency=None, maximum_frequency = None, use_dy_as_weights=False,
            **kwargs):

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
        use_fft: optional, bool (default: True)
            Uses the NFFT, otherwise just does direct summations (which
            are quite slow...)
        **kwargs

        Returns
        -------
        results: list of lists
            list of zip(freqs, pows) for each LS periodogram

        """

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in \
                     ['lomb', 'lomb_dirsum']]):
            self._compile_and_prepare_functions(**kwargs)


        freqs = [self.autofrequency(d[0], **kwargs) for d in data]
        nfreqs = [len(frq) for frq in freqs]
        if lsps is None or gpu_data is None:
            gpu_data, lsps = self.allocate(data, nfreqs=nfreqs, **kwargs)



        ls_kwargs = dict(sigma=self.nfft_proc.sigma,
                         m=self.nfft_proc.m,
                         block_size=self.block_size,
                         use_double=self.nfft_proc.use_double,
                         use_fft=use_fft,
                         samples_per_peak=samples_per_peak,
                         use_dy_as_weights=False)

        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        streams = [s for i, s in enumerate(self.streams) if i < len(data)]
        results = [lomb_scargle_async(stream, cdat, gdat, lsp, funcs, nf,
                                      **ls_kwargs)
                   for stream, cdat, gdat, lsp, nf in \
                                  zip(streams, data, gpu_data, lsps, nfreqs)]

        results = [(f, r) for f, r in zip(freqs, results)]
        return results


    def _get_pagelocked_data_buffers(self, n0, batch_size, **kwargs):
        # make pagelocked memory buffers to transfer the data
        data_pl_arrs = [tuple(cuda.aligned_zeros(shape=(n0,),
                                                 dtype=self.real_type,
                                                 alignment=resource.getpagesize())
                              for j in range(3)) for i in range(batch_size)]

        data_pl_arrs = [tuple(cuda.register_host_memory(arr)
                              for arr in d) for d in data_pl_arrs]

        return data_pl_arrs

    def _transfer_batch_to_pagelocked_buffers(self, data, plbuffs, **kwargs):

        for i, (t, y, w) in range(len(data)):
            plbuffs[i][0][len(t):] = 0.
            plbuffs[i][1][len(t):] = 0.
            plbuffs[i][2][len(t):] = 0.

            plbuffs[i][0][:len(t)] = t[:]
            plbuffs[i][1][:len(t)] = y[:]
            plbuffs[i][2][:len(t)] = w[:]

        return plbuffs

    def batched_run_const_nfreq(self, data, batch_size=10, df=None,
                                min_samples_per_peak=10., maximum_frequency=24.,
                                use_dy_as_weights=False, **kwargs):
        """
        Same as ``batched_run`` but is more efficient when the frequencies are the
        same for each lightcurve. Doesn't reallocate memory for each batch.
        """

        bsize = min([len(data), batch_size])
        if bsize < len(self.streams):
            self._create_streams(bsize - len(self.streams))

        streams = [self.streams[i] for i in range(bsize)]
        lsps = [np.zeros(nf) for i in range(len(data))]

        max_ndata = max([len(t) for t, y, dy in data])

        # set frequencies
        if df is None:
            max_baseline = max([max(t) - min(t) for t, y, dy in data])
            df = 1./(min_samples_per_peak * max_baseline)
        freqs = df * (1 + np.arange(int(maximum_frequency / df) - 1))

        # make pagelocked data buffers
        data_buffers = self._get_pagelocked_data_buffers(max_ndata, bsize)

        # make data batches
        batches = []
        while len(batches) * batch_size < len(data):
            off = len(batches) * batch_size
            end = off + min([batch_size, len(data) - off])
            batches.append([data[i] for i in range(off, end)])

        # allocate gpu and cpu (pinned) memory
        gpu_data, pl_lsps = self.allocate(batch[0], **kwargs)

        # setup keyword args for lomb scargle
        ls_kwargs = dict(sigma=self.nfft_proc.sigma,
                         m=self.nfft_proc.m,
                         block_size=self.block_size,
                         use_double=self.nfft_proc.use_double,
                         use_fft=use_fft,
                         use_dy_as_weights=use_dy_as_weights)

        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)

        for b, batch in enumerate(batches):
            bs = len(batch)

            # transfer batch to pagelocked buffers
            buffs = self._transfer_batch_to_pagelocked_buffers(batch, data_buffers)
            for i, (stream, gdat, buff, pl_lsp) \
                in enumerate(zip(streams, gpu_data, buffs, pl_lsps)):

                if i >= bs:
                    break

                samples_per_peak = 1./(df * (max(t) - min(t)))
                lomb_scargle_async(stream, buff, gdat, pl_lsp, funcs, len(freqs),
                                   n0=len(batch[i][0]), **ls_kwargs)

            # wait to finish, then transfer pagelocked memory to non-pagelocked memory
            for i, (stream, pl_lsp) in enumerate(zip(streams, pl_lsps)):
                if i >= bs:
                    break

                stream.synchronize()

                lsps[b * batch_size + i][:] = pl_lsp[:]

        return [(freqs, lsp) for lsp in lsps]


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
    results = proc.run([(t, y, w)], **kwargs)

    freqs, powers = results[0]

    proc.finish()

    return freqs, powers


"""
class LombScargleSpectrogram(BaseSpectrogram):
    def __init__(self, t, y, w, **kwargs):

        super(LombScargleSpectrogram, self).__init__(t, y, w, **kwargs)
        if self.proc is None:
            self.proc = LombScargleAsyncProcess()
"""
