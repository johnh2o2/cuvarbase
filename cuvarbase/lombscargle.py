import numpy as np
import pycuda.driver as cuda
import skcuda.fft as cufft
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource
from .core import GPUAsyncProcess, BaseSpectrogram
from .utils import weights, find_kernel, _module_reader
from .utils import autofrequency as utils_autofreq
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async, NFFTMemory


class LombScargleMemory(object):
    def __init__(self, sigma, stream, m, **kwargs):

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.precomp_psi = precomp_psi

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)

        self.t_g = kwargs.get('t_g', None)
        self.yw_g = kwargs.get('yw_g', None)
        self.w_g = kwargs.get('w_g', None)
        self.lsp_g = kwargs.get('lsp_g', None)

        self.nfft_mem_yw = kwargs.get('nfft_mem_yw', None)
        self.nfft_mem_w = kwargs.get('nfft_mem_w', None)

        if self.nfft_mem_yw is None:
            self.nfft_mem_yw = NFFTMemory(sigma, stream, m, **kwargs)

        if self.nfft_mem_w is None:
            self.nfft_mem_w = NFFTMemory(sigma, stream, m, **kwargs)

        self.real_type = self.nfft_mem_yw.real_type
        self.complex_type = self.nfft_mem_yw.complex_type

        self.buffered_transfer = kwargs.get('buffered_transfer', False)

        self.tbuff, self.ywbuff, self.wbuff = None, None, None
        self.lsp_c = kwargs.get('lsp_c', None)

    def allocate_data(self, n0=None, **kwargs):
        n0 = n0 if n0 is not None else self.n0

        assert(n0 is not None)

        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.yw_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.w_g = gpuarray.zeros(n0, dtype=self.real_type)

        self.nfft_mem_w.t_g = self.t_g
        self.nfft_mem_w.y_g = self.w_g

        self.nfft_mem_yw.t_g = self.t_g
        self.nfft_mem_yw.y_g = self.yw_g

        return self

    def allocate_grids(self, n0=None, nf=None, **kwargs):

        n0 = n0 if n0 is not None else self.n0

        assert(n0 is not None)

        nf = nf if nf is not None else self.nf

        assert(nf is not None)

        if self.nfft_mem_yw.precomp_psi:
            self.nfft_mem_yw.allocate_precomp_psi(n0=n0)

        # Only one precomp psi needed
        self.nfft_mem_w.precomp_psi = False
        self.nfft_mem_w.q1 = self.nfft_mem_yw.q1
        self.nfft_mem_w.q2 = self.nfft_mem_yw.q2
        self.nfft_mem_w.q3 = self.nfft_mem_yw.q3

        nfft_mem_yw.allocate_grid(nf=nf)
        nfft_mem_w.allocate_grid(nf=2*nf)

        self.lsp_g = gpuarray.zeros(nf, dtype=self.real_type)
        return self

    def allocate_pinned_cpu(self, nf=None, **kwargs):
        nf = nf if nf is not None else self.nf

        assert(nf is not None)
        self.lsp_c = cuda.aligned_zeros(shape=(nf,), dtype=self.complex_type,
                                        alignment=resource.getpagesize())
        self.lsp_c = cuda.register_host_memory(self.lsp_c)

        return self

    def is_ready(self):
        raise NotImplementedError()

    def allocate_buffered_data_arrays(self, n0=None, **kwargs):
        n0 = n0 if n0 is not None else self.n0

        assert(n0 is not None)
        self.tbuff = cuda.aligned_zeros(shape=(n0,),
                                        dtype=self.real_type,
                                        alignment=resource.getpagesize())
        self.tbuff = cuda.register_host_memory(self.tbuff)

        self.ywbuff = cuda.aligned_zeros(shape=(n0,),
                                         dtype=self.real_type,
                                         alignment=resource.getpagesize())
        self.ywbuff = cuda.register_host_memory(self.tbuff)

        self.wbuff = cuda.aligned_zeros(shape=(n0,),
                                        dtype=self.real_type,
                                        alignment=resource.getpagesize())
        self.wbuff = cuda.register_host_memory(self.tbuff)
        return self

    def allocate(self, n0, nf, **kwargs):
        self.n0 = np.int32(n0)
        self.nf = np.int32(nf)
        self.n = np.int32(self.sigma * self.nf)

        self.nfft_mem_yw.n = self.n
        self.nfft_mem_yw.nf = self.nf
        self.nfft_mem_yw.n0 = self.n0

        self.nfft_mem_w.n = 2 * self.n
        self.nfft_mem_w.nf = 2 * self.nf
        self.nfft_mem_w.n0 = self.n0

        self.allocate_data()
        self.allocate_grids()
        self.allocate_pinned_cpu()

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays()

        return self

    def transfer_data_to_gpu(self, t, y, dy,
                             convert_to_weights=True, **kwargs):

        w = dy if not convert_to_weights else weights(dy)

        self.ybar = self.real_type(np.dot(y, dy))
        yw = np.multiply(w, y - self.ybar)
        self.yy = self.real_type(np.dot(w, np.power(y - self.ybar, 2)))

        t_, yw_, w_ = None, None, None
        if self.buffered_transfer and len(t) < self.n0:
            self.n0 = np.int32(len(t))
            self.tbuff[:n0] = t[:]
            self.ywbuff[:n0] = yw[:]
            self.wbuff[:n0] = w[:]
            t_, yw_, w_ = self.tbuff, self.ybuff, self.wbuff
        else:
            t_ = np.asarray(t).astype(self.real_type)
            yw_ = np.asarray(yw).astype(self.real_type)
            w_ = np.asarray(w).astype(self.real_type)
        
        # Set minimum and maximum t values (needed to scale things
        # for the NFFT)
        self.tmin = self.real_type(min(t))
        self.tmax = self.real_type(max(t))
        
        self.nfft_mem_yw.tmin = self.tmin
        self.nfft_mem_w.tmin = self.tmin

        self.nfft_mem_yw.tmax = self.tmax
        self.nfft_mem_w.tmax = self.tmax

        # Do asynchronous data transfer
        self.t_g.set_async(t_, stream=self.stream)
        self.yw_g.set_async(yw_, stream=self.stream)
        self.w_g.set_async(w_, stream=self.stream)

    def transfer_lsp_to_cpu(self, **kwargs):
        cuda.memcpy_dtoh_async(self.lsp_c, self.lsp_g.ptr,
                               stream=self.stream)


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


def lomb_scargle_async(memory, functions, nf,
                       block_size=256, use_double=False, n0=None, use_fft=True,
                       minimum_frequency=0., python_dir_sums=False,
                       samples_per_peak=10, transfer_to_device=True, **kwargs):

    (lomb, lomb_dirsum), nfft_funcs = functions
    df = 1. / (memory.tmax - memory.tmin) / samples_per_peak
    df = real_type(df)
    stream = memory.stream

    # Use direct sums (on CPU)
    if python_dir_sums:
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

        args = (grid, block, stream,
                memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr,
                memory.lsp_g.ptr, memory.nf, memory.n0, memory.yy,
                df, df)

        lomb_dirsum.prepared_async_call(*args)

        memory.transfer_lsp_to_cpu()
        return lsp

    # NFFT
    nfft_kwargs = dict(transfer_to_host=False,
                       transfer_to_device=False,
                       min_freq=minimum_frequency,
                       samples_per_peak=samples_per_peak)

    nfft_kwargs.update(kwargs)
    """
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
    """
    else:
        # NFFT(w * (y - ybar))
        nfft_adjoint_async(memory.nfft_mem_yw, nfft_funcs, **nfft_kwargs)

        # NFFT(w)
        nfft_adjoint_async(memory.nfft_mem_w, nfft_funcs, **nfft_kwargs)

    args = (grid, block, stream)
    args += (memory.nfft_mem_w.ghat_g.ptr, memory.nfft_mem_yw.ghat_g.ptr)
    args += (memory.lsp_g.ptr, memory.nf, memory.yy)
    lomb.prepared_async_call(*args)

    if transfer_to_device:
        memory.transfer_lsp_to_cpu()

    return memory.lsp_c


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

        self.allocated_memory = []

    def _compile_and_prepare_functions(self, **kwargs):

        module_text = _module_reader(find_kernel('lomb'), self._cpp_defs)

        self.module = SourceModule(module_text, options=self.module_options)
        self.dtypes = dict(
            lomb=[np.intp, np.intp, np.intp, np.int32,
                  self.real_type],
            lomb_dirsum=[np.intp, np.intp, np.intp, np.intp,
                         np.int32, np.int32, self.real_type, self.real_type,
                         self.real_type]
        )

        self.nfft_proc._compile_and_prepare_functions(**kwargs)
        for fname, dtype in self.dtypes.iteritems():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))

    def memory_requirement(self, data, **kwargs):
        """ return an approximate GPU memory requirement in bytes """
        raise NotImplementedError()

    def allocate_for_single_lc(self, n0, nf, cu_plan_w=None,
                               cu_plan_yw=None, stream=None, **kwargs):

        m = self.nfft_proc.estimate_m(nf)

        mem = LombScargleMemory(sigma, stream, m, **kwargs)
        mem.allocate(n0, nf, **kwargs)

        return mem

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

            mem = self.allocate_for_single_lc(len(t), nf,
                                              stream=self.streams[i])



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
