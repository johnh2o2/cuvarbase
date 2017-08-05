import numpy as np
import pycuda.driver as cuda
import skcuda.fft as cufft
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import resource
from .core import GPUAsyncProcess
from .utils import weights, find_kernel, _module_reader
from .utils import autofrequency as utils_autofreq
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async, NFFTMemory


class LombScargleMemory(object):
    def __init__(self, sigma, stream, m, **kwargs):

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.precomp_psi = kwargs.get('precomp_psi', True)

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
        self.n0_buffer = kwargs.get('n0_buffer', None)

        self.lsp_c = kwargs.get('lsp_c', None)

        self.t = kwargs.get('t', None)
        self.yw = kwargs.get('yw', None)
        self.w = kwargs.get('w', None)

    def allocate_data(self, **kwargs):
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.yw_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.w_g = gpuarray.zeros(n0, dtype=self.real_type)

        self.nfft_mem_w.t_g = self.t_g
        self.nfft_mem_w.y_g = self.w_g

        self.nfft_mem_yw.t_g = self.t_g
        self.nfft_mem_yw.y_g = self.yw_g

        return self

    def allocate_grids(self, **kwargs):

        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        nf = kwargs.get('nf', self.nf)
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

    def allocate_pinned_cpu(self, **kwargs):
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)

        self.lsp_c = cuda.aligned_zeros(shape=(nf,), dtype=self.complex_type,
                                        alignment=resource.getpagesize())
        self.lsp_c = cuda.register_host_memory(self.lsp_c)

        return self

    def is_ready(self):
        raise NotImplementedError()

    def allocate_buffered_data_arrays(self, **kwargs):
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        self.t = cuda.aligned_zeros(shape=(n0,),
                                    dtype=self.real_type,
                                    alignment=resource.getpagesize())
        self.t = cuda.register_host_memory(self.t)

        self.yw = cuda.aligned_zeros(shape=(n0,),
                                     dtype=self.real_type,
                                     alignment=resource.getpagesize())
        self.yw = cuda.register_host_memory(self.yw)

        self.w = cuda.aligned_zeros(shape=(n0,),
                                    dtype=self.real_type,
                                    alignment=resource.getpagesize())
        self.w = cuda.register_host_memory(self.w)
        return self

    def allocate(self, **kwargs):
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        self.nf = kwargs.get('nf', self.nf)
        assert(self.nf is not None)

        self.n = np.int32(self.sigma * self.nf)

        self.nfft_mem_yw.n = self.n
        self.nfft_mem_yw.nf = self.nf
        self.nfft_mem_yw.n0 = self.n0

        self.nfft_mem_w.n = 2 * self.n
        self.nfft_mem_w.nf = 2 * self.nf
        self.nfft_mem_w.n0 = self.n0

        self.allocate_data(**kwargs)
        self.allocate_grids(**kwargs)
        self.allocate_pinned_cpu(**kwargs)

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays(**kwargs)

        return self

    def setdata(self, **kwargs):
        t = kwargs.get('t', self.t)
        yw = kwargs.get('yw', self.yw)
        w = kwargs.get('w', self.w)

        y = kwargs.get('y', None)
        dy = kwargs.get('dy', None)
        self.ybar = self.real_type(0)
        self.yy = self.real_type(kwargs.get('yy', 1.))

        self.n0 = kwargs.get('n0', np.int32(len(t)))

        if dy is not None:
            assert('w' is not in kwargs)
            w = weights(dy)

        if y is not None:
            assert('yw' is not in kwargs)

            self.ybar = self.real_type(np.dot(y, w))
            yw = np.multiply(w, y - self.ybar)
            y2 = np.power(y - self.ybar, 2)
            self.yy = self.real_type(np.dot(w, y2))

        t = np.asarray(t).astype(self.real_type)
        yw = np.asarray(yw).astype(self.real_type)
        w = np.asarray(w).astype(self.real_type)

        if self.buffered_transfer:
            if any([arr is None for arr in [self.t, self.yw, self.w]]):
                if self.buffered_transfer:
                    self.allocate_buffered_data_arrays(**kwargs)

            assert(self.n0 < len(self.tbuff))

            self.t[:self.n0] = t[:self.n0]
            self.yw[:self.n0] = yw[:self.n0]
            self.w[:self.n0] = w[:self.n0]
        else:
            self.t = np.asarray(t).astype(self.real_type)
            self.yw = np.asarray(yw).astype(self.real_type)
            self.w = np.asarray(w).astype(self.real_type)

        # Set minimum and maximum t values (needed to scale things
        # for the NFFT)
        self.tmin = self.real_type(min(t))
        self.tmax = self.real_type(max(t))

        self.nfft_mem_yw.tmin = self.tmin
        self.nfft_mem_w.tmin = self.tmin

        self.nfft_mem_yw.tmax = self.tmax
        self.nfft_mem_w.tmax = self.tmax

        return self

    def transfer_data_to_gpu(self, **kwargs):
        t, yw, w = self.t, self.yw, self.w
        if self.buffered_transfer:
            t, yw, w = self.tbuff, self.ybuff, self.wbuff

        assert(not any([arr is None for arr in [t, yw, w]]))

        # Do asynchronous data transfer
        self.t_g.set_async(t, stream=self.stream)
        self.yw_g.set_async(yw, stream=self.stream)
        self.w_g.set_async(w, stream=self.stream)

    def transfer_lsp_to_cpu(self, **kwargs):
        cuda.memcpy_dtoh_async(self.lsp_c, self.lsp_g.ptr,
                               stream=self.stream)

    def fromdata(self, **kwargs):
        self.setdata(**kwargs)

        if kwargs.get('allocate', True):
            self.allocate(**kwargs)

        return self


def lomb_scargle_direct_sums(t, yw, w, freqs, YY):
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

        CC -= (C * Cw + S * Sw) * (C * Cw + S * Sw)
        SS -= (S * Cw - C * Sw) * (S * Cw - C * Sw)

        P = (YC * YC / CC + YS * YS / SS) / YY

        if (np.isinf(P) or np.isnan(P) or P < 0 or P > 1):
            print(P, YC, YS, CC, SS, YY)
            P = 0.
        lsp[i] = P

    return lsp


def lomb_scargle_async(memory, functions,
                       block_size=256, use_double=False, use_fft=True,
                       minimum_frequency=0., python_dir_sums=False,
                       samples_per_peak=10, transfer_to_device=True,
                       transfer_to_host=True, **kwargs):

    (lomb, lomb_dirsum), nfft_funcs = functions
    df = 1. / (memory.tmax - memory.tmin) / samples_per_peak
    df = real_type(df)
    stream = memory.stream

    block = (block_size, 1, 1)
    grid = (int(np.ceil(nf / float(block_size))), 1)

    # lightcurve -> gpu
    if transfer_to_device:
        memory.transfer_data_to_gpu()

    # Use direct sums (on CPU)
    if python_dir_sums:
        freqs = df * (1 + np.arange(memory.nf))

        t = memory.t_g.get()
        yw = memory.yw_g.get()
        w = memory.w_g.get()

        return lomb_scargle_direct_sums(t, yw, w, freqs, memory.yy)

    # Use direct sums (on GPU)
    if not use_fft:
        args = (grid, block, stream,
                memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr,
                memory.lsp_g.ptr, memory.nf, memory.n0, memory.yy,
                df, df)

        lomb_dirsum.prepared_async_call(*args)
        if transfer_to_device:
            memory.transfer_lsp_to_cpu()
        return memory.lsp_c

    # NFFT
    nfft_kwargs = dict(transfer_to_host=False,
                       transfer_to_device=False,
                       min_freq=minimum_frequency,
                       samples_per_peak=samples_per_peak)

    nfft_kwargs.update(kwargs)

    else:
        # NFFT(w * (y - ybar))
        nfft_adjoint_async(memory.nfft_mem_yw, nfft_funcs, **nfft_kwargs)

        # NFFT(w)
        nfft_adjoint_async(memory.nfft_mem_w, nfft_funcs, **nfft_kwargs)

    args = (grid, block, stream)
    args += (memory.nfft_mem_w.ghat_g.ptr, memory.nfft_mem_yw.ghat_g.ptr)
    args += (memory.lsp_g.ptr, memory.nf, memory.yy)
    lomb.prepared_async_call(*args)

    if transfer_to_host:
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

    def allocate_for_single_lc(self, t, y, dy, nf,
                               stream=None, **kwargs):

        m = self.nfft_proc.estimate_m(nf)

        sigma = self.nfft_proc.sigma

        mem = LombScargleMemory(sigma, stream, m, **kwargs)

        mem.fromdata(t=t, y=y, dy=dy, nf=nf, **kwargs)

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
        allocated_memory: list of ``LombScargleMemory``
            list of allocated memory objects for each lightcurve

        """

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        allocated_memory = []

        nfrqs = nfreqs
        if nfrqs is None:
            nfrqs = [self._nfreqs(t, **kwargs) for (t, y, dy) in data]
        elif not hasattr(nfreqs, '__getitem__'):
            nfrqs = nfrqs * np.ones(len(data))

        for i, ((t, y, dy), nf) in enumerate(zip(data, nfrqs)):
            mem = self.allocate_for_single_lc(t, y, dy, nf,
                                              stream=self.streams[i])
            allocated_memory.append(mem)

        return allocated_memory

    def run(self, data,
            use_fft=True, memory=None,
            convert_to_weights=True, nyquist_factor=5, samples_per_peak=5,
            minimum_frequency=None, maximum_frequency=None,
            use_dy_as_weights=False, **kwargs):

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
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in
                     ['lomb', 'lomb_dirsum']]):
            self._compile_and_prepare_functions(**kwargs)

        freqs = [self.autofrequency(d[0], **kwargs) for d in data]
        nfreqs = [len(frq) for frq in freqs]
        if memory is None:
            memory = self.allocate(data, nfreqs=nfreqs, **kwargs)

        ls_kwargs = dict(block_size=self.block_size,
                         use_double=self.nfft_proc.use_double,
                         use_fft=use_fft)

        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        results = [lomb_scargle_async(mem, funcs, **ls_kwargs)
                   for mem in memory]

        results = [(f, r) for f, r in zip(freqs, results)]
        return results

    def batched_run_const_nfreq(self, data, batch_size=10, df=None,
                                min_samples_per_peak=10., lsps=None,
                                minimum_frequency=0., maximum_frequency=24.,
                                use_dy_as_weights=False, **kwargs):
        """
        Same as ``batched_run`` but is more efficient when the frequencies are
        the same for each lightcurve. Doesn't reallocate memory for each batch.

        Notes
        -----
        To get best efficiency, make sure the maximum number of observations
        is not much larger than the typical number of observations
        """

        bsize = min([len(data), batch_size])
        if bsize < len(self.streams):
            self._create_streams(bsize - len(self.streams))

        streams = [self.streams[i] for i in range(bsize)]
        max_ndata = max([len(t) for t, y, dy in data])

        # set frequencies
        if df is None:
            max_baseline = max([max(t) - min(t) for t, y, dy in data])
            df = 1./(min_samples_per_peak * max_baseline)
        freqs = df * (1 + np.arange(int(maximum_frequency / df) - 1))

        nfreqs = len(freqs)

        m = self.nfft_proc.estimate_m(nfreqs)

        sigma = self.nfft_proc.sigma

        if lsps is None:
            lsps = np.zeros((len(data), nfreqs), dtype=self.real_type)

        # make data batches
        batches = []
        while len(batches) * batch_size < len(data):
            off = len(batches) * batch_size
            end = off + min([batch_size, len(data) - off])
            batches.append([data[i] for i in range(off, end)])

        # allocate gpu and cpu (pinned) memory
        kwargs_lsmem = dict(buffered_transfer=True, n0_buffer=max_ndata)
        kwargs_lsmem.update(kwargs)

        # initialize memory containers
        memory = [LombScargleMemory(sigma, stream, m, **kwargs_lsmem)
                  for stream in streams]

        # allocate memory
        [mem.allocate(nf=nfreqs, **kwargs) for mem in memory]

        # setup keyword args for lomb scargle
        ls_kwargs = dict(block_size=self.block_size,
                         use_double=self.nfft_proc.use_double,
                         use_fft=use_fft)

        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)

        for b, batch in enumerate(batches):
            bs = len(batch)
            for i, (mem, (t, y, dy)) in enumerate(zip(memory, batch)):
                if i >= bs:
                    break

                # Set the data in CPU memory
                mem.setdata(t=t, y=y, dy=dy, **kwargs)

                ls_kwargs['samples_per_peak'] = 1./(df * (max(t) - min(t)))
                lomb_scargle_async(mem, funcs, **ls_kwargs)

            # wait for streams to finish, then transfer pagelocked
            # memory to non-pagelocked memory
            for i, mem in enumerate(memory):
                if i >= bs:
                    break

                mem.stream.synchronize()

                lsps[b * batch_size + i][:] = mem.lsp_c[:]

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
