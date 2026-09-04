"""
Implementation of Graham et al. 2013's Conditional Entropy
period finding algorithm

.. note:: **Maintenance status.** cuvarbase's conditional entropy
    implementation is in maintenance mode: it works and will keep
    working, but no further performance or feature development is
    planned. For new projects that need a fast GPU conditional-entropy
    (or AOV) search, consider `periodfind
    <https://github.com/scope-ml/periodfind>`_ (also on PyPI as
    ``periodfind``), an actively maintained GPU period-finding
    package developed for ZTF/SCoPe.
"""
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess, ensure_context
from .utils import _module_reader, find_kernel, normalize_light_curves
from .utils import autofrequency as utils_autofreq
from .memory import ConditionalEntropyMemory

import resource
import warnings


# Every kernel the CE module compiles, in the (sorted) order in which
# ``ConditionalEntropyAsyncProcess.function_tuple`` is unpacked by
# :func:`conditional_entropy` / :func:`conditional_entropy_fast`.
_CE_KERNELS = ('ce_classical_fast', 'ce_classical_faster', 'constdpdm_ce',
               'histogram_data_count', 'histogram_data_weighted',
               'log_prob', 'standard_ce', 'weighted_ce')


def _needs_compile(prepared_functions):
    """True unless every CE kernel has already been compiled and prepared.

    (The previous gate looked for a key ``'ce_wt'`` that no compile ever
    produced, so the module was rebuilt with nvcc on every call.)
    """
    if not prepared_functions:
        return True
    return not all(name in prepared_functions for name in _CE_KERNELS)


def _is_single_freq_grid(freqs):
    """True if ``freqs`` is one 1-D grid (to be shared by every lightcurve)
    rather than a sequence of per-lightcurve grids.

    Accepts any 1-D numeric array or list (float32, float64, integers,
    Python floats); previously only Python/np.float64 scalars were
    recognized, so a float32 grid was mistaken for a list of grids.
    """
    if isinstance(freqs, np.ndarray):
        return freqs.ndim == 1
    if len(freqs) == 0:
        return True
    return isinstance(freqs[0], (float, int, np.floating, np.integer))


def _freq_grids(freqs, nlcs):
    """Expand ``freqs`` into a list of ``nlcs`` per-lightcurve grids."""
    if _is_single_freq_grid(freqs):
        return [freqs] * nlcs
    return list(freqs)


def conditional_entropy(memory, functions, block_size=256,
                        transfer_to_host=True,
                        transfer_to_device=True,
                        **kwargs):
    block = (block_size, 1, 1)
    grid = (int(np.ceil((memory.n0 * memory.nf) / float(block_size))), 1)
    fast_ce, faster_ce, ce_dpdm, hist_count, hist_weight,\
        ce_logp, ce_std, ce_wt = functions

    if transfer_to_device:
        memory.transfer_data_to_gpu()

    # The histogram kernels accumulate into ``bins_g``: it must start from
    # zero on EVERY call, not only when ``run(set_data=True)`` zeroed it
    # (``run(memory=..., set_data=False)`` used to accumulate counts
    # across calls).
    memory.bins_g.fill(memory.bins_g.dtype.type(0), stream=memory.stream)

    if memory.weighted:
        args = (grid, block, memory.stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.dy_g.ptr)
        args += (memory.bins_g.ptr, memory.freqs_g.ptr)
        args += (np.uint32(memory.nf), np.uint32(memory.n0))
        args += (memory.real_type(memory.max_phi),)
        hist_weight.prepared_async_call(*args)

        grid = (int(np.ceil(memory.nf / float(block_size))), 1)

        args = (grid, block, memory.stream)
        args += (memory.bins_g.ptr, np.uint32(memory.nf), memory.ce_g.ptr)
        ce_wt.prepared_async_call(*args)

        if transfer_to_host:
            memory.transfer_ce_to_cpu()
        return memory.ce_c

    args = (grid, block, memory.stream)
    args += (memory.t_g.ptr, memory.y_g.ptr)
    args += (memory.bins_g.ptr, memory.freqs_g.ptr)
    args += (np.uint32(memory.nf), np.uint32(memory.n0))
    hist_count.prepared_async_call(*args)

    grid = (int(np.ceil(memory.nf / float(block_size))), 1)
    args = (grid, block, memory.stream)
    args += (memory.bins_g.ptr, np.uint32(memory.nf), memory.ce_g.ptr)

    if memory.balanced_magbins:
        args += (memory.mag_bwf_g.ptr,)
        ce_dpdm.prepared_async_call(*args)
    elif memory.compute_log_prob:
        args += (memory.mag_bin_fracs_g.ptr,)
        ce_logp.prepared_async_call(*args)
    else:
        ce_std.prepared_async_call(*args)

    if transfer_to_host:
        memory.transfer_ce_to_cpu()

    return memory.ce_c


def conditional_entropy_fast(memory, functions, block_size=256,
                             transfer_to_host=True,
                             transfer_to_device=True,
                             freq_batch_size=None,
                             shmem_lc=True,
                             shmem_lim=None,
                             max_nblocks=200,
                             force_nblocks=None,
                             stream=None,
                             **kwargs):
    fast_ce, faster_ce, ce_dpdm, hist_count, hist_weight,\
        ce_logp, ce_std, ce_wt = functions

    if shmem_lim is None:
        dev = ensure_context().device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = dev.get_attribute(att)

    if stream is None:
        # launch on the memory's own stream so the data upload, the
        # kernel and the result download are ordered and ``finish()``
        # (which synchronizes the process streams) covers all of them
        stream = memory.stream

    if transfer_to_device:
        memory.transfer_data_to_gpu()

    if freq_batch_size is None:
        freq_batch_size = int(memory.nf)

    block = (block_size, 1, 1)

    # Shared memory layout (must match ce_classical_fast/faster):
    #   block_bin[nmag * nphase] (uint32) | block_bin_phi[nphase] (uint32)
    #   | pad to sizeof(FLT) | Hc[nmag * nphase] (FLT)
    #   | t_sh[ndata] (FLT) | y_sh[ndata] (uint32)      (faster only)
    r = memory.real_type(1).nbytes
    u = np.uint32(1).nbytes
    shmem = (r + u) * memory.phase_bins * memory.mag_bins
    shmem += u * memory.phase_bins
    # The alignment pad sits between the uint32 histograms and Hc, so it
    # has to be added BEFORE the (optional) lightcurve block: computing
    # it after adding ``data_mem`` made it depend on the parity of ndata
    # and under-allocated by 4 bytes for odd ndata in double precision.
    shmem += (-shmem) % r
    data_mem = (r + u) * len(memory.t)

    func = fast_ce

    # Decide whether or not to use shared memory for
    # loading the lightcurve. Only if the user
    # wants and we have enough memory
    data_in_shared_mem = False
    if shmem_lc:
        data_in_shared_mem = shmem + data_mem < shmem_lim

    if data_in_shared_mem:
        shmem += data_mem
        func = faster_ce

    i_freq = 0
    while (i_freq < memory.nf):
        j_freq = min([i_freq + freq_batch_size, memory.nf])

        grid = (min([int(np.ceil((j_freq - i_freq) / block_size)),
                     max_nblocks]), 1)
        if data_in_shared_mem:
            grid = (int(np.floor(2 * float(shmem_lim) / shmem)), 1)
        if force_nblocks is not None:
            grid = (force_nblocks, 1)

        if not grid[0] > 0:
            raise RuntimeError(
                "computed CUDA grid size is 0: the shared-memory limit is "
                "too small for this configuration")

        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.y_g.ptr)
        args += (memory.freqs_g.ptr, memory.ce_g.ptr)
        args += (np.uint32(j_freq - i_freq), np.uint32(i_freq),
                 np.uint32(memory.n0))
        args += (np.uint32(memory.phase_bins), np.uint32(memory.mag_bins))
        args += (np.uint32(memory.phase_overlap),
                 np.uint32(memory.mag_overlap))

        func.prepared_async_call(*args, shared_size=shmem)

        i_freq += j_freq - i_freq

    if transfer_to_host:
        memory.transfer_ce_to_cpu()

    return memory.ce_c


class ConditionalEntropyAsyncProcess(GPUAsyncProcess):
    """
    GPUAsyncProcess for the Conditional Entropy period finder

    Parameters
    ----------
    phase_bins: int, optional (default: 10)
        Number of phase bins to use.
    mag_bins: int, optional (default: 10)
        Number of mag bins to use.
    max_phi: float, optional (default: 3.)
        For weighted CE; skips contibutions to bins that are more than
        ``max_phi`` sigma away.
    weighted: bool, optional (default: False)
        If true, uses the weighted version of the CE periodogram. Slower, but
        accounts for data uncertainties.
    block_size: int, optional (default: 256)
        Number of CUDA threads per CUDA block.
    phase_overlap: int, optional (default: 0)
        If > 0, the phase bins are overlapped with each other
    mag_overlap: int, optional (default: 0)
        If > 0, the mag bins are overlapped with each other
    use_fast: bool, optional (default: False)
        Use a somewhat experimental function to speed up
        computations. This is perfect for large Nfreqs and nobs <~ 2000.
        If True, use :func:`run` and not :func:`large_run` and set
        ``nstreams = 1``.
    compute_log_prob: bool, optional (default: False)
        Instead of computing CE, compute and return the log-probability periodogram.

    Example
    -------
    >>> proc = ConditionalEntropyAsyncProcess()
    >>> Ndata = 1000
    >>> t = np.sort(365 * np.random.rand(Ndata))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    >>> y += 0.01 * np.random.randn(len(t))
    >>> dy = 0.01 * np.ones_like(y)
    >>> results = proc.run([(t, y, dy)])
    >>> proc.finish()
    >>> ce_freqs, ce_powers = results[0]

    """
    def __init__(self, *args, **kwargs):
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 5)
        self.max_phi = kwargs.get('max_phi', 3.)
        self.weighted = kwargs.get('weighted', False)
        self.block_size = kwargs.get('block_size', 256)
        self.compute_log_prob = kwargs.get('compute_log_prob', False)

        self.phase_overlap = kwargs.get('phase_overlap', 0)
        self.mag_overlap = kwargs.get('mag_overlap', 0)

        self.balanced_magbins = kwargs.get('balanced_magbins', False)
        self.widen_mag_range = kwargs.get('widen_mag_range', False)
        self.use_fast = kwargs.get('use_fast', False)
        self.use_double = kwargs.get('use_double', False)

        # Reject unsupported option combinations before touching the GPU
        self._check_options(dict(weighted=self.weighted,
                                 balanced_magbins=self.balanced_magbins,
                                 compute_log_prob=self.compute_log_prob,
                                 mag_overlap=self.mag_overlap),
                            use_fast=self.use_fast)

        super(ConditionalEntropyAsyncProcess, self).__init__(*args, **kwargs)

        self.real_type = np.float32
        if self.use_double:
            self.real_type = np.float64

        self.call_func = conditional_entropy
        if self.use_fast:
            self.call_func = conditional_entropy_fast

        self.memory = kwargs.get('memory', None)
        self.shmem_lc = kwargs.get('shmem_lc', True)

    @staticmethod
    def _check_options(opts, use_fast=False):
        """
        Raise ``ValueError`` for option combinations that have no
        implementation (see ``docs/source/ce.rst``).

        Parameters
        ----------
        opts: dict
            Memory options (``weighted``, ``balanced_magbins``,
            ``compute_log_prob``, ``mag_overlap``); missing keys are
            treated as their defaults.
        use_fast: bool
            Whether the shared-memory kernels are in use.
        """
        weighted = opts.get('weighted', False)
        balanced = opts.get('balanced_magbins', False)
        log_prob = opts.get('compute_log_prob', False)
        mag_overlap = opts.get('mag_overlap', 0)

        if weighted and use_fast:
            raise ValueError("use_fast must be False if weighted is True")
        if weighted and balanced:
            raise ValueError("simultaneous balanced_magbins and weighted"
                             " options is not currently supported")
        if weighted and log_prob:
            raise ValueError("simultaneous compute_log_prob and weighted"
                             " options is not currently supported")
        if balanced and use_fast:
            raise ValueError("use_fast must be False if balanced_magbins "
                             "is True (the fast kernels only implement "
                             "uniform magnitude bins)")
        if balanced and log_prob:
            raise ValueError("simultaneous balanced_magbins and "
                             "compute_log_prob options is not currently "
                             "supported")
        if balanced and mag_overlap > 0:
            raise ValueError("mag_overlap must be zero "
                             "if balanced_magbins is True")

    def _memory_kwargs(self, **overrides):
        """
        Build the keyword arguments for ``ConditionalEntropyMemory`` from
        the process settings, apply ``overrides`` (per-call kwargs) and
        validate the resulting option combination.
        """
        kw = dict(phase_bins=self.phase_bins,
                  mag_bins=self.mag_bins,
                  mag_overlap=self.mag_overlap,
                  phase_overlap=self.phase_overlap,
                  max_phi=self.max_phi,
                  weighted=self.weighted,
                  use_double=self.use_double,
                  compute_log_prob=self.compute_log_prob,
                  balanced_magbins=self.balanced_magbins,
                  widen_mag_range=self.widen_mag_range)
        kw.update(overrides)
        self._check_options(kw, use_fast=self.use_fast)
        return kw

    def _ensure_compiled(self, **kwargs):
        """Compile and prepare the kernels once per process object."""
        if _needs_compile(getattr(self, 'prepared_functions', None)):
            self._compile_and_prepare_functions(**kwargs)

    def _compile_and_prepare_functions(self, **kwargs):

        cpp_defs = dict(NPHASE=self.phase_bins,
                        NMAG=self.mag_bins,
                        PHASE_OVERLAP=self.phase_overlap,
                        MAG_OVERLAP=self.mag_overlap)

        if self.use_double:
            cpp_defs['DOUBLE_PRECISION'] = None

        # Read kernel & replace with
        kernel_txt = _module_reader(find_kernel('ce'),
                                    cpp_defs=cpp_defs)

        # compile kernel
        self.module = SourceModule(kernel_txt, options=['--use_fast_math'])

        self.dtypes = dict(
            constdpdm_ce=[np.intp, np.int32, np.intp, np.intp],
            histogram_data_weighted=[np.intp, np.intp, np.intp, np.intp,
                                     np.intp, np.uint32, np.uint32,
                                     self.real_type],
            histogram_data_count=[np.intp, np.intp, np.intp, np.intp,
                                  np.uint32, np.uint32],
            log_prob=[np.intp, np.uint32, np.intp, np.intp],
            standard_ce=[np.intp, np.uint32, np.intp],
            weighted_ce=[np.intp, np.uint32, np.intp],
            ce_classical_fast=[np.intp, np.intp, np.intp,
                               np.intp, np.uint32,
                               np.uint32, np.uint32, np.uint32,
                               np.uint32, np.uint32, np.uint32],
            ce_classical_faster=[np.intp, np.intp, np.intp,
                                 np.intp, np.uint32,
                                 np.uint32, np.uint32, np.uint32,
                                 np.uint32, np.uint32, np.uint32]
        )
        if tuple(sorted(self.dtypes.keys())) != _CE_KERNELS:
            raise RuntimeError("CE kernel table does not match _CE_KERNELS")
        for fname, dtype in self.dtypes.items():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in _CE_KERNELS)

    def memory_requirement(self, n0, nf, **kwargs):
        """
        Return an approximate GPU memory requirement in bytes for one
        lightcurve with ``n0`` observations and ``nf`` trial
        frequencies.

        The histogram dominates: ``nf * phase_bins * mag_bins``
        entries (uint32, or ``real_type`` when ``weighted=True``).

        Parameters
        ----------
        n0: int
            Number of observations.
        nf: int
            Number of trial frequencies.

        Returns
        -------
        mem: int
            Approximate bytes of GPU memory required.
        """
        rsize = np.dtype(self.real_type).itemsize
        bin_size = rsize if self.weighted else np.dtype(np.uint32).itemsize

        # histogram bins
        mem = nf * self.phase_bins * self.mag_bins * bin_size
        # observation data: t, y (+ dy when weighted)
        mem += (3 if self.weighted else 2) * n0 * rsize
        # frequencies + CE result
        mem += 2 * nf * rsize

        return int(mem)

    def allocate_for_single_lc(self, t, y, freqs, dy=None,
                               stream=None, **kwargs):
        """
        Allocate GPU (and possibly CPU) memory for single lightcurve

        Parameters
        ----------
        t: array_like
            Observation times
        y: array_like
            Observations
        freqs: array_like
            frequencies
        dy: array_like, optional
            Observation uncertainties
        stream: pycuda.driver.Stream, optional
            CUDA stream you want this to run on
        **kwargs

        Returns
        -------
        mem: ~cuvarbase.memory.ce_memory.ConditionalEntropyMemory
            Memory object.
        """

        kw = self._memory_kwargs(**kwargs)
        kw['stream'] = stream
        mem = ConditionalEntropyMemory(**kw)

        mem.fromdata(t, y, dy=dy, freqs=freqs, allocate=True, **kwargs)

        return mem

    def autofrequency(self, *args, **kwargs):
        """ calls :func:`cuvarbase.utils.autofrequency` """
        return utils_autofreq(*args, **kwargs)

    def _nfreqs(self, *args, **kwargs):
        return len(self.autofrequency(*args, **kwargs))

    def allocate(self, data, freqs=None, **kwargs):
        """
        Allocate GPU memory for Conditional Entropy computations

        Parameters
        ----------
        data: list of (t, y, dy) tuples
            List of data, ``[(t_1, y_1, w_1), ...]``
            * ``t``: Observation times
            * ``y``: Observations
            * ``dy``: Observation uncertainties
        freqs: list, optional
            Either a list of floats (same frequencies for all data),
            or a list of length ``n=len(data)``, with element ``i`` of the
            list being a list of frequencies for the ``i``-th lightcurve.
        **kwargs

        Returns
        -------
        allocated_memory: list of ``ConditionalEntropyMemory``
            list of allocated memory objects for each lightcurve

        """

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        allocated_memory = []

        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(t, **kwargs) for (t, y, dy) in data]
        else:
            frqs = _freq_grids(freqs, len(data))

        for i, ((t, y, dy), f) in enumerate(zip(data, frqs)):
            mem = self.allocate_for_single_lc(t, y, dy=dy, freqs=f,
                                              stream=self.streams[i],
                                              **kwargs)
            allocated_memory.append(mem)

        return allocated_memory

    def preallocate(self, max_nobs, freqs,
                    nlcs=1, streams=None, **kwargs):
        """
        Preallocate memory for future runs.

        Parameters
        ----------
        max_nobs: int
            Upper limit for the number of observations
        freqs: array_like
            Frequency array to be used by future ``run`` calls
        nlcs: int, optional (default: 1)
            Maximum batch size for ``run`` calls
        streams: list of ``pycuda.driver.Stream``
            Length of list must be ``>= nlcs``; defaults to the process
            streams (created as needed)

        Returns
        -------
        self.memory: list
            List of ``ConditionalEntropyMemory`` objects
        """
        overrides = dict(n0_buffer=max_nobs,
                         buffered_transfer=True,
                         allocate=True,
                         freqs=freqs)
        overrides.update(kwargs)
        kw = self._memory_kwargs(**overrides)

        if streams is None:
            if len(self.streams) < nlcs:
                self._create_streams(nlcs - len(self.streams))
            streams = self.streams
        elif len(streams) < nlcs:
            raise ValueError("preallocate: %d streams given for nlcs=%d"
                             % (len(streams), nlcs))

        self.memory = []
        for i in range(nlcs):
            kw.update(dict(stream=streams[i]))
            mem = ConditionalEntropyMemory(**kw)
            mem.allocate(**kwargs)
            mem.transfer_freqs_to_gpu()
            self.memory.append(mem)

        return self.memory

    @staticmethod
    def _sync_memory_freqs(mem, freqs):
        """
        Make sure the frequency grid held by (and uploaded to) ``mem``
        is ``freqs``; re-upload when a ``run`` call passes a grid that
        differs from the one the memory was allocated with.
        """
        f = np.asarray(freqs, dtype=mem.real_type)
        if mem.nf is not None and len(f) != mem.nf:
            raise ValueError(
                "memory was allocated for %d frequencies but the call "
                "passes %d; allocate (or preallocate) the memory for the "
                "new grid" % (mem.nf, len(f)))
        if mem.freqs is None or not np.array_equal(mem.freqs, f):
            mem.freqs = f
            mem.transfer_freqs_to_gpu()

    def run(self, data,
            memory=None,
            freqs=None,
            set_data=True,
            **kwargs):

        """
        Run Conditional Entropy on a batch of data.

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, dy), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``dy``: observation uncertainties
        freqs: optional, list of ``np.ndarray`` frequencies
            List of custom frequencies. If not specified, calls
            ``autofrequency`` with default arguments
        memory: optional, list of ``ConditionalEntropyMemory`` objects
            List of memory objects, length of list must be ``>= len(data)``
        set_data: boolean, optional (default: True)
            Transfers data to gpu if memory is provided
        **kwargs

        Returns
        -------
        results: list of lists
            list of (freqs, ce) corresponding to CE for each element of
            the ``data`` array; the ce arrays are page-locked host
            buffers filled asynchronously — call :meth:`finish` before
            reading them (the batched entry points synchronize for you)

        """
        # compile module if not compiled already
        self._ensure_compiled(**kwargs)

        # Prepare data
        data = normalize_light_curves(data)

        # create and/or check frequencies
        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(d[0], **kwargs) for d in data]
        else:
            frqs = _freq_grids(freqs, len(data))

        if len(frqs) != len(data):
            raise ValueError(
                "number of frequency grids (%d) does not match number of "
            "lightcurves (%d)" % (len(frqs), len(data)))

        if not self.use_fast:
            for f, d in zip(frqs, data):
                if len(f) * len(d[0]) > 2**32-1:
                    raise OverflowError(
                        "Number of streams is too large - overflowing 32 bit integers\n"
                        "Decrease frequency range or use :func:`large_run` instead")

        memory = memory if memory is not None else self.memory

        if memory is None:
            memory = self.allocate(data, freqs=frqs,
                                   **kwargs)
            for mem in memory:
                mem.transfer_freqs_to_gpu()
        else:
            if len(memory) < len(data):
                raise ValueError(
                    "%d memory objects for %d lightcurves; preallocate "
                    "with nlcs >= the batch size" % (len(memory), len(data)))
            for i, (t, y, dy) in enumerate(data):
                self._sync_memory_freqs(memory[i], frqs[i])
                if set_data:
                    memory[i].set_gpu_arrays_to_zero(**kwargs)
                    memory[i].setdata(t, y, dy=dy, **kwargs)

        kw = dict(block_size=self.block_size,
                  shmem_lc=self.shmem_lc)
        kw.update(kwargs)
        results = [self.call_func(memory[i], self.function_tuple, **kw)
                   for i in range(len(data))]

        results = [(f, r) for f, r in zip(frqs, results)]
        return results

    def large_run(self, data,
                  freqs=None,
                  max_memory=None,
                  **kwargs):
        """
        Run Conditional Entropy on a large frequency grid

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, dy), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``dy``: observation uncertainties
        freqs: optional, list of ``np.ndarray`` frequencies
            List of custom frequencies. If not specified, calls
            ``autofrequency`` with default arguments
        max_memory: float, optional (default: None)
            Maximum memory per batch in bytes. If ``None``, it
            will use 90% of the total free memory available as specified by
            ``pycuda.driver.mem_get_info()``
        **kwargs

        Returns
        -------
        results: list of lists
            list of (freqs, ce) corresponding to CE for each element of
            the ``data`` array

        """

        # compile module if not compiled already
        self._ensure_compiled(**kwargs)

        if max_memory is None:
            free, total = cuda.mem_get_info()
            max_memory = 0.9 * free

        # create and/or check frequencies
        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(d[0], **kwargs) for d in data]
        else:
            frqs = _freq_grids(freqs, len(data))

        if len(frqs) != len(data):
            raise ValueError(
                "number of frequency grids (%d) does not match number of "
            "lightcurves (%d)" % (len(frqs), len(data)))

        cpers = []
        for d, f in zip(data, frqs):
            # Limit frequencies to ensure that
            # thread numbers are within the limits of single-precision
            max_threads_per_launch = 2**32 - 1
            total_threads = len(d[0]) * len(f)
            thread_nbatches = int(np.ceil(total_threads/max_threads_per_launch))

            size_of_real = self.real_type(1).nbytes

            # subtract of lc memory
            fmem = max_memory - len(d[0]) * size_of_real * 3

            tot_bins = self.phase_bins * self.mag_bins
            batch_size = int(np.floor(fmem / (size_of_real * (tot_bins + 2))))
            nbatches = int(np.ceil(len(f) / float(batch_size)))

            if thread_nbatches > nbatches:
                # Cap the batch size by the thread limit directly:
                # ceil(len(f) / thread_nbatches) can overshoot
                # max_threads_per_launch by up to len(d[0]) - 1 threads,
                # which would trip the overflow guard in run().
                batch_size = max(1, max_threads_per_launch // len(d[0]))
                nbatches = int(np.ceil(len(f) / float(batch_size)))

            cper = np.zeros(len(f))
            for i in range(nbatches):
                imin = i * batch_size
                imax = min([len(f), (i + 1) * batch_size])

                r = self.run([d], freqs=f[slice(imin, imax)], **kwargs)
                self.finish()

                cper[imin:imax] = r[0][1][:]

            cpers.append(cper)

        results = [(f, cper) for f, cper in zip(frqs, cpers)]
        return results

    def batched_run_const_nfreq(self, data, batch_size=10,
                                freqs=None,only_return_best_freqs=False,
                                **kwargs):
        """
        Same as ``batched_run`` but is more efficient when the frequencies are
        the same for each lightcurve. Doesn't reallocate memory for each batch.

        .. note::

            To get best efficiency, make sure the maximum number of
            observations is not much larger than the typical number
            of observations.
        """

        # create streams if needed
        bsize = min([len(data), batch_size])
        if len(self.streams) < bsize:
            self._create_streams(bsize - len(self.streams))

        streams = [self.streams[i] for i in range(bsize)]
        max_ndata = max([len(t) for t, y, dy in data])

        if freqs is None:
            data_with_max_baseline = max(data,
                                         key=lambda d: np.max(d[0]) - np.min(d[0]))
            freqs = self.autofrequency(data_with_max_baseline[0], **kwargs)

        df = freqs[1] - freqs[0]
        nf = len(freqs)

        ces = []

        # make data batches
        batches = []
        while len(batches) * batch_size < len(data):
            start = len(batches) * batch_size
            finish = start + min([batch_size, len(data) - start])
            batches.append([data[i] for i in range(start, finish)])

        # set up memory containers for gpu and cpu (pinned) memory
        overrides = dict(buffered_transfer=True,
                         n0_buffer=max_ndata)
        overrides.update(kwargs)
        kwargs_mem = self._memory_kwargs(**overrides)
        memory = [ConditionalEntropyMemory(stream=stream, **kwargs_mem)
                  for stream in streams]

        # allocate memory
        [mem.allocate(freqs=freqs, **kwargs) for mem in memory]

        [mem.transfer_freqs_to_gpu(**kwargs) for mem in memory]

        best_freqs, best_freq_significances = [], []

        for b, batch in enumerate(batches):
            results = self.run(batch, memory=memory, freqs=freqs, **kwargs)
            self.finish()

            for i, (f, ce) in enumerate(results):
                ce = np.copy(ce)
                significance = np.abs(np.mean(ce)-np.min(ce))/np.std(ce)
                if only_return_best_freqs:
                    best_freqs.append(freqs[np.argmin(ce)])
                    best_freq_significances.append(significance)
                else:
                    ces.append(ce)

        if only_return_best_freqs:
            return best_freqs, best_freq_significances
        else:
            return [(freqs, ce) for ce in ces]
