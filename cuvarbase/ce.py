"""
Implementation of Graham et al. 2013's Conditional Entropy
period finding algorithm
"""
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
#import pycuda.autoinit
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .utils import _module_reader, find_kernel
from .utils import autofrequency as utils_autofreq
from .memory import ConditionalEntropyMemory

import resource
import warnings


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
        dev = pycuda.autoprimaryctx.device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = pycuda.autoprimaryctx.device.get_attribute(att)

    if transfer_to_device:
        memory.transfer_data_to_gpu()

    if freq_batch_size is None:
        freq_batch_size = int(memory.nf)

    block = (block_size, 1, 1)

    # Get the shared memory requirement
    r = memory.real_type(1).nbytes
    u = np.uint32(1).nbytes
    shmem = (r + u) * memory.phase_bins * memory.mag_bins
    shmem += u * memory.phase_bins
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

    # Make sure we have extra memory for alignment
    shmem += shmem % r

    i_freq = 0
    while (i_freq < memory.nf):
        j_freq = min([i_freq + freq_batch_size, memory.nf])

        grid = (min([int(np.ceil((j_freq - i_freq) / block_size)),
                     max_nblocks]), 1)
        if data_in_shared_mem:
            grid = (int(np.floor(2 * float(shmem_lim) / shmem)), 1)
        if force_nblocks is not None:
            grid = (force_nblocks, 1)

        assert(grid[0] > 0)

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

    .. warning::
        This implementation uses a less efficient algorithm than the ``gce``
        package (Katz et al. 2020). For production CE searches, especially
        those including period derivatives, we recommend using ``gce`` instead:
        https://github.com/mikekatz04/gce

        This implementation remains available for backward compatibility and
        simple exploratory analysis.

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

    Notes
    -----
    Performance Considerations:

    This implementation allocates one GPU block per frequency and stores the
    full 2D (phase × magnitude) histogram in shared memory. This approach has
    two limitations:

    1. Memory: Uses ~n_mag more memory per frequency than optimal
    2. Parallelism: Limited to one block per frequency (lower GPU utilization)

    The ``gce`` package implements a more efficient algorithm that uses one
    thread per frequency and processes magnitude bins sequentially, requiring
    only 1D histograms. This enables:

    - Period derivative (Pdot) searches in tractable time
    - Better GPU utilization even for simple period searches
    - ~1/n_mag less memory usage

    References
    ----------
    Katz, M. L., Larson, S. L., Cohn, J., Vallisneri, M., & Graff, P. B. (2020).
    "Efficient computation of the Conditional Entropy period search."
    arXiv:2006.06866

    Example
    -------
    >>> proc = ConditionalEntropyAsyncProcess()
    >>> Ndata = 1000
    >>> t = np.sort(365 * np.random.rand(N))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    >>> y += 0.01 * np.random.randn(len(t))
    >>> dy = 0.01 * np.ones_like(y)
    >>> results = proc.run([(t, y, dy)])
    >>> proc.finish()
    >>> ce_freqs, ce_powers = results[0]

    """
    def __init__(self, *args, **kwargs):
        super(ConditionalEntropyAsyncProcess, self).__init__(*args, **kwargs)
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 5)
        self.max_phi = kwargs.get('max_phi', 3.)
        self.weighted = kwargs.get('weighted', False)
        self.block_size = kwargs.get('block_size', 256)

        self.phase_overlap = kwargs.get('phase_overlap', 0)
        self.mag_overlap = kwargs.get('mag_overlap', 0)

        if self.mag_overlap > 0:
            if kwargs.get('balanced_magbins', False):
                raise Exception("mag_overlap must be zero "
                                "if balanced_magbins is True")

        self.use_double = kwargs.get('use_double', False)

        self.real_type = np.float32
        if self.use_double:
            self.real_type = np.float64

        self.call_func = conditional_entropy
        if kwargs.get('use_fast', False):
            self.call_func = conditional_entropy_fast

        self.memory = kwargs.get('memory', None)
        self.shmem_lc = kwargs.get('shmem_lc', True)

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
        for fname, dtype in self.dtypes.items():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))

    def memory_requirement(self, data, **kwargs):
        """
        Return an approximate GPU memory requirement in bytes.
        Will throw a ``NotImplementedError`` if called, so ... don't call it.
        """
        raise NotImplementedError()

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
        mem: ConditionalEntropyMemory
            Memory object.
        """

        kw = dict(phase_bins=self.phase_bins,
                  mag_bins=self.mag_bins,
                  mag_overlap=self.mag_overlap,
                  phase_overlap=self.phase_overlap,
                  max_phi=self.max_phi,
                  stream=stream,
                  weighted=self.weighted,
                  use_double=self.use_double)

        kw.update(kwargs)
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

        elif isinstance(freqs[0], float):
            frqs = [freqs] * len(data)

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
            Length of list must be ``>= nlcs``

        Returns
        -------
        self.memory: list
            List of ``ConditionalEntropyMemory`` objects
        """
        kw = dict(phase_bins=self.phase_bins,
                  mag_bins=self.mag_bins,
                  mag_overlap=self.mag_overlap,
                  phase_overlap=self.phase_overlap,
                  max_phi=self.max_phi,
                  weighted=self.weighted,
                  use_double=self.use_double,
                  n0_buffer=max_nobs,
                  buffered_transfer=True,
                  allocate=True,
                  freqs=freqs)

        kw.update(kwargs)

        self.memory = []
        for i in range(nlcs):
            stream = None if streams is None else streams[i]
            kw.update(dict(stream=stream))
            mem = ConditionalEntropyMemory(**kw)
            mem.allocate(**kwargs)
            self.memory.append(mem)

        return self.memory

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
            the ``data`` array

        """
        # compile module if not compiled already
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in
                     ['ce_wt']]):
            self._compile_and_prepare_functions(**kwargs)

        # create and/or check frequencies
        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(d[0], **kwargs) for d in data]

        elif isinstance(frqs[0], float):
            frqs = [frqs] * len(data)

        assert(len(frqs) == len(data))

        memory = memory if memory is not None else self.memory

        if memory is None:
            memory = self.allocate(data, freqs=frqs,
                                   **kwargs)
            for mem in memory:
                mem.transfer_freqs_to_gpu()
        elif set_data:
            for i, (t, y, dy) in enumerate(data):
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
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in
                     ['ce_wt']]):
            self._compile_and_prepare_functions(**kwargs)

        if max_memory is None:
            free, total = cuda.mem_get_info()
            max_memory = 0.9 * free

        # create and/or check frequencies
        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(d[0], **kwargs) for d in data]

        elif isinstance(frqs[0], float):
            frqs = [frqs] * len(data)

        assert(len(frqs) == len(data))

        cpers = []
        for d, f in zip(data, frqs):
            size_of_real = self.real_type(1).nbytes

            # subtract of lc memory
            fmem = max_memory - len(d[0]) * size_of_real * 3

            tot_bins = self.phase_bins * self.mag_bins
            batch_size = int(np.floor(fmem / (size_of_real * (tot_bins + 2))))
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
                                         key=lambda d: max(d[0]) - min(d[0]))
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
        kwargs_mem = dict(buffered_transfer=True,
                          n0_buffer=max_ndata,
                          mag_overlap=self.mag_overlap,
                          phase_overlap=self.phase_overlap,
                          phase_bins=self.phase_bins,
                          mag_bins=self.mag_bins,
                          weighted=self.weighted,
                          max_phi=self.max_phi,
                          use_double=self.use_double)
        kwargs_mem.update(kwargs)
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
