"""
Implementation of Graham et al. 2013's Conditional Entropy
period finding algorithm
"""

from .core import GPUAsyncProcess
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from .utils import _module_reader, find_kernel
from .utils import autofrequency as utils_autofreq
from time import time
import resource


class ConditionalEntropyMemory(object):
    def __init__(self, **kwargs):
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 10)
        self.max_phi = kwargs.get('max_phi', 3.)
        self.stream = kwargs.get('stream', None)
        self.weighted = kwargs.get('weighted', False)
        self.widen_mag_range = kwargs.get('widen_mag_range', False)
        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)

        self.n0_buffer = kwargs.get('n0_buffer', None)
        self.buffered_transfer = kwargs.get('buffered_transfer', False)
        self.t = None
        self.y = None
        self.dy = None

        self.t_g = None
        self.y_g = None
        self.dy_g = None

        self.bins_g = None
        self.ce_c = None
        self.ce_g = None
        self.real_type = np.float32

        self.freqs = None
        self.freqs_g = None

        self.ytype = np.uint32 if not self.weighted else self.real_type

    def allocate_buffered_data_arrays(self, **kwargs):
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        self.t = cuda.aligned_zeros(shape=(n0,),
                                    dtype=self.real_type,
                                    alignment=resource.getpagesize())
        self.t = cuda.register_host_memory(self.t)

        self.y = cuda.aligned_zeros(shape=(n0,),
                                    dtype=self.ytype,
                                    alignment=resource.getpagesize())

        self.y = cuda.register_host_memory(self.y)
        if self.weighted:
            self.dy = cuda.aligned_zeros(shape=(n0,),
                                         dtype=self.real_type,
                                         alignment=resource.getpagesize())
            self.dy = cuda.register_host_memory(self.dy)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)

        self.ce_c = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
                                       alignment=resource.getpagesize())
        self.ce_c = cuda.register_host_memory(self.ce_c)

        return self

    def allocate_data(self, **kwargs):
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)

        assert(n0 is not None)
        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(n0, dtype=self.ytype)
        if self.weighted:
            self.dy_g = gpuarray.zeros(n0, dtype=self.real_type)

    def allocate_bins(self, **kwargs):
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)

        self.nbins = nf * self.phase_bins * self.mag_bins

        if self.weighted:
            self.bins_g = gpuarray.zeros(self.nbins, dtype=self.real_type)
        else:
            self.bins_g = gpuarray.zeros(self.nbins, dtype=np.uint32)

    def allocate_freqs(self, **kwargs):
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)
        self.freqs_g = gpuarray.zeros(nf, dtype=self.real_type)
        if self.ce_g is None:
            self.ce_g = gpuarray.zeros(nf, dtype=self.real_type)

    def allocate(self, **kwargs):
        self.freqs = kwargs.get('freqs', self.freqs)
        self.nf = kwargs.get('nf', len(self.freqs))

        if self.freqs is not None:
            self.freqs = np.asarray(self.freqs).astype(self.real_type)

        assert(self.nf is not None)

        self.allocate_data(**kwargs)
        self.allocate_bins(**kwargs)
        self.allocate_freqs(**kwargs)
        self.allocate_pinned_cpu(**kwargs)

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays(**kwargs)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        assert(not any([x is None for x in [self.t, self.y]]))

        self.t_g.set_async(self.t, stream=self.stream)
        self.y_g.set_async(self.y, stream=self.stream)

        if self.weighted:
            assert(self.dy is not None)
            self.dy_g.set_async(self.dy, stream=self.stream)

    def transfer_freqs_to_gpu(self, **kwargs):
        freqs = kwargs.get('freqs', self.freqs)
        assert(freqs is not None)

        self.freqs_g.set_async(freqs, stream=self.stream)

    def transfer_ce_to_cpu(self, **kwargs):
        cuda.memcpy_dtoh_async(self.ce_c, self.ce_g.ptr, stream=self.stream)

    def setdata(self, t, y, **kwargs):
        dy = kwargs.get('dy', self.dy)

        self.n0 = kwargs.get('n0', len(t))

        t = np.asarray(t).astype(self.real_type)
        y = np.asarray(y).astype(self.real_type)

        yscale = max(y[:self.n0]) - min(y[:self.n0])
        y0 = min(y[:self.n0])
        if self.weighted:
            dy = np.asarray(dy).astype(self.real_type)
            if self.widen_mag_range:
                med_sigma = np.median(dy[:self.n0])
                yscale += 2 * self.max_phi * med_sigma
                y0 -= self.max_phi * med_sigma

            dy /= yscale
        y = (y - y0) / yscale

        if not self.weighted:
            y = np.floor(y * self.mag_bins).astype(self.ytype)

        if self.buffered_transfer:
            arrs = [self.t, self.y]
            if self.weighted:
                arrs.append(self.dy)

            if any([arr is None for arr in arrs]):
                if self.buffered_transfer:
                    self.allocate_buffered_data_arrays(**kwargs)

            assert(self.n0 <= len(self.t))

            self.t[:self.n0] = t[:self.n0]
            self.y[:self.n0] = y[:self.n0]

            if self.weighted:
                self.dy[:self.n0] = dy[:self.n0]
        else:
            self.t = t
            self.y = y
            if self.weighted:
                self.dy = dy
        return self

    def set_gpu_arrays_to_zero(self, **kwargs):
        for x in [self.t_g, self.y_g, self.dy_g]:
            if x is not None:
                x.fill(self.real_type(0), stream=self.stream)
        if self.weighted:
            self.bins_g.fill(self.real_type(0), stream=self.stream)
        else:
            self.bins_g.fill(np.uint32(0), stream=self.stream)

    def fromdata(self, t, y, **kwargs):
        self.setdata(t, y, **kwargs)

        if kwargs.get('allocate', True):
            self.allocate(**kwargs)

        return self


def conditional_entropy(memory, functions, block_size=256,
                        transfer_to_host=True,
                        transfer_to_device=True,
                        **kwargs):
    block = (block_size, 1, 1)
    grid = (int(np.ceil((memory.n0 * memory.nf) / float(block_size))), 1)
    hist_count, hist_weight, ce_std, ce_wt = functions

    if transfer_to_device:
        memory.transfer_data_to_gpu()

    if memory.weighted:
        args = (grid, block, memory.stream)
        args += (memory.t_g.ptr, memory.y_g.ptr, memory.dy_g.ptr)
        args += (memory.bins_g.ptr, memory.freqs_g.ptr)
        args += (np.int32(memory.nf), np.int32(memory.n0))
        args += (np.float32(memory.max_phi),)
        hist_weight.prepared_async_call(*args)

        grid = (int(np.ceil(memory.nf / float(block_size))), 1)

        args = (grid, block, memory.stream)
        args += (memory.bins_g.ptr, np.int32(memory.nf), memory.ce_g.ptr)
        ce_wt.prepared_async_call(*args)

        if transfer_to_host:
            memory.transfer_ce_to_cpu()
        return memory.ce_c

    args = (grid, block, memory.stream)
    args += (memory.t_g.ptr, memory.y_g.ptr)
    args += (memory.bins_g.ptr, memory.freqs_g.ptr)
    args += (np.int32(memory.nf), np.int32(memory.n0))
    hist_count.prepared_async_call(*args)

    grid = (int(np.ceil(memory.nf / float(block_size))), 1)
    args = (grid, block, memory.stream)
    args += (memory.bins_g.ptr, np.int32(memory.nf), memory.ce_g.ptr)
    ce_std.prepared_async_call(*args)

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
        self.mag_bins = kwargs.get('mag_bins', 10)
        self.max_phi = kwargs.get('max_phi', 3.)
        self.weighted = kwargs.get('weighted', False)
        self.block_size = kwargs.get('block_size', 256)

        self.phase_overlap = kwargs.get('phase_overlap', 0)
        self.mag_overlap = kwargs.get('mag_overlap', 0)

    def _compile_and_prepare_functions(self, **kwargs):

        cpp_defs = dict(NPHASE=self.phase_bins,
                        NMAG=self.mag_bins,
                        PHASE_OVERLAP=self.phase_overlap,
                        MAG_OVERLAP=self.mag_overlap)
        # Read kernel
        kernel_txt = _module_reader(find_kernel('ce'),
                                    cpp_defs=cpp_defs)

        # compile kernel
        self.module = SourceModule(kernel_txt, options=['--use_fast_math'])

        self.dtypes = dict(
            histogram_data_weighted=[np.intp, np.intp, np.intp, np.intp,
                                     np.intp, np.int32, np.int32, np.float32],
            histogram_data_count=[np.intp, np.intp, np.intp, np.intp,
                                  np.int32, np.int32],
            weighted_ce=[np.intp, np.int32, np.intp],
            standard_ce=[np.intp, np.int32, np.intp]
        )
        for fname, dtype in self.dtypes.iteritems():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))

    def memory_requirement(self, data, **kwargs):
        """ return an approximate GPU memory requirement in bytes """
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
                  max_phi=self.max_phi,
                  stream=stream,
                  weighted=self.weighted)

        kw.update(kwargs)
        mem = ConditionalEntropyMemory(**kw)

        mem.fromdata(t, y, dy=dy, freqs=freqs, allocate=True, **kwargs)

        return mem

    def autofrequency(self, *args, **kwargs):
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

    def run(self, data,
            memory=None,
            freqs=None,
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

        if memory is None:
            memory = self.allocate(data, freqs=frqs,
                                   **kwargs)
            for mem in memory:
                mem.transfer_freqs_to_gpu()
        else:
            for i, (t, y, dy) in enumerate(data):
                memory[i].set_gpu_arrays_to_zero(**kwargs)
                memory[i].setdata(t, y, dy=dy, **kwargs)

        kw = dict(block_size=self.block_size)
        kw.update(kwargs)
        results = [conditional_entropy(memory[i], self.function_tuple, **kw)
                   for i in range(len(data))]

        results = [(f, r) for f, r in zip(frqs, results)]
        return results

    def batched_run_const_nfreq(self, data, batch_size=10,
                                freqs=None,
                                **kwargs):
        """
        Same as ``batched_run`` but is more efficient when the frequencies are
        the same for each lightcurve. Doesn't reallocate memory for each batch.

        Notes
        -----
        To get best efficiency, make sure the maximum number of observations
        is not much larger than the typical number of observations
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
                          n0_buffer=max_ndata)
        kwargs_mem.update(kwargs)
        memory = [ConditionalEntropyMemory(stream=stream, **kwargs_mem)
                  for stream in streams]

        # allocate memory
        [mem.allocate(freqs=freqs, **kwargs) for mem in memory]

        [mem.transfer_freqs_to_gpu(**kwargs) for mem in memory]

        for b, batch in enumerate(batches):
            results = self.run(batch, memory=memory, freqs=freqs, **kwargs)
            self.finish()

            for i, (f, ce) in enumerate(results):
                ces.append(np.copy(ce))

        return [(freqs, ce) for ce in ces]
