"""
Implementation of the box-least squares periodogram [K2002]_
and variants.

.. [K2002] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_

"""
import sys

#import pycuda.autoinit
import pycuda.autoprimaryctx
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader

import resource
import numpy as np

_default_block_size = 256
_all_function_names = ['full_bls_no_sol',
                       'bin_and_phase_fold_custom',
                       'reduction_max',
                       'store_best_sols',
                       'store_best_sols_custom',
                       'bin_and_phase_fold_bst_multifreq',
                       'binned_bls_bst']


_function_signatures = {
    'full_bls_no_sol': [np.intp, np.intp, np.intp,
                        np.intp, np.intp, np.intp,
                        np.intp, np.uint32, np.uint32,
                        np.uint32, np.uint32, np.uint32,
                        np.float32, np.float32, np.uint32],
    'bin_and_phase_fold_custom': [np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.int32,
                                  np.uint32, np.uint32, np.uint32,
                                  np.uint32],
    'reduction_max': [np.intp, np.intp, np.uint32, np.uint32, np.uint32,
                      np.intp, np.intp, np.uint32, np.uint32],
    'store_best_sols': [np.intp, np.intp, np.intp, np.uint32,
                        np.uint32, np.uint32, np.float32, np.uint32,
                        np.uint32],
    'store_best_sols_custom': [np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.uint32,
                               np.uint32, np.uint32, np.uint32],
    'bin_and_phase_fold_bst_multifreq':
        [np.intp, np.intp, np.intp, np.intp,
         np.intp, np.intp, np.uint32, np.uint32,
         np.uint32, np.uint32, np.uint32, np.uint32,
         np.float32, np.uint32],
    'binned_bls_bst': [np.intp, np.intp, np.intp, np.uint32, np.uint32]
}


def _reduction_max(max_func, arr, arr_args, nfreq, nbins,
                   stream, final_arr, final_argmax_arr,
                   final_index, block_size):
    # assert power of 2
    assert(block_size - 2 * (block_size / 2) == 0)

    block = (block_size, 1, 1)
    grid_size = int(np.ceil(float(nbins) / block_size)) * nfreq
    grid = (grid_size, 1)
    nbins0 = nbins

    init = np.uint32(1)
    while (grid_size > nfreq):

        max_func.prepared_async_call(grid, block, stream,
                                     arr.ptr, arr_args.ptr,
                                     np.uint32(nfreq), np.uint32(nbins0),
                                     np.uint32(nbins),
                                     arr.ptr, arr_args.ptr, np.uint32(0), init)
        init = np.uint32(0)

        nbins0 = grid_size / nfreq
        grid_size = int(np.ceil(float(nbins0) / block_size)) * nfreq
        grid = (grid_size, 1)

    max_func.prepared_async_call(grid, block, stream,
                                 arr.ptr,  arr_args.ptr,
                                 np.uint32(nfreq), np.uint32(nbins0),
                                 np.uint32(nbins),
                                 final_arr.ptr, final_argmax_arr.ptr,
                                 np.uint32(final_index), init)


def fmin_transit(t, rho=1., min_obs_per_transit=5, **kwargs):
    T = max(t) - min(t)
    qmin = float(min_obs_per_transit) / len(t)

    fmin1 = freq_transit(qmin, rho=rho)
    fmin2 = 2./(max(t) - min(t))
    return max([fmin1, fmin2])


def fmax_transit0(rho=1., **kwargs):
    return 8.6307 * np.sqrt(rho)


def q_transit(freq, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)

    f23 = np.power(freq / fmax0, 2./3.)
    f23 = np.minimum(1., f23)
    return np.arcsin(f23) / np.pi


def freq_transit(q, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return fmax0 * (np.sin(np.pi * q) ** 1.5)


def fmax_transit(rho=1., qmax=0.5, **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return min([fmax0, freq_transit(qmax, rho=rho, **kwargs)])


def transit_autofreq(t, fmin=None, fmax=None, samples_per_peak=2,
                     rho=1., qmin_fac=0.2, qmax_fac=None, **kwargs):
    """
    Produce list of frequencies for a given frequency range
    suitable for performing Keplerian BLS.

    Parameters
    ----------
    t: array_like, float
        Observation times.
    fmin: float, optional (default: ``None``)
        Minimum frequency. By default this is determined by ``fmin_transit``.
    fmax: float, optional (default: ``None``)
        Maximum frequency. By default this is determined by ``fmax_transit``.
    samples_per_peak: float, optional (default: 2)
        Oversampling factor. Frequency spacing is multiplied by
        ``1/samples_per_peak``.
    rho: float, optional (default: 1)
        Mean stellar density of host star in solar units
        :math:`\\rho=\\rho_{\\star} / \\rho_{\\odot}`, where
        :math:`\\rho_{\\odot}`
        is the mean density of the sun
    qmin_fac: float, optional (default: 0.2)
        The minimum :math:`q` value to search in units of the Keplerian
        :math:`q` value
    qmax_fac: float, optional (default: None)
        The maximum :math:`q` value to search in units of the Keplerian
        :math:`q` value. If ``None``, this defaults to ``1/qmin_fac``.

    Returns
    -------
    freqs: array_like
        The frequency grid
    q0vals: array_like
        The list of Keplerian :math:`q` values.

    """
    if qmax_fac is None:
        qmax_fac = 1./qmin_fac

    if fmin is None:
        fmin = fmin_transit(t, rho=rho, samples_per_peak=samples_per_peak,
                            **kwargs)
    if fmax is None:
        fmax = fmax_transit(rho=rho, **kwargs)

    T = max(t) - min(t)
    freqs = [fmin]
    while freqs[-1] < fmax:
        df = qmin_fac * q_transit(freqs[-1], rho=rho) / (samples_per_peak * T)
        freqs.append(freqs[-1] + df)
    freqs = np.array(freqs)
    q0vals = q_transit(freqs, rho=rho)
    return freqs, q0vals


def compile_bls(block_size=_default_block_size,
                function_names=_all_function_names,
                prepare=True,
                **kwargs):
    """
    Compile BLS kernel

    Parameters
    ----------
    block_size: int, optional (default: _default_block_size)
        CUDA threads per CUDA block.
    function_names: list, optional (default: _all_function_names)
        Function names to load and prepare
    prepare: bool, optional (default: True)
        Whether or not to prepare functions (for slightly faster
        kernel launching)

    Returns
    -------
    functions: dict
        Dictionary of (function name, PyCUDA function object) pairs

    """
    # Read kernel
    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel('bls'),
                                cpp_defs=cppd)

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    functions = {name: module.get_function(name) for name in function_names}

    # prepare functions
    if prepare:
        for name in functions.keys():
            sig = _function_signatures[name]
            functions[name] = functions[name].prepare(sig)

    return functions


class BLSMemory:
    def __init__(self, max_ndata, max_nfreqs, stream=None, **kwargs):
        self.max_ndata = max_ndata
        self.max_nfreqs = max_nfreqs
        self.t = None
        self.yw = None
        self.w = None

        self.t_g = None
        self.yw_g = None
        self.w_g = None

        self.freqs = None
        self.freqs_g = None

        self.qmin = None
        self.nbins0_g = None
        self.qmax = None
        self.nbinsf_g = None

        self.bls = None
        self.bls_g = None

        self.rtype = np.float32

        self.stream = stream

        self.allocate_pinned_arrays(nfreqs=max_nfreqs, ndata=max_ndata)

    def allocate_pinned_arrays(self, nfreqs=None, ndata=None):
        if nfreqs is None:
            nfreqs = int(self.max_nfreqs)
        if ndata is None:
            ndata = int(self.max_ndata)

        self.bls = cuda.aligned_zeros(shape=(nfreqs,),
                                      dtype=self.rtype,
                                      alignment=resource.getpagesize())

        self.nbins0 = cuda.aligned_zeros(shape=(nfreqs,),
                                         dtype=np.int32,
                                         alignment=resource.getpagesize())

        self.nbinsf = cuda.aligned_zeros(shape=(nfreqs,),
                                         dtype=np.int32,
                                         alignment=resource.getpagesize())

        self.t = cuda.aligned_zeros(shape=(ndata,),
                                    dtype=self.rtype,
                                    alignment=resource.getpagesize())

        self.yw = cuda.aligned_zeros(shape=(ndata,),
                                     dtype=self.rtype,
                                     alignment=resource.getpagesize())

        self.w = cuda.aligned_zeros(shape=(ndata,),
                                    dtype=self.rtype,
                                    alignment=resource.getpagesize())

    def allocate_freqs(self, nfreqs=None):
        if nfreqs is None:
            nfreqs = self.max_nfreqs

        self.freqs_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
        self.bls_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
        self.nbins0_g = gpuarray.zeros(nfreqs, dtype=np.uint32)
        self.nbinsf_g = gpuarray.zeros(nfreqs, dtype=np.uint32)

    def allocate_data(self, ndata=None):
        if ndata is None:
            ndata = len(self.t)
        self.t_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.yw_g = gpuarray.zeros(ndata, dtype=self.rtype)
        self.w_g = gpuarray.zeros(ndata, dtype=self.rtype)

    def transfer_data_to_gpu(self, transfer_freqs=True):
        self.t_g.set_async(self.t, stream=self.stream)
        self.yw_g.set_async(self.yw, stream=self.stream)
        self.w_g.set_async(self.w, stream=self.stream)

        if transfer_freqs:
            self.freqs_g.set_async(self.freqs, stream=self.stream)
            self.nbins0_g.set_async(self.nbins0, stream=self.stream)
            self.nbinsf_g.set_async(self.nbinsf, stream=self.stream)

    def transfer_data_to_cpu(self):
        # self.bls_g.get_async(ary=self.bls, stream=self.stream)
        if self.stream is None:
            self.bls = self.bls_g.get() / self.yy

        else:
            self.bls_g.get_async(ary=self.bls, stream=self.stream)
            self.bls /= self.yy

        # return self.bls

    def setdata(self, t, y, dy, qmin=None, qmax=None,
                freqs=None, nf=None, transfer=True,
                **kwargs):

        if freqs is not None:
            self.freqs = np.asarray(freqs).astype(self.rtype)
            self.nbinsf = (np.ones_like(self.freqs)/qmin).astype(np.uint32)
            self.nbins0 = (np.ones_like(self.freqs)/qmax).astype(np.uint32)

        self.t[:len(t)] = np.asarray(t).astype(self.rtype)[:]

        w = np.power(dy, -2)
        w /= sum(w)
        self.w[:len(t)] = np.asarray(w).astype(self.rtype)[:]

        self.ybar = sum(y * w)
        self.yy = np.dot(w, np.power(y - self.ybar, 2))

        u = (y - self.ybar) * w
        self.yw[:len(t)] = np.asarray(u).astype(self.rtype)[:]

        if any([x is None for x in [self.t_g, self.yw_g, self.w_g]]):
            self.allocate_data()

        if self.freqs_g is None:
            if nf is None:
                nf = len(freqs)
            self.allocate_freqs(nfreqs=nf)

        if transfer:
            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))

        return self

    @classmethod
    def fromdata(cls, t, y, dy, qmin=None, qmax=None,
                 freqs=None, nf=None, transfer=True,
                 **kwargs):

        max_ndata = kwargs.get('max_ndata', len(t))
        max_nfreqs = kwargs.get('max_nfreqs', nf if freqs is None
                                else len(freqs))
        c = cls(max_ndata, max_nfreqs, **kwargs)

        return c.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                         freqs=freqs, nf=nf, transfer=transfer,
                         **kwargs)


def eebls_gpu_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   ignore_negative_delta_sols=False,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, noverlap=2, max_nblocks=5000,
                   force_nblocks=None, dphi=0.0,
                   shmem_lim=None, freq_batch_size=None,
                   transfer_to_device=True,
                   transfer_to_host=True, **kwargs):
    """
    Box-Least Squares with PyCUDA but about 2-3 orders of magnitude
    faster than eebls_gpu. Uses shared memory for the binned data,
    which means that there is a lower limit on the q values that
    this function can handle.

    To save memory and improve speed, the best solution is not
    kept. To get the best solution, run ``eebls_gpu`` at the
    optimal frequency.

    .. warning::

        If you are running on a single-GPU machine, there may be a
        kernel time limit set by your OS. If running this function
        produces a timeout error, try setting ``freq_batch_size`` to a
        reasonable number (~10). That will split up the computations by
        frequency.

    .. note::

        No extra global memory is needed, meaning you likely do *not* need
        to use ``large_run`` with this function.

    .. note::

        There is no ``noverlap`` parameter here yet. This is only a problem
        if the optimal ``q`` value is close to ``qmin``. To alleviate this,
        you can run this function ``noverlap`` times with
        ``dphi = i/noverlap`` for the ``i``-th run. Then take the best solution
        of all runs.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    dphi: float, optional (default: 0.)
        Phase offset (in units of the finest grid spacing). If you
        want ``noverlap`` bins at the smallest ``q`` value, run this
        function ``noverlap`` times, with ``dphi = i / noverlap``
        for the ``i``-th run and take the best solution for all the runs.
    dlogq: float
        The logarithmic spacing of the q values to use. If negative,
        the q values increase by ``dq = qmin``.
    functions: dict
        Dictionary of compiled functions (see :func:`compile_bls`)
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; if
        ``None`` this will run a single batch for all frequencies
        simultaneously
    shmem_lim: int, optional (default: None)
        Maximum amount of shared memory to use per block in bytes.
        This is GPU-dependent but usually around 48KB. If ``None``,
        uses device information provided by PyCUDA (recommended).
    max_nblocks: int, optional (default: 200)
        Maximum grid size to use
    force_nblocks: int, optional (default: None)
        If this is set the gridsize is forced to be this value
    memory: :class:`BLSMemory` instance, optional (default: None)
        See :class:`BLSMemory`.
    transfer_to_host: bool, optional (default: True)
        Transfer BLS back to CPU.
    transfer_to_device: bool, optional (default: True)
        Transfer data to GPU
    **kwargs:
        passed to `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to
        :math:`1 - \chi_2(\omega) / \chi_2(constant)`

    """
    fname = 'full_bls_no_sol'

    if functions is None:
        functions = compile_bls(function_names=[fname], **kwargs)

    func = functions[fname]

    if shmem_lim is None:
        dev = pycuda.autoprimaryctx.device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = pycuda.autoprimaryctx.device.get_attribute(att)

    if memory is None:
        memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax,
                                    freqs=freqs, stream=stream,
                                    transfer=True,
                                    **kwargs)
    elif transfer_to_device:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                       freqs=freqs, transfer=True,
                       **kwargs)

    float_size = np.float32(1).nbytes
    block_size = kwargs.get('block_size', _default_block_size)

    if freq_batch_size is None:
        freq_batch_size = len(freqs)

    nbatches = int(np.ceil(len(freqs) / freq_batch_size))
    block = (block_size, 1, 1)

    # minimum q value that we can handle with the shared memory limit
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)
    i_freq = 0
    while(i_freq < len(freqs)):
        j_freq = min([i_freq + freq_batch_size, len(freqs)])
        nfreqs = j_freq - i_freq

        max_nbins = max(memory.nbinsf[i_freq:j_freq])

        mem_req = (block_size + 2 * max_nbins) * float_size

        if mem_req > shmem_lim:
            s = "qmin = %.2e requires too much shared memory." % (1./max_nbins)
            s += " Either try a larger value of qmin (> %e)" % (qmin_min)
            s += " or avoid using eebls_gpu_fast."
            raise Exception(s)
        # nblocks = int((2 * max_shmem / (mem_req + 4 * float_size)))
        nblocks = min([nfreqs, max_nblocks])
        if force_nblocks is not None:
            nblocks = force_nblocks

        grid = (nblocks, 1)
        args = (grid, block)
        if stream is not None:
            args += (stream,)
        args += (memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr)
        args += (memory.bls_g.ptr, memory.freqs_g.ptr)
        args += (memory.nbins0_g.ptr, memory.nbinsf_g.ptr)
        args += (np.uint32(len(t)), np.uint32(nfreqs),
                 np.uint32(i_freq))
        args += (np.uint32(max_nbins), np.uint32(noverlap))
        args += (np.float32(dlogq), np.float32(dphi))
        args += (np.uint32(ignore_negative_delta_sols),)

        if stream is not None:
            func.prepared_async_call(*args, shared_size=int(mem_req))
        else:
            func.prepared_call(*args, shared_size=int(mem_req))

        i_freq = j_freq

    if transfer_to_host:
        memory.transfer_data_to_cpu()
        if stream is not None:
            stream.synchronize()

    return memory.bls


def eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                     ignore_negative_delta_sols=False,
                     freq_batch_size=None, nstreams=5, max_memory=None,
                     functions=None, **kwargs):
    """
    Box-Least Squares, with custom q and phi values. Useful
    if you're honing the initial solution or testing between
    a relatively small number of possible solutions.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    q_values: array_like
        Set of q values to search at each trial frequency
    phi_values: float or array_like
        Set of phi values to search at each trial frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically by default based on ``max_memory``
    max_memory: float, optional (default: None)
        Maximum memory to use in bytes. Will ignore this if
        ``freq_batch_size`` is specified. If ``None``, will use the
        free memory given by ``pycuda.driver.mem_get_info()``
    functions: tuple of CUDA functions
        Dictionary of prepared functions from :func:`compile_bls`.
    **kwargs:
        passed to :func:`compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    qphi_sols: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    block_size = kwargs.get('block_size', _default_block_size)
    ndata = len(t)

    # read max_memory as total free memory available from driver
    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(0.9 * free)

    if freq_batch_size is None:
        # compute memory
        real_type_size = 4

        # data
        mem0 = ndata * 3 * real_type_size

        nq = len(q_values)
        nphi = len(phi_values)

        # q_values and phi_values
        mem0 += nq + nphi

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += len(freqs) * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nq * nphi * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size == 0:
            raise Exception("Not enough memory (freq_batch_size = 0)")

    nbtot = len(q_values) * len(phi_values) * freq_batch_size

    grid_size = int(np.ceil(float(nbtot) / block_size))

    # move data to GPU
    w = np.power(dy, -2)
    w /= sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(np.array(y) - ybar, 2))
    yw = (np.array(y) - ybar) * np.array(w)

    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))

    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nstreams):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(nbtot, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(nbtot, dtype=np.uint32))

    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.uint32)

    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)

    q_values_g = gpuarray.to_gpu(np.asarray(q_values).astype(np.float32))
    phi_values_g = gpuarray.to_gpu(np.asarray(phi_values).astype(np.float32))

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / freq_batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_custom']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols_custom']

    for batch in range(nbatches):
        imin = freq_batch_size * batch
        imax = min([len(freqs), freq_batch_size * (batch + 1)])

        nf = imax - imin
        j = batch % nstreams
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)
        bls_tmp_g.fill(np.float32(0), stream=stream)
        bls_tmp_sol_g.fill(np.int32(0), stream=stream)

        bin_grid = (int(np.ceil(float(len(t) * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr)
        args += (np.uint32(len(q_values)), np.uint32(len(phi_values)))
        args += (np.uint32(len(t)), np.uint32(nf))
        args += (np.uint32(freq_batch_size * batch),)
        bin_func.prepared_async_call(*args)

        nb = len(q_values) * len(phi_values)

        bls_grid = (int(np.ceil(float(nf * nb) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.uint32(nf * nb))
        args += (np.uint32(ignore_negative_delta_sols),)
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nb, stream, bls_g, bls_sol_g)
        args += (batch * freq_batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr)
        args += (np.uint32(len(q_values)), np.uint32(len(phi_values)))
        args += (np.uint32(nf), np.uint32(batch * freq_batch_size))
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))

    return bls_g.get()/YY, qphi_sols


def dnbins(nbins, dlogq):
    if (dlogq < 0):
        return 1

    n = int(np.floor(dlogq * nbins))

    return n if n > 0 else 1


def nbins_iter(i, nb0, dlogq):
    nb = nb0
    for j in range(i):
        nb += dnbins(nb, dlogq)

    return nb


def count_tot_nbins(nbins0, nbinsf, dlogq):
    ntot = 0

    i = 0
    while nbins_iter(i, nbins0, dlogq) <= nbinsf:
        ntot += nbins_iter(i, nbins0, dlogq)
        i += 1
    return ntot


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              ignore_negative_delta_sols=False,
              nstreams=5, noverlap=3, dlogq=0.2, max_memory=None,
              freq_batch_size=None, functions=None, **kwargs):

    """
    Box-Least Squares, accelerated with PyCUDA

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like
        Minimum q value(s) to test for each frequency
    qmax: float or array_like
        Maximum q value(s) to test for each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 3)
        Number of overlapping q bins to use
    dlogq: float, optional, (default: 0.5)
        logarithmic spacing of :math:`q` values, where :math:`d\log q = dq / q`
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically based on ``max_memory``
    max_memory: float, optional (default: None)
        Maximum memory to use in bytes. Will ignore this if
        ``freq_batch_size`` is specified, and will use the total free memory
        as returned by ``pycuda.driver.mem_get_info`` if this is ``None``.
    functions: tuple of CUDA functions
        returned by ``compile_bls``

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    qphi_sols: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

    """

    def locext(ext, arr, imin=None, imax=None):
        if isinstance(arr, float) or isinstance(arr, int):
            return arr
        return ext(arr[slice(imin, imax)])

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    if max_memory is None:
        free, total = cuda.mem_get_info()
        max_memory = int(0.9 * free)

    # smallest and largest number of bins
    nbins0_max = 1
    nbinsf_max = 1
    block_size = kwargs.get('block_size', _default_block_size)

    max_q_vals = locext(max, qmax)
    min_q_vals = locext(min, qmin)

    nbins0_max = int(np.floor(1./max_q_vals))
    nbinsf_max = int(np.ceil(1./min_q_vals))

    ndata = len(t)

    nbins_tot_max = count_tot_nbins(nbins0_max, nbinsf_max, dlogq)

    if freq_batch_size is None:
        # compute memory
        real_type_size = np.float32(1).nbytes

        # data
        mem0 = ndata * 3 * real_type_size

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += len(freqs) * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nbins_tot_max * noverlap * real_type_size

        freq_batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if freq_batch_size == 0:
            raise Exception("Not enough memory (freq_batch_size = 0)")

    gs = freq_batch_size * nbins_tot_max * noverlap

    grid_size = int(np.ceil(float(gs) / block_size))

    # move data to GPU
    w = np.power(dy, -2)
    w /= sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(np.array(y) - ybar, 2))
    yw = (np.array(y) - ybar) * np.array(w)

    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))

    yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams \
        = [], [], [], [], []
    for i in range(nstreams):
        streams.append(cuda.Stream())
        yw_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        w_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_gs.append(gpuarray.zeros(gs, dtype=np.float32))
        bls_tmp_sol_gs.append(gpuarray.zeros(gs, dtype=np.int32))

    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.int32)

    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / freq_batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_bst_multifreq']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols']

    for batch in range(nbatches):

        imin = freq_batch_size * batch
        imax = min([len(freqs), freq_batch_size * (batch + 1)])

        minq = locext(min, qmin, imin, imax)
        maxq = locext(max, qmax, imin, imax)

        nbins0 = int(np.floor(1./maxq))
        nbinsf = int(np.ceil(1./minq))

        nbins_tot = count_tot_nbins(nbins0, nbinsf, dlogq)

        nf = imax - imin
        j = batch % nstreams
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]
        # stream.synchronize()

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)
        bls_tmp_g.fill(np.float32(0), stream=stream)
        bls_tmp_sol_g.fill(np.int32(0), stream=stream)

        bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (np.int32(ndata), np.int32(nf))
        args += (np.int32(nbins0), np.int32(nbinsf))
        args += (np.int32(freq_batch_size * batch), np.int32(noverlap))
        args += (np.float32(dlogq), np.int32(nbins_tot))
        bin_func.prepared_async_call(*args)

        all_bins = nf * nbins_tot * noverlap

        bls_grid = (int(np.ceil(float(all_bins) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(all_bins))
        args += (np.uint32(ignore_negative_delta_sols),)
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * freq_batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (np.uint32(nbins0), np.uint32(nbinsf), np.uint32(noverlap))
        args += (np.float32(dlogq), np.uint32(nf))
        args += (np.uint32(batch * freq_batch_size),)
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))

    return bls_g.get()/YY, qphi_sols


def single_bls(t, y, dy, freq, q, phi0, ignore_negative_delta_sols=False, pdot=0.0):
    """
    Evaluate BLS power for a single set of (freq, q, phi0)
    parameters.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freq: float
        Frequency of the signal
    q: float
        Transit duration in phase
    phi0: float
        Phase offset of transit
    ignore_negative_delta_sols:
        Whether or not to ignore solutions with negative delta (inverted dips)
    pdot: float, optional (default: 0.0)
        Period derivative parameter. Phase is calculated as:
        phi = freq * t + 0.5 * pdot * t^2

    Returns
    -------
    bls: float
        BLS power for this set of parameters
    """
    t_arr = np.asarray(t).astype(np.float32)
    
    if pdot == 0.0:
        phi = t_arr * np.float32(freq)
    else:
        phi = t_arr * np.float32(freq) + 0.5 * np.float32(pdot) * t_arr * t_arr
    
    phi -= np.float32(phi0)
    phi -= np.floor(phi)

    mask = phi < np.float32(q)

    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))

    ybar = np.dot(w, np.asarray(y).astype(np.float32))
    YY = np.dot(w, np.power(np.asarray(y).astype(np.float32) - ybar, 2))

    W = np.sum(w[mask])
    YW = np.dot(w[mask], np.asarray(y).astype(np.float32)[mask]) - ybar * W

    if YW > 0 and ignore_negative_delta_sols:
        return 0
    return 0 if W < 1e-9 else (YW ** 2) / (W * (1 - W)) / YY


def sparse_bls_cpu(t, y, dy, freqs, ignore_negative_delta_sols=False, pdots=None):
    """
    Sparse BLS implementation for CPU (no binning, tests all pairs of observations).
    
    This is more efficient than traditional BLS when the number of observations
    is small, as it avoids redundant grid searching over finely-grained parameter
    grids. Based on https://arxiv.org/abs/2103.06193
    
    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies to test
    ignore_negative_delta_sols: bool, optional (default: False)
        Whether or not to ignore solutions with negative delta (inverted dips)
    pdots: array_like, float, optional (default: None)
        Period derivative values (one per frequency). If None, pdot=0 for all frequencies.
        Phase is calculated as: phi = freq * t + 0.5 * pdot * t^2
    
    Returns
    -------
    bls: array_like, float
        BLS power at each frequency
    solutions: list of (q, phi0) tuples
        Best (q, phi0) solution at each frequency
    """
    t = np.asarray(t).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    dy = np.asarray(dy).astype(np.float32)
    freqs = np.asarray(freqs).astype(np.float32)
    
    ndata = len(t)
    nfreqs = len(freqs)
    
    # Handle pdots
    if pdots is None:
        pdots = np.zeros(nfreqs, dtype=np.float32)
    else:
        pdots = np.asarray(pdots).astype(np.float32)
        if len(pdots) != nfreqs:
            raise ValueError("Length of pdots must match length of freqs")
    
    # Precompute weights
    w = np.power(dy, -2).astype(np.float32)
    w /= np.sum(w)
    
    # Precompute normalization
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))
    
    bls_powers = np.zeros(nfreqs, dtype=np.float32)
    best_q = np.zeros(nfreqs, dtype=np.float32)
    best_phi = np.zeros(nfreqs, dtype=np.float32)
    
    # For each frequency
    for i_freq, (freq, pdot) in enumerate(zip(freqs, pdots)):
        # Compute phases
        if pdot == 0.0:
            phi = (t * freq) % 1.0
        else:
            phi = (t * freq + 0.5 * pdot * t * t) % 1.0
        
        # Sort by phase
        sorted_indices = np.argsort(phi)
        phi_sorted = phi[sorted_indices]
        y_sorted = y[sorted_indices]
        w_sorted = w[sorted_indices]
        
        max_bls = 0.0
        best_q_val = 0.0
        best_phi_val = 0.0
        
        # Test all pairs of observations
        for i in range(ndata):
            for j in range(i + 1, ndata):
                # Transit from observation i to observation j
                phi0 = phi_sorted[i]
                q = phi_sorted[j] - phi_sorted[i]
                
                # Skip if q is too large (more than half the phase)
                if q > 0.5:
                    continue
                    
                # Observations in transit: indices i through j-1
                W = np.sum(w_sorted[i:j])
                
                # Skip if too few weight in transit
                if W < 1e-9 or W > 1.0 - 1e-9:
                    continue
                
                YW = np.dot(w_sorted[i:j], y_sorted[i:j]) - ybar * W
                
                # Check if we should ignore this solution
                if YW > 0 and ignore_negative_delta_sols:
                    continue
                    
                # Compute BLS
                bls = (YW ** 2) / (W * (1 - W)) / YY
                
                if bls > max_bls:
                    max_bls = bls
                    best_q_val = q
                    best_phi_val = phi0
        
        bls_powers[i_freq] = max_bls
        best_q[i_freq] = best_q_val
        best_phi[i_freq] = best_phi_val
    
    solutions = list(zip(best_q, best_phi))
    return bls_powers, solutions


def eebls_transit(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                  qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                  fmax=None, freqs=None, qvals=None, use_fast=False,
                  use_sparse=None, sparse_threshold=500,
                  ignore_negative_delta_sols=False,
                  pdots=None,
                  **kwargs):
    """
    Compute BLS for timeseries, automatically selecting between GPU and
    CPU implementations based on dataset size.
    
    For small datasets (ndata < sparse_threshold), uses the sparse BLS
    algorithm which avoids binning and grid searching. For larger datasets,
    uses the GPU-accelerated standard BLS.
    
    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    fmax_frac: float, optional (default: 1.0)
        Maximum frequency is `fmax_frac * fmax`, where
        `fmax` is automatically selected by `fmax_transit`.
    fmin_frac: float, optional (default: 1.0)
        Minimum frequency is `fmin_frac * fmin`, where
        `fmin` is automatically selected by `fmin_transit`.
    fmin: float, optional (default: None)
        Overrides automatic frequency minimum with this value
    fmax: float, optional (default: None)
        Overrides automatic frequency maximum with this value
    qmin_fac: float, optional (default: 0.5)
        Fraction of the fiducial q value to search
        at each frequency (minimum)
    qmax_fac: float, optional (default: 2.0)
        Fraction of the fiducial q value to search
        at each frequency (maximum)
    freqs: array_like, optional (default: None)
        Overrides the auto-generated frequency grid
    qvals: array_like, optional (default: None)
        Overrides the keplerian q values
    use_fast: bool, optional (default: False)
        Use fast GPU implementation (if not using sparse)
    use_sparse: bool, optional (default: None)
        If True, use sparse BLS. If False, use GPU BLS. If None (default),
        automatically select based on dataset size (sparse_threshold).
    sparse_threshold: int, optional (default: 500)
        Threshold for automatically selecting sparse BLS. If ndata < threshold
        and use_sparse is None, sparse BLS is used.
    ignore_negative_delta_sols: bool, optional (default: False)
        Whether or not to ignore inverted dips
    pdots: array_like, float, optional (default: None)
        Period derivative values (one per frequency). If provided, phase is
        calculated as: phi = freq * t + 0.5 * pdot * t^2.
        Note: GPU kernels do not currently support pdot; this parameter only
        works with use_sparse=True or when automatic selection chooses sparse BLS.
    **kwargs:
        passed to `eebls_gpu`, `eebls_gpu_fast`, `compile_bls`, 
        `fmax_transit`, `fmin_transit`, and `transit_autofreq`
    
    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    solutions: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency
        
        .. note::
        
            Only returned when ``use_fast=False``.
    
    """
    ndata = len(t)
    
    # Determine whether to use sparse BLS
    if use_sparse is None:
        use_sparse = ndata < sparse_threshold
    
    # Check if pdots is provided with non-sparse mode
    if pdots is not None and not use_sparse:
        raise ValueError("pdot parameter is only supported with use_sparse=True or "
                        "when automatic selection chooses sparse BLS (ndata < sparse_threshold)")
    
    # Generate frequency grid if not provided
    if freqs is None:
        if qvals is not None:
            raise Exception("qvals must be None if freqs is None")
        if fmin is None:
            fmin = fmin_transit(t, **kwargs) * fmin_frac
        if fmax is None:
            fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
        freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                        qmin_fac=qmin_fac, **kwargs)
    if qvals is None:
        qvals = q_transit(freqs, **kwargs)
    
    # Use sparse BLS for small datasets
    if use_sparse:
        powers, sols = sparse_bls_cpu(t, y, dy, freqs,
                                       ignore_negative_delta_sols=ignore_negative_delta_sols,
                                       pdots=pdots)
        return freqs, powers, sols
    
    # Use GPU BLS for larger datasets
    qmins = qvals * qmin_fac
    qmaxes = qvals * qmax_fac
    
    if use_fast:
        powers = eebls_gpu_fast(t, y, dy, freqs,
                                qmin=qmins, qmax=qmaxes,
                                ignore_negative_delta_sols=ignore_negative_delta_sols,
                                **kwargs)
        return freqs, powers
    
    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
    return freqs, powers, sols


def hone_solution(t, y, dy, f0, df0, q0, dlogq0, phi0, stop=1e-5,
                  samples_per_peak=5, max_iter=50, noverlap=3, **kwargs):
    """
    Experimental!
    """
    p0 = single_bls(t, y, dy, f0, q0, phi0)
    pn = None

    df = df0
    dlogq = dlogq0
    q = q0
    phi = phi0
    f = f0
    nol = noverlap

    baseline = max(t) - min(t)

    functions = compile_bls(**kwargs)
    i = 0
    while pn is None or i < 5 or ((pn - p0) / p0 > stop and i < max_iter):

        if pn is not None:
            p0 = pn

        fmin, fmax = f - 25 * df, f + 25 * df
        qmin = q / (1 + 5 * dlogq)
        qmax = q * (1 + 5 * dlogq)
        df *= 0.1
        dlogq *= 0.25

        nq = int(np.ceil(np.log(qmax/qmin)/dlogq))
        q_values = np.logspace(np.log(qmin), np.log(qmax), num=nq, base=np.e)
        dphi = 2 * ((fmax - fmin) * baseline * samples_per_peak + dlogq * q)

        phimin = 0.
        phimax = 1.
        if dphi < 0.25:
            phimin = phi - dphi
            phimax = phi + dphi

        nphi = max([10, int(np.ceil(10 * min([2 * dphi, 1.]) / qmin))])
        phi_values = np.linspace(phimin, phimax, nphi)
        nf = int((fmax - fmin)/df)
        freqs = np.linspace(fmin, fmax + df, nf)

        powers, sols = eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                                        freq_batch_size=5, nstreams=5,
                                        functions=functions, **kwargs)

        ibest = np.argmax(powers)
        f = freqs[ibest]
        q, phi = sols[ibest]
        pn = powers[ibest]
        i += 1
    return f, pn, i, (q, phi)


def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                      qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                      fmax=None, freqs=None, qvals=None, use_fast=False,
                      ignore_negative_delta_sols=False,
                      **kwargs):
    """
    Compute BLS for timeseries assuming edge-on keplerian
    orbit of a planet with Mp/Ms << 1, Rp/Rs < 1, Lp/Ls << 1 and
    negligible eccentricity.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    fmax_frac: float, optional (default: 1.0)
        Maximum frequency is `fmax_frac * fmax`, where
        `fmax` is automatically selected by `fmax_transit`.
    fmin_frac: float, optional (default: 1.5)
        Minimum frequency is `fmin_frac * fmin`, where
        `fmin` is automatically selected by `fmin_transit`.
    fmin: float, optional (default: None)
        Overrides automatic frequency minimum with this value
    fmax: float, optional (default: None)
        Overrides automatic frequency maximum with this value
    qmin_fac: float, optional (default: 0.5)
        Fraction of the fiducial q value to search
        at each frequency (minimum)
    qmax_fac: float, optional (default: 2.0)
        Fraction of the fiducial q value to search
        at each frequency (maximum)
    freqs: array_like, optional (default: None)
        Overrides the auto-generated frequency grid
    qvals: array_like, optional (default: None)
        Overrides the keplerian q values
    functions: tuple, optional (default=None)
        result of ``compile_bls(**kwargs)``.
    use_fast: bool, optional (default: False)

    ignore_negative_delta_sols: bool
        Whether or not to ignore inverted dips
    **kwargs:
        passed to `eebls_gpu`, `compile_bls`, `fmax_transit`,
        `fmin_transit`, and `transit_autofreq`


    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to :math:`1 - \chi^2(f) / \chi^2_0`
    solutions: list of ``(q, phi)`` tuples
        Best ``(q, phi)`` solution at each frequency

        .. note::

            Only returned when ``use_fast=False``.

    """

    if freqs is None:
        if qvals is not None:
            raise Exception("qvals must be None if freqs is None")
        if fmin is None:
            fmin = fmin_transit(t, **kwargs) * fmin_frac
        if fmax is None:
            fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
        freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                        qmin_fac=qmin_fac, **kwargs)
    if qvals is None:
        qvals = q_transit(freqs, **kwargs)

    qmins = qvals * qmin_fac
    qmaxes = qvals * qmax_fac

    if use_fast:
        powers = eebls_gpu_fast(t, y, dy, freqs,
                                qmin=qmins, qmax=qmaxes,
                                ignore_negative_delta_sols=ignore_negative_delta_sols,
                                **kwargs)

        return freqs, powers

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             ignore_negative_delta_sols=ignore_negative_delta_sols,
                             **kwargs)
    return freqs, powers, sols

