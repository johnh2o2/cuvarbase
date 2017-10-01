"""
Implementation of the box-least squares periodogram [K2002]_
and variants.

.. [K2002] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_

"""
from __future__ import print_function, division

from builtins import zip
from builtins import range
import sys

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader

import resource
import numpy as np

_default_block_size = 256
_all_function_names = ['full_bls_no_sol_fast',
                       'full_bls_no_sol_fast_sma',
                       'bin_and_phase_fold_custom',
                       'reduction_max',
                       'store_best_sols',
                       'store_best_sols_custom',
                       'bin_and_phase_fold_bst_multifreq',
                       'binned_bls_bst']


_function_signatures = {
    'full_bls_no_sol_fast': [np.intp, np.intp, np.intp,
                             np.intp, np.intp, np.intp,
                             np.intp, np.int32, np.int32,
                             np.int32, np.int32, np.float32],
    'full_bls_no_sol_fast_sma': [np.intp, np.intp, np.intp,
                                 np.intp, np.intp, np.intp,
                                 np.intp, np.int32, np.int32,
                                 np.int32, np.int32, np.float32],
    'bin_and_phase_fold_custom': [np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.intp,
                                  np.intp, np.intp, np.int32,
                                  np.int32, np.int32, np.int32,
                                  np.int32],
    'reduction_max': [np.intp, np.intp, np.int32, np.int32, np.int32,
                      np.intp, np.intp, np.int32, np.int32],
    'store_best_sols': [np.intp, np.intp, np.intp, np.int32,
                        np.int32, np.int32, np.float32, np.int32,
                        np.int32],
    'store_best_sols_custom': [np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.int32,
                               np.int32, np.int32, np.int32],
    'bin_and_phase_fold_bst_multifreq':
        [np.intp, np.intp, np.intp, np.intp,
         np.intp, np.intp, np.int32, np.int32,
         np.int32, np.int32, np.int32, np.int32,
         np.float32, np.int32],
    'binned_bls_bst': [np.intp, np.intp, np.intp, np.int32]
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

    init = np.int32(1)
    while (grid_size > nfreq):

        max_func.prepared_async_call(grid, block, stream,
                                     arr.ptr, arr_args.ptr,
                                     np.int32(nfreq), np.int32(nbins0),
                                     np.int32(nbins),
                                     arr.ptr, arr_args.ptr, np.int32(0), init)
        init = np.int32(0)

        nbins0 = grid_size / nfreq
        grid_size = int(np.ceil(float(nbins0) / block_size)) * nfreq
        grid = (grid_size, 1)

    max_func.prepared_async_call(grid, block, stream,
                                 arr.ptr,  arr_args.ptr,
                                 np.int32(nfreq), np.int32(nbins0),
                                 np.int32(nbins),
                                 final_arr.ptr, final_argmax_arr.ptr,
                                 np.int32(final_index), init)


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
        [func.prepare(_function_signatures[name])
         for name, func in functions.items()]

    return functions


class BLSMemory(object):
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
            nfreqs = self.max_nfreqs
        if ndata is None:
            ndata = self.max_ndata

        self.bls = cuda.aligned_zeros(shape=(nfreqs,),
                                      dtype=self.rtype,
                                      alignment=resource.getpagesize())
        self.bls = cuda.register_host_memory(self.bls)

        self.nbins0 = cuda.aligned_zeros(shape=(nfreqs,),
                                         dtype=np.int32,
                                         alignment=resource.getpagesize())
        self.nbins0 = cuda.register_host_memory(self.nbins0)

        self.nbinsf = cuda.aligned_zeros(shape=(nfreqs,),
                                         dtype=np.int32,
                                         alignment=resource.getpagesize())
        self.nbinsf = cuda.register_host_memory(self.nbinsf)

        self.t = cuda.aligned_zeros(shape=(ndata,),
                                    dtype=self.rtype,
                                    alignment=resource.getpagesize())
        self.t = cuda.register_host_memory(self.t)

        self.yw = cuda.aligned_zeros(shape=(ndata,),
                                     dtype=self.rtype,
                                     alignment=resource.getpagesize())
        self.yw = cuda.register_host_memory(self.yw)

        self.w = cuda.aligned_zeros(shape=(ndata,),
                                    dtype=self.rtype,
                                    alignment=resource.getpagesize())
        self.w = cuda.register_host_memory(self.w)

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

        self.yw[:len(t)] = np.asarray((y - self.ybar) * w).astype(self.rtype)[:]

        if any([x is None for x in [self.t_g, self.yw_g, self.w_g]]):
            self.allocate_data()

        if self.freqs_g is None:
            self.allocate_freqs(nfreqs=len(freqs))

        if transfer:
            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))

        return self

    @classmethod
    def fromdata(cls, t, y, dy, qmin=None, qmax=None,
                 freqs=None, nf=1e6, transfer=True,
                 **kwargs):

        max_ndata = kwargs.get('max_ndata', len(t))
        max_nfreqs = kwargs.get('max_nfreqs', nf if freqs is None
                                else len(freqs))
        c = cls(max_ndata, max_nfreqs, **kwargs)

        return c.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                         freqs=freqs, nf=nf, transfer=transfer,
                         **kwargs)


def eebls_gpu_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                   functions=None, stream=None, dlogq=0.3,
                   memory=None, use_sma=False, noverlap=2,
                   max_shmem=int(4.8e4), batch_size=None, **kwargs):

    fname = 'full_bls_no_sol_fast'
    if use_sma:
        fname = '{fname}_sma'.format(fname=fname)

    if functions is None:
        functions = compile_bls(function_names=[fname], **kwargs)

    func = functions[fname]

    if memory is None:
        memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax,
                                    freqs=freqs, stream=stream,
                                    transfer=True,
                                    **kwargs)
    else:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax,
                       freqs=freqs, stream=stream,
                       transfer=True,
                       **kwargs)
    float_size = 4
    block_size = kwargs.get('block_size', _default_block_size)
    nblocks = int(max_shmem / ((2 * block_size + 1) * float_size))

    if batch_size is None:
        batch_size = len(freqs)

    nbatches = int(np.ceil(len(freqs) / batch_size))
    block = (block_size, 1, 1)
    grid = (nblocks, 1)
    for i in range(nbatches):

        args = (grid, block, stream)
        args += (memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr)
        args += (memory.bls_g.ptr, memory.freqs_g.ptr)
        args += (memory.nbins0_g.ptr, memory.nbinsf_g.ptr)
        args += (np.uint32(len(t)), np.uint32(batch_size),
                 np.uint32(i * batch_size))
        args += (np.uint32(noverlap), np.float32(dlogq))
        #print(args)
        func.prepared_async_call(*args)

    memory.transfer_data_to_cpu()

    #print(memory.bls_g.get() / memory.yy, memory.yy)

    return memory.bls


def eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                     batch_size=None, nstreams=5, max_memory=1e8,
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
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically by default based on ``max_memory``
    max_memory: float, optional (default: 1e8)
        Maximum memory to use in bytes. Will ignore this if
        ``batch_size`` is specified.
    functions: tuple of CUDA functions
        gpu_bls, gpu_max, gpu_bin, gpu_store functions;
        returned by `compile_bls`
    **kwargs:
        passed to `compile_bls`

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
    nbtot = len(q_values) * len(phi_values) * batch_size

    grid_size = int(np.ceil(float(nbtot) / block_size))

    if batch_size is None:
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
        mem_per_f = 4 * nstreams * nbins_tot_max * noverlap * real_type_size

        batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if batch_size == 0:
            raise Exception("Not enough memory (batch_size = 0)")

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
        bls_tmp_sol_gs.append(gpuarray.zeros(nbtot, dtype=np.int32))

    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.int32)

    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)

    q_values_g = gpuarray.to_gpu(np.asarray(q_values).astype(np.float32))
    phi_values_g = gpuarray.to_gpu(np.asarray(phi_values).astype(np.float32))

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_custom']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols_custom']

    for batch in range(nbatches):
        imin = batch_size * batch
        imax = min([len(freqs), batch_size * (batch + 1)])

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
        args += (np.int32(len(q_values)), np.int32(len(phi_values)))
        args += (np.int32(len(t)), np.int32(nf))
        args += (np.int32(batch_size * batch),)
        bin_func.prepared_async_call(*args)

        nb = len(q_values) * len(phi_values)

        bls_grid = (int(np.ceil(float(nf * nb) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(nf * nb))
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nb, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr)
        args += (np.int32(len(q_values)), np.int32(len(phi_values)))
        args += (np.int32(nf), np.int32(batch * batch_size))
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))

    return bls_g.get()/YY, qphi_sols


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              nstreams=5, noverlap=3, dlogq=0.2, max_memory=1e8,
              batch_size=None, functions=None, **kwargs):

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
    nstreams: int, optional (default: 5)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 3)
        Number of overlapping q bins to use
    dlogq: float, optional, (default: 0.5)
        logarithmic spacing of :math:`q` values, where :math:`d\log q = dq / q`
    batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; determines
        this automatically based on ``max_memory``
    max_memory: float, optional (default: 1e8)
        Maximum memory to use in bytes. Will ignore this if
        batch_size is specified
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

    # smallest and largest number of bins
    nbins0_max = 1
    nbinsf_max = 1
    block_size = kwargs.get('block_size', _default_block_size)

    max_q_vals = locext(max, qmax)
    min_q_vals = locext(min, qmin)

    nbins0_max = int(np.floor(1./max_q_vals))
    nbinsf_max = int(np.ceil(1./min_q_vals))

    ndata = len(t)

    nbins_tot_max = 0
    x = 1.
    while(int(np.int32(x * nbins0_max)) <= nbinsf_max):
        nbins_tot_max += int(np.int32(x * nbins0_max))
        x *= (1 + dlogq)

    if batch_size is None:
        # compute memory
        real_type_size = 4

        # data
        mem0 = ndata * 3 * real_type_size

        # freqs + bls + best_phi + best_q + best_sol (int32)
        mem0 += len(freqs) * 5 * real_type_size

        # yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs (int32)
        mem_per_f = 4 * nstreams * nbins_tot_max * noverlap * real_type_size

        batch_size = int(float(max_memory - mem0) / (mem_per_f))

        if batch_size == 0:
            raise Exception("Not enough memory (batch_size = 0)")

    gs = batch_size * nbins_tot_max * noverlap

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

    nbatches = int(np.ceil(float(len(freqs)) / batch_size))

    bls = np.zeros(len(freqs))
    bin_func = functions['bin_and_phase_fold_bst_multifreq']
    bls_func = functions['binned_bls_bst']
    max_func = functions['reduction_max']
    store_func = functions['store_best_sols']

    for batch in range(nbatches):

        imin = batch_size * batch
        imax = min([len(freqs), batch_size * (batch + 1)])

        minq = locext(min, qmin, imin, imax)
        maxq = locext(max, qmax, imin, imax)

        nbins0 = int(np.floor(1./maxq))
        nbinsf = int(np.ceil(1./minq))

        nbins_tot = 0
        x = 1.
        while(int(np.int32(x * nbins0)) <= nbinsf):
            nbins_tot += int(np.int32(x * nbins0))
            x *= (1 + dlogq)

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
        args += (np.int32(batch_size * batch), np.int32(noverlap))
        args += (np.float32(dlogq), np.int32(nbins_tot))
        bin_func.prepared_async_call(*args)

        all_bins = nf * nbins_tot * noverlap

        bls_grid = (int(np.ceil(float(all_bins) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(all_bins))
        bls_func.prepared_async_call(*args)

        args = (max_func, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        _reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (np.int32(nbins0), np.int32(nbinsf), np.int32(noverlap))
        args += (np.float32(dlogq), np.int32(nf))
        args += (np.int32(batch * batch_size),)
        store_func.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = list(zip(best_q, best_phi))

    return bls_g.get()/YY, qphi_sols


def single_bls(t, y, dy, freq, q, phi0):
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

    Returns
    -------
    bls: float
        BLS power for this set of parameters
    """

    phi = np.asarray(t).astype(np.float32) * np.float32(freq)
    phi -= np.float32(phi0)
    phi -= np.floor(phi)

    mask = phi < np.float32(q)

    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))

    ybar = np.dot(w, np.asarray(y).astype(np.float32))
    YY = np.dot(w, np.power(np.asarray(y).astype(np.float32) - ybar, 2))

    W = np.sum(w[mask])
    YW = np.dot(w[mask], np.asarray(y).astype(np.float32)[mask]) - ybar * W

    return 0 if W < 1e-9 else (YW ** 2) / (W * (1 - W)) / YY


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
                                        batch_size=5, nstreams=5,
                                        functions=functions, **kwargs)

        ibest = np.argmax(powers)
        f = freqs[ibest]
        q, phi = sols[ibest]
        pn = powers[ibest]
        i += 1
    return f, pn, i, (q, phi)


def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
                      qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                      fmax=None, freqs=None, qvals=None, **kwargs):
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

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             **kwargs)
    return freqs, powers, sols
