import sys
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

from pycuda.compiler import SourceModule
from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader

import resource
import numpy as np


def reduction_max(max_func, arr, arr_args, nfreq, nbins,
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


def get_qphi(nbins0, nbinsf, alpha, noverlap):
    q_phi = []

    x = np.float32(1.)
    dphi = np.float32(np.float32(1.)/noverlap)
    while(np.int32(x * nbins0) <= nbinsf):

        nb = np.int32(x * nbins0)
        q = np.float32(1.)/nb

        qp = []
        for s in range(noverlap):
            for i in range(nb):
                phi = (float(i) + np.float32(s) * dphi) / nb
                phi = ((phi - 0.5) + 0.5 * q) % 1.0
                if phi < 0:
                    phi += 1.0
                qp.append((q, phi))

        q_phi.extend(qp)
        x *= np.float32(alpha)

    return q_phi


def eebls_gpu(t, y, dy, freqs, qmin=0.01, qmax=0.5, nstreams=10,
              noverlap=10, alpha=1.5, block_size=256,
              batch_size=1, plot_status=False, **kwargs):

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
        Frequencies to evaluate BLS spectrum
    qmin: float, optional (default: 0.01)
        Minimum q value to search
    qmax: float, optional (default: 0.5)
        Maximum q value to search
    nstreams: int, optional (default: 10)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 10)
        Number of overlapping bins to use
    alpha: float (> 1), optional, (default: 1.5)
        1 + dlog q, where dlog q = dq / q
    block_size: int, optional (default: 256)
        CUDA block size to use (must be power of 2)
    batch_size: int, optional (default: 1)
        Number of frequencies to compute in a single batch

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    qphi_sols: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """

    # Read kernel
    kernel_txt = _module_reader(find_kernel('bls'),
                                cpp_defs=dict(BLOCK_SIZE=block_size))

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    # get functions
    gpu_bls = module.get_function('binned_bls_bst')
    gpu_max = module.get_function('reduction_max')
    gpu_bin = module.get_function('bin_and_phase_fold_bst_multifreq')

    # Prepare the functions
    gpu_bls = gpu_bls.prepare([np.intp, np.intp, np.intp, np.int32])
    gpu_max = gpu_max.prepare([np.intp, np.intp, np.int32, np.int32, np.int32,
                               np.intp, np.intp, np.int32, np.int32])
    gpu_bin = gpu_bin.prepare([np.intp, np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.int32, np.int32,
                               np.int32, np.int32, np.int32, np.int32,
                               np.float32, np.int32])

    # smallest and largest number of bins
    nbins0 = 1
    nbinsf = 1
    while (int(1./(qmax)) > nbins0):
        nbins0 = np.ceil(alpha * nbins0)

    while (int(1./(qmin)) > nbinsf):
        nbinsf = np.ceil(alpha * nbinsf)

    nbins0 = np.int32(nbins0)
    nbinsf = np.int32(nbinsf)
    ndata = np.int32(len(t))

    q_phi = get_qphi(nbins0, nbinsf, alpha, noverlap)
    qvals, phivals = zip(*q_phi)

    nbins_tot = len(q_phi) / noverlap

    gs = batch_size * len(q_phi)

    grid_size = int(np.ceil(float(gs) / block_size))

    w = np.power(dy, -2)
    w /= sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))

    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
    yw_g = gpuarray.to_gpu(np.array((y - ybar) * w).astype(np.float32))
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

    block = (block_size, 1, 1)

    grid = (grid_size, 1)

    nbatches = int(np.ceil(float(len(freqs)) / batch_size))
    for batch in range(nbatches):

        nf = batch_size if batch < nbatches - 1 else \
                           len(freqs) - batch * batch_size

        nf = np.int32(nf)
        j = batch % nstreams
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)

        bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (ndata, nf, nbins0, nbinsf)
        args += (np.int32(batch_size * batch), noverlap)
        args += (alpha, nbins_tot)
        gpu_bin.prepared_async_call(*args)

        args = (grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  nf * nbins_tot * noverlap)
        gpu_bls.prepared_async_call(*args)

        bls_tmp = None if not plot_status else bls_tmp_g.get()

        args = (gpu_max, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        reduction_max(*args)

    bls_sols = bls_sol_g.get()

    assert(not any(bls_sols < 0))
    qphi_sols = [(qvals[b], phivals[b]) for b in bls_sols]

    return bls_g.get()/YY, qphi_sols


def q_transit(freq, rho, q0=0.038285):
    return q0 * (freq ** (4./3.)) * (rho ** (-1./3.))


def bls_fast_autofreq(t, fmin=1E-2, fmax=1E2, oversampling=2,
                      rho=1, qmin_fac=0.2):
    T = max(t) - min(t)
    freqs = [fmin]
    while freqs[-1] < fmax:
        df = qmin_fac * q_transit(freqs[-1], rho) / (oversampling * T)
        freqs.append(freqs[-1] + df)
    freqs = np.array(freqs)
    q0vals = q_transit(freqs, rho)
    return freqs, q0vals


def get_nbins(q, alpha):
    nb = 1
    while int(1./q) > nb:
        nb = int(np.ceil(nb * alpha))
    return nb


def eebls_gpu_fast(t, y, dy, freqs, rho=1, nstreams=10, qmin_fac=0.2,
                   qmax_fac=5., noverlap=5, alpha=1.5, block_size=256,
                   batch_size=1, plot_status=False, **kwargs):

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
    fmin: float, optional (default: 1E-2)
        Minimum frequency
    fmax: float, optional (default: 1E2)
        Maximum frequency
    oversampling: float, optional (default: 2)
        Factor that enhances frequency spacing relative
        to fiducial spacing
    rho: float, optional (default: 1)
        Stellar density in solar units
    nstreams: int, optional (default: 10)
        Number of CUDA streams to utilize.
    noverlap: int, optional (default: 10)
        Number of overlapping bins to use
    alpha: float (> 1), optional, (default: 1.5)
        1 + dlog q, where dlog q = dq / q
    block_size: int, optional (default: 256)
        CUDA block size to use (must be power of 2)
    batch_size: int, optional (default: 1)
        Number of frequencies to compute in a single batch

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    qphi_sols: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """

    # Read kernel
    kernel_txt = _module_reader(find_kernel('bls'),
                                cpp_defs=dict(BLOCK_SIZE=block_size))

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    # get functions
    gpu_bls = module.get_function('binned_bls_bst')
    gpu_max = module.get_function('reduction_max')
    gpu_bin = module.get_function('bin_and_phase_fold_bst_multifreq')
    gpu_store = module.get_function('store_best_sols')

    # Prepare the functions
    gpu_bls = gpu_bls.prepare([np.intp, np.intp, np.intp, np.int32])
    gpu_max = gpu_max.prepare([np.intp, np.intp, np.int32, np.int32, np.int32,
                               np.intp, np.intp, np.int32, np.int32])
    gpu_bin = gpu_bin.prepare([np.intp, np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.int32, np.int32,
                               np.int32, np.int32, np.int32, np.int32,
                               np.float32, np.int32])

    gpu_store = gpu_store.prepare([np.intp, np.intp, np.intp, np.int32,
                                   np.int32, np.int32, np.float32, np.int32,
                                   np.int32])

    # smallest and largest number of bins
    nbins0_max = 1
    nbinsf_max = 1

    qmin_all = qmin_fac * q_transit(min(freqs), rho)
    qmin_max_all = (qmax_fac / qmin_fac) * qmin_all

    nbins0_max = get_nbins(qmin_max_all, alpha)
    nbinsf_max = get_nbins(qmin_all, alpha)

    ndata = np.int32(len(t))

    q_phi = get_qphi(nbins0_max, nbinsf_max, alpha, noverlap)
    # qvals, phivals = zip(*q_phi)

    nbins_tot = len(q_phi) / noverlap

    gs = batch_size * len(q_phi)

    grid_size = int(np.ceil(float(gs) / block_size))

    w = np.power(dy, -2)
    w /= sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))

    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
    yw_g = gpuarray.to_gpu(np.array((y - ybar) * w).astype(np.float32))
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
    for batch in range(nbatches):

        qmin = qmin_fac * q_transit(freqs[batch_size * batch], rho)
        imax = min([len(freqs) - 1, batch_size * (batch + 1) - 1])
        qmax = qmax_fac * q_transit(freqs[imax], rho)

        nbins0 = get_nbins(qmax, alpha)
        nbinsf = get_nbins(qmin, alpha)

        nbins_tot = 0
        x = 1.
        while(np.int32(x * nbins0) <= nbinsf):
            nbins_tot += np.int32(x * nbins0)
            x *= alpha

        # q_phi = get_qphi(nbins0_max, nbinsf_max, alpha, noverlap)
        # qvals, phivals = zip(*q_phi)

        # nbins_tot = len(q_phi) / noverlap

        nf = batch_size if batch < nbatches - 1 else \
            len(freqs) - batch * batch_size

        nf = np.int32(nf)
        j = batch % nstreams
        yw_g_bin = yw_g_bins[j]
        w_g_bin = w_g_bins[j]
        bls_tmp_g = bls_tmp_gs[j]
        bls_tmp_sol_g = bls_tmp_sol_gs[j]

        stream = streams[j]

        yw_g_bin.fill(np.float32(0), stream=stream)
        w_g_bin.fill(np.float32(0), stream=stream)

        bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

        args = (bin_grid, block, stream)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr)
        args += (ndata, nf, np.int32(nbins0), np.int32(nbinsf))
        args += (np.int32(batch_size * batch), noverlap)
        args += (alpha, nbins_tot)
        gpu_bin.prepared_async_call(*args)

        args = (grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  nf * nbins_tot * noverlap)
        gpu_bls.prepared_async_call(*args)

        bls_tmp = None if not plot_status else bls_tmp_g.get()

        args = (gpu_max, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        reduction_max(*args)

        args = (grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (np.int32(nbins0), np.int32(nbinsf), np.int32(noverlap))
        args += (np.float32(alpha), np.int32(nf))
        args += (np.int32(batch * batch_size),)
        gpu_store.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = zip(best_q, best_phi)

    return bls_g.get()/YY, qphi_sols
