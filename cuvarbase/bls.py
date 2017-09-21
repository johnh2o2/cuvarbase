import sys
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

from pycuda.compiler import SourceModule
from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader

import resource
import numpy as np
from time import time

_default_block_size = 256


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


def fmin_transit(t, rho=1., min_obs_per_transit=5, **kwargs):
    T = max(t) - min(t)
    qmin = float(min_obs_per_transit) / len(t)
    fmin1 = np.power(np.sin(np.pi * qmin), 3./2.) * fmax_transit(rho=rho)
    fmin2 = 1./(max(t) - min(t))
    return max([fmin1, fmin2])


def fmax_transit0(rho=1., **kwargs):
    return 8.6307 * np.sqrt(rho)


def q_transit(freq, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return (1./np.pi) * np.arcsin(np.power(freq / fmax0, 2./3.))


def freq_transit(q, rho=1., **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return fmax0 * (np.sin(np.pi * q) ** 1.5)


def fmax_transit(rho=1., qmax=0.5, **kwargs):
    fmax0 = fmax_transit0(rho=rho)
    return min([fmax0, freq_transit(qmax, rho=rho, **kwargs)])


def transit_autofreq(t, fmin=1E-2, fmax=1E2, samples_per_peak=2,
                     rho=1., qmin_fac=0.2, **kwargs):
    T = max(t) - min(t)
    freqs = [fmin]
    while freqs[-1] < fmax:
        df = qmin_fac * q_transit(freqs[-1], rho=rho) / (samples_per_peak * T)
        freqs.append(freqs[-1] + df)
    freqs = np.array(freqs)
    q0vals = q_transit(freqs, rho=rho)
    return freqs, q0vals


def bin_data(t, y, freq, nbins, phi0=0.):
    phi_d = (t * freq) % 1.0 - phi0
    phi_d[phi_d < 0] += 1.

    bins = np.zeros(nbins)

    bs = np.floor(phi_d * nbins).astype(np.int32)

    for b, Y in zip(bs, y):
        bins[b] += Y

    return bins


def compile_bls(block_size=_default_block_size, **kwargs):
    """
    Compile BLS kernel

    Parameters
    ----------
    block_size: int, optional (default: _default_block_size)
        CUDA threads per CUDA block.

    Returns
    -------
    (gpu_bls, gpu_max, gpu_bin,
     gpu_bin_custom, gpu_store, gpu_store_custom)

    """
    # Read kernel
    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel('bls'),
                                cpp_defs=cppd)

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    # get functions
    gpu_bls = module.get_function('binned_bls_bst')
    gpu_max = module.get_function('reduction_max')
    gpu_bin = module.get_function('bin_and_phase_fold_bst_multifreq')
    gpu_bin_custom = module.get_function('bin_and_phase_fold_custom')
    gpu_store = module.get_function('store_best_sols')
    gpu_store_custom = module.get_function('store_best_sols_custom')

    # Prepare the functions
    gpu_bls = gpu_bls.prepare([np.intp, np.intp, np.intp, np.int32])
    gpu_max = gpu_max.prepare([np.intp, np.intp, np.int32, np.int32, np.int32,
                               np.intp, np.intp, np.int32, np.int32])
    gpu_bin = gpu_bin.prepare([np.intp, np.intp, np.intp, np.intp,
                               np.intp, np.intp, np.int32, np.int32,
                               np.int32, np.int32, np.int32, np.int32,
                               np.float32, np.int32])

    gpu_bin_custom = gpu_bin_custom.prepare([np.intp, np.intp, np.intp,
                                             np.intp, np.intp, np.intp,
                                             np.intp, np.intp,
                                             np.int32, np.int32, np.int32,
                                             np.int32, np.int32])

    gpu_store = gpu_store.prepare([np.intp, np.intp, np.intp, np.int32,
                                   np.int32, np.int32, np.float32, np.int32,
                                   np.int32])

    gpu_store_custom = gpu_store_custom.prepare([np.intp, np.intp, np.intp,
                                                 np.intp, np.intp, np.int32,
                                                 np.int32, np.int32, np.int32])
    return (gpu_bls, gpu_max, gpu_bin,
            gpu_bin_custom, gpu_store, gpu_store_custom)


def eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                     batch_size=5, nstreams=5, functions=None,
                     **kwargs):
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
    batch_size: int, optional (default: 5)
        Number of frequencies to compute in a single batch
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

    (gpu_bls, gpu_max, gpu_bin,
     gpu_bin_custom, gpu_store,
     gpu_store_custom) = functions

    block_size = kwargs.get('block_size', _default_block_size)
    nbtot = len(q_values) * len(phi_values) * batch_size

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
        gpu_bin_custom.prepared_async_call(*args)

        nb = len(q_values) * len(phi_values)

        bls_grid = (int(np.ceil(float(nf * nb) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(nf * nb))
        gpu_bls.prepared_async_call(*args)

        args = (gpu_max, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nb, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (q_values_g.ptr, phi_values_g.ptr)
        args += (np.int32(len(q_values)), np.int32(len(phi_values)))
        args += (np.int32(nf), np.int32(batch * batch_size))
        gpu_store_custom.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = zip(best_q, best_phi)

    return bls_g.get()/YY, qphi_sols


def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
              nstreams=5, noverlap=3, dlogq=0.5,
              batch_size=5, functions=None, **kwargs):

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
        logarithmic spacing of q values, where dlog q = dq / q
    batch_size: int, optional (default: 5)
        Number of frequencies to compute in a single batch
    functions: tuple of CUDA functions
        gpu_bls, gpu_max, gpu_bin, gpu_store functions;
        returned by `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    qphi_sols: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """

    def locext(ext, arr, imin=None, imax=None):
        if isinstance(arr, float) or isinstance(arr, int):
            return arr
        return ext(arr[slice(imin, imax)])

    functions = functions if functions is not None \
        else compile_bls(**kwargs)

    (gpu_bls, gpu_max, gpu_bin,
     gpu_bin_custom, gpu_store,
     gpu_store_custom) = functions

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
        gpu_bin.prepared_async_call(*args)

        all_bins = nf * nbins_tot * noverlap

        bls_grid = (int(np.ceil(float(all_bins) / block_size)), 1)
        args = (bls_grid, block, stream)
        args += (yw_g_bin.ptr, w_g_bin.ptr)
        args += (bls_tmp_g.ptr,  np.int32(all_bins))
        gpu_bls.prepared_async_call(*args)

        args = (gpu_max, bls_tmp_g, bls_tmp_sol_g)
        args += (nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g)
        args += (batch * batch_size, block_size)
        reduction_max(*args)

        store_grid = (int(np.ceil(float(nf) / block_size)), 1)
        args = (store_grid, block, stream)
        args += (bls_sol_g.ptr, bls_best_phi.ptr, bls_best_q.ptr)
        args += (np.int32(nbins0), np.int32(nbinsf), np.int32(noverlap))
        args += (np.float32(dlogq), np.int32(nf))
        args += (np.int32(batch * batch_size),)
        gpu_store.prepared_async_call(*args)

    best_q = bls_best_q.get()
    best_phi = bls_best_phi.get()

    qphi_sols = zip(best_q, best_phi)

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

        print (phimin, phimax, qmin, qmax, q, dphi, nphi)
        powers, sols = eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                                        batch_size=5, nstreams=5,
                                        functions=functions, **kwargs)

        ibest = np.argmax(powers)
        f = freqs[ibest]
        q, phi = sols[ibest]
        pn = powers[ibest]
        i += 1
        print i, (pn - p0) / p0, q, phi, pn
    return f, pn, i, (q, phi)


def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.5,
                      qmin_fac=0.5, qmax_fac=2.0, fmin=None,
                      fmax=None, **kwargs):
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
    **kwargs:
        passed to `eebls_gpu`, `compile_bls`, `fmax_transit`,
        `fmin_transit`, and `transit_autofreq`


    Returns
    -------
    freqs: array_like, float
        Frequencies where BLS is evaluated
    bls: array_like, float
        BLS periodogram, normalized to 1 - chi2(best_fit) / chi2(constant)
    solutions: list of (q, phi) tuples
        Best (q, phi) solution at each frequency

    """
    funcs = compile_bls(**kwargs)
    if fmin is None:
        fmin = fmin_transit(t, **kwargs) * fmin_frac
    if fmax is None:
        fmax = fmax_transit(qmax=0.5 / qmax_fac, **kwargs) * fmax_frac
    freqs, q0vals = transit_autofreq(t, fmin=fmin, fmax=fmax,
                                     qmin_fac=qmin_fac, **kwargs)

    qmins = q0vals * qmin_fac
    qmaxes = q0vals * qmax_fac

    powers, sols = eebls_gpu(t, y, dy, freqs,
                             qmin=qmins, qmax=qmaxes,
                             functions=funcs,
                             **kwargs)
    return freqs, powers, sols
