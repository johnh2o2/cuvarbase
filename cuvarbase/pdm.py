import numpy as np
import resource
import warnings
from typing import Literal

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .utils import weights, find_kernel, dphase, normalize_light_curves, autofrequency


def var_tophat(t, y, w, freq, dphi):
    var = 0.
    for i, (T, Y, W) in enumerate(zip(t, y, w)):
        mbar = 0.
        wtot = 0.
        for j, (T2, Y2, W2) in enumerate(zip(t, y, w)):
            dph = dphase(abs(T2 - T), freq)
            if dph < dphi:
                mbar += W2 * Y2
                wtot += W2

        var += W * (Y - mbar / wtot)**2

    return var


def var_gauss(t, y, w, freq, dphi):
    gaussian = lambda x: np.exp(-0.5 * x**2)
    var = 0.
    for i, (T, Y, W) in enumerate(zip(t, y, w)):
        mbar = 0.
        wtot = 0.

        for j, (T2, Y2, W2) in enumerate(zip(t, y, w)):
            dph = dphase(abs(T2 - T), freq)
            wgt   = W2 * gaussian(dph / dphi)
            mbar += wgt * Y2
            wtot += wgt

        var += W * (Y - mbar / wtot)**2

    return var


def binned_pdm_model(t, y, w, freq, nbins, linterp=True):

    if len(t) == 0:
        return lambda p, **kwargs: np.zeros_like(p)

    bin_means = np.zeros(nbins)
    phase = (t * freq) % 1.0
    bins = [int(p * nbins) % nbins for p in phase]

    for i in range(nbins):
        wtot = max([sum([W for j, W in enumerate(w) if bins[j] == i]), 1E-10])
        bin_means[i] = sum([W * Y for j, (Y, W) in enumerate(zip(y, w))
                            if bins[j] == i]) / wtot

    def pred_y(p, nbins=nbins, linterp=linterp, bin_means=bin_means):
        bs = np.array([int(P * nbins) % nbins for P in p])
        if not linterp:
            return bin_means[bs]
        alphas = p * nbins - np.floor(p * nbins) - 0.5
        di = np.floor(alphas).astype(np.int32)
        bins0 = bs + di
        bins1 = bins0 + 1

        alphas[alphas < 0] += 1
        bins0[bins0 < 0] += nbins
        bins1[bins1 >= nbins] -= nbins

        return (1 - alphas) * bin_means[bins0] + alphas * bin_means[bins1]

    return pred_y


def var_binned(t, y, w, freq, nbins, linterp=True):
    ypred = binned_pdm_model(t, y, w, freq, nbins, linterp=linterp)((t * freq) % 1.0)
    return np.dot(w, np.power(y - ypred, 2))


def binless_pdm_cpu(t, y, w, freqs, dphi=0.05, tophat=True):
    # Prepare data
    t -= np.mean(t)
    y -= np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    if tophat:
        return [1 - var_tophat(t, y, w, freq, dphi) / var for freq in freqs]
    else:
        return [1 - var_gauss(t, y, w, freq, dphi) / var for freq in freqs]


def pdm2_cpu(t, y, w, freqs, nbins=30, linterp=True):
    # Prepare data
    t -= np.mean(t)
    y -= np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    return [1 - var_binned(t, y, w, freq,
                           nbins=nbins, linterp=linterp) / var
            for freq in freqs]


def pdm2_single_freq(t, y, w, freq, nbins=30, linterp=True):
    # Prepare data
    t -= np.mean(t)
    y -= np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    return 1 - var_binned(t, y, w, freq, nbins=nbins, linterp=linterp) / var


def pdm_async(stream, data_cpu, data_gpu, pow_cpu, function,
              dphi=0.05, block_size=256, **kwargs):
    t, y, w, freqs = data_cpu
    t_g, y_g, w_g, freqs_g, pow_g = data_gpu

    if t_g is None:
        return pow_cpu

    # constants
    nfreqs = np.int32(len(freqs))
    ndata = np.int32(len(t))
    dphi = np.float32(dphi)

    # kernel size
    grid_size = int(np.ceil(float(nfreqs) / block_size))
    grid = (grid_size, 1)
    block = (block_size, 1, 1)

    # weighted mean + weighted variance
    ybar = np.dot(w, y)
    var = np.float32(np.dot(w, np.power(y - ybar, 2)))

    # transfer data
    w_g.set_async(np.asarray(w).astype(np.float32), stream=stream)
    t_g.set_async(np.asarray(t).astype(np.float32), stream=stream)

    # Ensure y is zero-weighted-meaned for fast kernels (one-pass SS_between)
    y_norm = (np.asarray(y) - ybar).astype(np.float32)
    y_g.set_async(y_norm, stream=stream)

    function.prepared_async_call(grid, block, stream,
                                 t_g.ptr, y_g.ptr, w_g.ptr,
                                 freqs_g.ptr, pow_g.ptr,
                                 ndata, nfreqs, dphi, var)

    pow_g.get_async(stream=stream, ary=pow_cpu)

    return pow_cpu


class PDMAsyncProcess(GPUAsyncProcess):

    def __init__(self, *args, **kwargs):
        super(PDMAsyncProcess, self).__init__(*args, **kwargs)

    def _compile_and_prepare_functions(self, nbins=10):
        pdm2_txt = open(find_kernel('pdm'), 'r').read()
        pdm2_txt = pdm2_txt.replace('//INSERT_NBINS_HERE',
                                    '#define NBINS %d' % nbins)

        self.module = SourceModule(pdm2_txt, options=['--use_fast_math'])

        self.dtypes = [np.intp, np.intp, np.intp, np.intp, np.intp,
                       np.int32, np.int32, np.float32, np.float32]
        for function in ['pdm_binless_tophat', 'pdm_binless_gauss',
                         'pdm_binned_linterp_%dbins' % nbins,
                         'pdm_binned_step_%dbins' % nbins,
                         'pdm_binned_linterp_fast_%dbins' % nbins,
                         'pdm_binned_step_fast_%dbins' % nbins,
                         'pdm_binless_tophat_fast',
                         'pdm_binless_gauss_fast']:
            func = function.replace('_%dbins' % nbins, '')
            func = self.module.get_function(func).prepare(self.dtypes)
            self.prepared_functions[function] = func

    def allocate(self, data, freqs=None, **kwargs):
        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, pow_cpus = [], []

        is_deprecated = len(data) > 0 and len(data[0]) == 4

        plot_data = []
        if is_deprecated:
            plot_data = data
        else:
            frqs = freqs
            if frqs is None:
                frqs = [autofrequency(d[0], **kwargs) for d in data]
            elif isinstance(frqs[0], (float, np.floating)):
                frqs = [frqs] * len(data)

            for i, (t, y, err) in enumerate(data):
                # We only need lengths for allocation
                plot_data.append((t, y, None, frqs[i]))

        for t, y, w, freqs in plot_data:

            pow_cpu = cuda.aligned_zeros(shape=(len(freqs),),
                                         dtype=np.float32,
                                         alignment=resource.getpagesize())

            t_g, y_g, w_g = None, None, None
            if len(t) > 0:
                t_g, y_g, w_g = tuple([gpuarray.zeros(len(t), dtype=np.float32)
                                       for _ in range(3)])

            pow_g = gpuarray.zeros(len(pow_cpu), dtype=pow_cpu.dtype)
            freqs_g = gpuarray.to_gpu(np.asarray(freqs).astype(np.float32))

            gpu_data.append((t_g, y_g, w_g, freqs_g, pow_g))
            pow_cpus.append(pow_cpu)
        return gpu_data, pow_cpus

    def run(self, data, gpu_data=None, pow_cpus=None, freqs=None,
            kind: Literal['binless_tophat', 'binless_gauss',
                          'binless_tophat_fast', 'binless_gauss_fast',
                          'binned_linterp', 'binned_step',
                          'binned_linterp_fast', 'binned_step_fast'] = 'binned_linterp',
            nbins=10, dphi=0.05, **pdm_kwargs):

        if kind in ['binless_tophat', 'binless_gauss',
                    'binless_tophat_fast', 'binless_gauss_fast']:
            function = 'pdm_%s' % kind
        elif kind in ['binned_linterp', 'binned_step',
                      'binned_linterp_fast', 'binned_step_fast']:
            function = 'pdm_%s_%dbins' % (kind, nbins)
        else:
            raise KeyError('Function not available. Please use one of the followings: '
                           'binless_tophat, binless_gauss, '
                           'binless_tophat_fast, binless_gauss_fast, '
                           'binned_linterp, binned_step, '
                           'binned_linterp_fast, binned_step_fast')

        if function not in self.prepared_functions:
            self._compile_and_prepare_functions(nbins=nbins)

        # Backward compatibility check
        is_deprecated = len(data) > 0 and len(data[0]) == 4
        if is_deprecated:
            warnings.warn("The (t, y, w, freqs) format is deprecated "
                          "and will be removed in the future. "
                          "Please use the (t, y, err) format "
                          "and pass freqs as a separate argument "
                          "or pass optional keyword arguments "
                          "passed to ``autofrequency``.",
                          DeprecationWarning, stacklevel=2)

        # Prepare data and determine frequencies
        if is_deprecated:
            norm_data = normalize_light_curves(data)
            frqs = [d[3] for d in data]
        else:
            frqs = freqs
            if frqs is None:
                frqs = [autofrequency(d[0], **pdm_kwargs) for d in data]
            elif isinstance(frqs[0], (float, np.floating)):
                frqs = [frqs] * len(data)

            # Normalize t and y
            norm_data_temp = normalize_light_curves(data)
            norm_data = []
            for i, (t, y, err) in enumerate(norm_data_temp):
                w = weights(err)
                norm_data.append((t, y, w, frqs[i]))

        if pow_cpus is None or gpu_data is None:
            gpu_data, pow_cpus = self.allocate(norm_data, freqs=frqs, **pdm_kwargs)

        streams = [s for i, s in enumerate(self.streams) if i < len(data)]
        func = self.prepared_functions[function]

        results = [pdm_async(stream, cdat, gdat, pcpu, func, dphi=dphi, **pdm_kwargs)
                   for stream, cdat, gdat, pcpu in
                   zip(streams, norm_data, gpu_data, pow_cpus)]

        if is_deprecated:
            return results
        return list(zip(frqs, results))
