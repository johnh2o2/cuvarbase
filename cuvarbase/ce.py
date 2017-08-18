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
from time import time


class ConditionalEntropyMemory(object):
    def __init__(self, **kwargs):
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 10)
        self.max_phi = kwargs.get('max_phi', 3.)
        self.stream = kwargs.get('stream', None)
        self.ndata = kwargs.get('ndata', None)
        self.nf = kwargs.get('nf', None)

    def allocate_data(self, **kwargs):
        raise NotImplementedError()


def conditional_entropy(t, y, freqs, dy=None, block_size=256, phase_bins=10,
                        mag_bins=10, max_phi=3., stream=None):
    # Read kernel
    kernel_txt = _module_reader(find_kernel('ce'),
                                cpp_defs=dict(BLOCK_SIZE=block_size,
                                              NPHASE=phase_bins,
                                              NMAG=mag_bins))

    # compile kernel
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    # get functions
    hist_weight = module.get_function('histogram_data_weighted')
    hist_count = module.get_function('histogram_data_count')
    ce_wt = module.get_function('compute_weighted_ce')
    ce_std = module.get_function('compute_standard_ce')

    # Prepare the functions
    hist_weight = hist_weight.prepare([np.intp, np.intp, np.intp, np.intp,
                                       np.intp, np.int32, np.int32,
                                       np.float32])

    hist_count = hist_count.prepare([np.intp, np.intp, np.intp, np.intp,
                                     np.int32, np.int32])

    ce_wt = ce_wt.prepare([np.intp, np.int32, np.intp])

    ce_std = ce_std.prepare([np.intp, np.int32, np.intp])

    t0 = time()

    dymax = 0. if dy is None else max_phi * max(dy)
    yscale = max(y) - min(y) + 2 * dymax
    ynorm = (y - min(y) + dymax) / yscale

    t_g = gpuarray.to_gpu(np.asarray(t).astype(np.float32))
    y_g = gpuarray.to_gpu(np.asarray(ynorm).astype(np.float32))

    nfreqs = len(freqs)
    ndata = len(t)
    nbins = nfreqs * phase_bins * mag_bins
    ce_g = gpuarray.zeros(nfreqs, dtype=np.float32)
    freqs_g = gpuarray.to_gpu(np.asarray(freqs).astype(np.float32))

    block = (block_size, 1, 1)
    grid = (int(np.ceil((ndata * nfreqs) / float(block_size))), 1)

    if dy is not None:
        dynorm = dy / yscale
        dy_g = gpuarray.to_gpu(np.asarray(dynorm).astype(np.float32))
        bins_g = gpuarray.zeros(nbins, dtype=np.float32)

        hist_weight.prepared_async_call(grid, block, stream,
                                        t_g.ptr, y_g.ptr, dy_g.ptr,
                                        bins_g.ptr, freqs_g.ptr,
                                        np.int32(nfreqs), np.int32(ndata),
                                        np.float32(max_phi))

        grid = (int(np.ceil(nfreqs / float(block_size))), 1)
        ce_wt.prepared_async_call(grid, block, stream,
                                  bins_g.ptr, np.int32(nfreqs), ce_g.ptr)
        ce = ce_g.get()
        print(time() - t0)
        return ce

    bins_g = gpuarray.zeros(nbins, dtype=np.uint32)
    hist_count.prepared_async_call(grid, block, stream,
                                   t_g.ptr, y_g.ptr, bins_g.ptr,
                                   freqs_g.ptr, np.int32(nfreqs),
                                   np.int32(ndata))

    grid = (int(np.ceil(nfreqs / float(block_size))), 1)
    ce_std.prepared_async_call(grid, block, stream,
                               bins_g.ptr, np.int32(nfreqs), ce_g.ptr)

    ce = ce_g.get()

    print(time() - t0)
    return ce
