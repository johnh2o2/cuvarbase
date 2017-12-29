from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from .utils import gaussian_window, tophat_window, get_autofreqs
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class GPUAsyncProcess(object):
    def __init__(self, *args, **kwargs):
        self.reader = kwargs.get('reader', None)
        self.nstreams = kwargs.get('nstreams', None)
        self.function_kwargs = kwargs.get('function_kwargs', {})
        self.device = kwargs.get('device', 0)
        self.memory = kwargs.get('memory', None)
        self.streams = []
        self.gpu_data = []
        self.results = []
        self._adjust_nstreams = self.nstreams is None
        if self.nstreams is not None:
                self._create_streams(self.nstreams)
        self.prepared_functions = {}

    def _create_streams(self, n):
        for i in range(n):
            self.streams.append(cuda.Stream())

    def _compile_and_prepare_functions(self):
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def finish(self):
        """ synchronize all active streams """
        for i, stream in enumerate(self.streams):
            stream.synchronize()

    def batched_run(self, data, batch_size=10, **kwargs):
        """ Run your data in batches (avoids memory problems) """
        nsubmit = 0
        results = []
        while nsubmit < len(data):
            batch = []
            while len(batch) < batch_size and nsubmit < len(data):
                batch.append(data[nsubmit])
                nsubmit += 1

            res = self.run(batch, **kwargs)
            self.finish()
            results.extend(res)

        return results

    def __call__(self, data, freqs=None, batch_size=10,
                 use_constant_freqs=False, **kwargs):

        # ensure data is [(x, y, dy), ...] format
        if isinstance(data[0][0], float):
            data = [data]

        if use_constant_freqs and hasattr(self, 'batched_run_const_nfreq'):
            assert(freqs is None or not is_list_of_lists(freqs))

            return self.batched_run_const_nfreq(data, freqs=freqs, batch_size=batch_size, **kwargs)

        elif len(data) > batch_size or (self.memory is not None and len(data) > len(self.memory)):
            return self.batched_run(data, freqs=freqs, batch_size=batch_size, **kwargs)

        return self.run(data, freqs=freqs, batch_size=batch_size, **kwargs)