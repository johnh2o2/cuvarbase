from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from .utils import gaussian_window, tophat_window, get_autofreqs
import pycuda.driver as cuda
import pycuda.tools as cutools
from pycuda.compiler import SourceModule
from functools import wraps
cuda.init()

class GPUAsyncProcess(object):
    _runfuncs = ['run']
    def __init__(self, *args, **kwargs):

        # initialize cuda
        cuda.init()

        self.reader = kwargs.get('reader', None)
        self.nstreams = kwargs.get('nstreams', None)
        self.function_kwargs = kwargs.get('function_kwargs', {})
        self.context = kwargs.get('context', cuda.Context.get_current())

        if self.context is None:
            self.context = cutools.make_default_context()

        self.device = kwargs.get('device', self.context.get_device())
        if isinstance(self.device, int):
            self.device = cuda.Device(self.device)

        assert(isinstance(self.device, cuda.Device))
        assert(self.context.get_device() == self.device)
        
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

    def synchronize_streams(self):
        for i, stream in enumerate(self.streams):
            stream.synchronize()

    def finish(self):
        """ synchronize all active streams """

        for stream in self.streams:
            stream.synchronize()

        return self

    def set_context(self, context):
        """
        Set the GPU context of this object.
        """

        if hasattr(self, 'context') and self.context is not None:
            self.scrub()
        
        self.context = context
        self.device = self.context.get_device()

        return self

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

            return self.batched_run_const_nfreq(data,
                                                freqs=freqs,
                                                batch_size=batch_size,
                                                **kwargs)

        elif len(data) > batch_size or (self.memory is not None and len(data) > len(self.memory)):
            return self.batched_run(data, freqs=freqs, batch_size=batch_size, **kwargs)

        return self.run(data, freqs=freqs, batch_size=batch_size, **kwargs)


def push_context_hook_for(cls, target_function):

    @wraps(target_function)
    def hooked(s, *args, **kwargs):
        if not hasattr(s, 'context') or s.context is None:
            if hasattr(s, 'device') and s.device is not None:
                s.context = s.device.make_context()
            else:
                s.context = cutools.make_default_context()
                s.device = s.context.get_device()

        if s.context != cuda.Context.get_current():
            s.context.push()

        return target_function(s, *args, **kwargs)

    return hooked

def pop_context_hook_for(cls, target_function):

    @wraps(target_function)
    def hooked(s, *args, **kwargs):
        rval = target_function(s, *args, **kwargs)
        s.context.pop()
        return rval
    return hooked


def for_all_runfuncs(decorator):
    def wrapper(cls):
        for func_name in cls._runfuncs:
            setattr(cls, func_name, decorator(cls, getattr(cls, func_name)))
        return cls
    return wrapper

ensure_context_push = for_all_runfuncs(push_context_hook_for)
ensure_context_pop = for_all_runfuncs(pop_context_hook_for)