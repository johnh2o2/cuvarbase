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


class BaseSpectrogram(object):
    def __init__(self, t, y, w, times=None, freqs=None, window='gaussian',
                 window_length=None, batch_size=10, block_size=128,
                 truncate=True, truncate_limit=1E-3, proc=None,
                 proc_kwargs=None, **kwargs):
        self.t = np.array(t)
        self.y = np.array(y)
        self.w = np.array(w)

        self.baseline = self.t.max() - self.t.min()

        self.batch_size = batch_size

        self.window_name = window
        self.window = None
        if self.window_name is 'gaussian':
            self.window = gaussian_window
        elif self.window_name is 'tophat':
            self.window = tophat_window
        else:
            raise ValueError("Don't understand window %s"%(self.window_name))

        self.times = times
        self.window_length = window_length

        self.freqs = freqs
        if self.freqs is None:
            self.freqs = get_autofreqs(self.t, **kwargs)

        self.truncate = truncate
        self.truncate_limit = truncate_limit

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.proc_kwargs = {} if proc_kwargs is None else proc_kwargs
        self.proc = proc

    def auto_time_split(self, nsplits=100):

        if self.window_length is None:
            self.window_length = self.baseline / nsplits

        dt = self.baseline / nsplits

        times = np.linspace(self.t.min(), self.t.max(), nsplits) + 0.5 * dt

        return times

    def weighted_local_data(self, time):
        w_window = self.window(self.t, time, self.window_length)

        inds = np.arange(len(self.t))
        if self.truncate:
            inds = inds[w_window > self.truncate_limit]

        if len(inds) == 0:
            return self.t[inds], self.y[inds], self.w[inds]

        return self.t[inds], self.y[inds], np.multiply(w_window[inds], self.w[inds])

    def localized_spectrum(self, time=None, freqs=None):
        t, y, w = self.t, self.y, self.w
        if not time is None:
            t, y, w = self.weighted_local_data(time)

        freqs = self.freqs if freqs is None else freqs

        p = self.proc.run([ (t, y, w, freqs) ], **self.proc_kwargs)

        self.proc.finish()

        return p[0]

    def split_data(self, times):
        return  [ self.weighted_local_data(time) for time in times ]


    def spectrogram(self, times=None, freqs=None, nsplits=100):
        if times is None:
            times = self.times
        if times is None:
            times = self.auto_time_split(nsplits = nsplits)

        if freqs is None:
            freqs = self.freqs

        specgram   = np.zeros((len(times), len(freqs)))
        split_data = self.split_data(times)
        split_data = [ tuple(split) + (freqs,) for split in split_data ]

        powers = self.proc.batched_run(split_data, batch_size=self.batch_size,
                                                **self.proc_kwargs)

        for i, power in enumerate(powers):
            specgram[i, :] = power[:]

        return times, freqs, specgram

    def model(self, *args, **kwargs):
        raise NotImplementedError()
