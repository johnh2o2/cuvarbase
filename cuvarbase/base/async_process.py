import warnings

from .context import ensure_context
import pycuda.driver as cuda


class GPUAsyncProcess:
    """Base class of every GPU periodogram process.

    ``reader``, ``function_kwargs`` and ``device`` have been accepted
    since 0.2.5 but are not read by any process; they are kept for 1.x
    and will be removed in 2.0. The device is selected by the
    ``CUDA_DEVICE`` environment variable (via ``pycuda.autoprimaryctx``,
    see :func:`cuvarbase.base.ensure_context`), so a ``device`` other
    than 0 is ignored with a ``UserWarning``.
    """

    def __init__(self, *args, **kwargs):
        # Constructing any GPU process is a "first GPU use" -- retain the
        # CUDA primary context now (no longer done eagerly at import).
        ensure_context()
        self.reader = kwargs.get('reader', None)
        self.nstreams = kwargs.get('nstreams', None)
        self.function_kwargs = kwargs.get('function_kwargs', {})
        self.device = kwargs.get('device', 0)
        if self.device is not None and self.device != 0:
            warnings.warn("GPUAsyncProcess(device=%r) is ignored: the "
                          "device is selected by the CUDA_DEVICE "
                          "environment variable (pycuda.autoprimaryctx). "
                          "The device= keyword is deprecated and will be "
                          "removed in 2.0" % (self.device,),
                          UserWarning, stacklevel=2)
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
