"""Replay GTLS's native float32 row scans without a Python loop per batch.

CuPy's cumsum with axis=None and with an explicit matrix axis can use different
summation orders. This plan captures the original one-dimensional operations;
it does not substitute a segmented scan with different floating-point outputs.
"""
import operator
import time

import cupy as cp
import numpy as np


def _shape(shape):
    shape = tuple(operator.index(n) for n in shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('A native prefix plan requires a positive two-dimensional shape')
    return shape


class NativePrefixPlan:
    """A reusable float32 row-prefix CUDA graph with owned buffers.

    ``plan(array)`` copies ``array`` into the plan's input buffer and returns
    its output buffer. The copy and graph replay use CuPy's current stream.
    The returned output is overwritten by the next call. Consumers must finish
    reading it before reuse, or run in the same ordered stream. Instances must
    not be called concurrently; a thread-local bounded cache is appropriate.

    The graph, input/output arrays, capture stream and a private temporary
    memory pool remain alive together. The private pool prevents captured CUB
    workspace pointers from being recycled by unrelated GPU work.

    ``max_bytes`` caps the owned input/output/workspace storage. CUDA's internal
    graph bookkeeping is driver-managed and is not included in that accounting.
    ``close()`` synchronizes the device before releasing resources; callers
    that already completed all consumers may use ``close(synchronize=False)``.
    """

    @staticmethod
    def key_for(shape, device_id=None):
        """Key for a cache belonging to one thread and CUDA context."""
        if device_id is None:
            device_id = cp.cuda.runtime.getDevice()
        return (int(device_id), _shape(shape), np.dtype(np.float32).str)

    @staticmethod
    def buffer_bytes_for(shape):
        """Required input/output storage, before the small CUB workspace."""
        rows, columns = _shape(shape)
        return 2 * rows * columns * np.dtype(np.float32).itemsize

    def __init__(self, shape, *, device_id=None, max_bytes=None):
        self.key = self.key_for(shape, device_id)
        self.device_id, self.shape, _ = self.key
        self.max_bytes = None if max_bytes is None else operator.index(max_bytes)
        self.buffer_bytes = self.buffer_bytes_for(self.shape)
        if self.max_bytes is not None and self.buffer_bytes > self.max_bytes:
            raise MemoryError('Native prefix buffers exceed max_bytes')
        self.input = self.output = None
        self._pool = self._capture_stream = self._graph = self._last_stream = None
        self.workspace_bytes = self.owned_bytes = 0
        self.closed = False
        begin = time.perf_counter()
        try:
            with cp.cuda.Device(self.device_id):
                self.input = cp.zeros(self.shape, dtype=cp.float32)
                self.output = cp.empty(self.shape, dtype=cp.float32)
                self._pool = cp.cuda.MemoryPool()
                self._capture_stream = cp.cuda.Stream(non_blocking=True)
                cp.cuda.get_current_stream().synchronize()
                with self._capture_stream, cp.cuda.using_allocator(self._pool.malloc):
                    # Warm every row alignment and allocate CUB's workspace
                    # before capture, where fresh cudaMalloc is prohibited.
                    for row in range(self.shape[0]):
                        cp.cumsum(self.input[row], out=self.output[row])
                    self._capture_stream.synchronize()
                    self._check_footprint()
                    self._capture_stream.begin_capture()
                    try:
                        for row in range(self.shape[0]):
                            cp.cumsum(self.input[row], out=self.output[row])
                        self._graph = self._capture_stream.end_capture()
                    except BaseException:
                        # End an invalidated capture so the stream and CuPy
                        # context are usable when the caller handles failure.
                        try:
                            self._capture_stream.end_capture()
                        except Exception:
                            pass
                        raise
                self._check_footprint()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise
        self.setup_seconds = time.perf_counter() - begin

    def _check_footprint(self):
        self.workspace_bytes = self._pool.total_bytes()
        self.owned_bytes = self.buffer_bytes + self.workspace_bytes
        if self.max_bytes is not None and self.owned_bytes > self.max_bytes:
            raise MemoryError('Native prefix workspace exceeds max_bytes')

    @property
    def footprint(self):
        return dict(device_id=self.device_id, shape=self.shape, dtype='float32',
                    buffer_bytes=0 if self.closed else self.buffer_bytes,
                    workspace_bytes=self.workspace_bytes, owned_bytes=self.owned_bytes,
                    driver_graph_storage_included=False)

    def __call__(self, array):
        if self.closed:
            raise RuntimeError('Native prefix plan is closed')
        if not isinstance(array, cp.ndarray):
            raise TypeError('Native prefix input must be a CuPy array')
        if array.shape != self.shape or array.dtype != cp.float32:
            raise ValueError('Native prefix input must match the plan shape and float32 dtype')
        if array.device.id != self.device_id or cp.cuda.runtime.getDevice() != self.device_id:
            raise ValueError('Native prefix plan and current CuPy device differ')
        stream = cp.cuda.get_current_stream()
        if self._last_stream is not None and self._last_stream.ptr != stream.ptr:
            # Serial use may change streams. Finish the prior replay before
            # another stream overwrites the shared input/output buffers.
            self._last_stream.synchronize()
        cp.copyto(self.input, array)
        self._graph.launch(stream)
        self._last_stream = stream
        return self.output

    def close(self, *, synchronize=True):
        if self.closed:
            return
        with cp.cuda.Device(self.device_id):
            if synchronize:
                cp.cuda.runtime.deviceSynchronize()
            self._graph = None
            self.input = self.output = None
            if self._pool is not None:
                self._pool.free_all_blocks()
            self._pool = self._capture_stream = self._last_stream = None
        self.workspace_bytes = self.owned_bytes = 0
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Native prefix plan is closed')
        return self

    def __exit__(self, *exc):
        self.close()
