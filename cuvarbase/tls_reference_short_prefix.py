"""Guarded reuse of CUB's native first-and-last-tile float32 scan.

Rows beyond one tile retain the native graph implementation. This wrapper
uses the installed CUB agent, including its load mapping and addition tree;
an explicit matrix-axis cumsum would use a different floating-point tree.
"""
import operator
import os
from pathlib import Path
import time

import cupy as cp
import numpy as np

from .utils import find_kernel


class NativeShortPrefixCache:
    """Small per-thread cache of audited modules belonging to CUDA contexts.

    ``prefix(array)`` returns a new output, or None to request native graphs.
    No input/output buffers or mutable scan state are retained. Unsupported
    builds and compilation/canary failures fall back; CUDA execution faults
    propagate. The cache retains at most ``max_contexts`` entries, including
    failed compilations. Additional contexts use native graphs. Retaining
    entries avoids unloading a module while an earlier stream uses it.
    """

    def __init__(self, max_contexts=2):
        self.max_contexts = operator.index(max_contexts)
        if self.max_contexts < 1:
            raise ValueError('Short prefix cache must retain at least one context')
        self.entries = {}
        self.compile_seconds = self.canary_seconds = 0.
        self.dispatch_calls = self.fallback_calls = 0
        self._status = dict(active=False, supported=False, fallback_reason='unused',
                            device=None, context=None)

    @staticmethod
    def unsupported_reason(array):
        if not isinstance(array, cp.ndarray) or array.ndim != 2:
            return 'input is not a CuPy matrix'
        rows, columns = array.shape
        if array.dtype != cp.float32 or not array.flags.c_contiguous:
            return 'input is not contiguous float32'
        if not 0 < rows <= np.iinfo(np.int32).max or not 0 < columns <= 1920:
            return 'shape exceeds the native single-tile domain'
        if cp.__version__ != '13.6.0' or cp.cuda.runtime.is_hip:
            return 'CuPy build is not the audited CUDA 13.6.0 build'
        if any(os.environ.get(name) for name in
               ('NVCC', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')):
            return 'custom NVCC command or injected compiler flags'
        from cupy._core import _accelerator
        from cupy.cuda import cub
        if cub.get_build_version() != 200800 or cp.cuda.driver.get_build_version() != 12090:
            return 'CUB or CUDA build differs from the audited wheel'
        if cp.cuda.Device().compute_capability != '86':
            return 'GPU architecture is not SM86'
        if array.device.id != cp.cuda.runtime.getDevice():
            return 'array and current CUDA device differ'
        # This is mutable process state, so recheck it on every dispatch.
        if _accelerator.ACCELERATOR_CUB not in _accelerator.get_routine_accelerators():
            return 'CUB routine accelerator is disabled'
        return None

    @staticmethod
    def compile():
        if not cp.cuda.get_nvcc_path():
            raise OSError('nvcc is unavailable')
        include = Path(cp.__file__).parent / '_core/include/cupy/_cccl'
        # Use the installed headers, retaining their native implementation
        # and license notices. The C++ wrapper pins nvcc and the CUB policy.
        with open(find_kernel('tls_reference_short_prefix')) as source:
            code = source.read()
        # RawModule's cache compiler unconditionally appends -ftz=true in
        # CuPy 13.6, unlike the native CUB wheel. Compile directly so subnormal
        # additions retain the native behavior; the bounded context cache
        # below amortizes this compilation without changing compiler flags.
        options = tuple(['--std=c++17', '-ftz=false'] +
                        ['-I' + str(include / name)
                         for name in ('cub', 'thrust', 'libcudacxx')])
        cubin = cp.cuda.compiler.compile_using_nvcc(code, options=options,
                                                   arch='86', code_type='cubin')
        module = cp.cuda.function.Module()
        module.load(cubin)
        return module, module.get_function('native_cub_short_rows')

    @staticmethod
    def canary(kernel):
        """Check the installed compiler/wheel before dispatching real inputs.

        This finite check supplements the pinned source argument. It does
        not establish equivalence for other algorithms, builds or shapes.
        Downloads synchronize every launched comparison, including failure.
        """
        for columns in (31, 513, 1475, 1920):
            index = np.arange(columns, dtype=np.int32)
            values = np.empty((4, columns), dtype=np.float32)
            values[0] = 1. + (index % 17 - 8) * np.float32(2**-16)
            values[1] = np.resize(np.array([2**18, -2**18, .001, -.003, 1.],
                                           dtype=np.float32), columns)
            words = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                              0x807fffff], dtype=np.uint32)
            values[2] = np.resize(words.view(np.float32), columns)
            values[3] = np.resize(np.array([1., -1., 2**-24, 2**24, -2**24],
                                           dtype=np.float32), columns)
            array = cp.asarray(values)
            actual = cp.empty_like(array)
            kernel((4,), (128,), (array, actual, np.int32(4), np.int32(columns)))
            expected = cp.empty_like(array)
            for row in range(4):
                cp.cumsum(array[row], out=expected[row])
            if not np.array_equal(actual.get().view(np.uint32),
                                  expected.get().view(np.uint32)):
                return False
        return True

    def _fallback(self, reason, *, supported=False):
        self.fallback_calls += 1
        self._status.update(active=False, supported=supported, fallback_reason=reason)
        return None

    def prefix(self, array):
        reason = self.unsupported_reason(array)
        if reason:
            return self._fallback(reason)
        key = (int(cp.cuda.runtime.getDevice()), int(cp.cuda.driver.ctxGetCurrent()))
        self._status.update(device=key[0], context=key[1])
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_contexts:
                return self._fallback('context cache limit reached', supported=True)
            before = time.perf_counter()
            try:
                module, kernel = self.compile()
            except (OSError, cp.cuda.compiler.CompileException) as error:
                entry = dict(module=None, kernel=None, failure=repr(error))
            else:
                entry = dict(module=module, kernel=kernel, failure=None)
            self.compile_seconds += time.perf_counter() - before
            if entry['kernel'] is not None:
                before = time.perf_counter()
                good = self.canary(entry['kernel'])
                self.canary_seconds += time.perf_counter() - before
                if not good:
                    entry = dict(module=None, kernel=None, failure='native scan canary mismatch')
            self.entries[key] = entry
        if entry['kernel'] is None:
            return self._fallback(entry['failure'], supported=True)
        rows, columns = array.shape
        result = cp.empty_like(array)
        # Runtime/launch errors deliberately propagate; they are not an
        # unsupported-build condition and must not silently change engines.
        entry['kernel']((rows,), (128,),
                        (array, result, np.int32(rows), np.int32(columns)))
        self.dispatch_calls += 1
        self._status.update(active=True, supported=True, fallback_reason=None)
        return result

    @property
    def status(self):
        """Private serializable diagnostics, separate from scientific results."""
        return dict(self._status, cached_context_count=len(self.entries),
                    cached_module_count=sum(item['kernel'] is not None
                                            for item in self.entries.values()),
                    compile_seconds=self.compile_seconds, canary_seconds=self.canary_seconds,
                    dispatch_calls=self.dispatch_calls, fallback_calls=self.fallback_calls)
