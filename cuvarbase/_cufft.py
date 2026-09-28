"""Minimal in-house cuFFT binding (replaces the abandoned scikit-cuda).

The NFFT / Lomb-Scargle path needs only batched 1D complex-to-complex
transforms: create a plan, run an inverse FFT, and estimate the cuFFT
work-area size. This module binds just those entry points of ``libcufft``
via ``ctypes`` and exposes a small scikit-cuda-compatible surface so the
existing call sites are unchanged:

    Plan(shape, in_dtype, out_dtype, stream=None, batch=1)
    fft(x_gpu, y_gpu, plan)         # forward
    ifft(x_gpu, y_gpu, plan)        # inverse
    cufft.cufftEstimate1d(nx, cufft.CUFFT_C2C)   # work-area bytes

``libcufft`` is loaded lazily on first use, so importing this module (and
hence ``cuvarbase.cunfft`` / ``cuvarbase.lombscargle``) does NOT require
CUDA -- only calling into it does. This drops the unmaintained
``scikit-cuda`` 0.5.3 dependency (issue #63), whose numpy>=1.24
incompatibility previously required a runtime monkeypatch.

Motivation for a direct binding over ``cupy``: cupy is a large,
CUDA-version-specific dependency, whereas the cuFFT surface cuvarbase
uses is three functions; scikit-cuda itself was just a ctypes binding
of the same calls.
"""
import atexit
import ctypes
import ctypes.util
import glob
import os
import sys

import numpy as np

# --- cufft.h constants -----------------------------------------------------
# cufftType
CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69

# transform direction (cufftExec*)
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# cufftResult names for error messages
_RESULT = {
    0: 'CUFFT_SUCCESS', 1: 'CUFFT_INVALID_PLAN', 2: 'CUFFT_ALLOC_FAILED',
    3: 'CUFFT_INVALID_TYPE', 4: 'CUFFT_INVALID_VALUE',
    5: 'CUFFT_INTERNAL_ERROR', 6: 'CUFFT_EXEC_FAILED',
    7: 'CUFFT_SETUP_FAILED', 8: 'CUFFT_INVALID_SIZE',
    9: 'CUFFT_UNALIGNED_DATA', 10: 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    11: 'CUFFT_INVALID_DEVICE', 12: 'CUFFT_PARSE_ERROR',
    13: 'CUFFT_NO_WORKSPACE', 14: 'CUFFT_NOT_IMPLEMENTED',
    15: 'CUFFT_LICENSE_ERROR', 16: 'CUFFT_NOT_SUPPORTED',
}

_lib = None

# Set once the interpreter starts shutting down. Plan.__del__ must not call
# into libcufft after this, because the CUDA primary context may already be
# torn down -- a C-level fault that try/except cannot catch. The OS reclaims
# the plans at process exit anyway.
_shutting_down = False


@atexit.register
def _mark_shutting_down():
    global _shutting_down
    _shutting_down = True


class CufftError(RuntimeError):
    """A cufft* call returned a non-success cufftResult."""


def _check(status):
    if status != 0:
        raise CufftError("cuFFT call failed: %s (%d)"
                         % (_RESULT.get(status, 'UNKNOWN'), status))


def _candidate_libs():
    """Ordered libcufft candidates: explicit paths first, SONAMEs last.

    Covers (a) the loader's own resolution, (b) the pip wheel layout
    ``<site-packages>/nvidia/cufft/lib/libcufft.so.*`` (nvidia-cufft-cuXX),
    (c) CUDA-toolkit ``lib64`` dirs, and (d) bare SONAMEs found via
    ``LD_LIBRARY_PATH``/ldconfig. Absolute paths are tried before bare
    names so a runtime-only install works without ``LD_LIBRARY_PATH``.
    """
    cands = []
    found = ctypes.util.find_library('cufft')
    if found:
        cands.append(found)

    patterns = []
    # pip wheel: site-packages/nvidia/cufft/lib/libcufft.so*
    for p in sys.path:
        if p and os.path.isdir(p):
            patterns.append(os.path.join(p, 'nvidia', 'cufft', 'lib',
                                          'libcufft.so*'))
    # CUDA toolkit install dirs
    patterns += ['/usr/local/cuda*/lib64/libcufft.so*',
                 '/usr/local/cuda/lib64/libcufft.so*',
                 '/opt/cuda*/lib64/libcufft.so*']
    for pat in patterns:
        # reverse-sort so a higher SONAME version (.so.11) precedes .so
        cands.extend(sorted(glob.glob(pat), reverse=True))

    # bare SONAMEs (resolved via the dynamic loader / LD_LIBRARY_PATH)
    cands += ['libcufft.so', 'libcufft.so.12', 'libcufft.so.11',
              'libcufft.so.10', 'libcufft.dylib',
              'cufft64_12.dll', 'cufft64_11.dll', 'cufft64_10.dll']

    seen, ordered = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _load():
    """Lazily load libcufft and declare the prototypes we use."""
    global _lib
    if _lib is not None:
        return _lib

    last_err = None
    lib = None
    for name in _candidate_libs():
        try:
            # RTLD_GLOBAL so libcufft's own deps (libcudart, cublas, ...)
            # and symbols are visible to the rest of the process.
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as exc:
            last_err = exc
    if lib is None:
        raise ImportError(
            "could not load libcufft (required for the NFFT / Lomb-Scargle "
            "GPU path). Install the CUDA cuFFT runtime and ensure it is on "
            "the loader path (e.g. LD_LIBRARY_PATH must include the CUDA "
            "lib64 directory, or `pip install nvidia-cufft-cu12`). If cuFFT "
            "is present, a missing sibling runtime (libcudart/libcublas) can "
            "also cause this. Last loader error: %s" % last_err)

    # cufftHandle is a plain int; cudaStream_t is an opaque pointer.
    lib.cufftPlan1d.restype = ctypes.c_int
    lib.cufftPlan1d.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                ctypes.c_int, ctypes.c_int]
    lib.cufftDestroy.restype = ctypes.c_int
    lib.cufftDestroy.argtypes = [ctypes.c_int]
    lib.cufftSetStream.restype = ctypes.c_int
    lib.cufftSetStream.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.cufftExecC2C.restype = ctypes.c_int
    lib.cufftExecC2C.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_int]
    lib.cufftExecZ2Z.restype = ctypes.c_int
    lib.cufftExecZ2Z.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_int]
    lib.cufftEstimate1d.restype = ctypes.c_int
    lib.cufftEstimate1d.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_size_t)]
    _lib = lib
    return _lib


def _fft_type(in_dtype, out_dtype):
    cin, cout = np.dtype(in_dtype), np.dtype(out_dtype)
    if cin == np.complex64 and cout == np.complex64:
        return CUFFT_C2C
    if cin == np.complex128 and cout == np.complex128:
        return CUFFT_Z2Z
    raise ValueError(
        "cuvarbase._cufft supports only complex64->complex64 (C2C) and "
        "complex128->complex128 (Z2Z); got %s -> %s" % (cin, cout))


def _devptr(x_gpu):
    """Device pointer (as an int) for a pycuda GPUArray or DeviceAllocation."""
    if hasattr(x_gpu, 'gpudata'):
        return int(x_gpu.gpudata)
    if hasattr(x_gpu, 'ptr'):
        return int(x_gpu.ptr)
    return int(x_gpu)


class Plan(object):
    """A batched 1D complex-to-complex cuFFT plan (scikit-cuda-compatible)."""

    def __init__(self, shape, in_dtype, out_dtype, batch=1, stream=None):
        lib = _load()
        if np.isscalar(shape):
            n = int(shape)
        else:
            n = int(np.prod(shape))
        self.n = n
        self.batch = int(batch)
        self.fft_type = _fft_type(in_dtype, out_dtype)
        self._exec = (lib.cufftExecC2C if self.fft_type == CUFFT_C2C
                      else lib.cufftExecZ2Z)

        self.handle = ctypes.c_int()
        _check(lib.cufftPlan1d(ctypes.byref(self.handle), n,
                               self.fft_type, self.batch))

        if stream is not None:
            handle = getattr(stream, 'handle', stream)
            _check(lib.cufftSetStream(self.handle, ctypes.c_void_p(int(handle))))

    def __del__(self):
        # Never raise from __del__. Skip the destroy during interpreter
        # shutdown: the CUDA context may already be gone, and calling into
        # libcufft then can fault below the Python level (the OS reclaims
        # the plan at exit regardless).
        try:
            if (not _shutting_down
                    and getattr(self, 'handle', None) is not None
                    and _lib is not None):
                _lib.cufftDestroy(self.handle)
                self.handle = None
        except Exception:
            pass


def _exec(plan, x_gpu, y_gpu, direction):
    _check(plan._exec(plan.handle, ctypes.c_void_p(_devptr(x_gpu)),
                      ctypes.c_void_p(_devptr(y_gpu)), direction))


def fft(x_gpu, y_gpu, plan):
    """Forward FFT of ``x_gpu`` into ``y_gpu`` using ``plan`` (in place ok)."""
    _exec(plan, x_gpu, y_gpu, CUFFT_FORWARD)


def ifft(x_gpu, y_gpu, plan):
    """Inverse (unnormalized) FFT of ``x_gpu`` into ``y_gpu`` using ``plan``."""
    _exec(plan, x_gpu, y_gpu, CUFFT_INVERSE)


def cufftEstimate1d(nx, fft_type, batch=1):
    """cuFFT work-area size in bytes for a 1D plan of size ``nx``."""
    lib = _load()
    work = ctypes.c_size_t(0)
    _check(lib.cufftEstimate1d(int(nx), int(fft_type), int(batch),
                               ctypes.byref(work)))
    return work.value


# scikit-cuda exposed the low-level entry points under ``skcuda.fft.cufft``
# (e.g. ``cufft.cufft.cufftEstimate1d``, ``cufft.cufft.CUFFT_C2C``). Mirror
# that nested attribute so call sites importing this module as ``cufft``
# keep working unchanged.
import types as _types  # noqa: E402

cufft = _types.SimpleNamespace(
    cufftEstimate1d=cufftEstimate1d,
    CUFFT_C2C=CUFFT_C2C,
    CUFFT_Z2Z=CUFFT_Z2Z,
    CUFFT_FORWARD=CUFFT_FORWARD,
    CUFFT_INVERSE=CUFFT_INVERSE,
)
