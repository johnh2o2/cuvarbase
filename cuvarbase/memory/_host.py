"""Host-array allocation for host<->device transfers.

cuvarbase's ``*Memory`` classes stage data in host arrays that are copied
to/from the GPU. For ``memcpy_*_async`` to actually run asynchronously
(overlapping transfers with compute), the host buffer must be *page-locked*
(pinned). Plain ``cuda.aligned_zeros`` is only page-aligned, so the driver
silently stages async copies through a synchronous bounce buffer.

:func:`host_array` allocates a pinned (``cuda.pagelocked_zeros``) buffer by
default and falls back to page-aligned memory if pinning fails -- e.g. when
the OS/driver page-locked-memory limit is exhausted -- so a pinning failure
degrades performance rather than crashing.
"""
import warnings

import pycuda.driver as cuda

_warned_fallback = False


def _warn_fallback(exc):
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        warnings.warn(
            "could not allocate page-locked (pinned) host memory (%s: %s); "
            "falling back to page-aligned host arrays. Async host<->device "
            "transfers will stage synchronously, reducing overlap. Pass "
            "pinned=False to silence this, or raise the system's locked-"
            "memory limit." % (type(exc).__name__, exc),
            RuntimeWarning)


def host_array(shape, dtype, pinned=True):
    """Allocate a zeroed host array for GPU transfers.

    Parameters
    ----------
    shape : int or tuple
        Array shape.
    dtype : numpy dtype
        Array dtype.
    pinned : bool, optional (default: True)
        If True, allocate page-locked (pinned) memory for true async
        transfer overlap, falling back to page-aligned memory if pinning
        fails. If False, allocate page-aligned memory directly.

    Returns
    -------
    numpy.ndarray
        A zeroed host array (pinned when possible).
    """
    if pinned:
        try:
            return cuda.pagelocked_zeros(shape, dtype=dtype)
        except Exception as exc:
            # Pinning failed. Try the page-aligned fallback; if THAT also
            # raises (e.g. the GPU is stubbed out in a CPU-only test run),
            # let it propagate rather than warn about a fallback that did
            # not actually happen.
            arr = cuda.aligned_zeros(shape, dtype=dtype)
            _warn_fallback(exc)
            return arr
    return cuda.aligned_zeros(shape, dtype=dtype)
