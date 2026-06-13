"""Lazy CUDA primary-context management.

Historically cuvarbase created (retained and pushed) the CUDA primary
context eagerly via ``import pycuda.autoprimaryctx`` at package import
time, so merely ``import cuvarbase`` required a working GPU. The context
is now created on first GPU use through :func:`ensure_context`, which
defers to the same ``pycuda.autoprimaryctx`` machinery: device selection
honors the ``CUDA_DEVICE`` environment variable (via pycuda's
``make_default_context``) and an ``atexit`` handler pops the context on
interpreter shutdown.

As a result ``import cuvarbase`` and the CPU-only helpers (e.g.
``sparse_bls_cpu``, ``single_bls``, ``fap_baluev``) no longer touch the
GPU. The ``pycuda`` *package* remains an import dependency of the GPU
modules -- they ``import pycuda.driver`` at module top -- but importing
them no longer allocates a CUDA context; that happens only when a kernel
is compiled or a periodogram process is constructed.
"""

_autoctx = None


def ensure_context():
    """Retain and activate the CUDA primary context, once, on first use.

    Returns the ``pycuda.autoprimaryctx`` module, which exposes the
    active ``context`` and the selected ``device``. The underlying
    context setup (``cuda.init()`` + ``retain_primary_context`` + push +
    ``atexit`` cleanup) runs only on the first call; subsequent calls
    return the cached module, so this is safe to call at the top of every
    GPU entry point.

    Device selection follows the ``CUDA_DEVICE`` environment variable
    (read by pycuda's ``make_default_context`` the first time this runs).
    """
    global _autoctx
    if _autoctx is None:
        import pycuda.autoprimaryctx as autoctx
        _autoctx = autoctx
    return _autoctx
