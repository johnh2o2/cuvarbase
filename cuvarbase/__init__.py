# The CUDA primary context is created lazily on first GPU use (see
# cuvarbase.base.ensure_context), NOT at import. `import cuvarbase` and
# the CPU-only helpers therefore require neither a GPU nor a CUDA
# context. The GPU modules still import pycuda.driver at module top, so
# the pycuda package remains a dependency for them -- but importing them
# no longer allocates a context.

# Version
__version__ = "1.0.0"

# The public top-level names are resolved lazily (PEP 562): importing
# the package imports none of the method modules, so `import cuvarbase`
# costs nothing and never fails because one backend (libcufft for the
# NFFT-based methods, batman for TLS templates, cufinufft) is missing.
# Each name below is fetched from its module on first access and is the
# same object as the module attribute (``cuvarbase.BLSMemory is
# cuvarbase.bls.BLSMemory``).
#
# This mapping IS the frozen 1.x top-level API: ``__all__`` is exactly
# its keys (``cuvarbase/tests/test_api_freeze.py`` asserts that) and
# nothing else resolves as ``cuvarbase.<name>`` except the submodules in
# ``_SUBMODULES``. Everything else lives in its module
# (``cuvarbase.bls.eebls_gpu``, ``cuvarbase.tls.tls_search_gpu``, ...).
_LAZY_ATTRS = {
    'GPUAsyncProcess': '.base',
    'NFFTMemory': '.memory',
    'ConditionalEntropyMemory': '.memory',
    'LombScargleMemory': '.memory',
    'BLSMemory': '.bls',
    'BLSBatchMemory': '.memory',
    'NFFTAsyncProcess': '.cunfft',
    'nfft_adjoint_async': '.cunfft',
    'ConditionalEntropyAsyncProcess': '.ce',
    'conditional_entropy': '.ce',
    'conditional_entropy_fast': '.ce',
    'LombScargleAsyncProcess': '.lombscargle',
    'lomb_scargle_async': '.lombscargle',
    'PDMAsyncProcess': '.pdm',
}

# Submodules reachable as attributes (``cuvarbase.bls``) without an
# explicit ``import cuvarbase.bls``. ``nufft_lrt`` is deliberately here
# and NOT in ``_LAZY_ATTRS``: the NUFFT likelihood-ratio test is
# quarantined as EXPERIMENTAL for 1.0 (importable as
# ``cuvarbase.nufft_lrt``, outside the 1.x API-stability promise, warns
# at construction): its Sep-2026 re-validation passed the correctness
# gate but showed that its defaults and return conventions should
# still change before the API is frozen (decision D1, 2026-09-06).
_SUBMODULES = {
    'base', 'memory', 'core', 'utils',
    'bls', 'bls_frequencies', 'ce', 'cunfft', 'lombscargle', 'pdm',
    'cufinufft_backend', 'nufft_lrt',
    'tls', 'tls_grids', 'tls_models', 'tls_stats',
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    import importlib

    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        return getattr(module, name)

    if name in _SUBMODULES:
        return importlib.import_module('.' + name, __name__)

    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY_ATTRS)
                      + list(_SUBMODULES)))
