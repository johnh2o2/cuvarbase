# import pycuda.autoinit causes problems when running e.g. FFT
import pycuda.autoprimaryctx

# Version
__version__ = "1.0.0"

# Public attributes are resolved lazily (PEP 562) so that importing the
# package does not drag in every backend. In particular, `import cuvarbase`
# must not require scikit-cuda (only the NFFT/Lomb-Scargle modules need
# cufft) — BLS/CE/PDM users can run on environments where scikit-cuda is
# broken (e.g. numpy >= 1.24 without the compat shim).
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

# NUFFT-LRT was cut from the v1.0 wheel (CPU-bound implementation with
# a uniform-grid-span limitation); the source lives on the
# feature/nufft-lrt-experimental branch pending a GPU rewire.
_SUBMODULES = {
    'base', 'memory', 'core', 'utils',
    'bls', 'bls_frequencies', 'ce', 'cunfft', 'lombscargle', 'pdm',
    'cufinufft_backend',
    'tls', 'tls_grids', 'tls_models', 'tls_stats',
}

__all__ = [
    'GPUAsyncProcess',
    'NFFTMemory',
    'ConditionalEntropyMemory',
    'LombScargleMemory',
    'NFFTAsyncProcess',
    'ConditionalEntropyAsyncProcess',
    'LombScargleAsyncProcess',
    'PDMAsyncProcess',
]


def __getattr__(name):
    import importlib

    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        return getattr(module, name)

    if name in _SUBMODULES:
        return importlib.import_module('.' + name, __name__)

    # Backward compatibility with the old eager `from .bls import *`:
    # any public name bls exposes is reachable as cuvarbase.<name>.
    if not name.startswith('_'):
        try:
            bls = importlib.import_module('.bls', __name__)
        except ImportError:
            raise AttributeError(
                "module %r has no attribute %r" % (__name__, name))
        if hasattr(bls, name):
            return getattr(bls, name)

    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(list(globals()) + __all__ + list(_SUBMODULES)))
