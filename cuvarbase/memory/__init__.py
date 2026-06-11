"""
Memory management classes for GPU operations.

This module contains classes for managing memory allocation and transfer
between CPU and GPU for various periodogram computations.

Attributes are resolved lazily (PEP 562) so that importing one memory
class does not drag in the others' backends — in particular,
``nfft_memory`` imports ``skcuda.fft``, which BLS/CE users must be able
to avoid.
"""

_LAZY_ATTRS = {
    'NFFTMemory': '.nfft_memory',
    'ConditionalEntropyMemory': '.ce_memory',
    'LombScargleMemory': '.lombscargle_memory',
    'weights': '.lombscargle_memory',
    'BLSBatchMemory': '.bls_memory',
}

__all__ = [
    'NFFTMemory',
    'ConditionalEntropyMemory',
    'LombScargleMemory',
    'weights',
    'BLSBatchMemory',
]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        return getattr(module, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(list(globals()) + __all__))
