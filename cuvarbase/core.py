"""
Deprecated alias of :mod:`cuvarbase.base`.

``cuvarbase.core`` shipped in 0.2.5 and is kept for the 1.x series so
old imports keep working; importing it emits a ``DeprecationWarning``.
It will be removed in 2.0. Import ``GPUAsyncProcess`` and
``ensure_context`` from :mod:`cuvarbase.base` instead.
"""
import warnings

from .base import GPUAsyncProcess, ensure_context

warnings.warn("cuvarbase.core is deprecated; import from cuvarbase.base. "
              "It will be removed in 2.0", DeprecationWarning,
              stacklevel=2)

__all__ = ['GPUAsyncProcess', 'ensure_context']
