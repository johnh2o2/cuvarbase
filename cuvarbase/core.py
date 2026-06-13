"""
Core classes for cuvarbase.

This module maintains backward compatibility by importing from the new
base module. New code should import from cuvarbase.base instead.
"""

# Import from new location for backward compatibility
from .base import GPUAsyncProcess, ensure_context

__all__ = ['GPUAsyncProcess', 'ensure_context']
