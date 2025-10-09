"""
Base classes and abstractions for cuvarbase.

This module contains the core abstractions used across different
periodogram implementations.
"""
from __future__ import absolute_import

from .async_process import GPUAsyncProcess

__all__ = ['GPUAsyncProcess']
