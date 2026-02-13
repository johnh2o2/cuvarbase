"""Root conftest to stub GPU dependencies for CPU-only testing."""
import sys
import types

# Only stub if pycuda is genuinely unavailable
try:
    import pycuda.driver
    _has_pycuda = True
except (ImportError, Exception):
    _has_pycuda = False

if not _has_pycuda:
    for name in ['pycuda', 'pycuda.autoprimaryctx', 'pycuda.driver',
                 'pycuda.gpuarray', 'pycuda.compiler',
                 'skcuda', 'skcuda.fft', 'skcuda.misc']:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules['pycuda.compiler'].SourceModule = None
