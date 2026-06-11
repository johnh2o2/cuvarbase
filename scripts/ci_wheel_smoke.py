"""CI packaging smoke test: import the *installed* wheel with pycuda stubbed.

Run from a clean environment where cuvarbase was installed from the built
wheel (pip install --no-deps dist/*.whl). Catches missing-subpackage and
missing-package-data bugs that source-tree testing hides (e.g. the v1.0
wheel that omitted cuvarbase.base/cuvarbase.memory entirely).
"""
import os
import sys
import types

# Stub pycuda so import works without CUDA
for name in ['pycuda', 'pycuda.autoprimaryctx', 'pycuda.autoinit',
             'pycuda.driver', 'pycuda.gpuarray', 'pycuda.compiler',
             'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object

# Make sure we import the installed package, not the source tree
sys.path = [p for p in sys.path if os.path.abspath(p) != os.getcwd()]

import cuvarbase  # noqa: E402
from cuvarbase import bls  # noqa: E402, F401
from cuvarbase.base import GPUAsyncProcess  # noqa: E402, F401
from cuvarbase.memory import BLSBatchMemory  # noqa: E402, F401
import cuvarbase.utils  # noqa: E402

kernel_path = cuvarbase.utils.find_kernel('bls')
assert os.path.exists(kernel_path), \
    "kernel file missing from wheel: %s" % kernel_path

print('wheel import OK:', cuvarbase.__version__)
