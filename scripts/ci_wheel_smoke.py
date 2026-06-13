"""CI packaging smoke test for the *installed* wheel.

Run from a clean environment where cuvarbase was installed from the built
wheel (pip install --no-deps dist/*.whl), so pycuda is genuinely absent.
Two things are checked:

1. ``import cuvarbase`` requires neither pycuda nor a CUDA context (the
   primary context is created lazily on first GPU use, not at import).
2. With pycuda stubbed, the GPU module surface and the packaged kernel
   files import/resolve -- catching missing-subpackage and
   missing-package-data bugs that source-tree testing hides (e.g. the
   v1.0 wheel that omitted cuvarbase.base/cuvarbase.memory entirely, or a
   shared .cuh left out of package-data).
"""
import os
import sys
import types

# Make sure we import the installed package, not the source tree.
sys.path = [p for p in sys.path if os.path.abspath(p) != os.getcwd()]

# --- Part 1: GPU-less, pycuda-less import ---------------------------------
# pycuda is not installed in this venv, so a successful import proves the
# package top-level does not import it (no eager CUDA context).
import cuvarbase  # noqa: E402
assert 'pycuda' not in sys.modules, \
    "import cuvarbase pulled in pycuda -- the CUDA context is no longer " \
    "supposed to be created at import time"
print('GPU-less import OK:', cuvarbase.__version__)

# --- Part 2: stubbed-pycuda deep import + packaged data -------------------
for name in ['pycuda', 'pycuda.autoprimaryctx', 'pycuda.autoinit',
             'pycuda.driver', 'pycuda.gpuarray', 'pycuda.compiler',
             'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object

from cuvarbase import bls  # noqa: E402, F401
from cuvarbase.base import GPUAsyncProcess, ensure_context  # noqa: E402, F401
from cuvarbase.memory import BLSBatchMemory  # noqa: E402, F401
import cuvarbase.utils  # noqa: E402

kernel_path = cuvarbase.utils.find_kernel('bls')
assert os.path.exists(kernel_path), \
    "kernel file missing from wheel: %s" % kernel_path
# The shared BLS device functions live in a .cuh inlined at load time;
# it must ship in the wheel or kernel compilation breaks at runtime.
common = os.path.join(os.path.dirname(kernel_path), 'bls_common.cuh')
assert os.path.exists(common), "bls_common.cuh missing from wheel"

print('wheel import OK:', cuvarbase.__version__)
