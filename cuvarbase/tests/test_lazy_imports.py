"""Lazy-import contract: `import cuvarbase` and every public process
must import even when scikit-cuda is broken/absent. As of v1.0 the cuFFT
binding is in-house (`cuvarbase._cufft`), so NO cuvarbase module imports
scikit-cuda anymore -- not even Lomb-Scargle/NFFT (they need libcufft
only when a transform actually runs)."""
import os
import subprocess
import sys

import pytest

_SCRIPT = r"""
import sys, types
for name in ['pycuda', 'pycuda.autoprimaryctx', 'pycuda.autoinit',
             'pycuda.driver', 'pycuda.gpuarray', 'pycuda.compiler',
             'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object

class _BrokenSkcudaFinder:
    def find_module(self, fullname, path=None):
        if fullname.startswith('skcuda'):
            return self
    def load_module(self, fullname):
        raise ImportError('simulated scikit-cuda failure')
sys.meta_path.insert(0, _BrokenSkcudaFinder())

import cuvarbase
from cuvarbase import bls
assert callable(cuvarbase.eebls_gpu)
from cuvarbase import ConditionalEntropyAsyncProcess
assert cuvarbase.BLSMemory is bls.BLSMemory
# Since v1.0 the cuFFT binding is in-house, so Lomb-Scargle no longer
# imports scikit-cuda: accessing it must succeed even with skcuda broken.
assert callable(cuvarbase.LombScargleAsyncProcess), \
    'LombScargleAsyncProcess should import without scikit-cuda'
assert callable(cuvarbase.NFFTAsyncProcess), \
    'NFFTAsyncProcess should import without scikit-cuda'
print('OK')
"""


def test_import_survives_broken_skcuda():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT],
        cwd=repo_root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


_IMPORT_WITHOUT_PYCUDA = r"""
# `import cuvarbase` must require neither pycuda nor a CUDA context: the
# primary context is now created lazily on first GPU use, not at import.
import builtins
_orig_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if name == 'pycuda' or name.startswith('pycuda.'):
        raise ImportError('pycuda blocked for this test')
    return _orig_import(name, *args, **kwargs)


builtins.__import__ = _blocked

import cuvarbase
assert cuvarbase.__version__
# a pure-CPU utility must be reachable without pycuda
from cuvarbase.utils import weights  # noqa: F401
print('OK')
"""

_NO_CONTEXT_UNTIL_GPU_USE = r"""
import sys, types
import numpy as np

# Harmless pycuda stubs so the GPU modules import without a real GPU.
for name in ['pycuda', 'pycuda.driver', 'pycuda.gpuarray',
             'pycuda.compiler', 'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object
# A fake retained-context module so ensure_context() succeeds off-GPU.
_autoctx = types.ModuleType('pycuda.autoprimaryctx')
_autoctx.device = object()
_autoctx.context = object()
sys.modules['pycuda.autoprimaryctx'] = _autoctx

import cuvarbase
from cuvarbase import bls
from cuvarbase.base import context as ctxmod, ensure_context

# Importing the package + the BLS module must NOT have created a context.
assert ctxmod._autoctx is None, 'CUDA context created at import time'

# A CPU-only helper must run without creating a context.
t = np.linspace(0, 10, 50)
y = np.sin(2 * np.pi * t)
dy = 0.1 * np.ones_like(t)
bls.single_bls(t, y, dy, 1.0, 0.1, 0.0)
assert ctxmod._autoctx is None, 'CPU helper created a CUDA context'

# First explicit GPU use retains the context (and caches it).
ensure_context()
assert ctxmod._autoctx is _autoctx, 'ensure_context did not retain the context'
print('OK')
"""


def _run_in_subprocess(script):
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    return subprocess.run(
        [sys.executable, '-c', script],
        cwd=repo_root, capture_output=True, text=True, timeout=120)


def test_import_cuvarbase_without_pycuda():
    result = _run_in_subprocess(_IMPORT_WITHOUT_PYCUDA)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_no_cuda_context_until_first_gpu_use():
    result = _run_in_subprocess(_NO_CONTEXT_UNTIL_GPU_USE)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_nufft_lrt_removed_from_package():
    # NUFFT-LRT was cut from the v1.0 wheel (source preserved on the
    # feature/nufft-lrt-experimental branch); the package must not
    # expose it anymore.
    import cuvarbase
    assert 'NUFFTLRTAsyncProcess' not in cuvarbase.__all__
    with pytest.raises(AttributeError):
        cuvarbase.nufft_lrt
    with pytest.raises(ImportError):
        import cuvarbase.nufft_lrt  # noqa: F401
