"""Lazy-import contract: `import cuvarbase` and the BLS/CE surface must
work even when scikit-cuda is broken (e.g. scikit-cuda 0.5.3 on
numpy >= 1.24). Only the NFFT/Lomb-Scargle modules may require skcuda,
and only at attribute-access time."""
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
try:
    cuvarbase.LombScargleAsyncProcess
except ImportError:
    pass  # expected: LS genuinely needs skcuda's cufft
else:
    raise SystemExit('LombScargle access should raise ImportError '
                     'when skcuda is broken')
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
