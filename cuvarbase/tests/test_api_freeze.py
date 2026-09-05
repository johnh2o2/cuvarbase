"""The 1.0 API freeze (Sep 2026): the frozen top-level namespace, the
NUFFT-LRT quarantine, the keyword-only markers on the 1.0-new
signatures and the per-module ``__all__`` lists. Everything here runs
on CPU (under the pycuda stub of ``conftest.py`` when no GPU is
present)."""
import importlib
import os
import subprocess
import sys
import warnings

import pytest

import cuvarbase


# ---------------------------------------------------------------------
# Top-level namespace (blocker 13)
# ---------------------------------------------------------------------

def test_all_equals_lazy_attrs():
    assert set(cuvarbase.__all__) == set(cuvarbase._LAZY_ATTRS)
    assert len(cuvarbase.__all__) == len(set(cuvarbase.__all__))


@pytest.mark.parametrize('name', sorted(cuvarbase._LAZY_ATTRS))
def test_public_name_resolves(name):
    obj = getattr(cuvarbase, name)
    module = importlib.import_module(cuvarbase._LAZY_ATTRS[name],
                                     'cuvarbase')
    assert obj is getattr(module, name)
    assert name in dir(cuvarbase)


def test_no_accidental_bls_names():
    # the unpublished v1.0 branch resolved any public name of
    # cuvarbase.bls (np, cuda, compile_bls, ...) as cuvarbase.<name>
    assert not hasattr(cuvarbase, 'np')
    assert not hasattr(cuvarbase, 'cuda')
    with pytest.raises(AttributeError):
        cuvarbase.eebls_gpu
    with pytest.raises(AttributeError):
        cuvarbase.compile_bls
    assert 'np' not in dir(cuvarbase)


def test_submodules_reachable_as_attributes():
    for name in cuvarbase._SUBMODULES:
        mod = getattr(cuvarbase, name)
        assert mod.__name__ == 'cuvarbase.' + name
        assert name in dir(cuvarbase)


# ---------------------------------------------------------------------
# NUFFT-LRT quarantine (decision D1)
# ---------------------------------------------------------------------

def test_nufft_lrt_not_top_level():
    assert 'NUFFTLRTAsyncProcess' not in cuvarbase.__all__
    assert 'NUFFTLRTMemory' not in cuvarbase.__all__
    assert 'nufft_lrt' in cuvarbase._SUBMODULES
    import cuvarbase.nufft_lrt as nufft_lrt
    assert callable(nufft_lrt.NUFFTLRTAsyncProcess)
    assert callable(nufft_lrt.NUFFTLRTMemory)


_STAR_IMPORT_SCRIPT = r"""
import sys, types
# Harmless pycuda stubs so the GPU modules import without a real GPU
# (the star-import resolves every lazy name, which imports every
# method module).
for name in ['pycuda', 'pycuda.driver', 'pycuda.gpuarray',
             'pycuda.compiler', 'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object
_autoctx = types.ModuleType('pycuda.autoprimaryctx')
_autoctx.device = object()
_autoctx.context = object()
sys.modules['pycuda.autoprimaryctx'] = _autoctx

import warnings
warnings.simplefilter('always')
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    from cuvarbase import *
    import cuvarbase.nufft_lrt
exp = [w for w in rec if 'EXPERIMENTAL' in str(w.message)]
assert not exp, [str(w.message) for w in exp]
names = sorted(n for n in dir() if not n.startswith('_')
               and n not in ('warnings', 'rec', 'exp', 'cuvarbase',
                             'sys', 'types', 'name'))
import cuvarbase
assert names == sorted(cuvarbase.__all__), (names, cuvarbase.__all__)
print('OK')
"""


def test_star_import_emits_no_experimental_warning():
    # star-import must not import nufft_lrt, and importing nufft_lrt
    # must not warn either: the warning is emitted at construction.
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(
        [sys.executable, '-c', _STAR_IMPORT_SCRIPT],
        cwd=repo_root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_nufft_lrt_warns_at_construction():
    from cuvarbase import nufft_lrt

    class _Proc(nufft_lrt.NUFFTLRTAsyncProcess):
        # GPUAsyncProcess.__init__ retains the CUDA context; skip it
        # (and the NFFT process) so the warning is testable on CPU.
        def __init__(self):
            warnings.warn(nufft_lrt._EXPERIMENTAL_MSG, UserWarning,
                          stacklevel=2)

    with pytest.warns(UserWarning,
                      match='cuvarbase.nufft_lrt is EXPERIMENTAL'):
        _Proc()
    # the real constructor's first statement is the same warning
    import inspect
    src = inspect.getsource(nufft_lrt.NUFFTLRTAsyncProcess.__init__)
    body = src.split('):', 1)[1].lstrip()
    assert body.startswith('warnings.warn(_EXPERIMENTAL_MSG')
