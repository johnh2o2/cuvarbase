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


# ---------------------------------------------------------------------
# Compatibility shims kept for 1.x (decision D3: shipped in 0.2.5)
# ---------------------------------------------------------------------

def test_core_module_is_deprecated_alias():
    sys.modules.pop('cuvarbase.core', None)
    with pytest.warns(DeprecationWarning, match='removed in 2.0'):
        import cuvarbase.core as core
    from cuvarbase import base
    assert core.GPUAsyncProcess is base.GPUAsyncProcess
    assert core.ensure_context is base.ensure_context


def test_no_internal_import_of_core():
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for dirpath, _, files in os.walk(pkg):
        if os.path.basename(dirpath) == 'tests':
            continue
        for f in files:
            if f.endswith('.py') and f != 'core.py':
                src = open(os.path.join(dirpath, f)).read()
                if 'from .core import' in src or 'cuvarbase.core' in src:
                    offenders.append(f)
    assert offenders == []


def test_bls_allocate_pinned_arrays_warns(monkeypatch):
    from cuvarbase.bls import BLSMemory
    mem = BLSMemory.__new__(BLSMemory)
    calls = []
    monkeypatch.setattr(mem, 'allocate_host_arrays',
                        lambda **kw: calls.append(kw) or 'ok',
                        raising=False)
    with pytest.warns(DeprecationWarning, match='removed in 2.0'):
        assert mem.allocate_pinned_arrays(nfreqs=3, ndata=4) == 'ok'
    assert calls == [{'nfreqs': 3, 'ndata': 4}]


def test_pdm_four_tuple_warning_wording():
    import inspect
    from cuvarbase import pdm
    src = inspect.getsource(pdm.PDMAsyncProcess.run)
    assert 'removed in 2.0' in src
    assert 'NORMALIZED WEIGHTS' in src


def test_gpu_async_process_device_keyword():
    from cuvarbase.base import GPUAsyncProcess
    # device=0 (the default) and the other legacy keywords are silent
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        proc = GPUAsyncProcess(reader=None, function_kwargs={}, device=0)
    assert proc.device == 0
    with pytest.warns(UserWarning, match='CUDA_DEVICE'):
        proc = GPUAsyncProcess(device=1)
    assert proc.device == 1


def test_utils_weights_is_canonical():
    import numpy as np
    from cuvarbase import utils
    from cuvarbase.memory import lombscargle_memory
    import cuvarbase.memory as memory
    assert lombscargle_memory.weights is utils.weights
    assert memory.weights is utils.weights
    err = np.array([0.1, 0.2, 0.4])
    w = utils.weights(err)
    assert w.dtype == np.float64
    assert w.sum() == pytest.approx(1.0)
