"""Root conftest: stub GPU dependencies for CPU-only test runs.

cuvarbase/__init__.py imports ``pycuda.autoprimaryctx`` at the top level, so on a
machine without CUDA the test suite cannot even be collected.  When pycuda is
genuinely unavailable, this conftest installs minimal stub modules so that:

* the full suite collects,
* pure-CPU tests (sparse BLS ground truth, TLS grids/models/stats, frequency
  grids, ...) run normally, and
* any test that actually touches the GPU raises :class:`GPUStubError`, which the
  hook below converts into a pytest *skip* rather than a failure.

On machines with a working pycuda installation this file does nothing.
"""
import sys
import types

import pytest

try:
    import pycuda.driver  # noqa: F401
    _HAS_PYCUDA = True
except Exception:
    _HAS_PYCUDA = False


class GPUStubError(RuntimeError):
    """Raised when stubbed GPU functionality is exercised without a GPU."""


if not _HAS_PYCUDA:

    class _GPUStub:
        """Attribute sink that raises GPUStubError when called."""

        def __init__(self, name):
            self._name = name

        def __getattr__(self, attr):
            if attr.startswith('__') and attr.endswith('__'):
                raise AttributeError(attr)
            return _GPUStub('%s.%s' % (self._name, attr))

        def __call__(self, *args, **kwargs):
            raise GPUStubError(
                '%s requires a GPU (pycuda is stubbed by conftest.py)'
                % self._name)

    def _make_module(name, **attrs):
        mod = types.ModuleType(name)
        for key, val in attrs.items():
            setattr(mod, key, val)
        sys.modules[name] = mod
        return mod

    pycuda_mod = _make_module('pycuda')
    _make_module('pycuda.autoprimaryctx')
    _make_module('pycuda.autoinit')

    def _module_getattr(modname):
        def _getattr(attr):
            # Dunders (__file__, __path__, ...) must follow normal module
            # semantics or inspect/import machinery breaks during collection.
            if attr.startswith('__') and attr.endswith('__'):
                raise AttributeError(attr)
            return _GPUStub('%s.%s' % (modname, attr))
        return _getattr

    driver = _make_module('pycuda.driver')
    driver.__getattr__ = _module_getattr('pycuda.driver')

    gpuarray = _make_module('pycuda.gpuarray')
    gpuarray.__getattr__ = _module_getattr('pycuda.gpuarray')

    _make_module('pycuda.compiler',
                 SourceModule=_GPUStub('pycuda.compiler.SourceModule'))

    # mark_cuda_test must be a passthrough decorator: it is applied at import
    # time, and the decorated tests then skip via GPUStubError when they run.
    _make_module('pycuda.tools',
                 mark_cuda_test=lambda f: f,
                 context_dependent_memoize=lambda f: f)

    pycuda_mod.driver = driver
    pycuda_mod.gpuarray = gpuarray

    skcuda_mod = _make_module('skcuda')
    fft = _make_module('skcuda.fft')
    fft.__getattr__ = _module_getattr('skcuda.fft')
    skcuda_mod.fft = fft


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert GPUStubError failures into skips on GPU-less machines."""
    outcome = yield
    rep = outcome.get_result()
    if rep.outcome == 'failed' and call.excinfo is not None:
        if call.excinfo.errisinstance(GPUStubError):
            rep.outcome = 'skipped'
            rep.longrepr = (str(item.fspath), item.location[1],
                            'requires GPU (pycuda stubbed by conftest.py)')
