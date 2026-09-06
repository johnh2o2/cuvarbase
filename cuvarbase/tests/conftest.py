"""Test-package conftest: stub GPU dependencies for CPU-only test runs.

The CUDA primary context is created lazily on first GPU use (see
``cuvarbase.base.ensure_context``), so ``import cuvarbase`` needs no
device -- but the GPU modules (``bls``, ``ce``, ``cunfft``,
``lombscargle``, ``pdm``, ``tls``, ``nufft_lrt``) still ``import
pycuda.driver`` at module top, and most test modules import them.  On a
machine without pycuda the suite could therefore not even be collected.
When pycuda is genuinely unavailable this conftest installs minimal stub
modules so that:

* the full suite collects,
* pure-CPU tests (sparse BLS ground truth, TLS grids/models/stats,
  frequency grids, input validation, ...) run normally, and
* any test that actually touches the GPU raises :class:`GPUStubError`,
  which the hook below converts into a pytest *skip* rather than a
  failure.

On machines with a working pycuda installation the stubs are not
installed and every test runs for real.

This file lives inside the package (``cuvarbase/tests/conftest.py``, not
the repository root) so that it ships in the wheel and is loaded by
``pytest --pyargs cuvarbase`` from an installed copy.

The ``gpu`` marker
------------------
``pytest_configure`` registers a ``gpu`` marker (so ``--strict-markers``
works with or without the ``[tool.pytest.ini_options]`` table) and
``pytest_collection_modifyitems`` applies it automatically.  The
heuristic is deliberately simple and module-grained: an item is marked
``gpu`` when its module either

* exposes a ``mark_cuda_test`` attribute (it imported
  ``pycuda.tools.mark_cuda_test`` to decorate device tests), or
* is listed in :data:`GPU_TEST_MODULES` below -- the modules whose tests
  are predominantly on-device (measured on a CPU-only host: >= 50 % of
  their items skip through the GPUStubError hook).

``-m "not gpu"`` therefore runs the CPU subset quickly.  It is a
*heuristic*: a handful of CPU-only tests that live in GPU-dominant
modules are excluded by it, and the few device tests that live in
CPU-dominant modules still run (and skip via the hook on a GPU-less
host, or pass on a device).  The hook, not the marker, is what keeps the
CPU run green; the marker is a selection convenience.
"""
import sys
import types

import pytest

try:
    import pycuda.driver  # noqa: F401
    _HAS_PYCUDA = True
except Exception:
    _HAS_PYCUDA = False


# Test modules (basename without .py) whose items are predominantly
# on-device.  Keep in sync with the heuristic documented above; a module
# that also imports ``mark_cuda_test`` is marked whether or not it is
# listed here.
GPU_TEST_MODULES = frozenset({
    'test_bls',
    'test_ce',
    'test_lombscargle',
    'test_nfft',
    'test_nufft_lrt',
    'test_pdm',
    'test_readme_examples',
    'test_tls_fast',
    'test_tls_golden',
})


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
                '%s requires a GPU (pycuda is stubbed by '
                'cuvarbase/tests/conftest.py)' % self._name)

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


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: needs a CUDA device')


def pytest_collection_modifyitems(config, items):
    """Apply ``pytest.mark.gpu`` per the module-grained heuristic
    documented in the module docstring."""
    for item in items:
        module = getattr(item, 'module', None)
        if module is None:
            continue
        name = module.__name__.rsplit('.', 1)[-1]
        if name in GPU_TEST_MODULES or hasattr(module, 'mark_cuda_test'):
            item.add_marker(pytest.mark.gpu)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert GPUStubError failures into skips on GPU-less machines."""
    outcome = yield
    rep = outcome.get_result()
    if rep.outcome == 'failed' and call.excinfo is not None:
        if call.excinfo.errisinstance(GPUStubError):
            rep.outcome = 'skipped'
            rep.longrepr = (str(item.fspath), item.location[1],
                            'requires GPU (pycuda stubbed by '
                            'cuvarbase/tests/conftest.py)')
