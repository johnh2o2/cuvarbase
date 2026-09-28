"""Preserved binned/legacy TLS t0-fidelity parameter plumbing (D1).

``T0_OVERSAMPLE`` (transit-epoch trial positions per duration) is now a
Python-level parameter (``t0_oversample``) plumbed into the kernel's
``#define`` via cpp_defs, the kernel cache key, and the older engine.
These checks run on CPU (no kernel compilation needed).
"""
import inspect

import cuvarbase.tls as tls_mod
from cuvarbase.tls import compile_tls, _get_cached_kernels, _tls_search_gpu_binned
from cuvarbase.tls_grids import t0_grid_size
from cuvarbase.utils import _module_reader, find_kernel


def test_t0_oversample_overrides_kernel_define():
    # The kernel guards its default with `#ifndef T0_OVERSAMPLE`, so the
    # cpp_defs `#define` must appear *before* that guard to take effect.
    txt = _module_reader(find_kernel('tls'),
                         cpp_defs={'BLOCK_SIZE': 128, 'T0_OVERSAMPLE': 33.0})
    assert '#define T0_OVERSAMPLE 33.0' in txt
    assert (txt.index('#define T0_OVERSAMPLE 33.0')
            < txt.index('#ifndef T0_OVERSAMPLE'))


def test_t0_oversample_in_binned_signatures():
    for fn in (compile_tls, _get_cached_kernels, _tls_search_gpu_binned):
        params = inspect.signature(fn).parameters
        assert 't0_oversample' in params, fn.__name__
    # default matches the kernel/grid default
    assert inspect.signature(compile_tls).parameters[
        't0_oversample'].default == 3.0


def test_t0_oversample_is_part_of_cache_key(monkeypatch):
    calls = []

    def fake_compile(block_size, t0_oversample=3.0):
        calls.append((block_size, t0_oversample))
        return {'standard': object(), 'keplerian': object()}

    monkeypatch.setattr(tls_mod, 'compile_tls', fake_compile)
    # Snapshot and restore the module-level cache: monkeypatch undoes
    # compile_tls but NOT cache contents — leaking the fake kernel
    # objects under keys like (128, 3.0) crashes any later test in the
    # same process that hits _get_cached_kernels with default settings.
    saved = dict(tls_mod._kernel_cache)
    tls_mod._kernel_cache.clear()
    try:
        _get_cached_kernels(128, t0_oversample=3.0)
        _get_cached_kernels(128, t0_oversample=3.0)   # cache hit
        _get_cached_kernels(128, t0_oversample=33.0)  # distinct key
        assert calls == [(128, 3.0), (128, 33.0)]
    finally:
        tls_mod._kernel_cache.clear()
        tls_mod._kernel_cache.update(saved)


def test_t0_grid_size_mirrors_oversample():
    # n_t0 = ceil(oversample / duration_phase), clamped to [30, 20000];
    # this is the Python mirror of the device t0_grid_size.
    assert t0_grid_size(0.01, oversample=3.0) == 300
    assert t0_grid_size(0.01, oversample=33.0) == 3300   # ~11x finer
    assert t0_grid_size(0.5, oversample=3.0) == 30       # floored at MIN_N_T0
