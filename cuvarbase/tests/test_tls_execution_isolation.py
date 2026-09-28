"""Host tests for release routing, immutable defaults and owned engine state.

Fake CUDA objects exercise real module loading/cache orchestration here; device
arithmetic and actual CUDA cache lifetimes require the separate GPU suite.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

import cuvarbase
from cuvarbase import tls_reference_math as baseline_math
from cuvarbase import tls_reference_experimental_math as experimental_math

PACKAGE = Path(cuvarbase.__file__).resolve().parent
BASELINE_PINS = {
    'tls_reference.py': 'a38d5a9b83b02825374ba6603d3c7b4106a10c939f98c9fb8004c09ce41100e2',
    'tls_reference_math.py': '27e5a575e6279e64ce02fd17c40d64e9885f9c285d3145e13e2bee5ac94afe13',
    'kernels/tls_reference.cu': 'b060dbd2063b03d3295f8469ecb4293a1118131d41a1c55fa0399d0a24fc7d97',
}


@pytest.mark.parametrize('relative,pin', BASELINE_PINS.items())
def test_default_execution_retains_6ced75d_source_bytes(relative, pin):
    # A deliberate default change requires its own review and evidence; the
    # historical optimized study did not qualify these changes for default use.
    assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == pin


def test_experimental_kernel_retains_measured_precursor_source():
    pin = '0725f64bb424334abef574c7d482e946377738488e7a84cf320b731a019bc7a0'
    assert hashlib.sha256((PACKAGE / 'kernels/tls_reference_experimental.cu').read_bytes()).hexdigest() == pin


def test_shared_math_uses_explicit_unchanged_dependencies_without_rebinding():
    for name in ('augment_duration_grid', 'build_cache', 'harmonic_candidate_indices',
                 'native_spectra', 'preprocess_inputs'):
        assert getattr(experimental_math, name) is getattr(baseline_math, name)
    for name in ('chunk_width_masks', 'refinement_candidate_indices'):
        assert getattr(experimental_math, name) is not getattr(baseline_math, name)
        assert getattr(baseline_math, name).__module__ == 'cuvarbase.tls_reference_math'


@pytest.fixture
def isolated_engines(monkeypatch):
    # Import the complete sources under private test names, with only their
    # CUDA dependency replaced. No import or state reset of real GPU engines.
    fake_cupy = types.ModuleType('cupy')
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    loaded = []
    for suffix in ('', '_experimental'):
        path = PACKAGE / ('tls_reference' + suffix + '.py')
        name = 'cuvarbase._isolated_execution' + suffix
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        # Prefix classes have no eager CUDA calls. Remove newly imported
        # dependency modules afterward so this fake cupy cannot leak to tests.
        for dependency in ('tls_reference_prefix', 'tls_reference_short_prefix'):
            key = 'cuvarbase.' + dependency
            if key not in sys.modules:
                dependency_spec = importlib.util.spec_from_file_location(key, PACKAGE / (dependency + '.py'))
                dependency_module = importlib.util.module_from_spec(dependency_spec)
                monkeypatch.setitem(sys.modules, key, dependency_module)
                dependency_spec.loader.exec_module(dependency_module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded, fake_cupy


def test_module_compilation_and_lookup_are_owned_by_each_backend(isolated_engines):
    (baseline, experimental), cp = isolated_engines
    compiled = []

    class Module:
        def __init__(self, code):
            self.code = code
            self.compiles = 0

        def compile(self):
            self.compiles += 1
            compiled.append(self)

    cp.RawModule = Module
    default_modules = baseline.modules()
    assert len(compiled) == 2
    assert experimental._MODULES is None
    optimized_modules = experimental.modules()
    assert len(compiled) == 4
    assert baseline.modules() is default_modules
    assert experimental.modules() is optimized_modules
    assert all(a is not b for a, b in zip(default_modules, optimized_modules))
    assert default_modules[1].code == (PACKAGE / 'kernels/tls_reference.cu').read_text()
    assert optimized_modules[1].code == (PACKAGE / 'kernels/tls_reference_experimental.cu').read_text()
    assert all(module.compiles == 1 for module in compiled)


def test_prefix_state_and_short_dispatch_are_isolated_across_modes_and_threads(isolated_engines, monkeypatch):
    (baseline, experimental), _ = isolated_engines
    assert baseline._PREFIX_PLANS is not experimental._PREFIX_PLANS
    short_calls = []

    class Graph:
        def prefix(self, value):
            return (self, value)

    class Short:
        def prefix(self, value):
            short_calls.append(value)
            return None

    for engine in (baseline, experimental):
        monkeypatch.setattr(engine, '_PrefixPlanCache', Graph)
    monkeypatch.setattr(experimental, 'NativeShortPrefixCache', Short)
    marker = object()
    default_cache, _ = baseline._native_flux_prefix(marker)
    assert not short_calls
    assert not hasattr(baseline._PREFIX_PLANS, 'short')
    experimental_cache, _ = experimental._native_flux_prefix(marker)
    assert short_calls == [marker]
    assert default_cache is not experimental_cache
    assert baseline._native_flux_prefix(marker)[0] is default_cache
    assert short_calls == [marker]
    assert experimental._native_flux_prefix(marker)[0] is experimental_cache

    def worker():
        return [engine._native_flux_prefix(marker)[0] for engine in (baseline, experimental)]

    with ThreadPoolExecutor(max_workers=1) as pool:
        child = pool.submit(worker).result()
    assert child[0] is not default_cache
    assert child[1] is not experimental_cache
    assert child[0] is not child[1]
    assert baseline._PREFIX_PLANS.cache is default_cache
    assert experimental._PREFIX_PLANS.cache is experimental_cache
