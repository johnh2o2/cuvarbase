"""Exact short CUB scan and protective native-graph dispatch regressions."""
import json

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from cuvarbase import tls_reference_experimental as engine
from cuvarbase.tls_reference_short_prefix import NativeShortPrefixCache

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def cuda_device():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device unavailable')


@pytest.fixture(scope='module')
def supported_cache():
    cache = NativeShortPrefixCache()
    array = cp.ones((2, 31), cp.float32)
    result = cache.prefix(array)
    if result is None:
        pytest.skip(cache.status['fallback_reason'])
    assert cache.status['active']
    assert cache.status['cached_module_count'] == 1
    return cache


def equal_bits(actual, expected):
    np.testing.assert_array_equal(actual.get().view(np.uint32), expected.get().view(np.uint32))


@pytest.mark.parametrize('columns', [1, 15, 16, 31, 32, 127, 128, 129,
                                    255, 256, 511, 512, 513, 1212, 1317,
                                    1475, 1477, 1919, 1920])
@pytest.mark.parametrize('offset', [0, 1])
def test_short_scan_preserves_native_bits_at_boundaries(supported_cache, columns, offset):
    rng = np.random.default_rng(columns)
    values = rng.normal(1., .005, (3, columns)).astype(np.float32)
    values[0, :max(1, columns // 50)] -= np.float32(.03)
    values[1] *= np.resize(np.array([-1., 2**18, 1., -2**18], np.float32), columns)
    # Include signed zeros and subnormals, which fast-math compilation could
    # flush or normalize even if ordinary normalized flux happened to agree.
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff], np.uint32)
    values[2] = np.resize(words.view(np.float32), columns)
    storage = cp.empty(values.size + offset, dtype=cp.float32)
    array = storage[offset:].reshape(values.shape)
    array.set(values)
    observed = supported_cache.prefix(array)
    assert observed is not None
    assert observed.data.ptr != array.data.ptr
    equal_bits(observed, engine._row_flux_prefix(array))


def test_short_scan_outputs_survive_later_calls_and_stream_changes(supported_cache):
    first = cp.asarray(np.random.default_rng(81).normal(1., .01, (7, 1475)), cp.float32)
    output = supported_cache.prefix(first)
    expected = engine._row_flux_prefix(first)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        second = cp.asarray(np.random.default_rng(82).normal(1., .01, (7, 1475)), cp.float32)
        other = supported_cache.prefix(second)
        equal_bits(other, engine._row_flux_prefix(second))
    equal_bits(output, expected)
    assert output.data.ptr != other.data.ptr


def test_single_element_rows_preserve_special_float_bits(supported_cache):
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                      0xff800000, 0x7fc00123, 0xffc00123, 0x3f800000], np.uint32)
    array = cp.asarray(words.view(np.float32).reshape(-1, 1))
    equal_bits(supported_cache.prefix(array), engine._row_flux_prefix(array))


@pytest.mark.parametrize('kind', ['long', 'strided', 'float64', 'empty', 'vector'])
def test_ineligible_shapes_never_compile(kind, monkeypatch):
    arrays = dict(long=cp.ones((2, 1921), cp.float32),
                  strided=cp.ones((2, 128), cp.float32)[:, ::2],
                  float64=cp.ones((2, 32), cp.float64),
                  empty=cp.ones((0, 32), cp.float32), vector=cp.ones(32, cp.float32))
    cache = NativeShortPrefixCache()
    monkeypatch.setattr(cache, 'compile', lambda: pytest.fail('unsupported shape compiled'))
    assert cache.prefix(arrays[kind]) is None
    assert not cache.status['supported']
    assert cache.status['cached_context_count'] == 0


@pytest.mark.parametrize('variable', ['NVCC', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'])
def test_injected_compiler_configuration_uses_native_graph(variable, monkeypatch):
    monkeypatch.setenv(variable, '--use_fast_math')
    array = cp.ones((2, 31), cp.float32)
    cache = NativeShortPrefixCache()
    monkeypatch.setattr(cache, 'compile', lambda: pytest.fail('injected flags compiled'))
    assert cache.prefix(array) is None
    assert 'compiler' in cache.status['fallback_reason'] or 'NVCC' in cache.status['fallback_reason']


def test_accelerator_changes_are_rechecked_after_success(supported_cache):
    from cupy._core import _accelerator
    array = cp.ones((2, 31), cp.float32)
    original = _accelerator.get_routine_accelerators()
    try:
        _accelerator.set_routine_accelerators([])
        assert supported_cache.prefix(array) is None
        assert 'disabled' in supported_cache.status['fallback_reason']
    finally:
        _accelerator.set_routine_accelerators(original)
    assert supported_cache.prefix(array) is not None
    assert supported_cache.status['active']


@pytest.mark.parametrize('failure', ['compiler', 'canary'])
def test_unavailable_compiler_or_canary_mismatch_is_cached_fallback(monkeypatch, failure):
    array = cp.ones((2, 31), cp.float32)
    cache = NativeShortPrefixCache()
    monkeypatch.setattr(cache, 'unsupported_reason', lambda array: None)
    attempts = []
    def compile():
        attempts.append(True)
        if failure == 'compiler':
            raise FileNotFoundError('nvcc unavailable')
        return object(), lambda *args: pytest.fail('failed canary dispatched real input')
    monkeypatch.setattr(cache, 'compile', compile)
    monkeypatch.setattr(cache, 'canary', lambda kernel: False)
    assert cache.prefix(array) is None
    assert cache.prefix(array) is None
    assert len(attempts) == 1
    assert cache.status['cached_context_count'] == 1
    assert cache.status['cached_module_count'] == 0
    assert cache.status['fallback_calls'] == 2
    json.dumps(cache.status, allow_nan=False)


def test_launched_kernel_failure_is_not_silently_fallback(monkeypatch):
    cache = NativeShortPrefixCache()
    monkeypatch.setattr(cache, 'unsupported_reason', lambda array: None)
    def broken(*args):
        raise RuntimeError('launched kernel failed')
    monkeypatch.setattr(cache, 'compile', lambda: (object(), broken))
    monkeypatch.setattr(cache, 'canary', lambda kernel: True)
    with pytest.raises(RuntimeError, match='launched kernel failed'):
        cache.prefix(cp.ones((2, 31), cp.float32))
    assert cache.status['fallback_calls'] == 0


def test_context_cache_does_not_reuse_modules_across_contexts(monkeypatch):
    cache = NativeShortPrefixCache(max_contexts=2)
    array = cp.ones((2, 31), cp.float32)
    monkeypatch.setattr(cache, 'unsupported_reason', lambda array: None)
    monkeypatch.setattr(cache, 'canary', lambda kernel: True)
    current = [101]
    used = []
    def compile():
        context = current[0]
        return object(), lambda *args: used.append(context)
    monkeypatch.setattr(cache, 'compile', compile)
    monkeypatch.setattr(cp.cuda.driver, 'ctxGetCurrent', lambda: current[0])
    assert cache.prefix(array) is not None
    current[0] = 202
    assert cache.prefix(array) is not None
    current[0] = 101
    assert cache.prefix(array) is not None
    assert used == [101, 202, 101]
    current[0] = 303
    assert cache.prefix(array) is None
    assert cache.status['fallback_reason'] == 'context cache limit reached'
    assert cache.status['cached_context_count'] == 2
    with pytest.raises(TypeError):
        NativeShortPrefixCache(max_contexts=1.5)


def test_engine_dispatches_both_short_and_native_graph_branches(monkeypatch, supported_cache):
    monkeypatch.setattr(engine._PREFIX_PLANS, 'short', supported_cache, raising=False)
    graph = engine._PrefixPlanCache()
    monkeypatch.setattr(engine._PREFIX_PLANS, 'cache', graph, raising=False)
    try:
        short = cp.ones((2, 1475), cp.float32)
        equal_bits(engine._native_flux_prefix(short), engine._row_flux_prefix(short))
        assert engine._native_short_prefix_status()['active']
        assert not graph.plans
        long = cp.ones((2, 1921), cp.float32)
        equal_bits(engine._native_flux_prefix(long), engine._row_flux_prefix(long))
        assert not engine._native_short_prefix_status()['active']
        assert len(graph.plans) == 1
        # A short but non-contiguous row view also retains graph semantics.
        strided = cp.ones((2, 128), cp.float32)[:, ::2]
        equal_bits(engine._native_flux_prefix(strided), engine._row_flux_prefix(strided))
        assert len(graph.plans) == 2
    finally:
        graph.close()
