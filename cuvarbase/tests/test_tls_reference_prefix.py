"""GPU regressions for exact native TLS scans and physical workspace chunks.

The baseline uses GTLS's one-dimensional float32 cumsum calls. Matrix-axis
scans can change their addition order and the subsequent mean-depth gates.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

cp = pytest.importorskip('cupy')

from cuvarbase import tls_reference as baseline_engine
from cuvarbase import tls_reference_experimental as experimental_engine
from cuvarbase.tls_reference_prefix import NativePrefixPlan


pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def cuda_device():
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip('CUDA device unavailable: {}'.format(error))
    if available < 1:
        pytest.skip('CUDA device unavailable')
    yield
    for backend in (baseline_engine, experimental_engine):
        cache = getattr(backend._PREFIX_PLANS, 'cache', None)
        if cache is not None:
            cache.close()


@pytest.fixture(params=[baseline_engine, experimental_engine], ids=['baseline', 'experimental'])
def engine(request):
    return request.param


def _bitwise_equal(actual, expected):
    actual = cp.asnumpy(actual)
    expected = cp.asnumpy(expected)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _literal_download_winners(scan, t, order, chi2, starts, indices, widths, depths):
    """Original six-download path, independent of the new pack kernel."""
    rows, ndata = order.shape
    safe_start = cp.clip(starts, 0, ndata - 1)
    epochs = t[order[cp.arange(rows, dtype=cp.int32), safe_start]]
    epochs = cp.where(starts >= 0, epochs, np.nan)
    return dict(chi2=chi2.get(), start=starts.get(), start_time=epochs.get(),
                width_index=indices.get(), width=widths.get(), depth=depths.get())


def test_packed_winners_preserve_float_bits_indices_and_absolute_epochs():
    engine = experimental_engine
    rows, ndata = 257, 13  # Cross the launch-block boundary.
    rng = np.random.default_rng(9251)
    t = cp.asarray(2457000. + rng.uniform(0, 30., ndata))
    order = cp.asarray(np.array([rng.permutation(ndata) for _ in range(rows)]), cp.int32)
    starts = cp.asarray(rng.integers(-1, ndata + 2, rows), cp.int32)
    indices = cp.asarray(rng.integers(-1, 17, rows), cp.int32)
    widths = cp.asarray(rng.integers(0, ndata, rows), cp.int32)
    # Include payload-bearing NaNs, infinities, subnormals and signed zeros;
    # none should be rounded or normalized by a float-to-float packing cast.
    words = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000,
                      0x7fc00123, 0xffc00123, 0x3f800000], dtype=np.uint32)
    chi2 = cp.asarray(np.resize(words, rows).view(np.float32))
    depths = cp.asarray(np.resize(words[::-1], rows).view(np.float32))
    args = (engine.modules()[1], t, order, chi2, starts, indices, widths, depths)
    expected = _literal_download_winners(*args)
    actual = engine._download_winners(*args)
    assert actual.dtype == engine._WINNER_DTYPE
    for field in expected:
        dtype = np.uint64 if expected[field].dtype.itemsize == 8 else np.uint32
        np.testing.assert_array_equal(actual[field].view(dtype), expected[field].view(dtype))


@pytest.mark.parametrize('columns', [17, 129, 513, 1301, 10003])
def test_graph_prefix_matches_native_float32_rows(engine, columns):
    rng = np.random.default_rng(columns)
    values = (1 + rng.normal(0., .005, (3, columns))).astype(np.float32)
    values[:, :max(1, columns // 50)] -= .02
    flux = cp.asarray(values)
    baseline = engine._row_flux_prefix(flux)
    with NativePrefixPlan(flux.shape) as plan:
        observed = plan(flux)
        assert observed is plan.output
        assert observed.data.ptr != flux.data.ptr
        _bitwise_equal(observed, baseline)
        assert plan.owned_bytes >= plan.buffer_bytes
        assert plan.owned_bytes == plan.buffer_bytes + plan.workspace_bytes
    assert plan.closed
    assert plan.owned_bytes == 0


def test_prefix_reuse_and_stream_changes_preserve_new_inputs(engine):
    rng = np.random.default_rng(732)
    values = (1 + rng.normal(0., .005, (3, 1301))).astype(np.float32)
    with NativePrefixPlan(values.shape) as plan:
        first = cp.asarray(values)
        _bitwise_equal(plan(first), engine._row_flux_prefix(first))
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            second = cp.asarray(values * np.float32(.999))
            _bitwise_equal(plan(second), engine._row_flux_prefix(second))
        # Returning to the default stream must not race the previous replay.
        third = cp.asarray(values + np.float32(.00001))
        _bitwise_equal(plan(third), engine._row_flux_prefix(third))
    plan.close()  # Explicit release is idempotent.
    with pytest.raises(RuntimeError, match='closed'):
        plan(third)


def test_prefix_contract_rejects_wrong_inputs_and_memory_budget(engine):
    shape = (2, 129)
    with pytest.raises(ValueError, match='positive two-dimensional'):
        NativePrefixPlan((0, 129))
    with pytest.raises(MemoryError, match='buffers'):
        NativePrefixPlan(shape, max_bytes=NativePrefixPlan.buffer_bytes_for(shape) - 1)
    with NativePrefixPlan(shape) as plan:
        with pytest.raises(TypeError, match='CuPy array'):
            plan(np.ones(shape, np.float32))
        with pytest.raises(ValueError, match='shape and float32'):
            plan(cp.ones(shape, cp.float64))
        with pytest.raises(ValueError, match='shape and float32'):
            plan(cp.ones((3, 129), cp.float32))
        _bitwise_equal(plan(cp.ones(shape, cp.float32)),
                       engine._row_flux_prefix(cp.ones(shape, cp.float32)))


def test_prefix_cache_lru_and_byte_limits_release_evicted_plans(engine):
    cache = engine._PrefixPlanCache(max_plans=2, max_bytes=64 * 1024)
    try:
        first, second, third = (cp.ones(shape, cp.float32)
                                for shape in ((2, 129), (3, 513), (4, 1024)))
        for flux in (first, second):
            _bitwise_equal(cache.prefix(flux), engine._row_flux_prefix(flux))
        old_second = cache.plans[NativePrefixPlan.key_for(second.shape)]
        _bitwise_equal(cache.prefix(first), engine._row_flux_prefix(first))
        _bitwise_equal(cache.prefix(third), engine._row_flux_prefix(third))
        assert old_second.closed
        assert len(cache.plans) == 2
        assert list(cache.plans) == [NativePrefixPlan.key_for(first.shape),
                                     NativePrefixPlan.key_for(third.shape)]
        assert cache.owned_bytes <= cache.max_bytes
    finally:
        cache.close()
    assert cache.owned_bytes == 0

    cache = engine._PrefixPlanCache(max_plans=4, max_bytes=20 * 1024)
    try:
        first, second = (cp.ones(shape, cp.float32)
                         for shape in ((2, 513), (3, 513)))
        _bitwise_equal(cache.prefix(first), engine._row_flux_prefix(first))
        old_first = next(iter(cache.plans.values()))
        _bitwise_equal(cache.prefix(second), engine._row_flux_prefix(second))
        assert old_first.closed
        assert len(cache.plans) == 1
        assert cache.owned_bytes <= cache.max_bytes
    finally:
        cache.close()


def test_oversized_prefix_uses_exact_uncached_row_scans(engine):
    cache = engine._PrefixPlanCache(max_bytes=64)
    flux = cp.asarray(np.random.default_rng(713).normal(1., .005, (2, 129)),
                      dtype=cp.float32)
    _bitwise_equal(cache.prefix(flux), engine._row_flux_prefix(flux))
    assert not cache.plans
    assert cache.owned_bytes == 0


def test_prefix_caches_are_thread_local(engine, monkeypatch):
    # Exercise the graph cache explicitly even when short native CUB scans
    # are supported on the test device.
    if engine is experimental_engine:
        monkeypatch.setattr(engine.NativeShortPrefixCache, 'unsupported_reason',
                            staticmethod(lambda array: 'graph-specific regression'))
    main_flux = cp.ones((2, 129), cp.float32)
    _bitwise_equal(engine._native_flux_prefix(main_flux),
                   engine._row_flux_prefix(main_flux))
    main_cache = engine._PREFIX_PLANS.cache
    device = cp.cuda.runtime.getDevice()

    def worker():
        with cp.cuda.Device(device):
            flux = cp.ones((2, 129), cp.float32)
            _bitwise_equal(engine._native_flux_prefix(flux),
                           engine._row_flux_prefix(flux))
            cache = engine._PREFIX_PLANS.cache
            different = cache is not main_cache
            cache.close()
            return different

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result()
    assert not next(iter(main_cache.plans.values())).closed


@pytest.mark.parametrize('full', [False, True])
def test_physical_workspace_budget_limits_rows_or_fails_before_search(engine, full):
    plan = engine._physical_chunk_plan(10003, 11205, 35, 1400, 256,
                                       full=full, free_bytes=64 * 1024**2)
    assert 1 <= plan['rows'] < 256
    assert plan['budget_bytes'] == 16 * 1024**2
    assert plan['estimated_chunk_bytes'] <= plan['budget_bytes']
    with pytest.raises(MemoryError, match='One TLS period'):
        engine._physical_chunk_plan(10003, 11205, 35, 1400, 256,
                                     full=full, free_bytes=4)


def _search_fixture(engine):
    pytest.importorskip('batman')
    rng = np.random.default_rng(9320)
    times = np.sort(rng.uniform(.1, 35., 1025))
    flux = 1 + rng.normal(0., .001, len(times))
    flux[np.remainder(times, 2.03) < .08] -= .01
    errors = rng.uniform(.0008, .0012, len(times))
    periods = np.linspace(2., 2.05, 6)
    prepared = engine.reference.preprocess_inputs(times, flux, errors)
    cache = engine.reference.build_cache(periods, len(prepared['t']))
    return periods, prepared, cache


@pytest.mark.parametrize('full', [False, True])
def test_raw_search_chunking_and_graph_preserve_scores_and_observation_epochs(engine, monkeypatch, full):
    periods, prepared, cache = _search_fixture(engine)
    inputs = (periods, prepared['t'], prepared['y'], prepared['dy'], cache)
    graph = engine._native_flux_prefix
    download = getattr(engine, '_download_winners', None)
    monkeypatch.setattr(engine, '_native_flux_prefix', engine._row_flux_prefix)
    if download is not None:
        monkeypatch.setattr(engine, '_download_winners', _literal_download_winners)
    old = engine.raw_search(*inputs, group_size=4, work_chunk=256,
                            capture=True, full=full)
    assert np.all(old['start'] >= 0)
    assert np.all(np.isfinite(old['chi2']) & (old['chi2'] > 0))
    per_row = min(plan['estimated_bytes_per_row'] for plan in old['work_chunk_plans'])
    monkeypatch.setattr(engine, '_WORKSPACE_BYTES', 2 * per_row)
    monkeypatch.setattr(engine, '_native_flux_prefix', graph)
    if download is not None:
        monkeypatch.setattr(engine, '_download_winners', download)
    new = engine.raw_search(*inputs, group_size=4, work_chunk=256, full=full)
    for field in ('chi2', 'start', 'start_time', 'width_index', 'width', 'depth',
                  'width_masks', 'group_ranges'):
        np.testing.assert_array_equal(new[field], old[field])
    assert all(chunk['rows'] <= 2 for chunk in new['work_chunks'])
    assert len(new['work_chunks']) > len(old['work_chunks'])
    for chunk in old['captured']:
        first, last = chunk['start'], chunk['stop']
        start = old['start'][first:last]
        original = prepared['t'][chunk['order'][np.arange(last - first), start]]
        np.testing.assert_array_equal(new['start_time'][first:last], original)


def test_explicit_duration_runs_never_broaden_another_periods_interval(engine):
    periods, prepared, cache = _search_fixture(engine)
    widths = cache['widths']
    assert len(widths) >= 3
    lower = np.array([widths[0], widths[0], widths[1], widths[2], widths[1], widths[1]])
    upper = lower.copy()
    selection = dict(width_minima=lower, width_maxima=upper)
    result = engine.raw_search(periods, prepared['t'], prepared['y'], prepared['dy'], cache,
        group_size=6, work_chunk=2, duration_selection=selection)
    np.testing.assert_array_equal(result['group_ranges'], [[0, 2], [2, 3], [3, 4], [4, 6]])
    assert result['width_masks'] is None  # No periods-by-widths retained table.
    assert np.all(result['start'] >= 0)
    np.testing.assert_array_equal(result['width'], lower)
    for group, (first, last) in enumerate(result['group_ranges']):
        admissible = (widths >= lower[first]) & (widths <= upper[first])
        assert np.count_nonzero(admissible) == 1
        assert np.all(lower[first:last] == lower[first])
    # Reconstruct each period independently to verify that run grouping does
    # not add a trial from another period, including nonconsecutive repeats.
    for index, period in enumerate(periods):
        single = engine.raw_search([period], prepared['t'], prepared['y'], prepared['dy'], cache,
            group_size=1, work_chunk=1,
            duration_selection=dict(width_minima=lower[index:index + 1],
                                    width_maxima=upper[index:index + 1]))
        for field in ('chi2', 'start', 'start_time', 'width_index', 'width', 'depth'):
            np.testing.assert_array_equal(result[field][index:index + 1], single[field])


def test_real_backend_modules_and_graph_buffers_do_not_cross_execution_modes(monkeypatch):
    import threading

    # New local state makes the lifetime proof independent of earlier tests.
    for backend in (baseline_engine, experimental_engine):
        monkeypatch.setattr(backend, '_PREFIX_PLANS', threading.local())
    short_calls = []

    def graph_fallback(self, flux):
        short_calls.append(flux)
        return None

    monkeypatch.setattr(experimental_engine.NativeShortPrefixCache, 'prefix', graph_fallback)
    flux = cp.ones((2, 129), cp.float32)
    try:
        default = baseline_engine._native_flux_prefix(flux)
        assert not short_calls
        assert not hasattr(baseline_engine._PREFIX_PLANS, 'short')
        optimized = experimental_engine._native_flux_prefix(flux)
        assert len(short_calls) == 1
        assert default.data.ptr != optimized.data.ptr
        assert baseline_engine._PREFIX_PLANS.cache is not experimental_engine._PREFIX_PLANS.cache
        _bitwise_equal(default, optimized)
        baseline_engine._native_flux_prefix(flux)
        assert len(short_calls) == 1
        baseline_modules = baseline_engine.modules()
        experimental_modules = experimental_engine.modules()
        assert all(a is not b for a, b in zip(baseline_modules, experimental_modules))
        assert baseline_engine.modules() is baseline_modules
        assert experimental_engine.modules() is experimental_modules
    finally:
        for backend in (baseline_engine, experimental_engine):
            cache = getattr(backend._PREFIX_PLANS, 'cache', None)
            if cache is not None:
                cache.close()
