"""CPU tests for the BLS kernel LRU cache in ``cuvarbase.bls``.

Ported from ``scripts/test_kernel_cache.py`` (which needed a device and
timed real nvcc compilations).  Here the compile functions behind the
cache (``compile_bls``, ``compile_sparse_bls``, ``compile_bls_batch``)
are replaced by counting stubs and ``ensure_context`` by a no-op, so the
cache's contract -- same object on a hit, one compile per key, bounded
size with least-recently-used eviction, no duplicate compiles under
concurrent first use -- is checked without a GPU.  The cache itself is
swapped for a fresh ``OrderedDict`` per test so nothing leaks into (or
from) the process-wide cache the other tests share.
"""
import threading
import time
from collections import OrderedDict

import pytest

from .. import bls


@pytest.fixture
def cache(monkeypatch):
    """Isolated cache + counting compile stubs. Returns a namespace with
    ``calls`` (list of keys compiled, in order) and ``store`` (the
    OrderedDict standing in for ``bls._kernel_cache``)."""
    calls = []
    store = OrderedDict()

    def fake_compile_bls(block_size=bls._default_block_size,
                         function_names=bls._all_function_names,
                         prepare=True, use_optimized=False, **kwargs):
        # a short sleep widens the window in which a second thread could
        # race into a duplicate compile if the lock were missing
        time.sleep(0.002)
        key = (block_size, use_optimized, tuple(sorted(function_names)))
        calls.append(key)
        return {'key': key, 'prepare': prepare}      # fresh object per call

    def fake_compile_sparse_bls(block_size=bls._default_block_size,
                                **kwargs):
        time.sleep(0.002)
        calls.append((block_size, 'sparse'))
        return {'key': (block_size, 'sparse')}

    def fake_compile_bls_batch(block_size=bls._default_block_size,
                               **kwargs):
        time.sleep(0.002)
        calls.append((block_size, 'batch'))
        return {'key': (block_size, 'batch')}

    monkeypatch.setattr(bls, 'compile_bls', fake_compile_bls)
    monkeypatch.setattr(bls, 'compile_sparse_bls', fake_compile_sparse_bls)
    monkeypatch.setattr(bls, 'compile_bls_batch', fake_compile_bls_batch)
    monkeypatch.setattr(bls, 'ensure_context', lambda: None)
    monkeypatch.setattr(bls, '_kernel_cache', store)

    class NS(object):
        pass
    ns = NS()
    ns.calls = calls
    ns.store = store
    return ns


FN = ['full_bls_no_sol_optimized']


def test_hit_returns_the_same_object_and_compiles_once(cache):
    f1 = bls._get_cached_kernels(256, use_optimized=True, function_names=FN)
    f2 = bls._get_cached_kernels(256, use_optimized=True, function_names=FN)
    assert f1 is f2
    assert len(cache.calls) == 1
    assert len(cache.store) == 1
    # the key is (block_size, use_optimized, sorted function names)
    assert (256, True, ('full_bls_no_sol_optimized',)) in cache.store

    # a different block size / kernel variant / function set is a miss
    bls._get_cached_kernels(128, use_optimized=True, function_names=FN)
    bls._get_cached_kernels(256, use_optimized=False,
                            function_names=['full_bls_no_sol'])
    bls._get_cached_kernels(256, use_optimized=True,
                            function_names=FN + ['full_bls_no_sol_fused'])
    assert len(cache.calls) == 4
    assert len(cache.store) == 4

    # function-name order does not change the key
    f3 = bls._get_cached_kernels(
        256, use_optimized=True,
        function_names=['full_bls_no_sol_fused'] + FN)
    assert len(cache.calls) == 4
    assert f3['key'][2] == ('full_bls_no_sol_fused',
                            'full_bls_no_sol_optimized')


def test_default_function_names_use_the_full_list(cache):
    f = bls._get_cached_kernels(256)
    assert f['key'] == (256, False, tuple(sorted(bls._all_function_names)))


def test_sparse_and_batch_routes_share_the_cache_with_distinct_keys(cache):
    s1 = bls._get_cached_sparse_kernel(256)
    s2 = bls._get_cached_sparse_kernel(256)
    b1 = bls._get_cached_batch_kernels(256)
    b2 = bls._get_cached_batch_kernels(256)
    k = bls._get_cached_kernels(256, use_optimized=False, function_names=FN)
    assert s1 is s2 and b1 is b2
    assert s1 is not b1 and k is not s1
    assert cache.calls == [(256, 'sparse'), (256, 'batch'),
                           (256, False, ('full_bls_no_sol_optimized',))]
    assert set(cache.store) == {(256, 'sparse'), (256, 'batch'),
                                (256, False, ('full_bls_no_sol_optimized',))}


def test_cached_compile_bls_routes_through_the_cache(cache):
    # the default entry points call compile_bls through this wrapper
    a = bls._cached_compile_bls(block_size=128, use_optimized=True,
                                function_names=FN)
    b = bls._cached_compile_bls(block_size=128, use_optimized=True,
                                function_names=FN)
    assert a is b
    assert len(cache.calls) == 1
    # prepare=False is the one option the cache does not model: it
    # falls through to a direct (uncached) compile every time
    c = bls._cached_compile_bls(block_size=128, use_optimized=True,
                                function_names=FN, prepare=False)
    d = bls._cached_compile_bls(block_size=128, use_optimized=True,
                                function_names=FN, prepare=False)
    assert c is not d and c['prepare'] is False
    assert len(cache.calls) == 3
    assert len(cache.store) == 1


def _unique_keys(n):
    """n distinct (block_size, use_optimized, function_names) keys."""
    keys = []
    block_sizes = [32, 64, 128, 256]
    fn_sets = [['full_bls_no_sol_optimized'], ['full_bls_no_sol'],
               ['reduction_max'], ['bin_and_phase_fold_bst_multifreq']]
    for i in range(n):
        bs = block_sizes[i % 4]
        opt = bool((i // 4) % 2)
        fns = fn_sets[(i // 8) % 4]
        keys.append((bs, opt, fns))
    assert len({(bs, opt, tuple(f)) for bs, opt, f in keys}) == n
    return keys


def test_cache_is_bounded_and_evicts_the_oldest(cache):
    max_size = bls._KERNEL_CACHE_MAX_SIZE
    assert max_size >= 2
    n_extra = 5
    keys = _unique_keys(max_size + n_extra)
    for bs, opt, fns in keys:
        bls._get_cached_kernels(bs, opt, fns)
        assert len(cache.store) <= max_size
    assert len(cache.store) == max_size
    assert len(cache.calls) == max_size + n_extra

    as_keys = [(bs, opt, tuple(sorted(f))) for bs, opt, f in keys]
    # the n_extra oldest insertions are gone, the rest retained, and the
    # OrderedDict order is insertion (= recency) order
    for k in as_keys[:n_extra]:
        assert k not in cache.store
    assert list(cache.store) == as_keys[n_extra:]

    # an evicted key recompiles (and lands at the most-recent end)
    bs, opt, fns = keys[0]
    bls._get_cached_kernels(bs, opt, fns)
    assert len(cache.calls) == max_size + n_extra + 1
    assert list(cache.store)[-1] == as_keys[0]
    assert len(cache.store) == max_size


def test_eviction_is_least_recently_used_not_fifo(cache):
    max_size = bls._KERNEL_CACHE_MAX_SIZE
    keys = _unique_keys(max_size + 1)
    as_keys = [(bs, opt, tuple(sorted(f))) for bs, opt, f in keys]
    # fill exactly to capacity
    for bs, opt, fns in keys[:max_size]:
        bls._get_cached_kernels(bs, opt, fns)
    assert len(cache.store) == max_size
    # touch the OLDEST entry: a hit must refresh its recency ...
    bs, opt, fns = keys[0]
    first = bls._get_cached_kernels(bs, opt, fns)
    assert len(cache.calls) == max_size            # a hit, no compile
    assert list(cache.store)[-1] == as_keys[0]
    # ... so the next insertion evicts the SECOND-oldest, not it
    bs, opt, fns = keys[max_size]
    bls._get_cached_kernels(bs, opt, fns)
    assert as_keys[0] in cache.store
    assert as_keys[1] not in cache.store
    assert len(cache.store) == max_size
    assert bls._get_cached_kernels(*keys[0]) is first


def _run_threads(n, target):
    errors = []
    results = [None] * n

    def worker(i):
        try:
            results[i] = target(i)
        except Exception as e:           # pragma: no cover - reported below
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    return results


def test_concurrent_first_use_of_one_key_compiles_once(cache):
    n = 20
    results = _run_threads(
        n, lambda i: bls._get_cached_kernels(128, use_optimized=True,
                                             function_names=FN))
    assert len(cache.calls) == 1, cache.calls
    assert len(cache.store) == 1
    assert all(r is results[0] for r in results)


def test_concurrent_mixed_keys_compile_each_key_once(cache):
    # 10 threads x 5 lookups over 4 distinct block sizes, with heavy
    # overlap between threads: exactly one compile per distinct key,
    # cache within bounds, every thread sees the cached object
    sizes = [32, 64, 128, 256, 32]
    per_thread = [(sizes * 2)[i % 5:i % 5 + 5] for i in range(10)]

    def target(i):
        return [bls._get_cached_kernels(bs, use_optimized=True,
                                        function_names=FN)
                for bs in per_thread[i]]

    results = _run_threads(10, target)
    distinct = {bs for row in per_thread for bs in row}
    assert len(cache.calls) == len(distinct), cache.calls
    assert len(cache.store) == len(distinct) <= bls._KERNEL_CACHE_MAX_SIZE
    by_bs = {}
    for row, objs in zip(per_thread, results):
        for bs, obj in zip(row, objs):
            assert by_bs.setdefault(bs, obj) is obj


def test_lock_is_a_real_lock_and_the_cache_an_ordered_dict():
    # the process-wide objects the routes above rely on
    assert isinstance(bls._kernel_cache, OrderedDict)
    assert hasattr(bls._kernel_cache_lock, 'acquire')
    assert bls._KERNEL_CACHE_MAX_SIZE == 20
