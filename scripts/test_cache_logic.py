#!/usr/bin/env python3
"""
Test kernel cache logic without GPU (unit tests for LRU and thread-safety).

Tests the cache implementation directly without requiring CUDA.
"""

import threading
import time
from collections import OrderedDict


# Simulated version of bls._get_cached_kernels for testing
class MockKernelCache:
    """Mock kernel cache for testing LRU and thread-safety."""

    def __init__(self, max_size=20):
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.max_size = max_size
        self.compilation_count = 0

    def _compile_kernel(self, key):
        """Simulate kernel compilation (slow operation)."""
        self.compilation_count += 1
        time.sleep(0.01)  # Simulate compilation time
        return f"kernel_{key}"

    def get_cached_kernels(self, block_size, use_optimized=False, function_names=None):
        """Get compiled kernels from cache with LRU eviction and thread-safety."""
        if function_names is None:
            function_names = ['default']

        key = (block_size, use_optimized, tuple(sorted(function_names)))

        with self.lock:
            # Check if key exists and move to end (most recently used)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            # Compile kernel (done inside lock to prevent duplicate compilation)
            compiled_kernel = self._compile_kernel(key)

            # Add to cache
            self.cache[key] = compiled_kernel
            self.cache.move_to_end(key)

            # Evict oldest entry if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest (FIFO = LRU)

            return compiled_kernel


def test_basic_caching():
    """Test basic caching functionality."""
    print("=" * 80)
    print("TEST 1: Basic Caching")
    print("=" * 80)

    cache = MockKernelCache(max_size=5)

    # First call should compile
    print("First call (should compile)...")
    result1 = cache.get_cached_kernels(256, use_optimized=True)
    assert cache.compilation_count == 1, "Should have compiled once"
    print(f"  ✓ Compiled (count={cache.compilation_count})")

    # Second call should be cached
    print("Second call (should be cached)...")
    result2 = cache.get_cached_kernels(256, use_optimized=True)
    assert cache.compilation_count == 1, "Should not compile again"
    assert result1 == result2, "Should return same result"
    print(f"  ✓ Cached (count={cache.compilation_count})")

    print()


def test_lru_eviction():
    """Test LRU eviction."""
    print("=" * 80)
    print("TEST 2: LRU Eviction")
    print("=" * 80)

    max_size = 5
    cache = MockKernelCache(max_size=max_size)

    print(f"Max cache size: {max_size}")
    print()

    # Fill cache beyond max size
    print("Filling cache with 8 entries...")
    keys = []
    for i in range(8):
        block_size = 32 * (i + 1)
        _ = cache.get_cached_kernels(block_size, use_optimized=True)
        keys.append((block_size, True, ('default',)))
        print(f"  Entry {i+1}: cache size = {len(cache.cache)}")

    print()
    print(f"Final cache size: {len(cache.cache)}")
    assert len(cache.cache) <= max_size, f"Cache size {len(cache.cache)} exceeds max {max_size}"
    print(f"  ✓ Cache bounded to {max_size}")

    # Verify oldest entries were evicted
    num_evicted = 8 - max_size
    for i, key in enumerate(keys[:num_evicted]):
        assert key not in cache.cache, f"Oldest key {i} should be evicted"
    print(f"  ✓ Oldest {num_evicted} entries evicted")

    # Verify newest entries retained
    for key in keys[-max_size:]:
        assert key in cache.cache, "Recent key should be retained"
    print(f"  ✓ Most recent {max_size} entries retained")

    print()


def test_lru_access_order():
    """Test that accessing an old entry moves it to the end."""
    print("=" * 80)
    print("TEST 3: LRU Access Order")
    print("=" * 80)

    cache = MockKernelCache(max_size=3)

    # Add 3 entries
    print("Adding 3 entries...")
    cache.get_cached_kernels(32, use_optimized=True)
    cache.get_cached_kernels(64, use_optimized=True)
    cache.get_cached_kernels(128, use_optimized=True)
    print(f"  Cache: {list(cache.cache.keys())}")
    print()

    # Access first entry (should move to end)
    print("Accessing first entry (32)...")
    cache.get_cached_kernels(32, use_optimized=True)
    print(f"  Cache: {list(cache.cache.keys())}")
    print(f"  ✓ Entry moved to end")
    print()

    # Add new entry (should evict 64, not 32)
    print("Adding new entry (should evict 64, not 32)...")
    cache.get_cached_kernels(256, use_optimized=True)
    print(f"  Cache: {list(cache.cache.keys())}")

    assert (32, True, ('default',)) in cache.cache, "32 should be retained (recently accessed)"
    assert (64, True, ('default',)) not in cache.cache, "64 should be evicted (oldest)"
    assert (256, True, ('default',)) in cache.cache, "256 should be added"
    print(f"  ✓ LRU eviction works correctly")

    print()


def test_thread_safety():
    """Test thread-safety."""
    print("=" * 80)
    print("TEST 4: Thread-Safety")
    print("=" * 80)

    cache = MockKernelCache(max_size=10)
    num_threads = 20
    results = [None] * num_threads
    errors = []

    def worker(thread_id):
        """Worker thread."""
        try:
            # Mix of shared and unique keys
            block_size = 128 if thread_id % 2 == 0 else 256
            result = cache.get_cached_kernels(block_size, use_optimized=True)
            results[thread_id] = result
        except Exception as e:
            errors.append((thread_id, str(e)))

    print(f"Launching {num_threads} threads...")

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print()

    if errors:
        print("ERRORS:")
        for thread_id, error in errors:
            print(f"  Thread {thread_id}: {error}")
        assert False, "Thread-safety test failed"
    else:
        print(f"  ✓ No errors from {num_threads} threads")

    # Should only have 2 unique keys (128 and 256)
    assert len(cache.cache) == 2, f"Expected 2 cache entries, got {len(cache.cache)}"
    print(f"  ✓ Cache has 2 entries (no duplicate compilations)")

    # Compilation count should be 2 (not 20)
    assert cache.compilation_count == 2, f"Expected 2 compilations, got {cache.compilation_count}"
    print(f"  ✓ Only 2 compilations (thread-safe)")

    print()


def test_concurrent_same_key():
    """Test concurrent compilation of same key."""
    print("=" * 80)
    print("TEST 5: Concurrent Same-Key Compilation")
    print("=" * 80)

    cache = MockKernelCache(max_size=10)
    num_threads = 50
    results = [None] * num_threads
    errors = []

    def worker(thread_id):
        """All threads compile same kernel."""
        try:
            result = cache.get_cached_kernels(256, use_optimized=True)
            results[thread_id] = result
        except Exception as e:
            errors.append((thread_id, str(e)))

    print(f"Launching {num_threads} threads for same kernel...")

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print()

    if errors:
        print("ERRORS:")
        for thread_id, error in errors:
            print(f"  Thread {thread_id}: {error}")
        assert False, "Concurrent compilation failed"
    else:
        print(f"  ✓ No errors from {num_threads} threads")

    # All should get same result
    assert len(set(results)) == 1, "All threads should get same result"
    print(f"  ✓ All threads got identical result")

    # Should only compile once
    assert cache.compilation_count == 1, f"Expected 1 compilation, got {cache.compilation_count}"
    print(f"  ✓ Only 1 compilation (no race conditions)")

    print()


def main():
    """Run all tests."""
    print()
    print("KERNEL CACHE LOGIC TEST SUITE")
    print("(Tests cache implementation without requiring GPU)")
    print()

    try:
        test_basic_caching()
        test_lru_eviction()
        test_lru_access_order()
        test_thread_safety()
        test_concurrent_same_key()

        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ Basic caching works correctly")
        print("  ✓ LRU eviction prevents unbounded growth")
        print("  ✓ LRU access ordering works correctly")
        print("  ✓ Thread-safe concurrent access")
        print("  ✓ No duplicate compilations from race conditions")
        print()
        print("The implementation in cuvarbase/bls.py uses the same logic")
        print("and should work identically with real CUDA kernels.")
        print()

        return True

    except AssertionError as e:
        print()
        print("=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
