#!/usr/bin/env python3
"""
Test kernel cache thread-safety and LRU eviction policy.

Tests:
1. Basic caching functionality
2. LRU eviction when cache is full
3. Thread-safety with concurrent kernel compilation
"""

import numpy as np
import threading
import time
import sys

try:
    from cuvarbase import bls
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU not available: {e}")
    sys.exit(1)


def test_basic_caching():
    """Test that kernels are cached and reused."""
    print("=" * 80)
    print("TEST 1: Basic Caching")
    print("=" * 80)
    print()

    # Clear cache
    bls._kernel_cache.clear()

    # First call should compile
    print("First call (should compile)...")
    start = time.time()
    funcs1 = bls._get_cached_kernels(256, use_optimized=True,
                                     function_names=['full_bls_no_sol_optimized'])
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1:.4f}s")
    print(f"  Cache size: {len(bls._kernel_cache)}")

    # Second call should be cached
    print("Second call (should be cached)...")
    start = time.time()
    funcs2 = bls._get_cached_kernels(256, use_optimized=True,
                                     function_names=['full_bls_no_sol_optimized'])
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2:.4f}s")
    print(f"  Cache size: {len(bls._kernel_cache)}")

    # Verify same object returned
    assert funcs1 is funcs2, "Cache should return same object"
    print(f"  ✓ Same object returned (funcs1 is funcs2)")

    # Verify speedup from caching
    speedup = elapsed1 / elapsed2
    print(f"  ✓ Speedup from caching: {speedup:.1f}x")
    assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"

    print()


def test_lru_eviction():
    """Test LRU eviction when cache exceeds max size."""
    print("=" * 80)
    print("TEST 2: LRU Eviction")
    print("=" * 80)
    print()

    # Clear cache
    bls._kernel_cache.clear()

    max_size = bls._KERNEL_CACHE_MAX_SIZE
    print(f"Max cache size: {max_size}")
    print()

    # Fill cache beyond max size
    block_sizes = [32, 64, 128, 256]
    use_optimized_vals = [True, False]

    print(f"Filling cache with {max_size + 5} different configurations...")

    cache_keys = []
    for i in range(max_size + 5):
        block_size = block_sizes[i % len(block_sizes)]
        use_optimized = use_optimized_vals[i % len(use_optimized_vals)]

        # Use different function subsets to create unique keys
        if i % 3 == 0:
            function_names = ['full_bls_no_sol_optimized']
        elif i % 3 == 1:
            function_names = ['full_bls_no_sol']
        else:
            function_names = ['reduction_max']

        key = (block_size, use_optimized, tuple(sorted(function_names)))
        cache_keys.append(key)

        _ = bls._get_cached_kernels(block_size, use_optimized, function_names)

        current_size = len(bls._kernel_cache)
        if i < 5 or i >= max_size:
            print(f"  Entry {i+1}: cache size = {current_size}")

    print()
    final_size = len(bls._kernel_cache)
    print(f"Final cache size: {final_size}")
    assert final_size <= max_size, f"Cache size {final_size} exceeds max {max_size}"
    print(f"  ✓ Cache size bounded to {max_size}")

    # Verify oldest entries were evicted
    print()
    print("Checking LRU eviction...")
    num_evicted = len(cache_keys) - max_size

    for i, key in enumerate(cache_keys[:num_evicted]):
        assert key not in bls._kernel_cache, f"Oldest key {i} should be evicted"
    print(f"  ✓ Oldest {num_evicted} entries evicted")

    # Verify newest entries are retained
    for i, key in enumerate(cache_keys[-max_size:]):
        assert key in bls._kernel_cache, f"Recent key should be retained"
    print(f"  ✓ Most recent {max_size} entries retained")

    print()


def test_thread_safety():
    """Test thread-safety with concurrent kernel compilation."""
    print("=" * 80)
    print("TEST 3: Thread-Safety")
    print("=" * 80)
    print()

    # Clear cache
    bls._kernel_cache.clear()

    num_threads = 10
    num_compilations_per_thread = 5

    compilation_times = []
    errors = []

    def worker(thread_id, block_sizes):
        """Worker thread that compiles kernels."""
        try:
            for i, block_size in enumerate(block_sizes):
                start = time.time()
                _ = bls._get_cached_kernels(block_size, use_optimized=True,
                                           function_names=['full_bls_no_sol_optimized'])
                elapsed = time.time() - start
                compilation_times.append(elapsed)

                if i == 0:
                    print(f"  Thread {thread_id}: first compilation = {elapsed:.4f}s")
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create block size sequences (some overlap to test concurrent access)
    block_sizes_per_thread = []
    for i in range(num_threads):
        # Mix of unique and shared block sizes
        sizes = [32, 64, 128, 256, 32][i % 5:i % 5 + num_compilations_per_thread]
        if len(sizes) < num_compilations_per_thread:
            sizes = sizes + [32] * (num_compilations_per_thread - len(sizes))
        block_sizes_per_thread.append(sizes)

    print(f"Launching {num_threads} threads, each compiling {num_compilations_per_thread} kernels...")
    print()

    # Launch threads
    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, block_sizes_per_thread[i]))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    total_time = time.time() - start_time

    print()
    print(f"All threads completed in {total_time:.4f}s")
    print(f"Total compilations: {len(compilation_times)}")
    print(f"Cache size: {len(bls._kernel_cache)}")
    print()

    # Check for errors
    if errors:
        print("ERRORS:")
        for thread_id, error in errors:
            print(f"  Thread {thread_id}: {error}")
        assert False, "Thread-safety test failed with errors"
    else:
        print("  ✓ No race condition errors")

    # Verify cache integrity
    assert len(bls._kernel_cache) <= bls._KERNEL_CACHE_MAX_SIZE, "Cache exceeded max size"
    print(f"  ✓ Cache size within bounds ({len(bls._kernel_cache)} <= {bls._KERNEL_CACHE_MAX_SIZE})")

    # Verify fast cached access
    cached_times = [t for t in compilation_times if t < 0.1]  # Cached should be <100ms
    print(f"  ✓ {len(cached_times)}/{len(compilation_times)} calls were cached (<100ms)")

    print()


def test_concurrent_same_key():
    """Test that concurrent compilation of same key doesn't cause issues."""
    print("=" * 80)
    print("TEST 4: Concurrent Same-Key Compilation")
    print("=" * 80)
    print()

    # Clear cache
    bls._kernel_cache.clear()

    num_threads = 20
    block_size = 128

    results = [None] * num_threads
    errors = []

    def worker(thread_id):
        """All threads try to compile the same kernel simultaneously."""
        try:
            funcs = bls._get_cached_kernels(block_size, use_optimized=True,
                                           function_names=['full_bls_no_sol_optimized'])
            results[thread_id] = funcs
        except Exception as e:
            errors.append((thread_id, str(e)))

    print(f"Launching {num_threads} threads to compile identical kernel...")

    # Launch all threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    print()

    # Check for errors
    if errors:
        print("ERRORS:")
        for thread_id, error in errors:
            print(f"  Thread {thread_id}: {error}")
        assert False, "Concurrent compilation test failed"
    else:
        print("  ✓ No errors from concurrent compilation")

    # Verify all got the same object (from cache)
    first_result = results[0]
    assert first_result is not None, "First thread should have result"

    for i, result in enumerate(results[1:], 1):
        assert result is first_result, f"Thread {i} got different object"

    print(f"  ✓ All {num_threads} threads got identical object (same memory address)")

    # Verify cache has only one entry
    assert len(bls._kernel_cache) == 1, "Should only have one cache entry"
    print(f"  ✓ Cache has exactly 1 entry (no duplicate compilations)")

    print()


def main():
    """Run all tests."""
    print()
    print("KERNEL CACHE TEST SUITE")
    print()

    if not GPU_AVAILABLE:
        print("ERROR: GPU not available")
        return False

    try:
        test_basic_caching()
        test_lru_eviction()
        test_thread_safety()
        test_concurrent_same_key()

        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ Basic caching works correctly")
        print("  ✓ LRU eviction prevents unbounded growth")
        print("  ✓ Thread-safe concurrent access")
        print("  ✓ No duplicate compilations from race conditions")
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
    except Exception as e:
        print()
        print("=" * 80)
        print("TEST ERROR")
        print("=" * 80)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
