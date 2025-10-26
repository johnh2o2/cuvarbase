#!/usr/bin/env python3
"""
Test correctness of adaptive BLS kernel across different block sizes.

Verifies that results are identical regardless of block size selection.
"""

import numpy as np
from cuvarbase import bls

def generate_test_data(ndata, seed=42):
    """Generate synthetic lightcurve data."""
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 100, ndata)).astype(np.float32)
    y = np.ones(ndata, dtype=np.float32)

    # Add transit signal
    period = 5.0
    depth = 0.01
    phase = (t % period) / period
    in_transit = (phase > 0.4) & (phase < 0.5)
    y[in_transit] -= depth

    # Add noise
    y += np.random.normal(0, 0.01, ndata).astype(np.float32)
    dy = np.ones(ndata, dtype=np.float32) * 0.01

    return t, y, dy


def test_block_sizes():
    """Test that all block sizes produce identical results."""
    print("=" * 80)
    print("ADAPTIVE BLS CORRECTNESS TEST")
    print("=" * 80)
    print()

    # Test different ndata values that trigger different block sizes
    test_configs = [
        (10, 32),    # Should use block_size=32
        (50, 64),    # Should use block_size=64
        (100, 128),  # Should use block_size=128
        (500, 256),  # Should use block_size=256
    ]

    freqs = np.linspace(0.05, 0.5, 100).astype(np.float32)

    all_passed = True

    for ndata, expected_block_size in test_configs:
        print(f"Testing ndata={ndata} (expected block_size={expected_block_size})...")

        t, y, dy = generate_test_data(ndata)

        # Get actual block size selected
        actual_block_size = bls._choose_block_size(ndata)
        print(f"  Selected block_size: {actual_block_size}")

        if actual_block_size != expected_block_size:
            print(f"  WARNING: Expected {expected_block_size}, got {actual_block_size}")

        # Run adaptive version
        power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

        # Run standard version with same block size for comparison
        functions_std = bls.compile_bls(block_size=actual_block_size, use_optimized=True,
                                        function_names=['full_bls_no_sol_optimized'])
        power_std = bls.eebls_gpu_fast_optimized(t, y, dy, freqs, functions=functions_std,
                                                  block_size=actual_block_size)

        # Compare
        diff = power_adaptive - power_std
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))

        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        if max_diff > 1e-6:
            print(f"  ✗ FAIL: Differences too large")
            all_passed = False

            # Show worst cases
            worst_idx = np.argsort(np.abs(diff))[::-1][:5]
            print("  Top 5 worst disagreements:")
            for idx in worst_idx:
                print(f"    freq={freqs[idx]:.4f}: adaptive={power_adaptive[idx]:.6f}, "
                      f"std={power_std[idx]:.6f}, diff={diff[idx]:+.2e}")
        else:
            print(f"  ✓ PASS")

        # Also test against fixed block_size=256 baseline
        functions_256 = bls.compile_bls(block_size=256, use_optimized=True,
                                        function_names=['full_bls_no_sol_optimized'])
        power_256 = bls.eebls_gpu_fast_optimized(t, y, dy, freqs, functions=functions_256,
                                                  block_size=256)

        diff_256 = power_adaptive - power_256
        max_diff_256 = np.max(np.abs(diff_256))

        print(f"  Comparison vs block_size=256:")
        print(f"    Max difference: {max_diff_256:.2e}")

        if max_diff_256 > 1e-6:
            print(f"    ✗ Results differ from baseline!")
            all_passed = False
        else:
            print(f"    ✓ Agrees with baseline")

        print()

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = test_block_sizes()
    exit(0 if success else 1)
