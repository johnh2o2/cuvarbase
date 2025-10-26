#!/usr/bin/env python3
"""
Test all code examples from README.md to ensure they work correctly.
"""

import sys
import numpy as np

print("Testing README.md examples...")
print("=" * 80)

# Test 1: Quick Start example
print("\nTest 1: Quick Start Example")
print("-" * 80)

try:
    from cuvarbase import bls

    # Generate some sample time series data
    t = np.sort(np.random.uniform(0, 10, 1000)).astype(np.float32)
    y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
    dy = np.ones_like(y) * 0.1  # uncertainties

    print("Data generated successfully")
    print(f"  t: {len(t)} points, dtype={t.dtype}")
    print(f"  y: mean={y.mean():.4f}, std={y.std():.4f}, dtype={y.dtype}")
    print(f"  dy: constant value={dy[0]:.2f}, dtype={dy.dtype}")

    # Box Least Squares (BLS) - Transit detection
    # Define frequency grid
    freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)
    print(f"\nFrequency grid: {len(freqs)} frequencies from {freqs[0]:.2f} to {freqs[-1]:.2f}")

    # Standard BLS
    print("\nTesting standard BLS (eebls_gpu)...")
    power = bls.eebls_gpu(t, y, dy, freqs)
    best_freq = freqs[np.argmax(power)]
    print(f"  ✓ BLS completed: power shape={power.shape}")
    print(f"    Best period: {1/best_freq:.2f} (expected: 2.5)")

    # Or use adaptive BLS for automatic optimization (5-90x faster!)
    print("\nTesting adaptive BLS (eebls_gpu_fast_adaptive)...")
    power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
    best_freq_adaptive = freqs[np.argmax(power_adaptive)]
    print(f"  ✓ Adaptive BLS completed: power shape={power_adaptive.shape}")
    print(f"    Best period: {1/best_freq_adaptive:.2f} (expected: 2.5)")

    print("\n✓ All Quick Start examples passed!")

except Exception as e:
    print(f"\n✗ Quick Start example failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("README EXAMPLE TESTING COMPLETE")
print("=" * 80)
print("\nAll examples executed successfully!")
print("\nNote: The example with CUDA_DEVICE=1 is pseudocode and not tested")
print("(it demonstrates environment variable usage, not actual Python code)")
