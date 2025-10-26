#!/usr/bin/env python3
"""
Test correctness of optimized BLS kernel.

Checks whether the optimized kernel produces identical results to the standard kernel.
"""

import numpy as np
from cuvarbase import bls

# Generate test data
np.random.seed(42)
ndata = 1000
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

# Create frequency grid
freqs = np.linspace(0.05, 0.5, 100).astype(np.float32)

print("Testing correctness...")
print(f"ndata = {ndata}")
print(f"nfreq = {len(freqs)}")

# Run standard kernel
print("\nRunning standard kernel...")
power_std = bls.eebls_gpu_fast(t, y, dy, freqs)

# Run optimized kernel
print("Running optimized kernel...")
power_opt = bls.eebls_gpu_fast_optimized(t, y, dy, freqs)

# Compare results
diff = power_std - power_opt
max_diff = np.max(np.abs(diff))
mean_diff = np.mean(np.abs(diff))
rms_diff = np.sqrt(np.mean(diff**2))

print(f"\nResults:")
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")
print(f"  RMS difference: {rms_diff:.2e}")
print(f"  Max relative difference: {max_diff / np.max(power_std):.2e}")

# Find where differences are largest
idx_max = np.argmax(np.abs(diff))
print(f"\nLargest difference at index {idx_max}:")
print(f"  Frequency: {freqs[idx_max]:.4f}")
print(f"  Standard: {power_std[idx_max]:.6f}")
print(f"  Optimized: {power_opt[idx_max]:.6f}")
print(f"  Difference: {diff[idx_max]:.6e}")

# Check if results are close enough
tolerance = 1e-4  # Relative tolerance
relative_diff = np.abs(diff) / (np.abs(power_std) + 1e-10)
max_relative = np.max(relative_diff)

print(f"\nMax relative difference: {max_relative:.2e}")
if max_relative < tolerance:
    print(f"✓ PASS: Results agree within {tolerance:.0e} relative tolerance")
else:
    print(f"✗ FAIL: Results differ by more than {tolerance:.0e}")

    # Show top 10 worst disagreements
    worst_idx = np.argsort(np.abs(diff))[::-1][:10]
    print("\nTop 10 worst disagreements:")
    print("  Idx    Freq    Standard   Optimized  AbsDiff    RelDiff")
    for idx in worst_idx:
        print(f"  {idx:<5d}  {freqs[idx]:.4f}  {power_std[idx]:.6f}  "
              f"{power_opt[idx]:.6f}  {diff[idx]:+.2e}  {relative_diff[idx]:.2e}")
