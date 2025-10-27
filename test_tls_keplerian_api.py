#!/usr/bin/env python3
"""Test TLS Keplerian API end-to-end"""
import numpy as np
from cuvarbase import tls

print("="*70)
print("TLS Keplerian API End-to-End Test")
print("="*70)

# Generate synthetic data with transit
np.random.seed(42)
ndata = 500
baseline = 50.0
period_true = 10.0
depth_true = 0.01

t = np.sort(np.random.uniform(0, baseline, ndata)).astype(np.float32)
y = np.ones(ndata, dtype=np.float32)

# Add transit
phase = (t % period_true) / period_true
in_transit = (phase < 0.01) | (phase > 0.99)
y[in_transit] -= depth_true
y += np.random.normal(0, 0.001, ndata).astype(np.float32)
dy = np.ones(ndata, dtype=np.float32) * 0.001

print(f"\nData: {ndata} points, transit at {period_true:.1f} days with depth {depth_true:.3f}")

# Test 1: tls_transit() with Keplerian constraints
print("\n" + "="*70)
print("Test 1: tls_transit() - Keplerian-Aware Search")
print("="*70)

results = tls.tls_transit(
    t, y, dy,
    R_star=1.0,
    M_star=1.0,
    R_planet=1.0,       # Earth-size planet
    qmin_fac=0.5,       # Search 0.5x to 2.0x Keplerian duration
    qmax_fac=2.0,
    n_durations=15,
    period_min=5.0,
    period_max=20.0
)

print(f"\nResults:")
print(f"  Period: {results['period']:.4f} days (true: {period_true:.1f})")
print(f"  Depth: {results['depth']:.6f} (true: {depth_true:.6f})")
print(f"  Duration: {results['duration']:.4f} days")
print(f"  T0: {results['T0']:.4f} days")
print(f"  SDE: {results['SDE']:.2f}")

# Check accuracy
period_error = abs(results['period'] - period_true)
depth_error = abs(results['depth'] - depth_true)

print(f"\nAccuracy:")
print(f"  Period error: {period_error:.4f} days ({period_error/period_true*100:.2f}%)")
print(f"  Depth error: {depth_error:.6f} ({depth_error/depth_true*100:.1f}%)")

# Test 2: Standard tls_search_gpu() for comparison
print("\n" + "="*70)
print("Test 2: tls_search_gpu() - Standard Search (Fixed Duration Range)")
print("="*70)

results_std = tls.tls_search_gpu(
    t, y, dy,
    period_min=5.0,
    period_max=20.0,
    R_star=1.0,
    M_star=1.0
)

print(f"\nResults:")
print(f"  Period: {results_std['period']:.4f} days (true: {period_true:.1f})")
print(f"  Depth: {results_std['depth']:.6f} (true: {depth_true:.6f})")
print(f"  Duration: {results_std['duration']:.4f} days")
print(f"  SDE: {results_std['SDE']:.2f}")

# Compare
print("\n" + "="*70)
print("Comparison: Keplerian vs Standard")
print("="*70)

print(f"\nPeriod Recovery:")
print(f"  Keplerian: {results['period']:.4f} days (error: {period_error/period_true*100:.2f}%)")
print(f"  Standard:  {results_std['period']:.4f} days (error: {abs(results_std['period']-period_true)/period_true*100:.2f}%)")

print(f"\nDepth Recovery:")
print(f"  Keplerian: {results['depth']:.6f} (error: {depth_error/depth_true*100:.1f}%)")
print(f"  Standard:  {results_std['depth']:.6f} (error: {abs(results_std['depth']-depth_true)/depth_true*100:.1f}%)")

# Verdict
print("\n" + "="*70)
success = (period_error < 0.5 and depth_error < 0.002)
if success:
    print("✓ Test PASSED: Keplerian API working correctly!")
    print("✓ Period recovered within 5% of true value")
    print("✓ Depth recovered within 20% of true value")
    exit(0)
else:
    print("✗ Test FAILED: Signal recovery outside acceptable tolerance")
    exit(1)
