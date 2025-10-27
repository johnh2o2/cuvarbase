#!/usr/bin/env python3
"""Test TLS GPU with realistic period grids"""
import numpy as np
from cuvarbase import tls, tls_grids

# Generate test data
ndata = 500
np.random.seed(42)
t = np.sort(np.random.uniform(0, 50, ndata)).astype(np.float32)
y = np.ones(ndata, dtype=np.float32)

# Add transit at period=10
period_true = 10.0
phase = (t % period_true) / period_true
in_transit = (phase < 0.01) | (phase > 0.99)
y[in_transit] -= 0.01
y += np.random.normal(0, 0.001, ndata).astype(np.float32)
dy = np.ones(ndata, dtype=np.float32) * 0.001

print(f"Data: {len(t)} points, transit at {period_true:.1f} days with depth 0.01")

# Generate realistic period grid
periods = tls_grids.period_grid_ofir(
    t, R_star=1.0, M_star=1.0,
    period_min=5.0,
    period_max=20.0
).astype(np.float32)

print(f"Period grid: {len(periods)} periods from {periods[0]:.2f} to {periods[-1]:.2f}")

# Run TLS
print("Running TLS...")
results = tls.tls_search_gpu(t, y, dy, periods=periods)

print(f"\nResults:")
print(f"  Period: {results['period']:.4f} (true: {period_true:.1f})")
print(f"  Depth: {results['depth']:.6f} (true: 0.010000)")
print(f"  Duration: {results['duration']:.4f} days")
print(f"  SDE: {results['SDE']:.2f}")

period_error = abs(results['period'] - period_true)
depth_error = abs(results['depth'] - 0.01)

print(f"\nAccuracy:")
print(f"  Period error: {period_error:.4f} days ({period_error/period_true*100:.1f}%)")
print(f"  Depth error: {depth_error:.6f} ({depth_error/0.01*100:.1f}%)")

if period_error < 0.5 and depth_error < 0.002:
    print("\n✓ Signal recovered successfully!")
    exit(0)
else:
    print("\n✗ Signal recovery failed")
    exit(1)
