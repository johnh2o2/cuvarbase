#!/usr/bin/env python3
"""Compare GPU and CPU TLS depth calculations"""
import numpy as np
from cuvarbase import tls as gpu_tls
from transitleastsquares import transitleastsquares as cpu_tls

# Generate test data
np.random.seed(42)
ndata = 500
t = np.sort(np.random.uniform(0, 50, ndata))
y = np.ones(ndata, dtype=np.float32)

# Add transit
period_true = 10.0
depth_true = 0.01  # Fractional dip
phase = (t % period_true) / period_true
in_transit = (phase < 0.01) | (phase > 0.99)
y[in_transit] -= depth_true
y += np.random.normal(0, 0.001, ndata).astype(np.float32)
dy = np.ones(ndata, dtype=np.float32) * 0.001

print(f"Test data:")
print(f"  N = {ndata}")
print(f"  Period = {period_true:.1f} days")
print(f"  Depth (fractional dip) = {depth_true:.3f}")
print(f"  Points in transit: {np.sum(in_transit)}")
print(f"  Measured depth: {np.mean(y[~in_transit]) - np.mean(y[in_transit]):.6f}")

# GPU TLS
print(f"\n--- GPU TLS ---")
gpu_result = gpu_tls.tls_search_gpu(
    t.astype(np.float32), y, dy,
    period_min=9.0,
    period_max=11.0
)

print(f"Period: {gpu_result['period']:.4f} (error: {abs(gpu_result['period'] - period_true)/period_true*100:.2f}%)")
print(f"Depth: {gpu_result['depth']:.6f}")
print(f"Duration: {gpu_result['duration']:.4f} days")
print(f"T0: {gpu_result['T0']:.4f}")

# CPU TLS
print(f"\n--- CPU TLS ---")
model = cpu_tls(t, y, dy)
cpu_result = model.power(
    period_min=9.0,
    period_max=11.0,
    n_transits_min=2
)

print(f"Period: {cpu_result.period:.4f} (error: {abs(cpu_result.period - period_true)/period_true*100:.2f}%)")
print(f"Depth (flux ratio): {cpu_result.depth:.6f}")
print(f"Depth (fractional dip): {1 - cpu_result.depth:.6f}")
print(f"Duration: {cpu_result.duration:.4f} days")
print(f"T0: {cpu_result.T0:.4f}")

# Compare
print(f"\n--- Comparison ---")
print(f"Period agreement: {abs(gpu_result['period'] - cpu_result.period):.4f} days")
print(f"Duration agreement: {abs(gpu_result['duration'] - cpu_result.duration):.4f} days")

# Check depth conventions
gpu_depth_frac = gpu_result['depth']  # GPU reports fractional dip
cpu_depth_frac = 1 - cpu_result.depth  # CPU reports flux ratio

print(f"\nDepth (fractional dip convention):")
print(f"  True: {depth_true:.6f}")
print(f"  GPU:  {gpu_depth_frac:.6f} (error: {abs(gpu_depth_frac - depth_true)/depth_true*100:.1f}%)")
print(f"  CPU:  {cpu_depth_frac:.6f} (error: {abs(cpu_depth_frac - depth_true)/depth_true*100:.1f}%)")
