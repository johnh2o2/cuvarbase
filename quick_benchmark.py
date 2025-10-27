#!/usr/bin/env python3
"""Quick GPU vs CPU benchmark"""
import numpy as np
import time
from cuvarbase import tls as gpu_tls, tls_grids
from transitleastsquares import transitleastsquares as cpu_tls

print("="*70)
print("Quick GPU vs CPU TLS Benchmark")
print("="*70)

# Test parameters
ndata_values = [500, 1000, 2000]
baseline = 50.0
period_true = 10.0
depth_true = 0.01

for ndata in ndata_values:
    print(f"\n--- N = {ndata} points ---")

    # Generate data
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, baseline, ndata)).astype(np.float32)
    y = np.ones(ndata, dtype=np.float32)
    phase = (t % period_true) / period_true
    in_transit = (phase < 0.01) | (phase > 0.99)
    y[in_transit] -= depth_true
    y += np.random.normal(0, 0.001, ndata).astype(np.float32)
    dy = np.ones(ndata, dtype=np.float32) * 0.001

    # GPU TLS
    t0_gpu = time.time()
    gpu_result = gpu_tls.tls_search_gpu(
        t, y, dy,
        period_min=5.0,
        period_max=20.0,
        use_simple=True
    )
    t1_gpu = time.time()
    gpu_time = t1_gpu - t0_gpu

    # CPU TLS
    model = cpu_tls(t, y, dy)
    t0_cpu = time.time()
    cpu_result = model.power(
        period_min=5.0,
        period_max=20.0,
        n_transits_min=2
    )
    t1_cpu = time.time()
    cpu_time = t1_cpu - t0_cpu

    # Compare
    speedup = cpu_time / gpu_time

    gpu_depth_frac = gpu_result['depth']
    cpu_depth_frac = 1 - cpu_result.depth

    print(f"GPU: {gpu_time:6.3f}s, period={gpu_result['period']:7.4f}, depth={gpu_depth_frac:.6f}")
    print(f"CPU: {cpu_time:6.3f}s, period={cpu_result.period:7.4f}, depth={cpu_depth_frac:.6f}")
    print(f"Speedup: {speedup:.1f}x")

    # Accuracy
    gpu_period_err = abs(gpu_result['period'] - period_true) / period_true * 100
    cpu_period_err = abs(cpu_result.period - period_true) / period_true * 100
    gpu_depth_err = abs(gpu_depth_frac - depth_true) / depth_true * 100
    cpu_depth_err = abs(cpu_depth_frac - depth_true) / depth_true * 100

    print(f"Period error: GPU={gpu_period_err:.2f}%, CPU={cpu_period_err:.2f}%")
    print(f"Depth error:  GPU={gpu_depth_err:.1f}%, CPU={cpu_depth_err:.1f}%")

print("\n" + "="*70)
print("Benchmark complete!")
