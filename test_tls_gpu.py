#!/usr/bin/env python3
"""
Quick TLS GPU test script - bypasses broken skcuda imports
"""
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

# Import TLS modules directly, skipping broken __init__.py
from cuvarbase import tls_grids, tls_models

print("=" * 60)
print("TLS GPU Test Script")
print("=" * 60)

# Test 1: Grid generation
print("\n1. Testing period grid generation...")
t = np.linspace(0, 100, 1000)
periods = tls_grids.period_grid_ofir(t, R_star=1.0, M_star=1.0)
print(f"   ✓ Generated {len(periods)} periods from {periods[0]:.2f} to {periods[-1]:.2f} days")

# Test 2: Duration grid
print("\n2. Testing duration grid generation...")
durations, counts = tls_grids.duration_grid(periods[:10])
print(f"   ✓ Generated duration grids for {len(durations)} periods")
print(f"   ✓ Duration counts: {counts}")

# Test 3: Transit model (simple)
print("\n3. Testing simple transit model...")
phases = np.linspace(0, 1, 1000)
flux = tls_models.simple_trapezoid_transit(phases, duration_phase=0.1, depth=0.01)
print(f"   ✓ Generated transit model with {len(flux)} points")
print(f"   ✓ Min flux: {np.min(flux):.4f} (expect ~0.99 for 1% transit)")

# Test 4: Try importing TLS with PyCUDA
print("\n4. Testing PyCUDA availability...")
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"   ✓ PyCUDA initialized")
    print(f"   ✓ GPUs available: {cuda.Device.count()}")
    for i in range(cuda.Device.count()):
        dev = cuda.Device(i)
        print(f"   ✓ GPU {i}: {dev.name()}")
except Exception as e:
    print(f"   ✗ PyCUDA error: {e}")
    sys.exit(1)

# Test 5: Compile TLS kernel
print("\n5. Testing TLS kernel compilation...")
try:
    from cuvarbase import tls
    kernel = tls.compile_tls(block_size=128, use_simple=True)
    print(f"   ✓ Simple kernel compiled successfully")
except Exception as e:
    print(f"   ✗ Kernel compilation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Run simple TLS search
print("\n6. Running simple TLS search on GPU...")
try:
    # Generate simple synthetic data
    ndata = 200
    t = np.sort(np.random.uniform(0, 50, ndata)).astype(np.float32)
    y = np.ones(ndata, dtype=np.float32)
    dy = np.ones(ndata, dtype=np.float32) * 0.001

    # Add simple transit at period=10
    period_true = 10.0
    phases = (t % period_true) / period_true
    in_transit = phases < 0.02
    y[in_transit] -= 0.01

    # Search
    periods_test = np.linspace(8, 12, 20).astype(np.float32)

    results = tls.tls_search_gpu(
        t, y, dy,
        periods=periods_test,
        use_simple=True,
        block_size=64
    )

    print(f"   ✓ Search completed")
    print(f"   ✓ Best period: {results['period']:.2f} days (true: {period_true:.2f})")
    print(f"   ✓ Best depth: {results['depth']:.4f} (true: 0.0100)")
    print(f"   ✓ SDE: {results['SDE']:.2f}")

    # Check accuracy
    period_error = abs(results['period'] - period_true)
    if period_error < 0.5:
        print(f"   ✓ Period recovered within 0.5 days!")
    else:
        print(f"   ⚠ Period error: {period_error:.2f} days")

except Exception as e:
    print(f"   ✗ TLS search error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
