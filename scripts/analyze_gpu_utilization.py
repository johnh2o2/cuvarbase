#!/usr/bin/env python3
"""
Analyze GPU utilization during BLS to understand batching opportunities.

Key questions:
1. Does a single lightcurve saturate the GPU?
2. How many SMs are we using?
3. Is there room for concurrent kernel execution?
"""

import numpy as np
import pycuda.driver as cuda
from cuvarbase import bls

# Get GPU info
cuda.init()
device = cuda.Device(0)

print("=" * 80)
print("GPU UTILIZATION ANALYSIS")
print("=" * 80)
print()
print("Device:", device.name())
print("Compute Capability:", device.compute_capability())
print("Multiprocessors:", device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT))
print("Max threads per multiprocessor:", device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR))
print("Max threads per block:", device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK))
print("Max blocks per multiprocessor:", device.get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR))
print()

# Calculate theoretical occupancy
n_sm = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
max_threads_per_sm = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
max_blocks_per_sm = device.get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR)

print("Theoretical Maximum Occupancy:")
print(f"  Total threads: {n_sm * max_threads_per_sm}")
print(f"  Total blocks: {n_sm * max_blocks_per_sm}")
print()

# Analyze different BLS configurations
configs = [
    ("Sparse ground-based", 100, 480224),
    ("Dense ground-based", 500, 734417),
    ("Space-based", 20000, 890539),
]

print("BLS Kernel Launch Configuration Analysis:")
print("-" * 80)

for desc, ndata, nfreq in configs:
    print(f"\n{desc} (ndata={ndata}, nfreq={nfreq}):")

    # Determine block size
    block_size = bls._choose_block_size(ndata)
    print(f"  Block size: {block_size} threads")

    # Grid size (number of blocks launched)
    # From eebls_gpu_fast: grid = min(nfreq, max_nblocks=5000)
    max_nblocks = 5000
    grid_size = min(nfreq, max_nblocks)
    print(f"  Grid size: {grid_size} blocks")

    # Total threads launched
    total_threads = grid_size * block_size
    print(f"  Total threads: {total_threads}")

    # Occupancy
    blocks_per_sm = grid_size / n_sm
    threads_per_sm = total_threads / n_sm

    occupancy_blocks = min(100, 100 * blocks_per_sm / max_blocks_per_sm)
    occupancy_threads = min(100, 100 * threads_per_sm / max_threads_per_sm)

    print(f"  Blocks per SM: {blocks_per_sm:.1f} / {max_blocks_per_sm} ({occupancy_blocks:.1f}% occupancy)")
    print(f"  Threads per SM: {threads_per_sm:.0f} / {max_threads_per_sm} ({occupancy_threads:.1f}% occupancy)")

    # Check if GPU is saturated
    if grid_size >= n_sm * max_blocks_per_sm:
        print(f"  ✓ GPU SATURATED - single lightcurve uses all SMs")
        print(f"  → No benefit from concurrent kernel execution")
    else:
        unused_blocks = n_sm * max_blocks_per_sm - grid_size
        print(f"  ⚠ GPU UNDERUTILIZED - {unused_blocks} blocks unused")
        print(f"  → Could run {unused_blocks / grid_size:.1f}x more kernels concurrently")

print()
print("=" * 80)
print("BATCHING OPPORTUNITIES")
print("=" * 80)
print()

# Analyze if we can batch multiple lightcurves
for desc, ndata, nfreq in configs:
    block_size = bls._choose_block_size(ndata)
    grid_size = min(nfreq, 5000)

    total_blocks_available = n_sm * max_blocks_per_sm

    if grid_size < total_blocks_available / 2:
        concurrent_lcs = int(total_blocks_available / grid_size)
        print(f"{desc}:")
        print(f"  Could run {concurrent_lcs} lightcurves concurrently")
        print(f"  → Use CUDA streams for concurrent execution")
        print(f"  → Expected speedup: {concurrent_lcs}x for batch processing")
    else:
        print(f"{desc}:")
        print(f"  Single LC saturates GPU")
        print(f"  → No benefit from concurrent streams")
    print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()
print("Based on GPU architecture, batching strategies:")
print()
print("1. Sparse ground-based (ndata~100):")
print("   - Small grid size → significant underutilization")
print("   - RECOMMENDATION: Use CUDA streams to run 10-20 LCs concurrently")
print("   - Expected: 10-20x throughput improvement")
print()
print("2. Dense ground-based (ndata~500):")
print("   - Moderate grid size → some underutilization")
print("   - RECOMMENDATION: Use streams to run 2-5 LCs concurrently")
print("   - Expected: 2-5x throughput improvement")
print()
print("3. Space-based (ndata~20k):")
print("   - Large grid size → GPU likely saturated")
print("   - RECOMMENDATION: Sequential processing is optimal")
print("   - Expected: No improvement from streams")
print("=" * 80)
