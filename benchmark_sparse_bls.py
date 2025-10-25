"""Benchmark sparse BLS CPU vs GPU performance"""
import numpy as np
import time
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu

def data(ndata=100, freq=1.0, q=0.05, phi0=0.3, seed=42):
    """Generate test data"""
    np.random.seed(seed)
    sigma = 0.1
    snr = 10
    baseline = 365.
    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    t = baseline * np.sort(np.random.rand(ndata))

    # Transit model
    phi = t * freq - phi0
    phi -= np.floor(phi)
    y = np.zeros(ndata)
    y[np.abs(phi) < q] -= delta
    y += sigma * np.random.randn(ndata)
    dy = sigma * np.ones(ndata)

    return t.astype(np.float32), y.astype(np.float32), dy.astype(np.float32)

print("Sparse BLS Performance Comparison")
print("=" * 70)
print(f"{'ndata':<10} {'nfreqs':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
print("=" * 70)

for ndata in [50, 100, 200, 500]:
    for nfreqs in [10, 50, 100]:
        t, y, dy = data(ndata=ndata)
        freqs = np.linspace(0.5, 2.0, nfreqs).astype(np.float32)

        # Warm up GPU
        _ = sparse_bls_gpu(t, y, dy, freqs[:5])

        # Benchmark CPU
        t_start = time.time()
        power_cpu, _ = sparse_bls_cpu(t, y, dy, freqs)
        t_cpu = (time.time() - t_start) * 1000  # ms

        # Benchmark GPU
        t_start = time.time()
        power_gpu, _ = sparse_bls_gpu(t, y, dy, freqs)
        t_gpu = (time.time() - t_start) * 1000  # ms

        speedup = t_cpu / t_gpu
        print(f"{ndata:<10} {nfreqs:<10} {t_cpu:<15.2f} {t_gpu:<15.2f} {speedup:<10.2f}x")

print("=" * 70)
