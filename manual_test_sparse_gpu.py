"""Manual test for sparse BLS GPU without pytest"""
import numpy as np
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu

def data(snr=10, q=0.01, phi0=0.2, freq=1.0, baseline=365., ndata=100, seed=42):
    """Generate test data"""
    np.random.seed(seed)
    sigma = 0.1
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

# Run tests
print("Testing GPU sparse BLS implementation")
print("=" * 60)

for ndata in [50, 100, 200]:
    for ignore_neg in [True, False]:
        t, y, dy = data(ndata=ndata, freq=1.0, q=0.05, phi0=0.3)
        df = 0.05 / (10 * (max(t) - min(t)))
        freqs = np.linspace(0.95, 1.05, 11).astype(np.float32)

        power_cpu, sols_cpu = sparse_bls_cpu(t, y, dy, freqs, ignore_negative_delta_sols=ignore_neg)
        power_gpu, sols_gpu = sparse_bls_gpu(t, y, dy, freqs, ignore_negative_delta_sols=ignore_neg)

        max_diff = np.abs(power_cpu - power_gpu).max()

        print(f"ndata={ndata}, ignore_neg={ignore_neg}: max_diff={max_diff:.2e}", end="")
        if max_diff < 1e-4:
            print(" ✓ PASS")
        else:
            print(" ✗ FAIL")
            print(f"  CPU powers: {power_cpu}")
            print(f"  GPU powers: {power_gpu}")

print("\nAll tests completed!")
