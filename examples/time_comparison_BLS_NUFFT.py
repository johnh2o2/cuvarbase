import numpy as np, time
from cuvarbase.bls import eebls_transit_gpu
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

# Synthetic gappy light curve
rng = np.random.default_rng(0)
n = 500
t = np.sort(rng.uniform(0, 30, n))
true_period = 2.5
y = (np.sin(2*np.pi*t/true_period) + 0.1*rng.normal(size=n)).astype(np.float32)

# Grids
periods = np.linspace(1.5, 4.0, 300).astype(np.float32)
durations = np.array([0.2], dtype=np.float32)
freqs = 1.0 / periods

# Warm up CUDA
_ = np.dot(np.ones(1000), np.ones(1000))

# NUFFT LRT timing
lrt = NUFFTLRTAsyncProcess()
start = time.perf_counter()
snr = lrt.run(t, y, periods, durations=durations)
lrt_time = time.perf_counter() - start

# BLS timing (transit variant over same freq span)
start = time.perf_counter()
# eebls_transit_gpu returns (freqs, power, sols) in standard mode
freqs_out, power, sols = eebls_transit_gpu(
    t, y, np.ones_like(y) * 0.1,
    fmin=freqs.min(), fmax=freqs.max(),
    samples_per_peak=2, noverlap=2
)
bls_time = time.perf_counter() - start

print(f"NUFFT LRT: {lrt_time:.3f} s, shape={snr.shape}")
print(f"BLS      : {bls_time:.3f} s, freqs={len(freqs_out)}")