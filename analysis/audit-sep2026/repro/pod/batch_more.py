import numpy as np, warnings, time
from cuvarbase.bls import eebls_gpu_batch, eebls_gpu_fast, transit_autofreq, q_transit, eebls_gpu, compile_bls, _get_cached_kernels
warnings.simplefilter('ignore')
from batch_exp import lc, stat

print("=== per-frequency Keplerian q bounds + conventions: batch vs fast")
t, y, dy = lc(800, 9)
freqs, qvals = transit_autofreq(t, fmin=0.2, fmax=3.0, qmin_fac=0.5)
qmins, qmaxes = qvals * 0.5, qvals * 2.0
for conv in ('chi2ratio', 'snr', 'loglik'):
    pf = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, noverlap=2, convention=conv)
    pb = eebls_gpu_batch([(t, y, dy), lc(300, 10)], freqs, qmin=qmins, qmax=qmaxes, noverlap=2, convention=conv)[0]
    print("  nfreq=%d conv=%s  %s" % (len(freqs), conv, stat(pb, pf)))

print("\n=== eebls_transit standard-path defaults: eebls_gpu compiles per call?")
t, y, dy = lc(750, 11)
freqs = np.linspace(0.2, 3.0, 20000)
def tmin(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)
fns = compile_bls()
a = tmin(lambda: eebls_gpu(t, y, dy, freqs))
b = tmin(lambda: eebls_gpu(t, y, dy, freqs, functions=fns))
c = tmin(lambda: eebls_gpu_fast(t, y, dy, freqs))
print("  eebls_gpu(functions=None) %.3fs | eebls_gpu(precompiled) %.3fs | eebls_gpu_fast (cached) %.3fs   (N=750, nfreq=20000, min of 3, shared noisy 4090)" % (a, b, c))

print("\n=== batch grid.y limit: max_batch_lcs > 65535?")
import pycuda.driver as cuda
dev = cuda.Context.get_device()
print("  MAX_GRID_DIM_Y =", dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y))
