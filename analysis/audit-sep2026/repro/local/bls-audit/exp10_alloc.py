"""Exp 10: eebls_gpu memory allocation vs need; fused-kernel availability on the
adaptive/optimized entry points; per-call compilation on eebls_gpu/custom/hone."""
import sys, time
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data
import pycuda.driver as cuda
from cuvarbase.core import ensure_context
ensure_context()
import cuvarbase.bls as B
from cuvarbase.bls import (eebls_gpu, eebls_gpu_fast, eebls_gpu_fast_optimized, eebls_gpu_fast_adaptive,
                           eebls_transit, _get_cached_kernels, compile_bls, transit_autofreq)

t, y, dy = make_data(ndata=2000, baseline=60., freq=0.7, q=0.03, phi0=0.3, snr=12, seed=1)
freqs = np.linspace(0.5, 1.0, 10)

# --- allocation footprint of eebls_gpu for a 10-frequency call
orig_zeros = B.gpuarray.zeros
peak = [0]
def spy_zeros(*a, **k):
    arr = orig_zeros(*a, **k)
    peak[0] += arr.nbytes
    return arr
B.gpuarray.zeros = spy_zeros
free0 = cuda.mem_get_info()[0]
p, s = eebls_gpu(t, y, dy, freqs)
B.gpuarray.zeros = orig_zeros
print("eebls_gpu(10 freqs, defaults): free before=%.2f GB; total gpuarray.zeros allocated during call = %.2f GB; needed for 10 freqs ~ %.1f MB"
      % (free0 / 1e9, peak[0] / 1e9, 10 * B.count_tot_nbins(2, 100, 0.2) * 3 * 4 * 5 * 4 / 1e6))

# --- fused kernel availability
for label, fns in [("eebls_gpu_fast default", _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])),
                   ("adaptive/eebls_transit(use_optimized) dict", _get_cached_kernels(256, True, ['full_bls_no_sol_optimized']))]:
    print("%s: has fused kernel = %s" % (label, 'full_bls_no_sol_fused' in fns))

# spy on which kernel is launched
launched = []
class Spy:
    def __init__(self, f, name): self.f, self.name = f, name
    def prepared_call(self, *a, **k): launched.append(self.name); return self.f.prepared_call(*a, **k)
    def prepared_async_call(self, *a, **k): launched.append(self.name); return self.f.prepared_async_call(*a, **k)
orig = B._get_cached_kernels
def spy_get(*a, **k):
    d = orig(*a, **k)
    return {n: Spy(f, n) for n, f in d.items()}
B._get_cached_kernels = spy_get
fr, _ = transit_autofreq(t, qmin_fac=0.5, fmin=0.3, fmax=1.0)
for label, fn in [("eebls_gpu_fast", lambda: eebls_gpu_fast(t, y, dy, fr)),
                  ("eebls_gpu_fast_optimized", lambda: eebls_gpu_fast_optimized(t, y, dy, fr)),
                  ("eebls_gpu_fast_adaptive", lambda: eebls_gpu_fast_adaptive(t, y, dy, fr)),
                  ("eebls_transit(use_fast=True)", lambda: eebls_transit(t, y, dy, use_fast=True, fmin=0.3, fmax=1.0)),
                  ("eebls_transit(use_optimized=True)", lambda: eebls_transit(t, y, dy, use_optimized=True, fmin=0.3, fmax=1.0))]:
    launched.clear(); fn()
    print("%-36s launches: %s" % (label, sorted(set(launched))), "x%d" % len(launched))
B._get_cached_kernels = orig

# --- per-call compile on eebls_gpu
ncomp = [0]
orig_compile = B.compile_bls
def spy_compile(*a, **k):
    ncomp[0] += 1; return orig_compile(*a, **k)
B.compile_bls = spy_compile
for _ in range(3): eebls_gpu(t, y, dy, freqs)
print("eebls_gpu x3: compile_bls called %d times" % ncomp[0]); ncomp[0] = 0
for _ in range(3): eebls_transit(t, y, dy, fmin=0.3, fmax=1.0)
print("eebls_transit (default, ndata=%d) x3: compile_bls called %d times" % (len(t), ncomp[0])); ncomp[0] = 0
for _ in range(3): B.eebls_gpu_custom(t, y, dy, freqs, np.array([0.02, 0.05]), np.linspace(0, 1, 50))
print("eebls_gpu_custom x3: compile_bls called %d times" % ncomp[0])
B.compile_bls = orig_compile
ts = []
for _ in range(3):
    t0 = time.perf_counter(); eebls_gpu(t, y, dy, freqs); ts.append(time.perf_counter() - t0)
fns = compile_bls()
ts2 = []
for _ in range(3):
    t0 = time.perf_counter(); eebls_gpu(t, y, dy, freqs, functions=fns); ts2.append(time.perf_counter() - t0)
print("eebls_gpu(10 freqs) wall: default %.0f ms median; with precompiled functions= %.0f ms median" % (1e3 * np.median(ts), 1e3 * np.median(ts2)))
