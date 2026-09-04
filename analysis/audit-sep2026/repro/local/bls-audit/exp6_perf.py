"""Exp 6: relative timings (noisy, shared 4090; repeated, report medians)."""
import sys, time
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data
from cuvarbase.bls import (eebls_gpu, eebls_gpu_fast, eebls_gpu_fast_optimized,
                           eebls_gpu_fast_adaptive, eebls_transit, transit_autofreq, q_transit,
                           compile_bls, eebls_gpu_custom, BLSMemory, _get_cached_kernels)
import pycuda.driver as cuda


def timeit(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.median(ts), np.min(ts)


for ndata, baseline, label in [(1200, 200., "ground-based 1200 pts / 200 d"), (18000, 27., "TESS-like 18000 pts / 27 d")]:
    t, y, dy = make_data(ndata=ndata, baseline=baseline, freq=0.4, q=0.03, phi0=0.6, snr=12, seed=11)
    freqs, q0 = transit_autofreq(t, qmin_fac=0.5)
    qmins, qmaxes = 0.5 * q0, 2.0 * q0
    print("=== %s: Keplerian grid nfreqs=%d, T=%.0f" % (label, len(freqs), baseline))
    for name, fn in [
        ("eebls_gpu (eebls_transit default) dlogq=.2 nov=3", lambda: eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, dlogq=0.2, noverlap=3)),
        ("eebls_gpu_fast (fused nov=2)", lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)),
        ("eebls_gpu_fast nov=3 (multipass)", lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, noverlap=3)),
        ("eebls_gpu_fast nov=4 (fused)", lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, noverlap=4)),
        ("eebls_gpu_fast_optimized (fused nov=2)", lambda: eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)),
        ("eebls_gpu_fast_adaptive", lambda: eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)),
        ("eebls_transit(use_fast=True)", lambda: eebls_transit(t, y, dy, use_fast=True)),
        ("eebls_transit() default", lambda: eebls_transit(t, y, dy)),
    ]:
        fn()  # warm
        med, mn = timeit(fn, 4 if 'eebls_gpu (' in name or 'default' in name else 6)
        print("  %-50s median %8.1f ms  min %8.1f ms" % (name, 1e3 * med, 1e3 * mn))
    # compile cost vs run cost
    t0 = time.perf_counter(); compile_bls(); tc = time.perf_counter() - t0
    print("  compile_bls() uncached: %.0f ms  (eebls_gpu / eebls_gpu_custom / hone_solution call this per invocation)" % (1e3 * tc))
    # fast path with default qmin/qmax scalars vs Keplerian arrays
    med, mn = timeit(lambda: eebls_gpu_fast(t, y, dy, freqs), 4)
    print("  %-50s median %8.1f ms" % ("eebls_gpu_fast scalar qmin=1e-2 qmax=.5", 1e3 * med))
    # host-side overhead of fast path: memory setup vs kernel
    mem = BLSMemory.fromdata(t, y, dy, qmin=qmins, qmax=qmaxes, freqs=freqs, transfer=True)
    fns = _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])
    med, mn = timeit(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, memory=mem, transfer_to_device=False, functions=fns), 6)
    print("  %-50s median %8.1f ms" % ("fast, memory reuse, no H2D", 1e3 * med))
    med, mn = timeit(lambda: BLSMemory.fromdata(t, y, dy, qmin=qmins, qmax=qmaxes, freqs=freqs, transfer=True), 6)
    print("  %-50s median %8.1f ms" % ("BLSMemory.fromdata alone", 1e3 * med))
    med, mn = timeit(lambda: transit_autofreq(t, qmin_fac=0.5), 3)
    print("  %-50s median %8.1f ms (python while-loop)" % ("transit_autofreq alone", 1e3 * med))
