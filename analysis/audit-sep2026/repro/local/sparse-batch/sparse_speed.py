"""Sparse-vs-standard crossover timing (relative, repeated-min, noisy shared
4090) + compile cost + a decode-optimized sparse kernel variant built in
scratch (closed-form pair decode instead of the O(N) while loop)."""
import numpy as np, time, sys
import pycuda.driver as cuda, pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from cuvarbase.bls import (sparse_bls_gpu, compile_sparse_bls, eebls_gpu_fast,
                           eebls_transit, transit_autofreq, q_transit, _get_cached_kernels)
from cuvarbase.utils import find_kernel, _module_reader


def lc(N, seed=0, baseline=365.0):
    r = np.random.RandomState(seed)
    t = np.sort(baseline * r.rand(N))
    y = 1.0 - 0.02 * (((t * 0.7) % 1.0) < 0.03) + 0.01 * r.randn(N)
    dy = 0.01 * np.ones(N)
    return t, y, dy


def tmin(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), np.median(ts)


# ---- compile cost (sparse_bls_gpu compiles per call when kernel=None)
t0 = time.perf_counter(); k = compile_sparse_bls(block_size=64); tc = time.perf_counter() - t0
t0 = time.perf_counter(); k = compile_sparse_bls(block_size=64); tc2 = time.perf_counter() - t0
print("compile_sparse_bls: %.3fs first, %.3fs second call (no caching in sparse_bls_gpu)" % (tc, tc2))

t, y, dy = lc(200)
freqs = np.linspace(0.5, 1.5, 500)
a = tmin(lambda: sparse_bls_gpu(t, y, dy, freqs))
b = tmin(lambda: sparse_bls_gpu(t, y, dy, freqs, kernel=k))
print("sparse_bls_gpu N=200 nf=500: kernel=None %.3fs | precompiled kernel %.3fs  (min of 5)" % (a[0], b[0]))

# ---- decode-optimized variant: replace the two while-loop decodes
src = _module_reader(find_kernel('sparse_bls'), cpp_defs=dict(BLOCK_SIZE=64))
old_nw = """                unsigned int idx = p;
                unsigned int i = 0;
                while (idx >= (N - i)) {
                    idx -= (N - i);
                    i++;
                }
                unsigned int j = i + 1 + idx; // j in [i+1, N]
"""
new_nw = """                // closed-form row decode: row i starts at i*N - i*(i-1)/2
                float Nf = (float) N;
                unsigned int i = (unsigned int) floorf(((2.f * Nf + 1.f) - sqrtf((2.f * Nf + 1.f) * (2.f * Nf + 1.f) - 8.f * (float) p)) * 0.5f);
                if (i > 0 && i * N - (i * (i - 1)) / 2 > p) i--;
                while ((i + 1) * N - ((i + 1) * i) / 2 <= p) i++;
                unsigned int idx = p - (i * N - (i * (i - 1)) / 2);
                unsigned int j = i + 1 + idx; // j in [i+1, N]
"""
old_w = """                unsigned int idx = p - total_nonwrap;
                unsigned int i = 1;
                while (idx >= i) {
                    idx -= i;
                    i++;
                }
                unsigned int k = idx; // k in [0, i)
"""
new_w = """                unsigned int idx = p - total_nonwrap;
                // row i (i>=1) starts at i*(i-1)/2
                unsigned int i = (unsigned int) floorf((1.f + sqrtf(1.f + 8.f * (float) idx)) * 0.5f);
                if (i > 1 && (i * (i - 1)) / 2 > idx) i--;
                while (((i + 1) * i) / 2 <= idx) i++;
                unsigned int k = idx - (i * (i - 1)) / 2; // k in [0, i)
"""
assert old_nw in src and old_w in src
src2 = src.replace(old_nw, new_nw).replace(old_w, new_w)
mod2 = SourceModule(src2, options=['--use_fast_math'])
k2 = mod2.get_function('sparse_bls_kernel')

print("\n--- sparse GPU: original vs closed-form-decode variant, block_size sweep (min of 5, seconds)")
for N in (100, 300, 500):
    t, y, dy = lc(N)
    freqs = np.linspace(0.5, 1.5, 2000)
    p_ref, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=k)
    for bs in (64, 128, 256, 512):
        kb = compile_sparse_bls(block_size=bs)
        kb2 = SourceModule(_module_reader(find_kernel('sparse_bls'), cpp_defs=dict(BLOCK_SIZE=bs)).replace(old_nw, new_nw).replace(old_w, new_w), options=['--use_fast_math']).get_function('sparse_bls_kernel')
        p1, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kb, block_size=bs)
        p2, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kb2, block_size=bs)
        t1 = tmin(lambda: sparse_bls_gpu(t, y, dy, freqs, kernel=kb, block_size=bs))
        t2 = tmin(lambda: sparse_bls_gpu(t, y, dy, freqs, kernel=kb2, block_size=bs))
        print("  N=%d bs=%3d orig %.4fs  decode-opt %.4fs  ratio %.2fx | bitwise-equal to bs=64 orig: %s / %s" % (
            N, bs, t1[0], t2[0], t1[0] / t2[0], np.array_equal(p1, p_ref), np.array_equal(p2, p_ref)))

print("\n--- crossover: sparse GPU (bs=64 default) vs eebls_gpu_fast on eebls_transit's own Keplerian grid (baseline 365 d)")
fns = _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])
for N in (50, 100, 200, 300, 500, 750, 1000):
    t, y, dy = lc(N)
    freqs, qvals = transit_autofreq(t, fmin=0.05, fmax=2.0, qmin_fac=0.5)
    qmins, qmaxes = qvals * 0.5, qvals * 2.0
    # eebls_transit(...) default path: sparse for N<500 (compiles per call), standard fast otherwise
    ts = tmin(lambda: sparse_bls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, kernel=k), reps=3)
    tf = tmin(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, noverlap=2, functions=fns), reps=3)
    tt = tmin(lambda: eebls_transit(t, y, dy, freqs=freqs), reps=3)
    print("  N=%5d nfreq=%6d  sparse(precompiled) %.3fs  fast(noverlap=2) %.3fs  ratio sparse/fast %.1fx | eebls_transit default %.3fs" % (
        N, len(freqs), ts[0], tf[0], ts[0] / tf[0], tt[0]))
