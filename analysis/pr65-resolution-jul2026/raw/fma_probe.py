"""Determine empirically whether the compiled BLS kernels FMA-contract
mod1(t * f): find (t, f) values where frac(rounded product) and
f32(frac(exact product)) fall in DIFFERENT bins, then check which bin
the kernel assigns.

Run on the pod from /workspace/cuvarbase.
"""
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

from cuvarbase.bls import compile_bls, _default_block_size
from cuvarbase.core import ensure_context

f32 = np.float32
ensure_context()

# ---- find discriminating test points ----
rng = np.random.RandomState(7)
nb = 22          # single bin level: nbins0 = nbinsf = 22, dlogq=-1
f = f32(0.9999707539228717)

cands_t, cands_plain, cands_fma = [], [], []
t_try = rng.uniform(300., 370., size=2000000).astype(f32)
u = t_try * f                                  # rounded product
phi_plain = u - np.floor(u)
exact = t_try.astype(np.float64) * np.float64(f)
phi_fma = (exact - np.floor(u.astype(np.float64))).astype(f32)
b_plain = np.floor(f32(nb) * phi_plain).astype(int) % nb
b_fma = np.floor(f32(nb) * phi_fma).astype(int) % nb
sel = b_plain != b_fma
print("discriminating points found:", sel.sum())
t_sel = t_try[sel][:64]
bp_sel = b_plain[sel][:64]
bf_sel = b_fma[sel][:64]

# ---- run the kernel on these points ----
functions = compile_bls(function_names=['bin_and_phase_fold_bst_multifreq'])
bin_func = functions['bin_and_phase_fold_bst_multifreq']

ndata = len(t_sel)
t_g = gpuarray.to_gpu(t_sel)
yw_g = gpuarray.to_gpu(np.ones(ndata, dtype=f32))
w_g = gpuarray.to_gpu(np.ones(ndata, dtype=f32))
freqs_g = gpuarray.to_gpu(np.array([f], dtype=f32))

noverlap = 1
nbins_tot = nb
yw_bin = gpuarray.zeros(nbins_tot * noverlap, dtype=f32)
w_bin = gpuarray.zeros(nbins_tot * noverlap, dtype=f32)

block = (_default_block_size, 1, 1)
grid = (int(np.ceil(ndata / float(_default_block_size))), 1)
bin_func.prepared_call(grid, block,
                       t_g.ptr, yw_g.ptr, w_g.ptr,
                       yw_bin.ptr, w_bin.ptr, freqs_g.ptr,
                       np.uint32(ndata), np.uint32(1),
                       np.uint32(nb), np.uint32(nb),
                       np.uint32(0), np.uint32(noverlap),
                       np.float32(-1.0), np.uint32(nbins_tot))

w_out = w_bin.get()

# each test point: does the kernel's occupied bin match plain or fma?
# run points ONE AT A TIME to attribute bins unambiguously
match_plain = match_fma = other = 0
for i in range(ndata):
    yw_bin.fill(f32(0)); w_bin.fill(f32(0))
    t1 = gpuarray.to_gpu(t_sel[i:i+1])
    bin_func.prepared_call((1, 1), block,
                           t1.ptr, yw_g.ptr, w_g.ptr,
                           yw_bin.ptr, w_bin.ptr, freqs_g.ptr,
                           np.uint32(1), np.uint32(1),
                           np.uint32(nb), np.uint32(nb),
                           np.uint32(0), np.uint32(noverlap),
                           np.float32(-1.0), np.uint32(nbins_tot))
    b_gpu = int(np.argmax(w_bin.get()))
    if b_gpu == bp_sel[i]:
        match_plain += 1
    elif b_gpu == bf_sel[i]:
        match_fma += 1
    else:
        other += 1
        print(f"  UNEXPECTED: t={t_sel[i]!r} gpu_bin={b_gpu} "
              f"plain={bp_sel[i]} fma={bf_sel[i]}")

print(f"kernel matches PLAIN (rounded-product frac): {match_plain}")
print(f"kernel matches FMA   (exact-product frac):   {match_fma}")
print(f"other: {other}")
