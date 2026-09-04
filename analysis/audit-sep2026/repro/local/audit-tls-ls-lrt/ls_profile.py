"""Lomb-Scargle stage profile at survey scale (GPU shared -> report medians of 5)."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from scipy.fft import next_fast_len
from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev, get_k0, check_k0
from cuvarbase.memory import LombScargleMemory
from cuvarbase import _cufft as cufft

rng = np.random.RandomState(3)
N = 1000; T = 1095.0
t = np.sort(rng.rand(N) * T); dy = 0.02 * (1 + rng.rand(N)); y = 15 + 0.1 * np.sin(2 * np.pi * t / 0.37) + dy * rng.randn(N)
nf = 365000; fmax = 100.0; df = fmax / nf; freqs = df * (1 + np.arange(nf))
k0 = get_k0(freqs); check_k0(freqs, k0)
data = [(t, y, dy)]
med = lambda xs: 1e3 * float(np.median(xs))

def bench(fn, reps=5):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return med(xs)

print("N=%d nf=%d k0=%d" % (N, nf, k0))
proc = LombScargleAsyncProcess()
proc.batched_run_const_nfreq(data, freqs=freqs)  # warm (compile)
print("batched_run_const_nfreq, 1 LC/call (alloc every call): %.2f ms" % bench(lambda: proc.batched_run_const_nfreq(data, freqs=freqs)))
print("batched_run_const_nfreq, 1 LC/call, only_return_best_freqs (FAP): %.2f ms" % bench(lambda: proc.batched_run_const_nfreq(data, freqs=freqs, only_return_best_freqs=True)))
d16 = data * 16
print("batched_run_const_nfreq, 16 LC/call: %.2f ms/LC" % (bench(lambda: proc.batched_run_const_nfreq(d16, freqs=freqs), reps=3) / 16))
# allocation cost
m = proc.nfft_proc.get_m(nf); sigma = proc.nfft_proc.sigma
def alloc():
    mem = LombScargleMemory(sigma, proc.streams[0], m, k0=k0, buffered_transfer=True, n0_buffer=N, use_double=False, nharmonics=1, use_fft=True)
    mem.allocate(nf=nf); return mem
print("LombScargleMemory alloc (pinned host + device + 2 cuFFT plans): %.2f ms" % bench(alloc))
n_yw = int(sigma * (nf + k0 - k0)); n_w = int(sigma * (2 * (nf + k0) - k0))
print("cuFFT plan creation: n_yw=%d %.2f ms ; n_w=%d %.2f ms" % (n_yw, bench(lambda: cufft.Plan(n_yw, np.complex64, np.complex64)), n_w, bench(lambda: cufft.Plan(n_w, np.complex64, np.complex64))))
mem = alloc()
print("run() with preallocated memory + finish: %.2f ms" % bench(lambda: (proc.run(data, memory=[mem], freqs=[freqs]), proc.finish())))

# ---- stage breakdown mirroring lomb_scargle_async ----
from cuvarbase.cunfft import nfft_adjoint_async
lomb, lomb_dirsum = proc.function_tuple
nfuncs = proc.nfft_proc.function_tuple
stream = mem.stream
def stages():
    S = {}
    def tick(name, t0):
        stream.synchronize(); S[name] = S.get(name, 0) + time.perf_counter() - t0
    t0 = time.perf_counter(); mem.set_gpu_arrays_to_zero(); tick('zero_arrays', t0)
    t0 = time.perf_counter(); mem.setdata(t=t, y=y, dy=dy); tick('setdata_cpu', t0)
    t0 = time.perf_counter(); mem.transfer_data_to_gpu(); tick('h2d', t0)
    spp = 1. / ((mem.tmax - mem.tmin) * df)
    kw = dict(transfer_to_host=False, transfer_to_device=False, minimum_frequency=freqs[0], samples_per_peak=spp, block_size=proc.block_size)
    # yw NFFT split: just_return_gridded_data path is not usable async; time whole adjoint per grid
    t0 = time.perf_counter(); nfft_adjoint_async(mem.nfft_mem_yw, nfuncs, **kw); tick('nfft_yw(grid+fft+norm)', t0)
    t0 = time.perf_counter(); nfft_adjoint_async(mem.nfft_mem_w, nfuncs, **kw); tick('nfft_w(grid+fft+norm)', t0)
    block = (proc.block_size, 1, 1); grid = (int(np.ceil(nf / float(proc.block_size))), 1)
    t0 = time.perf_counter()
    lomb.prepared_async_call(grid, block, stream, mem.nfft_mem_w.ghat_g.ptr, mem.nfft_mem_yw.ghat_g.ptr, mem.lsp_g.ptr, mem.reg_g.ptr, np.int32(nf), mem.real_type(mem.yy), mem.real_type(mem.ybar), np.int32(mem.k0), np.int32(mem.mode))
    tick('lomb_kernel', t0)
    t0 = time.perf_counter(); mem.transfer_lsp_to_cpu(); tick('d2h', t0)
    return S
acc = {}
for i in range(5):
    S = stages()
    for k, v in S.items(): acc.setdefault(k, []).append(v)
tot = sum(med(v) for v in acc.values())
for k, v in acc.items():
    print("  %-24s %7.3f ms  (%4.1f%%)" % (k, med(v), 100 * med(v) / tot))
print("  %-24s %7.3f ms" % ('sum', tot))
# FFT-only timing on the two grids and on padded sizes
for n in (n_yw, next_fast_len(n_yw), 1 << (n_yw - 1).bit_length(), n_w, next_fast_len(n_w), 1 << (n_w - 1).bit_length()):
    g = gpuarray.zeros(n, np.complex64); plan = cufft.Plan(n, np.complex64, np.complex64)
    def f():
        cufft.ifft(g, g, plan); cuda.Context.synchronize()
    print("  cuFFT C2C n=%8d: %.3f ms" % (n, bench(f, reps=7)))
# gridding sub-stages of the w grid (largest) and psi
mw = mem.nfft_mem_w; myw = mem.nfft_mem_yw
precompute_psi, fast_gaussian_grid, slow_gaussian_grid, nfft_shift, normalize = nfuncs
bs = proc.block_size; gs = lambda n: (int(np.ceil(n / bs)), 1)
spp = 1. / ((mem.tmax - mem.tmin) * df)
def psi():
    precompute_psi.prepared_async_call(gs(N + 2 * m + 1), (bs, 1, 1), stream, myw.t_g.ptr, myw.q1.ptr, myw.q2.ptr, myw.q3.ptr, np.int32(N), np.int32(myw.n), np.int32(m), np.float32(myw.b), np.float32(mem.tmin), np.float32(mem.tmax), np.float32(spp)); stream.synchronize()
def gridw():
    fast_gaussian_grid.prepared_async_call(gs(N), (bs, 1, 1), stream, mw.t_g.ptr, mw.y_g.ptr, mw.ghat_g.ptr, mw.q1.ptr, mw.q2.ptr, mw.q3.ptr, np.int32(N), np.int32(mw.n), np.int32(1), np.int32(m), np.float32(mem.tmin), np.float32(mem.tmax), np.float32(spp)); stream.synchronize()
def normw():
    normalize.prepared_async_call(gs(mw.nf), (bs, 1, 1), stream, mw.ghat_g.ptr, mw.ghat_g.ptr, np.int32(mw.n), np.int32(mw.nf), np.int32(1), np.float32(mw.b), np.float32(mem.tmin), np.float32(mem.tmax), np.float32(spp), np.float32(freqs[0])); stream.synchronize()
def shiftw():
    nfft_shift.prepared_async_call(gs(mw.n), (bs, 1, 1), stream, mw.ghat_g.ptr, mw.ghat_g.ptr, np.int32(mw.n), np.int32(1), np.float32(mem.tmin), np.float32(mem.tmax), np.float32(spp), np.float32(freqs[0])); stream.synchronize()
def zerow():
    mw.ghat_g.fill(np.complex64(0), stream=stream); stream.synchronize()
print("  w-grid sub-stages: psi %.3f ms | grid(m=%d, N=%d) %.3f ms | shift %.3f ms | normalize %.3f ms | zero-fill %.3f ms" % (bench(psi), m, N, bench(gridw), bench(shiftw), bench(normw), bench(zerow)))
# FAP
res = proc.run(data, memory=[mem], freqs=[freqs]); proc.finish(); p = np.array(res[0][1]); bi = int(np.argmax(p))
print("fap_baluev over full array (nf=%d): %.2f ms ; scalar at best: %.3f ms ; argmax+mask copies: %.2f ms"
      % (nf, bench(lambda: fap_baluev(t, dy, p, fmax)), bench(lambda: fap_baluev(t, dy, p[bi], fmax)), bench(lambda: (np.argmax(p[np.ones(nf, bool)]), freqs[np.ones(nf, bool)]))))
# m sweep (run-time) at fixed sigma=4 ; accuracy handled by ls_psi_bug-style patch elsewhere
for mm in (8, 6, 5, 4):
    pr = LombScargleAsyncProcess(m=mm); pr.batched_run_const_nfreq(data, freqs=freqs)
    mem2 = LombScargleMemory(sigma, pr.streams[0], mm, k0=k0, buffered_transfer=True, n0_buffer=N, use_double=False, nharmonics=1, use_fft=True); mem2.allocate(nf=nf)
    print("  m=%d: run() w/ prealloc memory: %.2f ms" % (mm, bench(lambda: (pr.run(data, memory=[mem2], freqs=[freqs]), pr.finish()))))
# larger N gridding cost check
N2 = 20000; t2 = np.sort(rng.rand(N2) * T); dy2 = 0.02 * np.ones(N2); y2 = 15 + dy2 * rng.randn(N2)
mem3 = LombScargleMemory(sigma, proc.streams[0], m, k0=k0, buffered_transfer=True, n0_buffer=N2, use_double=False, nharmonics=1, use_fft=True); mem3.allocate(nf=nf)
print("N=%d nf=%d run() w/ prealloc memory: %.2f ms" % (N2, nf, bench(lambda: (proc.run([(t2, y2, dy2)], memory=[mem3], freqs=[freqs]), proc.finish()))))
