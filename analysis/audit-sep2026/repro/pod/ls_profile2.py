"""LS: (1) where does N=20000 time go; (2) padded-FFT (next_fast_len) parity + speed; (3) psi-fix + m sweep accuracy at survey scale."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
import pycuda.driver as cuda
from scipy.fft import next_fast_len
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase.memory import LombScargleMemory, NFFTMemory
from cuvarbase.cunfft import nfft_adjoint_async
from cuvarbase import _cufft as cufft

rng = np.random.RandomState(3)
T = 1095.0; nf = 365000; fmax = 100.0; df = fmax / nf; freqs = df * (1 + np.arange(nf)); k0 = 1
med = lambda xs: 1e3 * float(np.median(xs))
def bench(fn, reps=5):
    xs = []
    for i in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return med(xs)
proc = LombScargleAsyncProcess(); m = proc.nfft_proc.get_m(nf); sigma = proc.nfft_proc.sigma
def mk(N):
    t = np.sort(rng.rand(N) * T); dy = 0.02 * (1 + rng.rand(N)); y = 15 + 0.1 * np.sin(2 * np.pi * t / 0.37) + dy * rng.randn(N); return t, y, dy
def alloc(N):
    mem = LombScargleMemory(sigma, proc.streams[0], m, k0=k0, buffered_transfer=True, n0_buffer=N, use_double=False, nharmonics=1, use_fft=True); mem.allocate(nf=nf); return mem
proc.batched_run_const_nfreq([mk(100)], freqs=freqs)
nfuncs = proc.nfft_proc.function_tuple
print("(1) scaling with N (run() w/ preallocated memory, and stage split):")
for N in (1000, 5000, 20000, 50000):
    t, y, dy = mk(N); mem = alloc(N); stream = mem.stream
    tot = bench(lambda: (proc.run([(t, y, dy)], memory=[mem], freqs=[freqs]), proc.finish()))
    S = {}
    def tick(name, t0): stream.synchronize(); S[name] = time.perf_counter() - t0
    t0 = time.perf_counter(); mem.set_gpu_arrays_to_zero(); tick('zero', t0)
    t0 = time.perf_counter(); mem.setdata(t=t, y=y, dy=dy); tick('setdata', t0)
    t0 = time.perf_counter(); mem.transfer_data_to_gpu(); tick('h2d', t0)
    spp = 1. / ((mem.tmax - mem.tmin) * df); kw = dict(transfer_to_host=False, transfer_to_device=False, minimum_frequency=freqs[0], samples_per_peak=spp, block_size=proc.block_size)
    t0 = time.perf_counter(); nfft_adjoint_async(mem.nfft_mem_yw, nfuncs, **kw); tick('nfft_yw', t0)
    t0 = time.perf_counter(); nfft_adjoint_async(mem.nfft_mem_w, nfuncs, **kw); tick('nfft_w', t0)
    # gridding kernel alone on the w grid
    precompute_psi, fast_gaussian_grid, _, _, _ = nfuncs; mw = mem.nfft_mem_w; bs = proc.block_size
    def gridw():
        fast_gaussian_grid.prepared_async_call((int(np.ceil(N / bs)), 1), (bs, 1, 1), stream, mw.t_g.ptr, mw.y_g.ptr, mw.ghat_g.ptr, mw.q1.ptr, mw.q2.ptr, mw.q3.ptr, np.int32(N), np.int32(mw.n), np.int32(1), np.int32(m), np.float32(mem.tmin), np.float32(mem.tmax), np.float32(spp)); stream.synchronize()
    # same with a random permutation of the points (contention test)
    perm = rng.permutation(N); tp, yp, dyp = t[perm], y[perm], dy[perm]
    memp = alloc(N)
    totp = bench(lambda: (proc.run([(tp, yp, dyp)], memory=[memp], freqs=[freqs]), proc.finish()))
    print("  N=%6d: run %.2f ms | stages: %s | w-grid kernel alone %.3f ms | run with time-SHUFFLED points: %.2f ms"
          % (N, tot, " ".join("%s=%.2f" % (k, 1e3 * v) for k, v in S.items()), bench(gridw), totp))

print("\n(2) padded FFT grids (next_fast_len) parity + speed, N=1000:")
t, y, dy = mk(1000)
ref_idx = None
res = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs); p0 = np.array(res[0][1], np.float64)
orig_alloc_grid = NFFTMemory.allocate_grid
def allocate_grid_padded(self, **kwargs):
    self.nf = kwargs.get('nf', self.nf); self.n = next_fast_len(int(self.sigma * self.nf))
    import pycuda.gpuarray as ga
    self.ghat_g = ga.zeros(self.n, dtype=self.complex_type); self.cu_plan = cufft.Plan(self.n, self.complex_type, self.complex_type, stream=self.stream); return self
NFFTMemory.allocate_grid = allocate_grid_padded
mem = alloc(1000)
tot_pad = bench(lambda: (proc.run([(t, y, dy)], memory=[mem], freqs=[freqs]), proc.finish()))
alloc_pad = bench(lambda: alloc(1000))
res = proc.run([(t, y, dy)], memory=[mem], freqs=[freqs]); proc.finish(); p1 = np.array(res[0][1], np.float64)
NFFTMemory.allocate_grid = orig_alloc_grid
mem = alloc(1000)
tot_orig = bench(lambda: (proc.run([(t, y, dy)], memory=[mem], freqs=[freqs]), proc.finish()))
alloc_orig = bench(lambda: alloc(1000))
ipk = int(np.argmax(p0)); sl = slice(max(0, ipk - 2000), ipk + 2000)
ref = LombScargle(t, y, dy).power(freqs[sl], method='cython')
print("  run: original grids %.2f ms -> padded %.2f ms ; alloc(incl plans): %.2f -> %.2f ms ; max|p_pad - p_orig| = %.2e ; vs exact astropy (4000 freqs @peak): orig max|d|=%.2e pad max|d|=%.2e"
      % (tot_orig, tot_pad, alloc_orig, alloc_pad, np.abs(p1 - p0).max(), np.abs(p0[sl] - ref).max(), np.abs(p1[sl] - ref).max()))

print("\n(3) psi-fix + m sweep accuracy at survey scale (N=1000, nf=365000; exact astropy on 4000 freqs around peak + 4000 at top of band):")
def patched_alloc_grids(orig):
    def allocate_grids(self, **kws):
        r = orig(self, **kws)
        if self.use_fft:
            self.nfft_mem_w.precomp_psi = True
            self.nfft_mem_w.allocate_precomp_psi(n0=self.n0_buffer if self.buffered_transfer else kws.get('n0', self.n0))
        return r
    return allocate_grids
sl2 = slice(nf - 4000, nf)
ref2 = LombScargle(t, y, dy).power(freqs[sl2], method='cython')
for label, mm, patch in [("m=8 (as shipped)", 8, False), ("m=8 psi-fix", 8, True), ("m=6 psi-fix", 6, True), ("m=5 psi-fix", 5, True), ("m=4 psi-fix", 4, True)]:
    orig = LombScargleMemory.allocate_grids
    if patch: LombScargleMemory.allocate_grids = patched_alloc_grids(orig)
    pr = LombScargleAsyncProcess(m=mm)
    r = pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs); p = np.array(r[0][1], np.float64)
    LombScargleMemory.allocate_grids = orig
    print("  %-18s peak: max|d|=%.2e rel@peak=%.2e | top-of-band: max|d|=%.2e (max power there %.2e)" % (label, np.abs(p[sl] - ref).max(), abs(p[ipk] - ref[ipk - sl.start]) / ref.max(), np.abs(p[sl2] - ref2).max(), ref2.max()))
