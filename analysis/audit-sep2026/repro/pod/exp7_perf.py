import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0, fap_baluev, lomb_scargle_async
from cuvarbase.cunfft import nfft_adjoint_async
import pycuda.driver as cuda

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def bench(fn, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts = np.sort(ts)
    return np.median(ts), ts[0]

print("=== per-LC time, batched_run_const_nfreq(batch_size=1), 10 LCs; relative comparisons only (shared 4090) ===")
for label, N, T, fmax in [("ZTF-like", 300, 3650.0, 20.0), ("Kepler-like", 65000, 1400.0, 50.0)]:
    freqs = grid(1.0 / (3 * T), fmax, T, spp=3)
    lcs = [make_lc(N=N, T=T, f0=1 + i, seed=i) for i in range(10)]
    print("%s: N=%d nf=%d" % (label, N, len(freqs)))
    for (dbl, sigma, m) in [(False, 4, 8), (False, 2, 8), (False, 4, 4), (False, 2, 4), (True, 4, 8), (False, 3, 6)]:
        proc = LombScargleAsyncProcess(use_double=dbl, sigma=sigma, m=m, autoset_m=False)
        proc.batched_run_const_nfreq(lcs[:1], freqs=freqs)  # warm-up / compile
        med, best = bench(lambda: proc.batched_run_const_nfreq(lcs, freqs=freqs), reps=5)
        print("   dbl=%s sigma=%d m=%2d : median %.1f ms/LC (best %.1f)" % (dbl, sigma, m, 1e3 * med / 10, 1e3 * best / 10))
        del proc

print("=== stage breakdown (ZTF-like, float32, sigma=4, m=8): sync after each stage ===")
N, T, fmax = 300, 3650.0, 20.0
freqs = grid(1.0 / (3 * T), fmax, T, spp=3)
t, y, dy = make_lc(N=N, T=T, f0=3.3, seed=1)
proc = LombScargleAsyncProcess(use_double=False, sigma=4, m=8, autoset_m=False)
proc.run([(t, y, dy)], freqs=freqs); proc.finish()
from cuvarbase.utils import normalize_light_curves
(tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
mem = proc.allocate([(tn, yn, dyn)], nfreqs=[len(freqs)], k0s=[get_k0(freqs)])[0]
mem.setdata(t=tn, y=yn, dy=dyn)
funcs = (proc.function_tuple, proc.nfft_proc.function_tuple)
df = freqs[1] - freqs[0]; spp = 1.0 / ((mem.tmax - mem.tmin) * df)
def stage_transfer(): mem.transfer_data_to_gpu(); mem.stream.synchronize()
def stage_nfft_yw(): nfft_adjoint_async(mem.nfft_mem_yw, funcs[1], transfer_to_host=False, transfer_to_device=False, minimum_frequency=freqs[0], samples_per_peak=spp); mem.stream.synchronize()
def stage_nfft_w(): nfft_adjoint_async(mem.nfft_mem_w, funcs[1], transfer_to_host=False, transfer_to_device=False, minimum_frequency=freqs[0], samples_per_peak=spp); mem.stream.synchronize()
def stage_all(): lomb_scargle_async(mem, funcs, freqs, block_size=256); mem.stream.synchronize()
def stage_all_nohost(): lomb_scargle_async(mem, funcs, freqs, block_size=256, transfer_to_host=False); mem.stream.synchronize()
for name, fn in [("transfer data", stage_transfer), ("nfft yw (ng=%d)" % mem.nfft_mem_yw.n, stage_nfft_yw), ("nfft w (ng=%d)" % mem.nfft_mem_w.n, stage_nfft_w), ("full lomb_scargle_async", stage_all), ("full, no host transfer", stage_all_nohost)]:
    med, best = bench(fn, reps=15)
    print("   %-28s median %.2f ms  best %.2f ms" % (name, 1e3 * med, 1e3 * best))
# memory-set-up cost per call of batched_run_const_nfreq
t0 = time.perf_counter(); proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs); t1 = time.perf_counter()
print("   batched_run_const_nfreq single LC call (incl. memory+plan setup): %.1f ms" % (1e3 * (t1 - t0)))

print("=== FAP cost in only_return_best_freqs (CPU): fap_baluev on full nf array vs scalar ===")
for nfx in (365000, 1825000):
    z = np.random.rand(nfx) * 0.5
    med, _ = bench(lambda: fap_baluev(t, dy, z, 20.0), reps=5)
    meds, _ = bench(lambda: fap_baluev(t, dy, z[:1], 20.0), reps=5)
    print("   nf=%d: full-array FAP %.1f ms ; scalar FAP %.3f ms" % (nfx, 1e3 * med, 1e3 * meds))
