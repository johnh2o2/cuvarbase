import sys, time, cProfile, pstats, io
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0, lomb_scargle_async
from cuvarbase.cunfft import nfft_adjoint_async
from cuvarbase.utils import normalize_light_curves

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

for label, N, T, fmax in [("ZTF-like", 300, 3650.0, 20.0), ("Kepler-like", 65000, 1400.0, 50.0)]:
    freqs = grid(1.0 / (3 * T), fmax, T, spp=3)
    lcs = [make_lc(N=N, T=T, f0=1 + i, seed=i) for i in range(10)]
    proc = LombScargleAsyncProcess(use_double=False, sigma=4, m=8, autoset_m=False)
    proc.batched_run_const_nfreq(lcs[:1], freqs=freqs)
    print("=================== %s N=%d nf=%d: cProfile of batched_run_const_nfreq(10 LCs) ===================" % (label, N, len(freqs)))
    pr = cProfile.Profile(); pr.enable()
    t0 = time.perf_counter(); proc.batched_run_const_nfreq(lcs, freqs=freqs); dt = time.perf_counter() - t0
    pr.disable()
    print("total %.1f ms (%.1f ms/LC)" % (1e3 * dt, 1e2 * dt))
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(28); print(s.getvalue()[:6000])
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(15); print(s.getvalue()[:3500])
    # GPU stage breakdown
    t, y, dy = lcs[0]
    (tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
    mem = proc.allocate([(tn, yn, dyn)], nfreqs=[len(freqs)], k0s=[get_k0(freqs)])[0]
    mem.setdata(t=tn, y=yn, dy=dyn)
    funcs = (proc.function_tuple, proc.nfft_proc.function_tuple)
    def bench(fn, reps=9):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return np.median(ts)
    def st_set(): mem.setdata(t=tn, y=yn, dy=dyn)
    def st_norm(): normalize_light_curves([(t, y, dy)])
    def st_xfer(): mem.transfer_data_to_gpu(); mem.stream.synchronize()
    def st_all(): lomb_scargle_async(mem, funcs, freqs, block_size=256, transfer_to_device=False); mem.stream.synchronize()
    for name, fn in [("normalize_light_curves (host)", st_norm), ("setdata (host)", st_set), ("H2D transfer", st_xfer), ("GPU: nfft x2 + lomb + D2H", st_all)]:
        print("   %-32s %.2f ms" % (name, 1e3 * bench(fn)))
    del proc
