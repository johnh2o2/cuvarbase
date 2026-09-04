"""Cross-path parity + launch counting (does adaptive/optimized/eebls_transit(use_optimized) use the fused kernel?)
+ astropy sanity + eebls_gpu default-memory arithmetic."""
import time, json, sys
import numpy as np
import pycuda.driver as cuda
from prof_common import *
import cuvarbase.bls as B
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu_fast_optimized, eebls_gpu_fast_adaptive,
                           eebls_gpu_batch, eebls_gpu, eebls_transit, BLSMemory, compile_bls)

log = []
_orig = B._get_cached_kernels
def patched(*a, **k):
    return proxied(_orig(*a, **k), log)
B._get_cached_kernels = patched

def stats(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), argmax_same=bool(np.argmax(a) == np.argmax(b)),
                max_abs_diff=float(np.max(np.abs(a - b))), n_bitdiff=int(np.sum(a != b)))

out = {}
for name in ('ZTF', 'TESS'):
    cfg = SURVEYS[name]; freqs, qmins, qmaxs = grid_for(cfg); t, y, dy = make_lc(cfg, 7)
    R = out[name] = {}
    log.clear(); p_fast = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs); R['fast_launches'] = summarize_log(log)
    log.clear(); p_opt = eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=qmins, qmax=qmaxs); R['optimized_launches'] = summarize_log(log)
    log.clear(); p_ad = eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxs); R['adaptive_launches'] = summarize_log(log)
    log.clear(); _, p_tr, _ = eebls_transit(t, y, dy, freqs=freqs, qvals=qmaxs/2.0, use_optimized=True, use_sparse=False); R['transit_use_optimized_launches'] = summarize_log(log)
    log.clear(); _, p_tf, _ = eebls_transit(t, y, dy, freqs=freqs, qvals=qmaxs/2.0, use_fast=True, use_sparse=False); R['transit_use_fast_launches'] = summarize_log(log)
    p_b = eebls_gpu_batch([(t, y, dy)], freqs, qmin=qmins, qmax=qmaxs)[0]
    R['fast_vs_optimized'] = stats(p_fast, p_opt); R['fast_vs_adaptive'] = stats(p_fast, p_ad)
    R['fast_vs_batch'] = stats(p_fast, p_b); R['fast_vs_transit_opt'] = stats(p_fast, p_tr); R['fast_vs_transit_fast'] = stats(p_fast, p_tf)
    R['peak_period_fast'] = float(1/freqs[np.argmax(p_fast)]); R['peak_power_fast'] = float(p_fast.max())
    # timing: adaptive vs fast, kernel-only, repeated
    mem = BLSMemory(cfg['ndata'], len(freqs)); mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    kw = dict(qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False)
    R['kernel_only_ms'] = {}
    for lab, fn in (('fast', eebls_gpu_fast), ('optimized', eebls_gpu_fast_optimized), ('adaptive', eebls_gpu_fast_adaptive)):
        m, allt = med(lambda: fn(t, y, dy, freqs, **kw), 7); R['kernel_only_ms'][lab] = dict(median=m*1e3, all=[x*1e3 for x in allt])
    # explicit multipass (noverlap=3, no fused) vs fused noverlap=4 for reference
    m3, _ = med(lambda: eebls_gpu_fast(t, y, dy, freqs, noverlap=3, **kw), 5); R['kernel_only_ms']['fast_noverlap3_multipass'] = m3*1e3
    m4, _ = med(lambda: eebls_gpu_fast(t, y, dy, freqs, noverlap=4, **kw), 5); R['kernel_only_ms']['fast_noverlap4_fused'] = m4*1e3
    print(name, json.dumps(R, indent=1, default=str), flush=True)

# ---- astropy sanity on TESS-like (uniform-ish comparison at the injected period) ----
try:
    from astropy.timeseries import BoxLeastSquares
    cfg = SURVEYS['TESS']; freqs, qmins, qmaxs = grid_for(cfg); t, y, dy = make_lc(cfg, 7)
    P0 = 2.5271
    p_cv = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, convention='loglik')
    i_cv = int(np.argmax(p_cv))
    bls = BoxLeastSquares(t, y, dy)
    # durations spanning the Keplerian search window at the injected period
    from cuvarbase.bls import q_transit
    q0 = float(q_transit(1/P0)); durs = P0*q0*np.geomspace(0.5, 2.0, 12)
    res = bls.power(1/freqs, durs, objective='likelihood', oversample=20)
    i_ap = int(np.argmax(res.power))
    out['astropy'] = dict(cuv_peak_P=float(1/freqs[i_cv]), ap_peak_P=float(res.period[i_ap]), true_P=P0,
                          cuv_loglik_at_peak=float(p_cv[i_cv]), ap_loglik_at_peak=float(res.power[i_ap]),
                          ap_depth=float(res.depth[i_ap]), ap_duration=float(res.duration[i_ap]),
                          corr_all=float(np.corrcoef(p_cv, res.power)[0, 1]))
    print('astropy', out['astropy'], flush=True)
except Exception as e:
    out['astropy'] = f'FAILED: {e!r}'; print(out['astropy'])

# ---- eebls_gpu default max_memory arithmetic (do NOT run default on the shared GPU) ----
free, total = cuda.mem_get_info()
cfg = SURVEYS['TESS']; freqs, qmins, qmaxs = grid_for(cfg); t, y, dy = make_lc(cfg, 7)
nbins0_max = int(np.floor(1./np.max(qmaxs))); nbinsf_max = int(np.ceil(1./np.min(qmins)))
ntot = B.count_tot_nbins(nbins0_max, nbinsf_max, 0.2)
mem_per_f = 4*5*ntot*3*4
fbs_default = int((0.9*free - (len(t)*12 + len(freqs)*20)) / mem_per_f)
out['eebls_gpu_default_memory'] = dict(device_free_MB=free/1e6, nbins0_max=nbins0_max, nbinsf_max=nbinsf_max, nbins_tot_max=ntot,
    bytes_per_freq_all_streams=mem_per_f, default_freq_batch_size=fbs_default, nfreq=len(freqs),
    default_alloc_MB=min(fbs_default, len(freqs))*mem_per_f/1e6 if fbs_default > len(freqs) else 0.9*free/1e6,
    note='freq_batch_size defaults to (0.9*free-mem0)/mem_per_f, so gs*nstreams*4 arrays == ~0.9*free bytes are zero-filled per call whenever fbs < nfreq; if fbs >= nfreq the allocation is nfreq*mem_per_f')
# measured: with a 300MB cap, what does one call allocate + how long do the zeros take?
zb = [0, 0.0]; oz = B.gpuarray.zeros
def tz(n, dtype=np.float32, **k):
    t0 = time.perf_counter(); r = oz(n, dtype=dtype, **k); sync(); zb[0] += r.nbytes; zb[1] += time.perf_counter()-t0; return r
B.gpuarray.zeros = tz
fr = compile_bls()
eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, functions=fr, max_memory=3e8)
B.gpuarray.zeros = oz
out['eebls_gpu_default_memory']['measured_alloc_MB_with_300MB_cap'] = zb[0]/1e6
out['eebls_gpu_default_memory']['measured_zeros_time_ms_with_300MB_cap'] = zb[1]*1e3
print('eebls_gpu memory', json.dumps(out['eebls_gpu_default_memory'], indent=1))
json.dump(out, open('/workspace/scratch/parity_paths.json', 'w'), indent=1, default=str)
