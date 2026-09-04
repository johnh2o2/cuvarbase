"""v2: GPU stages via CUDA events, host stages via process_time (CFS-throttling-immune) + wall min/median."""
import time, json, sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from prof_common import *
import cuvarbase.bls as B
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu_fast_optimized, eebls_gpu_fast_adaptive, eebls_gpu_batch, eebls_gpu,
                           BLSMemory, _get_cached_kernels, _get_cached_batch_kernels, compile_bls, compile_sparse_bls,
                           sparse_bls_gpu, count_tot_nbins)
from cuvarbase.memory.bls_memory import BLSBatchMemory
from cuvarbase.memory._host import host_array

RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
which = sys.argv[1].split(',') if len(sys.argv) > 1 else list(SURVEYS)

def hst(fn, runs=RUNS, warm=1, gsync=True):
    """host-stage timer: returns dict(wall_med, wall_min, cpu_med) in ms"""
    for _ in range(warm): fn()
    W, C = [], []
    for _ in range(runs):
        if gsync: sync()
        c0 = time.process_time(); t0 = time.perf_counter(); fn()
        if gsync: sync()
        W.append(time.perf_counter()-t0); C.append(time.process_time()-c0)
    return dict(wall_med=1e3*float(np.median(W)), wall_min=1e3*float(np.min(W)), cpu_med=1e3*float(np.median(C)))

def gst(fn, log, runs=RUNS, warm=1):
    """GPU-stage timer: fn issues launches through proxies; returns per-kernel event-time medians over runs + launches/run"""
    for _ in range(warm): fn(); log.clear()
    per = []
    for _ in range(runs):
        sync(); fn(); per.append(summarize_log(log))
    names = set(k for p in per for k in p)
    return {n: dict(launches=int(np.median([p.get(n, {}).get('launches', 0) for p in per])),
                    gpu_ms_med=float(np.median([p.get(n, {}).get('gpu_ms', 0) for p in per])),
                    gpu_ms_min=float(np.min([p.get(n, {}).get('gpu_ms', 0) for p in per]))) for n in names}

results = {}; thr0 = throttle(); print("GPU:", cuda.Context.get_device().name(), "throttle0", thr0, flush=True)
for name in which:
    cfg = SURVEYS[name]; ndata = cfg['ndata']; freqs, qmins, qmaxs = grid_for(cfg); nfreq = len(freqs); t, y, dy = make_lc(cfg, 1000)
    R = results[name] = dict(ndata=ndata, nfreq=nfreq); print(f"\n===== {name}: ndata={ndata} nfreq={nfreq} =====", flush=True)
    log = []; funcs = proxied(_get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused']), log)
    st = {}
    st['A_BLSMemory_init(6 pinned host arrays)'] = hst(lambda: BLSMemory(ndata, nfreq))
    st['A1_pinned_nfreq_arrays_x3'] = hst(lambda: [host_array((nfreq,), np.float32) for _ in range(3)])
    st['A2_pinned_ndata_arrays_x3'] = hst(lambda: [host_array((ndata,), np.float32) for _ in range(3)])
    st['A3_pinned_single_16B_array'] = hst(lambda: host_array((4,), np.float32))
    st['A4_pageable_numpy_nfreq_x3'] = hst(lambda: [np.zeros(nfreq, np.float32) for _ in range(3)], gsync=False)
    def gpu_alloc():
        m = BLSMemory.__new__(BLSMemory); m.rtype = np.float32; m.max_nfreqs = nfreq; m.t = np.empty(ndata, np.float32); m.allocate_freqs(nfreq); m.allocate_data(ndata)
    st['B_gpu_alloc_7_gpuarray_zeros'] = hst(gpu_alloc)
    mem = BLSMemory(ndata, nfreq); mem.allocate_freqs(nfreq); mem.allocate_data(ndata)
    st['C_setdata_host(freqs given)'] = hst(lambda: mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False), gsync=False)
    st['C2_setdata_host(freqs=None)'] = hst(lambda: mem.setdata(t, y, dy, freqs=None, transfer=False), gsync=False)
    st['C3_conflict_scatter_perm'] = hst(lambda: B.conflict_scatter_perm(ndata), gsync=False)
    st['C4_chi2_null'] = hst(lambda: B._chi2_null(y, dy), gsync=False)
    st['C5_nbins_from_qmin_qmax'] = hst(lambda: ((np.ones_like(mem.freqs)/qmins).astype(np.uint32), (np.ones_like(mem.freqs)/qmaxs).astype(np.uint32)), gsync=False)
    mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False)
    R['pinned_after_setdata'] = dict(freqs=type(mem.freqs.base).__name__ if mem.freqs.base is not None else 'ndarray', t=type(mem.t.base).__name__ if mem.t.base is not None else 'ndarray',
                                     nbinsf=type(mem.nbinsf.base).__name__ if mem.nbinsf.base is not None else 'ndarray')
    st['D_H2D(with freqs+nbins)'] = hst(lambda: mem.transfer_data_to_gpu(transfer_freqs=True))
    st['D2_H2D(data only)'] = hst(lambda: mem.transfer_data_to_gpu(transfer_freqs=False))
    kern = lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False, functions=funcs)
    st['E_kernel_wall(fused nov2)'] = hst(kern); R['E_kernel_events'] = gst(kern, log)
    kern1 = lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False, functions=funcs, noverlap=1)
    R['E1_kernel_events_1pass'] = gst(kern1, log)
    st['F_D2H+normalize'] = hst(lambda: mem.transfer_data_to_cpu())
    R['bls_pinned_after_d2h'] = type(mem.bls.base).__name__ if mem.bls.base is not None else 'ndarray'
    st['TOTAL_fast_naive_call'] = hst(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    mem2 = BLSMemory(ndata, nfreq); mem2.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    def reuse():
        mem2.setdata(t, y, dy, freqs=None, transfer=True); eebls_gpu_fast(t, y, dy, freqs, memory=mem2, transfer_to_device=False)
    st['TOTAL_fast_reuse_call'] = hst(reuse)
    st['TOTAL_fast_reuse(transfer_to_device=True)'] = hst(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem2, transfer_to_device=True))
    st['TOTAL_optimized_naive_call'] = hst(lambda: eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    st['TOTAL_adaptive_naive_call'] = hst(lambda: eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    kw = dict(qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False)
    _orig = B._get_cached_kernels; B._get_cached_kernels = lambda *a, **k: proxied(_orig(*a, **k), log)
    R['adaptive_kernel_events'] = gst(lambda: eebls_gpu_fast_adaptive(t, y, dy, freqs, **kw), log)
    R['optimized_kernel_events'] = gst(lambda: eebls_gpu_fast_optimized(t, y, dy, freqs, **kw), log)
    R['fast_nov3_multipass_events'] = gst(lambda: eebls_gpu_fast(t, y, dy, freqs, noverlap=3, **kw), log)
    B._get_cached_kernels = _orig
    R['fast'] = st
    for k, v in st.items(): print(f"  fast   {k:45s} wall med {v['wall_med']:9.3f}  min {v['wall_min']:9.3f}  cpu {v['cpu_med']:9.3f} ms", flush=True)
    for k in ('E_kernel_events', 'E1_kernel_events_1pass', 'adaptive_kernel_events', 'optimized_kernel_events', 'fast_nov3_multipass_events'): print(f"  fast   {k:45s} {R[k]}", flush=True)
    print("  pinned after setdata:", R['pinned_after_setdata'], " bls after d2h:", R['bls_pinned_after_d2h'], flush=True)

    # ---------- batch ----------
    nl = 8 if name != 'HAT' else 4; lcs = [make_lc(cfg, 1000+i) for i in range(nl)]
    blog = []; bfuncs = proxied(_get_cached_batch_kernels(256), blog); sb = {}; stream = cuda.Stream()
    sb['A_BLSBatchMemory_alloc'] = hst(lambda: BLSBatchMemory(ndata, nl, nfreq, stream=stream))
    bmem = BLSBatchMemory(ndata, nl, nfreq, stream=stream)
    sb['B_set_lightcurve_x%d' % nl] = hst(lambda: [bmem.set_lightcurve(j, *lcs[j]) for j in range(nl)], gsync=False)
    bmem.set_freqs(freqs, qmin=qmins, qmax=qmaxs)
    sb['C_transfer_to_gpu'] = hst(lambda: bmem.transfer_to_gpu(n_lcs_active=nl, transfer_freqs=False))
    sb['D_transfer_to_cpu+get_results'] = hst(lambda: (bmem.transfer_to_cpu(n_lcs_active=nl), bmem.get_results(n_lcs_active=nl, nfreq_active=nfreq)))
    bcall = lambda: eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs, memory=bmem, functions=bfuncs)
    sb['TOTAL_batch_reuse_call'] = hst(bcall); R['batch_kernel_events'] = gst(bcall, blog)
    sb['TOTAL_batch_fresh_memory_call'] = hst(lambda: eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs), runs=5)
    R['batch'] = sb; R['batch_n_lcs'] = nl
    for k, v in sb.items(): print(f"  batch  {k:45s} wall med {v['wall_med']:9.3f}  min {v['wall_min']:9.3f}  cpu {v['cpu_med']:9.3f} ms", flush=True)
    print("  batch  kernel events:", R['batch_kernel_events'], flush=True)

    # ---------- sparse (ZTF) ----------
    if ndata < 500:
        ss = {}
        ss['compile_sparse_bls(per-call)'] = hst(lambda: compile_sparse_bls(block_size=64), runs=3)
        ks = compile_sparse_bls(block_size=64)
        ss['TOTAL_sparse_gpu_call(kernel given)'] = hst(lambda: sparse_bls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, kernel=ks), runs=5)
        tt = (t - np.floor(t.min())).astype(np.float32)
        gs_ = [gpuarray.to_gpu(a.astype(np.float32)) for a in (tt, y, dy, freqs, qmins, qmaxs)]; outs = [gpuarray.zeros(nfreq, np.float32) for _ in range(3)]
        n_pow2 = 1
        while n_pow2 < ndata: n_pow2 *= 2
        def sk():
            e0, e1 = cuda.Event(), cuda.Event(); e0.record()
            ks(*gs_, np.uint32(ndata), np.uint32(nfreq), np.uint32(0), *outs, block=(64,1,1), grid=(min(nfreq,65535),1), shared=(3*n_pow2+2*ndata+3*64)*4)
            e1.record(); e1.synchronize(); return e0.time_till(e1)
        sk(); ss['sparse_kernel_event_ms'] = dict(med=float(np.median([sk() for _ in range(RUNS)])), min=float(np.min([sk() for _ in range(RUNS)])))
        R['sparse'] = ss
        for k, v in ss.items(): print(f"  sparse {k:45s} {v}", flush=True)

    # ---------- eebls_gpu standard path (eebls_transit default for ndata>=500) ----------
    sg = {}
    sg['compile_bls(per-call, pycuda cache warm)'] = hst(lambda: compile_bls(), runs=3)
    fr = compile_bls(); glog = []; gf = proxied(fr, glog)
    nf_std = nfreq if name != 'HAT' else 20000; fs, qn, qx = freqs[:nf_std], qmins[:nf_std], qmaxs[:nf_std]
    # pick a max_memory whose batch boundaries do not trip the count_tot_nbins sizing bug
    def overflows(maxmem):
        nb0 = int(np.floor(1./qx.max())); nbf = int(np.ceil(1./qn.min())); ntot = count_tot_nbins(nb0, nbf, 0.2)
        fbs = int((maxmem - (ndata*12 + len(fs)*20))/(4*5*ntot*3*4)); gs = fbs*ntot*3
        for b in range(int(np.ceil(len(fs)/fbs))):
            i0, i1 = fbs*b, min(len(fs), fbs*(b+1))
            if (i1-i0)*count_tot_nbins(int(np.floor(1./qx[i0:i1].max())), int(np.ceil(1./qn[i0:i1].min())), 0.2)*3 > gs: return True
        return False
    MAXMEM = next((m for m in (1.5e9, 2e9, 2.5e9, 3e9, 4e9, 5e9) if not overflows(m)), None)
    sg['max_memory_used'] = MAXMEM
    if MAXMEM is not None:
        zacc = [0.0, 0]; oz = B.gpuarray.zeros
        def tz(*a, **k):
            c0 = time.process_time(); r = oz(*a, **k); zacc[0] += time.process_time()-c0; zacc[1] += r.nbytes; return r
        B.gpuarray.zeros = tz
        gcall = lambda: eebls_gpu(t, y, dy, fs, qmin=qn, qmax=qx, functions=gf, max_memory=MAXMEM)
        sg['TOTAL_eebls_gpu_call'] = hst(gcall, runs=3); ncall = 4
        sg['gpuarray_zeros_cpu_ms_per_call'] = zacc[0]/ncall*1e3; sg['gpuarray_zeros_MB_per_call'] = zacc[1]/ncall/1e6
        B.gpuarray.zeros = oz
        R['std_kernel_events'] = gst(gcall, glog, runs=3)
    sg['nfreq_used'] = nf_std; R['std'] = sg
    for k, v in sg.items(): print(f"  std    {k:45s} {v}", flush=True)
    if 'std_kernel_events' in R: print("  std    kernel events:", R['std_kernel_events'], flush=True)
    del mem, mem2, bmem
results['throttle_delta'] = [a-b for a, b in zip(throttle(), thr0)]; print("throttle delta", results['throttle_delta'])
json.dump(results, open(f'/workspace/scratch/prof2_{"_".join(which)}.json', 'w'), indent=1, default=str)
