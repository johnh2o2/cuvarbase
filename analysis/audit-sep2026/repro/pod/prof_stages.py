"""Stage decomposition of every BLS path at ZTF/HAT/TESS sizes (RTX 4090, shared GPU)."""
import time, json, sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from prof_common import *
import cuvarbase.bls as B
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu_fast_optimized, eebls_gpu_fast_adaptive,
                           eebls_gpu_batch, eebls_gpu, BLSMemory, _get_cached_kernels,
                           _get_cached_batch_kernels, compile_bls, compile_sparse_bls, sparse_bls_gpu)
from cuvarbase.memory.bls_memory import BLSBatchMemory
from cuvarbase.memory._host import host_array

RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
which = sys.argv[1].split(',') if len(sys.argv) > 1 else list(SURVEYS)
results = {}
thr0 = throttle()
dev = cuda.Context.get_device()
print("GPU:", dev.name(), " cgroup throttle at start:", thr0, flush=True)

for name in which:
    cfg = SURVEYS[name]; ndata = cfg['ndata']
    freqs, qmins, qmaxs = grid_for(cfg); nfreq = len(freqs)
    t, y, dy = make_lc(cfg, 1000)
    R = results[name] = dict(ndata=ndata, nfreq=nfreq)
    print(f"\n===== {name}: ndata={ndata} nfreq={nfreq} =====", flush=True)

    # ---------- eebls_gpu_fast (default) : stage replica ----------
    log = []
    funcs_raw = _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])
    funcs = proxied(funcs_raw, log)
    st = {}
    st['kernel_cache_lookup'], _ = med(lambda: _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused']), RUNS)
    st['A_BLSMemory_init_pinned_host'], _ = med(lambda: BLSMemory(ndata, nfreq), RUNS)
    st['A1_pinned_nfreq_arrays_x3'], _ = med(lambda: [host_array((nfreq,), np.float32) for _ in range(3)], RUNS)
    st['A2_pinned_ndata_arrays_x3'], _ = med(lambda: [host_array((ndata,), np.float32) for _ in range(3)], RUNS)
    def gpu_alloc():
        m = BLSMemory.__new__(BLSMemory); m.rtype = np.float32; m.max_nfreqs = nfreq; m.t = np.empty(ndata, np.float32)
        m.allocate_freqs(nfreq); m.allocate_data(ndata)
    st['B_gpu_alloc_7_arrays'], _ = med(gpu_alloc, RUNS)
    mem = BLSMemory(ndata, nfreq); mem.allocate_freqs(nfreq); mem.allocate_data(ndata)
    st['C_setdata_host_compute_with_freqs'], _ = med(lambda: mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False), RUNS)
    st['C2_setdata_host_compute_no_freqs'], _ = med(lambda: mem.setdata(t, y, dy, freqs=None, transfer=False), RUNS)
    st['C3_conflict_scatter_perm'], _ = med(lambda: B.conflict_scatter_perm(ndata), RUNS)
    st['C4_chi2_null'], _ = med(lambda: B._chi2_null(y, dy), RUNS)
    mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False)
    R['freqs_pinned_after_setdata'] = type(mem.freqs.base).__name__ if mem.freqs.base is not None else 'ndarray(own)'
    R['t_pinned_after_setdata'] = type(mem.t.base).__name__ if mem.t.base is not None else 'ndarray(own)'
    st['D_H2D_with_freqs'], _ = med(lambda: mem.transfer_data_to_gpu(transfer_freqs=True), RUNS)
    st['D2_H2D_data_only'], _ = med(lambda: mem.transfer_data_to_gpu(transfer_freqs=False), RUNS)
    def kern(): eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False, functions=funcs)
    st['E_kernel_fused_nov2'], _ = med(kern, RUNS); R['fast_kernel_launches'] = summarize_log(log)
    def kern1(): eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False, functions=funcs, noverlap=1)
    st['E1_kernel_1pass'], _ = med(kern1, RUNS); R['fast_kernel_launches_1pass'] = summarize_log(log)
    st['F_D2H_norm'], _ = med(lambda: mem.transfer_data_to_cpu(), RUNS)
    R['bls_pinned_after_d2h'] = type(mem.bls.base).__name__ if mem.bls.base is not None else 'ndarray(own)'
    st['TOTAL_fast_naive_call'], allt = med(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs), RUNS)
    R['fast_naive_all_runs'] = allt
    mem2 = BLSMemory(ndata, nfreq); mem2.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    def reuse():
        mem2.setdata(t, y, dy, freqs=None, transfer=True)
        eebls_gpu_fast(t, y, dy, freqs, memory=mem2, transfer_to_device=False)
    st['TOTAL_fast_reuse_call'], _ = med(reuse, RUNS)
    def reuse_tdev():  # memory given but transfer_to_device=True (re-uploads freqs+nbins every call)
        eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem2, transfer_to_device=True)
    st['TOTAL_fast_reuse_transfer_to_device_True'], _ = med(reuse_tdev, RUNS)
    R['fast'] = st
    for k, v in st.items(): print(f"  fast   {k:45s} {v*1e3:9.3f} ms", flush=True)

    # ---------- optimized / adaptive whole-call and kernel-only ----------
    so = {}
    so['optimized_naive_call'], _ = med(lambda: eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=qmins, qmax=qmaxs), RUNS)
    so['adaptive_naive_call'], _ = med(lambda: eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxs), RUNS)
    # kernel-only for adaptive: pass memory (block size 256 for all three surveys since ndata>128)
    so['adaptive_kernel_only'], _ = med(lambda: eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False), RUNS)
    so['optimized_kernel_only'], _ = med(lambda: eebls_gpu_fast_optimized(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, memory=mem, transfer_to_device=False, transfer_to_host=False), RUNS)
    # eebls_transit(use_optimized=True) path: functions fetched with only the optimized name
    fo = _get_cached_kernels(256, True, ['full_bls_no_sol_optimized'])
    so['transit_use_optimized_functions_has_fused'] = 'full_bls_no_sol_fused' in fo
    R['opt'] = so
    for k, v in so.items(): print(f"  opt    {k:45s} {v*1e3 if isinstance(v,float) else v}", flush=True)

    # ---------- batch ----------
    nl = 8 if name != 'HAT' else 4
    lcs = [make_lc(cfg, 1000+i) for i in range(nl)]
    blog = []; bfuncs = proxied(_get_cached_batch_kernels(256), blog)
    sb = {}
    stream = cuda.Stream()
    sb['A_BLSBatchMemory_alloc'], _ = med(lambda: BLSBatchMemory(ndata, nl, nfreq, stream=stream), RUNS)
    bmem = BLSBatchMemory(ndata, nl, nfreq, stream=stream)
    # instrument
    acc = {}
    def wrap(obj, meth, key, sync_before=False, sync_after=False):
        orig = getattr(obj, meth)
        def f(*a, **k):
            if sync_before:
                t0 = time.perf_counter(); stream.synchronize(); acc['kernel_wait'] = acc.get('kernel_wait', 0) + time.perf_counter()-t0
            t0 = time.perf_counter(); r = orig(*a, **k)
            if sync_after: stream.synchronize()
            acc[key] = acc.get(key, 0) + time.perf_counter()-t0; return r
        setattr(obj, meth, f)
    wrap(bmem, 'set_lightcurve', 'set_lightcurve_host')
    wrap(bmem, 'transfer_to_gpu', 'H2D', sync_after=True)
    wrap(bmem, 'transfer_to_cpu', 'D2H', sync_before=True)
    wrap(bmem, 'get_results', 'get_results')
    def batch_reuse(): eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs, memory=bmem, functions=bfuncs)
    sb['TOTAL_batch_reuse_call'], _ = med(batch_reuse, RUNS)
    n_calls = RUNS + 1
    for k, v in acc.items(): sb['stage_' + k] = v / n_calls
    sb['batch_kernel_launches'] = summarize_log(blog)
    sb['TOTAL_batch_fresh_memory_call'], _ = med(lambda: eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs), RUNS)
    sb['per_lc_batch_reuse_ms'] = sb['TOTAL_batch_reuse_call'] / nl * 1e3
    sb['n_lcs'] = nl
    R['batch'] = sb
    for k, v in sb.items(): print(f"  batch  {k:45s} {v*1e3 if isinstance(v,float) and k!='per_lc_batch_reuse_ms' else v}", flush=True)

    # ---------- sparse (ZTF only: eebls_transit default for ndata<500) ----------
    if ndata < 500:
        ss = {}
        ss['compile_sparse_bls_warm(per-call cost)'], _ = med(lambda: compile_sparse_bls(block_size=64), RUNS)
        kern_s = compile_sparse_bls(block_size=64)
        ss['TOTAL_sparse_gpu_call_default(compiles)'], _ = med(lambda: sparse_bls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs), 3)
        ss['TOTAL_sparse_gpu_call_kernel_given'], _ = med(lambda: sparse_bls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, kernel=kern_s), RUNS)
        # replica for kernel-only time
        tt = (t - np.floor(t.min())).astype(np.float32)
        t_g = gpuarray.to_gpu(tt); y_g = gpuarray.to_gpu(y.astype(np.float32)); dy_g = gpuarray.to_gpu(dy.astype(np.float32))
        f_g = gpuarray.to_gpu(freqs.astype(np.float32)); qmn_g = gpuarray.to_gpu(qmins.astype(np.float32)); qmx_g = gpuarray.to_gpu(qmaxs.astype(np.float32))
        p_g = gpuarray.zeros(nfreq, np.float32); q_g = gpuarray.zeros(nfreq, np.float32); ph_g = gpuarray.zeros(nfreq, np.float32)
        n_pow2 = 1
        while n_pow2 < ndata: n_pow2 *= 2
        shm = (3*n_pow2 + 2*ndata + 3*64)*4
        def sk():
            kern_s(t_g, y_g, dy_g, f_g, qmn_g, qmx_g, np.uint32(ndata), np.uint32(nfreq), np.uint32(0), p_g, q_g, ph_g,
                   block=(64,1,1), grid=(min(nfreq,65535),1), shared=shm)
        ss['sparse_kernel_only_bs64'], _ = med(sk, RUNS)
        for bs in (128, 256):
            kb = compile_sparse_bls(block_size=bs); shmb = (3*n_pow2 + 2*ndata + 3*bs)*4
            def skb():
                kb(t_g, y_g, dy_g, f_g, qmn_g, qmx_g, np.uint32(ndata), np.uint32(nfreq), np.uint32(0), p_g, q_g, ph_g,
                   block=(bs,1,1), grid=(min(nfreq,65535),1), shared=shmb)
            ss[f'sparse_kernel_only_bs{bs}'], _ = med(skb, RUNS)
        R['sparse'] = ss
        for k, v in ss.items(): print(f"  sparse {k:45s} {v*1e3:9.3f} ms", flush=True)

    # ---------- eebls_gpu (standard, solutions; eebls_transit default for ndata>=500) ----------
    sg = {}
    sg['compile_bls_warm(per-call cost, pycuda disk cache)'], _ = med(lambda: compile_bls(), 3)
    fr = compile_bls(); glog = []; gf = proxied(fr, glog)
    # cap memory: default grabs 0.9*free device memory (shared GPU!)
    MAXMEM = 1.5e9
    nf_std = nfreq if name != 'HAT' else 20000
    fs, qn, qx = freqs[:nf_std], qmins[:nf_std], qmaxs[:nf_std]
    # instrument gpuarray.zeros to measure allocation+memset time
    zacc = [0.0]; orig_zeros = B.gpuarray.zeros
    def tz(*a, **k):
        t0 = time.perf_counter(); r = orig_zeros(*a, **k); sync(); zacc[0] += time.perf_counter()-t0; return r
    B.gpuarray.zeros = tz
    zacc[0] = 0.0
    sg['TOTAL_eebls_gpu_call(functions given, max_memory=1.5GB)'], _ = med(lambda: eebls_gpu(t, y, dy, fs, qmin=qn, qmax=qx, functions=gf, max_memory=MAXMEM), 3)
    sg['stage_gpuarray_zeros_alloc+memset'] = zacc[0] / 4
    B.gpuarray.zeros = orig_zeros
    sg['std_kernel_launches'] = summarize_log(glog)
    sg['nfreq_used'] = nf_std
    # how much memory would the DEFAULT max_memory grab?
    free, total = cuda.mem_get_info(); sg['device_free_MB_now'] = free/1e6; sg['default_max_memory_MB'] = 0.9*free/1e6
    R['std'] = sg
    for k, v in sg.items(): print(f"  std    {k:55s} {v*1e3 if isinstance(v,float) else v}", flush=True)
    del mem, mem2, bmem

results['throttle_delta'] = [a-b for a, b in zip(throttle(), thr0)]
print("cgroup throttle delta (nr, usec):", results['throttle_delta'])
json.dump(results, open(f'/workspace/scratch/prof_stages_{"_".join(which)}.json', 'w'), indent=1, default=str)
