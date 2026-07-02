"""E1 diagnosis: why is eebls_gpu_batch ~12x slower than the single-LC
fast path at TESS scale (ndata=20,000)?

Stage-times the batch path with CUDA events + wall clocks and compares
against a single-LC eebls_gpu_fast loop on identical data/parameters.
Configs cover the benchmark's regimes: small (batch wins), TESS scale
(the regression).

Prints a stage table; JSON_RESULT line at the end.
"""
import json
import time

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from cuvarbase.bls import (eebls_gpu_batch, eebls_gpu_fast,
                           compile_bls_batch, _choose_block_size,
                           _default_block_size, compile_bls)
from cuvarbase.memory.bls_memory import BLSBatchMemory


def make_lcs(n_lcs, ndata, seed=11):
    rand = np.random.RandomState(seed)
    lcs = []
    for i in range(n_lcs):
        t = np.sort(365.0 * rand.rand(ndata))
        phase = (t * 0.5) % 1.0
        y = 1.0 - 0.01 * (phase < 0.03) + 0.01 * rand.randn(ndata)
        lcs.append((t, y, 0.01 * np.ones(ndata)))
    return lcs


def profile_batch(lcs, freqs, noverlap=2, dlogq=0.3):
    """Manually reproduce eebls_gpu_batch's stages with timers."""
    stages = {}
    freqs32 = np.asarray(freqs).astype(np.float32)
    nfreq = len(freqs32)
    n_lcs = len(lcs)
    ndata = max(len(lc[0]) for lc in lcs)

    t0 = time.time()
    block_size = _choose_block_size(ndata)
    functions = compile_bls_batch(block_size=block_size)
    func = functions['full_bls_batch']
    stages['compile_s'] = time.time() - t0

    t0 = time.time()
    stream = cuda.Stream()
    mem = BLSBatchMemory(ndata, n_lcs, nfreq, stream=stream)
    max_nbins = mem.set_freqs(freqs32, qmin=1e-2, qmax=0.5)
    for j, (t, y, dy) in enumerate(lcs):
        mem.set_lightcurve(j, t, y, dy)
    stages['host_prep_s'] = time.time() - t0

    t0 = time.time()
    mem.transfer_to_gpu()
    stream.synchronize()
    stages['h2d_s'] = time.time() - t0

    float_size = 4
    mem_req = (block_size + 2 * max_nbins) * float_size
    grid = (min(nfreq, 5000), n_lcs)
    block = (block_size, 1, 1)

    # kernel passes, CUDA-event timed
    ev0, ev1 = cuda.Event(), cuda.Event()
    ev0.record(stream)
    best = None
    for i_pass in range(noverlap):
        args = (grid, block, stream)
        args += (mem.t_g.ptr, mem.yw_g.ptr, mem.w_g.ptr)
        args += (mem.bls_g.ptr, mem.freqs_g.ptr)
        args += (mem.nbins0_g.ptr, mem.nbinsf_g.ptr)
        args += (mem.ndata_per_lc_g.ptr,)
        args += (np.uint32(ndata), np.uint32(nfreq), np.uint32(0))
        args += (np.uint32(max_nbins), np.uint32(1))
        args += (np.float32(dlogq), np.float32(float(i_pass) / noverlap))
        args += (np.uint32(0), np.uint32(n_lcs))
        func.prepared_async_call(*args, shared_size=int(mem_req))
        if noverlap > 1:
            if best is None:
                best = mem.bls_g.copy()
            else:
                gpuarray.maximum(mem.bls_g, best, out=best, stream=stream)
    ev1.record(stream)
    ev1.synchronize()
    stages['kernel_ms_cuda'] = ev1.time_since(ev0)

    t0 = time.time()
    mem.transfer_to_cpu()
    _ = mem.get_results()
    stages['d2h_s'] = time.time() - t0
    return stages


def main():
    out = []
    for (n_lcs, ndata, nfreq) in [(10, 200, 5000),
                                  (10, 2000, 5000),
                                  (10, 20000, 1788),
                                  (2, 20000, 1788)]:
        lcs = make_lcs(n_lcs, ndata)
        freqs = np.linspace(0.1, 1.0, nfreq)

        # warm both paths (compile out of the timing)
        _ = eebls_gpu_fast(*lcs[0], freqs[:50])
        _ = eebls_gpu_batch(lcs[:1], freqs[:50])

        # single-LC loop wall time
        t0 = time.time()
        for lc in lcs:
            _ = eebls_gpu_fast(*lc, freqs)
        t_single = time.time() - t0

        # batch wall time (public API)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            t0 = time.time()
            _ = eebls_gpu_batch(lcs, freqs)
            t_batch = time.time() - t0

        stages = profile_batch(lcs, freqs)
        row = dict(n_lcs=n_lcs, ndata=ndata, nfreq=nfreq,
                   t_single_loop_s=t_single, t_batch_s=t_batch,
                   batch_over_single=t_batch / t_single, **stages)
        out.append(row)
        print("n_lcs=%-3d ndata=%-6d nfreq=%-5d single=%.3fs batch=%.3fs "
              "(%.2fx)  stages: compile=%.3f prep=%.3f h2d=%.3f "
              "kernel=%.1fms d2h=%.3f"
              % (n_lcs, ndata, nfreq, t_single, t_batch,
                 row['batch_over_single'], stages['compile_s'],
                 stages['host_prep_s'], stages['h2d_s'],
                 stages['kernel_ms_cuda'], stages['d2h_s']), flush=True)

    print("JSON_RESULT: " + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
