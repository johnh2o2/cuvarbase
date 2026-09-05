#!/usr/bin/env python
"""Decomposition diagnostic for the v0.2.6-vs-v1.0 warm BLS gap.

Times four nested variants of eebls_gpu_fast on the SAME data:
  A full        default product call (memory allocated per call)
  B precompiled functions= handle, memory allocated per call
  C mem_reuse   functions= + memory= reused, H2D + kernels + D2H
  D kernel_only functions= + memory= reused, no H2D/D2H (kernel + launch)

Run under each version's python. v1.0 rows use noverlap=1.
"""
import argparse
import json
import subprocess
import time

import numpy as np

BLS_PARAMS = dict(qmin=0.01, qmax=0.5, dlogq=0.3)


def make_lc(ndata, baseline, seed):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata))
    y = np.ones(ndata) + rng.randn(ndata) * 0.002
    dy = np.full(ndata, 0.002)
    phase = (t * (1.0 / 3.456)) % 1.0
    y[phase < 0.03] -= 0.008
    return t, y, dy


def gpu_state():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu',
             '--format=csv,noheader'])
        return out.decode().strip()
    except Exception:
        return '?'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ndata', type=int, default=20000)
    p.add_argument('--baseline', type=float, default=27.4)
    p.add_argument('--nfreq', type=int, default=13500)
    p.add_argument('--reps', type=int, default=15)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    import cuvarbase
    from cuvarbase import bls as B
    import pycuda.driver as drv

    is026 = cuvarbase.__version__.startswith('0.2')

    freqs = (np.arange(1, args.nfreq + 1) * (2.0 / args.nfreq))
    t, y, dy = make_lc(args.ndata, args.baseline, 42)

    kw = dict(BLS_PARAMS)
    if not is026:
        kw['noverlap'] = 1

    funcs = B.compile_bls(function_names=['full_bls_no_sol'])
    mem = B.BLSMemory.fromdata(t, y, dy, freqs=freqs,
                               qmin=kw['qmin'], qmax=kw['qmax'])

    def call_A():
        return B.eebls_gpu_fast(t, y, dy, freqs, **kw)

    def call_B():
        return B.eebls_gpu_fast(t, y, dy, freqs, functions=funcs, **kw)

    def call_C():
        return B.eebls_gpu_fast(t, y, dy, freqs, functions=funcs,
                                memory=mem, transfer_to_device=True,
                                transfer_to_host=True, **kw)

    def call_D():
        return B.eebls_gpu_fast(t, y, dy, freqs, functions=funcs,
                                memory=mem, transfer_to_device=False,
                                transfer_to_host=False, **kw)

    results = {}
    print('cuvarbase %s  ndata=%d nfreq=%d  gpu[%s]'
          % (cuvarbase.__version__, args.ndata, args.nfreq, gpu_state()))
    for name, fn in [('A_full', call_A), ('B_precompiled', call_B),
                     ('C_mem_reuse', call_C), ('D_kernel_only', call_D)]:
        for _ in range(3):
            fn()
        drv.Context.synchronize()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn()
            drv.Context.synchronize()
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        results[name] = dict(median_s=med, times_s=times)
        print('  %-14s median %8.2f ms   min %8.2f   max %8.2f   [%s]'
              % (name, 1e3 * med, 1e3 * min(times), 1e3 * max(times),
                 gpu_state()))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(version=cuvarbase.__version__,
                           ndata=args.ndata, nfreq=args.nfreq,
                           results=results), f)


if __name__ == '__main__':
    main()
