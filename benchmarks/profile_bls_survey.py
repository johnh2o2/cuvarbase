#!/usr/bin/env python3
"""
Minimal driver for attaching ncu/nsys to the survey-scale BLS kernels.

Runs ONLY kernel launches (data resident on device) for one survey config
so profilers see a clean stream of full_bls_no_sol / full_bls_batch
launches without allocation noise.

Usage:
  ncu --launch-skip 2 --launch-count 2 -k "regex:full_bls" --set full \
      python benchmarks/profile_bls_survey.py --survey Kepler --variant fast
  nsys profile -o rep python benchmarks/profile_bls_survey.py ...
"""
import argparse
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx  # noqa: F401

from cuvarbase.bls import eebls_gpu_fast, eebls_gpu_batch, BLSMemory
from bench_bls_survey import SURVEYS, make_lc, grid_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--survey', default='Kepler')
    ap.add_argument('--variant', default='fast',
                    choices=['fast', 'batch', 'naive'])
    ap.add_argument('--niter', type=int, default=4)
    ap.add_argument('--noverlap', type=int, default=2)
    ap.add_argument('--nlcs', type=int, default=2)
    ap.add_argument('--freq-stride', type=int, default=1,
                    help='subsample the freq grid (keeps the nbins mix) '
                         'so ncu kernel replay stays affordable')
    args = ap.parse_args()

    cfg = SURVEYS[args.survey]
    freqs, qmins, qmaxs = grid_for(cfg)
    if args.freq_stride > 1:
        freqs = freqs[::args.freq_stride].copy()
        qmins = qmins[::args.freq_stride].copy()
        qmaxs = qmaxs[::args.freq_stride].copy()
    print(f"{args.survey}: ndata={cfg['ndata']} nfreq={len(freqs)} "
          f"variant={args.variant}")

    if args.variant == 'fast':
        t, y, dy = make_lc(cfg, seed=12345)
        mem = BLSMemory(cfg['ndata'], len(freqs))
        mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs,
                    transfer=True)
        cuda.Context.synchronize()
        # one warmup (kernel compile happens here via cache)
        eebls_gpu_fast(t, y, dy, freqs, memory=mem,
                       transfer_to_device=False, transfer_to_host=False,
                       noverlap=args.noverlap)
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.niter):
            eebls_gpu_fast(t, y, dy, freqs, memory=mem,
                           transfer_to_device=False,
                           transfer_to_host=False,
                           noverlap=args.noverlap)
        cuda.Context.synchronize()
        print(f"per-iter: {(time.perf_counter()-t0)/args.niter*1e3:.1f} ms")
    elif args.variant == 'naive':
        # full public path incl. per-call allocations (host-overhead view)
        lcs = [make_lc(cfg, seed=1000 + i) for i in range(args.nlcs)]
        eebls_gpu_fast(*lcs[0], freqs, qmin=qmins, qmax=qmaxs,
                       noverlap=args.noverlap)
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.niter):
            for lc in lcs:
                eebls_gpu_fast(*lc, freqs, qmin=qmins, qmax=qmaxs,
                               noverlap=args.noverlap)
        cuda.Context.synchronize()
        n = args.niter * len(lcs)
        print(f"per-lc: {(time.perf_counter()-t0)/n*1e3:.1f} ms")
    else:
        lcs = [make_lc(cfg, seed=1000 + i) for i in range(args.nlcs)]
        eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs,
                        noverlap=args.noverlap)
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.niter):
            eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs,
                            noverlap=args.noverlap)
        cuda.Context.synchronize()
        print(f"per-iter: {(time.perf_counter()-t0)/args.niter*1e3:.1f} ms")


if __name__ == '__main__':
    main()
