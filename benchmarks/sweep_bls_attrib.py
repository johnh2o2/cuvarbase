#!/usr/bin/env python3
"""
Empirical kernel-time attribution for the fast BLS kernel WITHOUT GPU
performance counters (RunPod blocks them): vary one work axis at a
time and read the marginal costs off the slopes.

Sweeps (kernel-only, data resident):
  A) ndata sweep at fixed grid   -> d(t)/d(ndata) = fold+histogram cost
  B) bin-scale sweep (qmin,qmax scaled by 1/k) at fixed ndata
                                 -> d(t)/d(scan work) = box-scan cost
  C) noverlap 1 vs 2             -> per-pass multiplier
  D) block_size sweep            -> occupancy sensitivity

Writes JSON to benchmarks/results/bls_survey_speed_jul2026/raw/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx  # noqa: F401

from cuvarbase.bls import eebls_gpu_fast, BLSMemory
from bench_bls_survey import SURVEYS, make_lc, grid_for, RESULTS_DIR


def ktime(t, y, dy, freqs, qmins, qmaxs, noverlap=2, runs=5,
          block_size=None):
    kw = {}
    if block_size:
        kw['block_size'] = block_size
    mem = BLSMemory(len(t), len(freqs))
    mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs,
                transfer=True)
    cuda.Context.synchronize()
    eebls_gpu_fast(t, y, dy, freqs, memory=mem, transfer_to_device=False,
                   transfer_to_host=False, noverlap=noverlap, **kw)
    cuda.Context.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        eebls_gpu_fast(t, y, dy, freqs, memory=mem,
                       transfer_to_device=False, transfer_to_host=False,
                       noverlap=noverlap, **kw)
        cuda.Context.synchronize()
        times.append(time.perf_counter() - t0)
    del mem
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--survey', default='HAT-Net')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--freq-stride', type=int, default=1)
    args = ap.parse_args()

    cfg = dict(SURVEYS[args.survey])
    freqs, qmins, qmaxs = grid_for(cfg)
    if args.freq_stride > 1:
        freqs = freqs[::args.freq_stride].copy()
        qmins = qmins[::args.freq_stride].copy()
        qmaxs = qmaxs[::args.freq_stride].copy()
    res = dict(survey=args.survey, nfreq=len(freqs),
               freq_stride=args.freq_stride, sweeps={})
    print(f"survey={args.survey} nfreq={len(freqs)}")

    # A) ndata sweep
    nds = [150, 600, 2400, 9600, 38400]
    sweep = []
    for nd in nds:
        c = dict(cfg)
        c['ndata'] = nd
        t, y, dy = make_lc(c, seed=7)
        s = ktime(t, y, dy, freqs, qmins, qmaxs, runs=args.runs)
        sweep.append(dict(ndata=nd, s=s))
        print(f"  A ndata={nd:6d}: {s*1e3:8.2f} ms")
    res['sweeps']['ndata'] = sweep

    # B) bin-scale sweep at ndata from config
    t, y, dy = make_lc(cfg, seed=7)
    sweep = []
    for k in (1.0, 2.0, 4.0):
        qmn = np.maximum(qmins / k, 2.5e-4)  # shared-mem guard
        s = ktime(t, y, dy, freqs, qmn, qmaxs, runs=args.runs)
        nbf_max = int(1.0 / qmn.min())
        sweep.append(dict(bin_scale=k, nbf_max=nbf_max, s=s))
        print(f"  B bin_scale={k}: nbf_max={nbf_max} {s*1e3:8.2f} ms")
    res['sweeps']['bins'] = sweep

    # C) noverlap sweep
    sweep = []
    for nov in (1, 2, 3):
        s = ktime(t, y, dy, freqs, qmins, qmaxs, noverlap=nov,
                  runs=args.runs)
        sweep.append(dict(noverlap=nov, s=s))
        print(f"  C noverlap={nov}: {s*1e3:8.2f} ms")
    res['sweeps']['noverlap'] = sweep

    # D) block size
    sweep = []
    for bs in (64, 128, 256, 512):
        try:
            s = ktime(t, y, dy, freqs, qmins, qmaxs, runs=args.runs,
                      block_size=bs)
            sweep.append(dict(block_size=bs, s=s))
            print(f"  D block={bs:4d}: {s*1e3:8.2f} ms")
        except Exception as e:
            print(f"  D block={bs:4d}: failed ({e})")
    res['sweeps']['block_size'] = sweep

    outdir = RESULTS_DIR / 'raw'
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f'attrib_{args.survey.replace("-", "")}.json'
    with open(fn, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"wrote {fn}")


if __name__ == '__main__':
    main()
