#!/usr/bin/env python3
"""
Microbenchmark for punchlist #2 item A4: is the ndata-only
_choose_block_size heuristic within ~10% of the best block size once
the number of phase bins (driven by qmin) is taken into account?

Sweeps (ndata, qmin, block_size) on the fast BLS kernels with a
preallocated/pretransferred BLSMemory so the timing isolates kernel
execution (no alloc/H2D/D2H inside the timed region). For each
(ndata, qmin) cell it reports the per-block-size median time, the
heuristic's choice, the empirically best choice, and the penalty
ratio time[heuristic] / time[best].

Run on the pod:
    python scripts/benchmark_block_size.py            # both kernels
    python scripts/benchmark_block_size.py --quick    # smaller grid
"""
import argparse
import json
import time

import numpy as np

import pycuda.autoprimaryctx
from cuvarbase.bls import (BLSMemory, _choose_block_size,
                           eebls_gpu_fast, eebls_gpu_fast_optimized)

BLOCK_SIZES = [32, 64, 128, 256, 512]
NDATA_GRID = [50, 200, 1000, 5000, 20000]
QMIN_GRID = [1e-3, 5e-3, 2e-2, 1e-1]
NFREQ = 2000
NTRIALS = 7


def generate_data(ndata, seed=42, baseline=100.0):
    rand = np.random.RandomState(seed)
    t = np.sort(rand.uniform(0, baseline, ndata))
    y = np.ones(ndata)
    phase = (t % 5.0) / 5.0
    y[(phase > 0.4) & (phase < 0.5)] -= 0.01
    y += 0.01 * rand.randn(ndata)
    dy = 0.01 * np.ones(ndata)
    return t, y, dy


def time_cell(fn, t, y, dy, freqs, qmin, block_size, ntrials=NTRIALS):
    """Median kernel-only wall time for one (data, qmin, block) cell."""
    mem = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=0.5,
                             freqs=freqs, transfer=True)
    kw = dict(qmin=qmin, qmax=0.5, memory=mem, noverlap=1,
              transfer_to_device=False, transfer_to_host=False,
              block_size=block_size)

    # warm-up: compile + cache the kernel for this block size
    fn(t, y, dy, freqs, **kw)
    pycuda.autoprimaryctx.context.synchronize()

    times = []
    for _ in range(ntrials):
        t0 = time.perf_counter()
        fn(t, y, dy, freqs, **kw)
        pycuda.autoprimaryctx.context.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def sweep(kernel, ndata_grid, qmin_grid, block_sizes, ntrials):
    fn = (eebls_gpu_fast_optimized if kernel == 'optimized'
          else eebls_gpu_fast)
    cells = []
    for ndata in ndata_grid:
        t, y, dy = generate_data(ndata)
        freqs = np.linspace(0.1, 2.0, NFREQ)
        for qmin in qmin_grid:
            timings = {}
            for bs in block_sizes:
                timings[str(bs)] = time_cell(fn, t, y, dy, freqs,
                                             qmin, bs, ntrials)
            heur_bs = _choose_block_size(ndata)
            best_bs = min(timings, key=timings.get)
            penalty = timings[str(heur_bs)] / timings[best_bs]
            cell = dict(ndata=ndata, qmin=qmin,
                        nbins=int(np.ceil(1.0 / qmin)),
                        timings_s=timings,
                        heuristic_block_size=heur_bs,
                        best_block_size=int(best_bs),
                        penalty=round(penalty, 4))
            cells.append(cell)
            print("%s ndata=%-6d qmin=%-7g heur=%-4d best=%-4s "
                  "penalty=%.3f" % (kernel, ndata, qmin, heur_bs,
                                    best_bs, penalty))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smaller grid / fewer trials')
    ap.add_argument('--kernels', nargs='+',
                    default=['standard', 'optimized'],
                    choices=['standard', 'optimized'])
    ap.add_argument('--output', default='benchmark_block_size.json')
    args = ap.parse_args()

    ndata_grid = [200, 5000] if args.quick else NDATA_GRID
    qmin_grid = [1e-3, 1e-1] if args.quick else QMIN_GRID
    ntrials = 3 if args.quick else NTRIALS

    device = pycuda.autoprimaryctx.device.name()
    results = dict(device=device,
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
                   nfreq=NFREQ, ntrials=ntrials,
                   block_sizes=BLOCK_SIZES, kernels={})

    for kernel in args.kernels:
        results['kernels'][kernel] = sweep(kernel, ndata_grid,
                                           qmin_grid, BLOCK_SIZES,
                                           ntrials)

    penalties = [c['penalty'] for k in results['kernels'].values()
                 for c in k]
    results['summary'] = dict(
        max_penalty=max(penalties),
        median_penalty=float(np.median(penalties)),
        cells_over_10pct=[
            dict(kernel=k, ndata=c['ndata'], qmin=c['qmin'],
                 penalty=c['penalty'])
            for k, cs in results['kernels'].items()
            for c in cs if c['penalty'] > 1.10])

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\ndevice: %s" % device)
    print("max penalty:    %.3f" % results['summary']['max_penalty'])
    print("median penalty: %.3f" % results['summary']['median_penalty'])
    n_bad = len(results['summary']['cells_over_10pct'])
    print("cells > 10%% over best: %d" % n_bad)
    print("wrote %s" % args.output)


if __name__ == '__main__':
    main()
