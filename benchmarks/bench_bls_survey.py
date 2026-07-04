#!/usr/bin/env python3
"""
Survey-scale BLS benchmark: end-to-end wall clock + decomposition.

Measures eebls_gpu_fast (naive per-call, memory-reuse, kernel-only) and
eebls_gpu_batch at four realistic survey scales on Keplerian frequency
grids (qmin = 0.5 q_kep, qmax = 2 q_kep, matching eebls_transit_gpu
defaults):

  ZTF     : 150 obs   x ~60K  freqs
  HAT-Net : 6K  obs   x ~301K freqs
  TESS    : 20K obs   x ~1.8K freqs
  Kepler  : 65K obs   x ~131K freqs

Variants
--------
fast_naive  : eebls_gpu_fast, fresh BLSMemory every call (public default path)
fast_reuse  : eebls_gpu_fast with a persistent BLSMemory (steady-state)
kernel      : kernel launches only (data resident, no H2D/D2H), noverlap=2
kernel_1pass: same but noverlap=1 (isolates the per-pass cost)
batch       : eebls_gpu_batch over the whole LC list (as-is, incl. its
              per-call BLSBatchMemory allocation)

Output: JSON (+ optional parity .npz) under
benchmarks/results/bls_survey_speed_jul2026/raw/

Timing discipline: warm-cache medians over >= --runs runs; the cold
(first-call) number is recorded separately.
"""
import argparse
import json
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

import pycuda.driver as cuda   # noqa: E402
import pycuda.autoprimaryctx   # noqa: F401,E402

from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu_batch, BLSMemory)
from cuvarbase.bls_frequencies import keplerian_freq_grid

RESULTS_DIR = Path(__file__).parent / 'results' / 'bls_survey_speed_jul2026'
POD_USD_PER_HR = 0.27  # RTX A5000 on-demand

SURVEYS = OrderedDict([
    ('ZTF', dict(ndata=150, baseline=730.0, period_min=0.5,
                 period_max=100.0, cadence=None)),
    ('HAT-Net', dict(ndata=6000, baseline=3650.0, period_min=0.5,
                     period_max=100.0, cadence=None)),
    ('TESS', dict(ndata=20000, baseline=27.0, period_min=0.5,
                  period_max=13.5, cadence=None)),
    ('Kepler', dict(ndata=65000, baseline=1460.0, period_min=0.5,
                    period_max=500.0, cadence=None)),
])

# per-survey loop sizes (kept small for the heavy configs; medians are
# still over >= 5 runs of the whole loop)
DEFAULT_NLCS = {'ZTF': 10, 'HAT-Net': 4, 'TESS': 10, 'Kepler': 2}


def make_lc(cfg, seed, bjd=False):
    rng = np.random.RandomState(seed)
    ndata, baseline = cfg['ndata'], cfg['baseline']
    t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float64)
    period, q0, depth = 2.5271, 0.035, 0.01
    phase = (t % period) / period
    y = np.ones(ndata)
    y[phase < q0] -= depth
    y += 0.002 * rng.randn(ndata)
    dy = np.full(ndata, 0.002)
    if bjd:
        t = t + 2455197.5
    return t, y, dy


def grid_for(cfg):
    freqs, qvals = keplerian_freq_grid(cfg['period_min'], cfg['period_max'],
                                       cfg['baseline'], oversampling=2,
                                       return_qvals=True)
    qmins = 0.5 * qvals
    qmaxs = 2.0 * qvals
    return freqs.astype(np.float64), qmins, qmaxs


def sync():
    cuda.Context.synchronize()


def timed(fn, runs, warmup=1):
    """Return (cold_s, warm_median_s, all_warm)."""
    cold = None
    for i in range(warmup):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        dt = time.perf_counter() - t0
        if i == 0:
            cold = dt
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return cold, float(np.median(times)), times


def bench_survey(name, cfg, n_lcs, runs, variants, noverlap=2):
    freqs, qmins, qmaxs = grid_for(cfg)
    nfreq = len(freqs)
    print(f"\n=== {name}: ndata={cfg['ndata']}, nfreq={nfreq} "
          f"(n_lcs={n_lcs}, runs={runs}) ===", flush=True)

    lcs = [make_lc(cfg, seed=1000 + i) for i in range(n_lcs)]
    out = dict(ndata=cfg['ndata'], nfreq=nfreq, n_lcs=n_lcs, runs=runs,
               noverlap=noverlap, variants={})

    # ---------------- fast_naive: fresh memory per call ----------------
    if 'fast_naive' in variants:
        def run_naive():
            for (t, y, dy) in lcs:
                eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs,
                               noverlap=noverlap)
        cold, med, all_t = timed(run_naive, runs)
        out['variants']['fast_naive'] = dict(
            cold_total_s=cold, warm_median_total_s=med, all_s=all_t,
            per_lc_s=med / n_lcs)
        print(f"  fast_naive : {med/n_lcs*1e3:9.2f} ms/lc "
              f"(cold total {cold:.3f}s)", flush=True)

    # ---------------- fast_reuse: persistent BLSMemory -----------------
    mem = None
    if 'fast_reuse' in variants or 'kernel' in variants \
            or 'kernel_1pass' in variants:
        mem = BLSMemory(cfg['ndata'], nfreq)
        t0, y0, dy0 = lcs[0]
        # first call sets freqs + nbins & allocates GPU arrays
        mem.setdata(t0, y0, dy0, qmin=qmins, qmax=qmaxs, freqs=freqs,
                    transfer=True)
        sync()

    if 'fast_reuse' in variants:
        def run_reuse():
            for (t, y, dy) in lcs:
                mem.setdata(t, y, dy, freqs=None, transfer=True)
                eebls_gpu_fast(t, y, dy, freqs, memory=mem,
                               transfer_to_device=False,
                               noverlap=noverlap)
        cold, med, all_t = timed(run_reuse, runs)
        out['variants']['fast_reuse'] = dict(
            cold_total_s=cold, warm_median_total_s=med, all_s=all_t,
            per_lc_s=med / n_lcs)
        print(f"  fast_reuse : {med/n_lcs*1e3:9.2f} ms/lc", flush=True)

    # ---------------- kernel only (data resident) ----------------------
    for vname, nov in (('kernel', noverlap), ('kernel_1pass', 1)):
        if vname not in variants:
            continue
        t0, y0, dy0 = lcs[0]
        mem.setdata(t0, y0, dy0, freqs=None, transfer=True)
        sync()

        def run_kernel():
            eebls_gpu_fast(t0, y0, dy0, freqs, memory=mem,
                           transfer_to_device=False,
                           transfer_to_host=False, noverlap=nov)
        cold, med, all_t = timed(run_kernel, runs)
        out['variants'][vname] = dict(
            cold_s=cold, warm_median_s=med, all_s=all_t, per_lc_s=med)
        print(f"  {vname:11s}: {med*1e3:9.2f} ms/lc", flush=True)

    # ---------------- decomposition pieces ------------------------------
    if 'pieces' in variants:
        t0, y0, dy0 = lcs[0]
        # host-side conversion + H2D (no freq transfer)
        def run_setdata():
            mem.setdata(t0, y0, dy0, freqs=None, transfer=True)
        _, med_sd, _ = timed(run_setdata, runs)
        # D2H + normalize
        def run_d2h():
            mem.transfer_data_to_cpu()
        _, med_d2h, _ = timed(run_d2h, runs)
        # fresh BLSMemory construction (pinned-host allocs)
        def run_alloc():
            m = BLSMemory(cfg['ndata'], nfreq)
            m.allocate_data(cfg['ndata'])
            m.allocate_freqs(nfreq)
        cold_al, med_al, _ = timed(run_alloc, max(3, runs // 2))
        out['variants']['pieces'] = dict(
            setdata_h2d_s=med_sd, d2h_norm_s=med_d2h, alloc_s=med_al,
            alloc_cold_s=cold_al)
        print(f"  pieces     : setdata+h2d {med_sd*1e3:.2f} ms, "
              f"d2h+norm {med_d2h*1e3:.2f} ms, alloc {med_al*1e3:.2f} ms",
              flush=True)

    # ---------------- batch ---------------------------------------------
    if 'batch' in variants:
        def run_batch():
            eebls_gpu_batch(lcs, freqs, qmin=qmins, qmax=qmaxs,
                            noverlap=noverlap)
        cold, med, all_t = timed(run_batch, runs)
        out['variants']['batch'] = dict(
            cold_total_s=cold, warm_median_total_s=med, all_s=all_t,
            per_lc_s=med / n_lcs)
        print(f"  batch      : {med/n_lcs*1e3:9.2f} ms/lc "
              f"(cold total {cold:.3f}s)", flush=True)

    # $/lightcurve for whatever variants we have
    for v, d in out['variants'].items():
        if 'per_lc_s' in d:
            d['usd_per_million_lc'] = (d['per_lc_s'] / 3600.0) \
                * POD_USD_PER_HR * 1e6
    if mem is not None:
        del mem
    return out


def dump_parity(tag, surveys, noverlap=2):
    """Save reference periodograms for before/after parity checks."""
    pdir = RESULTS_DIR / 'raw' / 'parity'
    pdir.mkdir(parents=True, exist_ok=True)
    for name in surveys:
        cfg = SURVEYS[name]
        freqs, qmins, qmaxs = grid_for(cfg)
        t, y, dy = make_lc(cfg, seed=12345)
        p_fast = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs,
                                noverlap=noverlap)
        tb, yb, dyb = make_lc(cfg, seed=12345, bjd=True)
        p_bjd = eebls_gpu_fast(tb, yb, dyb, freqs, qmin=qmins, qmax=qmaxs,
                               noverlap=noverlap)
        p_batch = eebls_gpu_batch([(t, y, dy)], freqs, qmin=qmins,
                                  qmax=qmaxs, noverlap=noverlap)[0]
        fn = pdir / f'parity_{name.replace("-", "")}_{tag}.npz'
        np.savez_compressed(fn, freqs=freqs.astype(np.float32),
                            fast=p_fast.astype(np.float32),
                            fast_bjd=p_bjd.astype(np.float32),
                            batch=p_batch.astype(np.float32))
        print(f"  parity dump: {fn} "
              f"(peak fast @ {freqs[np.argmax(p_fast)]:.6f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surveys', nargs='+', default=list(SURVEYS.keys()))
    ap.add_argument('--variants', nargs='+',
                    default=['fast_naive', 'fast_reuse', 'kernel',
                             'kernel_1pass', 'pieces', 'batch'])
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--nlcs', type=int, default=None)
    ap.add_argument('--noverlap', type=int, default=2)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--parity', action='store_true')
    args = ap.parse_args()

    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent.parent).decode().strip()
    except Exception:
        sha = 'unknown'

    dev = cuda.Context.get_device()
    meta = dict(gpu=dev.name(), git_sha=sha, tag=args.tag,
                noverlap=args.noverlap,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                pod_usd_per_hr=POD_USD_PER_HR)
    print(f"GPU: {meta['gpu']}  sha={sha}  tag={args.tag}")

    results = dict(meta=meta, surveys={})
    for name in args.surveys:
        cfg = SURVEYS[name]
        n_lcs = args.nlcs or DEFAULT_NLCS[name]
        results['surveys'][name] = bench_survey(
            name, cfg, n_lcs, args.runs, args.variants,
            noverlap=args.noverlap)

    outdir = RESULTS_DIR / 'raw'
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f'bench_{args.tag}.json'
    with open(fn, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {fn}")

    if args.parity:
        dump_parity(args.tag, args.surveys, noverlap=args.noverlap)


if __name__ == '__main__':
    main()
