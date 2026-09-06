#!/usr/bin/env python
"""Head-to-head benchmark: cuvarbase v1.0.0 (RC 2cc1f96) vs PyPI cuvarbase==0.2.6.

Version-agnostic: run the SAME script under each version's venv.
Implements the fairness rules of analysis/BENCHMARK_PROTOCOL_V1.md (section 4):

* identical seeded inputs (float64 host arrays; each version does its own cast)
* warm = steady-state with compile excluded on BOTH sides:
    - 0.2.6 gets precompiled ``functions=`` handles (its API supports this)
    - v1.0 uses its LRU kernel cache (>=2 discarded warmups on both sides)
* 0.2.6's fast path SILENTLY IGNORES noverlap (kernel arg unused in the
  compiled linear-bin branch), so the apples-to-apples v1.0 row is noverlap=1.
  v1.0 noverlap=2 (the default) is reported separately as a correctness
  improvement (~2x work: two dphi-shifted passes).
* cold = single fresh-process call including nvcc compile (clear the pycuda
  disk compiler cache BEFORE the process starts to make it a true cold start).
* median of >= 5 timed runs, explicit context synchronize inside each timing.

Modes
-----
warm         steady-state BLS timing (one JSON row per variant)
cold         fresh-process first-call + second-call BLS timing
loop         naive per-lightcurve loop (no functions= handle; what a naive
             pipeline pays), N distinct light curves, per-call times recorded
correctness  injected-transit periodograms at near-zero t and BJD-scale t
             (t + 2457000), noverlap=1; full periodograms stored in JSON
ls           Lomb-Scargle steady-state timing (process reused)

Usage: python bench_v026_head_to_head.py --mode warm --config canonical --out x.json
"""
from __future__ import print_function

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------

def get_config(name):
    """BLS benchmark configurations. freqs are k*df (k0=1) so they are also
    valid LS grids."""
    if name == 'canonical':
        # matches the 7-GPU campaign config (scripts/benchmark_algorithms.py):
        # 10-yr baseline, 10k obs, 5k freqs = k * (2.0/5000)
        return dict(ndata=10000, baseline=3652.5, nfreq=5000, fmax=2.0)
    if name == 'small':
        return dict(ndata=500, baseline=3652.5, nfreq=5000, fmax=2.0)
    if name == 'tess':
        # TESS-scale: 20k obs, 27.4-d sector, ~13.5k freqs up to P=0.5d
        return dict(ndata=20000, baseline=27.4, nfreq=13500, fmax=2.0)
    if name == 'correctness':
        return dict(ndata=3000, baseline=27.4, nfreq=7800, fmax=2.0)
    raise ValueError(name)


BLS_PARAMS = dict(qmin=0.01, qmax=0.5, dlogq=0.3)

# injected transit for correctness/BJD demo
INJ = dict(freq=1.0 / 3.456, q=0.03, depth=0.008)

BJD_OFFSET = 2457000.0


def make_freqs(cfg):
    df = cfg['fmax'] / cfg['nfreq']
    return (np.arange(1, cfg['nfreq'] + 1) * df).astype(np.float64)


def make_lc(ndata, baseline, seed, inject=None, t_offset=0.0):
    """Seeded light curve; float64 host arrays (each version casts itself)."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata))
    y = np.ones(ndata) + rng.randn(ndata) * 0.002
    dy = np.full(ndata, 0.002)
    if inject is not None:
        phase = (t * inject['freq']) % 1.0
        y[phase < inject['q']] -= inject['depth']
    return t + t_offset, y, dy


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

def env_info():
    import cuvarbase
    import pycuda
    import pycuda.driver as drv
    dev = _get_device()
    info = dict(
        cuvarbase=cuvarbase.__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        pycuda=getattr(pycuda, 'VERSION_TEXT', 'unknown'),
        cuda_driver_version=drv.get_driver_version(),
        gpu=dev.name(),
        compute_capability='%d.%d' % dev.compute_capability(),
        hostname=platform.node(),
    )
    try:
        out = subprocess.check_output(['nvcc', '--version'],
                                      stderr=subprocess.STDOUT)
        info['nvcc'] = out.decode().strip().splitlines()[-2].strip()
    except Exception as e:  # pragma: no cover
        info['nvcc'] = 'unavailable: %s' % e
    try:
        import skcuda
        info['scikit_cuda'] = skcuda.__version__
    except Exception:
        info['scikit_cuda'] = None
    return info


def _get_device():
    try:  # v1.0: lazy context helper
        from cuvarbase.base import ensure_context
        return ensure_context().device
    except Exception:
        pass
    import pycuda.autoprimaryctx  # 0.2.6: context made at cuvarbase import
    return pycuda.autoprimaryctx.device


def _sync():
    import pycuda.driver as drv
    drv.Context.synchronize()


def is_v026():
    import cuvarbase
    return cuvarbase.__version__.startswith('0.2')


# ----------------------------------------------------------------------------
# timing helper
# ----------------------------------------------------------------------------

def time_call(fn, n_warm=2, n_timed=7):
    for _ in range(n_warm):
        fn()
    _sync()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    times = sorted(times)
    med = float(np.median(times))
    iqr = [float(np.percentile(times, 25)), float(np.percentile(times, 75))]
    return med, iqr, times


# ----------------------------------------------------------------------------
# BLS variants
# ----------------------------------------------------------------------------

def bls_variants(mode):
    """Return list of (label, callable_factory) for this cuvarbase version.

    callable_factory(t, y, dy, freqs) -> zero-arg callable that runs one full
    eebls_gpu_fast call (H2D + kernels + D2H).
    """
    from cuvarbase import bls as cvb_bls

    variants = []

    if is_v026():
        if mode == 'warm':
            # fairness rule: precompiled handles for the baseline warm rows
            funcs = cvb_bls.compile_bls(function_names=['full_bls_no_sol'])

            def factory_warm(t, y, dy, freqs):
                def call():
                    return cvb_bls.eebls_gpu_fast(
                        t, y, dy, freqs, functions=funcs, **BLS_PARAMS)
                return call
            variants.append(('v026_fast_warm_precompiled', factory_warm))
        else:
            # naive product path: functions=None -> compile_bls every call
            def factory_naive(t, y, dy, freqs):
                def call():
                    return cvb_bls.eebls_gpu_fast(t, y, dy, freqs,
                                                  **BLS_PARAMS)
                return call
            variants.append(('v026_fast_naive', factory_naive))
        return variants

    # ---- v1.0 ----
    def factory_nov(noverlap):
        def factory(t, y, dy, freqs):
            def call():
                return cvb_bls.eebls_gpu_fast(t, y, dy, freqs,
                                              noverlap=noverlap, **BLS_PARAMS)
            return call
        return factory

    variants.append(('v10_fast_noverlap1', factory_nov(1)))
    variants.append(('v10_fast_noverlap2_default', factory_nov(2)))

    if mode == 'warm' and hasattr(cvb_bls, 'eebls_gpu_fast_optimized'):
        def factory_opt(t, y, dy, freqs):
            def call():
                return cvb_bls.eebls_gpu_fast_optimized(
                    t, y, dy, freqs, noverlap=1, **BLS_PARAMS)
            return call
        variants.append(('v10_fast_optimized_noverlap1', factory_opt))
    return variants


def run_warm(cfg, out):
    from cuvarbase import bls  # noqa: F401  (import before timing anything)
    freqs = make_freqs(cfg)
    t, y, dy = make_lc(cfg['ndata'], cfg['baseline'], seed=42, inject=INJ)

    rows = []
    for label, factory in bls_variants('warm'):
        call = factory(t, y, dy, freqs)
        med, iqr, times = time_call(call, n_warm=2, n_timed=7)
        rows.append(dict(label=label, median_s=med, iqr_s=iqr, times_s=times))
        print('  %-34s median %.4f s  IQR [%.4f, %.4f]'
              % (label, med, iqr[0], iqr[1]))
    out['rows'] = rows


def run_cold(cfg, out):
    """One fresh-process call including compile. Caller must have cleared
    ~/.cache/pycuda before starting this process for a true cold start."""
    freqs = make_freqs(cfg)
    t, y, dy = make_lc(cfg['ndata'], cfg['baseline'], seed=42, inject=INJ)

    t_imp0 = time.perf_counter()
    from cuvarbase import bls as cvb_bls
    _get_device()  # force context creation now; not part of call timing
    import_s = time.perf_counter() - t_imp0

    kwargs = dict(BLS_PARAMS)
    if not is_v026():
        kwargs['noverlap'] = 1

    t0 = time.perf_counter()
    cvb_bls.eebls_gpu_fast(t, y, dy, freqs, **kwargs)
    _sync()
    first_call_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cvb_bls.eebls_gpu_fast(t, y, dy, freqs, **kwargs)
    _sync()
    second_call_s = time.perf_counter() - t0

    out['rows'] = [dict(label=('v026_fast_naive' if is_v026()
                               else 'v10_fast_noverlap1'),
                        import_and_context_s=import_s,
                        first_call_s=first_call_s,
                        second_call_s=second_call_s)]
    print('  import+ctx %.3f s, first call %.3f s, second call %.3f s'
          % (import_s, first_call_s, second_call_s))


def run_loop(cfg, out, nlc=20):
    """Naive per-LC loop: fresh process, product defaults (no functions=).
    v1.0 noverlap=1 for apples-to-apples work per call."""
    from cuvarbase import bls as cvb_bls
    freqs = make_freqs(cfg)
    lcs = [make_lc(cfg['ndata'], cfg['baseline'], seed=100 + i, inject=INJ)
           for i in range(nlc)]

    kwargs = dict(BLS_PARAMS)
    if not is_v026():
        kwargs['noverlap'] = 1

    per_call = []
    t_loop0 = time.perf_counter()
    for (t, y, dy) in lcs:
        t0 = time.perf_counter()
        cvb_bls.eebls_gpu_fast(t, y, dy, freqs, **kwargs)
        _sync()
        per_call.append(time.perf_counter() - t0)
    loop_s = time.perf_counter() - t_loop0

    steady = float(np.median(per_call[1:]))
    out['rows'] = [dict(label=('v026_fast_naive_loop' if is_v026()
                               else 'v10_fast_noverlap1_loop'),
                        nlc=nlc, loop_total_s=loop_s,
                        per_call_s=per_call,
                        first_call_s=per_call[0],
                        steady_per_call_median_s=steady,
                        extrapolated_100lc_s=per_call[0] + 99 * steady)]
    print('  %d-LC loop: total %.3f s; first %.3f s; steady median %.4f s;'
          ' 100-LC extrapolation (first + 99*steady) %.2f s'
          % (nlc, loop_s, per_call[0], steady,
             per_call[0] + 99 * steady))


def run_correctness(cfg, out):
    """Injected transit at near-zero t and at BJD-scale t; noverlap=1 rows."""
    from cuvarbase import bls as cvb_bls
    freqs = make_freqs(cfg)

    kwargs = dict(BLS_PARAMS)
    if not is_v026():
        kwargs['noverlap'] = 1

    rows = []
    for tag, offset in (('near_zero', 0.0), ('bjd', BJD_OFFSET)):
        t, y, dy = make_lc(cfg['ndata'], cfg['baseline'], seed=7,
                           inject=INJ, t_offset=offset)
        power = np.asarray(
            cvb_bls.eebls_gpu_fast(t, y, dy, freqs, **kwargs), dtype=float)
        _sync()
        imax = int(np.argmax(power))
        i_inj = int(np.argmin(np.abs(freqs - INJ['freq'])))
        rows.append(dict(
            label=('v026' if is_v026() else 'v10_noverlap1') + '_' + tag,
            timescale=tag, t_offset=offset,
            injected_freq=INJ['freq'],
            peak_freq=float(freqs[imax]),
            peak_power=float(power[imax]),
            power_at_injected_freq=float(power[i_inj]),
            recovered=bool(abs(freqs[imax] - INJ['freq'])
                           < 5 * (freqs[1] - freqs[0])),
            periodogram=power.tolist()))
        print('  %-22s peak %.6f/d (inj %.6f/d) power %.5g  recovered=%s'
              % (rows[-1]['label'], freqs[imax], INJ['freq'],
                 power[imax], rows[-1]['recovered']))
    out['freqs'] = freqs.tolist()
    out['rows'] = rows


def run_ls(cfg_name, out):
    """Lomb-Scargle steady state, process reused (compile once)."""
    from cuvarbase.lombscargle import LombScargleAsyncProcess

    ls_configs = {
        'ls_large_grid': dict(ndata=3000, baseline=365.0, nfreq=100000,
                              fmax=None, df=1.0 / (4 * 365.0)),
        'ls_canonical': dict(ndata=10000, baseline=3652.5, nfreq=5000,
                             fmax=None, df=2.0 / 5000),
    }
    cfg = ls_configs[cfg_name]
    frq = (np.arange(1, cfg['nfreq'] + 1) * cfg['df']).astype(np.float64)
    t, y, dy = make_lc(cfg['ndata'], cfg['baseline'], seed=13)
    # add a sinusoid so the periodogram is non-trivial
    t64 = np.asarray(t)
    y = y + 0.005 * np.sin(2 * np.pi * 0.7431 * t64)

    proc = LombScargleAsyncProcess()

    def call():
        results = proc.run([(t, y, dy)], freqs=[frq])
        proc.finish()
        return results

    res = call()  # warmup + compile; also grab result for peak check
    fgrid, power = res[0]
    imax = int(np.argmax(power))

    med, iqr, times = time_call(call, n_warm=1, n_timed=7)
    out['rows'] = [dict(label='ls_' + ('v026' if is_v026() else 'v10'),
                        config=cfg, median_s=med, iqr_s=iqr, times_s=times,
                        peak_freq=float(np.asarray(fgrid)[imax]),
                        peak_power=float(np.asarray(power)[imax]))]
    print('  LS %-14s median %.4f s IQR [%.4f, %.4f]  peak %.4f/d'
          % (cfg_name, med, iqr[0], iqr[1], np.asarray(fgrid)[imax]))


def run_pdm(out):
    """PDM steady state, process reused. Legacy (t, y, w, freqs) data
    format (accepted by both versions); binned_linterp, nbins=10.
    v1.0 additionally reports the new *_fast kernel."""
    import warnings
    warnings.simplefilter('ignore')
    from cuvarbase.pdm import PDMAsyncProcess

    ndata, baseline, nfreq = 3000, 365.0, 10000
    df = 2.0 / nfreq
    frq = (np.arange(1, nfreq + 1) * df).astype(np.float64)
    t, y, dy = make_lc(ndata, baseline, seed=13)
    y = y + 0.005 * np.sin(2 * np.pi * 0.7431 * np.asarray(t))
    w = np.power(dy, -2.0)
    w /= w.sum()

    rows = []
    kinds = ['binned_linterp']
    if not is_v026():
        kinds.append('binned_linterp_fast')
    for kind in kinds:
        proc = PDMAsyncProcess()

        def call():
            r = proc.run([(np.asarray(t, dtype=np.float32),
                           np.asarray(y, dtype=np.float32),
                           np.asarray(w, dtype=np.float32),
                           np.asarray(frq, dtype=np.float32))],
                         kind=kind, nbins=10)
            proc.finish()
            return r

        res = call()
        power = np.asarray(res[0])
        imax = int(np.argmax(power))
        med, iqr, times = time_call(call, n_warm=1, n_timed=7)
        rows.append(dict(label='pdm_%s_%s' % (
                             'v026' if is_v026() else 'v10', kind),
                         ndata=ndata, nfreq=nfreq, kind=kind,
                         median_s=med, iqr_s=iqr, times_s=times,
                         peak_freq=float(frq[imax]),
                         peak_power=float(power[imax])))
        print('  PDM %-22s median %.4f s IQR [%.4f, %.4f] peak %.4f/d'
              % (kind, med, iqr[0], iqr[1], frq[imax]))
    out['rows'] = rows


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['warm', 'cold', 'loop', 'correctness', 'ls',
                            'pdm'])
    p.add_argument('--config', default='canonical')
    p.add_argument('--nlc', type=int, default=20)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = dict(mode=args.mode, config_name=args.config,
               bls_params=BLS_PARAMS, injection=INJ,
               timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))

    if args.mode == 'ls':
        out['env'] = None  # filled after import inside run_ls path
        run_ls(args.config, out)
    elif args.mode == 'pdm':
        run_pdm(out)
    else:
        cfg = get_config(args.config if args.mode != 'correctness'
                         else 'correctness')
        out['config'] = cfg
        out['freq_grid'] = dict(df=cfg['fmax'] / cfg['nfreq'],
                                nfreq=cfg['nfreq'], k0=1)
        if args.mode == 'warm':
            run_warm(cfg, out)
        elif args.mode == 'cold':
            run_cold(cfg, out)
        elif args.mode == 'loop':
            run_loop(cfg, out, nlc=args.nlc)
        elif args.mode == 'correctness':
            run_correctness(cfg, out)

    out['env'] = env_info()
    print(json.dumps(out['env'], indent=2))

    with open(args.out, 'w') as f:
        json.dump(out, f)
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
