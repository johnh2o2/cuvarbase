#!/usr/bin/env python
"""Survey-scale TLS throughput benchmark (end-to-end, GPU).

Measures wall time per lightcurve (grid generation + preprocessing + H2D +
kernel + D2H + statistics) for N lightcurves per survey regime, comparing up
to three implementations (each degrades gracefully if unavailable):

  new        cuvarbase.tls.tls_search_batch (batch API)
  old        cuvarbase.tls.tls_transit looped per LC (ndata <= 3300 only;
             capped at --old-nlc LCs, per-LC median extrapolated)
  reference  CPU transitleastsquares (--ref-nlc LCs; kepler-4yr skipped
             unless --ref-all)

Half the lightcurves carry an injected box transit (Keplerian duration,
Sun-like), half are pure noise; recovery + median SDE reported per half.

Usage (RunPod pod):
    ./scripts/run-remote.sh python scripts/benchmark_tls_survey.py \\
        [--regimes tess-ffi,k2] [--impls new,old,reference] [--quick]

Output: JSON via --output plus a human-readable summary table.
"""

import argparse
import json
import multiprocessing
import platform
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

OLD_NDATA_CAP = 3300  # old per-LC kernel's shared-memory cap on ndata
REF_SKIP_DEFAULT = ('kepler-4yr',)  # CPU ref >> 15 min; needs --ref-all

# ----------------------------------------------------------------------------
# Survey regimes (period ranges chosen for comparability with the reference
# TLS paper / GTLS 2026 paper). cadence in days; noise/depth fractional flux.
# ----------------------------------------------------------------------------
MIN30 = 30.0 / (60.0 * 24.0)
MIN2 = 2.0 / (60.0 * 24.0)

REGIMES = OrderedDict([
    ('tess-ffi', dict(ndata=1310, baseline=27.4, cadence=MIN30, noise=1e-3,
                      inject_period=7.7, inject_depth=0.005,
                      period_min=0.6, period_max=13.7, nlc=100)),
    ('k2', dict(ndata=4320, baseline=90.0, cadence=MIN30, noise=8e-4,
                inject_period=12.4, inject_depth=0.004,
                period_min=0.6, period_max=45.0, nlc=50)),
    ('tess-2min', dict(ndata=19710, baseline=27.4, cadence=MIN2, noise=2e-3,
                       inject_period=7.7, inject_depth=0.005,
                       period_min=0.6, period_max=13.7, nlc=50)),
    ('tess-yr', dict(ndata=16850, baseline=351.0, cadence=MIN30, noise=1e-3,
                     inject_period=21.7, inject_depth=0.004,
                     period_min=0.6, period_max=175.0, nlc=20)),
    ('kepler-4yr', dict(ndata=65440, baseline=1363.0, cadence=MIN30,
                        noise=6e-4, inject_period=41.3, inject_depth=0.003,
                        period_min=0.6, period_max=500.0, nlc=10)),
])


# ----------------------------------------------------------------------------
# Lightcurve generation
# ----------------------------------------------------------------------------

def make_lc(cfg, seed, inject):
    """Regular-cadence LC, flux ~1.0, optional box transit at t0 = 0.3 * P
    with Keplerian duration q = 0.0763 * P^(-2/3) (fraction of period,
    Sun-like). A box (not limb-darkened) is fine: recovery is on period."""
    rng = np.random.default_rng(seed)
    t = np.arange(cfg['ndata'], dtype=np.float64) * cfg['cadence']
    y = 1.0 + rng.normal(0.0, cfg['noise'], cfg['ndata'])
    if inject:
        P = cfg['inject_period']
        q = 0.0763 * P ** (-2.0 / 3.0)
        t0 = 0.3 * P
        in_transit = np.abs(((t - t0 + 0.5 * P) % P) - 0.5 * P) < 0.5 * q * P
        y[in_transit] -= cfg['inject_depth']
    dy = np.full(cfg['ndata'], cfg['noise'])
    return t, y, dy


def make_regime_lcs(key, cfg, nlc):
    """~Half injected, half pure noise; seeded per (regime, lc_index)."""
    regime_idx = list(REGIMES).index(key)
    flags = [i % 2 == 0 for i in range(nlc)]
    lcs = [make_lc(cfg, 100000 * (regime_idx + 1) + i, flags[i])
           for i in range(nlc)]
    return lcs, flags


# ----------------------------------------------------------------------------
# GPU / environment helpers (imports deferred so --help works anywhere)
# ----------------------------------------------------------------------------

def gpu_sync():
    try:
        import pycuda.driver as drv
        drv.Context.synchronize()
    except Exception:
        pass


def _get_device():
    try:  # v1.0 lazy context helper
        from cuvarbase.base import ensure_context
        return ensure_context().device
    except Exception:
        import pycuda.autoprimaryctx
        return pycuda.autoprimaryctx.device


def env_info():
    info = dict(python=platform.python_version(), numpy=np.__version__,
                hostname=platform.node(),
                cpu_count=multiprocessing.cpu_count())
    try:
        import cuvarbase
        info['cuvarbase'] = cuvarbase.__version__
    except Exception as e:
        info['cuvarbase'] = 'unavailable: %s' % e
    try:
        import pycuda
        import pycuda.driver as drv
        dev = _get_device()
        info['pycuda'] = getattr(pycuda, 'VERSION_TEXT', 'unknown')
        info['cuda_driver_version'] = drv.get_driver_version()
        info['gpu'] = dev.name()
        info['compute_capability'] = '%d.%d' % dev.compute_capability()
    except Exception as e:
        info['gpu'] = 'unavailable: %s' % e
    try:
        out = subprocess.check_output(['nvcc', '--version'],
                                      stderr=subprocess.STDOUT)
        info['nvcc'] = out.decode().strip().splitlines()[-2].strip()
    except Exception as e:
        info['nvcc'] = 'unavailable: %s' % e
    return info


def probe_nperiods(cfg):
    """Number of Ofir-grid periods this regime's search covers."""
    try:
        from cuvarbase import tls_grids
        t = np.arange(cfg['ndata'], dtype=np.float64) * cfg['cadence']
        periods = tls_grids.period_grid_ofir(
            t, R_star=1.0, M_star=1.0, oversampling_factor=3,
            period_min=cfg['period_min'], period_max=cfg['period_max'])
        return int(len(periods))
    except Exception as e:
        print('  nperiods probe failed: %r' % e)
        return None


# ----------------------------------------------------------------------------
# Recovery / statistics
# ----------------------------------------------------------------------------

def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def eval_recovery(results, flags, inject_period):
    """Recovery on the injected half; |P/P_inj - 1| < 0.01 counts as
    recovered, 2x / 0.5x aliases (1% relative) counted separately."""
    n_inj = n_rec = n_alias = 0
    sde_inj, sde_noise = [], []
    for res, injected in zip(results, flags):
        res = res or {}
        sde = _f(res.get('SDE'))
        if injected:
            n_inj += 1
            if sde is not None:
                sde_inj.append(sde)
            p = _f(res.get('period'))
            if p:
                r = p / inject_period
                if abs(r - 1.0) < 0.01:
                    n_rec += 1
                elif abs(r / 2.0 - 1.0) < 0.01 or abs(2.0 * r - 1.0) < 0.01:
                    n_alias += 1
        elif sde is not None:
            sde_noise.append(sde)
    return dict(
        n_injected=n_inj, n_recovered=n_rec, n_alias=n_alias,
        recovery_frac=(n_rec / n_inj) if n_inj else None,
        median_sde_injected=float(np.median(sde_inj)) if sde_inj else None,
        median_sde_noise=float(np.median(sde_noise)) if sde_noise else None)


def compact_per_lc(results, flags):
    out = []
    for res, injected in zip(results, flags):
        res = res or {}
        rec = dict(injected=bool(injected))
        for k in ('period', 'T0', 'duration', 'depth', 'SDE', 'chi2_min'):
            rec[k] = _f(res.get(k))
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Implementations
# ----------------------------------------------------------------------------

def _warmup_new(tls_search_batch, cfg):
    """A 2-LC batch with the REGIME's own config so every phase-bin
    band variant this regime needs is compiled before timing (band
    structure depends on the period range)."""
    print('  [new] warmup: 2-LC regime batch (absorbs compile)...',
          flush=True)
    wlcs = [make_lc(cfg, 900 + i, inject=True) for i in range(2)]
    tls_search_batch(wlcs, R_star=1.0, M_star=1.0,
                     period_min=cfg['period_min'],
                     period_max=cfg['period_max'],
                     oversampling_factor=3, n_durations=15,
                     t0_oversample=3.0,
                     block_size=None, nbins=None, return_arrays=False)
    gpu_sync()


def run_new(cfg, lcs, flags, args, state):
    entry = dict(nlc=len(lcs))
    try:
        from cuvarbase.tls import tls_search_batch
    except Exception as e:
        traceback.print_exc()
        entry['error'] = 'import failed: %r' % e
        return entry
    try:
        warm_key = 'new_warmed_%s_%s' % (cfg['period_min'],
                                          cfg['period_max'])
        if not state.get(warm_key):
            _warmup_new(tls_search_batch, cfg)
            state[warm_key] = True
        print('  [new] timing %d-LC batch (x%d iter)...'
              % (len(lcs), args.n_iter), flush=True)
        times, results = [], None
        for _ in range(args.n_iter):
            gpu_sync()
            t0 = time.perf_counter()
            results = tls_search_batch(
                lcs, R_star=1.0, M_star=1.0,
                period_min=cfg['period_min'], period_max=cfg['period_max'],
                oversampling_factor=3, n_durations=15, t0_oversample=3.0,
                block_size=None, nbins=None,
                return_arrays=False)
            gpu_sync()
            times.append(time.perf_counter() - t0)
        total = float(np.median(times))
        entry.update(total_s=total, times_s=times,
                     ms_per_lc=1000.0 * total / len(lcs),
                     lc_per_s=len(lcs) / total)
        entry.update(eval_recovery(results, flags, cfg['inject_period']))
        entry['per_lc'] = compact_per_lc(results, flags)
    except Exception as e:
        traceback.print_exc()
        entry['error'] = repr(e)
    return entry


def run_old(cfg, lcs, flags, args):
    entry = dict()
    if cfg['ndata'] > OLD_NDATA_CAP:
        entry['skipped'] = 'ndata cap'
        print('  [old] skipped: ndata=%d > %d (shared-memory cap)'
              % (cfg['ndata'], OLD_NDATA_CAP))
        return entry
    try:
        from cuvarbase.tls import tls_transit
    except Exception as e:
        traceback.print_exc()
        entry['error'] = 'import failed: %r' % e
        return entry
    n_old = min(args.old_nlc, len(lcs))
    sub, subflags = lcs[:n_old], flags[:n_old]
    kwargs = dict(R_star=1.0, M_star=1.0, period_min=cfg['period_min'],
                  period_max=cfg['period_max'], use_fast=False)
    try:
        print('  [old] warmup (1 LC, absorbs compile)...', flush=True)
        tls_transit(*sub[0], **kwargs)
        gpu_sync()
        print('  [old] timing %d LCs (per-LC loop)...' % n_old, flush=True)
        per_call, results = [], []
        for (t, y, dy) in sub:
            gpu_sync()
            t0 = time.perf_counter()
            results.append(tls_transit(t, y, dy, **kwargs))
            gpu_sync()
            per_call.append(time.perf_counter() - t0)
        med = float(np.median(per_call))
        entry.update(nlc=n_old, total_s=float(np.sum(per_call)),
                     per_call_s=per_call, ms_per_lc=1000.0 * med,
                     lc_per_s=1.0 / med, extrapolated=True,
                     note='per-LC median over %d LCs' % n_old)
        entry.update(eval_recovery(results, subflags, cfg['inject_period']))
        entry['per_lc'] = compact_per_lc(results, subflags)
    except Exception as e:
        traceback.print_exc()
        entry['error'] = repr(e)
    return entry


def run_reference(key, cfg, lcs, flags, args):
    entry = dict()
    if key in REF_SKIP_DEFAULT and not args.ref_all:
        entry['skipped'] = ('expected CPU runtime >~15 min; '
                            'pass --ref-all to run')
        print('  [reference] skipped: %s' % entry['skipped'])
        return entry
    try:
        from transitleastsquares import transitleastsquares
    except Exception as e:
        entry['error'] = 'import failed: %r' % e
        print('  [reference] %s' % entry['error'])
        return entry
    n_ref = min(args.ref_nlc, len(lcs))
    sub, subflags = lcs[:n_ref], flags[:n_ref]
    ncpu = multiprocessing.cpu_count()
    try:
        print('  [reference] timing %d LCs (CPU, %d threads)...'
              % (n_ref, ncpu), flush=True)
        per_call, results = [], []
        for (t, y, dy) in sub:
            t0 = time.perf_counter()
            # reference TLS expects flux normalized around 1.0; our
            # generator already produces y ~ 1.0
            model = transitleastsquares(t, y, dy)
            res = model.power(R_star=1.0, M_star=1.0,
                              period_min=cfg['period_min'],
                              period_max=cfg['period_max'],
                              oversampling_factor=3, use_threads=ncpu,
                              show_progress_bar=False)
            per_call.append(time.perf_counter() - t0)
            results.append({k: _f(getattr(res, k, None)) for k in
                            ('period', 'SDE', 'T0', 'duration', 'depth')})
        med = float(np.median(per_call))
        entry.update(nlc=n_ref, total_s=float(np.sum(per_call)),
                     per_call_s=per_call, ms_per_lc=1000.0 * med,
                     lc_per_s=1.0 / med, extrapolated=True,
                     note='per-LC median over %d LCs' % n_ref)
        entry.update(eval_recovery(results, subflags, cfg['inject_period']))
        entry['per_lc'] = compact_per_lc(results, subflags)
    except Exception as e:
        traceback.print_exc()
        entry['error'] = repr(e)
    return entry


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def _row(key, impl, nlc, total, mslc, lcs, rec, notes):
    print('%-11s %-10s %5s %10s %10s %9s %10s  %s'
          % (key, impl, nlc, total, mslc, lcs, rec, notes))


def print_summary(out):
    print('\n' + '=' * 96 + '\nSUMMARY\n' + '=' * 96)
    _row('regime', 'impl', 'nlc', 'total s', 'ms/LC', 'LC/s', 'recovery',
         'notes')
    print('-' * 96)
    for key, regime in out['regimes'].items():
        for impl, e in regime['impls'].items():
            if 'skipped' in e:
                _row(key, impl, *['-'] * 5, 'skipped: %s' % e['skipped'])
            elif 'error' in e:
                _row(key, impl, *['-'] * 5,
                     'error: %s' % str(e['error'])[:40])
            else:
                rec = '%d/%d' % (e.get('n_recovered', 0),
                                 e.get('n_injected', 0))
                if e.get('n_alias'):
                    rec += '+%da' % e['n_alias']
                _row(key, impl, '%d' % e['nlc'], '%.3f' % e['total_s'],
                     '%.2f' % e['ms_per_lc'], '%.2f' % e['lc_per_s'],
                     rec, e.get('note', ''))
    print('=' * 96)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Survey-scale TLS throughput benchmark (GPU)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--regimes', default=','.join(REGIMES),
                   help='comma-separated regime keys')
    p.add_argument('--nlc', type=int, default=None,
                   help='override per-regime lightcurve count')
    p.add_argument('--impls', default='new,old',
                   help='comma-separated: new,old,reference')
    p.add_argument('--ref-nlc', type=int, default=1,
                   help='LCs for the CPU reference implementation')
    p.add_argument('--old-nlc', type=int, default=5,
                   help='LCs for the old per-LC GPU path (extrapolated)')
    p.add_argument('--n-iter', type=int, default=1,
                   help='timed iterations per batch (median reported)')
    p.add_argument('--output', default='tls_survey_bench_results.json')
    p.add_argument('--quick', action='store_true',
                   help='smoke test: nlc=4 per regime')
    p.add_argument('--ref-all', action='store_true',
                   help='run CPU reference on all regimes incl. kepler-4yr')
    return p.parse_args()


def main():
    args = parse_args()

    regime_keys = [k.strip() for k in args.regimes.split(',') if k.strip()]
    bad = [k for k in regime_keys if k not in REGIMES]
    if bad:
        sys.exit('unknown regime(s) %s; choose from %s'
                 % (bad, list(REGIMES)))
    impls = [s.strip() for s in args.impls.split(',') if s.strip()]
    bad = [s for s in impls if s not in ('new', 'old', 'reference')]
    if bad:
        sys.exit('unknown impl(s) %s; choose from new,old,reference' % bad)

    out = dict(script='benchmark_tls_survey.py',
               timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
               args=vars(args), regimes=OrderedDict())
    state = {}

    for key in regime_keys:
        cfg = dict(REGIMES[key])
        nlc = args.nlc if args.nlc else (4 if args.quick else cfg['nlc'])
        print('\n' + '=' * 70)
        print('%s: ndata=%d, baseline=%.1fd, P=[%.2g, %.4g]d, nlc=%d'
              % (key, cfg['ndata'], cfg['baseline'], cfg['period_min'],
                 cfg['period_max'], nlc))
        print('=' * 70)

        lcs, flags = make_regime_lcs(key, cfg, nlc)
        nperiods = probe_nperiods(cfg)
        if nperiods:
            print('  Ofir grid: %d periods' % nperiods)

        regime_entry = dict(config=cfg, nlc=nlc, nperiods=nperiods,
                            impls=OrderedDict())
        for impl in impls:
            if impl == 'new':
                e = run_new(cfg, lcs, flags, args, state)
            elif impl == 'old':
                e = run_old(cfg, lcs, flags, args)
            else:
                e = run_reference(key, cfg, lcs, flags, args)
            if 'total_s' in e:
                print('  [%s] total %.3f s | %.2f ms/LC | %.2f LC/s | '
                      'recovered %d/%d (+%d alias)'
                      % (impl, e['total_s'], e['ms_per_lc'], e['lc_per_s'],
                         e.get('n_recovered', 0), e.get('n_injected', 0),
                         e.get('n_alias', 0)))
            regime_entry['impls'][impl] = e
        out['regimes'][key] = regime_entry

    out['env'] = env_info()
    print('\n' + json.dumps(out['env'], indent=2))

    print_summary(out)

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
