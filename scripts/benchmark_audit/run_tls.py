#!/usr/bin/env python3
"""TLS timing and recovery runner. Inputs are generated once and hashed."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from scipy.signal import medfilt

from common import array_hash, environment, measure, write_json


def cpu_search(job):
    lc, periods, threads = job
    # The reference API has no explicit-period argument. Inject ONLY the shared
    # grid at its grid factory, and verify the returned periods below. No search
    # kernel, template, or duration/epoch evaluation is patched.
    import importlib
    main = importlib.import_module('transitleastsquares.main')
    old = main.period_grid
    main.period_grid = lambda **kwargs: periods.copy()
    try:
        from transitleastsquares import transitleastsquares
        return transitleastsquares(*lc, verbose=False).power(
            use_threads=threads, show_progress_bar=False, verbose=False,
            R_star=1, M_star=1, R_star_min=.05, R_star_max=4,
            M_star_min=.05, M_star_max=1, oversampling_factor=3,
            T0_fit_margin=.125, duration_grid_step=1.1)
    finally:
        main.period_grid = old


def identical_sde(chi2):
    # Frozen cuvarbase v1.0 / reference SR definition. Historical 1-chi2/max is
    # also saved for comparison. This is a score, NOT a calibrated FAP.
    c = np.asarray(chi2, float)
    valid = np.isfinite(c) & (c > 0) & (c < 1e29)
    c = c[valid]
    if len(c) < 5:
        return dict(current=None, historical=None)
    out = {}
    for name, sr in [('current', c.min()/c), ('historical', 1-c/c.max())]:
        trend = medfilt(sr, 91) if len(sr) > 91 else np.zeros_like(sr)
        residual = sr-trend
        out[name] = float((residual.max()-residual.mean())/residual.std()) \
            if residual.std() > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cuvarbase', 'gtls', 'cpu'], required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--n-lcs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--default-window', action='store_true')
    ap.add_argument('--evaluate-only', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    d = np.load(args.input)
    meta = json.loads(str(d['metadata']))
    periods = d['periods']
    lcs = [(d[f't_{i}'], d[f'y_{i}'], d[f'dy_{i}']) for i in range(args.n_lcs)]
    record = dict(algorithm='TLS', args=vars(args), environment=environment(),
                  status='running', input_metadata=meta,
                  input_sha256=array_hash(periods, *[a for lc in lcs for a in lc]),
                  boundary='API wall time, host lightcurves to host search results '
                           'and spectra; imports, grid generation, external '
                           're-scoring/validation excluded')
    write_json(args.out, record)
    executor = None
    try:
        sync = lambda: None
        if args.backend == 'cuvarbase':
            from cuvarbase.tls import tls_search_batch
            from cuvarbase.base import ensure_context
            import pycuda.driver as drv
            ensure_context()
            sync = drv.Context.synchronize
            kw = dict(periods=periods, R_star=1, M_star=1,
                      oversampling_factor=3, t0_oversample=8, n_durations=38,
                      refine_top_k=50, return_arrays=True,
                      u=[.4804, .1867], qmin=d['qmin'], qmax=d['qmax'])
            if args.default_window:
                kw.update(t0_oversample=3, n_durations=15)
                kw.pop('qmin')
                kw.pop('qmax')

            def run():
                return tls_search_batch(lcs, **kw)

        elif args.backend == 'gtls':
            import cupy as cp
            from gputls import gtls
            sync = cp.cuda.runtime.deviceSynchronize

            def run():
                return [gtls(*lc, verbose=False).power(
                    periods=periods, R_star=1, M_star=1, oversampling_factor=3,
                    T0_fit_margin=.125, duration_grid_step=1.1,
                    transit_template='default', verbose=False,
                    show_progress_bar=False) for lc in lcs]
        else:
            jobs = [(lc, periods, args.threads) for lc in lcs]
            if args.workers > 1:
                executor = ProcessPoolExecutor(max_workers=args.workers)

                def run():
                    return list(executor.map(cpu_search, jobs))
            else:
                def run():
                    return [cpu_search(job) for job in jobs]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if args.evaluate_only:
                sync()
                start = time.perf_counter()
                results = run()
                sync()
                stats = dict(evaluation_wall_s=time.perf_counter()-start)
            else:
                stats, results = measure(run, sync, args.reps)
            record['warnings'] = sorted(set(str(w.message) for w in caught))
        rows, spectra = [], {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and 'error' in result:
                raise RuntimeError(result['error'])
            def field(name):
                return result[name] if hasattr(result, '__getitem__') else getattr(result, name)
            c = np.asarray(np.ma.filled(field('chi2'), np.nan), float)
            p = np.asarray(np.ma.filled(field('periods'), np.nan), float)
            valid = np.isfinite(c) & np.isfinite(p) & (c > 0) & (c < 1e29)
            truth = meta['cases'][i]
            found = float(field('period'))
            native_sde = float(field('SDE'))
            raw = float(p[valid][np.argmin(c[valid])]) if valid.any() else None
            rows.append(dict(index=i, period=found if np.isfinite(found) else None,
                             chi2_best_period=raw,
                             true_period=truth['period'], injected=truth['injected'],
                             exact_recovery=bool(abs(found/truth['period']-1) < .002)
                             if truth['injected'] else None,
                             alias_recovery=bool(any(abs(found/(truth['period']*k)-1) < .002
                                                     for k in [.5, 1, 2, 1/3, 3]))
                             if truth['injected'] else None,
                             native_sde=native_sde if np.isfinite(native_sde) else None,
                             identical_sde=identical_sde(c),
                             periods_returned=len(p), periods_finite=int(valid.sum()),
                             period_grid_max_error=float(np.max(np.abs(
                                 np.sort(p[np.isfinite(p)])-periods)))
                             if np.isfinite(p).all() and len(p)==len(periods) else None))
            spectra.update({f'periods_{i}': p, f'chi2_{i}': c})
        np.savez_compressed(Path(args.out).with_suffix('.npz'), **spectra)
        record.update(status='ok', timing=stats, recovery=rows,
                      native_outputs_finite=all(r['period'] is not None and
                                                r['native_sde'] is not None for r in rows),
                      seconds_per_lc=stats.get('median_s', stats.get('evaluation_wall_s')) / args.n_lcs)
    except Exception:
        record.update(status='error', error=traceback.format_exc())
        print(record['error'], flush=True)
    finally:
        if executor:
            executor.shutdown()
    record['environment_after'] = environment()
    write_json(args.out, record)
    print(json.dumps({k: record[k] for k in ['status', 'seconds_per_lc'] if k in record}), flush=True)
    if record['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
