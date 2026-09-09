#!/usr/bin/env python3
"""Version-agnostic, host-to-host LS comparison; see README.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import traceback

import numpy as np

from common import LS_CONFIGS, array_hash, environment, measure, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cuvarbase', 'nifty_cpu', 'nifty_gpu',
                                          'astropy'], required=True)
    ap.add_argument('--config', default='small')
    ap.add_argument('--n-lcs', type=int, default=1)
    ap.add_argument('--shared-times', action='store_true')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--double', action='store_true')
    ap.add_argument('--float32', action='store_true',
                    help='Cast nifty-ls inputs inside the timed call; default is float64')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    input_path = Path(__file__).resolve().parent / 'inputs' / (
        f'ls_{args.config}_{"shared" if args.shared_times else "distinct"}.npz')
    with np.load(input_path) as data:
        freqs = data['freqs']
        t, y, dy = data['t'], data['y'], data['dy']
        lcs = [(t[i], y[i], dy[i]) for i in range(args.n_lcs)]
    cfg = LS_CONFIGS[args.config]
    record = dict(algorithm='LS', args=vars(args), config=cfg,
                  environment=environment(), status='running',
                  input_sha256=array_hash(freqs, *[a for lc in lcs for a in lc]),
                  frequency_sha256=array_hash(freqs),
                  input_file=input_path.name,
                  dtype_input='float64',
                  boundary='API wall time, host inputs to full host periodograms; '
                           'imports, input generation and validation excluded')
    write_json(args.out, record)
    try:
        sync = lambda: None
        executor = None
        if args.backend == 'cuvarbase':
            from cuvarbase.lombscargle import LombScargleAsyncProcess
            import pycuda.driver as drv
            proc = LombScargleAsyncProcess(use_double=args.double)
            sync = drv.Context.synchronize

            def run():
                if len(lcs) == 1:
                    res = proc.run(lcs, freqs=[freqs])
                else:
                    res = proc.batched_run_const_nfreq(
                        lcs, batch_size=args.batch_size, freqs=freqs,
                        only_return_best_freqs=False)
                proc.finish()
                return np.asarray([p for _, p in res])

        elif args.backend.startswith('nifty'):
            import nifty_ls
            kw = dict(fmin=float(freqs[0]), fmax=float(freqs[-1]), Nf=len(freqs),
                      center_data=True, fit_mean=True, normalization='standard')
            if args.backend == 'nifty_cpu':
                kw.update(backend='finufft', nthreads=args.threads)
            else:
                import cupy as cp
                kw.update(backend='cufinufft')
                sync = cp.cuda.runtime.deviceSynchronize

            def single(lc):
                if args.float32:
                    lc = tuple(np.asarray(a, dtype=np.float32) for a in lc)
                result = nifty_ls.lombscargle(*lc, **kw).power
                if hasattr(result, 'get'):
                    result = result.get()
                return np.asarray(result)

            if args.shared_times and len(lcs) > 1:
                # Stacking is inside timing: raw host LCs are the input contract.
                def run():
                    return single((lcs[0][0], np.stack([x[1] for x in lcs]),
                                   np.stack([x[2] for x in lcs])))
            elif args.workers > 1:
                executor = ThreadPoolExecutor(max_workers=args.workers)

                def run():
                    return np.asarray(list(executor.map(single, lcs)))
            else:
                def run():
                    return np.asarray([single(lc) for lc in lcs])
        else:
            from astropy.timeseries import LombScargle

            def single(lc):
                return LombScargle(*lc, fit_mean=True, center_data=True,
                                  normalization='standard').power(freqs, method='fast',
                                                                 assume_regular_frequency=True)
            if args.workers > 1:
                executor = ThreadPoolExecutor(max_workers=args.workers)

                def run():
                    return np.asarray(list(executor.map(single, lcs)))
            else:
                def run():
                    return np.asarray([single(lc) for lc in lcs])

        stats, power = measure(run, sync, args.reps)
        if executor:
            executor.shutdown()
        power = np.atleast_2d(power)
        assert power.shape == (args.n_lcs, len(freqs)), power.shape
        if not np.all(np.isfinite(power)):
            raise ValueError('Non-finite LS periodogram')
        peak = np.argmax(power, axis=1)
        # Store exact computed samples for cross-version accuracy checks in the
        # modern environment. Legacy environment need not install Astropy.
        sample = np.unique(np.concatenate([
            np.linspace(0, len(freqs) - 1, 256).astype(int), peak,
            np.clip(peak - 1, 0, len(freqs) - 1),
            np.clip(peak + 1, 0, len(freqs) - 1)]))
        record.update(status='ok', timing=stats,
                      seconds_per_lc=stats['median_s'] / args.n_lcs,
                      peak_frequency=freqs[peak].tolist(),
                      peak_power=power[np.arange(args.n_lcs), peak].tolist(),
                      validation_indices=sample.tolist(),
                      validation_power=power[:, sample].tolist(),
                      output_sha256=array_hash(power))
    except Exception:
        record.update(status='error', error=traceback.format_exc())
        print(record['error'], flush=True)
    record['environment_after'] = environment()
    write_json(args.out, record)
    print(json.dumps({k: record[k] for k in ['status', 'seconds_per_lc'] if k in record}),
          flush=True)
    if record['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
