#!/usr/bin/env python3
"""Serial campaign controller: no competing benchmarks on the same host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke', 'ls', 'ls_legacy', 'ls_shared_followup',
                                       'tls', 'tls_followup', 'ensemble'], required=True)
    ap.add_argument('--root', default='/tmp/cuvarbase-benchmark-audit')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / 'results'
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PATH='/usr/local/cuda/bin:' + env['PATH'], CUDA_HOME='/usr/local/cuda',
               LD_LIBRARY_PATH='/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONUTF8='1')
    records = []

    def job(label, script, opts, legacy=False, head=False, timeout=600):
        penv = env.copy()
        if head:
            penv['PYTHONPATH'] = str(root / 'gtls-head-install')
        cmd = [str(root / ('legacy' if legacy else 'modern') / 'bin/python'),
               str(root / script), *map(str, opts), '--out', str(out / (label + '.json'))]
        print('START', label, flush=True)
        start = time.time()
        with (out / (label + '.log')).open('w') as log:
            try:
                p = subprocess.run(cmd, cwd=root, env=penv, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                status = 'exited'
                rc = p.returncode
            except subprocess.TimeoutExpired:
                status, rc = 'timeout', None
        records.append(dict(label=label, argv=cmd, status=status, returncode=rc,
                            started_unix=start,
                            wall_s=time.time()-start, timeout_s=timeout,
                            gtls_head=head, legacy=legacy))
        (out / ('controller_' + args.stage + '.json')).write_text(json.dumps(records, indent=2))
        print('END', label, status, rc, round(time.time()-start, 2), flush=True)

    if args.stage == 'smoke':
        for backend in ['cuvarbase', 'nifty_cpu', 'nifty_gpu', 'astropy']:
            job('smoke_ls_' + backend, 'run_ls.py',
                ['--backend', backend, '--config', 'small', '--reps', 3])
        job('smoke_ls_legacy', 'run_ls.py', ['--backend', 'cuvarbase', '--config', 'small', '--reps', 3], legacy=True)
        for backend in ['cuvarbase', 'gtls', 'cpu']:
            job('smoke_tls_' + backend, 'run_tls.py',
                ['--backend', backend, '--input', root/'inputs/tls27.npz', '--reps', 3])
        job('smoke_tls_gtls_head', 'run_tls.py',
            ['--backend', 'gtls', '--input', root/'inputs/tls27.npz', '--reps', 3], head=True)
    elif args.stage == 'ls_legacy':
        for cfg in ['small', 'tess', 'ztf', 'kepler']:
            for n in [1, 32]:
                job(f'ls_{cfg}_{n}_pypi', 'run_ls.py',
                    ['--backend', 'cuvarbase', '--config', cfg, '--n-lcs', n, '--reps', 5], legacy=True)
        job('ls_shared_tess_pypi', 'run_ls.py',
            ['--backend', 'cuvarbase', '--config', 'tess', '--n-lcs', 32,
             '--shared-times', '--reps', 5], legacy=True)
    elif args.stage == 'ls':
        for cfg in ['small', 'tess', 'ztf', 'kepler']:
            for n in [1, 32]:
                base = ['--config', cfg, '--n-lcs', n, '--reps', 5]
                prefix = f'ls_{cfg}_{n}'
                job(prefix+'_v1', 'run_ls.py', ['--backend', 'cuvarbase', *base])
                job(prefix+'_pypi', 'run_ls.py', ['--backend', 'cuvarbase', *base], legacy=True)
                job(prefix+'_v1_double', 'run_ls.py', ['--backend', 'cuvarbase', '--double', *base])
                job(prefix+'_pypi_double', 'run_ls.py', ['--backend', 'cuvarbase', '--double', *base], legacy=True)
                job(prefix+'_nifty_gpu', 'run_ls.py', ['--backend', 'nifty_gpu', *base])
                job(prefix+'_nifty_gpu_float32', 'run_ls.py', ['--backend', 'nifty_gpu', '--float32', *base])
                for threads in [1, 4, 8]:
                    job(prefix+f'_nifty_cpu_t{threads}', 'run_ls.py',
                        ['--backend', 'nifty_cpu', '--threads', threads, *base])
                if n > 1:
                    for workers in [4, 8]:
                        job(prefix+f'_nifty_cpu_w{workers}', 'run_ls.py',
                            ['--backend', 'nifty_cpu', '--workers', workers, *base])
                # This baseline was substantially slower in the pilot. Bound
                # the cost and archive timeouts as censored, never as timings.
                job(prefix+'_astropy', 'run_ls.py', ['--backend', 'astropy', *base], timeout=45)
        # Same-epoch batching is a distinct workload. Both versions of cuvarbase
        # and nifty-ls receive the same 32 series with shared timestamps.
        for backend in ['cuvarbase', 'nifty_cpu', 'nifty_gpu']:
            job('ls_shared_tess_'+backend, 'run_ls.py',
                ['--backend', backend, '--config', 'tess', '--n-lcs', 32,
                 '--shared-times', '--reps', 5])
        job('ls_shared_tess_pypi', 'run_ls.py',
            ['--backend', 'cuvarbase', '--config', 'tess', '--n-lcs', 32,
             '--shared-times', '--reps', 5], legacy=True)
    elif args.stage == 'ls_shared_followup':
        base = ['--config', 'tess', '--n-lcs', 32, '--shared-times', '--reps', 5]
        for legacy, label in [(False, 'cuvarbase'), (True, 'pypi')]:
            job('ls_shared_tess_'+label+'_double', 'run_ls.py',
                ['--backend', 'cuvarbase', '--double', *base], legacy=legacy)
        for threads in [4, 8]:
            job(f'ls_shared_tess_nifty_cpu_t{threads}', 'run_ls.py',
                ['--backend', 'nifty_cpu', '--threads', threads, *base])
        job('ls_shared_tess_nifty_gpu_float32', 'run_ls.py',
            ['--backend', 'nifty_gpu', '--float32', *base])
    elif args.stage == 'tls_followup':
        # Preserve non-finite native GTLS outputs as nulls in JSON, and cover
        # the remaining CPU allocation points for the displayed 27-day cases.
        for n in [1, 16]:
            job(f'tls_27_{n}_gtls_pypi', 'run_tls.py',
                ['--backend', 'gtls', '--input', root/'inputs/tls27.npz', '--n-lcs', n, '--reps', 3])
        for threads in [1, 8]:
            job(f'tls_27_1_cpu_t{threads}', 'run_tls.py',
                ['--backend', 'cpu', '--input', root/'inputs/tls27.npz', '--threads', threads, '--reps', 3])
        job('tls_27_16_cpu_w8', 'run_tls.py',
            ['--backend', 'cpu', '--input', root/'inputs/tls27.npz', '--n-lcs', 16,
             '--workers', 8, '--reps', 3])
    elif args.stage == 'tls':
        for baseline, n in [(27, 1), (27, 16), (200, 1), (1500, 1)]:
            base = ['--input', root/f'inputs/tls{baseline}.npz', '--n-lcs', n, '--reps', 3]
            prefix = f'tls_{baseline}_{n}'
            job(prefix+'_v1_wide', 'run_tls.py', ['--backend', 'cuvarbase', *base])
            job(prefix+'_v1_default', 'run_tls.py', ['--backend', 'cuvarbase', '--default-window', *base])
            if baseline < 1500:
                job(prefix+'_gtls_pypi', 'run_tls.py', ['--backend', 'gtls', *base], timeout=900)
            job(prefix+'_gtls_head', 'run_tls.py', ['--backend', 'gtls', *base], head=True, timeout=1200)
            if baseline < 1500:
                job(prefix+'_cpu', 'run_tls.py',
                    ['--backend', 'cpu', '--threads', 4 if n==1 else 1,
                     '--workers', 1 if n==1 else 4, *base], timeout=1200)
    else:
        base = ['--input', root/'inputs/tls27_ensemble.npz', '--n-lcs', 64, '--evaluate-only']
        for backend in ['cuvarbase', 'gtls', 'cpu']:
            job('ensemble_'+backend, 'run_tls.py',
                ['--backend', backend, '--workers', 4 if backend=='cpu' else 1, *base],
                head=backend=='gtls', timeout=1800)
        job('ensemble_cuvarbase_default', 'run_tls.py',
            ['--backend', 'cuvarbase', '--default-window', *base], timeout=600)


if __name__ == '__main__':
    main()
