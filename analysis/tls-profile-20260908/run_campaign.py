#!/usr/bin/env python3
"""Sequential bounded profiling jobs; retain exits and kill timeout groups."""
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time


def main():
    root = Path('/tmp/cuvarbase-tls-profile')
    os.chdir(root)
    out = root / 'results'
    jobs = []
    for profile in ['ztf', 'rubin']:
        for backend, version, variant in [('gtls', 'head', 'native'), ('gtls', 'pypi', 'native'),
                                          ('gtls', 'head', 'union'), ('gtls', 'head', 'both'),
                                          ('v1', 'release', 'native')]:
            name = f'{profile}_{backend}_{version}_{variant}'
            command = ['modern/bin/python', 'profile_tls.py', '--backend', backend,
                       '--input', f'inputs/tls_sparse_{profile}.npz', '--out', f'results/{name}.json',
                       '--variant', variant, '--profile-reps', '2' if backend == 'v1' else '1']
            jobs.append(dict(name=name, command=command, version=version, timeout=900))
    random.Random(901337).shuffle(jobs)
    for profile in ['ps1', 'gaia', 'ztf', 'rubin']:
        name = f'{profile}_cpu_failure'
        jobs.append(dict(name=name, version='cpu', timeout=1200,
                         command=['modern/bin/python', 'diagnose_cpu.py', '--input',
                                  f'inputs/tls_sparse_{profile}.npz', '--out', f'results/{name}.json']))
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    env_base = os.environ.copy()
    env_base.update(LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONUTF8='1',
                    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    NUMBA_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='0',
                    PATH='/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
                    CUDA_HOME='/usr/local/cuda',
                    LD_LIBRARY_PATH='/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''))
    for job in jobs:
        env = env_base.copy()
        if job['version'] == 'head':
            env['PYTHONPATH'] = str(root / 'gtls-head-install')
        else:
            env.pop('PYTHONPATH', None)
        started = time.time()
        print('START', job['name'], flush=True)
        with (out / (job['name'] + '.log')).open('w') as log:
            child = subprocess.Popen(job['command'], stdout=log, stderr=log, env=env,
                                     start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=job['timeout'])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 'timeout'
        execution = dict(job=job, started_epoch=started, elapsed_s=time.time()-started,
                         exit_code=code, timed_out=timed_out, pid=child.pid)
        (out / (job['name'] + '.execution.json')).write_text(json.dumps(execution, indent=2) + '\n')
        print('FINISHED', job['name'], code, round(execution['elapsed_s'], 2), flush=True)
    print('PROFILE_CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
