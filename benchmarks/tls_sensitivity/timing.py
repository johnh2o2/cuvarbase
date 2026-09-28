#!/usr/bin/env python3
"""Measure TLS latency and throughput with one GPU and isolated method processes.

The parent randomizes configuration order. Each child warms its exact workload
and measures five synchronized repetitions. Inputs must be the earlier tuning
archives, not the independent sensitivity cohorts. No cloud resources are
created by this script.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run import Backend, array_hash, cpu_quota, dump, sha


PROFILES = ('tess_200s', 'tess_gap', 'ztf')
METHODS = ('v1_original', 'v1_resolved', 'v1_fine')
INDICES = list(range(8)) + list(range(128, 136))


def observe(backend, lightcurves):
    backend.sync()
    start = time.perf_counter()
    outputs = backend.search(lightcurves)
    backend.sync()
    elapsed = time.perf_counter() - start
    if len(outputs) != len(lightcurves):
        raise RuntimeError('Wrong number of timed outputs')
    candidates = []
    for output in outputs:
        c = output['candidate']
        if output.get('error') or not c.get('api_result_valid', True) or c['finite_fraction'] != 1.:
            raise RuntimeError('Invalid timed result: ' + str(output.get('error', c)))
        candidates.append(dict(**c, spectrum_sha256={
            k: array_hash(output[k]) for k in ('periods', 'power')}))
    return elapsed, candidates


def child(args):
    profile, method, mode = args.job.split('/')
    path = args.inputs / f'{profile}_heldout.npz'
    data = np.load(path)
    config = json.loads((args.configs / f'{method}.json').read_text())
    if method.startswith('gtls') and mode == 'single':
        config['workers'] = 1
    lcs = [tuple(data[f'{key}_{i}'] for key in ('t', 'y', 'dy')) for i in INDICES]
    truth = json.loads(str(data['metadata']))
    result = dict(status='running', profile=profile, method=method, mode=mode,
                  config=config, indices=INDICES, input_sha256=sha(path),
                  grid_sha256={k: array_hash(data[k]) for k in ('freqs', 'q', 'tls_periods')},
                  input_array_sha256=[{k: array_hash(v) for k, v in zip(('t', 'y', 'dy'), lc)} for lc in lcs],
                  truth=[truth['cases'][i] for i in INDICES],
                  runner_sha256=sha(__file__),
                  adapter_sha256=sha(Path(__file__).resolve().parents[1] / 'transit/worker.py'),
                  cpu_quota=cpu_quota(), repetitions=[])
    output = args.out / profile / method / mode / 'summary.json'
    dump(output, result)
    try:
        start = time.perf_counter()
        backend = Backend(config, data, max(len(lc[0]) for lc in lcs))
        backend.sync()
        result['initialization_s'] = time.perf_counter() - start
        elapsed, candidates = observe(backend, lcs[:1])
        result['first_api_s'] = elapsed
        result['first_api_candidates'] = candidates
        # Warm the same workload that will be timed, including all pool workers.
        warm = [observe(backend, [lc]) for lc in lcs] if mode == 'single' else [observe(backend, lcs)]
        result['warmup'] = [dict(elapsed_s=t, candidates=c) for t, c in warm]
        for rep in range(args.reps):
            # Rotating source order prevents one particular source always being first.
            order = np.roll(np.arange(len(lcs)), rep).tolist()
            if mode == 'single':
                calls = [dict(index=INDICES[i], elapsed_s=t, candidates=c)
                         for i in order for t, c in [observe(backend, [lcs[i]])]]
                seconds = float(np.mean([c['elapsed_s'] for c in calls]))
            else:
                elapsed, candidates = observe(backend, [lcs[i] for i in order])
                calls = [dict(indices=[INDICES[i] for i in order], elapsed_s=elapsed,
                              candidates=candidates)]
                seconds = elapsed / len(lcs)
            result['repetitions'].append(dict(rep=rep, seconds_per_source=seconds, calls=calls))
            dump(output, result)
        values = [r['seconds_per_source'] for r in result['repetitions']]
        result.update(status='ok', seconds_per_source=float(np.median(values)),
                      min_seconds_per_source=min(values), max_seconds_per_source=max(values),
                      n_sources=len(lcs), n_repetitions=args.reps,
                      single_statistic='Median across repetitions of mean latency over 16 distinct sources',
                      batch_statistic='Median 16-source API call duration divided by 16')
        result['installed_sources'] = {}
        for name in ('cuvarbase', 'gputls'):
            if name in sys.modules:
                root = Path(sys.modules[name].__file__).parent
                result['installed_sources'][name] = {
                    str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                    if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.so')}
        dump(output, result)
    except Exception:
        import traceback
        result.update(status='error', error=traceback.format_exc())
        dump(output, result)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026090927)
    ap.add_argument('--job', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.job:
        child(a)
        return
    jobs = [f'{p}/{m}/{mode}' for p in PROFILES for m in (*METHODS, f'gtls_{p}')
            for mode in ('single', 'batch16')]
    np.random.default_rng(a.seed).shuffle(jobs)
    a.out.mkdir(parents=True, exist_ok=True)
    plan = dict(order=jobs, seed=a.seed, indices=INDICES, repetitions=a.reps,
                boundary='Prepared host arrays and explicit period grid through host periodograms, native candidate and score, including transfers and synchronization.',
                exclusions='Imports, context initialization, grid creation, synthetic data generation, disk I/O and survey preprocessing.',
                hardware={
                    'gpu': subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True),
                    'cpu': subprocess.check_output(['lscpu'], text=True),
                    'cpu_quota': cpu_quota(),
                    'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)})
    dump(a.out / 'plan.json', plan)
    statuses = []
    for job in jobs:
        print('Timing ' + job, flush=True)
        started = time.time()
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--inputs', str(a.inputs),
                                    '--configs', str(a.configs), '--out', str(a.out), '--reps', str(a.reps),
                                    '--job', job])
        statuses.append(dict(job=job, exit_code=completed.returncode,
                             started_epoch=started, finished_epoch=time.time()))
        dump(a.out / 'status.json', statuses)


if __name__ == '__main__':
    main()
