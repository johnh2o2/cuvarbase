#!/usr/bin/env python3
"""Run one independent TLS recovery shard through pinned public APIs.

All candidates and scores are retained. Full spectra are retained for the first
four cases of every shard; every spectrum is hashed before disposal. This
avoids committing large synthetic periodogram archives to a release repository.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from generate import array_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transit'))
from worker import Backend, dump, sha


def cpu_quota():
    paths = ['/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
             '/sys/fs/cgroup/cpu/cpu.cfs_period_us',
             '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us',
             '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us']
    return {p: Path(p).read_text().strip() for p in paths if Path(p).exists()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    data = np.load(a.input)
    metadata = json.loads(str(data['metadata']))
    lcs = [tuple(data[f'{k}_{i}'] for k in ('t', 'y', 'dy'))
           for i in range(len(metadata['cases']))]
    record = dict(status='running', profile=metadata['profile'], split=metadata['split'],
                  start=metadata.get('start', 0), count=len(lcs), config=cfg,
                  input_file=a.input.name, input_sha256=sha(a.input),
                  config_sha256=sha(a.config), adapter_sha256=sha(Path(__file__).resolve().parents[1] / 'transit/worker.py'),
                  runner_sha256=sha(__file__), cpu_quota=cpu_quota(),
                  threads={k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')},
                  boundary='Prepared host arrays and explicit period grid to host periodograms and native candidate/score; GPU synchronized. Generation, imports, contexts, disk output excluded.',
                  packages={}, cases=[])
    for name in ('numpy', 'scipy', 'batman-package', 'pycuda', 'cupy-cuda12x', 'gputls', 'cuvarbase'):
        try:
            record['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        record['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.free,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        record['gpu'] = None
    a.out.mkdir(parents=True, exist_ok=True)
    dump(a.out / 'summary.json', record)
    try:
        begin = time.perf_counter()
        backend = Backend(cfg, data, max(len(lc[0]) for lc in lcs))
        backend.sync()
        record['initialization_s'] = time.perf_counter() - begin
        begin = time.perf_counter()
        warm = backend.search(lcs[:1])
        backend.sync()
        record['first_api_s'] = time.perf_counter() - begin
        record['warmup_candidates'] = [o['candidate'] for o in warm]
        record['installed_sources'] = {}
        for name in ('cuvarbase', 'gputls'):
            if name in sys.modules:
                package = Path(sys.modules[name].__file__).parent
                record['installed_sources'][name] = {str(p.relative_to(package)): sha(p)
                    for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.so')}
        for start in range(0, len(lcs), cfg.get('eval_chunk', 16)):
            stop = min(start + cfg.get('eval_chunk', 16), len(lcs))
            backend.sync()
            begin = time.perf_counter()
            try:
                outputs = backend.search(lcs[start:stop])
                backend.sync()
            except Exception:
                error = traceback.format_exc()
                elapsed = time.perf_counter() - begin
                for i in range(start, stop):
                    record['cases'].append(dict(**metadata['cases'][i], api_result_valid=False,
                                                period_found=None, score=None, error=error,
                                                recovered=False, alias_recovered=False,
                                                search_s=elapsed / (stop - start)))
                dump(a.out / 'summary.json', record)
                continue
            elapsed = time.perf_counter() - begin
            if len(outputs) != stop - start:
                raise RuntimeError('Public API returned wrong number of outputs')
            for i, output in zip(range(start, stop), outputs):
                truth = metadata['cases'][i]
                candidate = dict(output['candidate'])
                candidate.setdefault('api_result_valid', True)
                found = candidate.pop('period')
                epoch_found = candidate.pop('epoch', None)
                drift = abs(found / truth['period'] - 1) * metadata['baseline'] if found is not None else None
                recovered = bool(drift is not None and drift <= .5 * truth['duration']) if truth['injected'] else None
                alias = bool(found is not None and any(abs(found / (truth['period'] * k) - 1) * metadata['baseline'] <= .5 * truth['duration']
                              for k in (.5, 1., 2., 1/3, 3.))) if truth['injected'] else None
                row = dict(**truth, **candidate, period_found=found, epoch_found=epoch_found, phase_drift_days=drift,
                           recovered=recovered, alias_recovered=alias, search_s=elapsed / (stop - start),
                           evaluation_chunk=stop - start, n_periods=len(output['periods']),
                           spectrum_sha256={k: array_hash(output[k]) for k in ('periods', 'power')})
                if output.get('error'):
                    row['error'] = output['error']
                if i < 4:
                    path = a.out / f'case_{i:04}.npz'
                    np.savez_compressed(path, periods=output['periods'], power=output['power'])
                    row.update(output_file=path.name, output_sha256=sha(path))
                record['cases'].append(row)
            dump(a.out / 'summary.json', record)
            # Progress deliberately omits held-out outcomes until analysis.
            print(json.dumps(dict(completed=stop, count=len(lcs), elapsed_s=elapsed)), flush=True)
        record['status'] = 'ok'
    except Exception:
        record.update(status='error', error=traceback.format_exc())
    dump(a.out / 'summary.json', record)
    if record['status'] != 'ok':
        raise RuntimeError(record['error'])


if __name__ == '__main__':
    main()
