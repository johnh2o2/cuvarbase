"""Provenance helpers; no GPU imports."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def environment():
    packages = {}
    for name in ['numpy', 'scipy', 'cuvarbase', 'nifty-ls', 'finufft',
                 'cufinufft', 'pycuda', 'cupy-cuda12x', 'astropy', 'periodfind',
                 'periodfind_cpu', 'numba', 'gputls', 'transitleastsquares']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    out = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
               python=platform.python_version(), packages=packages,
               threads={k: os.environ.get(k) for k in ['OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS']})
    for name in ['cpu.max', 'cpu.stat', 'cpuset.cpus.effective']:
        p = Path('/sys/fs/cgroup') / name
        if p.exists():
            out[name] = p.read_text().strip()
    for name in ['cpu.cfs_quota_us','cpu.cfs_period_us','cpu.stat']:
        p = Path('/sys/fs/cgroup/cpu') / name
        if p.exists():
            out['cgroup_v1/'+name] = p.read_text().strip()
    try:
        out['gpu'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    out['harness_sha256'] = {p.name: sha(p) for p in
                            sorted(Path(__file__).parent.glob('*.py'))}
    return out


def load_inputs(path, nsource=None):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        count = min(nsource or meta['nsource'], meta['nsource'])
        sources = []
        for s in range(count):
            sources.append([tuple(z[f'{k}_{s}_{b}'].copy() for k in ('t', 'y', 'dy'))
                            for b in range(meta['nband'])])
        return z['freqs'].copy(), sources, meta


def measure(fn,sync,reps=3):
    sync();start=time.perf_counter();result=fn();sync();first=time.perf_counter()-start
    fn();sync();times=[]
    for _ in range(reps):
        sync();start=time.perf_counter();result=fn();sync();times.append(time.perf_counter()-start)
    return dict(first_call_s=first,times_s=times,median_s=float(np.median(times))),result
