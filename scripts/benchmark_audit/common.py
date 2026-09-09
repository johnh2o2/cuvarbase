"""Small, GPU-independent helpers for the September benchmark audit."""
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def environment():
    names = ['cuvarbase', 'numpy', 'scipy', 'pycuda', 'scikit-cuda',
             'astropy', 'nifty-ls', 'finufft', 'cufinufft', 'cupy-cuda12x',
             'gputls', 'transitleastsquares', 'batman-package', 'numba']
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    out = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
               python=platform.python_version(), host=platform.node(),
               platform=platform.platform(), packages=packages,
               threads={k: os.environ.get(k) for k in
                        ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']})
    for filename in ['cpu.max', 'cpu.stat', 'cpuset.cpus.effective']:
        p = Path('/sys/fs/cgroup') / filename
        if p.exists():
            out[filename] = p.read_text().strip()
    try:
        out['gpu'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def measure(fn, sync, reps=5):
    """First API call separately, one additional warmup, then equal repeats.

    All calls consume host inputs and return host outputs. No subtraction of
    estimated compilation or transfer time. Import/context time is excluded.
    """
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    first = time.perf_counter() - start
    print('first_api_call_s', first, flush=True)
    fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        samples.append(time.perf_counter() - start)
        print('warm_sample_s', samples[-1], flush=True)
    return dict(first_call_s=first, times_s=samples,
                median_s=float(np.median(samples)),
                min_s=min(samples), max_s=max(samples)), result


LS_CONFIGS = {
    'small': dict(ndata=1000, baseline=27.0, nfreq=5000, fmax=10.0),
    'tess': dict(ndata=20000, baseline=27.0, nfreq=13500, fmax=100.0),
    'ztf': dict(ndata=150, baseline=730.0, nfreq=365000, fmax=100.0),
    'kepler': dict(ndata=65000, baseline=1460.0, nfreq=730000, fmax=100.0),
}


def ls_input(config, n_lcs, shared_times=False):
    cfg = LS_CONFIGS[config]
    n, baseline, nf = cfg['ndata'], cfg['baseline'], cfg['nfreq']
    # Integer k0, float64 grid. Identical arrays delivered to every backend.
    df = cfg['fmax'] / nf
    k0 = max(1, round((1.0 / baseline) / df))
    freqs = (k0 + np.arange(nf, dtype=np.float64)) * df
    lcs = []
    for i in range(n_lcs):
        rt = np.random.RandomState(9100 if shared_times else 9100 + i)
        t = np.sort(rt.uniform(0.0, baseline, n))
        r = np.random.RandomState(19100 + i)
        dy = 0.003 * r.uniform(0.8, 1.2, n)
        y = 1.0 + 0.01 * np.sin(2 * np.pi * 0.7431 * t + 0.27 * i)
        y += r.normal(size=n) * dy
        lcs.append((t, y, dy))
    return lcs, freqs, cfg
