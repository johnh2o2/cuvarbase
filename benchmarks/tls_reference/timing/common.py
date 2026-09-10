"""Shared inputs, literal public calls and post-measurement validation.

This module performs no cloud/resource actions. Scientific source is imported
from the installed packages and never edited by the timing runner.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


REGIMES = ('tess_solar', 'tess_gap', 'ztf_solar')
SINGLE_REPETITIONS = 5
BATCH_REPETITIONS = 3
NATIVE_BACKENDS = ('gtls', 'gtls_corrected')
_CORRECTION_CONTEXTS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def array_hash(value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise TypeError('Object-array pointer bytes are not a numerical fingerprint')
    digest = hashlib.sha256()
    # Identical encoding to the independent study's parity_harness.array_hash.
    digest.update(json.dumps(value.dtype.descr if value.dtype.names else value.dtype.str).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def masked_hash(value):
    value = np.ma.asarray(value)
    return dict(data=array_hash(value.data), mask=array_hash(np.ma.getmaskarray(value)))


def selected_names(regime):
    if regime not in REGIMES:
        raise ValueError('Unknown timing regime: ' + regime)
    return [f'{regime}_null_{index:04d}.npz' for index in range(16)]


def load_cases(manifest_path, regime, names=None):
    """Load and verify all bytes before clocks start; never generate a grid."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = {entry['file']: entry for entry in manifest['cases']}
    names = selected_names(regime) if names is None else names
    cases = []
    for name in names:
        entry = entries[name]
        path = manifest_path.parent / name
        if sha(path) != entry['sha256']:
            raise ValueError('Fixture differs from its manifest: ' + name)
        with np.load(path, allow_pickle=False) as source:
            data = {key: np.array(source[key], copy=True) for key in ('t', 'y', 'dy', 'periods')}
            metadata = json.loads(str(source['metadata']))
        if metadata['regime'] != regime:
            raise ValueError('Fixture belongs to another regime: ' + name)
        if np.any(data['t'] <= 0) or np.any(~np.isfinite(data['t'])):
            raise ValueError('Timing requires the same finite positive-origin inputs for both APIs')
        if np.any(np.diff(data['periods']) <= 0):
            raise ValueError('The sealed timing grid must be strictly increasing, matching study public-output order')
        options = dict(metadata['search_kwargs'])
        cases.append(dict(name=name, data=data, options=options, metadata=metadata,
                          input_sha256=entry['sha256'],
                          arrays={key: array_hash(value) for key, value in data.items()},
                          error_scale=float(np.mean(data['dy']))))
    if not cases:
        raise ValueError('No timing cases selected')
    first = cases[0]
    for case in cases[1:]:
        if (case['arrays']['periods'] != first['arrays']['periods'] or
                case['options'] != first['options']):
            raise ValueError('Public batch requires one identical period grid and search configuration')
    return cases


def case_identity(case):
    return {key: case[key] for key in ('name', 'input_sha256', 'arrays', 'options')}


def package_sources(backend):
    if backend in NATIVE_BACKENDS:
        import gputls
        root = Path(gputls.__file__).parent
        paths = sorted(path for path in root.rglob('*')
                       if path.is_file() and path.suffix in ('.py', '.cu', '.cuh')
                       and '__pycache__' not in path.parts)
    else:
        import cuvarbase
        root = Path(cuvarbase.__file__).parent
        paths = [root / name for name in ('tls.py', 'tls_reference.py',
                 'tls_reference_math.py', 'tls_reference_frontend.py',
                 'tls_reference_prefix.py', 'tls_grids.py', 'tls_stats.py',
                 'tls_models.py', 'kernels/tls_reference.cu',
                 'kernels/tls_reference_prepare.cu')]
    return dict(root=str(root), files={str(path.relative_to(root)): sha(path) for path in paths})


def initialize_backend(backend, prefix='graph', correction_adapter=None):
    import cupy as cp
    cp.cuda.Device(0).use()
    if backend == 'candidate':
        from cuvarbase import tls, tls_reference
        if prefix == 'row':
            tls_reference._native_flux_prefix = tls_reference._row_flux_prefix
        elif prefix != 'graph':
            raise ValueError('Unknown prefix implementation')
    elif backend in NATIVE_BACKENDS:
        from gputls import gtls, core
        if backend == 'gtls_corrected':
            if correction_adapter is None:
                raise ValueError('Corrected-native timing requires the frozen validation adapter')
            path = Path(correction_adapter).resolve()
            spec = importlib.util.spec_from_file_location('_tls_timing_corrected_reference', path)
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            context = adapter.apply(core)
            provenance = context.__enter__()
            # One dedicated process uses this explicitly labeled correction
            # throughout its lifetime. Installed source files remain unchanged.
            _CORRECTION_CONTEXTS.append(context)
    else:
        raise ValueError('Unknown backend')
    cp.cuda.runtime.deviceSynchronize()
    sources = package_sources(backend)
    if backend == 'gtls_corrected':
        sources['reference_correction'] = provenance
    return sources


def public_single(backend, case):
    data, options = case['data'], case['options']
    if backend in NATIVE_BACKENDS:
        from gputls import gtls
        model = gtls(data['t'], data['y'], data['dy'], verbose=False)
        return model.power(periods=data['periods'], fast=False,
                           verbose=False, show_progress_bar=False, **options)
    from cuvarbase.tls import tls_search_gpu
    return tls_search_gpu(data['t'], data['y'], data['dy'], periods=data['periods'],
                          full=True, return_arrays=True, **options)


def public_batch(backend, cases):
    if backend in NATIVE_BACKENDS:
        return [public_single(backend, case) for case in cases]
    from cuvarbase.tls import tls_search_batch
    curves = [(case['data']['t'], case['data']['y'], case['data']['dy']) for case in cases]
    return tls_search_batch(curves, periods=cases[0]['data']['periods'],
                            full=True, return_arrays=True, **cases[0]['options'])


def fingerprint(backend, case, result):
    """Only call after the measurement barrier; hashes all returned arrays.

    The strict pool gate uses native period/chi2/power bytes including masks.
    A separate common representation compares public output units and ignores
    only data hidden by the scientific mask. It never replaces the strict gate.
    """
    values = vars(result) if backend in NATIVE_BACKENDS else result
    if backend == 'candidate' and 'error' in values:
        raise ValueError('Candidate returned an error result: ' + str(values['error']))
    for key in ('periods', 'power', 'chi2', 'period', 'SDE'):
        if key not in values:
            raise ValueError('Public result lacks ' + key)
    period, sde = float(values['period']), float(values['SDE'])
    if not np.isfinite(period) or not np.isfinite(sde):
        raise ValueError('Public result has no finite full-search detection')
    fields = {}
    for key, value in sorted(values.items()):
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            fields[key] = masked_hash(value)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float, np.number)):
            fields[key] = masked_hash(np.asarray(value))
        else:
            fields[key] = plain(value)
    strict = {key: masked_hash(values[key]) for key in ('periods', 'power', 'chi2')}
    strict['period'] = array_hash(np.array(period, dtype=np.float64))
    strict['SDE'] = array_hash(np.array(sde, dtype=np.float64))
    chi2 = np.ma.asarray(values['chi2'])
    mask = np.ma.getmaskarray(chi2) | ~np.isfinite(np.asarray(chi2.data))
    if backend == 'candidate':
        mask |= ~np.asarray(values['valid_periods'])
    common = {}
    for key in ('periods', 'power', 'chi2'):
        array = np.asarray(np.ma.getdata(values[key]), dtype=np.float64).copy()
        if key == 'chi2' and backend in NATIVE_BACKENDS:
            array /= case['error_scale']**2
        array[mask] = np.nan
        common[key] = array_hash(array)
    common['mask'] = array_hash(mask)
    common['period'], common['SDE'] = strict['period'], strict['SDE']
    return dict(case=case['name'], strict=strict, common=common, fields=fields,
                full_digest=hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                primary_period=period, SDE=sde, nperiods=len(chi2))


def environment():
    versions = {}
    for package in ('numpy', 'scipy', 'cupy-cuda12x', 'pycuda', 'batman-package', 'numba', 'pynvml'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    quota = {}
    for name in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu.stat',
                 '/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'):
        path = Path(name)
        if path.exists():
            quota[name] = path.read_text().strip()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,power.limit',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=False)
    cpu_quota = None
    if '/sys/fs/cgroup/cpu.max' in quota:
        amount, interval = quota['/sys/fs/cgroup/cpu.max'].split()
        if amount != 'max':
            cpu_quota = int(amount) / int(interval)
    threads = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cgroup=quota, cpu_quota_cores=cpu_quota,
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                cpu_math_thread_environment=threads,
                nvidia_smi=gpu.stdout.strip(), nvidia_smi_error=gpu.stderr.strip())
