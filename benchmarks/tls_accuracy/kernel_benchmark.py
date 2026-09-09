#!/usr/bin/env python3
"""Compare TLS empty-bin traversal with the same search using dense traversal.

This is an engineering performance/parity check, not a detection-sensitivity
study. Both variants use identical light curves, templates, trial grids and
statistics. Imports and compilation are excluded; each timed call includes
host preprocessing, transfers, the GPU search/refinement and host results.

Use --baseline-kernel to compile an unmodified, compatible TLS CUDA source as
the reference. Without it, the reference is the current kernel compiled with
TLS_SKIP_EMPTY_BINS=0. Input archives are those published with the September 8
transit benchmark. No cloud resources are created by this script.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    """Hash dtype, shape and C-order bytes so array identity is explicit."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    # Same convention as the archived TLS sensitivity study.
    h.update(array.dtype.str.encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def package_versions():
    versions = {}
    for name in ('numpy', 'scipy', 'pycuda', 'batman-package'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def compare(old, new, chi2_0, source_indices, rep):
    """Compare delta-chi-squared, avoiding a loose tolerance on total chi2."""
    rows = []
    for i, (a, b) in enumerate(zip(old, new)):
        old_score, new_score = chi2_0[i] - a['chi2'], chi2_0[i] - b['chi2']
        valid = np.isfinite(old_score) & np.isfinite(new_score)
        diff = np.abs(old_score[valid] - new_score[valid])
        rel = diff / np.maximum(np.abs(old_score[valid]), 1.)
        rows.append(dict(
            rep=rep, source_index=source_indices[i],
            same_valid=bool(np.array_equal(a['valid_periods'], b['valid_periods'])),
            n_valid_compared=int(valid.sum()),
            max_abs_score_diff=float(diff.max(initial=0.)),
            max_rel_score_diff=float(rel.max(initial=0.)),
            percentile99_rel_score_diff=float(np.percentile(rel, 99)) if len(rel) else 0.,
            same_period=bool(number(a['period']) == number(b['period'])),
            old_period=number(a['period']), new_period=number(b['period']),
            old_sde=number(a['SDE']), new_sde=number(b['SDE']),
            delta_sde=number(b['SDE'] - a['SDE']),
            old_snr=number(a['SNR']), new_snr=number(b['SNR']),
            changed_coarse_t0=int(np.count_nonzero(a['best_t0_per_period'][valid] != b['best_t0_per_period'][valid])),
            changed_coarse_duration=int(np.count_nonzero(a['best_duration_per_period'][valid] != b['best_duration_per_period'][valid]))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--baseline-kernel', type=Path,
                    help='unmodified baseline CUDA file with the same kernel ABI')
    ap.add_argument('--profile', default='ztf', choices=('ztf', 'tess_gap', 'tess_200s'))
    ap.add_argument('--nbins', type=int, default=8192,
                    help='fixed bins; 0 selects the public API automatic rule')
    ap.add_argument('--period-limit', type=int, default=4096,
                    help='evenly selected trial periods; 0 uses the full grid')
    ap.add_argument('--sources', type=int, default=4,
                    help='balanced injection/null subset, at most 256 sources')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--t0-oversample', type=float, default=16.)
    ap.add_argument('--n-durations', type=int, default=32)
    ap.add_argument('--refine-top-k', type=int, default=50)
    ap.add_argument('--refine-oversample', type=float, default=33.)
    ap.add_argument('--block-size', type=int, default=None)
    a = ap.parse_args()
    if not 2 <= a.sources <= 256:
        ap.error('--sources must be between 2 and 256')
    if a.reps < 1 or a.period_limit < 0 or a.nbins < 0:
        ap.error('reps must be positive; period-limit and nbins cannot be negative')

    from cuvarbase import tls
    from pycuda import driver as cuda

    tls.ensure_context()
    input_path = a.inputs / (a.profile + '_heldout.npz')
    indices = list(range(a.sources // 2)) + list(range(128, 128 + (a.sources + 1) // 2))
    with np.load(input_path) as data:
        lcs = [tuple(data[f'{k}_{i}'] for k in ('t', 'y', 'dy')) for i in indices]
        periods = np.asarray(data['tls_periods'], dtype=np.float64)
        truth = json.loads(str(data['metadata']))
    full_period_count = len(periods)
    if a.period_limit and len(periods) > a.period_limit:
        selected_periods = np.linspace(0, len(periods) - 1, a.period_limit).astype(int)
        periods = periods[selected_periods]
    else:
        selected_periods = np.arange(len(periods))
    # Frozen benchmark worker's qtransit formula. The public helper uses
    # slightly different stellar constants and is not substituted here.
    q = np.arcsin(np.minimum(1., (1. / (periods * 8.6307)) ** (2. / 3.))) / np.pi
    qmin, qmax = .5 * q, np.minimum(2. * q, .333)
    kwargs = dict(periods=periods, qmin=qmin, qmax=qmax,
                  n_durations=a.n_durations, t0_oversample=a.t0_oversample,
                  nbins=a.nbins or None, block_size=a.block_size,
                  refine_top_k=a.refine_top_k,
                  refine_oversample=a.refine_oversample,
                  limb_dark='quadratic', u=[.4804, .1867],
                  return_arrays=True)
    config = {k: v for k, v in kwargs.items() if k not in ('periods', 'qmin', 'qmax')}
    config.update(profile=a.profile, source_indices=indices,
                  n_periods=len(periods), full_period_count=full_period_count,
                  period_selection='all' if len(periods) == full_period_count else 'evenly spaced indices',
                  duration_window='q = arcsin(min(1, (1/(P*8.6307))**(2/3)))/pi; qmin = .5*q; qmax = min(2*q, .333)',
                  n_repetitions=a.reps)
    package = Path(tls.__file__).resolve().parent
    device = cuda.Context.get_device()
    output = dict(
        status='running', config=config,
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        profile=a.profile, nbins=a.nbins or None,
        n_periods=len(periods), source_indices=indices,
        ndata=[len(lc[0]) for lc in lcs],
        truth=[truth['cases'][i] for i in indices],
        input_sha256=sha(input_path),
        input_array_sha256=[{k: array_sha(v) for k, v in zip(('t', 'y', 'dy'), lc)} for lc in lcs],
        grid_sha256={k: array_sha(v) for k, v in dict(periods=periods, qmin=qmin, qmax=qmax,
                                                    selected_indices=selected_periods).items()},
        runner_sha256=sha(__file__),
        kernel_sha256=sha(tls.find_kernel('tls_fast')),
        baseline_kernel_sha256=sha(a.baseline_kernel) if a.baseline_kernel else None,
        reference='frozen CUDA source' if a.baseline_kernel else 'current CUDA source, dense traversal',
        installed_sources={str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*'))
                           if p.is_file() and p.suffix in ('.py', '.cu', '.cuh') and 'tests' not in p.parts},
        hardware=dict(gpu=device.name(), compute_capability=device.compute_capability(),
                      gpu_memory_bytes=int(device.total_memory()), cuda_driver=cuda.get_driver_version(),
                      platform=platform.platform(), python=sys.version),
        cpu_quota={str(p): p.read_text().strip() for p in (
            Path('/sys/fs/cgroup/cpu.max'), Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
            Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')) if p.exists()},
        packages=package_versions(),
        boundary='Prepared host arrays and explicit grid through complete host results, including preprocessing, allocations, transfers, synchronization, search, refinement and statistics.',
        exclusions='Imports, compilation, context initialization, grid creation, data generation and disk I/O.',
        repetitions=[], comparisons=[], original_repeat_diff=[])
    dump(a.out, output)

    reader = tls._module_reader
    original_getter = tls._get_cached_fast_kernels
    cache = {}
    mode = 0

    def get_kernels(block_size, nbins, t0_oversample, refine_nd=3):
        key = (mode, block_size, nbins, float(t0_oversample), refine_nd)
        if key not in cache:
            def variant_reader(*args, **kw):
                if mode == 0 and a.baseline_kernel:
                    return reader(str(a.baseline_kernel), *args[1:], **kw)
                return '#define TLS_SKIP_EMPTY_BINS %d\n' % mode + reader(*args, **kw)
            tls._module_reader = variant_reader
            try:
                cache[key] = tls.compile_tls_fast(block_size, nbins, t0_oversample, refine_nd)
            finally:
                tls._module_reader = reader
        return cache[key]

    tls._get_cached_fast_kernels = get_kernels
    elapsed = {0: [], 1: []}
    chi2_0 = tls._preprocess_batch(lcs)[6]
    reference_first = None
    try:
        for mode in (0, 1):
            tls.tls_search_batch(lcs, **kwargs)
            cuda.Context.synchronize()
        for rep in range(a.reps):
            pair = {}
            order = (0, 1) if rep % 2 == 0 else (1, 0)
            for mode in order:
                cuda.Context.synchronize()
                start = time.perf_counter()
                pair[mode] = tls.tls_search_batch(lcs, **kwargs)
                cuda.Context.synchronize()
                elapsed[mode].append(time.perf_counter() - start)
            if reference_first is None:
                reference_first = pair[0]
            elif rep == a.reps - 1:
                output['original_repeat_diff'] = compare(reference_first, pair[0], chi2_0, indices, rep)
            output['comparisons'].extend(compare(pair[0], pair[1], chi2_0, indices, rep))
            output['repetitions'].append(dict(rep=rep, order=list(order),
                                              elapsed_s={str(k): v[-1] for k, v in elapsed.items()}))
            dump(a.out, output)
        output.update(
            status='ok', elapsed_s={str(k): v for k, v in elapsed.items()},
            median_seconds_per_source={str(k): float(np.median(v) / len(lcs)) for k, v in elapsed.items()},
            speedup=float(np.median(elapsed[0]) / np.median(elapsed[1])),
            compiled_registers={str(k): {n: int(f.num_regs) for n, f in v.items()} for k, v in cache.items()})
        dump(a.out, output)
    except Exception:
        import traceback
        output.update(status='error', error=traceback.format_exc())
        dump(a.out, output)
        raise
    finally:
        tls._module_reader = reader
        tls._get_cached_fast_kernels = original_getter
    print(json.dumps({k: output[k] for k in ('status', 'profile', 'nbins', 'n_periods',
                                           'ndata', 'median_seconds_per_source', 'speedup')}, indent=2))


if __name__ == '__main__':
    main()
