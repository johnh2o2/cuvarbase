#!/usr/bin/env python3
"""Synchronized wall-phase profiling and isolated GTLS host-loop ablations.

Installed sources and CUDA kernels stay unchanged. Rebuilt Python functions
are retained separately; native runs are timed before instrumentation.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import sys
import textwrap
import time
import traceback
import warnings

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


class Profiler:
    def __init__(self, sync):
        self.sync = sync
        self.clear()

    def clear(self):
        self.rows = {}
        self.stack = []

    @contextmanager
    def segment(self, label):
        self.sync()
        state = [time.perf_counter(), 0.]
        self.stack.append(state)
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - state[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows.setdefault(label, dict(calls=0, inclusive_s=0., exclusive_s=0.))
            row['calls'] += 1
            row['inclusive_s'] += elapsed
            row['exclusive_s'] += elapsed - state[1]


def assign_name(node):
    if isinstance(node, ast.Assign):
        return ast.unparse(node.targets[0])
    return ''


def call_name(node):
    value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
    return ast.unparse(value.func) if isinstance(value, ast.Call) else ''


def segment_node(label, statements):
    return ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Name(id='_TLS_PROFILE', ctx=ast.Load()),
                           attr='segment', ctx=ast.Load()),
        args=[ast.Constant(label)], keywords=[]))], body=statements)


def partition(body, kind, scope='top'):
    """Wrap contiguous statement groups; preserve branches and evaluation order."""
    default = {'gtls': 'GTLS setup and allocations',
               'v1': 'v1 grid/configuration', 'power': 'GTLS input/template setup'}[kind]
    if scope == 'chunk':
        default = 'GTLS duration-mask union' if kind == 'gtls' else 'v1 lightcurve transfers'
    output, block, label = [], [], default

    def flush():
        nonlocal block
        if block:
            output.append(segment_node(label, block))
            block = []

    for node in body:
        name, call = assign_name(node), call_name(node)
        new = None
        is_chunk = isinstance(node, ast.For) and (
            (kind == 'gtls' and ast.unparse(node.target) == 'iterFlag') or
            (kind == 'v1' and ast.unparse(node.target) == '(i0, i1)'))
        if is_chunk:
            flush()
            node.body = partition(node.body, kind, 'chunk')
            output.append(node)
            label = 'GTLS result/statistics processing' if kind == 'gtls' else 'v1 final packaging'
            continue
        if kind == 'gtls':
            if scope == 'top':
                if name == 'GPUCode': new = 'GTLS CUDA module lookup/compile'
                elif name == '(durations, indices)': new = 'GTLS setup and allocations'
                elif name == 'raw_chi2': new = 'GTLS result/statistics processing'
                elif call == 'search_multi_periods_again': new = 'GTLS candidate refinement'
                elif call == 'search_single_periods': new = 'GTLS best-period fit/diagnostics'
                elif name.startswith('chi2['): new = 'GTLS result/statistics processing'
            else:
                if name == 'single_lc_arr': new = 'GTLS chunk allocations/transfers'
                elif name == 'fastFoldGPU': new = 'GTLS folding/sorting'
                elif name == 'patchDataGPU': new = 'GTLS reorder/weights'
                elif isinstance(node, ast.For) and ast.unparse(node.iter) == 'range(singleCalcPeriods)':
                    new = 'GTLS row-wise flux prefix sums'
                elif name == 'cumsumGPU[:]' or name == 'cumsumGPU[:, :]':
                    new = 'GTLS row-wise flux prefix sums'
                elif name == 'patchedDatasSize_local': new = 'GTLS error prefixes/out-of-transit terms'
                elif name == 'calcAllLowestResidualsGPU': new = 'GTLS transit residual kernel'
                elif name == 'start_idx': new = 'GTLS reductions/chunk cleanup'
        elif kind == 'v1':
            if scope == 'top':
                if name == 'band_launches': new = 'v1 kernel lookup/grid transfers'
                elif name == '(T_tab, S1_tab, S2_tab)': new = 'v1 template tables'
                elif call == '_preprocess_batch': new = 'v1 host lightcurve preparation'
                elif name == 'periods_g': new = 'v1 buffer allocations/transfers'
            else:
                if isinstance(node, ast.For) and ast.unparse(node.iter) == 'band_launches':
                    new = 'v1 coarse search kernel'
                elif name == 'score_h': new = 'v1 coarse spectrum transfer'
                elif name == 'rscore_h': new = 'v1 candidate selection/refinement/transfers'
                elif isinstance(node, ast.If) and ast.unparse(node.test).startswith('return_arrays'):
                    new = 'v1 parameter-spectrum transfers'
                elif isinstance(node, ast.FunctionDef) and node.name == '_finish_lc':
                    new = 'v1 CPU statistics/results'
        elif kind == 'power':
            if name == 'use_multi_gpu': new = 'GTLS full search call'
        if new and new != label:
            flush()
            label = new
        block.append(node)
    flush()
    return output


class Vectorize(ast.NodeTransformer):
    def __init__(self, variant):
        self.variant = variant
        self.changes = []

    def visit_For(self, node):
        self.generic_visit(node)
        if ast.unparse(node.iter) == 'range(start_idx + 1, end_idx)':
            assert len(node.body) == 1 and call_name(node.body[0]) == 'cp.logical_or'
            self.changes.append('Boolean duration union: loop to axis reduction')
            return ast.parse('temp_bool = cp.any(durationBoolArrayGPU[start_idx:end_idx], axis=0)').body[0]
        if self.variant == 'both' and ast.unparse(node.iter) == 'range(singleCalcPeriods)':
            if len(node.body) == 1 and call_name(node.body[0]) == 'cp.cumsum':
                self.changes.append('Flux prefix sums: row loop to batched axis scan')
                return ast.parse('cumsumGPU[:] = cp.cumsum(patchedDatasGPU, axis=1)').body[0]
        return node


def rebuild(owner, name, output, variant='native', instrument=False, kind=None, profiler=None):
    original = getattr(owner, name)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    fn = tree.body[0]
    changes = []
    if variant != 'native':
        transformer = Vectorize(variant)
        fn = transformer.visit(fn)
        changes = transformer.changes
        assert changes, 'Expected ablation sites were not found'
    if instrument:
        fn.body = partition(fn.body, kind)
    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree) + '\n'
    suffix = 'instrumented' if instrument else variant
    path = output.with_name(output.stem + f'.{name}.{suffix}.py')
    path.write_text(rendered)
    namespace = original.__globals__
    namespace['_TLS_PROFILE'] = profiler
    local = {}
    exec(compile(tree, str(path), 'exec'), namespace, local)
    setattr(owner, name, local[name])
    return dict(function=name, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                transformed_sha256=sha(path), file=path.name, changes=changes,
                instrumentation=instrument)


def output_arrays(results):
    arrays = {}
    scalars = []
    for i, r in enumerate(results):
        get = lambda name: r[name] if hasattr(r, '__getitem__') else getattr(r, name)
        for name in ['periods', 'chi2']:
            arrays[f'{name}_{i}'] = np.asarray(np.ma.filled(get(name), np.nan), dtype=np.float64)
        scalars.append({name: float(get(name)) for name in ['period', 'SDE']})
    return arrays, scalars


def compare(a, b):
    checks = {}
    for name in a:
        aa, bb = a[name], b[name]
        same_mask = bool(np.array_equal(np.isfinite(aa), np.isfinite(bb)))
        finite = np.isfinite(aa) & np.isfinite(bb)
        checks[name] = dict(same_shape=aa.shape == bb.shape, same_finite_mask=same_mask,
                           exact=bool(np.array_equal(aa, bb, equal_nan=True)),
                           max_abs=float(np.max(np.abs(aa[finite] - bb[finite]))) if finite.any() else None)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['gtls', 'v1'], required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=['native', 'union', 'both'], default='native')
    ap.add_argument('--nsource', type=int, default=1)
    ap.add_argument('--offset', type=int, default=4)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--profile-reps', type=int, default=1)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    periods = data['periods']
    lcs_raw = [(data[f't_{i}'], data[f'y_{i}'], data[f'dy_{i}'], data[f'scale_{i}'])
               for i in range(args.offset, args.offset + args.nsource)]

    def prepare():
        result = []
        for t, y, dy, scale in lcs_raw:
            order = np.argsort(t)
            result.append((t[order], (y / scale)[order], (dy / scale)[order]))
        return result

    record = dict(status='running', args=vars(args), input_sha256=sha(args.input),
                  runner_sha256=sha(__file__), python=sys.version, platform=platform.platform(),
                  source_files={}, transformations=[],
                  interpretation='New-host component diagnostic. Native API medians are uninstrumented. '
                                 'Phase times include synchronization and instrumentation overhead; '
                                 'ablations are diagnostic Python changes, not released GTLS.')
    write(args.out, record)
    try:
        if args.backend == 'gtls':
            import cupy as cp
            import gputls
            from gputls import gtls
            core = importlib.import_module('gputls.core')
            cp.cuda.Device(0).use()
            sync = cp.cuda.runtime.deviceSynchronize
            package = Path(gputls.__file__).parent
            for p in sorted(package.rglob('*.py')):
                record['source_files'][str(p.relative_to(package))] = sha(p)
            if args.variant != 'native':
                record['transformations'].append(rebuild(core, 'search_multi_periods', args.out,
                                                          variant=args.variant))

            def run():
                return [gtls(*lc, verbose=False).power(
                    periods=periods, R_star=1, M_star=1, oversampling_factor=3,
                    T0_fit_margin=.125, duration_grid_step=1.1,
                    transit_template='default', verbose=False, show_progress_bar=False)
                    for lc in prepare()]
        else:
            from cuvarbase import tls
            from cuvarbase.base import ensure_context
            import pycuda.driver as drv
            ensure_context()
            sync = drv.Context.synchronize
            package = Path(tls.__file__).parent
            for p in sorted(package.rglob('*')):
                if p.suffix in ['.py', '.cu', '.cuh']:
                    record['source_files'][str(p.relative_to(package))] = sha(p)

            binned_name = '_tls_search_batch_binned' if hasattr(tls, '_tls_search_batch_binned') else 'tls_search_batch'

            def run():
                return getattr(tls, binned_name)(prepare(), periods=periods,
                    qmin=data['qmin'], qmax=data['qmax'], n_durations=38,
                    t0_oversample=8, refine_top_k=50, refine_oversample=33,
                    R_star=1, M_star=1, oversampling_factor=3, return_arrays=True,
                    u=[.4804, .1867])
        sync()
        first_start = time.perf_counter()
        first = run()
        sync()
        record['first_api_s'] = time.perf_counter() - first_start
        baseline_arrays, baseline_scalars = output_arrays(first)
        times = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for _ in range(args.reps):
                sync()
                start = time.perf_counter()
                result = run()
                sync()
                times.append(time.perf_counter() - start)
        record['warnings'] = sorted(set(str(w.message) for w in caught))
        arrays, scalars = output_arrays(result)
        np.savez_compressed(args.out.with_suffix('.npz'), **arrays)
        record.update(native_times_s=times, native_median_s=float(np.median(times)),
                      native_results=scalars, first_vs_last=compare(baseline_arrays, arrays),
                      native_outputs_finite=all(np.isfinite(v) for r in scalars for v in r.values()))
        write(args.out, record)
        profiler = Profiler(sync)
        if args.profile_reps:
            if args.backend == 'gtls':
                record['transformations'].append(rebuild(core, 'search_multi_periods', args.out,
                    instrument=True, kind='gtls', profiler=profiler))
                record['transformations'].append(rebuild(gtls, 'power', args.out,
                    instrument=True, kind='power', profiler=profiler))
            else:
                record['transformations'].append(rebuild(tls, binned_name, args.out,
                    instrument=True, kind='v1', profiler=profiler))
        profiles = []
        for _ in range(args.profile_reps):
            profiler.clear()
            start = time.perf_counter()
            with profiler.segment('API preparation/remaining host work'):
                result = run()
            elapsed = time.perf_counter() - start
            profile_arrays, profile_scalars = output_arrays(result)
            profiles.append(dict(total_s=elapsed, phases=profiler.rows,
                                 output_comparison=compare(arrays, profile_arrays), results=profile_scalars))
        record.update(status='ok', profiles=profiles, output_file_sha256=sha(args.out.with_suffix('.npz')))
    except Exception:
        record.update(status='error', error=traceback.format_exc())
    write(args.out, record)
    print(json.dumps({k: record[k] for k in ['status', 'native_median_s', 'error'] if k in record}), flush=True)
    if record['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
