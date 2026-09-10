#!/usr/bin/env python3
"""Complete TLS differential runner with explicit native-reference provenance.

The reproduction runner invokes backends serially in one process and frees
unused CuPy pool blocks before each search. Native GTLS is instrumented only by
wrapping CPU postprocessing/profiling returns; the numerical source and CUDA
code are unchanged. Recorded wall time is diagnostic accounting, not a
headline performance benchmark.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps(value.dtype.descr if value.dtype.names else value.dtype.str).encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sources(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.cu', '.cuh') and '__pycache__' not in p.parts}


def production_sources(root):
    root = Path(root)
    paths = list((root/'cuvarbase').glob('*.py'))
    paths += list((root/'cuvarbase/kernels').glob('*.cu'))
    paths += list((root/'cuvarbase/kernels').glob('*.cuh'))
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths) if p.is_file()}


def put_masked(arrays, name, value):
    value = np.ma.asarray(value)
    arrays[name] = np.asarray(value.data).copy()
    arrays[name+'_mask'] = np.ma.getmaskarray(value).copy()


def native_gpu_locals(local, arrays, prefix):
    """Small logical-domain outputs retained from the actual native frame."""
    metadata = {}
    for key in ('singleCalcPeriods', 'tSize', 'patchedDatasSize', 'maxDuration', 'TotalIter'):
        if key in local:
            metadata[key] = int(local[key])
    for source, target in [('durationsGridCollectionGPU', 'chunk_width_masks'),
                           ('fulldurationsMinGPU', 'minimum_width'),
                           ('fulldurationsMaxGPU', 'maximum_width'),
                           ('locationGPU', 'winning_local_flat_index')]:
        if source in local:
            arrays[prefix+target] = local[source].get()
    if 'durations' in local:
        arrays[prefix+'widths'] = np.asarray(local['durations']).copy()
    if 'periods' in local:
        arrays[prefix+'periods'] = np.asarray(local['periods']).copy()
    return metadata


def normalize_native_winners(arrays, prefix, ndata, group_size, chi2):
    """Decode a native chunk-local duration/start without reading undefined rows."""
    locations = arrays[prefix+'winning_local_flat_index']
    masks, widths = arrays[prefix+'chunk_width_masks'], arrays[prefix+'widths']
    start, width = np.full(len(chi2), -1, np.int32), np.full(len(chi2), -1, np.int32)
    for i, value in enumerate(chi2):
        eligible = widths[masks[i//group_size]]
        if np.isfinite(value) and len(eligible):
            local = int(locations[i])
            if not 0 <= local < len(eligible)*ndata:
                raise ValueError('Native finite residual has invalid logical winner')
            start[i], width[i] = local % ndata, eligible[local//ndata]
    arrays[prefix+'winning_start'], arrays[prefix+'winning_width'] = start, width


def run_gtls(data, options, mode, auto_grid=False, case_search_kwargs=None, correction=False):
    import cupy as cp
    import gputls
    from gputls import core, gtls, constants

    arrays, stages, refinements, final_fit = {}, [], [], {}
    model = gtls(data['t'], data['y'], data['dy'], verbose=False)
    for key in ('t', 'y', 'dy'):
        arrays['prepared_'+key] = np.asarray(getattr(model, key)).copy()
    original_spectra = core.spectra
    previous_profile = sys.getprofile()

    def spectra_spy(chi2, oversampling_factor):
        # Inspect actual locals at the point the first GPU residual vector
        # has already reached the host, before full mode deletes its buffers.
        caller = inspect.currentframe().f_back
        stage = len(stages)
        prefix = 'stage%d_' % stage
        details = native_gpu_locals(caller.f_locals, arrays, prefix) if stage == 0 else {}
        put_masked(arrays, prefix+'chi2', chi2)
        result = original_spectra(chi2, oversampling_factor)
        for name, value in zip(('SR', 'power_raw', 'power'), result[:3]):
            put_masked(arrays, prefix+name, value)
        if stage == 2 and 'period' in caller.f_locals:
            prior = caller.f_locals['period']
            details['preceding_selected_period_masked'] = bool(np.ma.is_masked(prior))
            details['preceding_selected_period_finite'] = bool(not np.ma.is_masked(prior) and np.isfinite(prior))
        stages.append(dict(index=stage, SDE_raw=plain(result[3]), SDE=plain(result[4]), **details))
        for key in ('possiblePeriodsIndices', 'possiblePeriods',
                    'possiblePeriodsIndices_multi', 'possiblePeriods_multi'):
            if key in caller.f_locals:
                arrays[prefix+key] = np.asarray(caller.f_locals[key]).copy()
        return result

    def profile(frame, event, value):
        if (event == 'return' and value is not None and frame.f_code.co_filename == core.__file__ and
                frame.f_code.co_name == 'search_multi_periods_again'):
            prefix = 'refinement%d_' % len(refinements)
            details = native_gpu_locals(frame.f_locals, arrays, prefix)
            if value is not None:
                arrays[prefix+'chi2'] = np.asarray(value).copy()
            refinements.append(details)
        elif (event == 'return' and value is not None and frame.f_code.co_filename == core.__file__ and
                frame.f_code.co_name == 'search_single_periods'):
            local = frame.f_locals
            location = int(local['bestLocation'])
            arrays['final_chi2'] = local['lowestResidualsGPU'].ravel()[location:location+1].get()
            arrays['final_winning_start'] = np.array([location % len(local['t'])], np.int32)
            arrays['final_winning_width'] = np.array([local['durationPointsNum']], np.int32)
            for key, item in zip(('fractional_duration', 'width_in_samples', 'duration', 'depth', 'T0',
                                  'transit_times', 'native_gtls_snr', 'native_gtls_snr_pink',
                                  'native_gtls_snrFit', 'native_gtls_snrFitPink'), value):
                final_fit[key] = plain(item)

    core.spectra = spectra_spy
    sys.setprofile(profile)
    try:
        kwargs = dict(case_search_kwargs or {})
        kwargs.update(options.get('gtls_kwargs', {}))
        # These are the public reference's stellar-input validation bounds.
        # Its duration cache and CUDA domain remain the pinned hardcoded
        # defaults; accepting a supplied dense host does not widen those.
        for parameter, default_min, default_max in (
                ('R_star', constants.R_STAR_MIN, constants.R_STAR_MAX),
                ('M_star', constants.M_STAR_MIN, constants.M_STAR_MAX)):
            value = kwargs.get(parameter, 1.)
            kwargs.setdefault(parameter+'_min', min(default_min, value))
            kwargs.setdefault(parameter+'_max', max(default_max, value))
        kwargs.update(fast=mode == 'fast', verbose=False, show_progress_bar=False)
        if correction:
            import corrected_reference
            with corrected_reference.apply(core) as correction_provenance:
                result = model.power(periods=[] if auto_grid else data['periods'], **kwargs)
        else:
            correction_provenance = None
            result = model.power(periods=[] if auto_grid else data['periods'], **kwargs)
    finally:
        core.spectra = original_spectra
        sys.setprofile(previous_profile)
    if not stages:
        raise RuntimeError('GTLS did not produce a captured native spectrum')
    if mode == 'fast':
        periods, power = result
        found = None
        extra = {}
    else:
        periods, power, found = result.periods, result.power, result.period
        extra = {key: plain(getattr(result, key)) for key in
                 ('duration', 'rawDuration', 'depth', 'T0', 'SDE', 'snr', 'snr_pink')
                 if hasattr(result, key)}
        for key in ('possiblePeriodsIndices', 'possiblePeriods'):
            if hasattr(result, key):
                arrays[key] = np.asarray(getattr(result, key)).copy()
    put_masked(arrays, 'periods', periods)
    put_masked(arrays, 'power', power)
    final = 'stage%d_' % (len(stages)-1)
    arrays['chi2'], arrays['chi2_mask'] = arrays[final+'chi2'], arrays[final+'chi2_mask']
    arrays['coarse_chi2'], arrays['coarse_chi2_mask'] = arrays['stage0_chi2'], arrays['stage0_chi2_mask']
    normalize_native_winners(arrays, 'stage0_', len(model.t), stages[0]['singleCalcPeriods'], arrays['coarse_chi2'])
    for i, details in enumerate(refinements):
        normalize_native_winners(arrays, 'refinement%d_' % i, len(model.t),
                                 details['singleCalcPeriods'], arrays['refinement%d_chi2' % i])
    if mode == 'full':
        arrays['refinement_indices'] = arrays['stage1_possiblePeriodsIndices'].astype(np.int64)
        arrays['harmonic_indices'] = arrays['stage2_possiblePeriodsIndices_multi'].astype(np.int64)
    overview = np.asarray(model.lc_cache_overview)
    widths, unique = np.unique(overview['width_in_samples'], return_index=True)
    curves = [np.asarray(model.lc_arr[i]) for i in unique]
    arrays['cache_overview'] = overview.copy()
    arrays['cache_widths'] = widths.astype(np.int32)
    arrays['cache_overshoot'] = np.asarray(overview['overshoot'][unique], dtype=np.float32)
    arrays['cache_signal_lengths'] = np.array([len(c) for c in curves], dtype=np.int32)
    arrays['cache_template_deficits'] = np.asarray(
        1-np.array([np.pad(c, (0, int(max(widths))-len(c)), 'constant') for c in curves]),
        dtype=np.float32)
    selected = np.where(~arrays['power_mask'], arrays['power'], np.nan)
    global_index = int(np.nanargmax(selected)) if np.any(np.isfinite(selected)) else None
    index = global_index if found is None else int(np.flatnonzero(arrays['periods'] == found)[0])
    if found is None and index is not None:
        found = float(arrays['periods'][index])
    valid_input = np.ones(len(data['t']), bool)
    for field in ('t', 'y', 'dy'):
        valid_input &= np.isfinite(data[field]) & (data[field] > 0)
    error_scale = float(np.mean(data['dy'][valid_input]))
    return arrays, dict(primary_index=index, period=plain(found),
                        score=plain(stages[-1]['SDE']), global_power_primary_index=global_index,
                        error_scale=error_scale,
                        effective_search_kwargs=plain(kwargs), reference_correction=correction_provenance,
                        stages=stages, refinements=refinements,
                        group_size=stages[0].get('singleCalcPeriods'), extra=extra, final_fit=final_fit,
                        package_sources=sources(Path(gputls.__file__).parent),
                        reference_instrumentation='CPU spectra wrapper and refinement-return profile; CUDA source unchanged')


def run_candidate(data, options, mode, engine_root, reference_record, work_chunk, chunk_policy='replay',
                  engine_kind='public', auto_grid=False, case_search_kwargs=None):
    sys.path.insert(0, str(engine_root))
    engine = importlib.import_module('cuvarbase.tls_reference')
    function = getattr(engine, 'search_'+mode)
    kwargs = dict(options.get('candidate_kwargs', {}))
    kwargs.setdefault('work_chunk', work_chunk)
    if reference_record is not None and chunk_policy == 'replay' and engine_kind != 'public':
        kwargs['group_size'] = reference_record['result']['group_size']
    public = None
    if engine_kind == 'public':
        if chunk_policy != 'default':
            raise ValueError('The public API must use its actual default logical-group policy')
        public_api = importlib.import_module('cuvarbase.tls')
        capture = []
        def runner_spy(*args, **options):
            result = function(*args, **options)
            capture.append(result)
            return result
        setattr(engine, 'search_'+mode, runner_spy)
        try:
            public_kwargs = dict(case_search_kwargs or {})
            public_kwargs.update(kwargs)
            public = public_api.tls_search_gpu(data['t'], data['y'], data['dy'],
                periods=None if auto_grid else data['periods'], full=mode == 'full', **public_kwargs)
        finally:
            setattr(engine, 'search_'+mode, function)
        if len(capture) != 1:
            raise RuntimeError('Public API did not invoke its default numerical engine exactly once')
        result = capture[0]
    else:
        if auto_grid:
            raise ValueError('Automatic-grid dispatch must be tested through the public API')
        result = function(data['t'], data['y'], data['dy'], data['periods'], **kwargs)
    arrays = {}
    for key in ('t', 'y', 'dy'):
        arrays['prepared_'+key] = np.asarray(result['prepared'][key]).copy()
    for key in ('overview', 'widths', 'overshoot', 'signal_lengths', 'template_deficits'):
        arrays['cache_'+key] = np.asarray(result['cache'][key]).copy()
    put_masked(arrays, 'periods', np.ma.array(result['periods'],
                 mask=np.ma.getmaskarray(result['spectra']['chi2'])))
    put_masked(arrays, 'power', result['spectra']['power'])
    put_masked(arrays, 'chi2', result['spectra']['chi2'])
    raw = result.get('coarse_raw', result['raw'])
    arrays['coarse_chi2'] = np.asarray(raw['chi2']).copy()
    arrays['coarse_chi2_mask'] = np.ma.getmaskarray(result.get('coarse_spectra', result['spectra'])['chi2']).copy()
    for source, target in [('minima', 'minimum_width'), ('maxima', 'maximum_width'),
                           ('width_masks', 'chunk_width_masks'), ('start', 'winning_start'),
                           ('width', 'winning_width')]:
        if source in raw:
            arrays['stage0_'+target] = np.asarray(raw[source]).copy()
    arrays['stage0_widths'] = arrays['cache_widths'].copy()
    arrays['stage0_periods'] = arrays['periods'].copy()
    for key, target in (('candidates', 'refinement_indices'), ('harmonics', 'harmonic_indices')):
        if key in result:
            arrays[target] = np.asarray(result[key], dtype=np.int64).copy()
    for i, key in enumerate(('refined', 'harmonic_results')):
        if key in result:
            refined = result[key]
            for source, target in (('chi2', 'chi2'), ('start', 'winning_start'), ('width', 'winning_width'),
                                   ('minima', 'minimum_width'), ('maxima', 'maximum_width'),
                                   ('width_masks', 'chunk_width_masks')):
                arrays['refinement%d_%s' % (i, target)] = np.asarray(refined[source]).copy()
    final_fit = {}
    if 'final' in result:
        final = result['final']
        arrays['final_chi2'] = np.asarray(final['chi2']).copy()
        arrays['final_winning_start'] = np.asarray(final['start']).copy()
        arrays['final_winning_width'] = np.asarray(final['width']).copy()
        final_fit = engine.reference.final_parameters(
            result['prepared']['t'], result['prepared']['y'], result['prepared']['dy'],
            result['period'], result['cache'], int(final['width_index'][0]), int(final['start'][0]),
            fit_chi2=float(final['chi2'][0]), error_scale=result['prepared']['error_scale'])
    history = result.get('spectra_history', [result['spectra']])
    for i, stage in enumerate(history):
        for key in ('chi2', 'SR', 'power_raw', 'power'):
            if key in stage:
                put_masked(arrays, 'stage%d_%s' % (i, key), stage[key])
    public_contract = None
    if public is not None:
        public_order = np.argsort(public['periods']) if 'periods' in public else None
        for key in ('periods', 'chi2', 'power', 'SR', 'valid_periods'):
            if key in public:
                arrays['public_'+key] = np.asarray(public[key])[public_order].copy()
        public_contract = {key: plain(public.get(key)) for key in
            ('period', 'T0', 'duration', 'depth', 'fractional_duration', 'SDE', 'SDE_raw', 'SNR',
             'chi2_min', 'chi2_null', 'native_gtls_snr', 'search_configuration')}
    index = result.get('primary_index', result['spectra']['primary_index'])
    score = result['spectra']['SDE']
    return arrays, dict(primary_index=index, period=plain(result['period']), score=plain(score),
                        global_power_primary_index=result['spectra']['primary_index'], final_fit=plain(final_fit),
                        public_contract=public_contract,
                        error_scale=float(result['prepared']['error_scale']),
                        public_capture='Actual public API; selected numerical runner wrapped only to retain its unchanged return value' if public else None,
                        group_size=int(raw['group_size']), work_chunk=int(raw['work_chunk']),
                        stages=[dict(index=i, SDE=plain(s.get('SDE'))) for i, s in enumerate(history)])


def run(args):
    import cupy as cp
    input_path = args.case.resolve()
    with np.load(input_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files if key != 'metadata'}
        metadata = json.loads(str(source['metadata']))
    positive_origin = getattr(args, 'positive_origin', False)
    engine_kind = getattr(args, 'engine_kind', 'public')
    auto_grid = getattr(args, 'auto_grid', False)
    if positive_origin:
        finite_times = data['t'][np.isfinite(data['t'])]
        if not len(finite_times):
            raise ValueError('Common positive-origin execution requires at least one finite time')
        shift = float(np.floor(np.min(finite_times))-1.)
        data['t'] = data['t']-shift
        metadata = dict(metadata, execution_time_shift_days=shift,
                        truth_epoch=metadata['truth_epoch']-shift,
                        execution_preprocessing='Same positive-origin arrays supplied to native and candidate')
    metadata = dict(metadata, auto_grid_api_exercised=auto_grid)
    if getattr(args, 'replay', False):
        metadata = dict(metadata, replayed_original_cohort=metadata.get('cohort'), cohort='reproduction',
                        interpretation='Reproduction of frozen inputs; never a new independent holdout')
    options = json.loads(args.options.read_text()) if args.options else {}
    seal = json.loads(args.seal.read_text()) if args.seal else None
    if getattr(args, 'replay', False) and seal is not None:
        if (options != seal['options'] or args.mode not in seal['modes'] or
                engine_kind != seal['engine_kind'] or args.chunk_policy != seal['chunk_policy'] or
                auto_grid != seal['auto_grid']):
            raise ValueError('Reproduction options differ from the published experiment')
        if production_sources(args.engine_root) != seal['source_identity']['production_sources']:
            raise ValueError('Production source differs from the published seal; use the recorded source snapshot/revision')
    if metadata.get('cohort') == 'heldout':
        if seal is None or metadata.get('seal_sha256') != sha(args.seal):
            raise ValueError('Heldout execution requires the exact seal used to generate inputs')
        if args.mode not in seal['modes'] or options != seal['options']:
            raise ValueError('Heldout mode/options differ from the frozen protocol')
        if (engine_kind != seal.get('engine_kind') or args.chunk_policy != seal.get('chunk_policy') or
                auto_grid != seal.get('auto_grid')):
            raise ValueError('Heldout API/group/grid mode differs from the frozen protocol')
        identity = seal['source_identity']
        if sha(__file__) != identity['harness_sha256']:
            raise ValueError('Harness changed after holdout freeze')
        for name, expected_hash in identity['validation_sources'].items():
            if sha(Path(__file__).parent/name) != expected_hash:
                raise ValueError('Validation source changed after freeze: '+name)
        if args.backend == 'gtls_corrected' and seal.get('reference_correction') != 'finite_candidates_before_ranking_v1':
            raise ValueError('Corrected reference is not declared in this seal')
        actual_sources = production_sources(args.engine_root)
        frozen_sources = identity['production_sources']
        if actual_sources != frozen_sources:
            raise ValueError('Candidate numerical source changed after holdout freeze')
    reference = json.loads(args.reference_record.read_text()) if args.reference_record else None
    if reference and reference['input_sha256'] != sha(input_path):
        raise ValueError('Reference input differs from candidate input')
    source_getter = production_sources
    identity = dict(input_sha256=sha(input_path), input_metadata=metadata,
                    input_arrays={key: array_hash(value) for key, value in data.items()},
                    backend=args.backend, mode=args.mode, options=options, chunk_policy=args.chunk_policy,
                    engine_kind=engine_kind, positive_origin=positive_origin, auto_grid=auto_grid,
                    seal_sha256=sha(args.seal) if args.seal else None,
                    harness_sha256=sha(__file__), reference_record_sha256=sha(args.reference_record) if reference else None,
                    engine_sources=source_getter(args.engine_root) if args.backend == 'candidate' else None)
    if reference and reference.get('input_arrays') != identity['input_arrays']:
        raise ValueError('Actual execution arrays differ after input preprocessing')
    if args.out.exists():
        raise ValueError('Output exists; refuse to overwrite: '+str(args.out))
    args.out.mkdir(parents=True)
    write(args.out/'record.json', dict(identity, status='running'))
    # A fresh pool avoids a previous case's retained temporary buffers
    # silently changing the reference's duration-union group size.
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    start = time.perf_counter()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if args.backend in ('gtls', 'gtls_corrected'):
                arrays, result = run_gtls(data, options, args.mode, auto_grid,
                                          metadata.get('search_kwargs'), correction=args.backend == 'gtls_corrected')
                if seal is not None and result['package_sources'] != seal['reference_package_sources']:
                    raise ValueError('Native GTLS source differs from the frozen reference package')
            else:
                arrays, result = run_candidate(data, options, args.mode, args.engine_root,
                                               reference, args.work_chunk, args.chunk_policy,
                                               engine_kind, auto_grid, metadata.get('search_kwargs'))
            cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter()-start
        path = args.out/'arrays.npz'
        np.savez_compressed(path, **arrays)
        record = dict(identity, status='ok', result=result, elapsed_seconds=elapsed,
                      free_memory_before=int(free_before), total_memory=int(total_memory),
                      arrays_file='arrays.npz', arrays_sha256=sha(path),
                      arrays={key: dict(sha256=array_hash(value), dtype=str(value.dtype), shape=value.shape)
                              for key, value in arrays.items()},
                      warnings=sorted(set(str(v.message) for v in caught)))
    except Exception:
        record = dict(identity, status='error', elapsed_seconds=time.perf_counter()-start,
                      free_memory_before=int(free_before), total_memory=int(total_memory),
                      error=traceback.format_exc(), warnings=sorted(set(str(v.message) for v in caught)))
    if args.backend == 'candidate' and record['engine_sources'] != source_getter(args.engine_root):
        record.update(status='error', error='Candidate source files changed during execution')
    write(args.out/'record.json', record)
    print(json.dumps(dict(status=record['status'], output=str(args.out), elapsed_seconds=record['elapsed_seconds'])))
    if record['status'] != 'ok':
        raise RuntimeError(record['error'])


def compare_array(a, b, atol, rtol):
    if a.shape != b.shape:
        return dict(shape_equal=False, passed=False, shape_reference=a.shape, shape_candidate=b.shape)
    if a.dtype.names or a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
        equal = bool(np.array_equal(a, b))
        return dict(shape_equal=True, dtype_equal=a.dtype == b.dtype, bitwise=equal, passed=equal)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    classifications = bool(np.array_equal(np.isnan(a), np.isnan(b)) and
                           np.array_equal(np.isposinf(a), np.isposinf(b)) and
                           np.array_equal(np.isneginf(a), np.isneginf(b)))
    take = finite_a & finite_b
    error = np.abs(a[take].astype(np.float64)-b[take].astype(np.float64))
    allowance = atol+rtol*np.abs(a[take].astype(np.float64))
    bitwise = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return dict(shape_equal=True, dtype_equal=a.dtype == b.dtype, bitwise=bitwise,
                same_nonfinite_classification=classifications,
                max_abs_error=float(error.max()) if len(error) else 0.,
                changed_finite_cells=int(np.sum(error != 0)), compared_finite_cells=int(take.sum()),
                outside_tolerance=int(np.sum(error > allowance)),
                passed=classifications and bool(np.all(error <= allowance)))


def compare(args):
    rpath, cpath = args.reference, args.candidate
    reference, candidate = (json.loads(p.read_text()) for p in (rpath, cpath))
    if reference['input_sha256'] != candidate['input_sha256'] or reference['mode'] != candidate['mode']:
        raise ValueError('Different inputs or GTLS modes cannot be parity-compared')
    if reference.get('input_arrays') != candidate.get('input_arrays'):
        raise ValueError('Different actual arrays cannot be parity-compared')
    if reference['status'] != 'ok' or candidate['status'] != 'ok':
        write(args.out, dict(status='execution_failure', reference_status=reference['status'],
                             candidate_status=candidate['status'], passed=False,
                             reference_failure_extension=reference['status'] == 'error' and candidate['status'] == 'ok',
                             warning='Reference failures and candidate extensions do not establish numerical parity.'))
        return
    for path, record in ((rpath, reference), (cpath, candidate)):
        if sha(path.parent/record['arrays_file']) != record['arrays_sha256']:
            raise ValueError('Result arrays hash mismatch')
    gates = json.loads(args.gates.read_text()) if args.gates else {}
    exact = ('periods', 'periods_mask', 'prepared_t', 'prepared_y', 'prepared_dy',
             'cache_widths', 'cache_template_deficits', 'cache_overshoot', 'cache_signal_lengths',
             'chi2_mask', 'power_mask', 'coarse_chi2_mask', 'stage0_minimum_width',
             'stage0_maximum_width', 'stage0_chunk_width_masks', 'stage0_winning_start', 'stage0_winning_width')
    numeric = ('coarse_chi2', 'chi2', 'power')
    if reference['mode'] == 'full':
        exact += ('refinement_indices', 'harmonic_indices', 'final_winning_start', 'final_winning_width')
        numeric += ('final_chi2',)
        for i in range(2):
            exact += tuple('refinement%d_%s' % (i, field) for field in
                           ('winning_start', 'winning_width', 'minimum_width', 'maximum_width', 'chunk_width_masks'))
            numeric += ('refinement%d_chi2' % i,)
    checks, public_checks = {}, {}
    with np.load(
            rpath.parent/reference['arrays_file'], allow_pickle=False) as r, np.load(
            cpath.parent/candidate['arrays_file'], allow_pickle=False) as c:
        for name in exact+numeric:
            if name not in r or name not in c:
                checks[name] = dict(passed=False, missing_reference=name not in r, missing_candidate=name not in c)
                continue
            gate = {} if name in exact else gates.get(name, gates.get('chi2', {}) if name.endswith('chi2') else {})
            checks[name] = compare_array(r[name], c[name], float(gate.get('atol', 0)), float(gate.get('rtol', 0)))
        # Full-stage evidence must be explicitly present. Matching only the
        # final maximum cannot substitute for matching candidate/refinement paths.
        stage_counts = (len(reference['result']['stages']), len(candidate['result']['stages']))
        for i in range(stage_counts[0]):
            for field in ('chi2', 'chi2_mask', 'power', 'power_mask'):
                name = 'stage%d_%s' % (i, field)
                if name in r and name in c:
                    gate = gates.get('chi2' if field.startswith('chi2') else 'power', {})
                    checks[name] = compare_array(r[name], c[name], float(gate.get('atol', 0)), float(gate.get('rtol', 0)))
                else:
                    checks[name] = dict(passed=False, missing_reference=name not in r, missing_candidate=name not in c)
        rp = np.where(r['power_mask'], np.nan, r['power']).astype(float)
        cp = np.where(c['power_mask'], np.nan, c['power']).astype(float)
        finite = np.isfinite(rp) & np.isfinite(cp)
        epsilon = float(np.max(np.abs(rp[finite]-cp[finite]))) if finite.any() else 0.
        rid, cid = reference['result']['primary_index'], candidate['result']['primary_index']
        ordered = np.sort(rp[np.isfinite(rp)])
        margin = float(ordered[-1]-ordered[-2]) if len(ordered) >= 2 else None
        ranking = dict(reference_index=rid, candidate_index=cid, same_primary=rid == cid,
                       reference_period=reference['result']['period'], candidate_period=candidate['result']['period'],
                       same_period=reference['result']['period'] == candidate['result']['period'],
                       reference_margin=margin, max_power_error=epsilon,
                       changed_within_error_band=rid != cid and margin is not None and margin <= 2*epsilon)
        thresholds = []
        for threshold in args.threshold:
            rs, cs = reference['result']['score'], candidate['result']['score']
            rd = rs is not None and rs > threshold
            cd = cs is not None and cs > threshold
            thresholds.append(dict(threshold=threshold, reference_above=rd, candidate_above=cd,
                                   same_decision=rd == cd,
                                   reference_margin=None if rs is None else abs(rs-threshold)))
        if candidate.get('engine_kind') == 'public':
            scale = reference['result']['error_scale']
            expected = dict(periods=r['periods'],
                chi2=np.where(r['chi2_mask'], np.nan, r['chi2']).astype(np.float64)/scale**2,
                power=np.where(r['power_mask'], np.nan, r['power']),
                SR=np.where(r['stage%d_SR_mask' % (stage_counts[0]-1)], np.nan,
                            r['stage%d_SR' % (stage_counts[0]-1)]),
                valid_periods=np.isfinite(r['chi2']) & ~r['chi2_mask'])
            for field, value in expected.items():
                name = 'public_'+field
                if name not in c:
                    public_checks[field] = dict(passed=False, missing_candidate=True)
                    continue
                gate = gates.get(name, {})
                public_checks[field] = compare_array(np.asarray(value), c[name],
                    float(gate.get('atol', 0)), float(gate.get('rtol', 0)))
            contract = candidate['result'].get('public_contract') or {}
            final = reference['result'].get('final_fit', {})
            for key, value in dict(period=reference['result']['period'], SDE=reference['result']['score'],
                                   **{k: v for k, v in final.items() if k in
                                      ('T0', 'duration', 'depth', 'fractional_duration', 'native_gtls_snr')}).items():
                if key not in contract:
                    public_checks[key] = dict(passed=False, missing_candidate=True)
                elif value is None or contract[key] is None:
                    public_checks[key] = dict(passed=value is None and contract[key] is None,
                                              nonfinite_or_missing=True)
                else:
                    gate = gates.get('public_fit', {})
                    public_checks[key] = compare_array(np.atleast_1d(value), np.atleast_1d(contract[key]),
                        float(gate.get('atol', 0)), float(gate.get('rtol', 0)))
            configuration = contract.get('search_configuration') or {}
            public_checks['standard_engine'] = dict(passed=configuration.get('method') == 'reference' and
                configuration.get('phase_binning') is False and configuration.get('samples_used') == len(r['prepared_t']) and
                configuration.get('input_count') == len(r['prepared_t']) and configuration.get('time_origin') == 0.)
    final_checks = {}
    if reference['mode'] == 'full':
        for key in ('fractional_duration', 'width_in_samples', 'duration', 'depth', 'T0', 'transit_times', 'native_gtls_snr'):
            fits = [record['result'].get('final_fit', {}) for record in (reference, candidate)]
            if any(key not in fit for fit in fits):
                final_checks[key] = dict(passed=False, missing_reference=key not in fits[0], missing_candidate=key not in fits[1])
                continue
            rvalue, cvalue = (fit[key] for fit in fits)
            if rvalue is None or cvalue is None:
                final_checks[key] = dict(passed=rvalue is None and cvalue is None, nonfinite_or_missing=True)
            else:
                gate = gates.get('final_fit', {})
                final_checks[key] = compare_array(np.atleast_1d(rvalue), np.atleast_1d(cvalue),
                                                  float(gate.get('atol', 0)), float(gate.get('rtol', 0)))
    numeric_passed = (all(v['passed'] for v in checks.values()) and all(v['passed'] for v in final_checks.values()) and
                      all(v['passed'] for v in public_checks.values()) and stage_counts[0] == stage_counts[1])
    decisions_passed = ranking['same_primary'] and ranking['same_period'] and all(v['same_decision'] for v in thresholds)
    passed = numeric_passed and decisions_passed
    write(args.out, dict(status='compared', passed=passed, numeric_passed=numeric_passed,
                         decisions_passed=decisions_passed, checks=checks, final_checks=final_checks,
                         public_checks=public_checks,
                         stage_counts=stage_counts, ranking=ranking, thresholds=thresholds,
                         reference_record_sha256=sha(rpath), candidate_record_sha256=sha(cpath),
                         gates=gates, input_metadata=reference['input_metadata'],
                         warning='Numeric parity is distinct from a statistically established recovery margin. Near-tie outcome disagreements are retained.'))
    print(json.dumps(dict(passed=passed, same_primary=ranking['same_primary'], output=str(args.out))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--case', type=Path, required=True)
    run_parser.add_argument('--backend', choices=('gtls', 'gtls_corrected', 'candidate'), required=True)
    run_parser.add_argument('--mode', choices=('fast', 'full'), default='full')
    run_parser.add_argument('--engine-root', type=Path, required=True)
    run_parser.add_argument('--engine-kind', choices=('production', 'public'), default='public')
    run_parser.add_argument('--positive-origin', action='store_true', help='Apply the same positive-origin time shift before either backend.')
    run_parser.add_argument('--auto-grid', action='store_true', help='Exercise both public APIs automatic period-grid generation.')
    run_parser.add_argument('--reference-record', type=Path)
    run_parser.add_argument('--options', type=Path)
    run_parser.add_argument('--replay', action='store_true', help='Rerun published frozen inputs as reproduction, preserving original seed/source metadata.')
    run_parser.add_argument('--seal', type=Path, help='Required for frozen heldout inputs.')
    run_parser.add_argument('--work-chunk', type=int, default=256)
    run_parser.add_argument('--chunk-policy', choices=('replay', 'default'), default='default',
                            help='Replay actual native group size, or independently test candidate default grouping.')
    run_parser.add_argument('--out', type=Path, required=True)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--reference', type=Path, required=True)
    compare_parser.add_argument('--candidate', type=Path, required=True)
    compare_parser.add_argument('--gates', type=Path, help='Pre-frozen per-array atol/rtol JSON; omitted means exact numerical equality.')
    compare_parser.add_argument('--threshold', type=float, action='append', default=[])
    compare_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else compare)(args)


if __name__ == '__main__':
    main()
