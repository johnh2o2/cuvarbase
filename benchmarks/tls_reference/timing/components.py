#!/usr/bin/env python3
"""Separate component measurements; never substitute them for public timings.

The native common-search endpoint is stamped immediately after the pinned
final single-period argmin().get(), before physical/SNR postprocessing. The
candidate endpoint is immediately after engine.search_full returns. Added
wrappers and one in-memory timestamp statement do not alter numerical source
or returned values; every completed result must match a literal public call.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'

if __package__:
    from .common import (NATIVE_BACKENDS, REGIMES, SINGLE_REPETITIONS, case_identity, environment,
                         fingerprint, initialize_backend, load_cases, public_single,
                         masked_hash, sha, write)
    from .cohort import frozen_outputs, select, verify_worker_sources
    from .benchmark import (GPUOwnership, Monitor, exclusive_gpu_processes,
                            ownership_valid, process_ids, retain_cuda_context)
else:
    from common import (NATIVE_BACKENDS, REGIMES, SINGLE_REPETITIONS, case_identity, environment,
                        fingerprint, initialize_backend, load_cases, public_single,
                        masked_hash, sha, write)
    from cohort import frozen_outputs, select, verify_worker_sources
    from benchmark import (GPUOwnership, Monitor, exclusive_gpu_processes,
                           ownership_valid, process_ids, retain_cuda_context)


class Components:
    def __init__(self, backend):
        self.backend = backend
        self.patches = []
        self.reset()

    def reset(self):
        self.events = []
        self.search_end = None
        self.search_result = None
        self.boundary_count = 0

    def patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def timed(self, function, label, boundary=False):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            before = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                if boundary:
                    self.search_end = time.perf_counter()
                    self.boundary_count += 1
                    self.search_result = result
                return result
            finally:
                self.events.append(dict(stage=label, started=before, ended=time.perf_counter()))
        return wrapped

    def mark_native_search_end(self):
        self.search_end = time.perf_counter()
        self.boundary_count += 1
        # Keep CPU spectrum references only after recording the timestamp.
        # They can document a finished search if later native diagnostics fail.
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_code.co_name not in (
                'search_multi_periods', 'search_multi_periods_multiGPU'):
            frame = frame.f_back
        if frame is None:
            raise RuntimeError('Pinned native final search caller was not found')
        values = frame.f_locals
        self.search_result = {key: values[key] for key in ('periods', 'period', 'power', 'chi2', 'SDE')}

    def __enter__(self):
        try:
            return self.install()
        except BaseException:
            self.__exit__()
            raise

    def install(self):
        if self.backend in NATIVE_BACKENDS:
            from gputls import core, stats
            self.patch(core, 'spectra', self.timed(core.spectra, 'native_spectra'))
            self.patch(core, 'search_multi_periods_again',
                       self.timed(core.search_multi_periods_again, 'native_candidate_or_harmonic_refinement'))
            self.patch(core, 'snr_stats', self.timed(core.snr_stats, 'native_snr_stats_inclusive'))
            self.patch(stats, 'pink_noise', self.timed(stats.pink_noise, 'native_pink_noise_nested'))
            original = core.search_single_periods
            source = inspect.getsource(original)
            needle = '    bestLocation = lowestResidualsGPU.argmin().get()\n'
            if source.count(needle) != 1:
                raise ValueError('Pinned GTLS search-end statement was not found exactly once')
            self.endpoint_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            instrumented = source.replace(needle, needle + '    _benchmark_search_end()\n')
            namespace = dict(core.__dict__, _benchmark_search_end=self.mark_native_search_end)
            exec(compile(instrumented, str(core.__file__) + ':benchmark_timestamp', 'exec'), namespace)
            self.patch(core, 'search_single_periods',
                       self.timed(namespace['search_single_periods'], 'native_final_window_and_diagnostics'))
        else:
            from cuvarbase import tls_reference as engine
            self.endpoint_source_sha256 = sha(engine.__file__)
            self.patch(engine, 'search_full',
                       self.timed(engine.search_full, 'candidate_search_full', boundary=True))
            original = engine.raw_search
            @functools.wraps(original)
            def raw(*args, **kwargs):
                label = ('candidate_full_window_stage' if kwargs.get('full', False)
                         else 'candidate_coarse_stage')
                return self.timed(original, label)(*args, **kwargs)
            self.patch(engine, 'raw_search', raw)
            for name, label in (('build_cache', 'candidate_template_cache'),
                                ('native_spectra', 'candidate_spectra'),
                                ('final_parameters', 'candidate_final_physical_parameters')):
                self.patch(engine.reference, name, self.timed(getattr(engine.reference, name), label))
        return self

    def __exit__(self, *unused):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches = []

    def accounting(self, before, after):
        durations = defaultdict(float)
        calls = defaultdict(int)
        for event in self.events:
            durations[event['stage']] += event['ended'] - event['started']
            calls[event['stage']] += 1
        return dict(public_instrumented_seconds=after-before,
                    common_search_seconds=None if self.search_end is None else self.search_end-before,
                    after_common_search_seconds=None if self.search_end is None else after-self.search_end,
                    endpoint_count=self.boundary_count,
                    endpoint_valid=(self.boundary_count == 1 and self.search_end is not None and
                                    before <= self.search_end <= after),
                    inclusive_stage_seconds=dict(durations), stage_call_counts=dict(calls),
                    events=[dict(stage=event['stage'], start_from_public=event['started']-before,
                                 end_from_public=event['ended']-before) for event in self.events],
                    overlap_note='Stage times are inclusive and can overlap: pink_noise is nested in snr_stats, which is nested in final_window_and_diagnostics.')


def attempt(backend, case):
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    before = time.perf_counter()
    result, error = None, None
    try:
        result = public_single(backend, case)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as failure:
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
    after = time.perf_counter()
    return before, after, result, error


def run_owned(args, sources, ownership):
    """Numerical instrumentation stays in one child, after parent approval."""
    output = args.output.resolve()
    allowed = sorted(value['host_pid'] for value in ownership['bindings'])
    plan = dict(backend=args.backend, manifest_sha256=sha(args.manifest),
                measurement_scope=getattr(args, 'measurement_scope', 'full'),
                repetitions=SINGLE_REPETITIONS, full=True,
                source_files=sources, environment=environment(),
                harness_sources={path.name: sha(path) for path in Path(__file__).parent.glob('*.py')},
                common_endpoint=('Immediately after native final bestLocation = lowestResidualsGPU.argmin().get()'
                                 if args.backend in NATIVE_BACKENDS else 'Immediately after production engine.search_full returns'),
                timing_scope='From public-call entry including constructor/validation/cache through final window selection; candidate also transfers compact winner fields before its endpoint',
                reporting='Separate component experiment; public headline timings must come from benchmark.py',
                gpu_ownership=ownership,
                cohorts={})
    write(output/'plan.json', plan)
    records = {}
    with Monitor(output/'monitor.jsonl'):
        for regime in args.regimes:
            selection = select(args.manifest, args.paired_results, regime)
            verify_worker_sources(args.backend, sources, selection)
            plan['cohorts'][regime] = selection
            write(output/'plan.json', plan)
            if selection['single_case'] is None:
                records[regime] = dict(status='no_paired_single_case')
                continue
            if args.backend == 'gtls_corrected' and not selection['correction_timing']['required']:
                records[regime] = dict(status='correction_proved_no_op_on_entire_timing_cohort')
                continue
            case = load_cases(args.manifest, regime, [selection['single_case']])[0]
            expected = frozen_outputs(args.paired_results, [case], backends=(args.backend,),
                                      manifest_path=args.manifest)[args.backend][case['name']]
            literal_exclusive_before = exclusive_gpu_processes(allowed)
            before, after, literal, error = attempt(args.backend, case)
            literal_exclusive_after = exclusive_gpu_processes(allowed, strict=False)
            if not literal_exclusive_after['exclusive']:
                error = dict(type='ConcurrencyError', snapshot=literal_exclusive_after)
            baseline = None
            if literal is not None:
                try:
                    baseline = fingerprint(args.backend, case, literal)
                except Exception:
                    error = dict(type='FingerprintError', traceback=traceback.format_exc())
            record = dict(case=case_identity(case), literal_warmup_seconds=after-before,
                          literal_error=error, literal_outputs=baseline, repetitions=[],
                          gpu_ownership=ownership,
                          literal_exclusive_before=literal_exclusive_before,
                          literal_exclusive_after=literal_exclusive_after,
                          frozen_outputs=expected,
                          literal_matches_frozen=error is None and baseline is not None and baseline['strict'] == expected['strict'])
            literal = None
            with Components(args.backend) as instrument:
                record['endpoint_source_sha256'] = instrument.endpoint_source_sha256
                for _ in range(SINGLE_REPETITIONS):
                    instrument.reset()
                    exclusive_before = exclusive_gpu_processes(allowed)
                    before, after, result, error = attempt(args.backend, case)
                    exclusive_after = exclusive_gpu_processes(allowed, strict=False)
                    if not exclusive_after['exclusive']:
                        error = dict(type='ConcurrencyError', snapshot=exclusive_after)
                    measured = instrument.accounting(before, after)
                    observed = None
                    if result is not None:
                        try:
                            observed = fingerprint(args.backend, case, result)
                        except Exception:
                            error = dict(type='FingerprintError', traceback=traceback.format_exc())
                    exact = bool(baseline is not None and observed is not None and
                                 observed['full_digest'] == baseline['full_digest'])
                    measured.update(error=error, outputs=observed,
                                    exclusive_before=exclusive_before, exclusive_after=exclusive_after,
                                    output_identical_to_literal=exact,
                                    denominator_eligible=exact and error is None and measured['endpoint_valid'] and record['literal_matches_frozen'],
                                    completed_search_before_api_failure=instrument.search_end is not None and error is not None)
                    if instrument.search_result is not None and args.backend in NATIVE_BACKENDS:
                        # Retained arrays are hashed after the measurement,
                        # including when optional native postprocessing failed.
                        measured['search_outputs'] = {key: masked_hash(value)
                                                      for key, value in instrument.search_result.items()}
                    record['repetitions'].append(measured)
                    write(output/(regime+'.json'), record)
                    result = None
            record['status'] = ('ok' if all(rep['denominator_eligible'] for rep in record['repetitions'])
                                else 'failed_literal_output_gate')
            records[regime] = dict(status=record['status'], file=regime+'.json')
            write(output/(regime+'.json'), record)
    # The supervisor alone can complete the lifecycle after this process exits.
    write(output/'summary.json', dict(status='awaiting_worker_exit', backend=args.backend, regimes=records))


def component_worker(connection, args):
    try:
        sources = initialize_backend(args.backend, correction_adapter=args.correction_adapter)
        context_allocation = retain_cuda_context()
        connection.send(dict(kind='ready', pid=os.getpid(), namespace_pids=process_ids(), sources=sources,
                             cuda_context_allocation_bytes=1, cuda_context_synchronized=True))
        command = connection.recv()
        if command.get('kind') != 'bind':
            raise RuntimeError('Component worker was not given its verified host PID binding')
        run_owned(args, sources, command['ownership'])
        connection.send(dict(kind='complete'))
    except BaseException:
        connection.send(dict(kind='fatal', traceback=traceback.format_exc()))
        raise
    finally:
        connection.close()


def run(args):
    """Supervise context birth and exit outside every component clock."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ownership, process, parent, child = None, None, None, None
    forced, failure = [], None
    try:
        ownership = GPUOwnership()
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=component_worker, args=(child, args))
        process.start()
        child.close()
        if not parent.poll(180.):
            raise TimeoutError('Component worker startup timed out')
        ready = parent.recv()
        if ready.get('kind') != 'ready':
            raise RuntimeError('Component startup failed: ' + repr(ready))
        ownership.bind([ready], [process])
        write(output/'ownership.json', ownership.receipt)
        parent.send(dict(kind='bind', ownership=ownership.receipt))
        if not parent.poll(600.):
            raise TimeoutError('Component worker did not complete')
        completed = parent.recv()
        if completed.get('kind') != 'complete':
            raise RuntimeError('Component worker failed: ' + repr(completed))
    except BaseException:
        failure = traceback.format_exc()
    finally:
        receipt = (dict(status='failed', passed=False) if ownership is None else ownership.receipt)
        if parent is not None:
            parent.close()
        if child is not None:
            child.close()
        if process is not None and process.pid is not None:
            process.join(timeout=5.)
            if process.is_alive():
                forced.append(process.pid)
                process.terminate()  # Only the Process object started here.
                process.join(timeout=5.)
        if ownership is not None:
            try:
                receipt = ownership.finish([] if process is None else [process], forced=forced)
                write(output/'ownership.json', receipt)
                if receipt['passed'] is not True:
                    failure = (failure or '') + '\nComponent GPU ownership lifecycle did not pass'
            except Exception:
                failure = (failure or '') + '\n' + traceback.format_exc()
        summary_path = output/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
            backend=args.backend, regimes={})
        for regime in args.regimes:
            path = output/(regime+'.json')
            if path.exists():
                record = json.loads(path.read_text())
                record['gpu_ownership'] = receipt
                if not ownership_valid(record):
                    record['status'] = 'ownership_failure'
                    failure = (failure or '') + '\nInvalid component ownership receipt: ' + regime
                write(path, record)
                summary['regimes'].setdefault(regime, dict(file=regime+'.json'))['status'] = record['status']
        summary.update(status='error' if failure else 'complete', gpu_ownership=receipt)
        if failure:
            summary['error'] = failure
        write(summary_path, summary)
    if failure:
        raise RuntimeError('Component supervision failed: ' + failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--paired-results', type=Path, required=True)
    parser.add_argument('--backend', choices=('gtls', 'gtls_corrected', 'candidate'), required=True)
    parser.add_argument('--correction-adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurement-scope', choices=('full', 'single'), default='full',
                        help='Record the associated public timing scope; components always measure one source')
    parser.add_argument('--regimes', nargs='+', choices=REGIMES, default=list(REGIMES))
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    run(args)


if __name__ == '__main__':
    main()
