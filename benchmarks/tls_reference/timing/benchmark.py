#!/usr/bin/env python3
"""Uninstrumented complete-public-call TLS timings with deferred validation.

Persistent workers retain their raw results until the parent has stopped the
clock. Only then are result arrays hashed and released. No numerical routines
are patched in headline graph mode. The optional row-prefix variant is an
explicit attribution experiment, kept separate from competitor selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import traceback

if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'

if __package__:
    from .common import (BATCH_REPETITIONS, NATIVE_BACKENDS, REGIMES, SINGLE_REPETITIONS, case_identity,
                         environment, fingerprint, initialize_backend, load_cases,
                         public_batch, public_single, sha, write)
    from .cohort import frozen_outputs, select, verify_worker_sources
else:
    from common import (BATCH_REPETITIONS, NATIVE_BACKENDS, REGIMES, SINGLE_REPETITIONS, case_identity,
                        environment, fingerprint, initialize_backend, load_cases,
                        public_batch, public_single, sha, write)
    from cohort import frozen_outputs, select, verify_worker_sources


def process_ids():
    """Record any visible outer/inner namespace IDs for NVML process checks."""
    result = {os.getpid()}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('NSpid:'):
                result.update(int(value) for value in line.split()[1:])
    return sorted(result)


def retain_cuda_context():
    """Make this worker visible to NVML before any search or timing clock."""
    import cupy as cp
    allocation = cp.cuda.alloc(1)
    cp.cuda.runtime.deviceSynchronize()
    return allocation


def worker(connection, config):
    """One serial public-API consumer, with no timed result serialization."""
    try:
        cases = load_cases(config['manifest'], config['regime'], config.get('names'))
        sources = initialize_backend(config['backend'], config['prefix'], config.get('correction_adapter'))
        import cupy as cp
        context_allocation = retain_cuda_context()  # Kept alive until worker exit.
        connection.send(dict(kind='ready', pid=os.getpid(), namespace_pids=process_ids(), sources=sources,
                             cuda_context_allocation_bytes=1, cuda_context_synchronized=True))
        retained = []
        while True:
            command = connection.recv()
            if command['kind'] == 'close':
                break
            if command['kind'] == 'validate':
                records, errors = [], []
                for index, result in retained:
                    try:
                        records.append(fingerprint(config['backend'], cases[index], result))
                    except Exception:
                        errors.append(dict(case=cases[index]['name'], traceback=traceback.format_exc()))
                retained = []
                connection.send(dict(kind='validation', records=records, errors=errors))
                continue
            if command['kind'] != 'run':
                raise ValueError('Unknown worker command')
            if retained:
                raise RuntimeError('Previous results were not validated before reuse')
            indices = command['indices']
            selected = [cases[index] for index in indices]
            cp.cuda.runtime.deviceSynchronize()
            started = time.perf_counter()
            error = None
            case_failures = []
            try:
                if command['single']:
                    results = [public_single(config['backend'], selected[0])]
                elif config['backend'] in NATIVE_BACKENDS:
                    # Preserve the identity and elapsed time of a failed
                    # public call, then let independent cases finish. A batch
                    # with any failure is never a timing denominator.
                    for index in indices:
                        call_started = time.perf_counter()
                        try:
                            result = public_single(config['backend'], cases[index])
                            cp.cuda.runtime.deviceSynchronize()
                            retained.append((index, result))
                        except Exception:
                            case_failures.append(dict(case=cases[index]['name'],
                                failure_seconds=time.perf_counter()-call_started,
                                traceback=traceback.format_exc()))
                    results = None
                else:
                    results = public_batch(config['backend'], selected)
                cp.cuda.runtime.deviceSynchronize()
                # Keep the raw returned objects alive through the barrier.
                if results is not None:
                    retained = list(zip(indices, results))
            except Exception:
                error = traceback.format_exc()
                try:
                    cp.cuda.runtime.deviceSynchronize()
                except Exception:
                    pass
            ended = time.perf_counter()
            connection.send(dict(kind='complete', pid=os.getpid(), started=started,
                                 ended=ended, api_seconds=ended-started,
                                 completed_cases=len(retained), error=error,
                                 case_failures=case_failures))
    except Exception:
        try:
            connection.send(dict(kind='fatal', pid=os.getpid(), traceback=traceback.format_exc()))
        except Exception:
            pass
    finally:
        connection.close()


def exclusive_gpu_processes(allowed, *, strict=True):
    """Require exactly the bound host PIDs, including their presence."""
    import pynvml as nvml
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        observed = sorted({int(process.pid) for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle)})
    finally:
        nvml.nvmlShutdown()
    foreign = sorted(set(observed) - set(allowed))
    missing = sorted(set(allowed) - set(observed))
    if (foreign or missing) and strict:
        raise RuntimeError('Foreign GPU compute processes or missing owned contexts prevent exclusive timings: '
                           + repr(dict(foreign=foreign, missing=missing)))
    return dict(allowed_pids=sorted(allowed), observed_pids=observed,
                foreign_pids=foreign, missing_pids=missing, exclusive=not foreign and not missing,
                monotonic=time.perf_counter(), utc=time.time())


class GPUOwnership:
    """Prove a worker's NVML host identity across its controlled lifetime.

    A container may expose only its inner PID in NSpid. On a previously empty
    GPU, N distinct initialized children may then own exactly N new host PIDs
    as a set. Hidden IDs are never assigned arbitrarily to individual children.
    The set is provisional until all contexts disappear after clean pool exit.
    """
    def __init__(self):
        self.allowed_pids = []
        self.receipt = dict(version=1, status='starting', passed=False,
                            before_start=exclusive_gpu_processes([], strict=False))
        if not self.receipt['before_start']['exclusive']:
            self.receipt['status'] = 'failed'
            error = RuntimeError('GPU must be empty before any owned worker starts: '
                                 + repr(self.receipt['before_start']))
            error.gpu_ownership = self.receipt
            raise error

    def bind(self, ready, processes, timeout=5.):
        self.receipt.update(workers=ready, owned_worker_pids=[process.pid for process in processes])
        if (len(ready) != len(processes) or not ready or
                len({process.pid for process in processes}) != len(processes)):
            raise RuntimeError('Missing owned worker readiness records')
        for value, process in zip(ready, processes):
            if (not process.is_alive() or value.get('pid') != process.pid or
                    process.pid not in value.get('namespace_pids', []) or
                    value.get('cuda_context_allocation_bytes') != 1 or
                    value.get('cuda_context_synchronized') is not True):
                raise RuntimeError('Owned worker did not prove a live synchronized CUDA context')
        deadline = time.monotonic() + timeout
        observations = []
        while True:
            current = exclusive_gpu_processes([], strict=False)
            observations.append(current)
            observed = current['observed_pids']
            if len(observed) == len(ready):
                break
            if (len(observed) > len(ready) or time.monotonic() >= deadline or
                    any(not process.is_alive() for process in processes)):
                self.receipt['startup_observations'] = observations
                raise RuntimeError('Ambiguous or missing newly visible CUDA contexts: ' + repr(observed))
            time.sleep(.1)
        bindings = []
        for value in ready:
            matches = sorted(set(value['namespace_pids']) & set(observed))
            if len(matches) == 1:
                host_pid, method = matches[0], 'visible_namespace_pid'
            elif len(ready) == 1 and not matches:
                host_pid, method = observed[0], 'single_worker_lifecycle'
            elif not matches:
                host_pid, method = None, 'pool_lifecycle_member'
            else:
                raise RuntimeError('Ambiguous directly visible namespace IDs')
            bindings.append(dict(worker_pid=value['pid'], namespace_pids=value['namespace_pids'],
                                 host_pid=host_pid, method=method))
        assigned = [value['host_pid'] for value in bindings if value['host_pid'] is not None]
        if len(set(assigned)) != len(assigned):
            raise RuntimeError('Multiple workers cannot share one claimed NVML identity')
        self.allowed_pids = observed
        self.receipt.update(status='bound', bindings=bindings, host_pids=observed,
            identity_scope='individual' if len(assigned) == len(ready) else 'owned_pool_set',
            live_workers_at_binding=[process.pid for process in processes if process.is_alive()],
            startup_observations=observations,
            after_start=exclusive_gpu_processes(self.allowed_pids))
        return self.allowed_pids

    def finish(self, processes, forced=(), timeout=5.):
        """Call only after joining owned children; never terminate NVML PIDs."""
        exits = [dict(worker_pid=process.pid, exit_code=process.exitcode,
                      alive=process.is_alive(), forced=process.pid in forced) for process in processes]
        deadline = time.monotonic() + timeout
        observations = []
        while True:
            current = exclusive_gpu_processes([], strict=False)
            observations.append(current)
            if current['exclusive'] or time.monotonic() >= deadline:
                break
            # Unexpected foreign contexts cannot be attributed to delayed cleanup.
            if set(current['observed_pids']) - set(self.allowed_pids):
                break
            time.sleep(.1)
        clean = bool(exits) and all(value['exit_code'] == 0 and not value['alive'] and
                                    not value['forced'] for value in exits)
        passed = self.receipt['status'] == 'bound' and clean and current['exclusive']
        self.receipt.update(status='complete' if passed else 'failed', passed=passed,
            worker_exits=exits, exit_observations=observations, after_exit=current)
        return self.receipt


def ownership_valid(record):
    """Check the lifecycle receipt, not just an asserted success flag."""
    receipt = record.get('gpu_ownership', {})
    bindings = receipt.get('bindings', [])
    hosts = receipt.get('host_pids', [])
    workers = sorted(value.get('worker_pid', -1) for value in bindings)
    exits = receipt.get('worker_exits', [])
    ready = receipt.get('workers', [])
    if (receipt.get('version') != 1 or receipt.get('status') != 'complete' or
            receipt.get('passed') is not True or not hosts or len(hosts) != len(bindings) or
            hosts != sorted(set(hosts)) or
            len(set(workers)) != len(workers) or min(hosts + workers) <= 0 or
            sorted(receipt.get('owned_worker_pids', [])) != workers or
            sorted(receipt.get('live_workers_at_binding', [])) != workers or
            sorted(value.get('pid', -1) for value in ready) != workers or
            any(value.get('cuda_context_allocation_bytes') != 1 or
                value.get('cuda_context_synchronized') is not True for value in ready) or
            sorted(value.get('worker_pid', -1) for value in exits) != workers or
            any(value.get('exit_code') != 0 or value.get('alive') is not False or
                value.get('forced') is not False for value in exits)):
        return False
    if 'pool_width' in record and record['pool_width'] != len(workers):
        return False
    if any(value['namespace_pids'] != next(worker['namespace_pids'] for worker in ready
            if worker['pid'] == value['worker_pid']) for value in bindings):
        return False
    for name, expected in (('before_start', []), ('after_start', hosts), ('after_exit', [])):
        value = receipt.get(name, {})
        if (value.get('exclusive') is not True or value.get('allowed_pids') != expected or
                value.get('observed_pids') != expected or value.get('foreign_pids') != [] or
                value.get('missing_pids') != []):
            return False
    if not all(value.get('method') in ('visible_namespace_pid', 'single_worker_lifecycle', 'pool_lifecycle_member') and
               value['worker_pid'] in value.get('namespace_pids', []) and
               (value['host_pid'] in value['namespace_pids'] and value['host_pid'] in hosts
                if value['method'] == 'visible_namespace_pid' else
                len(bindings) == 1 and value['host_pid'] == hosts[0]
                if value['method'] == 'single_worker_lifecycle' else
                len(bindings) > 1 and value['host_pid'] is None)
               for value in bindings):
        return False
    assigned = [value['host_pid'] for value in bindings if value['host_pid'] is not None]
    if (len(set(assigned)) != len(assigned) or receipt.get('identity_scope') !=
            ('individual' if len(assigned) == len(bindings) else 'owned_pool_set')):
        return False
    calls = record.get('single', []) + record.get('batch', []) + record.get('repetitions', [])
    if 'warmup' in record:
        calls = [record['warmup']] + calls
    if 'literal_exclusive_before' in record:
        calls = [dict(exclusive_before=record['literal_exclusive_before'],
                      exclusive_after=record.get('literal_exclusive_after'))] + calls
    for call in calls:
        for name in ('exclusive_before', 'exclusive_after'):
            value = call.get(name) or {}
            if (value.get('exclusive') is not True or value.get('allowed_pids') != hosts or
                    value.get('observed_pids') != hosts or value.get('foreign_pids') != [] or
                    value.get('missing_pids') != []):
                return False
    return True


class WorkerPool:
    def __init__(self, config, width, timeout=1800.):
        self.timeout = timeout
        self.names = config['names']
        self.connections, self.processes = [], []
        self.ownership = GPUOwnership()  # Before spawning or initializing CUDA.
        self.closed = False
        begin = time.perf_counter()
        context = mp.get_context('spawn')
        try:
            for _ in range(width):
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child, config))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.ready = [self._receive(connection, 'ready') for connection in self.connections]
            self.ownership.bind(self.ready, self.processes)
        except Exception as error:
            self.close()
            error.gpu_ownership = self.ownership.receipt
            raise
        self.startup_seconds = time.perf_counter() - begin

    def _receive(self, connection, expected):
        if not connection.poll(self.timeout):
            raise TimeoutError('Worker did not finish within the declared timeout')
        result = connection.recv()
        if result.get('kind') != expected:
            raise RuntimeError('Unexpected worker response: ' + repr(result))
        return result

    def measure(self, indices, *, single=False, warmup=False):
        if single and len(self.connections) != 1:
            raise ValueError('Single-source latency uses one persistent worker')
        # Round-robin assignment is fixed; completion order cannot alter inputs.
        assignments = ([indices[:1]] * len(self.connections) if warmup else
                       [indices[i::len(self.connections)] for i in range(len(self.connections))])
        if any(not values for values in assignments):
            raise ValueError('Every worker must receive at least one case')
        if any(not process.is_alive() for process in self.processes):
            raise RuntimeError('An owned GPU worker exited before measurement')
        allowed_pids = self.ownership.allowed_pids
        exclusive_before = exclusive_gpu_processes(allowed_pids)
        before = time.perf_counter()
        for connection, values in zip(self.connections, assignments):
            connection.send(dict(kind='run', indices=values, single=single))
        completions = [self._receive(connection, 'complete') for connection in self.connections]
        after = time.perf_counter()  # All public returns + CUDA work completed.
        exclusive_after = exclusive_gpu_processes(allowed_pids, strict=False)
        # No hashing, NumPy comparisons or array IPC occurs before this point.
        validation_begin = time.perf_counter()
        for connection in self.connections:
            connection.send(dict(kind='validate'))
        validation = [self._receive(connection, 'validation') for connection in self.connections]
        records = [record for value in validation for record in value['records']]
        errors = ([value['error'] for value in completions if value['error']] +
                  [error for value in completions for error in value['case_failures']] +
                  [error for value in validation for error in value['errors']])
        if not exclusive_after['exclusive']:
            errors.append(dict(reason='GPU ownership changed across the measurement',
                               snapshot=exclusive_after))
        if any(not process.is_alive() for process in self.processes):
            errors.append(dict(reason='An owned worker exited across the measurement'))
        expected = sum(len(values) for values in assignments)
        expected_names = [self.names[index] for values in assignments for index in values]
        observed_names = [value['case'] for value in records]
        if Counter(expected_names) != Counter(observed_names):
            errors.append(dict(reason='Post-barrier result membership differs from the assigned inputs',
                               expected=expected_names, observed=observed_names))
        complete = not errors and sum(value['completed_cases'] for value in completions) == expected
        return dict(status='ok' if complete else 'error', elapsed_seconds=after-before,
                    denominator_seconds=after-before if complete else None,
                    validation_seconds=time.perf_counter()-validation_begin,
                    worker_public_calls=completions, errors=errors, outputs=records,
                    assignments=assignments, source_count=expected, warmup=warmup,
                    exclusive_before=exclusive_before, exclusive_after=exclusive_after)

    def close(self):
        if self.closed:
            return self.ownership.receipt
        self.closed = True
        forced = []
        for connection in self.connections:
            try:
                connection.send(dict(kind='close'))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.)
            if process.is_alive():
                forced.append(process.pid)
                process.terminate()
                process.join(timeout=5.)
        for connection in self.connections:
            connection.close()
        return self.ownership.finish(self.processes, forced=forced)


class Monitor:
    """One-Hz read-only NVML/cgroup telemetry, outside worker API execution."""
    def __init__(self, path):
        self.path = Path(path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import json
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:
            handle, nvml = None, None
            setup_error = repr(error)
        with self.path.open('w') as output:
            while not self.stop_event.is_set():
                row = dict(monotonic=time.perf_counter(), utc=time.time())
                try:
                    if handle is not None:
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                        row.update(gpu_percent=utilization.gpu, memory_percent=utilization.memory,
                                   used_gpu_bytes=memory.used,
                                   power_mw=nvml.nvmlDeviceGetPowerUsage(handle),
                                   sm_clock_mhz=nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                                   temperature_c=nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
                        row['compute_processes'] = [dict(pid=int(process.pid),
                            used_gpu_bytes=int(process.usedGpuMemory))
                            for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                    else:
                        row['nvml_error'] = setup_error
                    for file in ('cpu.stat', 'memory.current'):
                        path = Path('/sys/fs/cgroup') / file
                        if path.exists():
                            row[file] = path.read_text().strip()
                except Exception as error:
                    row['error'] = repr(error)
                output.write(json.dumps(row) + '\n')
                output.flush()
                self.stop_event.wait(1.)
        if nvml is not None:
            nvml.nvmlShutdown()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop_event.set()
        self.thread.join(timeout=5.)


def consistency(records, reference=None, expected_names=None, expected_repetitions=None,
                field='strict'):
    """All cases and repetitions must pass before a denominator is eligible."""
    baseline = {} if reference is None else dict(reference)
    problems = []
    if not records or (expected_repetitions is not None and len(records) != expected_repetitions):
        problems.append(dict(reason='Missing declared repetitions', actual=len(records),
                             expected=expected_repetitions))
    expected = (Counter(expected_names) if expected_names is not None else
                Counter(reference.keys()) if reference is not None else None)
    for repetition, record in enumerate(records):
        if record['status'] != 'ok':
            problems.append(dict(repetition=repetition, reason='incomplete public result'))
            continue
        observed = Counter(value['case'] for value in record['outputs'])
        if expected is None:
            expected = observed
        if expected != observed or any(count != 1 for count in observed.values()):
            problems.append(dict(repetition=repetition, reason='Result membership changed',
                                 expected=dict(expected), observed=dict(observed)))
        if record.get('source_count', sum(observed.values())) != sum(observed.values()):
            problems.append(dict(repetition=repetition, reason='Source count differs from returned outputs'))
        for output in record['outputs']:
            name, current = output['case'], output[field]
            if name not in baseline:
                if reference is not None:
                    problems.append(dict(case=name, reason='absent from one-worker reference'))
                else:
                    baseline[name] = current
            elif baseline[name] != current:
                problems.append(dict(case=name, repetition=repetition,
                                     reason=field + ' output hashes changed'))
    return dict(eligible=not problems, problems=problems, reference=baseline)


def run(args):
    measurement_scope = getattr(args, 'measurement_scope', 'full')
    if measurement_scope not in ('full', 'single'):
        raise ValueError('Unknown measurement scope')
    if measurement_scope == 'single' and args.pool_widths != [1]:
        raise ValueError('Single-source latency uses exactly one worker per method')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Apply one CPU math thread per worker before fresh spawned imports. This
    # leaves the declared 1/2/4 worker count as the CPU parallelism control.
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    plan = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
                regimes=args.regimes, measurement_scope=measurement_scope,
                single_repetitions=SINGLE_REPETITIONS,
                batch_repetitions=BATCH_REPETITIONS if measurement_scope == 'full' else 0,
                native_pool_widths=args.pool_widths,
                cpu_math_threads_per_worker=1, full=True, return_arrays=True,
                instrumentation='None in public-call runner; only start/end clocks and completion synchronization',
                scope='Public constructor/validation/template cache/full search/final fit; preloaded files and explicit period-grid generation excluded',
                barrier='Every worker API return plus device synchronization before parent stops; result hashes afterward',
                native_extras='Literal GTLS full calls include CPU per-transit SNR and pink-noise diagnostics absent from cuvarbase',
                process_ownership='Empty device before workers; synchronized retained CUDA allocation; '
                    'bound NVML host PID(s) present and exclusive at every call boundary; clean exit and empty device afterward',
                sources={path.name: sha(path) for path in Path(__file__).parent.glob('*.py')},
                environment=environment(), cases={})
    selections = {regime: select(args.manifest, args.paired_results, regime) for regime in args.regimes}
    plan['cohort_selection'] = selections
    plan['failure_policy'] = ('Preserve every failed original, use deterministic manifest-order paired-success '
        'replacements before timing, report actual batch size if fewer than16. A new runtime failure '
        'invalidates its run/configuration; rerun a complete paired cohort for any later replacement, '
        'never mix different input cohorts or use failed elapsed times as speed denominators.')
    all_cases = {regime: load_cases(args.manifest, regime,
                    [selections[regime]['single_case']] if measurement_scope == 'single'
                    else selections[regime]['selected_cases'])
                 for regime in args.regimes if selections[regime]['single_case'] is not None}
    plan['cases'] = {regime: [case_identity(case) for case in cases] for regime, cases in all_cases.items()}
    plan['frozen_outputs'] = {regime: frozen_outputs(args.paired_results, cases,
        backends=('gtls', 'candidate', 'gtls_corrected') if selections[regime]['correction_timing']['required']
                 else ('gtls', 'candidate'), manifest_path=args.manifest) for regime, cases in all_cases.items()}
    correction_needed = any(selection['correction_timing']['required'] for selection in selections.values())
    if correction_needed and args.correction_adapter is None:
        raise ValueError('An affected selected case requires --correction-adapter for the additional native cross-check')
    plan['correction_adapter'] = (None if args.correction_adapter is None else
                                 dict(path=str(args.correction_adapter.resolve()), sha256=sha(args.correction_adapter)))
    write(output/'plan.json', plan)
    summary = dict(status='running', regimes={})
    write(output/'summary.json', summary)
    with Monitor(output/'monitor.jsonl'):
        for regime, cases in all_cases.items():
            regimes = summary['regimes'][regime] = {}
            configs = [('candidate', 1, 'graph')] + [('gtls', width, 'graph')
                        for width in args.pool_widths if width <= len(cases)]
            if selections[regime]['correction_timing']['required']:
                configs.append(('gtls_corrected', 1, 'graph'))
            if args.row_ab:
                configs.append(('candidate', 1, 'row'))
            native_reference = None
            for backend, width, prefix in configs:
                label = f'{backend}_{prefix}_{width}worker'
                target = output/regime/label
                target.mkdir(parents=True)
                config = dict(manifest=str(args.manifest.resolve()), regime=regime,
                              backend=backend, prefix=prefix,
                              correction_adapter=None if args.correction_adapter is None else str(args.correction_adapter.resolve()),
                              names=[case['name'] for case in cases], measurement_scope=measurement_scope)
                record = dict(config=config, pool_width=width, single=[], batch=[])
                pool = None
                try:
                    pool = WorkerPool(config, width, timeout=args.timeout)
                    for ready in pool.ready:
                        verify_worker_sources(backend, ready['sources'], selections[regime])
                    record.update(pool_startup_seconds=pool.startup_seconds, workers=pool.ready)
                    record['warmup'] = pool.measure([0], warmup=True,
                                                   single=measurement_scope == 'single')
                    if record['warmup']['status'] != 'ok':
                        raise RuntimeError('Warmup did not complete correctly')
                    write(target/'record.json', record)
                    if width == 1 and selections[regime]['single_case'] is not None:
                        single_index = config['names'].index(selections[regime]['single_case'])
                        for _ in range(SINGLE_REPETITIONS):
                            record['single'].append(pool.measure([single_index], single=True))
                            write(target/'record.json', record)
                    # Optional row-prefix A/B is deliberately single-source;
                    # it is not another entrant in strongest-pool selection.
                    if prefix == 'graph' and measurement_scope == 'full':
                        for _ in range(BATCH_REPETITIONS):
                            record['batch'].append(pool.measure(list(range(len(cases)))))
                            write(target/'record.json', record)
                    gate = consistency(record['batch'] or record['single'],
                                       native_reference if backend == 'gtls' else None,
                                       expected_names=(config['names'] if record['batch'] else
                                                       [selections[regime]['single_case']]),
                                       expected_repetitions=(BATCH_REPETITIONS if record['batch'] else
                                                             SINGLE_REPETITIONS))
                    if backend == 'gtls' and width == 1:
                        if gate['eligible'] and len(gate['reference']) == len(cases):
                            native_reference = gate['reference']
                        else:
                            gate['eligible'] = False
                            gate['problems'].append(dict(reason='No stable complete one-worker reference'))
                    elif backend == 'gtls' and native_reference is None:
                        gate['eligible'] = False
                        gate['problems'].append(dict(reason='One-worker baseline missing or unstable'))
                    record['parity'] = gate
                    expected = {name: value['strict'] for name, value in
                                plan['frozen_outputs'][regime][backend].items()}
                    if prefix == 'row':
                        expected = {selections[regime]['single_case']: expected[selections[regime]['single_case']]}
                    record['frozen_output_gate'] = consistency(record['batch'] or record['single'],
                        reference=expected, expected_names=list(expected),
                        expected_repetitions=BATCH_REPETITIONS if record['batch'] else SINGLE_REPETITIONS)
                    if record['single']:
                        single_name = selections[regime]['single_case']
                        reference = ({single_name: gate['reference'][single_name]}
                                     if single_name in gate['reference'] else None)
                        record['single_parity'] = consistency(record['single'], reference=reference,
                            expected_names=[single_name], expected_repetitions=SINGLE_REPETITIONS)
                        record['full_repeatability_gate'] = consistency(record['single'],
                            expected_names=[single_name], expected_repetitions=SINGLE_REPETITIONS,
                            field='full_digest')
                    else:
                        record['single_parity'] = None
                    record['status'] = ('ok' if all(rep['status'] == 'ok'
                        for rep in record['single'] + record['batch']) else 'measurement_failure')
                except Exception as error:
                    record.update(status='error', error=traceback.format_exc(),
                                  parity=dict(eligible=False, problems=[dict(reason='runner failure')]))
                    if hasattr(error, 'gpu_ownership'):
                        record['gpu_ownership'] = error.gpu_ownership
                finally:
                    if pool is not None:
                        try:
                            record['gpu_ownership'] = pool.close()
                            if not ownership_valid(record):
                                record.update(status='ownership_failure',
                                    parity=dict(eligible=False, problems=[dict(reason='GPU ownership lifecycle failed')]))
                        except Exception:
                            record.update(status='ownership_failure', ownership_error=traceback.format_exc(),
                                parity=dict(eligible=False, problems=[dict(reason='GPU ownership cleanup failed')]))
                write(target/'record.json', record)
                regimes[label] = dict(status=record['status'], record=str(target.relative_to(output)/'record.json'),
                    parity=record['parity'],
                    single_seconds=[item['denominator_seconds'] for item in record['single']],
                    batch_seconds=[item['denominator_seconds'] for item in record['batch']])
                write(output/'summary.json', summary)
                if (measurement_scope == 'single' or width == 1) and (record['status'] != 'ok' or
                        not record['parity']['eligible'] or
                        not record.get('frozen_output_gate', {}).get('eligible') or
                        not record.get('full_repeatability_gate', {}).get('eligible')):
                    summary['status'] = 'error'
                    write(output/'summary.json', summary)
                    raise RuntimeError('Required public timing failed its complete-output gate: ' + regime + '/' + label)
    summary['cohort_selection'] = selections
    summary['measurement_scope'] = measurement_scope
    summary['status'] = ('complete' if all(
        selection['single_case'] is not None if measurement_scope == 'single'
        else selection['actual_batch_size'] == 16
        for selection in selections.values()) else 'insufficient_paired_cases')
    write(output/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--paired-results', type=Path, required=True,
                        help='Completed heldout per-case native/candidate records; selection uses API success only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regimes', nargs='+', choices=REGIMES, default=list(REGIMES))
    parser.add_argument('--measurement-scope', choices=('full', 'single'), default='full',
                        help='Single measures five single-source calls only; full also measures three batches and native pools')
    parser.add_argument('--pool-widths', nargs='+', type=int, choices=(1, 2, 4))
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--row-ab', action='store_true', help='Separate five single-source original-row prefix calls')
    parser.add_argument('--correction-adapter', type=Path,
                        help='Frozen corrected_reference.py from the independent study; used only for affected regimes')
    args = parser.parse_args()
    if args.pool_widths is None:
        args.pool_widths = [1] if args.measurement_scope == 'single' else [1, 2, 4]
    if args.measurement_scope == 'single' and args.pool_widths != [1]:
        parser.error('Single-source latency uses exactly one worker per method')
    if args.pool_widths != sorted(set(args.pool_widths)) or args.pool_widths[0] != 1:
        parser.error('Pool widths must be unique, ascending and include the one-worker baseline first')
    run(args)


if __name__ == '__main__':
    main()
