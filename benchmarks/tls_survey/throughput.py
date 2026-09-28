#!/usr/bin/env python3
"""Bounded, sustained single-GPU queues with independently tuned process pools.

Run one configuration at a time; compare configurations only after the numerical
and ownership gates pass. No cloud actions occur here. All imports of CUDA or a
scientific backend happen in spawned workers, so a baseline checkout can be
measured alongside the edited tree without monkey-patching either implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import cProfile
import hashlib
import io
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pstats
import resource
import shutil
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.tls_reference.timing.common import (array_hash, environment,
    fingerprint, initialize_backend, sha, write)
from benchmarks.tls_reference.timing.benchmark import (GPUOwnership,
    exclusive_gpu_processes, process_ids, retain_cuda_context)


def science_bls_module():
    """Load the sealed science runner without ambiguous top-level common imports."""
    name = '_tls_survey_throughput_science'
    if name in sys.modules:
        return sys.modules[name]
    from benchmarks.tls_survey import common as science_common
    previous = sys.modules.get('common')
    sys.modules['common'] = science_common
    try:
        return science_common.module(Path(__file__).with_name('run.py'), name)
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous


def configure_bls(cases, seal_path):
    """Use only the per-regime method/ranker frozen by scientific development."""
    from benchmarks.tls_survey.common import BLS_CONFIGS, source_identity, method_applicable
    seal = json.loads(Path(seal_path).read_text())
    if seal['source_identity'] != source_identity():
        raise ValueError('BLS science sources differ from the frozen seal')
    for case in cases:
        selection = seal['bls_selected'][case['metadata']['regime']]
        if selection['method'] not in BLS_CONFIGS or not method_applicable(
                selection['method'], case['metadata']['regime']):
            raise ValueError('Selected BLS configuration is not applicable')
        if selection['ranker'] not in ('raw', 'likelihood', 'detrended'):
            raise ValueError('Unknown frozen BLS ranker')
        case['bls_selection'] = selection
        case['group'] = hashlib.sha256((case['group'] +
            json.dumps(selection, sort_keys=True)).encode()).hexdigest()
    return seal


def load_manifest(path, names=(), regimes=()):
    """Accept reference and survey manifests; every array file is byte checked."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    selected = set(names)
    cases = []
    for entry in manifest['cases']:
        if selected and entry['file'] not in selected:
            continue
        filename = path.parent / entry['file']
        if sha(filename) != entry['sha256']:
            raise ValueError('Input hash mismatch: ' + str(filename))
        with np.load(filename, allow_pickle=False) as source:
            metadata = json.loads(str(source['metadata']))
            if regimes and metadata['regime'] not in regimes:
                continue
            data = {key: np.array(source[key], copy=True)
                    for key in ('t', 'y', 'dy', 'periods')}
        if np.any(data['t'] <= 0):
            raise ValueError('Competitors require identical positive-origin timestamps')
        if 'metadata' in entry and metadata != entry['metadata']:
            raise ValueError('Input metadata differs from its manifest entry')
        options = dict(metadata['search_kwargs'])
        case = dict(name=entry['file'], data=data, options=options,
                    metadata=metadata, input_sha256=entry['sha256'],
                    error_scale=float(np.mean(data['dy'])))
        case['group'] = hashlib.sha256((array_hash(data['periods']) +
            json.dumps(options, sort_keys=True)).encode()).hexdigest()
        cases.append(case)
    if not cases or selected - {case['name'] for case in cases}:
        raise ValueError('Empty cohort or missing requested input files')
    return cases


def batches(cases, batch_size):
    """One immutable grid/options group per API batch, in manifest order."""
    grouped = defaultdict(list)
    for index, case in enumerate(cases):
        grouped[case['group']].append(index)
    return [indices[start:start + batch_size] for indices in grouped.values()
            for start in range(0, len(indices), batch_size)]


def public_call(backend, cases, *, arrays):
    if backend == 'bls':
        science = science_bls_module()
        return [compact_bls(case, science, arrays=arrays) for case in cases]
    if backend in ('gtls', 'gtls_corrected'):
        from gputls import gtls
        return [gtls(case['data']['t'], case['data']['y'], case['data']['dy'],
                     verbose=False).power(periods=case['data']['periods'],
                     fast=False, verbose=False, show_progress_bar=False,
                     **case['options']) for case in cases]
    from cuvarbase.tls import tls_search_batch
    return tls_search_batch([(case['data']['t'], case['data']['y'], case['data']['dy'])
                             for case in cases], periods=cases[0]['data']['periods'],
                             full=True, return_arrays=arrays, **cases[0]['options'])


def compact_bls(case, science, *, arrays=False):
    """Same sealed GPU search; additional full diagnostics only on qualification."""
    from cuvarbase.bls import eebls_gpu_fast
    from benchmarks.tls_survey.common import BLS_CONFIGS
    selection = case['bls_selection']
    t, y, dy, periods = (case['data'][key] for key in ('t', 'y', 'dy', 'periods'))
    qmin, qmax = science.bls_bounds(periods)
    config = dict(BLS_CONFIGS[selection['method']])
    qmin *= config.pop('qmin_factor')
    power = np.asarray(eebls_gpu_fast(t, y, dy, 1/periods, qmin=qmin, qmax=qmax,
                                     ignore_negative_delta_sols=True, **config))
    if arrays:
        weight = dy**-2
        weighted_mean = np.dot(weight, y)/weight.sum()
        chi2_null = float(np.dot(weight, (y-weighted_mean)**2))
        candidates = science.bls_candidates(periods, power, chi2_null)
        chosen = candidates[selection['ranker']]
        return dict(period=chosen['period'], score=chosen['score'], candidates=candidates,
                    spectra=dict(periods=array_hash(periods), power=array_hash(power),
                                 valid_mask=array_hash(np.isfinite(power))),
                    _arrays=dict(periods=periods, power=power))
    if selection['ranker'] == 'detrended':
        from benchmarks.transit.worker import spectral_candidate
        chosen = spectral_candidate(periods, power)
    else:
        good = np.isfinite(power)
        if not good.any():
            raise ValueError('No finite BLS powers')
        index = int(np.argmax(np.where(good, power, -np.inf)))
        value = float(power[index])
        if selection['ranker'] == 'likelihood':
            weight = dy**-2
            weighted_mean = np.dot(weight, y)/weight.sum()
            chi2_null = float(np.dot(weight, (y-weighted_mean)**2))
            value = float(power[index]*chi2_null)
        chosen = dict(period=float(periods[index]), score=value)
    return dict(period=chosen['period'], score=chosen['score'])


def scalar_fingerprint(backend, result):
    """Shared selected-detection fields, excluding package-specific SNR units."""
    values = vars(result) if backend in ('gtls', 'gtls_corrected') else result
    if 'error' in values:
        raise RuntimeError('API returned error: ' + str(values['error']))
    fields = {name: array_hash(np.asarray(float(values[name]), dtype=np.float64))
              for name in (('period', 'score') if backend == 'bls' else ('period', 'SDE'))}
    if any(not np.isfinite(float(values[name])) for name in fields):
        raise ValueError('Nonfinite selected detection')
    return fields


def complete_fingerprint(backend, case, result):
    if backend != 'bls':
        return fingerprint(backend, case, result)
    strict = {key: result['spectra'][key] for key in ('periods', 'valid_mask')}
    strict.update(scalar_fingerprint(backend, result))
    # Native BLS float32 atomic accumulation varies even within one worker.
    # Its selected endpoint remains exact; retain nonwinning powers and unused
    # rankers as diagnostics rather than falsely calling this full equivalence.
    fields = {key: value for key, value in result.items()
              if key not in ('_arrays', 'spectrum_artifact')}
    return dict(case=case['name'], strict=strict, common=strict, fields=fields,
                spectrum_artifact=result.get('spectrum_artifact'),
                qualification_contract='BLS exact periods/mask and selected period/score; full powers diagnostic',
                full_digest=hashlib.sha256(json.dumps(fields, sort_keys=True,
                                                     allow_nan=False).encode()).hexdigest(),
                primary_period=result['period'], selected_score=result['score'],
                nperiods=len(case['data']['periods']))


def archive_bls_spectra(result, directory, name):
    """Persist full qualification arrays outside the API and sustained clocks."""
    path = Path(directory)/name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result['_arrays'])
    result['spectrum_artifact'] = dict(path=str(path.resolve()), sha256=sha(path))


def bls_repeat_diagnostics(rows, reference_rows=None):
    """Report native repeat variation; this function never grants eligibility."""
    expected = {}
    for row in rows if reference_rows is None else reference_rows:
        for value in row['outputs']:
            expected.setdefault(value['case'], value)
    records = []
    for row in rows:
        for actual in row['outputs']:
            name = actual['case']
            reference = expected[name]
            arrays = []
            for output in (reference, actual):
                artifact = output['spectrum_artifact']
                path = Path(artifact['path'])
                if sha(path) != artifact['sha256']:
                    raise ValueError('Archived BLS qualification spectrum changed')
                with np.load(path, allow_pickle=False) as source:
                    arrays.append(np.array(source['power'], copy=True))
            first, current = arrays
            common_finite = np.isfinite(first) & np.isfinite(current)
            delta = np.abs(current[common_finite].astype(np.float64) -
                           first[common_finite].astype(np.float64))
            relative = delta/np.maximum(np.abs(first[common_finite].astype(np.float64)), 1e-30)
            candidates = {}
            for ranker, value in actual['fields']['candidates'].items():
                previous = reference['fields']['candidates'][ranker]
                score_delta = (None if value.get('score') is None or previous.get('score') is None else
                               float(value['score'] - previous['score']))
                candidates[ranker] = dict(exact=value == previous,
                    period_changed=value.get('period') != previous.get('period'),
                    score_difference=score_delta,
                    reference=previous, actual=value)
            records.append(dict(case=name, worker=row['worker'],
                reference_artifact=reference['spectrum_artifact'],
                actual_artifact=actual['spectrum_artifact'],
                full_power_hash_equal=reference['fields']['spectra']['power'] == actual['fields']['spectra']['power'],
                finite_masks_equal=bool(np.array_equal(np.isfinite(first), np.isfinite(current))),
                changed_finite_power_values=int(np.count_nonzero(delta)),
                max_absolute_power_difference=float(delta.max(initial=0.)),
                max_relative_power_difference=float(relative.max(initial=0.)),
                selected_endpoint_exact=all(reference['strict'][key] == actual['strict'][key]
                                            for key in ('period', 'score')),
                candidates=candidates))
    return dict(contract='Descriptive native BLS variation, not a TLS-equivalence or tolerance gate',
                comparisons=records,
                changed_power_comparisons=sum(not value['full_power_hash_equal'] for value in records),
                changed_selected_endpoints=sum(not value['selected_endpoint_exact'] for value in records),
                max_absolute_power_difference=max((value['max_absolute_power_difference'] for value in records), default=0.))


def prefix_dispatch_status(backend):
    if backend not in ('candidate', 'baseline'):
        return dict(status='not_applicable', backend=backend)
    from cuvarbase import tls_reference
    helper = getattr(tls_reference, '_native_short_prefix_status', None)
    if helper is None:
        return dict(status='helper_absent_in_this_source', backend=backend)
    return dict(status='recorded', backend=backend, dispatch=helper())


def rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def resource_environment():
    """Record both cgroup versions; host logical CPU count is not a quota."""
    result = environment()
    result['nvcc_command_path'] = shutil.which('nvcc')
    paths = ('cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us', 'cpu/cpu.stat',
             'memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes',
             'memory/memory.max_usage_in_bytes')
    for name in paths:
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result['cgroup'][str(path)] = path.read_text().strip()
    quota = result['cgroup'].get('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = result['cgroup'].get('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota is not None and period is not None and int(quota) > 0:
        result['cpu_quota_cores'] = int(quota)/int(period)
    memory = result['cgroup'].get('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                                 result['cgroup'].get('/sys/fs/cgroup/memory.max'))
    result['host_memory_limit_bytes'] = int(memory) if memory and memory != 'max' else None
    return result


def prepare_grids(cases):
    """Measure one reusable explicit grid per shared survey configuration."""
    from cuvarbase.tls_reference_math import period_grid
    groups, records = {}, []
    for case in cases:
        if case['group'] in groups:
            case['data']['periods'] = groups[case['group']]
            continue
        metadata = case['metadata']
        if 'grid_kwargs' not in metadata:
            records.append(dict(group=case['group'], status='sealed_array_only_no_recipe', seconds=None))
            groups[case['group']] = case['data']['periods']
            continue
        before = time.perf_counter()
        periods = np.sort(period_grid(metadata['baseline_days'], **metadata['grid_kwargs']))
        elapsed = time.perf_counter()-before
        if not np.array_equal(periods, case['data']['periods']):
            raise ValueError('Regenerated common grid differs from sealed input')
        groups[case['group']] = case['data']['periods'] = periods
        records.append(dict(group=case['group'], status='exact_regeneration', seconds=elapsed,
                            nperiods=len(periods), sha256=array_hash(periods), kwargs=metadata['grid_kwargs']))
    return records


def worker(connection, config):
    try:
        started = time.perf_counter()
        if config['source_root']:
            sys.path.insert(0, str(Path(config['source_root']).resolve()))
        cases = load_manifest(config['manifest'], config['names'], config['regimes'])
        seal = configure_bls(cases, config['science_seal']) if config['backend'] == 'bls' else None
        load_seconds = time.perf_counter() - started
        grid_preparation = prepare_grids(cases)
        internal_backend = 'candidate' if config['backend'] == 'baseline' else config['backend']
        if internal_backend == 'bls':
            from cuvarbase.base import ensure_context
            ensure_context()
            import cuvarbase
            science = science_bls_module()
            production = science.production_identity()
            if production != seal['production_sources']:
                raise ValueError('BLS production sources differ from the scientific seal')
            sources = dict(root=str(Path(cuvarbase.__file__).parent), files=production,
                           science_seal_sha256=sha(config['science_seal']),
                           bls_selected=seal['bls_selected'])
        else:
            sources = initialize_backend(internal_backend, correction_adapter=config['correction_adapter'])
        if internal_backend in ('candidate', 'bls'):
            package = Path(sources['root'])
            sources['files'] = {str(path.relative_to(package)): sha(path)
                               for path in sorted(package.rglob('*')) if path.is_file()
                               and path.suffix in ('.py', '.cu', '.cuh')}
        if config['source_root'] and config['backend'] in ('baseline', 'candidate', 'bls'):
            if Path(sources['root']).resolve().parent != Path(config['source_root']).resolve():
                raise RuntimeError('Scientific source import escaped the requested isolated checkout')
        import cupy as cp
        from cupy._core import _accelerator
        owned_allocation = retain_cuda_context()
        connection.send(dict(kind='ready', pid=os.getpid(), namespace_pids=process_ids(),
            cuda_context_allocation_bytes=1, cuda_context_synchronized=True,
            source_files=sources, input_load_seconds=load_seconds,
            cuda_environment=dict(runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                                  driver_version=cp.cuda.runtime.driverGetVersion(),
                                  compute_capability=cp.cuda.Device().compute_capability,
                                  routine_accelerators=_accelerator.get_routine_accelerators(),
                                  cupy_accelerators_environment=os.environ.get('CUPY_ACCELERATORS'),
                                  nvcc_environment=os.environ.get('NVCC')),
            grid_preparation=grid_preparation,
            ready_seconds=time.perf_counter()-started))
        qualification_index = 0
        while True:
            command = connection.recv()
            if command['kind'] == 'close':
                break
            if command['kind'] == 'memory':
                connection.send(dict(kind='memory', pid=os.getpid(), host_peak_rss_bytes=rss_peak_bytes(),
                    cupy_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
                    cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
                    prefix_dispatch=prefix_dispatch_status(config['backend'])))
                continue
            selected = [cases[index] for index in command['indices']]
            profile = cProfile.Profile() if command['kind'] == 'profile' else None
            cp.cuda.runtime.deviceSynchronize()
            before = time.perf_counter()
            error, results, fingerprints, scalars = None, None, [], []
            try:
                if profile:
                    profile.enable()
                results = public_call(internal_backend, selected,
                                      arrays=command['kind'] != 'run')
                cp.cuda.runtime.deviceSynchronize()
            except Exception:
                error = traceback.format_exc()
            finally:
                if profile:
                    profile.disable()
            after = time.perf_counter()
            # Qualifying spectra are hashed after the individual API clock.
            # Scalar checks on measured tasks remain inside sustained wall time.
            if results is not None:
                try:
                    scalars = [dict(case=case['name'], fields=scalar_fingerprint(internal_backend, result))
                               for case, result in zip(selected, results)]
                    if internal_backend == 'bls':
                        for scalar, result in zip(scalars, results):
                            scalar['values'] = {key: result[key] for key in ('period', 'score')}
                    if command['kind'] != 'run':
                        if internal_backend == 'bls':
                            for case, result in zip(selected, results):
                                filename = f"{qualification_index:05d}-{command['kind']}-{case['name']}"
                                archive_bls_spectra(result, Path(config['output'])/'spectra'/str(os.getpid()), filename)
                            qualification_index += 1
                        fingerprints = [complete_fingerprint(internal_backend, case, result)
                                        for case, result in zip(selected, results)]
                    if len(results) != len(selected):
                        raise ValueError('API output count differs from assigned count')
                except Exception:
                    error = traceback.format_exc()
            profile_text = None
            if profile:
                stream = io.StringIO()
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
                profile_text = stream.getvalue()
            del results
            connection.send(dict(kind='complete', pid=os.getpid(), task=command.get('task'),
                indices=command['indices'], started=before, ended=after,
                api_seconds=after-before, error=error, outputs=fingerprints,
                scalars=scalars, profile=profile_text, host_peak_rss_bytes=rss_peak_bytes()))
    except BaseException:
        try:
            connection.send(dict(kind='fatal', traceback=traceback.format_exc()))
        except Exception:
            pass
    finally:
        connection.close()


class Telemetry:
    """Sample the whole device and owned pool; peaks are sampled lower bounds."""
    def __init__(self, path, pids, interval=.1):
        self.path, self.pids, self.interval = Path(path), pids, interval
        self.stop = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        import pynvml as nvml
        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            with self.path.open('w') as output:
                while not self.stop.is_set():
                    row = dict(monotonic=time.perf_counter())
                    try:
                        row['gpu_used_bytes'] = int(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                        row['gpu_utilization_percent'] = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                        row['gpu_power_mw'] = int(nvml.nvmlDeviceGetPowerUsage(handle))
                        row['gpu_sm_clock_mhz'] = int(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
                        row['gpu_temperature_c'] = int(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
                        row['gpu_processes'] = [int(p.pid) for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                        row['host_pool_rss_bytes'] = 0
                        for pid in self.pids:
                            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                                if line.startswith('VmRSS:'):
                                    row['host_pool_rss_bytes'] += int(line.split()[1])*1024
                        for name in ('memory.current', 'memory.peak'):
                            file = Path('/sys/fs/cgroup') / name
                            if file.exists():
                                row['cgroup_' + name] = int(file.read_text())
                        for name, key in (('memory/memory.usage_in_bytes', 'cgroup_memory.current'),
                                          ('memory/memory.max_usage_in_bytes', 'cgroup_memory.peak')):
                            file = Path('/sys/fs/cgroup') / name
                            if file.exists():
                                row[key] = int(file.read_text())
                    except Exception:
                        row['error'] = traceback.format_exc()
                    self.rows.append(row)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    self.stop.wait(self.interval)
        finally:
            nvml.nvmlShutdown()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=5)

    def summarize(self, allowed_pids):
        foreign = sorted({pid for row in self.rows for pid in row.get('gpu_processes', [])
                          if pid not in allowed_pids})
        return dict(sample_interval_seconds=self.interval, samples=len(self.rows),
            scope='Worker startup, full qualification, measured queues, post-qualification and teardown. '
                  'Worker RSS sampling begins after worker readiness; lifetime RSS high-water marks '
                  'also cover imports/startup. GPU and container sampling include startup.',
            foreign_gpu_pids=foreign,
            ownership_passed=bool(self.rows) and not foreign,
            peaks_are_sampled_lower_bounds=True,
            cgroup_peak_scope='Container lifetime high-water mark; may include earlier configurations. '
                              'Per-configuration host/GPU memory uses sampled current memory plus worker lifetime RSS.',
            operating_ranges={name: [min((row[name] for row in self.rows if name in row), default=None),
                                     max((row[name] for row in self.rows if name in row), default=None)]
                              for name in ('gpu_power_mw', 'gpu_sm_clock_mhz', 'gpu_temperature_c',
                                           'gpu_utilization_percent')},
            **{name: max((row[name] for row in self.rows if name in row), default=None)
               for name in ('gpu_used_bytes', 'host_pool_rss_bytes',
                            'cgroup_memory.current', 'cgroup_memory.peak')})


class Pool:
    def __init__(self, config, width, timeout):
        self.timeout, self.connections, self.processes = timeout, [], []
        self.ownership = GPUOwnership()
        self.closed = False
        started = time.perf_counter()
        try:
            context = mp.get_context('spawn')
            for unused in range(width):
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child, config))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.ready = [self.receive(connection, 'ready') for connection in self.connections]
            self.ownership.bind(self.ready, self.processes)
        except BaseException as error:
            error.gpu_ownership = self.close()
            raise
        self.startup_seconds = time.perf_counter() - started

    def receive(self, connection, kind='complete'):
        if not connection.poll(self.timeout):
            raise TimeoutError('Worker exceeded declared per-task timeout')
        message = connection.recv()
        if message['kind'] != kind:
            raise RuntimeError('Unexpected worker response: ' + repr(message))
        return message

    def qualify(self, cohort, kind='qualify'):
        """Every worker sees every batch, concurrently, before and after queues."""
        rows = []
        for indices in cohort:
            for connection in self.connections:
                connection.send(dict(kind=kind, indices=indices))
            rows.extend(dict(worker=index, **self.receive(connection))
                        for index, connection in enumerate(self.connections))
        return rows

    def run_queue(self, cohort, cases, min_sources, min_seconds, scalars):
        """Keep at most one task per worker in flight; stop at whole input cycles."""
        before_owner = exclusive_gpu_processes(self.ownership.allowed_pids)
        if not before_owner['exclusive']:
            raise RuntimeError('GPU ownership check failed before the measured queue')
        started = time.perf_counter()
        jobs = []
        cycles = max(1, int(np.ceil(min_sources / len(cases))),
                     int(np.ceil(len(self.connections) / len(cohort))))
        for cycle in range(cycles):
            jobs.extend(cohort)
        pending, records, submitted = {}, [], 0

        def submit(connection):
            nonlocal submitted
            indices = jobs[submitted]
            connection.send(dict(kind='run', indices=indices, task=submitted))
            pending[connection] = submitted
            submitted += 1

        for connection in self.connections[:len(jobs)]:
            submit(connection)
        while pending:
            ready = wait(list(pending), timeout=self.timeout)
            if not ready:
                raise TimeoutError('Sustained queue worker timeout')
            for connection in ready:
                row = self.receive(connection)
                expected_task = pending.pop(connection)
                if row['task'] != expected_task:
                    raise ValueError('Worker returned a different assigned task')
                row['scalar_match'] = all(value['fields'] == scalars.get(value['case'])
                                          for value in row['scalars'])
                expected_names = Counter(cases[index]['name'] for index in jobs[expected_task])
                row['membership_match'] = Counter(value['case'] for value in row['scalars']) == expected_names
                records.append(row)
                if row['error'] or not row['scalar_match'] or not row['membership_match']:
                    error = RuntimeError('Measured task failed numerical/membership gate: ' + repr(row))
                    error.failed_queue = dict(status='error', elapsed_seconds=time.perf_counter()-started,
                                              tasks=records, failed_task=expected_task,
                                              expected_scalars=scalars)
                    raise error
                if submitted == len(jobs) and time.perf_counter()-started < min_seconds:
                    jobs.extend(cohort)
                if submitted < len(jobs):
                    submit(connection)
        ended = time.perf_counter()
        after_owner = exclusive_gpu_processes(self.ownership.allowed_pids)
        if not after_owner['exclusive']:
            raise RuntimeError('GPU ownership check failed after the measured queue')
        counts = Counter(cases[index]['metadata']['regime'] for job in jobs for index in job)
        return dict(status='ok', source_count=sum(counts.values()), regime_counts=dict(counts),
            elapsed_seconds=ended-started, lightcurves_per_second=sum(counts.values())/(ended-started),
            completed_input_cycles=len(jobs)//len(cohort), tasks=records,
            exclusive_before=before_owner, exclusive_after=after_owner)

    def memory(self):
        for connection in self.connections:
            connection.send(dict(kind='memory'))
        return [self.receive(connection, 'memory') for connection in self.connections]

    def close(self):
        if self.closed:
            return self.ownership.receipt
        self.closed = True
        forced = []
        for connection in self.connections:
            try:
                connection.send(dict(kind='close'))
            except (EOFError, BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                forced.append(process.pid)
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        return self.ownership.finish(self.processes, forced=forced)


def qualify_rows(rows, reference=None, expected_names=None, workers=None):
    """Exact predeclared backend fields; BLS nonwinning powers are diagnostic."""
    expected = {} if reference is None else dict(reference)
    problems = []
    scalars = {}
    by_worker = defaultdict(Counter)
    for row in rows:
        if row['error']:
            problems.append(dict(reason='API failure', error=row['error']))
        for value in row['outputs']:
            name = value['case']
            by_worker[row['worker']][name] += 1
            identity = value['strict']
            if name not in expected:
                if reference is not None:
                    problems.append(dict(reason='Unfrozen case', case=name))
                else:
                    expected[name] = identity
            elif expected[name] != identity:
                problems.append(dict(reason='Changed required search output', case=name,
                                     worker=row['worker']))
        for value in row['scalars']:
            if value['case'] in scalars and scalars[value['case']] != value['fields']:
                problems.append(dict(reason='Changed selected detection', case=value['case']))
            scalars[value['case']] = value['fields']
    if not rows or not expected:
        problems.append(dict(reason='No qualifying outputs'))
    if expected_names is not None:
        target = Counter(expected_names)
        if workers is not None and set(by_worker) != set(range(workers)):
            problems.append(dict(reason='Missing or unexpected qualifying workers'))
        for index, observed in by_worker.items():
            if observed != target:
                problems.append(dict(reason='Qualifying case membership differs', worker=index,
                                     expected=dict(target), observed=dict(observed)))
    return dict(passed=not problems, problems=problems, strict=expected, scalars=scalars)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cases = load_manifest(args.manifest, args.names, args.regimes)
    if args.backend == 'bls':
        configure_bls(cases, args.science_seal)
    cohort = batches(cases, args.batch_size)
    record = dict(status='running', schema_version=1, backend=args.backend,
        workers=args.workers, batch_size=args.batch_size,
        actual_api_batch_sizes=[len(indices) for indices in cohort],
        manifest_sha256=sha(args.manifest), harness_sha256=sha(__file__),
        harness_dependency_sha256={name: sha(ROOT/'benchmarks/tls_reference/timing'/name)
                                   for name in ('common.py', 'benchmark.py')},
        science_seal_sha256=sha(args.science_seal) if args.science_seal else None,
        environment=resource_environment(), config={key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()},
        cohort=[dict(name=case['name'], regime=case['metadata']['regime'],
                     nobs=len(case['data']['t']), nperiods=len(case['data']['periods']),
                     input_sha256=case['input_sha256']) for case in cases],
        timing_boundary='Persistent bounded queue: dispatch, complete public API including validation, '
            'template preparation, GPU transfers, search/refinement, requested result construction, '
            'scalar verification and task completion. Input-file loading, imports/context setup and '
            'first calls recorded separately and amortized in end-to-end rate. Explicit grids are regenerated '
            'once per shared configuration/worker from sealed metadata, byte checked and amortized in cold '
            'preparation. Historical inputs without a grid recipe are explicitly marked array-only.',
        output_policy='cuvarbase TLS return_arrays=False; native public API always returns arrays and extra diagnostics. '
            'BLS uses the sealed GPU search and only its selected ranker during measured calls; '
            'complete power/period arrays, masks, hashes and all rankers are retained only for qualification. '
            'BLS eligibility requires exact period arrays, finite masks, selected period and selected score; '
            'native atomic variation in nonwinning powers and unused rankers is reported separately. '
            'TLS complete-spectrum eligibility is unchanged. '
            'All distinct case spectra checked on every worker before and after queues.',
        cold_cache_policy='Fresh worker processes; existing filesystem compiler/kernel caches are retained. '
            'Cold-start values include actual first-use setup/compilation/canary costs incurred in this state, '
            'but are not empty-disk-cache installation or first-ever compilation measurements. '
            'The guarded short-row CUB kernel uses direct NVCC compilation with explicit FTZ disabled; '
            'it has only a bounded process/context memory cache and pays its compilation/canary on each '
            'fresh supported worker/context, even if other CuPy filesystem caches are warm.',
        queue_population_note='Fixed distinct source cohort repeated in whole cycles; repetitions measure '
            'steady-state execution, not independent astrophysical population draws.',
        qualification=[], repetitions=[])
    write(args.output/'result.json', record)
    pool = None
    telemetry = Telemetry(args.output/'telemetry.jsonl', [])
    try:
        telemetry.__enter__()
        config = dict(manifest=str(args.manifest.resolve()), names=args.names, regimes=args.regimes,
                      backend=args.backend, source_root=args.source_root,
                      output=str(args.output.resolve()),
                      science_seal=str(args.science_seal.resolve()) if args.science_seal else None,
                      correction_adapter=str(args.correction_adapter.resolve()))
        pool = Pool(config, args.workers, args.timeout)
        telemetry.pids[:] = [process.pid for process in pool.processes]
        record.update(worker_ready=pool.ready, startup_seconds=pool.startup_seconds,
                      parent_input_load_and_setup_seconds=time.perf_counter()-started-pool.startup_seconds)
        reference, reference_rows = None, None
        if args.reference:
            frozen = json.loads(args.reference.read_text())
            if frozen['status'] != 'ok' or frozen['workers'] != 1:
                raise ValueError('Pool reference must be a successful one-worker run')
            if frozen['backend'] != args.backend:
                raise ValueError('Pool qualification reference must use the same backend')
            if frozen.get('science_seal_sha256') != record['science_seal_sha256']:
                raise ValueError('Pool reference uses another scientific BLS selection')
            if frozen['manifest_sha256'] != record['manifest_sha256']:
                raise ValueError('Pool reference belongs to another input manifest')
            reference = frozen['qualification'][0]['gate']['strict']
            reference_rows = frozen['qualification'][0]['rows']
        cold_started = time.perf_counter()
        cold = pool.qualify(cohort)
        record['first_full_cohort_seconds'] = time.perf_counter()-cold_started
        record['cold_first_public_batch_api_seconds_by_worker'] = [row['api_seconds'] for row in cold[:args.workers]]
        record['cold_accounting_note'] = ('First-cohort time includes qualification hashing and runs each '
            'distinct input on every worker. Cold-amortized throughput therefore conservatively charges '
            'validation warmup overhead; first-public-batch API latency is separately retained.')
        expected_names = [case['name'] for case in cases]
        gate = qualify_rows(cold, reference, expected_names, args.workers)
        record['qualification'].append(dict(phase='before', rows=cold, gate=gate))
        if args.backend == 'bls':
            record['qualification'][-1]['native_repeat_diagnostics'] = bls_repeat_diagnostics(cold, reference_rows)
        if not gate['passed'] or set(gate['strict']) != {case['name'] for case in cases}:
            raise RuntimeError('Pre-queue required-output qualification failed')
        record['worker_after_warmup'] = pool.memory()
        if args.profile:
            record['profiles'] = pool.qualify(cohort, kind='profile')
            if not qualify_rows(record['profiles'], gate['strict'], expected_names, args.workers)['passed']:
                raise RuntimeError('Profile instrumentation changed search outputs')
        write(args.output/'result.json', record)
        for repetition in range(args.repetitions):
            measured = pool.run_queue(cohort, cases, args.min_sources, args.min_seconds, gate['scalars'])
            measured['repetition'] = repetition
            measured['estimated_workload_compute_usd'] = args.hourly_usd*measured['elapsed_seconds']/3600
            record['repetitions'].append(measured)
            write(args.output/'result.json', record)
        record['worker_after_queues'] = pool.memory()
        after = pool.qualify(cohort)
        final_gate = qualify_rows(after, gate['strict'], expected_names, args.workers)
        record['qualification'].append(dict(phase='after', rows=after, gate=final_gate))
        if args.backend == 'bls':
            record['qualification'][-1]['native_repeat_diagnostics'] = bls_repeat_diagnostics(after, reference_rows or cold)
        if not final_gate['passed']:
            raise RuntimeError('Post-queue required-output qualification failed')
        record['worker_memory_before_teardown'] = pool.memory()
        record['status'] = 'ok'
    except BaseException as error:
        record.update(status='error', error=traceback.format_exc())
        if hasattr(error, 'failed_queue'):
            record['failed_queue'] = error.failed_queue
        if hasattr(error, 'gpu_ownership'):
            record['gpu_ownership'] = error.gpu_ownership
    finally:
        if pool is not None:
            record['gpu_ownership'] = pool.close()
            if not record['gpu_ownership']['passed']:
                record['status'] = 'error'
        telemetry.__exit__()
        record['memory'] = telemetry.summarize(pool.ownership.allowed_pids if pool is not None else [])
        if (not record['memory']['ownership_passed'] or
                record['memory']['gpu_used_bytes'] is None or
                record['memory']['host_pool_rss_bytes'] is None):
            record['status'] = 'error'
            record.setdefault('error', 'Missing memory telemetry or unexpected GPU process during the run')
        record['total_campaign_seconds'] = time.perf_counter()-started
        if record['status'] == 'ok':
            elapsed = sum(row['elapsed_seconds'] for row in record['repetitions'])
            count = sum(row['source_count'] for row in record['repetitions'])
            preparation = record['startup_seconds'] + record['first_full_cohort_seconds'] + record['parent_input_load_and_setup_seconds']
            median_rate = float(np.median([row['lightcurves_per_second'] for row in record['repetitions']]))
            record['summary'] = dict(steady_state_lightcurves_per_second=count/elapsed,
                median_repetition_lightcurves_per_second=median_rate,
                total_measured_sources=count, total_measured_seconds=elapsed,
                total_measured_compute_usd=args.hourly_usd*elapsed/3600,
                observed_rate_min=min(row['lightcurves_per_second'] for row in record['repetitions']),
                observed_rate_max=max(row['lightcurves_per_second'] for row in record['repetitions']),
                cold_first_cohort_including_startup_seconds=preparation,
                cold_preparation_compute_usd=args.hourly_usd*preparation/3600,
                cold_amortized_lightcurves_per_second=count/(elapsed+preparation),
                usd_per_million_steady=args.hourly_usd*1e6/(3600*median_rate),
                usd_per_million_cold_amortized=args.hourly_usd*(elapsed+preparation)*1e6/(3600*count),
                estimated_run_compute_usd=args.hourly_usd*record['total_campaign_seconds']/3600,
                timing_scope_note='Cost projection includes measured reusable grid preparation in cold amortization; '
                    'excludes survey data acquisition/preprocessing and vetting; no million-source execution claim.')
        write(args.output/'result.json', record)
    print(json.dumps({key: record[key] for key in ('status', 'summary', 'error') if key in record}), flush=True)
    return 0 if record['status'] == 'ok' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('baseline', 'candidate', 'gtls', 'gtls_corrected', 'bls'), required=True)
    parser.add_argument('--science-seal', type=Path, help='Frozen scientific BLS method/ranker choices')
    parser.add_argument('--source-root', help='Isolated checkout/package root for this worker backend')
    parser.add_argument('--correction-adapter', type=Path,
        default=ROOT/'benchmarks/tls_reference/corrected_reference.py')
    parser.add_argument('--reference', type=Path, help='Same-backend one-worker result.json frozen before tuning')
    parser.add_argument('--names', nargs='*', default=[])
    parser.add_argument('--regimes', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--min-sources', type=int, default=96)
    parser.add_argument('--min-seconds', type=float, default=60.)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--profile', action='store_true', help='Extra qualified cProfile run, excluded from headline timings')
    args = parser.parse_args()
    for name in ('workers', 'batch_size', 'repetitions', 'min_sources'):
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if args.min_seconds < 0 or args.hourly_usd < 0:
        parser.error('Time/cost controls must be nonnegative')
    if args.backend == 'baseline' and not args.source_root:
        parser.error('baseline requires --source-root')
    if args.backend == 'bls' and not args.science_seal:
        parser.error('BLS requires --science-seal from independent scientific development')
    if args.workers > 1 and not args.reference:
        parser.error('concurrent pools require --reference from the same-backend one-worker run')
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
