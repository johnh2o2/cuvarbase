#!/usr/bin/env python3
"""Separate native-BLS execution timing; original numerical exclusion is retained.

This supplement never grants numerical qualification. It uses the unchanged
compact BLS API wrapper, input recipes, resource monitor and ownership machinery.
Only native API/invalid-output errors are counted as failed completions. Broken
instrumentation, membership, source identity or resource ownership invalidates
the entire configuration's rates. No CUDA imports occur in the parent process.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.tls_survey import throughput as native
from benchmarks.tls_survey import throughput_campaign as primary

SCOPES = (*primary.REGIMES, 'varied')
CLEANUP_RESERVE_SECONDS = 120


class Interrupted(RuntimeError):
    pass


def deadline_check(deadline):
    if time.time() >= deadline - CLEANUP_RESERVE_SECONDS:
        raise Interrupted('Shared supplemental deadline reached; reserving cleanup time')


def process_start_ticks():
    path = Path('/proc/self/stat')
    return int(path.read_text().rsplit(')',1)[1].split()[19]) if path.exists() else None


def source_identity(protocol):
    paths = [Path(__file__), Path(protocol), Path(native.__file__), Path(primary.__file__),
             ROOT/'benchmarks/tls_reference/timing/common.py',
             ROOT/'benchmarks/tls_reference/timing/benchmark.py']
    return {str(p.relative_to(ROOT)): native.sha(p) for p in paths}


def allocation(env):
    values = (env.get('nvidia_smi'), env.get('cpu_quota_cores'), env.get('host_memory_limit_bytes'))
    if any(v is None for v in values):
        raise ValueError('Incomplete GPU/CPU/memory allocation identity')
    if any(value != '1' for value in env['cpu_math_thread_environment'].values()):
        raise ValueError('Numerical CPU threads must remain one')
    return values


def cohort_identity(cases):
    return [dict(name=c['name'], regime=c['metadata']['regime'], nobs=len(c['data']['t']),
                 nperiods=len(c['data']['periods']), input_sha256=c['input_sha256']) for c in cases]


def first_anchors(rows):
    """Only the first scheduled worker-0 observation can anchor each case."""
    anchors, seen = {}, set()
    for row in rows:
        if row['worker'] != 0:
            continue
        for observation in row['observations']:
            name = observation['case']
            if name in seen:
                continue
            seen.add(name)
            if observation['status'] == 'success':
                anchors[name] = observation
    return anchors


def compare_observation(observation, anchor):
    """Diagnostic exact comparisons without any passing tolerance or gate."""
    result = dict(case=observation['case'], status=observation['status'])
    if observation['status'] != 'success':
        return dict(result, comparison='API failure; no output comparison')
    if anchor is None:
        return dict(result, comparison='fixed first-observation reference unavailable')
    changed = [k for k in ('period', 'score') if
               observation['scalar']['fields'][k] != anchor['scalar']['fields'][k]]
    result.update(comparison='recorded', changed_selected_fields=changed)
    if changed:
        result.update(expected=anchor['scalar']['values'], actual=observation['scalar']['values'],
                      differences={k:observation['scalar']['values'][k]-anchor['scalar']['values'][k]
                                   for k in changed})
    if 'full' in observation and 'full' in anchor:
        old, new = anchor['full'], observation['full']
        result['changed_complete_fields'] = [k for k in sorted(set(old['fields']) | set(new['fields']))
                                             if old['fields'].get(k) != new['fields'].get(k)]
        arrays = []
        for full in (old, new):
            artifact = full['spectrum_artifact']
            if native.sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Archived complete BLS spectrum changed')
            with np.load(artifact['path'], allow_pickle=False) as data:
                arrays.append(np.array(data['power'], copy=True))
        if arrays[0].shape != arrays[1].shape:
            raise ValueError('BLS power shape changed within a fixed input')
        finite = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
        delta = np.abs(arrays[1][finite].astype(float)-arrays[0][finite].astype(float))
        result.update(changed_finite_power_values=int(np.count_nonzero(delta)),
                      maximum_absolute_power_difference=float(delta.max()) if delta.size else None,
                      changed_finite_mask_values=int(np.count_nonzero(
                          np.isfinite(arrays[0]) != np.isfinite(arrays[1]))))
    return result


def account_task(row, expected_indices, cases, anchors):
    """Fail closed on instrumentation, while retaining native failures and drift."""
    if row['indices'] != expected_indices:
        raise ValueError('Returned indices differ from assigned task')
    expected = [cases[i]['name'] for i in expected_indices]
    actual = [v['case'] for v in row['observations']]
    if actual != expected:
        raise ValueError('Attempted-case membership/order differs from assigned task')
    statuses = [v['status'] for v in row['observations']]
    if any(s not in ('success', 'api_error') for s in statuses):
        raise ValueError('Unknown or instrumental observation status')
    for value in row['observations']:
        if value['status'] == 'success' and not all(np.isfinite(value['scalar']['values'][k])
                                                   for k in ('period', 'score')):
            raise ValueError('Worker labeled nonfinite output successful')
    return dict(attempted_count=len(expected), successful_count=statuses.count('success'),
                failed_count=statuses.count('api_error'),
                comparisons=[compare_observation(v, anchors.get(v['case'])) for v in row['observations']])


def queue_summary(rows, elapsed):
    if elapsed <= 0:
        raise ValueError('Elapsed queue time must be positive')
    attempted = sum(r['accounting']['attempted_count'] for r in rows)
    success = sum(r['accounting']['successful_count'] for r in rows)
    failed = sum(r['accounting']['failed_count'] for r in rows)
    if attempted != success+failed or not attempted:
        raise ValueError('Incomplete attempt accounting')
    return dict(attempted_count=attempted, successful_count=success, failed_count=failed,
                elapsed_seconds=elapsed, successful_lightcurves_per_second=success/elapsed,
                attempted_lightcurves_per_second=attempted/elapsed,
                failure_fraction=failed/attempted, completion_fraction=success/attempted,
                sum_worker_api_seconds=sum(v.get('api_seconds',0.) for row in rows
                                           for v in row['observations']),
                api_seconds_note='Diagnostic sum across possibly overlapping workers; '
                'excludes per-case journals and is never the throughput denominator.')


def execution_winner(records):
    valid = [r for r in records if r.get('execution_rates_valid') and
             r['summary']['successful_lightcurves_per_second'] > 0]
    return min(valid, key=lambda r: (-r['summary']['median_repetition_successful_lightcurves_per_second'],
                                    r['workers'], r['batch_size'])) if valid else None


def validate_diagnostic_coverage(rows, cohort, workers, cases):
    expected = Counter((w,tuple(indices)) for indices in cohort for w in range(workers))
    observed = Counter((row['worker'],tuple(row['indices'])) for row in rows)
    if observed != expected:
        raise ValueError('Diagnostic worker/batch coverage differs from assigned cohort')
    for row in rows:
        account_task(row,row['indices'],cases,{})


def worker(connection, config):
    journal = None
    try:
        started = time.perf_counter()
        sys.path.insert(0, config['source_root'])
        cases = native.load_manifest(config['manifest'], config['names'])
        seal = native.configure_bls(cases, config['science_seal'])
        load_seconds = time.perf_counter()-started
        grids = native.prepare_grids(cases)
        from cuvarbase.base import ensure_context
        ensure_context()
        import cuvarbase
        import cupy as cp
        science = native.science_bls_module()
        if science.production_identity() != seal['production_sources']:
            raise ValueError('Production sources changed from science seal')
        package = Path(cuvarbase.__file__).resolve().parent
        if package.parent != Path(config['source_root']).resolve():
            raise ValueError('BLS import escaped requested checkout')
        sources = {str(p.relative_to(package)): native.sha(p) for p in sorted(package.rglob('*'))
                   if p.is_file() and p.suffix in ('.py', '.cu', '.cuh')}
        owned_context = native.retain_cuda_context()
        connection.send(dict(kind='ready', pid=os.getpid(), namespace_pids=native.process_ids(),
                             process_start_ticks=process_start_ticks(),
                             supplement_owner_token=os.environ.get('CUVARBASE_SURVEY_BLS_SUPPLEMENT_OWNER'),
                             cuda_context_allocation_bytes=1, cuda_context_synchronized=True,
                             source_files=dict(root=str(package), files=sources),
                             input_load_seconds=load_seconds, grid_preparation=grids,
                             ready_seconds=time.perf_counter()-started))
        counter = 0
        journal = (Path(config['output'])/f'worker-{os.getpid()}-attempts.jsonl').open('x')

        def event(value):
            journal.write(json.dumps(dict(pid=os.getpid(),monotonic=time.perf_counter(),**value))+'\n')
            journal.flush()

        while True:
            command = connection.recv()
            if command['kind'] == 'close':
                break
            if command['kind'] == 'memory':
                connection.send(dict(kind='memory', pid=os.getpid(), host_peak_rss_bytes=native.rss_peak_bytes(),
                                     cupy_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
                                     cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes()))
                continue
            before = time.perf_counter()
            observations = []
            for index in command['indices']:
                deadline_check(config['deadline_epoch'])
                case = cases[index]
                observation = dict(case=case['name'], index=index)
                event(dict(event='attempt_started',sequence=counter,case=case['name'],
                           input_sha256=case['input_sha256'],kind=command['kind'],task=command.get('task'),
                           repetition=command.get('repetition')))
                api_started = time.perf_counter()
                try:
                    cp.cuda.runtime.deviceSynchronize()
                    result = native.compact_bls(case, science, arrays=command['kind'] != 'run')
                    cp.cuda.runtime.deviceSynchronize()
                    # Invalid selected native outputs are failed executions, not fast successes.
                    if not all(np.isfinite(result[k]) for k in ('period', 'score')):
                        raise ValueError('Native BLS returned a nonfinite selected endpoint')
                except Exception:
                    observation.update(status='api_error', error=traceback.format_exc(),
                                       api_seconds=time.perf_counter()-api_started)
                else:
                    observation['api_seconds'] = time.perf_counter()-api_started
                    event(dict(event='api_returned',sequence=counter,case=case['name'],
                               period=result['period'],score=result['score']))
                    # Any failure in diagnostic construction/archive is instrumental and fatal.
                    observation.update(status='success', scalar=dict(
                        fields=native.scalar_fingerprint('bls', result),
                        values={k:result[k] for k in ('period', 'score')}))
                    if command['kind'] != 'run':
                        native.archive_bls_spectra(result, Path(config['output'])/'spectra'/str(os.getpid()),
                                                   f'{counter:06d}-{case["name"]}')
                        observation['full'] = native.complete_fingerprint('bls', case, result)
                    del result
                event(dict(event='attempt_completed',sequence=counter,kind=command['kind'],
                           task=command.get('task'),repetition=command.get('repetition'),observation=observation))
                counter += 1
                observations.append(observation)
            connection.send(dict(kind='complete', pid=os.getpid(), task=command.get('task'),
                                 repetition=command.get('repetition'),
                                 indices=command['indices'], observations=observations,
                                 api_and_diagnostic_seconds=time.perf_counter()-before,
                                 host_peak_rss_bytes=native.rss_peak_bytes()))
    except BaseException:
        error = traceback.format_exc()
        if journal is not None:
            journal.write(json.dumps(dict(event='worker_interrupted',pid=os.getpid(),error=error))+'\n')
            journal.flush()
        try:
            connection.send(dict(kind='fatal', traceback=error))
        except Exception:
            pass
    finally:
        if journal is not None:
            journal.close()
        connection.close()


class Pool(native.Pool):
    def __init__(self, config, width, deadline):
        self.deadline = deadline
        self.timeout, self.connections, self.processes = 1800, [], []
        self.ownership, self.closed = native.GPUOwnership(), False
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
            self.ready = [self.receive(c, 'ready') for c in self.connections]
            self.ownership.bind(self.ready, self.processes)
        except BaseException as error:
            error.gpu_ownership = self.close()
            raise
        self.startup_seconds = time.perf_counter()-started

    def receive(self, connection, kind='complete'):
        deadline_check(self.deadline)
        timeout = min(self.timeout, self.deadline-CLEANUP_RESERVE_SECONDS-time.time())
        if not connection.poll(max(0, timeout)):
            raise Interrupted('Worker timeout/deadline; incomplete configuration invalid')
        value = connection.recv()
        if value['kind'] != kind:
            raise RuntimeError('Worker instrumentation failed: '+repr(value))
        return value

    def diagnostic(self, cohort, output):
        rows = []
        with Path(output).open('x') as stream:
            for indices in cohort:
                for connection in self.connections:
                    connection.send(dict(kind='qualify', indices=indices))
                for index, connection in enumerate(self.connections):
                    row = dict(worker=index, **self.receive(connection))
                    if row['indices'] != indices or row['task'] is not None:
                        raise ValueError('Diagnostic worker returned a different assigned task')
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
        return rows

    def queue(self, cohort, cases, min_attempts, min_seconds, anchors, journal, repetition=0):
        ownership = native.exclusive_gpu_processes(self.ownership.allowed_pids)
        if not ownership['exclusive']:
            raise ValueError('GPU ownership failed before queue')
        started = time.perf_counter()
        cycles = max(1, int(np.ceil(min_attempts/len(cases))),
                     int(np.ceil(len(self.connections)/len(cohort))))
        jobs = cohort*cycles
        submitted, pending, rows = 0, {}, []

        def submit(connection):
            nonlocal submitted
            connection.send(dict(kind='run', indices=jobs[submitted], task=submitted,repetition=repetition))
            pending[connection] = submitted
            submitted += 1

        with Path(journal).open('x') as stream:
            for connection in self.connections[:len(jobs)]:
                submit(connection)
            while pending:
                deadline_check(self.deadline)
                ready = wait(list(pending), timeout=min(self.timeout, max(0,
                             self.deadline-CLEANUP_RESERVE_SECONDS-time.time())))
                if not ready:
                    raise Interrupted('Queue timeout/deadline; journal retains completed attempts')
                for connection in ready:
                    row = self.receive(connection)
                    task = pending.pop(connection)
                    if row['task'] != task or row['repetition'] != repetition:
                        raise ValueError('Worker task identity changed')
                    row['accounting'] = account_task(row, jobs[task], cases, anchors)
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    if submitted == len(jobs) and time.perf_counter()-started < min_seconds:
                        jobs.extend(cohort)
                    if submitted < len(jobs):
                        submit(connection)
        elapsed = time.perf_counter()-started
        final_owner = native.exclusive_gpu_processes(self.ownership.allowed_pids)
        if not final_owner['exclusive']:
            raise ValueError('GPU ownership failed after queue')
        summary = queue_summary(rows, elapsed)
        if summary['attempted_count'] < min_attempts or elapsed < min_seconds:
            raise ValueError('Completed queue does not meet predeclared minima')
        return dict(status='completed_queue', **summary, completed_input_cycles=len(jobs)//len(cohort),
                    tasks=rows, exclusive_before=ownership, exclusive_after=final_owner,
                    journal=dict(path=str(Path(journal).resolve()), sha256=native.sha(journal)))


def numerical_summary(diagnostics, repetitions):
    comparisons = [value for phase in diagnostics for row in phase['rows']
                   for value in row['accounting']['comparisons']]
    comparisons += [value for rep in repetitions for row in rep['tasks']
                    for value in row['accounting']['comparisons']]
    return dict(original_qualification_passed=False,
                interpretation='Original exact repeatability qualification remains failed. '
                'This supplement supplies execution rates, no new numerical acceptance threshold.',
                selected_mismatch_count=sum(bool(v.get('changed_selected_fields')) for v in comparisons),
                complete_output_mismatch_count=sum(bool(v.get('changed_complete_fields')) for v in comparisons),
                reference_missing_count=sum(v['comparison']=='fixed first-observation reference unavailable'
                                            for v in comparisons),
                comparison_count=len(comparisons))


def run_configuration(args, manifest, names, scope, workers, batch_size, output,
                      expected_allocation, expected_cohort, anchors=None, repetitions=1,
                      min_attempts=24, min_seconds=30):
    deadline_check(args.deadline_epoch)
    if repetitions*min_seconds > args.deadline_epoch-time.time()-CLEANUP_RESERVE_SECONDS:
        raise Interrupted('Known minimum configuration duration cannot fit shared deadline')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    record = dict(schema_version=1, status='running', scope=scope, backend='native_bls_execution',
                  workers=workers, batch_size=batch_size, execution_rates_valid=False,
                  original_qualification_passed=False,
                  original_numerical_qualification_passed=False, repetitions=[], diagnostics=[],
                  source_identity=source_identity(args.protocol), science_seal_sha256=native.sha(args.science_seal),
                  manifest_sha256=native.sha(manifest), config=dict(manifest=str(manifest), names=names,
                                                                  hourly_usd=args.hourly_usd),
                  timing_boundary='Same preparation/full-grid/transfer/CPU/GPU boundaries as primary timing. '
                  'Instrumentation overhead differs: the supplement flushes three per-case worker journal events '
                  'for successful calls (start, API return, completion), two for failures, and one parent task '
                  'record, in addition to exact scalar comparisons. All are inside elapsed queue time. '
                  'Diagnostic summed worker API seconds exclude these journals and never supply a rate denominator. '
                  'Failed API calls consume elapsed time and '
                  'count as attempts, never successful completions. Full-array diagnostics run outside queues.',
                  cold_cache_policy='Fresh processes; existing filesystem compiler caches retained. '
                  'First-cohort complete-array diagnostic cost is conservatively charged to cold amortization.')
    record.update(args.authorization_identity)
    pool, telemetry = None, native.Telemetry(output/'telemetry.jsonl', [])
    try:
        cases = native.load_manifest(manifest, names)
        native.configure_bls(cases, args.science_seal)
        record.update(cohort=cohort_identity(cases), environment=native.resource_environment())
        if record['cohort'] != expected_cohort:
            raise ValueError('Supplement cohort differs from original timing inputs')
        if allocation(record['environment']) != tuple(expected_allocation):
            raise ValueError('GPU/CPU/RAM allocation differs from primary campaign')
        cohort = native.batches(cases, batch_size)
        record['actual_api_batch_sizes'] = [len(x) for x in cohort]
        native.write(output/'result.json', record)
        telemetry.__enter__()
        pool = Pool(dict(manifest=str(manifest), names=names, science_seal=str(args.science_seal),
                         source_root=str(args.source_root), output=str(output),
                         deadline_epoch=args.deadline_epoch), workers, args.deadline_epoch)
        telemetry.pids[:] = [p.pid for p in pool.processes]
        record.update(worker_ready=pool.ready, startup_seconds=pool.startup_seconds,
                      parent_input_load_and_setup_seconds=time.perf_counter()-start-pool.startup_seconds)
        cold_start = time.perf_counter()
        before = pool.diagnostic(cohort, output/'diagnostic-before.jsonl')
        validate_diagnostic_coverage(before,cohort,workers,cases)
        record['first_full_cohort_seconds'] = time.perf_counter()-cold_start
        record['cold_first_public_call_seconds_by_worker'] = [
            row['observations'][0]['api_seconds'] for row in before[:workers] if row['observations']]
        if anchors is None:
            anchors = first_anchors(before)
        for row in before:
            row['accounting'] = account_task(row, row['indices'], cases, anchors)
        record['fixed_reference_anchors'] = anchors
        record['diagnostics'].append(dict(phase='before', rows=before))
        record['worker_after_warmup'] = pool.memory()
        native.write(output/'result.json', record)
        for i in range(repetitions):
            if min_seconds > args.deadline_epoch-time.time()-CLEANUP_RESERVE_SECONDS:
                raise Interrupted('Known minimum queue duration cannot fit shared deadline')
            result = pool.queue(cohort, cases, min_attempts, min_seconds, anchors,
                                output/f'queue-{i}.jsonl',repetition=i)
            record['repetitions'].append(dict(repetition=i, **result))
            native.write(output/'result.json', record)
        record['worker_after_queues'] = pool.memory()
        after = pool.diagnostic(cohort, output/'diagnostic-after.jsonl')
        validate_diagnostic_coverage(after,cohort,workers,cases)
        for row in after:
            row['accounting'] = account_task(row, row['indices'], cases, anchors)
        record['diagnostics'].append(dict(phase='after', rows=after))
        record['worker_memory_before_teardown'] = pool.memory()
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='partial' if isinstance(error, Interrupted) else 'error',
                      error=traceback.format_exc())
        if hasattr(error, 'gpu_ownership'):
            record['gpu_ownership'] = error.gpu_ownership
    finally:
        if pool is not None:
            telemetry.pids[:] = []
            teardown_start = time.perf_counter()
            try:
                record['gpu_ownership'] = pool.close()
            except BaseException:
                record.update(status='error',cleanup_error=traceback.format_exc())
                record['gpu_ownership'] = dict(passed=False,status='cleanup_instrumentation_failure',
                                               receipt=pool.ownership.receipt)
            record['teardown_seconds'] = time.perf_counter()-teardown_start
        if telemetry.thread.ident is not None:
            telemetry.__exit__()
        record['memory'] = telemetry.summarize(pool.ownership.allowed_pids if pool else [])
        record['telemetry_errors'] = [r for r in telemetry.rows if r.get('error')]
        record['total_campaign_seconds'] = time.perf_counter()-start
        record['numerical'] = numerical_summary(record['diagnostics'], record['repetitions'])
        instrument_ok = (record.get('gpu_ownership', {}).get('passed') is True and
                         record['memory']['ownership_passed'] and not record['telemetry_errors'] and
                         record['memory']['gpu_used_bytes'] is not None and
                         record['memory']['host_pool_rss_bytes'] is not None)
        record['execution_rates_valid'] = record['status']=='complete' and instrument_ok
        if not instrument_ok:
            record['status'] = 'error'
            record.setdefault('error', 'Ownership or memory instrumentation failed')
        if record['execution_rates_valid']:
            elapsed = sum(v['elapsed_seconds'] for v in record['repetitions'])
            tasks = [t for v in record['repetitions'] for t in v['tasks']]
            summary = queue_summary(tasks, elapsed)
            preparation = (record['startup_seconds']+record['first_full_cohort_seconds']+
                           record['parent_input_load_and_setup_seconds'])
            rates = [r['successful_lightcurves_per_second'] for r in record['repetitions']]
            success = summary['successful_count']
            record['summary'] = dict(summary,
                median_repetition_successful_lightcurves_per_second=float(np.median(rates)),
                observed_rate_min=min(rates), observed_rate_max=max(rates),
                cold_first_cohort_including_startup_seconds=preparation,
                cold_amortized_successful_lightcurves_per_second=success/(elapsed+preparation),
                total_measured_compute_usd=args.hourly_usd*elapsed/3600,
                cold_preparation_compute_usd=args.hourly_usd*preparation/3600,
                estimated_run_compute_usd=args.hourly_usd*record['total_campaign_seconds']/3600,
                cost_per_attempt_usd=args.hourly_usd*elapsed/(3600*summary['attempted_count']),
                cost_per_successful_lightcurve_usd=None if not success else args.hourly_usd*elapsed/(3600*success),
                whole_configuration_successful_lightcurves_per_second=success/record['total_campaign_seconds'],
                whole_configuration_cost_per_successful_lightcurve_usd=None if not success else
                    args.hourly_usd*record['total_campaign_seconds']/(3600*success),
                usd_per_million_successful=None if not success else args.hourly_usd*elapsed*1e6/(3600*success))
        record['artifact_sha256'] = {str(p.relative_to(output)):native.sha(p) for p in sorted(output.rglob('*'))
                                     if p.is_file() and p.name != 'result.json'}
        native.write(output/'result.json', record)
    return record


def checked_json(path, expected_sha=None):
    if expected_sha is not None and native.sha(path) != expected_sha:
        raise ValueError('Receipt changed: '+str(path))
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_authorization(args):
    """Validate the prospective source seal and mechanical post-primary binding."""
    seal = checked_json(args.supplement_seal,args.supplement_seal_sha256)
    binding = checked_json(args.supplement_binding,args.supplement_binding_sha256)
    if seal['kind'] != 'native_bls_execution_supplement' or seal['schema'] != 1:
        raise ValueError('Unexpected supplement authorization schema')
    if seal['budget'] != dict(gpu_cap_seconds=3600,cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS):
        raise ValueError('Supplement budget/cleanup policy changed')
    if binding['schema'] != 1 or binding['supplement_seal_sha256'] != args.supplement_seal_sha256:
        raise ValueError('Mechanical binding uses another supplement seal')
    for key in ('science_seal_sha256','auxiliary_plan_sha256'):
        if binding[key] != seal[key]:
            raise ValueError('Mechanical binding changes reviewed identity: '+key)
    if native.sha(args.science_seal) != seal['science_seal_sha256']:
        raise ValueError('Scientific seal changed')
    required = [str(ROOT/name) for name in source_identity(args.protocol)]
    if any(path not in seal['remote_files'] for path in required):
        raise ValueError('A supplement runner/timing dependency was not source sealed')
    for path,digest in seal['remote_files'].items():
        if native.sha(path) != digest:
            raise ValueError('Supplement source changed: '+path)
    rule = seal['binding_rule']
    if rule['primary_tuning_path'] not in seal['remote_files']:
        raise ValueError('Original development tuning was not prospectively sealed')
    if (str(args.primary_tuning) != rule['primary_tuning_path'] or
            str(args.primary_measurement) != rule['primary_measurement_path']):
        raise ValueError('Primary paths differ from reviewed mechanical binding rule')
    checked_json(args.primary_tuning,binding['primary_tuning_sha256'])
    measurement = checked_json(args.primary_measurement,binding['primary_measurement_sha256'])
    if measurement['status'] != 'complete' or measurement['stage'] != 'measure':
        raise ValueError('Mechanical binding requires completed primary measurement')
    if measurement['science_seal_sha256'] != seal['science_seal_sha256']:
        raise ValueError('Primary measurement science seal differs')
    if native.sha(rule['primary_state_path']) != binding['primary_state_sha256']:
        raise ValueError('Bound primary completion state changed')
    if native.sha(rule['primary_bundle_receipt_path']) != binding['primary_bundle_receipt_sha256']:
        raise ValueError('Bound primary archive receipt changed')
    if (measurement['manifest_sha256'] != binding['primary_measurement_manifest_sha256'] or
            measurement['varied_manifest_sha256'] != binding['primary_measurement_varied_manifest_sha256']):
        raise ValueError('Bound primary manifest identity changed')
    if native.sha(measurement['varied_manifest']) != measurement['varied_manifest_sha256']:
        raise ValueError('Bound varied-cohort manifest changed')
    expected = []
    for row in measurement['configs']:
        result = checked_json(args.primary_measurement.parent/row['result'],row['result_sha256'])
        expected.append(dict(scope=row['scope'],result=row['result'],result_sha256=row['result_sha256'],
                             cohort_sha256=canonical_sha(result['cohort']),
                             environment_sha256=canonical_sha(result['environment'])))
    if binding['primary_configs'] != expected:
        raise ValueError('Mechanical binding changes original cohort/resource receipts')
    return dict(supplement_seal_sha256=args.supplement_seal_sha256,
                supplement_binding_sha256=args.supplement_binding_sha256,
                science_seal_sha256=seal['science_seal_sha256'],
                auxiliary_plan_sha256=seal['auxiliary_plan_sha256'],
                primary_tuning_sha256=binding['primary_tuning_sha256'],
                primary_measurement_sha256=binding['primary_measurement_sha256'])


def primary_panel(measurement_path, campaign, scope):
    for row in campaign['configs']:
        if row['scope'] == scope:
            result_path = Path(measurement_path).parent/row['result']
            result = checked_json(result_path, row['result_sha256'])
            if 'cohort' in result and 'environment' in result:
                return result
    raise ValueError('No original timing cohort identity for '+scope)


def main(argv=None):
    stage_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['tune','measure'])
    for name in ('manifest','output','science-seal','primary-tuning','primary-measurement',
                 'source-root','supplement-seal','supplement-binding'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--supplement-seal-sha256', required=True)
    parser.add_argument('--supplement-binding-sha256', required=True)
    parser.add_argument('--tuning', type=Path, help='Separate tuning-seal.json for measure')
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('BLS_EXECUTION_PROTOCOL.md'))
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.deadline_epoch-time.time() <= 3600:
        parser.error('Shared absolute deadline must be within one hour')
    if args.hourly_usd < 0 or not np.isfinite(args.hourly_usd):
        parser.error('Hourly cost must be finite and nonnegative')
    if args.stage == 'measure' and (not args.tuning or not args.primary_measurement):
        parser.error('measure requires --tuning and --primary-measurement')
    for key,value in vars(args).items():
        if isinstance(value, Path):
            setattr(args,key,value.resolve())
    if args.output.exists():
        parser.error('Fresh output directory required; no resume or retry of existing stages')
    args.authorization_identity = verify_authorization(args)
    original = checked_json(args.primary_tuning)
    if original['status'] != 'complete' or original['stage'] != 'tune':
        raise ValueError('Completed original tuning required')
    science_sha = native.sha(args.science_seal)
    if original['science_seal_sha256'] != science_sha:
        raise ValueError('Original tuning science identity changed')
    failure = next(r for r in original['configs'] if r['id']=='bls-mixed-w1-b1')
    if failure['eligible']:
        raise ValueError('Supplement requires the retained original BLS exclusion')
    checked_json(args.primary_tuning.parent/failure['result'], failure['result_sha256'])
    identity = source_identity(args.protocol)
    reference = primary_panel(args.primary_tuning, original, 'mixed')
    if args.hourly_usd != reference['config']['hourly_usd']:
        raise ValueError('Rental price differs from frozen primary timing receipt')
    expected_allocation = allocation(reference['environment'])
    campaign = dict(schema_version=1, stage=args.stage, status='running', source_identity=identity,
                    science_seal_sha256=science_sha, primary_tuning_sha256=native.sha(args.primary_tuning),
                    original_bls_exclusion_sha256=failure['result_sha256'],
                    original_qualification_passed=False,
                    original_numerical_qualification_passed=False, manifest_sha256=native.sha(args.manifest),
                    deadline_epoch=args.deadline_epoch, hourly_usd=args.hourly_usd,
                    allocation=list(expected_allocation), configs=[], unavailable=[], selected=None)
    campaign.update(args.authorization_identity)
    args.output.mkdir(parents=True)
    campaign['stage_preparation_seconds'] = time.perf_counter()-stage_started

    def interrupted(signum, frame):
        raise Interrupted('Signal '+str(signum)+'; preserve partial evidence and clean owned workers')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01,args.deadline_epoch-time.time()-CLEANUP_RESERVE_SECONDS))

    def run(scope, manifest, names, expected_cohort, workers, batch, anchors=None,
            reps=1, min_attempts=24, seconds=30, prefix=''):
        identifier = f'{prefix}bls-execution-{scope}-w{workers}-b{batch}'
        directory = args.output/identifier
        result = run_configuration(args, manifest, names, scope, workers, batch, directory,
                                   expected_allocation, expected_cohort, anchors, reps,min_attempts,seconds)
        campaign['configs'].append(dict(id=identifier, scope=scope, workers=workers, batch_size=batch,
            reference_only=bool(prefix), result=str(directory.relative_to(args.output)/'result.json'),
            result_sha256=native.sha(directory/'result.json'),
            execution_rates_valid=result['execution_rates_valid'], failure_reason=result.get('error')))
        native.write(args.output/'campaign.json', campaign)
        if result['status'] == 'partial':
            raise Interrupted('Configuration interrupted; do not select partial-stage results')
        return result

    try:
        native.write(args.output/'campaign.json', campaign)
        if args.stage == 'tune':
            if campaign['manifest_sha256'] != original['manifest_sha256']:
                raise ValueError('Development manifest differs from original tuning')
            names = original['names']
            expected_cohort = reference['cohort']
            records = [run('mixed',args.manifest,names,expected_cohort,1,1)]
            anchors = records[0].get('fixed_reference_anchors', {})
            for workers in (2,4):
                records.append(run('mixed',args.manifest,names,expected_cohort,workers,1,anchors))
            chosen = execution_winner(records)
            if chosen is not None:
                campaign['worker_stage_selection'] = chosen['workers']
                native.write(args.output/'campaign.json', campaign)
                for batch in (4,8):
                    records.append(run('mixed',args.manifest,names,expected_cohort,chosen['workers'],batch,anchors))
                chosen = execution_winner([r for r in records if r['workers']==campaign['worker_stage_selection']])
                campaign['selected'] = dict(workers=chosen['workers'],batch_size=chosen['batch_size'])
            else:
                campaign['unavailable'].append(dict(scope='mixed',reason='No positive valid execution rate'))
        else:
            sealed = checked_json(args.tuning)
            tuning = checked_json(Path(sealed['campaign_path']),sealed['campaign_sha256'])
            for key in (*args.authorization_identity,'source_identity','allocation','deadline_epoch','hourly_usd'):
                if sealed[key] != campaign[key] or sealed[key] != tuning[key]:
                    raise ValueError('Supplement tuning identity changed: '+key)
            for name,digest in tuning['artifact_sha256'].items():
                if native.sha(Path(sealed['campaign_path']).parent/name) != digest:
                    raise ValueError('Supplement tuning output changed: '+name)
            if tuning['status'] != 'complete' or tuning['selected'] != sealed['selected']:
                raise ValueError('Incomplete or changed separate tuning selection')
            campaign.update(tuning_seal_sha256=native.sha(args.tuning),tuning_seal_path=str(args.tuning),
                            selected=sealed['selected'])
            measurement = checked_json(args.primary_measurement)
            if measurement['status'] != 'complete' or measurement['stage'] != 'measure':
                raise ValueError('Completed original measurement required')
            if measurement['science_seal_sha256'] != science_sha or measurement['manifest_sha256'] != campaign['manifest_sha256']:
                raise ValueError('Original measurement science or input identity changed')
            campaign['primary_measurement_sha256'] = native.sha(args.primary_measurement)
            for scope in SCOPES:
                if campaign['selected'] is None:
                    campaign['unavailable'].append(dict(scope=scope,reason='No valid execution tuning selection'))
                    continue
                prior = primary_panel(args.primary_measurement,measurement,scope)
                if allocation(prior['environment']) != expected_allocation:
                    raise ValueError('Original panel allocation differs from tuning')
                if prior['config']['hourly_usd'] != args.hourly_usd:
                    raise ValueError('Original panel rental price differs from tuning')
                manifest = Path(measurement['varied_manifest']) if scope=='varied' else args.manifest
                if scope=='varied' and native.sha(manifest) != measurement['varied_manifest_sha256']:
                    raise ValueError('Original varied manifest changed')
                names = [v['name'] for v in prior['cohort']]
                chosen = campaign['selected']
                anchors = None
                if chosen['workers'] > 1:
                    ref = run(scope,manifest,names,prior['cohort'],1,1,reps=1,min_attempts=1,seconds=0,prefix='reference-')
                    if ref['status'] != 'complete':
                        campaign['unavailable'].append(dict(scope=scope,reason='Reference instrumentation failed'))
                        continue
                    anchors = ref['fixed_reference_anchors']
                measured = run(scope,manifest,names,prior['cohort'],chosen['workers'],chosen['batch_size'],anchors,
                               reps=3,min_attempts=96,seconds=120)
                if not measured['execution_rates_valid']:
                    campaign['unavailable'].append(dict(scope=scope,reason=measured.get('error',
                                                        'Execution ownership/accounting invalid')))
        campaign['status'] = 'complete'
    except BaseException as error:
        campaign.update(status='partial' if isinstance(error, Interrupted) else 'error',error=traceback.format_exc())
        if campaign['status'] != 'complete':
            campaign['selected'] = None
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        campaign['total_stage_seconds'] = time.perf_counter()-stage_started
        campaign['estimated_stage_compute_usd'] = args.hourly_usd*campaign['total_stage_seconds']/3600
        try:
            campaign['gpu_empty_after'] = native.exclusive_gpu_processes([])
        except BaseException:
            campaign['gpu_empty_after'] = dict(exclusive=False,error=traceback.format_exc())
        if not campaign['gpu_empty_after']['exclusive']:
            campaign['status']='error'
            campaign['selected']=None
        campaign['artifact_sha256'] = {str(p.relative_to(args.output)):native.sha(p)
            for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ('campaign.json','tuning-seal.json')}
        native.write(args.output/'campaign.json',campaign)
    if args.stage=='tune' and campaign['status']=='complete':
        native.write(args.output/'tuning-seal.json',dict(schema_version=1,
            original_qualification_passed=False,
            campaign_path=str(args.output/'campaign.json'),campaign_sha256=native.sha(args.output/'campaign.json'),
            **{k:campaign[k] for k in ('science_seal_sha256','source_identity','primary_tuning_sha256',
                                      'primary_measurement_sha256','supplement_seal_sha256',
                                      'supplement_binding_sha256','auxiliary_plan_sha256','deadline_epoch',
                                      'hourly_usd','allocation','selected','manifest_sha256','original_bls_exclusion_sha256')}))
    print(json.dumps(dict(status=campaign['status'], selected=campaign['selected'],
                         configurations=len(campaign['configs']))),flush=True)
    return 0 if campaign['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
