#!/usr/bin/env python3
"""Assess complete comparisons separately from a failed full timing campaign.

This is an explicitly post hoc reporting assessment. It never edits or relabels
original campaign acceptance, records, or normalization. The only permitted
campaign incompleteness is a disclosed optional native pool's warmup OOM.
Every included configuration retains the original numerical, source, cohort,
repetition, and process-ownership requirements, with full-object repeatability
checked additionally. All medians and pool choices are recomputed from records.
No GPU, cloud, or benchmark execution is performed by this program.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REGIMES = ('tess_solar', 'tess_gap', 'ztf_solar')
CANDIDATE_FILES = {'tls.py', 'tls_reference.py', 'tls_reference_math.py',
    'tls_reference_frontend.py', 'tls_reference_prefix.py', 'tls_grids.py',
    'tls_stats.py', 'tls_models.py', 'kernels/tls_reference.cu',
    'kernels/tls_reference_prepare.cu'}
ALLOWED_OMISSION_REASONS = {
    'Configuration did not complete successfully', 'Warmup did not complete',
    'Batch repetition count is not three'}
REPORTING_SCOPE = 'post_hoc_complete_configurations_after_optional_native_warmup_oom'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(Path(path).parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def frozen_modules(source_root):
    """Import byte-verified source extracted from the executed archive."""
    source_root = Path(source_root)
    manifest = read(source_root/'execution-sources.json')
    archive = source_root/manifest['archive']
    demand(sha(archive) == manifest['archive_sha256'], 'Execution archive hash mismatch')
    with tempfile.TemporaryDirectory(prefix='tls-frozen-report-') as tmp:
        tmp = Path(tmp)
        with tarfile.open(archive) as stream:
            members = [member for member in stream if member.isfile()]
            demand(set(member.name for member in members) == set(manifest['files']),
                   'Execution archive file inventory differs')
            for member in members:
                path = Path(member.name)
                demand(not path.is_absolute() and '..' not in path.parts,
                       'Unsafe execution source member')
                data = stream.extractfile(member).read()
                demand(hashlib.sha256(data).hexdigest() == manifest['files'][member.name],
                       'Execution source member hash differs: ' + member.name)
                target = tmp/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        tools = tmp/'latency/tools/timing'
        package = '_tls_reporting_frozen'
        for name in list(sys.modules):
            if name == package or name.startswith(package+'.'):
                del sys.modules[name]
        load_module(package, tools/'__init__.py', package=True)
        benchmark = load_module(package+'.benchmark', tools/'benchmark.py')
        cohort = sys.modules[package+'.cohort']
        common = sys.modules[package+'.common']
        summarize = load_module(package+'.summarize', tools/'summarize.py')
        analyze = load_module(package+'.analyzer', tools.parent/'analyze_timing.py')
        try:
            yield dict(root=tmp, benchmark=benchmark, cohort=cohort, common=common,
                       summarize=summarize, analyze=analyze, manifest=manifest)
        finally:
            for name in list(sys.modules):
                if name == package or name.startswith(package+'.'):
                    del sys.modules[name]


def verify_collection(checkpoint):
    """A failed pipeline may still have complete, verified collection."""
    checkpoint = Path(checkpoint)
    outcome = read(checkpoint/'outcome.json')
    demand(outcome.get('final_compact_verified') is True and
           outcome.get('missing_final_paths') == [],
           'Final critical evidence was not completely collected')
    termination = read(checkpoint/'termination.json')
    demand(termination.get('verified') is True, 'Allocation termination is unverified')
    index = read(checkpoint/'file-index.json')
    prefix = 'timing-continuation/results/'
    selected = {name: entry for name, entry in index.items() if name.startswith(prefix)}
    demand(bool(selected), 'No collected timing campaign')
    for name, entry in selected.items():
        path = checkpoint/'files'/name
        demand(path.is_file() and path.stat().st_size == entry['size'] and
               sha(path) == entry['sha256'], 'Collected file mismatch: ' + name)
    root = checkpoint/'files'/prefix
    disk = {str(path.relative_to(checkpoint/'files')) for path in root.rglob('*') if path.is_file()}
    demand(disk == set(selected), 'Unindexed or missing collected campaign files')
    return root, dict(outcome_sha256=sha(checkpoint/'outcome.json'),
        termination_sha256=sha(checkpoint/'termination.json'),
        file_index_sha256=sha(checkpoint/'file-index.json'),
        verified_files={name: dict(sha256=item['sha256'], size=item['size'])
                        for name, item in sorted(selected.items())},
        collection_scope=outcome.get('completeness_scope'),
        original_pipeline_exit_code=outcome.get('exit_code'))


def check_driver_terminal(timing, prepared):
    terminal, status = read(timing/'terminal.json'), read(timing/'status.json')
    expected = ['preflight', 'components_candidate', 'components_gtls']
    if any(item['correction_timing']['required'] for item in prepared['selections'].values()):
        expected.append('components_gtls_corrected')
    expected += ['public', 'summarize', 'normalize']
    stages = status.get('stages', [])
    demand(terminal.get('status') == 'error' and type(terminal.get('exit_code')) is int and
           terminal['exit_code'] == 2 and
           'ValueError: Final timing acceptance failed: ' in terminal.get('error', ''),
           'Original driver did not fail specifically at its final audit')
    demand(status.get('status') == 'error' and [item.get('name') for item in stages] == expected and
           all(item.get('status') == 'complete' and type(item.get('exit_code')) is int and
               item['exit_code'] == 0 for item in stages),
           'An earlier original stage is missing or failed')
    plan = read(timing/'pipeline-plan.json')
    demand([item['name'] for item in plan['stages']] == expected and
           all(actual['command'] == declared['command'] and
               actual['timeout_seconds'] == declared['timeout_seconds']
               for actual, declared in zip(stages, plan['stages'])),
           'Executed stage inventory differs from its original pipeline plan')
    return terminal


def replay_original_audit(timing, frozen):
    """Run the exact original audit against read-only symlinks in a new folder."""
    with tempfile.TemporaryDirectory(prefix='tls-original-audit-') as folder:
        folder = Path(folder)
        for path in timing.iterdir():
            if path.name != 'acceptance.json':
                (folder/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        code = """import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import run_timing
try:
    run_timing.audit(SimpleNamespace(output=Path(sys.argv[2])))
except ValueError as error:
    if not str(error).startswith('Final timing acceptance failed: '):
        raise
    sys.exit(3)
sys.exit(4)
"""
        run = subprocess.run([sys.executable, '-c', code,
            str(frozen['root']/'latency'), str(folder)], capture_output=True, text=True, timeout=60)
        demand(run.returncode == 3 and (folder/'acceptance.json').is_file(),
               'Exact original audit did not reproduce its rejection: ' + run.stderr)
        replay, original = read(folder/'acceptance.json'), read(timing/'acceptance.json')
        demand({k:v for k,v in replay.items() if k != 'created_utc'} ==
               {k:v for k,v in original.items() if k != 'created_utc'},
               'Original audit reasons or recorded inputs could not be reproduced')
        return replay


def verify_worker(backend, worker, selection, cohort):
    actual = worker['sources']
    expected_names = (set(selection['expected_native_sources']) if backend != 'candidate'
                      else CANDIDATE_FILES)
    demand(set(actual['files']) == expected_names, 'Worker source inventory is incomplete')
    cohort.verify_worker_sources(backend, actual, selection)


def verify_output(output, case):
    digest = hashlib.sha256(json.dumps(output['fields'], sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    demand(output['full_digest'] == digest, 'Returned-object digest does not match retained fields')
    demand(output['case'] == case['name'], 'Output names differ from the execution cohort')
    # Full strict array/mask/primary/SDE hashes are checked against the frozen
    # result by consistency(). Scalar metadata still must describe that spectrum.
    demand(output['nperiods'] > 0 and
           math.isfinite(output['primary_period']) and math.isfinite(output['SDE']),
           'Output has no finite complete search')


def measurement_problems(record, names, single_name, width, benchmark):
    problems = []
    if record.get('status') != 'ok': problems.append('Configuration did not complete successfully')
    if record.get('pool_width') != width: problems.append('Worker count differs from declaration')
    if record.get('warmup', {}).get('status') != 'ok': problems.append('Warmup did not complete')
    if len(record.get('batch', [])) != 3: problems.append('Batch repetition count is not three')
    if len(record.get('single', [])) != (5 if width == 1 else 0):
        problems.append('Single repetition count differs from declaration')
    if not benchmark.ownership_valid(record): problems.append('GPU ownership lifecycle failed')
    for kind in ('single', 'batch'):
        expected = Counter([single_name] if kind == 'single' else names)
        for rep in record.get(kind, []):
            seconds = rep.get('denominator_seconds')
            if (rep.get('status') != 'ok' or rep.get('errors') or
                    not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0 or
                    seconds != rep.get('elapsed_seconds') or rep.get('source_count') != sum(expected.values()) or
                    Counter(item['case'] for item in rep.get('outputs', [])) != expected):
                problems.append('Incomplete or invalid public repetition')
    return problems


def optional_warmup_oom(record, backend, width, problems):
    """The post hoc scope permits this resource failure, never a partial time."""
    errors = record.get('warmup', {}).get('errors', [])
    api_errors = [item for item in errors if 'traceback' in item]
    membership = [item for item in errors if 'traceback' not in item]
    warmup_outputs = record.get('warmup', {}).get('outputs', [])
    names = record.get('config', {}).get('names', [])
    expected = [names[0]] * width if names else []
    observed = [item['case'] for item in warmup_outputs]
    accounting = (len(membership) == 1 and
        membership[0].get('reason') == 'Post-barrier result membership differs from the assigned inputs' and
        membership[0].get('expected') == expected and membership[0].get('observed') == observed and
        len(observed) + len(api_errors) == width and
        Counter(observed + [item.get('case') for item in api_errors]) == Counter(expected))
    return (backend == 'gtls' and width in (2, 4) and record.get('status') == 'error' and
            set(problems) == ALLOWED_OMISSION_REASONS and
            not record.get('single') and not record.get('batch') and
            record.get('warmup', {}).get('status') == 'error' and
            record['warmup'].get('denominator_seconds') is None and bool(api_errors) and accounting and
            all('cupy.cuda.memory.OutOfMemoryError:' in item.get('traceback', '') for item in api_errors))


def audit_config(record, backend, width, regime, selection, plan, frozen):
    names = selection['selected_cases']
    config = record['config']
    demand(config.get('backend') == backend and config.get('prefix') == 'graph' and
           config.get('regime') == regime and config.get('measurement_scope') == 'full' and
           config.get('manifest') == plan['manifest'] and config.get('names') == names and
           record.get('pool_width') == width, 'A configuration changed its cohort, backend, or width')
    demand(record.get('status') in ('ok', 'error', 'measurement_failure', 'ownership_failure'),
           'A requested configuration is not terminal')
    workers = record.get('workers', [])
    demand(len(workers) == width and len({item['pid'] for item in workers}) == width,
           'A requested configuration lacks its complete worker/source inventory')
    demand(workers == record['gpu_ownership']['workers'], 'Source worker and ownership inventory differ')
    for worker in workers:
        verify_worker(backend, worker, selection, frozen['cohort'])
    # Ownership applies to failed configurations as well as selected timings.
    demand(frozen['benchmark'].ownership_valid(record), 'Requested configuration ownership failed')
    case_by_name = {item['name']: item for item in plan['cases'][regime]}
    expected = plan['frozen_outputs'][regime][backend]
    for rep in [record.get('warmup', {})] + record.get('single', []) + record.get('batch', []):
        for output in rep.get('outputs', []):
            demand(output['case'] in case_by_name, 'Unexpected source in a returned output')
            verify_output(output, case_by_name[output['case']])
    problems = measurement_problems(record, names, selection['single_case'], width, frozen['benchmark'])
    excluded_failure = bool(problems)
    if excluded_failure:
        demand(optional_warmup_oom(record, backend, width, problems),
               'Failure is outside the narrowly declared optional native warmup-OOM scope')
        return dict(status=record['status'], complete=False, eligible=False,
            exclusion='warmup_gpu_out_of_memory', problems=problems,
            completed_single_repetitions=0, completed_batch_repetitions=0,
            warmup_failure=record['warmup']['errors'], failed_times_are_speed_denominators=False)
    gates = {}
    for kind, selected in (('batch', names), ('single', [selection['single_case']])):
        if kind == 'single' and width != 1: continue
        ref = {name: expected[name]['strict'] for name in selected}
        gates[kind+'_frozen'] = frozen['summarize'].gate(record, kind, selected, reference=ref)
        gates[kind+'_full_repeatability'] = frozen['summarize'].gate(record, kind, selected, field='full_digest')
    if width == 1:
        ref = gates['batch_full_repeatability']['reference']
        gates['single_batch_full_identity'] = frozen['summarize'].gate(record, 'single',
            [selection['single_case']], reference={selection['single_case']:ref[selection['single_case']]},
            field='full_digest')
    eligible = all(item['eligible'] for item in gates.values())
    demand(eligible or (backend == 'gtls' and width in (2, 4)),
           'Mandatory candidate/native-one configuration failed a raw output gate')
    return dict(status=record['status'], complete=True, eligible=eligible,
        exclusion=None if eligible else 'complete_but_ineligible_output',
        gates={name:dict(eligible=value['eligible'], problems=value['problems']) for name,value in gates.items()},
        completed_single_repetitions=len(record['single']), completed_batch_repetitions=len(record['batch']),
        failed_times_are_speed_denominators=False)


def verify_audit_reasons(acceptance, configurations):
    expected = []
    omissions = []
    for regime, records in configurations.items():
        for label, result in records.items():
            if result.get('exclusion') == 'warmup_gpu_out_of_memory':
                omissions.append(regime+'/'+label)
                expected.extend(regime+': '+label+': '+reason for reason in result['problems'])
    actual = acceptance.get('publication_gate', {}).get('problems')
    demand(acceptance.get('status') == 'rejected' and
           acceptance.get('publication_gate', {}).get('pass') is False and bool(omissions) and
           Counter(actual or []) == Counter(expected),
           'Original rejection includes failures beyond optional native warmup OOM completeness')
    return omissions


def assess(checkpoint, source_root, manifest_path):
    campaign, collection = verify_collection(checkpoint)
    timing = campaign/'timing'
    prepared, plan = read(timing/'prepared.json'), read(timing/'public/plan.json')
    terminal = check_driver_terminal(timing, prepared)
    manifest = read(manifest_path)
    demand(sha(manifest_path) == prepared['manifest_sha256'] == plan['manifest_sha256'],
           'Timing input manifest changed')
    with frozen_modules(source_root) as frozen:
        source_inventory = frozen['manifest']['files']
        expected_tools = {Path(name).name:digest for name,digest in source_inventory.items()
                          if name.startswith('latency/tools/timing/') and name.endswith('.py')}
        demand(prepared['timing_sources'] == expected_tools == plan['sources'],
               'Executed timing sources differ from the frozen archive')
        demand(read(timing/'pipeline-plan.json')['driver_sha256'] == source_inventory['latency/run_timing.py'],
               'Executed driver differs from frozen source')
        demand(plan['cohort_selection'] == prepared['selections'] and
               plan['frozen_outputs'] == prepared['frozen_outputs'], 'Cohort/frozen outputs changed')
        demand(plan['regimes'] == list(REGIMES) and plan['native_pool_widths'] == [1,2,4] and
               plan['measurement_scope'] == 'full' and plan['single_repetitions'] == 5 and
               plan['batch_repetitions'] == 3 and plan['cpu_math_threads_per_worker'] == 1,
               'Original timing declaration changed')
        accepted = frozen['cohort'].accepted_study(manifest_path, Path(manifest_path).parent/'main')
        configurations, expected_paths = {}, set()
        public_summary = read(timing/'public/summary.json')
        demand(public_summary.get('status') == 'complete' and
               public_summary.get('cohort_selection') == prepared['selections'],
               'Original public runner did not account for the complete cohort')
        for regime in REGIMES:
            selection = prepared['selections'][regime]
            demand(selection['accepted_study'] == accepted, 'Accepted numerical origins differ')
            cases = frozen['common'].load_cases(manifest_path, regime, selection['selected_cases'])
            recomputed_selection = frozen['cohort'].select(manifest_path, Path(manifest_path).parent/'main', regime)
            demand(recomputed_selection == selection, 'Original deterministic source selection cannot be reproduced')
            demand([frozen['common'].case_identity(case) for case in cases] == plan['cases'][regime],
                   'Actual execution input arrays or options differ from the original plan')
            backends = ('gtls', 'candidate', 'gtls_corrected') if selection['correction_timing']['required'] else ('gtls', 'candidate')
            expected = frozen['cohort'].frozen_outputs(Path(manifest_path).parent/'main', cases,
                backends=backends, manifest_path=manifest_path)
            demand(expected == plan['frozen_outputs'][regime], 'Original frozen output bank differs')
            required = [('candidate',1), ('gtls',1), ('gtls',2), ('gtls',4)]
            if selection['correction_timing']['required']: required.append(('gtls_corrected',1))
            details = configurations[regime] = {}
            demand(selection['actual_batch_size'] == len(cases) == 16 and
                   len({item['name'] for item in cases}) == 16 and selection['single_case'] in selection['selected_cases'],
                   'The declared fixed 16-source cohort changed')
            for backend,width in required:
                label = f'{backend}_graph_{width}worker'
                path = timing/'public'/regime/label/'record.json'
                expected_paths.add(path.resolve())
                record = read(path)
                detail = audit_config(record, backend, width, regime, selection, plan, frozen)
                detail.update(record=str(path.relative_to(timing)), record_sha256=sha(path),
                              backend=backend, workers=width)
                details[label] = detail
                logged = public_summary['regimes'][regime][label]
                demand(logged['status'] == record['status'] and logged['record'] == str(path.relative_to(timing/'public')) and
                       logged['single_seconds'] == [item['denominator_seconds'] for item in record['single']] and
                       logged['batch_seconds'] == [item['denominator_seconds'] for item in record['batch']],
                       'Original public configuration accounting differs from its raw record')
            demand(set(public_summary['regimes'][regime]) == set(details), 'Public configuration inventory differs')
        actual_paths = {path.resolve() for path in (timing/'public').glob('*/*/record.json')}
        demand(actual_paths == expected_paths and set(public_summary['regimes']) == set(REGIMES),
               'Missing or extra attempted configurations')
        corrected = timing/'components/gtls_corrected' if any(
            value['correction_timing']['required'] for value in prepared['selections'].values()) else None
        component_backends = ['gtls','candidate'] + (['gtls_corrected'] if corrected else [])
        for backend in component_backends:
            component_summary = read(timing/'components'/backend/'summary.json')
            demand(component_summary.get('status') == 'complete' and component_summary.get('backend') == backend,
                   'Component stage is not complete')
            for regime in REGIMES:
                if backend == 'gtls_corrected' and not prepared['selections'][regime]['correction_timing']['required']: continue
                record = read(timing/'components'/backend/(regime+'.json'))
                selection = prepared['selections'][regime]
                demand(record['gpu_ownership'] == component_summary['gpu_ownership'], 'Component ownership receipt differs')
                demand(len(record['gpu_ownership']['workers']) == 1, 'Component worker cardinality changed')
                verify_worker(backend, record['gpu_ownership']['workers'][0], selection, frozen['cohort'])
                demand(record['frozen_outputs'] == plan['frozen_outputs'][regime][backend][selection['single_case']],
                       'Component frozen expectation changed')
        checks = frozen['summarize'].summarize(timing/'public', timing/'components/gtls',
            timing/'components/candidate', corrected)
        demand(checks == read(timing/'summary.json'), 'Raw summary cannot be recomputed from original records')
        for regime in REGIMES:
            for key in ('public_single','public_batch','common_search_components'):
                demand(checks['regimes'][regime][key]['eligible'] is True, 'A mandatory original comparison gate failed')
            # Supplement the original strict eligibility with full-object stability.
            pools = checks['regimes'][regime]['public_batch']['native_pool_configurations']
            for width, pool in pools.items():
                detail = configurations[regime][f'gtls_graph_{width}worker']
                pool['eligible'] = pool['eligible'] and detail['eligible']
            eligible = [(item['elapsed']['median_seconds'],int(width))
                        for width,item in pools.items() if item['eligible']]
            demand(bool(eligible), 'No complete eligible native pool')
            selected_width = min(eligible)[1]
            batch = checks['regimes'][regime]['public_batch']
            batch['strongest_tested_native_workers'] = selected_width
            batch['strongest_tested_native'] = pools[str(selected_width)]['elapsed']
            batch['speedup'] = frozen['summarize'].ratio(batch['strongest_tested_native'],batch['candidate'])
        normalized = frozen['analyze'].analyze(checks, manifest,
            dict(merged_origin_checks=accepted), sha(manifest_path))
        original_normalized = read(timing/'timing_analysis.json')
        # Before supplementary full-object filtering, exact normalization must
        # reproduce the frozen raw summary. A new selected pool is possible only
        # when an original optional pool fails the added full-object gate.
        original_replay = frozen['analyze'].analyze(read(timing/'summary.json'), manifest,
            dict(merged_origin_checks=accepted), sha(manifest_path))
        demand(original_replay == {k:v for k,v in original_normalized.items() if k != 'sources'},
               'Original normalized output cannot be reproduced')
        demand(original_normalized['sources']['timing_checks']['sha256'] == sha(timing/'summary.json') and
               original_normalized['sources']['inputs']['sha256'] == sha(manifest_path) and
               original_normalized['sources']['numerical_acceptance']['sha256'] in
                   {item['receipt_sha256'] for item in accepted['accepted_studies'].values()},
               'Original normalized source hashes differ')
        rejection = replay_original_audit(timing, frozen)
        omissions = verify_audit_reasons(rejection, configurations)
        normalized.update(campaign_pass=False, reporting_scope=REPORTING_SCOPE,
            reporting_scope_note='Post hoc assessment after an optional native pool warmup OOM; '
                'the original all-configuration campaign acceptance remains failed. '
                'Only complete configurations enter timing denominators, with all attempts disclosed.',
            excluded_configurations=[regime+'/'+label for regime, rows in configurations.items()
                                     for label, item in rows.items() if not item['eligible']],
            configuration_outcomes=configurations,
            sources=original_normalized['sources'])
        normalized['verification'].update(campaign_pass=False,
            reporting_scope=REPORTING_SCOPE, complete=True,
            full_returned_object_repeatability=True,
            scope='Separate post hoc complete-configuration reporting gate; original full campaign failed. '
                'Original numerical/source/cohort/exclusivity/repetition gates are unchanged.')
        receipt = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
            reporting_scope=REPORTING_SCOPE, campaign_pass=False,
            original_campaign_acceptance=dict(file='timing/acceptance.json',
                sha256=sha(timing/'acceptance.json'), passed=False,
                publication_gate=rejection['publication_gate']),
            original_terminal_sha256=sha(timing/'terminal.json'),
            original_summary_sha256=sha(timing/'summary.json'),
            original_normalization_sha256=sha(timing/'timing_analysis.json'),
            exact_original_audit_replayed=True, all_attempts_accounted=True,
            original_numerical_rules_changed=False, measurements_rerun=False,
            failed_times_are_speed_denominators=False,
            omitted_optional_configurations=omissions, configurations=configurations,
            collection=collection, execution_source_manifest=frozen['manifest'],
            assessment_source_sha256=sha(__file__),
            scope_note='This gate authorizes only a report of complete comparisons, not the original campaign. '
                'The report scope was chosen after observing the warmup OOM; pool ranking uses the original '
                'fastest-eligible rule, additionally requiring stable complete returned-object digests.')
        receipt['reporting_gate']={'pass':True,'problems':[]}
        return normalized, receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','sources','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    demand(not args.output.exists(), 'Use a new post hoc reporting directory')
    normalized, receipt=assess(args.checkpoint,args.sources,args.manifest)
    args.output.mkdir(parents=True)
    write(args.output/'timing_analysis.json',normalized)
    receipt['timing_analysis_sha256']=sha(args.output/'timing_analysis.json')
    write(args.output/'reporting_acceptance.json',receipt)
    print(json.dumps(dict(reporting_gate=receipt['reporting_gate'],campaign_pass=False,
        timing_analysis_sha256=receipt['timing_analysis_sha256'],
        reporting_acceptance_sha256=sha(args.output/'reporting_acceptance.json'))))


if __name__=='__main__':
    main()
