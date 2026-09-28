"""CPU checks for benchmark accounting and output gates; no CUDA imports."""
import copy
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import benchmark
from .benchmark import consistency
from .cohort import accepted_study, frozen_outputs, select, verify_worker_sources
from .common import array_hash, fingerprint, masked_hash, selected_names, sha
from .components import Components
from .summarize import compare_components, compare_public, compare_single_public, distribution


def output(name='a', digest='same'):
    return dict(case=name, strict={'periods': digest}, common={'periods': digest},
                full_digest=digest)


def exclusive_snapshot(pids=(101,)):
    return dict(allowed_pids=list(pids), observed_pids=list(pids),
                foreign_pids=[], missing_pids=[], exclusive=True)


def ownership_receipt():
    return dict(version=1, status='complete', passed=True, host_pids=[101], identity_scope='individual',
        before_start=exclusive_snapshot(()), after_start=exclusive_snapshot(),
        after_exit=exclusive_snapshot(()),
        owned_worker_pids=[11], live_workers_at_binding=[11],
        workers=[dict(pid=11, namespace_pids=[11], cuda_context_allocation_bytes=1,
                      cuda_context_synchronized=True)],
        bindings=[dict(worker_pid=11, namespace_pids=[11], host_pid=101,
                       method='single_worker_lifecycle')],
        worker_exits=[dict(worker_pid=11, exit_code=0, alive=False, forced=False)])


def repetition(names=('a', 'b'), seconds=2.):
    return dict(status='ok', outputs=[output(name) for name in names],
                source_count=len(names), denominator_seconds=seconds,
                exclusive_before=exclusive_snapshot(), exclusive_after=exclusive_snapshot())


def configuration(names=('a', 'b'), seconds=2.):
    return dict(status='ok', gpu_ownership=ownership_receipt(), batch=[repetition(names, seconds) for _ in range(3)],
                single=[repetition(names[:1], seconds/2) for _ in range(5)])


def single_configuration(seconds=1.):
    return dict(status='ok', pool_width=1, batch=[], gpu_ownership=ownership_receipt(),
                single=[repetition(('a',), seconds) for _ in range(5)])


def test_single_scope_has_no_batch_or_strongest_pool_result():
    records = {'candidate_graph_1worker': single_configuration(),
               'gtls_graph_1worker': single_configuration(2.)}
    frozen = {name: {'a': {'strict': {'periods': 'same'}}} for name in ('candidate', 'gtls')}
    result = compare_single_public(records, 'a', frozen)
    assert result['public_single']['eligible']
    assert result['public_single']['speedup'] == 2
    assert result['public_batch']['status'] == 'not_measured'
    assert result['public_batch']['source_count'] == 0
    assert 'strongest_tested_native_workers' not in result['public_batch']


@pytest.mark.parametrize('change', ('missing_rep', 'wrong_case', 'changed', 'full_output', 'failed', 'batch', 'pool'))
def test_single_scope_rejects_incomplete_or_different_results(change):
    records = {'candidate_graph_1worker': single_configuration(),
               'gtls_graph_1worker': single_configuration(2.)}
    record = records['candidate_graph_1worker']
    if change == 'missing_rep':
        record['single'].pop()
    elif change == 'wrong_case':
        record['single'][0]['outputs'][0]['case'] = 'another'
    elif change == 'changed':
        record['single'][0]['outputs'][0]['strict'] = {'periods': 'changed'}
    elif change == 'full_output':
        record['single'][0]['outputs'][0]['full_digest'] = 'extra-field-changed'
    elif change == 'failed':
        record['single'][0].update(status='error', denominator_seconds=None)
    elif change == 'batch':
        record['batch'] = [repetition()]
    else:
        record['pool_width'] = 2
    frozen = {name: {'a': {'strict': {'periods': 'same'}}} for name in ('candidate', 'gtls')}
    result = compare_single_public(records, 'a', frozen)
    assert not result['public_single']['eligible']
    assert result['public_single']['speedup'] is None


def test_single_corrected_comparison_is_separate_and_requires_matching_outputs():
    records = {name+'_graph_1worker': single_configuration(seconds) for name, seconds in
               (('candidate', 1), ('gtls', 2), ('gtls_corrected', 1.5))}
    frozen = {name: {'a': {'strict': {'periods': 'same'}}} for name in ('candidate', 'gtls', 'gtls_corrected')}
    result = compare_single_public(records, 'a', frozen)
    assert result['public_single']['speedup'] == 2
    assert result['corrected_native_crosscheck']['single']['speedup'] == 1.5
    records['gtls_corrected_graph_1worker']['single'][0]['outputs'][0]['common'] = {'periods': 'changed'}
    assert not compare_single_public(records, 'a', frozen)['corrected_native_crosscheck']['single']['eligible']


@pytest.mark.parametrize('fail', (False, True))
def test_single_runner_executes_only_five_public_single_calls_and_halts_on_mismatch(tmp_path, monkeypatch, fail):
    import contextlib
    calls = []
    selection = dict(single_case='a', selected_cases=['a', 'b'], actual_batch_size=2,
                     correction_timing=dict(required=False))
    monkeypatch.setattr(benchmark, 'select', lambda *args: copy.deepcopy(selection))
    monkeypatch.setattr(benchmark, 'load_cases', lambda manifest, regime, names:
                        [dict(name=name) for name in names])
    monkeypatch.setattr(benchmark, 'case_identity', lambda case: case)
    monkeypatch.setattr(benchmark, 'frozen_outputs', lambda *args, **kwargs:
        {name: {'a': {'strict': {'periods': 'same'}}} for name in ('candidate', 'gtls')})
    monkeypatch.setattr(benchmark, 'verify_worker_sources', lambda *args: None)
    monkeypatch.setattr(benchmark, 'environment', lambda: {})
    monkeypatch.setattr(benchmark, 'Monitor', lambda *args: contextlib.nullcontext())
    class Pool:
        def __init__(self, config, width, timeout):
            assert config['names'] == ['a'] and width == 1
            self.backend = config['backend']
            self.ready = [dict(sources={})]
            self.startup_seconds = .01
        def measure(self, indices, *, single=False, warmup=False):
            assert single and indices == [0]
            calls.append((self.backend, warmup))
            result = repetition(('a',))
            if fail and not warmup:
                result['outputs'][0]['strict'] = {'periods': 'changed'}
            return result
        def close(self):
            return ownership_receipt()
    monkeypatch.setattr(benchmark, 'WorkerPool', Pool)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    args = SimpleNamespace(measurement_scope='single', pool_widths=[1], output=tmp_path/'output',
        manifest=manifest, paired_results=tmp_path, regimes=['tess_solar'], correction_adapter=None,
        row_ab=False, timeout=10)
    if fail:
        with pytest.raises(RuntimeError, match='complete-output gate'):
            benchmark.run(args)
        assert len(calls) == 6  # Fail before starting the other implementation.
    else:
        benchmark.run(args)
        assert len(calls) == 12
        plan = json.loads((args.output/'plan.json').read_text())
        assert plan['measurement_scope'] == 'single'
        assert plan['batch_repetitions'] == 0


def test_numerical_fingerprints_cover_shape_dtype_mask_and_objects():
    assert array_hash(np.ones(2, np.float32)) != array_hash(np.ones(2, np.float64))
    assert array_hash(np.ones(2)) != array_hash(np.ones((1, 2)))
    assert masked_hash(np.ma.array([1, 2], mask=[0, 1])) != masked_hash([1, 2])
    with pytest.raises(TypeError, match='Object-array'):
        array_hash(np.array([object()], object))


def test_public_unit_conversion_preserves_strict_native_bits():
    case = dict(name='case', error_scale=2.)
    native = SimpleNamespace(periods=np.array([1., 2.]),
        power=np.ma.array([2., 999.], mask=[0, 1]),
        chi2=np.ma.array([8., 123.], mask=[0, 1]), period=1., SDE=3., extra=np.array([1.]))
    candidate = dict(periods=np.array([1., 2.]), power=np.array([2., np.nan]),
        chi2=np.array([2., np.nan]), valid_periods=np.array([True, False]), period=1., SDE=3.)
    old = fingerprint('gtls', case, native)
    new = fingerprint('candidate', case, candidate)
    assert old['common'] == new['common']
    assert old['strict'] != new['strict']
    native.extra[0] = 2.
    changed = fingerprint('gtls', case, native)
    assert old['strict'] == changed['strict']
    assert old['full_digest'] != changed['full_digest']


@pytest.mark.parametrize('change', ('empty', 'missing_rep', 'missing_case', 'duplicate', 'changed', 'failure', 'count'))
def test_consistency_rejects_incomplete_or_changed_repetitions(change):
    records = [repetition() for _ in range(3)]
    if change == 'empty':
        records = []
    elif change == 'missing_rep':
        records.pop()
    elif change == 'missing_case':
        records[1]['outputs'].pop()
    elif change == 'duplicate':
        records[1]['outputs'][1] = output('a')
    elif change == 'changed':
        records[1]['outputs'][1] = output('b', 'changed')
    elif change == 'failure':
        records[1]['status'] = 'error'
    else:
        records[1]['source_count'] = 3
    assert not consistency(records, expected_names=['a', 'b'], expected_repetitions=3)['eligible']


def test_fastest_native_pool_requires_original_complete_searches():
    records = {'candidate_graph_1worker': configuration(seconds=1.),
               'gtls_graph_1worker': configuration(seconds=8.),
               'gtls_graph_2worker': configuration(seconds=6.),
               'gtls_graph_4worker': configuration(seconds=2.)}
    records['gtls_graph_4worker']['batch'][1]['outputs'][1] = output('b', 'changed')
    summary = compare_public(records, ['a', 'b'], 'a', [1, 2, 4])
    assert summary['public_batch']['strongest_tested_native_workers'] == 2
    assert summary['public_batch']['speedup'] == 6.
    assert not summary['public_batch']['native_pool_configurations']['4']['eligible']
    assert summary['public_single']['speedup'] == 8.


def test_cross_backend_difference_and_failures_never_become_denominators():
    records = {'candidate_graph_1worker': configuration(seconds=1.),
               'gtls_graph_1worker': configuration(seconds=8.)}
    records['candidate_graph_1worker']['batch'][0]['outputs'][1]['common'] = {'periods': 'changed'}
    result = compare_public(records, ['a', 'b'], 'a', [1])
    assert result['public_batch']['speedup'] is None
    records['candidate_graph_1worker']['status'] = 'measurement_failure'
    result = compare_public(records, ['a', 'b'], 'a', [1])
    assert result['public_single']['speedup'] is None
    assert distribution([1., None, 0.01]) is None
    assert distribution([1., float('inf')]) is None


def test_single_batch_and_full_output_prefix_ab_gates():
    records = {'candidate_graph_1worker': configuration(seconds=1.),
               'candidate_row_1worker': configuration(seconds=2.),
               'gtls_graph_1worker': configuration(seconds=8.)}
    records['candidate_row_1worker']['single'][1]['outputs'][0]['full_digest'] = 'extra_array_changed'
    records['candidate_graph_1worker']['single'][2]['outputs'][0]['strict'] = {'periods': 'changed'}
    result = compare_public(records, ['a', 'b'], 'a', [1])
    assert not result['public_single']['eligible']
    assert not result['prefix_ab']['eligible']
    assert result['public_batch']['eligible']


def test_declared_native_bug_difference_keeps_method_specific_frozen_gates():
    records = {'candidate_graph_1worker': configuration(seconds=1.),
               'gtls_graph_1worker': configuration(seconds=8.)}
    for kind in ('single', 'batch'):
        for rep in records['candidate_graph_1worker'][kind]:
            for value in rep['outputs']:
                value['common'] = {'periods': 'declared-native-bug-difference'}
    frozen = {backend: {name: {'strict': {'periods': 'same'}} for name in ('a', 'b')}
              for backend in ('gtls', 'candidate')}
    result = compare_public(records, ['a', 'b'], 'a', [1], frozen=frozen)
    assert result['public_batch']['eligible']
    assert not result['public_batch']['common_output_gate']['eligible']
    assert result['public_single']['eligible']
    frozen['candidate']['b']['strict'] = {'periods': 'unexpected'}
    assert not compare_public(records, ['a', 'b'], 'a', [1], frozen=frozen)['public_batch']['eligible']


def test_corrected_crosscheck_does_not_enter_literal_pool_selection():
    records = {'candidate_graph_1worker': configuration(seconds=1.),
               'gtls_graph_1worker': configuration(seconds=8.),
               'gtls_graph_2worker': configuration(seconds=6.),
               'gtls_corrected_graph_1worker': configuration(seconds=.5)}
    frozen = {backend: {name: {'strict': {'periods': 'same'}} for name in ('a', 'b')}
              for backend in ('gtls', 'gtls_corrected', 'candidate')}
    result = compare_public(records, ['a', 'b'], 'a', [1, 2], frozen=frozen)
    assert result['public_batch']['strongest_tested_native_workers'] == 2
    assert result['public_batch']['speedup'] == 6.
    assert result['corrected_native_crosscheck']['batch']['speedup'] == .5
    assert result['corrected_native_crosscheck']['batch']['source_count'] == 2
    records['gtls_corrected_graph_1worker']['batch'][1]['outputs'][1] = output('b', 'changed')
    bad = compare_public(records, ['a', 'b'], 'a', [1, 2], frozen=frozen)
    assert not bad['corrected_native_crosscheck']['batch']['eligible']
    assert bad['public_batch']['eligible']


def test_own_frozen_public_output_archives_are_verified_before_measurement(tmp_path):
    case = dict(name='case.npz', input_sha256='input', error_scale=2.)
    for backend in ('gtls', 'candidate'):
        folder = tmp_path/'case'/backend
        folder.mkdir(parents=True)
        if backend == 'gtls':
            arrays = dict(periods=np.array([1., 2.]), power=np.array([2., 3.]), chi2=np.array([8., 12.]))
            arrays.update({key+'_mask': np.array([False, False]) for key in ('periods', 'power', 'chi2')})
            result = dict(period=1., score=3.)
        else:
            arrays = dict(public_periods=np.array([1., 2.]), public_power=np.array([2., 3.]),
                          public_chi2=np.array([2., 3.]), public_valid_periods=np.array([True, True]))
            result = dict(public_contract=dict(period=1., SDE=3.))
        archive = folder/'arrays.npz'
        np.savez_compressed(archive, **arrays)
        record = dict(status='ok', input_sha256='input', result=result, arrays_file='arrays.npz',
                      arrays_sha256=sha(archive),
                      arrays={key: dict(sha256=array_hash(value), shape=list(value.shape), dtype=str(value.dtype))
                              for key, value in arrays.items()})
        (folder/'record.json').write_text(json.dumps(record))
    result = frozen_outputs(tmp_path, [case])
    assert result['candidate']['case.npz']['strict']['power'] == result['gtls']['case.npz']['strict']['power']
    (tmp_path/'case/candidate/arrays.npz').unlink()
    assert frozen_outputs(tmp_path, [case]) == result


def make_cohort(root, failing=(), failed_reserves=()):
    originals = selected_names('tess_solar')
    names = originals + [f'tess_solar_null_{index:04d}.npz' for index in range(16, 20)]
    manifest = dict(cases=[dict(file=name, sha256=name, metadata=dict(regime='tess_solar', null=True, search_kwargs={})) for name in names],
                    seal_sha256='seal', source_identity=dict(production_sources={'cuvarbase/tls.py': 'source'}))
    path = root/'manifest.json'
    path.write_text(json.dumps(manifest))
    results = root/'results'
    for name in names:
        for backend in ('gtls', 'candidate', 'gtls_corrected'):
            folder = results/Path(name).stem/backend
            folder.mkdir(parents=True)
            bad = backend == 'gtls' and name in set(failing) | set(failed_reserves)
            record = dict(status='error' if bad else 'ok', input_sha256=name, seal_sha256='seal',
                          elapsed_seconds=0.001 if bad else 999999.,
                          engine_sources=manifest['source_identity']['production_sources'],
                          result=dict(package_sources={'core.py': 'native'},
                                      reference_correction={'correction': 'finite_candidates_before_ranking_v1'}))
            (folder/'record.json').write_text(json.dumps(record))
        trace = dict(correction='finite_candidates_before_ranking_v1', proved_no_op=True)
        (results/Path(name).stem/'correction_trace.json').write_text(json.dumps(trace))
    acceptance = dict(publication_gate={'pass': True}, inputs_manifest_sha256=sha(path),
                      seal_sha256=manifest['seal_sha256'], source_identity=manifest['source_identity'],
                      reference_package_sources={'core.py': 'native'},
                      counts={'planned': len(names), 'accounted': len(names)})
    (results/'acceptance.json').write_text(json.dumps(acceptance))
    return path, results


def test_replacements_are_manifest_order_success_only_and_keep_failures(tmp_path):
    first = selected_names('tess_solar')[0]
    manifest, results = make_cohort(tmp_path, [first], ['tess_solar_null_0016.npz'])
    result = select(manifest, results, 'tess_solar')
    assert result['actual_batch_size'] == 16
    assert result['single_case'] == 'tess_solar_null_0001.npz'
    assert result['replacement_cases'] == [dict(original_case=first, replacement_case='tess_solar_null_0017.npz')]
    assert result['single_replaced']
    assert result['excluded_cases'] == [first, 'tess_solar_null_0016.npz']
    assert result['examined'][0]['backends']['gtls']['study_elapsed_seconds'] == 0.001
    assert result['study_times_are_not_benchmark_denominators']
    verify_worker_sources('gtls', {'files': {'core.py': 'native'}}, result)
    with pytest.raises(ValueError, match='native package'):
        verify_worker_sources('gtls', {'files': {'core.py': 'changed'}}, result)


def test_insufficient_successes_and_missing_results_are_explicit(tmp_path):
    originals = selected_names('tess_solar')
    reserve = [f'tess_solar_null_{index:04d}.npz' for index in range(16, 20)]
    manifest, results = make_cohort(tmp_path, originals[:2], reserve)
    result = select(manifest, results, 'tess_solar')
    assert result['actual_batch_size'] == 14
    assert result['single_case'] == 'tess_solar_null_0002.npz'
    assert result['single_replaced']
    assert result['single_case'] in result['selected_cases']
    assert result['status'] == 'insufficient_paired_successes'
    (results/Path(originals[0]).stem/'gtls/record.json').unlink()
    with pytest.raises(ValueError, match='not complete'):
        select(manifest, results, 'tess_solar')


def test_no_successful_null_does_not_invent_single_latency_input(tmp_path):
    originals = selected_names('tess_solar')
    reserve = [f'tess_solar_null_{index:04d}.npz' for index in range(16, 20)]
    manifest, results = make_cohort(tmp_path, originals, reserve)
    result = select(manifest, results, 'tess_solar')
    assert result['actual_batch_size'] == 0
    assert result['single_case'] is None
    assert result['single_unavailable']
    assert not result['single_replaced']


def test_conditional_correction_timing_uses_trace_not_runtime_or_recovery(tmp_path):
    manifest, results = make_cohort(tmp_path)
    original = selected_names('tess_solar')[3]
    before = select(manifest, results, 'tess_solar')
    assert not before['correction_timing']['required']
    path = results/Path(original).stem/'correction_trace.json'
    trace = json.loads(path.read_text())
    trace['proved_no_op'] = False
    path.write_text(json.dumps(trace))
    after = select(manifest, results, 'tess_solar')
    assert before['selected_cases'] == after['selected_cases']
    assert after['correction_timing']['required']
    assert after['correction_timing']['affected_cases'] == [original]
    verify_worker_sources('gtls_corrected', dict(files={'core.py': 'native'},
                          reference_correction=after['expected_correction']), after)
    with pytest.raises(ValueError, match='correction differs'):
        verify_worker_sources('gtls_corrected', dict(files={'core.py': 'native'},
                              reference_correction={'correction': 'changed'}), after)


def test_unaccepted_independent_study_prevents_any_timing_selection(tmp_path):
    manifest, results = make_cohort(tmp_path)
    path = results/'acceptance.json'
    receipt = json.loads(path.read_text())
    receipt['publication_gate']['pass'] = False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='publication gate'):
        select(manifest, results, 'tess_solar')


def merged_cohort(root):
    studies, entries, production = {}, [], None
    for index, label in enumerate(('main', 'supplement')):
        origin = root/label
        origin.mkdir()
        path, results = make_cohort(origin)
        manifest = json.loads(path.read_text())
        manifest['seal_sha256'] = label+'-seal'
        manifest['source_identity']['generator_sha256'] = label+'-generator'
        path.write_text(json.dumps(manifest))
        for record_path in results.glob('*/*/record.json'):
            record = json.loads(record_path.read_text())
            record['seal_sha256'] = manifest['seal_sha256']
            record_path.write_text(json.dumps(record))
        acceptance_path = results/'acceptance.json'
        acceptance = json.loads(acceptance_path.read_text())
        acceptance.update(inputs_manifest_sha256=sha(path), seal_sha256=manifest['seal_sha256'],
                          source_identity=manifest['source_identity'])
        acceptance_path.write_text(json.dumps(acceptance))
        studies[label] = dict(manifest_path=str(path), acceptance_path=str(acceptance_path),
                              results_root=str(results), manifest_sha256=sha(path),
                              seal_sha256=manifest['seal_sha256'])
        production = manifest['source_identity']['production_sources']
        entries += [dict(value, study_id=label, result_root=str(results/Path(value['file']).stem))
                    for value in manifest['cases'][index*8:(index+1)*8]]
    merged = root/'timing_manifest.json'
    merged.write_text(json.dumps(dict(source_identity={'production_sources': production},
                                     studies=studies, cases=entries)))
    return merged


def test_two_study_null_manifest_preserves_separate_source_and_seal_chains(tmp_path):
    manifest = merged_cohort(tmp_path)
    accepted = accepted_study(manifest, tmp_path/'unused')
    assert set(accepted['independently_accepted_studies']) == {'main', 'supplement'}
    selected = select(manifest, tmp_path/'unused', 'tess_solar')
    assert selected['actual_batch_size'] == 16
    assert selected['single_case'] == 'tess_solar_null_0000.npz'
    assert selected['selected_cases'] == selected_names('tess_solar')
    assert not selected['replacement_cases']


@pytest.mark.parametrize('mutation', ('unaccepted', 'foreign_result_root', 'wrong_manifest_hash', 'changed_metadata'))
def test_merged_null_manifest_rejects_broken_origin_chain(tmp_path, mutation):
    path = merged_cohort(tmp_path)
    manifest = json.loads(path.read_text())
    if mutation == 'unaccepted':
        receipt_path = Path(manifest['studies']['supplement']['acceptance_path'])
        receipt = json.loads(receipt_path.read_text())
        receipt['publication_gate']['pass'] = False
        receipt_path.write_text(json.dumps(receipt))
    elif mutation == 'foreign_result_root':
        manifest['cases'][8]['result_root'] = str(tmp_path/'elsewhere')
    elif mutation == 'wrong_manifest_hash':
        manifest['studies']['supplement']['manifest_sha256'] = 'changed'
    else:
        manifest['cases'][8]['metadata']['search_kwargs']['oversampling_factor'] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        accepted_study(path, tmp_path/'unused')


def test_worker_retains_results_until_after_completion_ack(monkeypatch):
    events = []
    cases = [dict(name='a'), dict(name='b')]
    commands = iter([dict(kind='run', indices=[0, 1], single=False),
                     dict(kind='validate'), dict(kind='close')])
    class Connection:
        def recv(self):
            return next(commands)
        def send(self, value):
            events.append(value['kind'])
        def close(self):
            pass
    monkeypatch.setattr(benchmark, 'load_cases', lambda *args: cases)
    monkeypatch.setattr(benchmark, 'initialize_backend', lambda *args: {})
    monkeypatch.setattr(benchmark, 'retain_cuda_context', lambda: object())
    monkeypatch.setattr(benchmark, 'public_batch', lambda *args: [object(), object()])
    def deferred(*args):
        events.append('hash')
        return {'case': args[1]['name']}
    monkeypatch.setattr(benchmark, 'fingerprint', deferred)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(deviceSynchronize=lambda: events.append('sync')))))
    benchmark.worker(Connection(), dict(manifest='unused', regime='unused', backend='candidate', prefix='graph'))
    assert events.index('complete') < events.index('hash') < events.index('validation')
    assert events[:events.index('complete')].count('sync') == 2


def test_foreign_gpu_process_prevents_timing_and_is_retained_afterward(monkeypatch):
    fake = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=10), SimpleNamespace(pid=20)])
    monkeypatch.setitem(sys.modules, 'pynvml', fake)
    with pytest.raises(RuntimeError, match='Foreign GPU'):
        benchmark.exclusive_gpu_processes([10])
    result = benchmark.exclusive_gpu_processes([10], strict=False)
    assert result['foreign_pids'] == [20]
    assert not result['exclusive']
    assert benchmark.exclusive_gpu_processes([10, 20])['exclusive']


def install_fake_native(monkeypatch, source):
    package = ModuleType('gputls')
    core = ModuleType('gputls.core')
    core.__file__ = 'fake_core.py'
    stats = ModuleType('gputls.stats')
    stats.pink_noise = lambda *args: 1.
    core.spectra = lambda *args: None
    core.search_multi_periods_again = lambda *args: None
    core.snr_stats = lambda *args: stats.pink_noise()
    class Lowest:
        def argmin(self):
            return self
        def get(self):
            return 7
    core.lowestResidualsGPU = Lowest()
    exec(source, core.__dict__)
    original = core.search_single_periods
    monkeypatch.setattr(inspect, 'getsource', lambda function: source if function is original else '')
    package.core, package.stats = core, stats
    monkeypatch.setitem(sys.modules, 'gputls', package)
    monkeypatch.setitem(sys.modules, 'gputls.core', core)
    monkeypatch.setitem(sys.modules, 'gputls.stats', stats)
    return core, stats, original


def test_component_injection_preserves_values_and_finds_real_caller(monkeypatch):
    source = ('def search_single_periods():\n'
              '    bestLocation = lowestResidualsGPU.argmin().get()\n'
              '    diagnostic = snr_stats()\n'
              '    return bestLocation, diagnostic\n')
    core, stats, original = install_fake_native(monkeypatch, source)
    literal = original()
    # The real native caller remains below an instrumentation wrapper, so
    # the callback must walk the frame stack rather than assume two frames.
    exec('def search_multi_periods():\n'
         '    periods, period, power, chi2, SDE = [1., 2.], 1., [2., 3.], [5., 6.], 4.\n'
         '    return search_single_periods()\n', core.__dict__)
    with Components('gtls') as instrument:
        import time
        before = time.perf_counter()
        assert core.search_multi_periods() == literal
        after = time.perf_counter()
        result = instrument.accounting(before, after)
        assert result['endpoint_valid']
        assert result['endpoint_count'] == 1
        assert instrument.search_result['period'] == 1.
        assert result['stage_call_counts']['native_pink_noise_nested'] == 1
        assert result['stage_call_counts']['native_snr_stats_inclusive'] == 1
        assert result['inclusive_stage_seconds']['native_snr_stats_inclusive'] >= result['inclusive_stage_seconds']['native_pink_noise_nested']
    assert core.search_single_periods is original


def test_failed_component_install_restores_every_prior_patch(monkeypatch):
    source = 'def search_single_periods():\n    return 7\n'
    core, stats, original = install_fake_native(monkeypatch, source)
    snapshot = (core.spectra, core.search_multi_periods_again, core.snr_stats, stats.pink_noise)
    with pytest.raises(ValueError, match='exactly once'):
        with Components('gtls'):
            pass
    assert snapshot == (core.spectra, core.search_multi_periods_again, core.snr_stats, stats.pink_noise)


def test_component_summary_rechecks_literal_output_identity():
    one = dict(case={'name': 'a'}, status='ok', literal_outputs={'common': {'x': 'hash'}, 'full_digest': 'full'},
        gpu_ownership=ownership_receipt(),
        repetitions=[dict(denominator_eligible=True, endpoint_valid=True, output_identical_to_literal=True,
                          exclusive_before=exclusive_snapshot(), exclusive_after=exclusive_snapshot(),
                          outputs={'full_digest': 'full'}, common_search_seconds=1.,
                          public_instrumented_seconds=2., after_common_search_seconds=1.,
                          inclusive_stage_seconds={'stage': 0.5}) for _ in range(5)])
    assert compare_components(one, copy.deepcopy(one), 'a')['eligible']
    changed = copy.deepcopy(one)
    changed['repetitions'][2]['outputs']['full_digest'] = 'changed'
    assert not compare_components(one, changed, 'a')['eligible']
