"""CPU checks of queue membership and exact numerical qualification gates."""
from collections import Counter
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.tls_survey.throughput import (Pool, batches, prepare_grids, qualify_rows,
    configure_bls, public_call, scalar_fingerprint, complete_fingerprint, Telemetry,
    archive_bls_spectra, bls_repeat_diagnostics)
from benchmarks.tls_survey.throughput_campaign import (prepare_varied_manifest, winner, select_names,
                                                     validate_tuning_identity)


def row(worker=0, name='a', digest='same'):
    return dict(worker=worker, error=None,
                outputs=[dict(case=name, strict=dict(power=digest))],
                scalars=[dict(case=name, fields=dict(period='one', SDE='two'))])


def test_complete_spectrum_gate_rejects_changed_nonwinning_power():
    assert qualify_rows([row()], {'a': {'power': 'same'}}, ['a'], 1)['passed']
    assert not qualify_rows([row(digest='changed')], {'a': {'power': 'same'}}, ['a'], 1)['passed']


def test_membership_gate_requires_every_case_once_on_every_worker():
    valid = [row(0), row(1)]
    assert qualify_rows(valid, expected_names=['a'], workers=2)['passed']
    assert not qualify_rows(valid[:1], expected_names=['a'], workers=2)['passed']
    assert not qualify_rows(valid + [row(1)], expected_names=['a'], workers=2)['passed']
    assert not qualify_rows(valid, expected_names=['a', 'b'], workers=2)['passed']


def test_batches_never_mix_grid_options_groups_or_lose_inputs():
    cases = [dict(group=value) for value in ('x', 'y', 'x', 'x', 'y')]
    jobs = batches(cases, 2)
    assert Counter(index for job in jobs for index in job) == Counter(range(5))
    assert all(len(job) <= 2 and len({cases[i]['group'] for i in job}) == 1 for job in jobs)
    assert jobs == [[0, 2], [3], [1, 4]]


def test_bounded_queue_completes_whole_cycles_and_preserves_population(monkeypatch):
    import benchmarks.tls_survey.throughput as module
    cases = [dict(name=name, metadata=dict(regime=regime))
             for name, regime in (('a', 'tess'), ('b', 'ztf'), ('c', 'tess'))]
    scalars = {case['name']: dict(period='one', SDE='two') for case in cases}
    class Connection:
        def __init__(self):
            self.command = None
            self.maximum_pending = 0
        def send(self, command):
            assert self.command is None
            self.command = command
            self.maximum_pending = 1
    pool = object.__new__(Pool)
    pool.connections = [Connection(), Connection()]
    pool.ownership = SimpleNamespace(allowed_pids=[10, 20])
    pool.timeout = 1
    def receive(connection):
        command, connection.command = connection.command, None
        return dict(task=command['task'], error=None,
            scalars=[dict(case=cases[i]['name'], fields=scalars[cases[i]['name']])
                     for i in command['indices']])
    pool.receive = receive
    monkeypatch.setattr(module, 'wait', lambda values, timeout: values[:1])
    monkeypatch.setattr(module, 'exclusive_gpu_processes', lambda values: dict(exclusive=True))
    measured = pool.run_queue([[0, 2], [1]], cases, 7, 0, scalars)
    assert measured['source_count'] == 9
    assert measured['completed_input_cycles'] == 3
    assert measured['regime_counts'] == dict(tess=6, ztf=3)
    assert all(connection.command is None for connection in pool.connections)


def test_fastest_failure_cannot_win_and_ties_prefer_smaller_pools():
    def result(speed, workers=1, batch=1, valid=True):
        return dict(status='ok', workers=workers, batch_size=batch,
            gpu_ownership=dict(passed=True),
            qualification=[dict(gate=dict(passed=valid))]*2,
            repetitions=[dict(status='ok')],
            summary=dict(median_repetition_lightcurves_per_second=speed))
    selected = winner([result(100, valid=False), result(2, workers=4),
                       result(2, workers=2, batch=4), result(2, workers=2, batch=1)])
    assert selected['workers'] == 2 and selected['batch_size'] == 1


def test_measurement_cannot_change_frozen_timing_runner_or_protocol():
    tuning = dict(status='complete', driver_sha256='driver', runner_sha256='runner',
                  protocol_sha256='protocol', harness_dependency_sha256={'common.py':'dependency'})
    validate_tuning_identity(tuning,dict(tuning))
    for key in ('driver_sha256','runner_sha256','protocol_sha256','harness_dependency_sha256'):
        changed = dict(tuning)
        changed[key] = 'modified'
        with pytest.raises(ValueError,match='Timing definitions changed'):
            validate_tuning_identity(tuning,changed)
    with pytest.raises(ValueError,match='completed tuning'):
        validate_tuning_identity(dict(tuning,status='running'),tuning)


def test_common_preparation_reproduces_sealed_grid_or_fails():
    from cuvarbase.tls_reference_math import period_grid
    options = dict(R_star=1., M_star=1., period_min=.6, period_max=12.,
                   oversampling_factor=3, n_transits_min=2)
    periods = np.sort(period_grid(27., **options))
    def case():
        return dict(group='shared', data=dict(periods=periods.copy()),
                    metadata=dict(baseline_days=27., grid_kwargs=options))
    cohort = [case(), case()]
    records = prepare_grids(cohort)
    assert len(records) == 1 and records[0]['status'] == 'exact_regeneration'
    assert cohort[0]['data']['periods'] is cohort[1]['data']['periods']
    changed = case()
    changed['data']['periods'][0] = np.nextafter(periods[0], np.inf)
    with pytest.raises(ValueError, match='differs from sealed input'):
        prepare_grids([changed])


def test_varied_nulls_preserve_grid_endpoints_and_aligned_samples(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    manifest = dict(cases=[])
    for regime in ('tess_solar', 'tess_gap_long', 'ztf_solar'):
        for index in range(2):
            name = f'{regime}_nulls_{index:04d}.npz'
            metadata = dict(regime=regime, null=True, search_kwargs=dict(R_star=1., M_star=1.),
                            grid_kwargs={}, baseline_days=19.)
            path = source/name
            np.savez_compressed(path, t=np.arange(1,21.), y=np.arange(20.)+100,
                                dy=np.arange(20.)+1, periods=np.array([2.,3.]),
                                metadata=json.dumps(metadata))
            manifest['cases'].append(dict(file=name, metadata=metadata,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (source/'manifest.json').write_text(json.dumps(manifest))
    result = prepare_varied_manifest(source/'manifest.json', tmp_path/'derived', count=2)
    entries = json.loads(result.read_text())['cases']
    assert len(entries) == 6
    assert sorted({entry['metadata']['ndata'] for entry in entries}) == [16,20]
    for entry in entries:
        with np.load(result.parent/entry['file']) as data:
            assert data['t'][0] == 1 and data['t'][-1] == 20
            np.testing.assert_array_equal(data['y'], data['t']+99)
            np.testing.assert_array_equal(data['dy'], data['t'])
            np.testing.assert_array_equal(data['periods'], [2.,3.])
        assert entry['metadata']['do_not_use_for_recovery_or_false_alarm']
    # Resume verifies bytes, preserving the original derivation rather than
    # regenerating a silently different population.
    assert prepare_varied_manifest(source/'manifest.json', tmp_path/'derived', count=2) == result


def test_science_bls_selection_and_exact_selected_candidate_gate(tmp_path, monkeypatch):
    import benchmarks.tls_survey.throughput as module
    from benchmarks.tls_survey import common
    monkeypatch.setattr(common, 'source_identity', lambda: {'science.py': 'frozen'})
    selection = dict(method='bls_finest', ranker='likelihood')
    seal = dict(source_identity={'science.py': 'frozen'}, bls_selected={'tess_solar': selection})
    seal_path = tmp_path/'seal.json'
    seal_path.write_text(json.dumps(seal))
    cases = [dict(name='source', metadata=dict(regime='tess_solar'), group='original',
                  data={'periods':np.array([2.,3.])})]
    configure_bls(cases, seal_path)
    calls = []
    def compact(case, science, *, arrays=False):
        calls.append((case['bls_selection']['method'], arrays))
        return dict(period=2., score=9.,
            candidates={'raw': {'period':2.,'score':.2}, 'likelihood':{'period':2.,'score':9.}},
            spectra={'power':'complete-spectrum','valid_mask':'mask','periods':'periods'})
    monkeypatch.setattr(module, 'science_bls_module', lambda: SimpleNamespace())
    monkeypatch.setattr(module, 'compact_bls', compact)
    result = public_call('bls', cases, arrays=True)[0]
    assert calls == [('bls_finest', True)] and result['score'] == 9.
    assert 'score' in scalar_fingerprint('bls', result) and 'SDE' not in scalar_fingerprint('bls', result)
    before = complete_fingerprint('bls', cases[0], result)
    result['spectra']['power'] = 'changed-nonwinning-bin'
    after = complete_fingerprint('bls', cases[0], result)
    assert before['strict'] == after['strict']
    assert before['full_digest'] != after['full_digest']
    result['score'] = np.nextafter(result['score'], np.inf)
    assert before['strict'] != complete_fingerprint('bls', cases[0], result)['strict']
    seal['source_identity']['science.py'] = 'edited'
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='science sources'):
        configure_bls(cases, seal_path)


def test_final_science_regime_names_select_all_timing_groups(tmp_path):
    from benchmarks.tls_survey.common import REGIMES as scientific_regimes
    from benchmarks.tls_survey.throughput_campaign import REGIMES as timing_regimes
    assert set(timing_regimes) <= set(scientific_regimes)
    manifest = dict(cases=[dict(file=r+'_development_0000.npz', metadata=dict(regime=r, null=False))
                           for r in scientific_regimes])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    assert len(select_names(path, 1)) == 3
    assert any(name.startswith('tess_gap_long_') for name in select_names(path, 1))


def test_compact_bls_all_selected_rankers_equal_full_science_path(monkeypatch):
    import sys
    from benchmarks.tls_survey.throughput import compact_bls, science_bls_module
    rng = np.random.default_rng(183)
    periods = np.linspace(.6, 12., 103)
    power = rng.uniform(.01, .1, len(periods))
    power[17] = np.nan
    calls = []
    def fake_gpu(t, y, dy, frequencies, **kwargs):
        calls.append((frequencies.copy(), kwargs))
        return power.copy()
    monkeypatch.setitem(sys.modules, 'cuvarbase.bls', SimpleNamespace(eebls_gpu_fast=fake_gpu))
    science = science_bls_module()
    data = dict(t=np.arange(1., 51.), y=1+rng.normal(0,.01,50),
                dy=rng.uniform(.01,.04,50), periods=periods)
    candidates, _ = science.search(data, {}, 'bls_finest')
    for ranker in ('raw', 'likelihood', 'detrended'):
        case = dict(data=data, bls_selection=dict(method='bls_finest', ranker=ranker))
        compact = compact_bls(case, science)
        assert compact == {key:candidates[ranker][key] for key in ('period','score')}
        full = compact_bls(case, science, arrays=True)
        assert full['candidates'] == candidates
        assert compact == {key:full[key] for key in ('period','score')}
        np.testing.assert_array_equal(full['_arrays']['power'], power)
        np.testing.assert_array_equal(full['_arrays']['periods'], periods)
        np.testing.assert_array_equal(calls[-1][0], calls[0][0])
        for key in ('qmin','qmax'):
            np.testing.assert_array_equal(calls[-1][1][key], calls[0][1][key])
        assert calls[-1][1]['noverlap'] == calls[0][1]['noverlap']


def test_bls_spectra_retained_and_variation_reported_without_changing_tls_gate(tmp_path):
    period = np.array([2., 3.])
    power = np.array([.5, .25], dtype=np.float32)
    case = dict(name='a', data=dict(periods=period))
    def make(value, filename, score=.5):
        result = dict(period=2., score=score,
            candidates={'raw': dict(period=2., score=score)},
            spectra={'periods':'periods', 'valid_mask':'mask',
                     'power':hashlib.sha256(value.tobytes()).hexdigest()},
            _arrays=dict(periods=period, power=value))
        archive_bls_spectra(result, tmp_path, filename)
        fingerprint = complete_fingerprint('bls', case, result)
        return dict(worker=0, error=None, outputs=[fingerprint],
                    scalars=[dict(case='a', fields=scalar_fingerprint('bls', result))])
    before = make(power, 'before.npz')
    changed = power.copy()
    changed[1] = np.nextafter(changed[1], np.float32(np.inf))
    after = make(changed, 'after.npz')
    expected = qualify_rows([before])['strict']
    assert qualify_rows([after], expected, ['a'], 1)['passed']
    diagnostic = bls_repeat_diagnostics([after], [before])
    assert diagnostic['changed_power_comparisons'] == 1
    assert diagnostic['changed_selected_endpoints'] == 0
    assert diagnostic['max_absolute_power_difference'] == float(changed[1]-power[1])
    assert diagnostic['comparisons'][0]['changed_finite_power_values'] == 1
    mismatch = make(changed, 'selected-changed.npz', score=np.nextafter(.5, np.inf))
    assert not qualify_rows([mismatch], expected, ['a'], 1)['passed']
    assert bls_repeat_diagnostics([mismatch], [before])['changed_selected_endpoints'] == 1
    # TLS and native-GTLS retain their existing complete-spectrum equality.
    assert not qualify_rows([row(digest='different')], {'a': {'power':'same'}}, ['a'], 1)['passed']
    (tmp_path/'after.npz').write_bytes(b'corrupted archive')
    with pytest.raises(ValueError, match='spectrum changed'):
        bls_repeat_diagnostics([after], [before])


def test_selected_endpoint_queue_failure_retains_actual_values(monkeypatch):
    import benchmarks.tls_survey.throughput as module
    connection = SimpleNamespace(send=lambda command: None)
    pool = object.__new__(Pool)
    pool.connections = [connection]
    pool.ownership = SimpleNamespace(allowed_pids=[10])
    pool.timeout = 1
    # Dictionary keys need a hashable connection, as multiprocessing pipes are.
    class Connection:
        def send(self, command):
            pass
    pool.connections = [Connection()]
    pool.receive = lambda unused: dict(task=0, error=None,
        scalars=[dict(case='a', fields={'period':'same', 'score':'changed'},
                      values=dict(period=2., score=.50000001))])
    monkeypatch.setattr(module, 'wait', lambda values, timeout: values)
    monkeypatch.setattr(module, 'exclusive_gpu_processes', lambda values: dict(exclusive=True))
    with pytest.raises(RuntimeError) as captured:
        pool.run_queue([[0]], [dict(name='a', metadata=dict(regime='tess'))], 1, 0,
                       {'a':{'period':'same', 'score':'previous'}})
    failure = captured.value.failed_queue
    assert failure['status'] == 'error'
    assert failure['tasks'][0]['scalars'][0]['values']['score'] == .50000001
    assert not failure['tasks'][0]['scalar_match']


def test_figure_retains_missing_competitors_and_suppresses_failed_paired_ratio(tmp_path):
    from benchmarks.tls_survey.plot_throughput import read_campaign, SCOPES, BACKENDS
    configurations = []
    for backend in ('baseline', 'candidate'):
        record = dict(status='ok', gpu_ownership=dict(passed=True),
                      qualification=[dict(gate=dict(passed=True))]*2,
                      repetitions=[dict(status='ok', lightcurves_per_second=2.)]*3,
                      summary={}, environment=dict(nvidia_smi='GPU, UUID, 1', cpu_quota_cores=7.65,
                                                   host_memory_limit_bytes=50_000_000_000))
        filename = backend+'.json'
        (tmp_path/filename).write_text(json.dumps(record))
        configurations.append(dict(backend=backend, scope='tess_solar', eligible=True, result=filename,
                                   result_sha256=hashlib.sha256((tmp_path/filename).read_bytes()).hexdigest()))
    campaign = dict(stage='measure', status='complete', configs=configurations,
                    baseline_candidate_spectra=dict(checks=[dict(scope='tess_solar', exact=False)]))
    path = tmp_path/'campaign.json'
    path.write_text(json.dumps(campaign))
    _, data, missing, paired, _ = read_campaign(path)
    assert ('baseline','tess_solar') in data and ('candidate','tess_solar') not in data
    assert not paired['tess_solar']
    assert len(data)+len(missing) == len(SCOPES)*len(BACKENDS)
    assert 'Paired' in missing[('candidate','tess_solar')]


def test_telemetry_rejects_foreign_gpu_context_even_if_it_exits_before_teardown(tmp_path):
    telemetry = Telemetry(tmp_path/'telemetry.jsonl', [1])
    telemetry.rows = [dict(gpu_processes=[10], gpu_used_bytes=5),
                      dict(gpu_processes=[10,99], gpu_used_bytes=15),
                      dict(gpu_processes=[], gpu_used_bytes=0)]
    result = telemetry.summarize([10])
    assert not result['ownership_passed'] and result['foreign_gpu_pids'] == [99]
    assert result['gpu_used_bytes'] == 15
