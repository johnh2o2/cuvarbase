"""CPU tests for strict NVML host/container identity and lifetime gates."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from . import benchmark, components
from .test_timing import exclusive_snapshot, ownership_receipt, single_configuration


def nvml(monkeypatch, pids):
    state = dict(pids=list(pids))
    fake = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle:
            [SimpleNamespace(pid=pid) for pid in state['pids']])
    monkeypatch.setitem(sys.modules, 'pynvml', fake)
    return state


class Process:
    def __init__(self, pid=11):
        self.pid, self.exitcode, self.alive = pid, None, True
    def is_alive(self):
        return self.alive


def ready(pid=11, namespace=None):
    return dict(pid=pid, namespace_pids=[pid] if namespace is None else namespace,
                cuda_context_allocation_bytes=1, cuda_context_synchronized=True)


def test_hidden_outer_pid_requires_empty_start_one_live_context_and_exit(monkeypatch):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    process = Process()
    state['pids'] = [3212228]
    assert ownership.bind([ready()], [process], timeout=0) == [3212228]
    assert ownership.receipt['bindings'][0]['method'] == 'single_worker_lifecycle'
    assert ownership.receipt['passed'] is False  # Exit proof is still pending.
    process.alive, process.exitcode = False, 0
    state['pids'] = []
    receipt = ownership.finish([process], timeout=0)
    assert benchmark.ownership_valid(dict(gpu_ownership=receipt))
    assert receipt['before_start']['observed_pids'] == []
    assert receipt['after_start']['observed_pids'] == [3212228]
    assert receipt['after_exit']['observed_pids'] == []


def test_preexisting_gpu_context_rejected_before_spawning(monkeypatch):
    nvml(monkeypatch, [500])
    with pytest.raises(RuntimeError, match='must be empty') as caught:
        benchmark.GPUOwnership()
    assert caught.value.gpu_ownership['before_start']['observed_pids'] == [500]


@pytest.mark.parametrize('observed', ([], [101, 202]))
def test_missing_or_ambiguous_new_context_is_rejected(monkeypatch, observed):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = observed
    with pytest.raises(RuntimeError, match='Ambiguous or missing'):
        ownership.bind([ready()], [Process()], timeout=0)


@pytest.mark.parametrize('mutation', ('wrong_pid', 'dead', 'no_allocation', 'not_synchronized'))
def test_bind_requires_the_actual_live_initialized_child(monkeypatch, mutation):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = [101]
    process, value = Process(), ready()
    if mutation == 'wrong_pid':
        value['pid'] = 12
    elif mutation == 'dead':
        process.alive = False
    elif mutation == 'no_allocation':
        value['cuda_context_allocation_bytes'] = 0
    else:
        value['cuda_context_synchronized'] = False
    with pytest.raises(RuntimeError, match='live synchronized'):
        ownership.bind([value], [process], timeout=0)


def test_multiworker_hidden_ids_are_proved_as_a_set_without_invented_mapping(monkeypatch):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = [101, 102]
    processes = [Process(11), Process(12)]
    assert ownership.bind([ready(11), ready(12)], processes, timeout=0) == [101, 102]
    assert ownership.receipt['identity_scope'] == 'owned_pool_set'
    assert all(value['host_pid'] is None for value in ownership.receipt['bindings'])
    for process in processes:
        process.alive, process.exitcode = False, 0
    state['pids'] = []
    assert benchmark.ownership_valid(dict(gpu_ownership=ownership.finish(processes, timeout=0)))


@pytest.mark.parametrize('count', (2, 4))
def test_multiworker_direct_ids_are_preserved(monkeypatch, count):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = list(range(101, 101+count))
    processes = [Process(11+i) for i in range(count)]
    values = [ready(11+i, [101+i, 11+i]) for i in range(count)]
    assert ownership.bind(values, processes, timeout=0) == state['pids']
    assert ownership.receipt['identity_scope'] == 'individual'


@pytest.mark.parametrize('observed', ([101], [101, 102, 103]))
def test_pool_requires_exactly_as_many_new_host_pids_as_live_children(monkeypatch, observed):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = observed
    with pytest.raises(RuntimeError, match='Ambiguous or missing'):
        ownership.bind([ready(11), ready(12)], [Process(11), Process(12)], timeout=0)


def test_pool_rejects_duplicate_or_ambiguous_direct_identity(monkeypatch):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    state['pids'] = [101, 102]
    with pytest.raises(RuntimeError, match='share one'):
        ownership.bind([ready(11, [11,101]), ready(12, [12,101])], [Process(11), Process(12)], timeout=0)
    with pytest.raises(RuntimeError, match='Ambiguous directly'):
        ownership.bind([ready(11, [11,101,102]), ready(12)], [Process(11), Process(12)], timeout=0)


@pytest.mark.parametrize('observed', ([], [102], [101, 102]))
def test_call_checks_reject_disappeared_replaced_or_extra_owner(monkeypatch, observed):
    nvml(monkeypatch, observed)
    with pytest.raises(RuntimeError, match='exclusive timings'):
        benchmark.exclusive_gpu_processes([101])
    result = benchmark.exclusive_gpu_processes([101], strict=False)
    assert result['exclusive'] is False
    assert result['observed_pids'] == observed


@pytest.mark.parametrize('mode', ('stale_context', 'foreign_context', 'alive', 'forced', 'nonzero_exit'))
def test_exit_must_be_clean_and_context_must_disappear(monkeypatch, mode):
    state = nvml(monkeypatch, [])
    ownership = benchmark.GPUOwnership()
    process = Process()
    state['pids'] = [101]
    ownership.bind([ready()], [process], timeout=0)
    process.alive, process.exitcode = False, 0
    state['pids'] = []
    forced = []
    if mode == 'stale_context':
        state['pids'] = [101]
    elif mode == 'foreign_context':
        state['pids'] = [202]
    elif mode == 'alive':
        process.alive = True
    elif mode == 'forced':
        forced = [11]
    else:
        process.exitcode = 1
    result = ownership.finish([process], forced=forced, timeout=0)
    assert result['passed'] is False
    assert not benchmark.ownership_valid(dict(gpu_ownership=result))


@pytest.mark.parametrize('mode', ('missing_receipt', 'changed_owner', 'missing_exit', 'nonempty_baseline'))
def test_summary_rechecks_raw_identity_receipts(monkeypatch, mode):
    from .summarize import gate
    record = single_configuration()
    if mode == 'missing_receipt':
        del record['gpu_ownership']
    elif mode == 'changed_owner':
        record['single'][0]['exclusive_after'] = exclusive_snapshot([202])
    elif mode == 'missing_exit':
        del record['gpu_ownership']['after_exit']
    else:
        record['gpu_ownership']['before_start'] = exclusive_snapshot([101])
    assert not gate(record, 'single', ['a'])['eligible']


def test_cuda_allocation_is_retained_after_startup_synchronization(monkeypatch):
    events, allocation = [], object()
    def alloc(size):
        assert size == 1
        events.append('allocate')
        return allocation
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(
        alloc=alloc, runtime=SimpleNamespace(deviceSynchronize=lambda: events.append('sync')))))
    assert benchmark.retain_cuda_context() is allocation
    assert events == ['allocate', 'sync']


@pytest.mark.parametrize('exit_context', ([], [101]))
def test_component_supervisor_gates_start_run_and_exit(tmp_path, monkeypatch, exit_context):
    state = nvml(monkeypatch, [])
    events, process = [], Process()
    args = SimpleNamespace(output=tmp_path/'output', regimes=['tess_solar'], backend='candidate')
    class Connection:
        def poll(self, timeout):
            return True
        def recv(self):
            events.append('receive')
            return dict(kind='ready', **ready()) if events.count('receive') == 1 else dict(kind='complete')
        def send(self, command):
            assert command['kind'] == 'bind'
            assert command['ownership']['after_start']['observed_pids'] == [101]
            events.append('run_after_binding')
            record = dict(case={'name': 'a'}, status='ok', gpu_ownership=command['ownership'],
                          repetitions=[dict(exclusive_before=exclusive_snapshot(), exclusive_after=exclusive_snapshot())])
            benchmark.write(args.output/'tess_solar.json', record)
            benchmark.write(args.output/'summary.json', dict(status='awaiting_worker_exit', regimes={}))
        def close(self):
            pass
    def start():
        assert state['pids'] == []
        state['pids'] = [101]
        events.append('start')
    def join(timeout):
        process.alive, process.exitcode = False, 0
        state['pids'] = exit_context
        events.append('join')
    process.start, process.join = start, join
    context = SimpleNamespace(Pipe=lambda: (Connection(), Connection()), Process=lambda **kwargs: process)
    monkeypatch.setattr(components.mp, 'get_context', lambda name: context)
    original = benchmark.GPUOwnership.finish
    monkeypatch.setattr(benchmark.GPUOwnership, 'finish', lambda self, processes, forced=():
                        original(self, processes, forced=forced, timeout=0))
    if exit_context:
        with pytest.raises(RuntimeError, match='ownership'):
            components.run(args)
    else:
        components.run(args)
    record = json.loads((args.output/'tess_solar.json').read_text())
    assert record['gpu_ownership']['passed'] is (not exit_context)
    assert events.index('start') < events.index('run_after_binding') < events.index('join')
