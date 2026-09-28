"""Operational failures must wake the owner without losing completed evidence."""
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tools.watch_jobs import assess, deliver, read, repair_services, service_alive
from tools.watch_monitor import check
from tools import watch_jobs


def preserved_success():
    return dict(campaign=dict(status='complete', finished_epoch=50,
                              steps=[dict(label='release-gate', exit_code=0)]),
                required_steps=['release-gate'], require_backup=True,
                termination=dict(provider_absence_verified=True),
                verification=dict(status='archive_and_all_members_verified'),
                backup=dict(cloud_readback_verified=True))


def test_archived_failed_release_never_becomes_success():
    snapshot = preserved_success()
    snapshot.update(finalizer=dict(status='complete'), completion=dict(
        cloud_readback_verified=True, release_validation=dict(status='failed', gate_exit_code=1,
        suite_counts=dict(failures=0, errors=0, skipped=0, xfailed=1))))
    outcome, issues = assess(snapshot, 100)
    assert outcome == 'collected_with_failures'
    assert 'release_validation_failed' in dict(issues)
    assert 'suite_incomplete' not in dict(issues)


@pytest.mark.parametrize('missing', ['backup', 'verification', 'termination', 'step'])
def test_success_requires_validation_preservation_shutdown_and_backup(missing):
    snapshot = preserved_success()
    if missing == 'step':
        snapshot['campaign']['steps'] = []
    else:
        snapshot.pop(missing)
    assert assess(snapshot, 1000)[0] != 'ready_for_review'
    assert assess(preserved_success(), 1000)[0] == 'ready_for_review'


def test_nonzero_substep_is_actionable_before_campaign_exits():
    snapshot = dict(campaign=dict(status='running', stages=[dict(name='gpu-validation', exit_code=1)]))
    outcome, issues = assess(snapshot, 100)
    assert outcome == 'needs_attention'
    assert 'step_failed:gpu-validation' in dict(issues)


def test_crash_silent_stall_and_lost_connection_are_distinct():
    base = dict(campaign=dict(status='running'), controller_alive=True,
                remote_activity_epoch=990, stall_seconds=100)
    assert not assess(base, 1000)[1]
    for changed, expected in [({'controller_alive': False}, 'controller_dead'),
                              ({'remote_activity_epoch': 500}, 'stalled'),
                              ({'connection_failures': 3}, 'connection_lost')]:
        assert expected in dict(assess(dict(base, **changed), 1000)[1])


def test_unavailable_panels_remain_actionable_after_collection():
    snapshot = preserved_success()
    snapshot['completion'] = dict(available_panels=11, planned_panels=16)
    outcome, issues = assess(snapshot, 100)
    assert outcome == 'collected_with_failures'
    assert 'panels_unavailable' in dict(issues)


def test_failed_delivery_is_retried_and_success_survives_monitor_restart(tmp_path):
    event = dict(id='incident', job='release', path=str(tmp_path), message='A check failed.')
    config = dict(codex='/codex', thread_id='same-thread', repository=str(tmp_path),
                  _path=str(tmp_path/'config.json'))
    calls = []

    def queue(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0,
                               stdout='' if len(calls) == 1 else 'Queued message accepted for thread same-thread.')

    deliver(event, config, tmp_path, 100, runner=queue)
    assert not event.get('queued')
    deliver(event, config, tmp_path, 110, runner=queue)
    assert len(calls) == 1
    deliver(event, config, tmp_path, 161, runner=queue)
    assert event['queued'] and event['attempts'] == 2
    restored = json.loads(json.dumps(event))
    deliver(restored, config, tmp_path, 10000, runner=queue)
    assert len(calls) == 2
    assert calls[1][calls[1].index('--thread')+1] == 'same-thread'


def test_timed_out_queue_does_not_lose_notification(tmp_path):
    event = dict(id='incident', job='release', path=str(tmp_path), message='A check failed.')
    config = dict(codex='/codex', thread_id='same-thread', repository=str(tmp_path),
                  _path=str(tmp_path/'config.json'))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('codex', 30)

    deliver(event, config, tmp_path, 100, runner=timeout)
    assert not event.get('queued')
    assert event['retry_after'] > 100


def test_independent_watchdog_retries_one_incident_and_recovers(tmp_path):
    config = dict(state_directory=str(tmp_path), codex='/codex', thread_id='existing-thread',
                  repository='/project', heartbeat_grace_seconds=180)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    calls = []

    def queue(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0,
                               stdout='' if len(calls) == 1 else 'Queued message accepted.')

    first = check(path, queue, now=1000)
    assert not first['healthy'] and not first['incident'].get('queued')
    recovered_delivery = check(path, queue, now=1061)
    assert recovered_delivery['incident']['queued']
    check(path, queue, now=1200)
    assert len(calls) == 2
    (tmp_path/'heartbeat.json').write_text(json.dumps(dict(last_success_epoch=1200)))
    healthy = check(path, queue, now=1210)
    assert healthy['healthy'] and 'incident' not in healthy
    assert healthy['last_incident']['id'] == first['incident']['id']


def test_new_failure_after_recovery_gets_a_new_event(tmp_path, monkeypatch):
    job = tmp_path/'job'
    job.mkdir()
    state = tmp_path/'state'
    state.mkdir()
    config = dict(jobs=[dict(id='a', path=str(job), require_backup=False)])
    monkeypatch.setattr(watch_jobs, 'notification', lambda message: True)
    monkeypatch.setattr(watch_jobs, 'deliver', lambda *args: None)
    failed = dict(status='running', steps=[dict(label='step', exit_code=1)])
    path = job/'live-campaign-state.json'
    path.write_text(json.dumps(failed))
    watch_jobs.tick(config, state)
    watch_jobs.tick(config, state)
    assert len(read(state/'state.json')['events']) == 1
    path.write_text(json.dumps(dict(status='running')))
    watch_jobs.tick(config, state)
    path.write_text(json.dumps(failed))
    watch_jobs.tick(config, state)
    assert len(read(state/'state.json')['events']) == 2


def test_review_cannot_silence_an_unverified_backup(tmp_path, monkeypatch):
    job = tmp_path/'job'
    job.mkdir()
    (job/'termination.json').write_text(json.dumps(dict(provider_absence_verified=True)))
    (job/'collection-verification.json').write_text(json.dumps(
        dict(status='archive_and_all_members_verified')))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(state_directory=str(tmp_path/'state'), jobs=[
        dict(id='a', path=str(job), require_backup=True)])))
    monkeypatch.setattr(sys, 'argv', ['watch_jobs.py', '--config', str(config),
        'review', '--job', 'a', '--outcome', 'done'])
    with pytest.raises(ValueError, match='cloud preservation'):
        watch_jobs.main()


def test_guard_and_collector_restart_without_resetting_rental(tmp_path, monkeypatch):
    # Process identity must survive a narrow ps display without restarting
    # a healthy collector alongside the deliberately stopped guard.
    monkeypatch.setenv('COLUMNS', '40')
    ops = tmp_path/'ops'
    ops.mkdir()
    script = '''from pathlib import Path
import json, os, sys, time
root = Path(__file__).resolve().parents[1]
role = 'guard' if len(sys.argv) > 1 else 'monitor'
(root/(role+'-ready.json')).write_text(json.dumps(dict(pid=os.getpid())))
time.sleep(30)
'''
    for name in ['rental.py', 'monitor.py']:
        (ops/name).write_text(script)
    rental = dict(deadline_epoch=12345, cap_usd=1.5, id='existing-owned-rental')
    (tmp_path/'rental.json').write_text(json.dumps(rental))
    memory = {}
    pids = set()

    def wait_for_guard(previous=None):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            guard = read(tmp_path/'guard-ready.json', {}).get('pid')
            monitor = read(tmp_path/'monitor-ready.json', {}).get('pid')
            if guard and monitor and guard != previous:
                pids.update([guard, monitor])
                return guard
            time.sleep(.02)
        pytest.fail('Recovery processes did not start')

    try:
        notices = repair_services(tmp_path, memory, 100)
        assert len(notices) == 2
        pid = wait_for_guard()
        assert service_alive(pid, ops/'rental.py')
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic()+3
        while service_alive(pid, ops/'rental.py') and time.monotonic() < deadline:
            time.sleep(.02)
        notices = repair_services(tmp_path, memory, 110)
        replacement = wait_for_guard(pid)
        assert replacement != pid
        assert [name for name, _ in notices] == ['service_restarted:guard']
        assert read(tmp_path/'rental.json') == rental
    finally:
        for role in ['guard', 'monitor']:
            receipt = read(tmp_path/(role+'-supervisor-restart.json'), {})
            if receipt.get('pid'):
                pids.add(receipt['pid'])
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
