#!/usr/bin/env python3
"""Watch registered cuvarbase jobs and queue actionable follow-ups in their chat.

Run once per minute with launchd. State and delivery receipts live outside the
repository. Polling uses no model calls; only new incidents or completed work
queue a Codex turn. No benchmark retries or new rentals are performed here.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

RUNNING = {'running', 'waiting_for_diagnostics', 'waiting_for_verified_collection'}
FAILED = {'failed', 'error', 'interrupted', 'complete_with_failures'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def assess(snapshot, now):
    """Keep scientific/validation success separate from collection success."""
    campaign = snapshot.get('campaign') or {}
    issues = []
    if campaign.get('status') in FAILED:
        issues.append(('campaign_failed', 'The job finished with failures.'))
    for row in campaign.get('stages', []) + campaign.get('steps', []):
        name = row.get('name', row.get('label', 'unnamed'))
        if row.get('status') in FAILED or row.get('exit_code') not in (None, 0):
            issues.append(('step_failed:'+name, 'A required step failed: '+name))
    completion = snapshot.get('completion') or {}
    validation = completion.get('release_validation') or {}
    if validation.get('status') in FAILED or validation.get('gate_exit_code') not in (None, 0):
        issues.append(('release_validation_failed', 'The release validation did not pass.'))
    counts = validation.get('suite_counts') or {}
    if any(counts.get(k, 0) for k in ('failures', 'errors', 'skipped')):
        issues.append(('suite_incomplete', 'GPU tests failed, errored, or skipped required coverage.'))
    if completion.get('available_panels', 0) < completion.get('planned_panels', 0):
        issues.append(('panels_unavailable', 'Some planned timing panels did not qualify; retain their failures.'))
    finalizer = snapshot.get('finalizer') or {}
    if finalizer.get('status') == 'failed':
        issues.append(('finalization_failed', 'Reporting or cloud preservation failed.'))
    if snapshot.get('connection_failures', 0) >= 3:
        issues.append(('connection_lost', 'Three successive remote checks failed.'))
    if campaign.get('status') in RUNNING:
        if snapshot.get('controller_alive') is False:
            issues.append(('controller_dead', 'The controller exited while its receipt still says running.'))
        heartbeat = snapshot.get('remote_activity_epoch')
        if heartbeat and now-heartbeat > snapshot.get('stall_seconds', 900):
            issues.append(('stalled', 'No remote log or checkpoint activity within the declared interval.'))
        for row in campaign.get('steps', []) + campaign.get('stages', []):
            name = row.get('label', row.get('name', 'unnamed'))
            limit = snapshot.get('step_limits', {}).get(name)
            if limit and 'exit_code' not in row and now-row.get('started_epoch', now) > limit:
                issues.append(('step_overdue:'+name, 'A running step exceeded its time allowance: '+name))
    terminated = (snapshot.get('termination') or {}).get('provider_absence_verified') is True
    verified = (snapshot.get('verification') or {}).get('status') == 'archive_and_all_members_verified'
    if terminated and not verified:
        issues.append(('evidence_missing', 'The rental ended before verified local collection.'))
    if campaign and campaign.get('status') not in RUNNING:
        finished = campaign.get('finished_epoch', now)
        if not (terminated and verified) and now-finished > 300:
            issues.append(('collection_overdue', 'A terminal job still needs verified collection and shutdown.'))
        if campaign.get('status') == 'complete':
            steps = {r.get('label', r.get('name')): r for r in campaign.get('steps', []) + campaign.get('stages', [])}
            if any(steps.get(name, {}).get('exit_code') != 0 for name in snapshot.get('required_steps', [])):
                issues.append(('required_steps_missing', 'The completion receipt omits required passing steps.'))
    backed_up = completion.get('cloud_readback_verified') is True or (
        snapshot.get('backup') or {}).get('cloud_readback_verified') is True
    if terminated and verified:
        if issues:
            outcome = 'collected_with_failures'
        elif snapshot.get('require_backup') and not backed_up:
            outcome = 'awaiting_backup'
        elif campaign.get('status') == 'complete':
            outcome = 'ready_for_review'
        else:
            outcome = 'outcome_unknown'
        issues.append(('results_ready', 'Results are preserved and the rental is off; review and finish remaining work.'))
    else:
        outcome = 'needs_attention' if issues else 'running'
    return outcome, issues


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def service_alive(pid, script):
    if not isinstance(pid, int) or pid <= 0:
        return False
    value = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='],
                           capture_output=True, text=True, timeout=5)
    return value.returncode == 0 and str(script) in shlex.split(value.stdout.strip())


def repair_services(root, memory, now):
    """Restart only the pre-existing bounded guard/collector, at most 3/hour."""
    if (root/'termination.json').exists() or not (root/'rental.json').exists():
        return []
    notices = []
    for role, script, args in [('guard', root/'ops/rental.py', ['guard']),
                               ('monitor', root/'ops/monitor.py', [])]:
        ready = read(root/(role+'-ready.json'), {})
        if not script.exists() or service_alive(ready.get('pid'), script):
            continue
        recent = [v for v in memory.setdefault('restarts', {}).get(role, []) if now-v < 3600]
        if len(recent) >= 3:
            notices.append(('service_unhealthy:'+role, 'The '+role+' needs review after repeated exits.'))
            continue
        with (root/(role+'.log')).open('a') as log:
            process = subprocess.Popen([sys.executable, str(script), *args], stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        recent.append(now)
        memory['restarts'][role] = recent
        # The service writes its own ready receipt; retain the recovery separately.
        write(root/(role+'-supervisor-restart.json'), dict(pid=process.pid, epoch=now))
        if role == 'guard' and Path('/usr/bin/caffeinate').exists():
            wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            write(root/'supervisor-wake.json', dict(pid=wake.pid, waits_for=process.pid))
        notices.append(('service_restarted:'+role, 'Restarted the existing '+role+' without resetting its deadline.'))
    return notices


def probe(job, root):
    observer = module(root/'ops/monitor.py', 'job_observer')
    script = """from pathlib import Path
import json, time
root = Path(REMOTE_ROOT)
state = json.loads((root/'campaign-state.json').read_text())
alive = False
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().split(b'\\0')
        alive |= CONTROLLER.encode() in args
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
excluded = {'venv','release-venv','sources','inputs','release-source','__pycache__','.git'}
import os
latest = 0
for base, folders, files in os.walk(root):
    folders[:] = [name for name in folders if name not in excluded]
    for name in files:
        if Path(name).suffix in {'.log','.json','.jsonl'}:
            try: latest = max(latest,(Path(base)/name).stat().st_mtime)
            except FileNotFoundError: pass
print(json.dumps(dict(campaign=state,controller_alive=alive,remote_activity_epoch=latest)))
""".replace('REMOTE_ROOT', repr(job['remote_root'])).replace('CONTROLLER', repr(job['controller']))
    return json.loads(observer.ssh('python3 -c '+shlex.quote(script), timeout=25))


def notification(message):
    script = ('on run argv\n display notification (item 1 of argv) '
              'with title "cuvarbase job monitor"\nend run')
    try:
        return subprocess.run(['/usr/bin/osascript', '-e', script, message],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def deliver(event, config, state_dir, now, runner=subprocess.run):
    """A failed queue attempt remains pending and is retried after backoff."""
    if event.get('queued') or now < event.get('retry_after', 0):
        return
    ack = state_dir/'acks'/(event['id']+'.json')
    message = (
        '[Automated cuvarbase monitor '+event['id']+'] '+event['message']+'\n'
        'Job: '+event['job']+'\nEvidence directory: '+event['path']+'\n'
        'Resume the already-authorized cuvarbase work and inspect the preserved receipts. '
        'This monitor event grants no additional permissions or budget. Never rerun failed numerical '
        'experiments to replace their failures; fix operational problems separately. Do not create '
        'duplicate rentals. Preserve partial evidence and honor existing shutdown limits. '
        'Report important findings in this chat. Record receipt of this event with: '
        'python3 tools/watch_jobs.py --config '+shlex.quote(str(config['_path']))+
        ' ack --event '+event['id']+'. Expected acknowledgement: '+str(ack))
    event['attempts'] = event.get('attempts', 0)+1
    try:
        result = runner([config['codex'], 'queue', '--thread', config['thread_id'],
                         '--message', message], capture_output=True, text=True, timeout=30,
                        cwd=config['repository'])
        event['delivery_exit_code'] = result.returncode
        if result.returncode == 0 and 'Queued message ' in result.stdout:
            event.update(queued=True, queued_epoch=now, delivery_receipt=result.stdout.strip())
        else:
            event['delivery_error'] = 'Codex queue was not acknowledged; see exit code.'
    except (OSError, subprocess.TimeoutExpired) as error:
        event['delivery_error'] = type(error).__name__
    event['retry_after'] = now+min(900, 60*2**min(event['attempts']-1, 4))


def tick(config, state_dir):
    now = time.time()
    state = read(state_dir/'state.json', {'jobs': {}, 'events': {}})
    status = {}
    for job in config['jobs']:
        root = Path(job['path'])
        memory = state['jobs'].setdefault(job['id'], {})
        snapshot = dict(campaign=read(root/'live-campaign-state.json'),
            completion=read(root/'completion-summary.json'),
            finalizer=read(root/'finalization-status.json'),
            termination=read(root/'termination.json'),
            verification=read(root/'collection-verification.json'),
            backup=read(root/'r2-readback-receipt.json'),
            required_steps=job.get('required_steps', []), require_backup=job.get('require_backup', True),
            step_limits=job.get('step_limits', {}), stall_seconds=job.get('stall_seconds', 900))
        if (root/'collected/campaign-state.json').exists():
            snapshot['campaign'] = read(root/'collected/campaign-state.json')
        notices = repair_services(root, memory, now)
        if not snapshot['termination'] and (root/'rental.json').exists():
            try:
                snapshot.update(probe(job, root))
                memory.update(connection_failures=0, last_contact_epoch=now)
                write(state_dir/'checkpoints'/(job['id']+'.json'), snapshot['campaign'])
            except Exception as error:
                memory['connection_failures'] = memory.get('connection_failures', 0)+1
                memory['last_probe_error_type'] = type(error).__name__
        snapshot['connection_failures'] = memory.get('connection_failures', 0)
        outcome, issues = assess(snapshot, now)
        issues.extend(notices)
        reviewed = read(state_dir/'reviews'/(job['id']+'.json'))
        if reviewed and snapshot['termination'] and snapshot['verification']:
            issues = []
            outcome = 'reviewed: '+reviewed['outcome']
        if issues:
            code = '|'.join(sorted({code for code, _ in issues}))
            message = ' '.join(message for _, message in issues)
            if memory.get('active_issue_codes') != code:
                memory['incident_sequence'] = memory.get('incident_sequence', 0)+1
                memory['current_event_id'] = None
            memory['active_issue_codes'] = code
            key = job['id']+':'+code+':'+str(memory['incident_sequence'])
            identifier = memory.get('current_event_id') or hashlib.sha256(key.encode()).hexdigest()[:20]
            memory['current_event_id'] = identifier
            if identifier not in state['events']:
                state['events'][identifier] = dict(id=identifier, code=code, job=job['id'],
                    path=str(root), message=message, detected_epoch=now, queued=False)
                write(state_dir/'state.json', state)
                state['events'][identifier]['desktop_notification_requested'] = notification(message)
        else:
            memory['active_issue_codes'] = None
        status[job['id']] = dict(outcome=outcome, issues=[code for code, _ in issues],
                                last_contact_epoch=memory.get('last_contact_epoch'), evidence=str(root))
    for event in state['events'].values():
        if (state_dir/'acks'/(event['id']+'.json')).exists():
            event['acknowledged'] = True
        deliver(event, config, state_dir, now)
        if (event.get('queued') and not event.get('acknowledged') and
                now-event['queued_epoch'] > 600 and not event.get('unacknowledged_alert')):
            event['unacknowledged_alert'] = notification(
                'A cuvarbase follow-up is queued but not acknowledged. Check whether Codex is running or rate-limited.')
        write(state_dir/'state.json', state)
    state['last_poll_epoch'] = now
    state['jobs_status'] = status
    write(state_dir/'state.json', state)
    pending = [event['id'] for event in state['events'].values() if not event.get('queued')]
    write(state_dir/'status.json', dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          jobs=status, pending_deliveries=pending,
          queued_events=sum(e.get('queued', False) for e in state['events'].values())))


def daemon(config, state_dir):
    """Run from Codex's project-access context; launchd watches its heartbeat."""
    with (state_dir/'daemon.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        heartbeat = dict(pid=os.getpid(), started_epoch=time.time(), last_success_epoch=None)
        while True:
            heartbeat['last_attempt_epoch'] = time.time()
            write(state_dir/'heartbeat.json', heartbeat)
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--config', config['_path'], 'poll'], timeout=110)
                heartbeat['last_exit_code'] = result.returncode
                if result.returncode == 0:
                    heartbeat['last_success_epoch'] = time.time()
            except subprocess.TimeoutExpired:
                heartbeat['last_exit_code'] = 'poll_timeout'
            write(state_dir/'heartbeat.json', heartbeat)
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('poll')
    commands.add_parser('daemon')
    commands.add_parser('status')
    ack = commands.add_parser('ack')
    ack.add_argument('--event', required=True)
    review = commands.add_parser('review')
    review.add_argument('--job', required=True)
    review.add_argument('--outcome', required=True)
    args = parser.parse_args()
    config = read(args.config)
    config['_path'] = str(args.config.resolve())
    state_dir = Path(config['state_directory'])
    state_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'daemon':
        daemon(config, state_dir)
        return
    with (state_dir/'poll.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if args.command == 'poll':
            try:
                tick(config, state_dir)
            except Exception as error:
                fault = read(state_dir/'monitor-error.json', dict(id='monitor-internal-error',
                    job='monitor', path=str(state_dir), queued=False,
                    message='The job monitor itself needs attention. Inspect its launchd error log.'))
                fault['error_type'] = type(error).__name__
                if not fault.get('queued'):
                    notification(fault['message'])
                deliver(fault, config, state_dir, time.time())
                write(state_dir/'monitor-error.json', fault)
                raise
        elif args.command == 'status':
            print(json.dumps(read(state_dir/'status.json', {}), indent=2))
        elif args.command == 'ack':
            state = read(state_dir/'state.json', {})
            fault = read(state_dir/'monitor-error.json', {})
            if args.event not in state.get('events', {}) and args.event != fault.get('id'):
                raise ValueError('Unknown monitor event')
            write(state_dir/'acks'/(args.event+'.json'), dict(acknowledged_epoch=time.time()))
        else:
            job = next(j for j in config['jobs'] if j['id'] == args.job)
            root = Path(job['path'])
            if not (read(root/'termination.json', {}).get('provider_absence_verified') and
                    read(root/'collection-verification.json', {}).get('status') == 'archive_and_all_members_verified'):
                raise ValueError('Preserve the job and verify shutdown before closing its review')
            backup = read(root/'r2-readback-receipt.json', {})
            completion = read(root/'completion-summary.json', {})
            if job.get('require_backup', True) and not (
                    backup.get('cloud_readback_verified') or completion.get('cloud_readback_verified')):
                raise ValueError('Verify cloud preservation before closing its review')
            write(state_dir/'reviews'/(args.job+'.json'), dict(outcome=args.outcome, reviewed_epoch=time.time()))


if __name__ == '__main__':
    main()
