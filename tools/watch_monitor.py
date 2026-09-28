#!/usr/bin/env python3
"""Independent launchd heartbeat watchdog, installed outside Documents.

It reads only its own Application Support state and queues recovery into the
existing Codex conversation. It never attempts to bypass macOS project access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return default


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def check(config_path, runner=subprocess.run, now=None):
    now = time.time() if now is None else now
    config = read(config_path, {})
    root = Path(config['state_directory'])
    heartbeat = read(root/'heartbeat.json', {})
    state = read(root/'watchdog.json', {})
    successful = heartbeat.get('last_success_epoch') or 0
    healthy = now-successful < config.get('heartbeat_grace_seconds', 180)
    state.update(checked_epoch=now, healthy=healthy, last_success_epoch=successful)
    if healthy:
        if state.get('incident'):
            state['last_incident'] = dict(state.pop('incident'), recovered_epoch=now)
    else:
        incident = state.setdefault('incident', dict(
            id=hashlib.sha256(str(now).encode()).hexdigest()[:20], detected_epoch=now))
        if not incident.get('queued') and now >= incident.get('retry_after', 0):
            message = (
                '[Automated cuvarbase monitor recovery '+incident['id']+'] '
                'The persistent job observer has no recent successful heartbeat. '
                'Inspect '+str(root/'watchdog.json')+' and '+str(root/'observer.log')+'. '
                'This is the user-requested monitoring watchdog, not a new authorization. '
                'If the observer is absent or stuck, restore it from this Codex context using '
                'python3 '+shlex.quote(config['repository']+'/tools/watch_jobs.py')+
                ' --config '+shlex.quote(str(config_path))+
                ' daemon, detached with output in observer.log. Check the daemon lock and PID '
                'before replacing a live process. Then inspect job status and continue authorized '
                'work within existing budgets. Do not create duplicate rentals or repeat failed '
                'scientific experiments. Record acknowledgement in '+str(root/'watchdog-ack.json')+'.')
            incident['attempts'] = incident.get('attempts', 0)+1
            try:
                result = runner([config['codex'], 'queue', '--thread', config['thread_id'],
                                 '--message', message], cwd=root, capture_output=True,
                                text=True, timeout=30)
                incident['exit_code'] = result.returncode
                if result.returncode == 0 and 'Queued message ' in result.stdout:
                    incident.update(queued=True, queued_epoch=now, receipt=result.stdout.strip())
            except (OSError, subprocess.TimeoutExpired) as error:
                incident['error_type'] = type(error).__name__
            incident['retry_after'] = now+min(900, 60*2**min(incident['attempts']-1, 4))
    write(root/'watchdog.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    check(parser.parse_args().config)
