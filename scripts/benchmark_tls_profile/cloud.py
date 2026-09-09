#!/usr/bin/env python3
"""Credential-silent lifecycle for the bounded TLS profiling experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(os.environ.get('CUVARBASE_BENCHMARK_POD_DIR', 'analysis/tls-profile-20260908'))
STATE = ROOT / 'pod.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    result = {}
    for line in Path('.runpod.env').read_text().splitlines():
        match = re.match(r'(?:export\s+)?(RUNPOD_\w+)=(.*)', line)
        if match:
            result[match[1]] = match[2].strip().strip('\"\'')
    return result


def api(query):
    key = config()['RUNPOD_API_KEY']
    try:
        response = subprocess.run(['curl', '--silent', '--fail', '--max-time', '30',
            '--request', 'POST', '--header', 'Content-Type: application/json',
            '--url', 'https://api.runpod.io/graphql?api_key=' + key, '--data-binary', '@-'],
            input=json.dumps({'query': query}), capture_output=True, text=True)
        if response.returncode:
            raise RuntimeError('Transport failure')
        data = json.loads(response.stdout)
    except Exception:
        raise RuntimeError('RunPod API transport failed; credentials omitted') from None
    if data.get('errors'):
        raise RuntimeError(json.dumps(data['errors']).replace(key, '[redacted]'))
    return data['data']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def terminate(reason):
    pod = json.loads(STATE.read_text())
    if pod.get('termination_verified'):
        return
    response = api('mutation { podTerminate(input:{podId:' + json.dumps(pod['id']) + '}) }')
    save(ROOT / 'termination-response.json', response)
    for _ in range(10):
        active = api('query { myself { pods { id name desiredStatus costPerHr } } }')
        if all(p['id'] != pod['id'] for p in active['myself']['pods']):
            elapsed = time.time() - pod['created_epoch']
            pod.update(termination_verified=True, terminated_utc=now(), termination_reason=reason,
                       estimated_usd=elapsed / 3600 * pod['costPerHr'])
            save(STATE, pod)
            save(ROOT / 'pods-after-termination.json', active)
            print(json.dumps({'terminated': pod['id'], 'verified': True,
                              'estimated_usd': pod['estimated_usd']}), flush=True)
            return
        time.sleep(3)
    raise RuntimeError('Termination absence not yet verified')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['create', 'status', 'guard', 'ssh', 'put', 'get', 'terminate'])
    ap.add_argument('arguments', nargs='*')
    args = ap.parse_args()
    if args.action == 'create':
        if STATE.exists() and not json.loads(STATE.read_text()).get('termination_verified'):
            raise RuntimeError('Profiling pod already recorded; inspect status')
        before = api('query { myself { pods { id name desiredStatus costPerHr } } }')
        save(ROOT / 'pods-before.json', before)
        quoted = api('query { gpuTypes { id displayName securePrice communityPrice } }')
        save(ROOT / 'gpu-quotes.json', [g for g in quoted['gpuTypes'] if g['id'] == 'NVIDIA A40'])
        created = time.time()
        data = api('mutation { podFindAndDeployOnDemand(input: {cloudType: ALL, gpuCount: 1, '
            'volumeInGb: 0, containerDiskInGb: 35, minVcpuCount: 8, minMemoryInGb: 20, '
            'gpuTypeId: "NVIDIA A40", name: "cuvarbase-tls-profiling", '
            'imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", '
            'ports: "22/tcp", volumeMountPath: "/workspace"}) {id costPerHr} }')
        pod = data['podFindAndDeployOnDemand']
        pod.update(created_epoch=created, created_utc=now(), experiment_cap_usd=5.,
                   prior_estimated_usd=3.7664614732111117, authorized_total_usd=50.)
        save(STATE, pod)
        if pod['costPerHr'] > .70:
            terminate('Advertised rate exceeds this experiment limit')
            raise RuntimeError('Rate exceeded local experiment cap')
        with (ROOT / 'budget-guard.log').open('a') as log:
            guard = subprocess.Popen([sys.executable, __file__, 'guard'], stdout=log,
                                     stderr=log, start_new_session=True)
        pod['guard_pid'] = guard.pid
        save(STATE, pod)
        print(json.dumps({'created': pod['id'], 'rate': pod['costPerHr'], 'cap': 5.}), flush=True)
        return
    if args.action == 'guard':
        while True:
            pod = json.loads(STATE.read_text())
            if pod.get('termination_verified'):
                return
            spent = (time.time() - pod['created_epoch']) / 3600 * pod['costPerHr']
            # Independent hard limit on this experiment, including idle time.
            if spent >= pod['experiment_cap_usd'] - .10:
                terminate('Experiment budget guard')
                return
            time.sleep(30)
    elif args.action == 'status':
        pod = json.loads(STATE.read_text())
        if pod.get('termination_verified'):
            print(json.dumps({'terminated': True, 'estimated_usd': pod['estimated_usd']}))
            return
        data = api('query { pod(input:{podId:' + json.dumps(pod['id']) + '}) '
                   '{id name desiredStatus costPerHr lastStartedAt runtime {uptimeInSeconds '
                   'ports {ip isIpPublic privatePort publicPort type}}} }')['pod']
        save(ROOT / 'pod-provider-status.json', data)
        for port in (data.get('runtime') or {}).get('ports', []):
            if port['privatePort'] == 22 and port['isIpPublic']:
                pod.update(ssh_host=port['ip'], ssh_port=port['publicPort'])
        if data.get('lastStartedAt'):
            pod['provider_started_utc'] = data['lastStartedAt']
        save(STATE, pod)
        print(json.dumps({'status': data['desiredStatus'], 'ssh_ready': 'ssh_host' in pod,
                          'elapsed_minutes': round((time.time()-pod['created_epoch'])/60, 2)}))
    elif args.action == 'terminate':
        terminate('Profiling evidence transferred and checked')
    else:
        pod = json.loads(STATE.read_text())
        if pod.get('termination_verified'):
            raise RuntimeError('Profiling pod is terminated')
        cfg = config()
        key = os.path.expanduser(cfg.get('RUNPOD_SSH_KEY', '~/.ssh/id_ed25519'))
        opts = ['-i', key, '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=15']
        target = 'root@' + pod['ssh_host']
        if args.action == 'ssh':
            command = ['ssh', *opts, '-p', str(pod['ssh_port']), target, *args.arguments]
        elif args.action == 'put':
            command = ['scp', '-q', *opts, '-P', str(pod['ssh_port']), *args.arguments[:-1],
                       target + ':' + args.arguments[-1]]
        else:
            command = ['scp', '-q', *opts, '-P', str(pod['ssh_port']),
                       target + ':' + args.arguments[0], *args.arguments[1:]]
        raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
