#!/usr/bin/env python3
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'scripts/benchmark_tls_profile')
import cloud
ap=argparse.ArgumentParser();ap.add_argument('name');a=ap.parse_args()
p=Path('analysis/transit-recovery-20260908/compute')/a.name;d=json.loads((p/'pod.json').read_text())
key=Path(os.path.expanduser(cloud.config().get('RUNPOD_SSH_KEY','~/.ssh/id_ed25519')))
pub=key.with_suffix(key.suffix+'.pub').read_text().strip()
commands="ssh-keygen -A >/dev/null 2>&1\nservice ssh start\nmkdir -p /root/.ssh\nchmod 700 /root/.ssh\nprintf '%s\\n' "+shlex.quote(pub)+" >> /root/.ssh/authorized_keys\nchmod 600 /root/.ssh/authorized_keys\necho SSHD_SETUP_DONE\nexit\n"
cmd=['ssh','-tt','-i',str(key),'-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','LogLevel=ERROR','-o','ConnectTimeout=15',d['proxy_host_id']+'@ssh.runpod.io']
r=subprocess.run(cmd,input=commands,text=True,capture_output=True,timeout=45)
(p/'ssh-bootstrap.log').write_text(r.stdout+r.stderr)
print(a.name,'SSH bootstrap',r.returncode,'marker', 'SSHD_SETUP_DONE' in r.stdout)
if r.returncode:raise SystemExit(r.returncode)
