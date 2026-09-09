#!/usr/bin/env python3
"""Download the original node only after every performance job has finished."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    env=os.environ.copy();env.pop('CUVARBASE_BENCHMARK_POD_DIR',None)
    scripts=Path(__file__).resolve().parent;cloud=scripts.parent/'benchmark_tls_profile/cloud.py'
    program="""import json
from pathlib import Path
r=Path('/tmp/cuvarbase-tls-profile/recovery');out={}
for name in ['validation_main','timings','components','cpu_operations']:
 p=r/(name+'.execution.json');d=json.loads(p.read_text()) if p.exists() else []
 out[name]=dict(completed=len(d),planned=len(json.loads((r/(name+'.json')).read_text())),failed=[v['name'] for v in d if v['exit_code']!=0])
print(json.dumps(out))
"""
    last=None
    while True:
        v=subprocess.run([sys.executable,str(cloud),'ssh','python -c '+shlex.quote(program)],env=env,capture_output=True,text=True)
        if v.returncode:print('Original status transport retry',flush=True);time.sleep(20);continue
        state=json.loads(v.stdout)
        if state!=last:print(json.dumps(state),flush=True);last=state
        assert not any(v['failed'] for v in state.values()),state
        if all(v['completed']==v['planned'] for v in state.values()):break
        time.sleep(20)
    command='LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1 /tmp/cuvarbase-tls-profile/modern/bin/python /tmp/cuvarbase-tls-profile/recovery/scripts/collect_evidence.py --manifest-only --node original'
    subprocess.run([sys.executable,str(cloud),'ssh',command],env=env,check=True)
    subprocess.run([sys.executable,str(scripts/'download_evidence.py'),'--root',str(r),'--node','original'],env=env,check=True)
    print('ORIGINAL_EVIDENCE_DOWNLOADED_AND_TRANSFER_HASHES_VERIFIED; retain pod until analysis verification succeeds.',flush=True)


if __name__=='__main__':main()
