#!/usr/bin/env python3
"""Collect, verify and terminate one auxiliary pod once its frozen jobs finish."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--node',choices=['validation_a','validation_b'],required=True);a=ap.parse_args();r=a.root
    env=os.environ.copy();env['CUVARBASE_BENCHMARK_POD_DIR']=str(r/'compute'/a.node)
    cloud=Path(__file__).resolve().parents[1]/'benchmark_tls_profile/cloud.py';scripts=Path(__file__).resolve().parent
    command=("import json;from pathlib import Path;r=Path('/tmp/cuvarbase-tls-profile/recovery');"
        "p=r/"+repr(a.node+'.execution.json')+";d=json.loads(p.read_text()) if p.exists() else [];"
        "print(json.dumps(dict(completed=len(d),failed=[v['name'] for v in d if v['exit_code']!=0])))")
    last=None
    while True:
        v=subprocess.run([sys.executable,str(cloud),'ssh','python -c '+shlex.quote(command)],env=env,capture_output=True,text=True)
        if v.returncode:print('Status transport retry',a.node,flush=True);time.sleep(20);continue
        state=json.loads(v.stdout)
        if state!=last:print(a.node,state,flush=True);last=state
        assert not state['failed'],state
        if state['completed']==9:break
        time.sleep(20)
    command='LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1 /tmp/cuvarbase-tls-profile/modern/bin/python /tmp/cuvarbase-tls-profile/recovery/scripts/collect_evidence.py --manifest-only --node '+a.node
    subprocess.run([sys.executable,str(cloud),'ssh',command],env=env,check=True)
    for name in ['download_evidence.py','verify_validation_node.py']:
        subprocess.run([sys.executable,str(scripts/name),'--root',str(r),'--node',a.node],env=env,check=True)
    subprocess.run([sys.executable,str(cloud),'terminate'],env=env,check=True)
    print('NODE_EVIDENCE_VERIFIED_AND_RENTAL_TERMINATED',a.node,flush=True)


if __name__=='__main__':main()
