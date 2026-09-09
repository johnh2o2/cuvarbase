#!/usr/bin/env python3
"""Run a declared manifest sequentially, with process-group timeouts and checkpoints."""
import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path('/tmp/cuvarbase-tls-profile')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('--after');a=ap.parse_args()
    if a.after:
        previous=Path(a.after);expected=len(json.loads(previous.read_text()))
        while True:
            try:
                if len(json.loads(previous.with_suffix('.execution.json').read_text()))==expected:break
            except (FileNotFoundError,json.JSONDecodeError):pass
            time.sleep(5)
    p=Path(a.manifest);jobs=json.loads(p.read_text());records=[]
    for job in jobs:
        out=ROOT/'recovery/results'/job['name'];out.mkdir(parents=True,exist_ok=True)
        if (out/'execution.json').exists():
            old=json.loads((out/'execution.json').read_text())
            if old.get('exit_code')==0:records.append(old);continue
        cfg=job['config'];env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='7',
                   RAYON_NUM_THREADS=str(cfg.get('threads',7)),CUDA_HOME='/usr/local/cuda',
                   PATH='/usr/local/cuda/bin:'+env['PATH'],PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
        if cfg.get('gtls_version')=='head':env['PYTHONPATH']=str(ROOT/'gtls-head-install')
        else:env.pop('PYTHONPATH',None)
        python=ROOT/('legacy' if cfg['backend'].startswith('pypi') else 'modern')/'bin/python'
        cmd=[str(python),str(ROOT/'recovery/scripts/worker.py'),'--input',str(ROOT/'recovery/inputs'/job['input']),
             '--config',json.dumps(cfg),'--indices',job.get('indices','all'),'--out',str(out)]
        if job.get('timing'):cmd+=['--timing','--reps',str(job.get('reps',3))]
        start=time.time();print('START',job['name'],datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
        with (out/'stdout.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
            try:code=proc.wait(timeout=job.get('timeout',1800));timeout=False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:code=proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
                timeout=True
        record=dict(name=job['name'],exit_code=code,timeout=timeout,elapsed_s=time.time()-start,command=cmd,
                    started_epoch=start,finished_epoch=time.time())
        (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
        p.with_suffix('.execution.json').write_text(json.dumps(records,indent=2)+'\n')
        print('END',job['name'],code,round(record['elapsed_s'],2),flush=True)
    print('MANIFEST_COMPLETE',p.name,flush=True)


if __name__=='__main__':main()
