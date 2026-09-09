#!/usr/bin/env python3
"""Profile the selected TLS API configuration on the new observed cadences."""
import argparse,importlib,json,sys,time
from pathlib import Path
import numpy as np
from worker import Backend,sha,dump
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tls_profile'))
from profile_tls import Profiler,rebuild,compare


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--variant',default='native')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);cfg=json.loads(a.config);cfg['workers']=1
    d=np.load(a.input);meta=json.loads(str(d['metadata']));lc=tuple(np.array(d[f'{k}_2']) for k in ['t','y','dy'])
    b=Backend(cfg,d,len(lc[0]));transforms=[];path=a.out/'profile.json'
    if cfg['backend']=='gtls':
        core=importlib.import_module('gputls.core')
        if a.variant!='native':transforms.append(rebuild(core,'search_multi_periods',path,variant=a.variant))
    b.search([lc]);times=[]
    for rep in range(3):
        b.sync();start=time.perf_counter();native=b.search([lc])[0];b.sync();times.append(time.perf_counter()-start)
    profiler=Profiler(b.sync)
    if cfg['backend']=='gtls':
        from gputls import gtls
        transforms.append(rebuild(core,'search_multi_periods',path,instrument=True,kind='gtls',profiler=profiler))
        transforms.append(rebuild(gtls,'power',path,instrument=True,kind='power',profiler=profiler))
    else:
        from cuvarbase import tls
        transforms.append(rebuild(tls,'tls_search_batch',path,instrument=True,kind='v1',profiler=profiler));b.tls=tls.tls_search_batch
    profiles=[]
    for rep in range(2):
        profiler.clear()
        with profiler.segment('API remainder'):
            out=b.search([lc])[0]
        profiles.append(dict(phases=profiler.rows,total_s=sum(v['exclusive_s'] for v in profiler.rows.values())))
    np.savez_compressed(a.out/'outputs.npz',periods=native['periods'],native=native['power'],profiled=out['power'])
    dump(a.out/'summary.json',dict(status='ok',profile=meta['profile'],config=cfg,variant=a.variant,
        input_sha256=sha(a.input),worker_sha256=sha(__file__),instrumentation_sha256=sha('/tmp/cuvarbase-tls-profile/profile_tls.py'),
        native_times_s=times,native_median_s=float(np.median(times)),profiles=profiles,transformations=transforms,
        native_candidate=native['candidate'],profiled_candidate=out['candidate'],
        output_agreement=compare({'periods':native['periods'],'power':native['power']},{'periods':out['periods'],'power':out['power']}),
        output_sha256=sha(a.out/'outputs.npz'),meaning='Synchronized wall phases in one worker. Diagnostic Python ablations are not released GTLS.'))
    print('TLS_COMPONENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
