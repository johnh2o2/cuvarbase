#!/usr/bin/env python3
"""Declare randomized, exclusive single-source and 16-source timing jobs."""
import argparse,datetime,hashlib,json,random
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    p=r/'validation-methods.json';declared=json.loads(p.read_text())
    if (r/'timings.json').exists():raise RuntimeError('Timing manifest already exists')
    jobs=[];methods=[]
    for m in declared['methods']:
        if m['tag'] in ['bls_v1_batch','gtls_batch']:continue
        for mode in ['single','batch16']:
            cfg=m['config'].copy();tag=m['tag'];n=1 if mode=='single' else 16
            if tag=='bls_v1' and mode=='batch16':
                cfg=next(x['config'].copy() for x in declared['methods'] if x['profile']==m['profile'] and x['tag']=='bls_v1_batch')
            if tag=='gtls' and mode=='batch16':
                cfg=next(x['config'].copy() for x in declared['methods'] if x['profile']==m['profile'] and x['tag']=='gtls_batch')
            # Worker count is a measured operational choice, independent of physics settings.
            if tag=='gtls' and mode=='single':cfg['workers']=1
            cfg.pop('eval_chunk',None)
            name=f"timing_{m['profile']}_{tag}_{mode}"
            jobs.append(dict(name=name,input=m['profile']+'_heldout.npz',config=cfg,
                             indices=','.join(map(str,range(128,128+n))),timing=True,reps=5 if n==1 else 3,timeout=2200))
            methods.append(dict(profile=m['profile'],tag=tag,family=m['family'],mode=mode,n=n,job=name,config=cfg))
    random.Random(936811).shuffle(jobs)
    (r/'timings.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (r/'timing-methods.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),methods=methods,
        boundary='Prepared host arrays and explicit grid to host spectra and ranked candidate. Grid generation, imports, disk I/O and preprocessing excluded.',
        projected_gpu_hourly_usd=.49,batch_note='16 distinct independent null sources. Costs per million are linear projections, not measured full-pipeline costs.'),indent=2)+'\n')
    print('Declared',len(jobs),'exclusive timing jobs')


if __name__=='__main__':main()
