#!/usr/bin/env python3
"""Select configurations exclusively from complete tuning-injection results."""
import argparse,datetime,json
from pathlib import Path
import numpy as np


def family(config):
    b=config['backend']
    if b.startswith('v1_bls'):return 'BLS v1'
    if b.startswith('pypi_bls'):return 'BLS PyPI'
    if b in ['astropy','periodfind_cpu','fbls']:return 'BLS CPU'
    if b=='periodfind_gpu':return 'BLS GPU'
    if b=='v1_tls':return 'TLS v1'
    if b=='gtls':return 'GTLS'
    raise ValueError(b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--freeze',action='store_true');ap.add_argument('--use-repeats',action='store_true')
    a=ap.parse_args();r=a.root;groups={};excluded=[]
    for p in sorted((r/'results').glob('*/summary.json')):
        if not p.parent.name.startswith(('tune_','screen_')):continue
        d=json.loads(p.read_text())
        if d['split']!='tune' or 'times_s' in d:continue
        cases=d['cases'];good=d['status']=='ok' and len(cases)==32 and all(c['injected'] for c in cases)
        execution=p.with_name('execution.json')
        good=good and execution.exists() and json.loads(execution.read_text()).get('exit_code')==0
        good=good and all(c.get('api_result_valid',True) and c.get('finite_fraction',0)==1. for c in cases)
        if not good:
            excluded.append(dict(job=p.parent.name,status=d['status'],n=len(cases),reason='Incomplete, nonfinite, failed, or fewer than 32 tuning injections'))
            continue
        item=dict(job=p.parent.name,config=d['config'],recovered=sum(c['recovered'] for c in cases),n=32,
            mean_s=float(np.mean([c['search_s'] for c in cases])),median_s=float(np.median([c['search_s'] for c in cases])),
            source=str(p.relative_to(r)),worker_sha256=d['worker_sha256'])
        key=(d['profile'],family(d['config']));groups.setdefault(key,[]).append(item)
    selected={};allgroups=[]
    for (profile,fam),items in sorted(groups.items()):
        best=max(i['recovered'] for i in items);eligible=[i for i in items if i['recovered']>=best-1]
        close=[i for i in eligible if i['mean_s']<=1.2*min(x['mean_s'] for x in eligible)]
        if a.use_repeats and len(close)>1:
            for item in close:
                rp=r/'results'/('repeat_'+item['job'])/'summary.json'
                rd=json.loads(rp.read_text());execution=json.loads(rp.with_name('execution.json').read_text())
                assert rd['status']=='ok' and execution['exit_code']==0 and rd['config']==item['config']
                item['repeat_seconds_per_source']=rd['seconds_per_source'];item['repeat_source']=str(rp.relative_to(r))
            winner=min(close,key=lambda i:i['repeat_seconds_per_source'])
        else:winner=min(eligible,key=lambda i:i['mean_s'])
        selected.setdefault(profile,{})[fam]=winner
        allgroups.append(dict(profile=profile,family=fam,best_tuning_recovery=best,selected=winner,
                              close_timing_candidates=[i for i in eligible if i['mean_s']<=1.2*winner['mean_s']],all_candidates=items))
    name='selection.json' if a.freeze else 'selection-preview.json'
    if a.freeze and (r/name).exists():raise RuntimeError('Selection is already frozen; do not overwrite after held-out inspection')
    out=dict(frozen=a.freeze,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             rule='Fastest complete setting within one recovered tuning injection of best in its family; close timing choices repeated when requested; no held-out results read.',used_repeats=a.use_repeats,
             selected=selected,groups=allgroups,excluded=excluded)
    (r/name).write_text(json.dumps(out,indent=2)+'\n')
    for g in allgroups:
        w=g['selected'];print(g['profile'],g['family'],w['job'],w['recovered'],round(w['mean_s'],5),
                               'close',len(g['close_timing_candidates']))


if __name__=='__main__':main()
