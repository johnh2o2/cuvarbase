#!/usr/bin/env python3
"""Validate timing outputs against recovery runs; compute directly traceable ratios."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze import table,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--verify-arrays',action='store_true')
    a=ap.parse_args();r=a.root;declared=json.loads((r/'timing-methods.json').read_text())
    assert sha(r/'validation-methods.json')==declared['validation_manifest_sha256']
    rows=[];errors=[];agreements=[];repetitions=[]
    recovery=json.loads((r/'recovery_analysis.json').read_text())
    thresholds={(v['profile'],v['method']):v['threshold'] for v in recovery['methods']}
    jobs={j['name']:j for j in json.loads((r/'timings.json').read_text())}
    if (r/'cpu_operations.json').exists():jobs.update({j['name']:j for j in json.loads((r/'cpu_operations.json').read_text())})
    for m in declared['methods']:
        folder=r/'results'/m['job'];d=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'execution.json').read_text())
        if d['status']!='ok' or e['exit_code']!=0:errors.append('Unsuccessful timing '+m['job']);continue
        if d['config']!=m['config']:errors.append('Timing config mismatch '+m['job'])
        if sha(r/'inputs'/d['input_file'])!=d['input_sha256']:errors.append('Timing input mismatch '+m['job'])
        ref_tag=m['tag']+'_batch' if m['tag'] in ['bls_v1','gtls'] and m['mode']=='batch16' else m['tag']
        ref_folder=r/'results'/f"heldout_{m['profile']}_{ref_tag}"
        reference={c['index']:c for c in json.loads((ref_folder/'summary.json').read_text())['cases']}
        threshold=thresholds[m['profile'],ref_tag]
        if [c['index'] for c in d['cases']]!=list(range(128,128+m['n'])):errors.append('Timing source indices '+m['job'])
        expected_reps=jobs[m['job']]['reps']
        if len(d['times_s'])!=expected_reps or len(d['timed_candidates'])!=expected_reps:errors.append('Timing repetition count '+m['job'])
        for rep,values in enumerate(d['timed_candidates']):
            values=[values] if isinstance(values,dict) else values
            if len(values)!=m['n']:errors.append('Timed candidate count '+m['job']);continue
            for offset,c in enumerate(values):
                rc=reference[128+offset]
                valid=c.get('api_result_valid',True) and c.get('finite_fraction')==1. and c.get('period') is not None and c.get('score') is not None
                valid=valid and bool(np.isfinite(c['period']) and np.isfinite(c['score']))
                if not valid:errors.append('Invalid repetition '+m['job']+':'+str(rep))
                repetitions.append(dict(job=m['job'],rep=rep,index=128+offset,valid=valid,
                    candidate_numerically_equal=bool(valid and np.isclose(c['period'],rc['period'],rtol=1e-12,atol=0.)),
                    score_abs_difference=abs(c['score']-rc['score']) if valid else None,
                    calibrated_null_decision_equal=bool(valid and (c['score']>threshold)==(rc['score']>threshold))))
        for c in d['cases']:
            rc=reference[c['index']];valid=c.get('api_result_valid',True)
            agree=dict(job=m['job'],index=c['index'],valid=valid,
                       candidate_equal=c['period']==rc['period'],candidate_numerically_equal=bool(np.isclose(c['period'],rc['period'],rtol=1e-12,atol=0.)) if valid else False,score_abs_difference=abs(c['score']-rc['score']) if valid else None)
            agree['calibrated_null_decision_equal']=bool(valid and (c['score']>threshold)==(rc['score']>threshold))
            if not valid:errors.append('Invalid timed output '+m['job']+str(c['index']))
            if a.verify_arrays and c.get('output_file') and rc.get('output_file'):
                p=folder/c['output_file'];rp=ref_folder/rc['output_file']
                if sha(p)!=c['output_sha256'] or sha(rp)!=rc['output_sha256']:errors.append('Timing array hash '+str(p))
                with np.load(p) as v,np.load(rp) as rv:
                    agree['period_arrays_equal']=bool(np.array_equal(v['periods'],rv['periods']))
                    agree['power_arrays_equal']=bool(np.array_equal(v['power'],rv['power']))
                    agree['power_max_abs_difference']=float(np.max(np.abs(v['power']-rv['power'])))
            agreements.append(agree)
        t=np.array(d['times_s']);per=float(np.median(t))/m['n']
        if not np.all(np.isfinite(t)&(t>0)):errors.append('Invalid elapsed time '+m['job'])
        if not np.isclose(d['seconds_per_source'],per,rtol=1e-14):errors.append('Timing arithmetic '+m['job'])
        rows.append(dict(profile=m['profile'],method=m['tag'],family=m['family'],mode=m['mode'],n=m['n'],job=m['job'],
            median_total_s=float(np.median(t)),seconds_per_source=per,min_total_s=float(t.min()),max_total_s=float(t.max()),reps=len(t),
            initialization_s=d['initialization_s'],first_api_s=d['first_api_s'],
            projected_gpu_usd_per_million=per*1e6*.49/3600 if m['tag']!='bls_cpu' else None))
    by={(v['profile'],v['method'],v['mode']):v for v in rows};ratios=[]
    for profile in ['tess_200s','tess_gap','ztf']:
        for mode in ['single','batch16']:
            for v1,other in [('bls_v1','bls_pypi'),('bls_v1','bls_cpu'),('bls_v1','bls_gpu'),('tls_v1','gtls')]:
                aa=by[(profile,v1,mode)];bb=by[(profile,other,mode)];speed=bb['seconds_per_source']/aa['seconds_per_source']
                ratios.append(dict(profile=profile,mode=mode,v1=v1,comparator=other,speedup=speed,
                    cpu_break_even_hourly_usd=.49/speed if other=='bls_cpu' else None))
        aa=by[profile,'bls_v1','fresh_grid'];bb=by[profile,'bls_pypi','fresh_grid']
        ratios.append(dict(profile=profile,mode='fresh_grid',v1='bls_v1',comparator='bls_pypi',
            speedup=bb['seconds_per_source']/aa['seconds_per_source'],cpu_break_even_hourly_usd=None))
    result=dict(timings=rows,ratios=ratios,agreements=agreements,repetition_agreements=repetitions,verification=dict(errors=errors,arrays_verified=a.verify_arrays,complete=not errors),
        cost_note='GPU bundle rental-equivalent linear search-only projections at $0.49/hour. CPU break-even price is an estimate, not a measured standalone CPU rental.')
    table(r/'timing_summary.csv',rows);table(r/'speedups.csv',ratios)
    (r/'timing_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(timing_jobs=len(rows),errors=errors,candidate_disagreements=sum(not x['candidate_equal'] for x in agreements),
        repetition_candidate_disagreements=sum(not x['candidate_numerically_equal'] for x in repetitions),
        repetition_null_decision_disagreements=sum(not x['calibrated_null_decision_equal'] for x in repetitions)),indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
