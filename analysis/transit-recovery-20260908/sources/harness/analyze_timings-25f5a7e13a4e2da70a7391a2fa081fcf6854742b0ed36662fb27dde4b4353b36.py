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
    rows=[];errors=[];agreements=[]
    for m in declared['methods']:
        folder=r/'results'/m['job'];d=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'execution.json').read_text())
        if d['status']!='ok' or e['exit_code']!=0:errors.append('Unsuccessful timing '+m['job']);continue
        if d['config']!=m['config']:errors.append('Timing config mismatch '+m['job'])
        if sha(r/'inputs'/d['input_file'])!=d['input_sha256']:errors.append('Timing input mismatch '+m['job'])
        ref_tag='bls_v1_batch' if m['tag']=='bls_v1' and m['mode']=='batch16' else m['tag']
        ref_folder=r/'results'/f"heldout_{m['profile']}_{ref_tag}"
        reference={c['index']:c for c in json.loads((ref_folder/'summary.json').read_text())['cases']}
        if [c['index'] for c in d['cases']]!=list(range(128,128+m['n'])):errors.append('Timing source indices '+m['job'])
        for c in d['cases']:
            rc=reference[c['index']];valid=c.get('api_result_valid',True)
            agree=dict(job=m['job'],index=c['index'],valid=valid,
                       candidate_equal=c['period']==rc['period'],score_abs_difference=abs(c['score']-rc['score']) if valid else None)
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
    result=dict(timings=rows,ratios=ratios,agreements=agreements,verification=dict(errors=errors,complete=not errors),
        cost_note='GPU bundle rental-equivalent linear search-only projections at $0.49/hour. CPU break-even price is an estimate, not a measured standalone CPU rental.')
    table(r/'timing_summary.csv',rows);table(r/'speedups.csv',ratios)
    (r/'timing_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(timing_jobs=len(rows),errors=errors,candidate_disagreements=sum(not x['candidate_equal'] for x in agreements)),indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
