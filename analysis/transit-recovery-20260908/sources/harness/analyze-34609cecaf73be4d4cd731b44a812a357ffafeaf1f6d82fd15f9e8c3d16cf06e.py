#!/usr/bin/env python3
"""Verify held-out bookkeeping and derive recovery, FPR and paired comparisons."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from recovery_statistics import paired,summarize


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--verify-arrays',action='store_true')
    a=ap.parse_args();r=a.root
    declared=json.loads((r/'validation-methods.json').read_text())
    assert declared['selection_sha256']==sha(r/'selection.json')
    records={};summaries=[];errors=[];checked=0;snrrows=[]
    for m in declared['methods']:
        profile,tag=m['profile'],m['tag'];parts={}
        for split in ['calibration','heldout']:
            folder=r/'results'/f'{split}_{profile}_{tag}';p=folder/'summary.json'
            if not p.exists():errors.append('Missing job '+str(folder));continue
            d=json.loads(p.read_text());e=json.loads((folder/'execution.json').read_text())
            if d['status']!='ok' or e['exit_code']!=0:errors.append('Incomplete job '+str(folder));continue
            inp=r/'inputs'/d['input_file']
            if sha(inp)!=d['input_sha256']:errors.append('Input hash '+str(inp))
            with np.load(inp) as data:meta=json.loads(str(data['metadata']))
            cases=sorted(d['cases'],key=lambda c:c['index'])
            if [c['index'] for c in cases]!=list(range(len(meta['cases']))):errors.append('Incomplete case indices '+str(folder));continue
            for c in cases:
                truth=meta['cases'][c['index']]
                if c['injected']!=truth['injected'] or c['snr']!=truth['target_white_oracle_snr']:errors.append('Truth metadata mismatch '+str(folder))
                expected=bool(c['period'] is not None and abs(c['period']/truth['period']-1)*meta['baseline']<=.5*truth['duration']) if truth['injected'] else None
                if expected!=c['recovered']:errors.append('Recovery classification '+str(folder)+str(c['index']))
                if c.get('output_file'):
                    out=folder/c['output_file']
                    if a.verify_arrays:
                        if sha(out)!=c['output_sha256']:errors.append('Output hash '+str(out))
                        with np.load(out) as v:
                            if len(v['periods'])!=c['n_periods'] or v['periods'].shape!=v['power'].shape:errors.append('Output shape '+str(out))
                            finite=float(np.isfinite(v['power']).mean()) if len(v['power']) else 0.
                            if abs(finite-c['finite_fraction'])>1e-12:errors.append('Finite coverage '+str(out))
                            checked+=1
            parts[split]=cases
        if len(parts)!=2:continue
        summary=dict(profile=profile,method=tag,family=m['family'],config=m['config'],**summarize(parts['calibration'],parts['heldout']))
        records[(profile,tag)]=summary;summaries.append(summary)
        for row in summary['by_snr']:snrrows.append(dict(profile=profile,method=tag,**row))
    comparisons=[]
    for profile in ['ztf','tess_200s','tess_gap']:
        for v1,competitor in [('bls_v1','bls_pypi'),('bls_v1','bls_cpu'),('bls_v1','bls_gpu'),('bls_v1_batch','bls_pypi'),
                              ('bls_v1_batch','bls_cpu'),('bls_v1_batch','bls_gpu'),('tls_v1','gtls')]:
            if (profile,v1) not in records or (profile,competitor) not in records:continue
            av,bv=records[(profile,v1)],records[(profile,competitor)]
            period=paired(av['period_recovered_vector'],bv['period_recovered_vector'])
            detection=paired(av['detected_vector'],bv['detected_vector'])
            falsepositive=paired(av['false_positive_vector'],bv['false_positive_vector'])
            valid=all(x['invalid_calibration']==x['invalid_heldout']==0 for x in [av,bv])
            comparisons.append(dict(profile=profile,v1=v1,comparator=competitor,all_api_results_valid=valid,
                period_noninferior=valid and period['noninferior_5pp'],detection_noninferior=valid and detection['noninferior_5pp'],
                fpr_increase_upper_95=falsepositive['upper_95_one_sided'],
                comparable_detection=valid and detection['noninferior_5pp'] and falsepositive['upper_95_one_sided']<.05,
                period_comparison=period,detection_comparison=detection,false_positive_comparison=falsepositive))
    compact=[]
    for s in summaries:
        compact.append({k:v for k,v in s.items() if k not in ['config','by_snr'] and not k.endswith('_vector')})
    table(r/'recovery_summary.csv',compact);table(r/'recovery_by_snr.csv',snrrows)
    table(r/'paired_comparisons.csv',comparisons)
    result=dict(methods=summaries,comparisons=comparisons,verification=dict(errors=errors,output_arrays_checked=checked,complete=not errors))
    (r/'recovery_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(methods=len(summaries),comparisons=len(comparisons),errors=errors,output_arrays_checked=checked),indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
