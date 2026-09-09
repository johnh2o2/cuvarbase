#!/usr/bin/env python3
"""Summarize synchronized wall phases and numerical effects of diagnostic ablations."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze import sha,table


def candidate(d):return d['native_candidate']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    declared=json.loads((r/'components.json').read_text());records={};phases=[];summaries=[];errors=[]
    for job in declared:
        folder=r/'results'/job['name'];d=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'execution.json').read_text())
        if d['status']!='ok' or e['exit_code']!=0:errors.append('Failed component job '+job['name']);continue
        if sha(r/'inputs'/job['input'])!=d['input_sha256']:errors.append('Component input hash '+job['name'])
        if sha(folder/'outputs.npz')!=d['output_sha256']:errors.append('Component output hash '+job['name'])
        records[job['name']]=d;profiles=[v.get('phases',v) for v in d['profiles']]
        totals=[sum(p['exclusive_s'] for p in profile.values()) for profile in profiles]
        names=sorted(set(k for profile in profiles for k in profile))
        for name in names:
            values=[profile.get(name,{}).get('exclusive_s',0.) for profile in profiles]
            phases.append(dict(job=job['name'],profile=d['profile'],phase=name,exclusive_mean_s=float(np.mean(values)),
                               calls_mean=float(np.mean([profile.get(name,{}).get('calls',0) for profile in profiles])),
                               fraction_of_profile=float(np.mean(values))/float(np.mean(totals))))
        with np.load(folder/'outputs.npz') as v:
            instrumentation_delta=float(np.max(np.abs(v['native']-v['profiled'])))
        summaries.append(dict(job=job['name'],profile=d['profile'],native_median_s=d['native_median_s'],
            synchronized_profile_mean_s=float(np.mean(totals)),profile_over_native=float(np.mean(totals))/d['native_median_s'],
            instrumentation_max_abs_power_difference=instrumentation_delta,
            grid_median_s=d.get('grid_median_s'),grid_float32_equal=d.get('grid_float32_equal')))
    ablations=[]
    for profile in ['tess_200s','tess_gap','ztf']:
        for base,variant,meaning in [(f'component_{profile}_bls_v1',f'component_{profile}_bls_v1_unfused','Public unfused phase passes versus fused histogram'),
                                      (f'component_{profile}_bls_v1',f'component_{profile}_bls_v1_no_scatter','Diagnostic chronological input versus conflict-scattered observation order'),
                                      (f'component_{profile}_gtls_native',f'component_{profile}_gtls_both','Diagnostic batching of two GTLS host loops')]:
            aa,bb=records[base],records[variant]
            with np.load(r/'results'/base/'outputs.npz') as av,np.load(r/'results'/variant/'outputs.npz') as bv:
                equal=bool(np.array_equal(av['periods'],bv['periods']));assert equal
                delta=float(np.max(np.abs(av['native']-bv['native'])));exact=bool(np.array_equal(av['native'],bv['native']))
            ca,cb=candidate(aa),candidate(bb)
            ablations.append(dict(profile=profile,baseline=base,variant=variant,meaning=meaning,
                baseline_s=aa['native_median_s'],variant_s=bb['native_median_s'],variant_over_baseline=bb['native_median_s']/aa['native_median_s'],
                periods_equal=equal,powers_equal=exact,max_abs_power_difference=delta,candidate_period_equal=ca['period']==cb['period'],
                candidate_score_abs_difference=abs(ca['score']-cb['score']),power_units='native SDE' if aa['config'].get('fast') else 'BLS chi2 ratio'))
    table(r/'component_summary.csv',summaries);table(r/'component_phases.csv',phases);table(r/'component_ablations.csv',ablations)
    result=dict(components=summaries,phases=phases,ablations=ablations,verification=dict(complete=not errors,errors=errors),
        note='Synchronized wall regions perturb execution; attribution uses profile fractions alongside ordinary uninstrumented API timings. Ablations are diagnostic, not released competitors, and their gains cannot simply be multiplied.')
    (r/'component_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(jobs=len(records),ablations=len(ablations),errors=errors),indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
