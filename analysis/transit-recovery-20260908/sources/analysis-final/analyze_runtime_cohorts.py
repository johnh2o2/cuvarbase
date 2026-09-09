#!/usr/bin/env python3
"""Check flux-dependent runtime on all held-out cases measured on the original pod."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze import table


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    rows=[];ratios=[]
    for m in json.loads((r/'validation-methods.json').read_text())['methods']:
        # The serial GTLS cases were run on other machines and cannot supply same-host ratios.
        if m['tag']=='gtls':continue
        d=json.loads((r/'results'/f"heldout_{m['profile']}_{m['tag']}"/'summary.json').read_text())
        assert d['status']=='ok' and not d.get('partitioned_recovery')
        chunk=m['config'].get('eval_chunk',1);cases=sorted(d['cases'],key=lambda c:c['index']);stats={}
        for injected,label in [(True,'injected'),(False,'null')]:
            chosen=[c for c in cases if c['injected']==injected];assert len(chosen)==128
            assert all(c.get('api_result_valid',True) and c.get('finite_fraction',0.)>0. for c in chosen)
            times=np.array([c['search_s'] for c in chosen]);assert np.all(np.isfinite(times)&(times>0))
            # Each eval_chunk source group has one shared wall measurement; don't count
            # its repeated per-source value as independent timing repetitions.
            for j in range(0,len(chosen),chunk):
                assert len({c['search_s'] for c in chosen[j:j+chunk]})==1
            row=dict(profile=m['profile'],method=m['tag'],cohort=label,sources=128,independent_search_calls=128//chunk,
                sources_per_call=chunk,mean_seconds_per_source=float(times.mean()),median_seconds_per_source=float(np.median(times)),
                partial_spectra=sum(c.get('finite_fraction',0.)<1. for c in chosen),
                p10_seconds_per_source=float(np.quantile(times,.1)),p90_seconds_per_source=float(np.quantile(times,.9)))
            rows.append(row);stats[label]=row
        ratios.append(dict(profile=m['profile'],method=m['tag'],sources_per_call=chunk,
            injection_over_null_mean=stats['injected']['mean_seconds_per_source']/stats['null']['mean_seconds_per_source']))
    table(r/'runtime_by_cohort.csv',rows);table(r/'runtime_cohort_ratios.csv',ratios)
    result=dict(complete=True,cohorts=rows,ratios=ratios,
        note='Secondary runtime check from original-pod validation calls, after API warmup and excluding result compression/I/O. Independent distinct sources; one call per source or source group, not repeated timing trials. Serial GTLS recovery from auxiliary nodes is excluded. Main speed/cost ratios use the separate exclusive timing manifest.')
    (r/'runtime_cohort_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(ratios,indent=2))


if __name__=='__main__':main()
