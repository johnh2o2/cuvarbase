#!/usr/bin/env python3
"""Measure full-spectrum agreement for public operational batch choices."""
import argparse,json
from pathlib import Path
import numpy as np
from worker import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;rows=[]
    for profile in ['ztf','tess_gap']:
        rf=r/'results'/f'screen_{profile}_gtls_fast';ref=json.loads((rf/'summary.json').read_text());rc={c['index']:c for c in ref['cases']}
        for prefix,w in [('screen',2),('probe',2),('probe',4)]:
            f=r/'results'/f'{prefix}_{profile}_gtls_fast_w{w}';p=f/'summary.json'
            if not p.exists():continue
            d=json.loads(p.read_text());cases=[]
            for c in d['cases']:
                b=rc[c['index']];pp=f/c['output_file'];rp=rf/b['output_file'];assert sha(pp)==c['output_sha256'] and sha(rp)==b['output_sha256']
                with np.load(pp) as v,np.load(rp) as rv:
                    cases.append(dict(index=c['index'],candidate_equal=c['period']==b['period'],score_abs_difference=abs(c['score']-b['score']),
                        periods_equal=bool(np.array_equal(v['periods'],rv['periods'])),powers_equal=bool(np.array_equal(v['power'],rv['power'])),
                        max_abs_power_difference=float(np.max(np.abs(v['power']-rv['power'])))))
            rows.append(dict(profile=profile,workers=w,job=f.name,status=d['status'],seconds_per_source=d.get('seconds_per_source'),
                             oom_in_log='OutOfMemoryError' in (f/'stdout.log').read_text(),cases=cases))
    dump(r/'operational-probe-agreement.json',dict(rows=rows));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
