#!/usr/bin/env python3
"""Declare explanatory profiles/ablations separately from the main competitor."""
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    selection=json.loads((r/'selection.json').read_text());assert selection['frozen'];jobs=[]
    for p,groups in selection['selected'].items():
        for fam,tag in [('BLS v1','bls_v1'),('BLS PyPI','bls_pypi')]:
            cfg=groups[fam]['config'].copy()
            jobs.append(dict(name=f'component_{p}_{tag}',input=f'{p}_tune.npz',config=cfg,script='components.py',timeout=400))
        cfg=groups['BLS v1']['config'].copy();cfg['unfused']=True
        jobs.append(dict(name=f'component_{p}_bls_v1_unfused',input=f'{p}_tune.npz',config=cfg,script='components.py',timeout=400))
        cfg=groups['BLS v1']['config'].copy();cfg['no_scatter']=True
        jobs.append(dict(name=f'component_{p}_bls_v1_no_scatter',input=f'{p}_tune.npz',config=cfg,script='components.py',timeout=400))
        for fam,tag in [('TLS v1','tls_v1'),('GTLS','gtls')]:
            for variant in (['native','both'] if fam=='GTLS' else ['native']):
                jobs.append(dict(name=f'component_{p}_{tag}_{variant}',input=f'{p}_tune.npz',config=groups[fam]['config'],
                                 script='components_tls.py',variant=variant,timeout=600))
    (r/'components.json').write_text(json.dumps(jobs,indent=2)+'\n');print('Declared',len(jobs),'profiles and ablations')


if __name__=='__main__':main()
