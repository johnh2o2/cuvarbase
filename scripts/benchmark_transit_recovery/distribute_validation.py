#!/usr/bin/env python3
"""Distribute independent serial-GTLS recovery cases; timing remains on original pod."""
import argparse,hashlib,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    jobs=json.loads((r/'validation.json').read_text());main_jobs=[];chunks=[];parents=[]
    for j in jobs:
        if j['config']['backend']!='gtls' or not j['name'].endswith('_gtls'):
            main_jobs.append(j);continue
        n=128 if j['name'].startswith('calibration_') else 256;part_names=[]
        for start in range(0,n,64):
            part=j.copy();part['name']=j['name']+f'__part{start:03}';part['indices']=','.join(map(str,range(start,start+64)))
            profile=j['input'].split('_calibration')[0].split('_heldout')[0]
            estimate={'ztf':18.,'tess_gap':7.6,'tess_200s':.5}[profile]*64
            chunks.append((estimate,part));part_names.append(part['name'])
        parents.append(dict(parent=j['name'],parts=part_names,n=n))
    assignment={'validation_a':[],'validation_b':[]};loads={k:0. for k in assignment}
    for estimate,part in sorted(chunks,key=lambda x:-x[0]):
        dest=min(loads,key=loads.get);assignment[dest].append(part);loads[dest]+=estimate
    (r/'validation_main.json').write_text(json.dumps(main_jobs,indent=2)+'\n')
    for name,parts in assignment.items():(r/(name+'.json')).write_text(json.dumps(parts,indent=2)+'\n')
    out=dict(validation_sha256=hashlib.sha256((r/'validation.json').read_bytes()).hexdigest(),parents=parents,
             assignment={k:[j['name'] for j in v] for k,v in assignment.items()},estimated_compute_s=loads,
             note='Only recovery cases distributed; full inputs, spectra, source hashes and per-part hardware provenance retained. No distributed evaluation times enter speed ratios.')
    (r/'validation-distribution.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Main jobs',len(main_jobs),'auxiliary parts', {k:len(v) for k,v in assignment.items()},'estimated minutes',{k:round(v/60,1) for k,v in loads.items()})


if __name__=='__main__':main()
