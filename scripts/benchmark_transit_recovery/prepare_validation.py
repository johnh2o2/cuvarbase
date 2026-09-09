#!/usr/bin/env python3
"""Declare held-out jobs only after a configuration-selection artifact is frozen."""
import argparse,datetime,hashlib,json,random
from pathlib import Path

TAGS={'BLS v1':'bls_v1','BLS PyPI':'bls_pypi','BLS CPU':'bls_cpu','BLS GPU':'bls_gpu','TLS v1':'tls_v1','GTLS':'gtls'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    selection=json.loads((r/'selection.json').read_text());assert selection['frozen']
    if (r/'validation.json').exists():raise RuntimeError('Validation jobs already declared; do not overwrite')
    operations=json.loads((r/'operational-selection.json').read_text());methods=[];jobs=[]
    for profile,groups in selection['selected'].items():
        for family,item in groups.items():
            tag=TAGS[family];cfg=item['config'].copy()
            methods.append(dict(profile=profile,tag=tag,family=family,config=cfg,selected_from=item['job']))
            if family=='GTLS':
                batch=cfg.copy();batch.update(workers=operations['gtls_batch_workers'][profile],eval_chunk=16)
                methods.append(dict(profile=profile,tag=tag+'_batch',family=family,config=batch,selected_from=item['job'],
                                    purpose='Separate recovery calibration for concurrent GTLS; memory-dependent chunking can alter the spectrum'))
            if family=='BLS v1':
                batch=cfg.copy();batch.update(backend='v1_bls_batch',reuse_batch=True,batch_capacity=16,eval_chunk=16)
                methods.append(dict(profile=profile,tag=tag+'_batch',family=family,config=batch,selected_from=item['job'],
                                    purpose='Validate the public batch implementation at the same selected scientific settings'))
    random.Random(744219).shuffle(methods)
    for method in methods:
        for split in ['calibration','heldout']:
            jobs.append(dict(name=f"{split}_{method['profile']}_{method['tag']}",input=f"{method['profile']}_{split}.npz",
                             config=method['config'],indices='all',timeout=14000))
    (r/'validation-methods.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection_sha256=hashlib.sha256((r/'selection.json').read_bytes()).hexdigest(),operational_selection_sha256=hashlib.sha256((r/'operational-selection.json').read_bytes()).hexdigest(),methods=methods),indent=2)+'\n')
    (r/'validation.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Declared',len(jobs),'jobs for',len(methods),'method/profile combinations')


if __name__=='__main__':main()
