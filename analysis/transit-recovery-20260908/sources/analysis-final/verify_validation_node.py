#!/usr/bin/env python3
"""Verify a collected auxiliary node before releasing its rental."""
import argparse,json
from pathlib import Path
import numpy as np
from verify_provenance import archive_sources,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--node',choices=['validation_a','validation_b'],required=True);a=ap.parse_args();r=a.root
    folder=r/'compute'/a.node;dest=folder/'evidence'
    assert json.loads((folder/'download-verification.json').read_text())['complete']
    freeze=json.loads((r/'harness-freeze.json').read_text())['files']
    for script in ['worker.py','controller.py']:assert sha(dest/'scripts'/script)==freeze[script]
    expected=archive_sources(dest/'sources/runtime/gtls-head.tar','src/gputls/');assert len(expected)>10
    jobs=json.loads((r/(a.node+'.json')).read_text());assert len(jobs)==9
    checked=[]
    for job in jobs:
        p=dest/'results'/job['name'];d=json.loads((p/'summary.json').read_text());e=json.loads((p/'execution.json').read_text())
        assert d['status']=='ok' and e['exit_code']==0 and not e['timeout'],job['name']
        assert d['config']==job['config'] and d['worker_sha256']==freeze['worker.py']
        assert d['input_sha256']==sha(r/'inputs'/job['input'])==sha(dest/'inputs'/job['input'])
        ids=[int(i) for i in job['indices'].split(',')];assert ids==d['indices']==[c['index'] for c in d['cases']]
        for c in d['cases']:
            assert c['api_result_valid'] and c['finite_fraction']==1. and np.isfinite(c['period']) and np.isfinite(c['score']), (job['name'],c['index'])
            assert sha(p/c['output_file'])==c['output_sha256']
        actual=d['installed_sources']['gputls']
        for name,digest in expected.items():
            if name in ['GPUFun.cu','GPUFun_bak.cu'] and name not in actual:continue
            assert actual.get(name)==digest,(job['name'],name)
        checked.append(dict(job=job['name'],cases=len(d['cases'])))
    result=dict(complete=True,node=a.node,jobs=checked,cases=sum(j['cases'] for j in checked),
        note='Verified complete planned partitions, successful APIs, frozen worker/controller, pinned installed GTLS sources, identical input hashes and every retained output hash before rental termination.')
    (folder/'node-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
