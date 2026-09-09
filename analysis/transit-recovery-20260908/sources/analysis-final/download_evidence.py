#!/usr/bin/env python3
"""Stream a finished node's evidence without doubling remote disk use."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--node',choices=['original','validation_a','validation_b'],required=True);a=ap.parse_args()
    r=a.root;folder=r/'compute'/a.node;folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    if a.node!='original':env['CUVARBASE_BENCHMARK_POD_DIR']=str(folder)
    else:env.pop('CUVARBASE_BENCHMARK_POD_DIR',None)
    cloud=Path(__file__).resolve().parents[1]/'benchmark_tls_profile/cloud.py'
    command='cd /tmp/cuvarbase-tls-profile/recovery && tar --null -T transfer-'+a.node+'.files0 -cf -'
    archive=folder/'evidence.tar';partial=folder/'evidence.tar.partial';start=time.time()
    if not archive.exists():
        print('Streaming',a.node,flush=True)
        with partial.open('wb') as out,(folder/'stream.stderr.log').open('w') as err:
            subprocess.run([sys.executable,str(cloud),'ssh',command],env=env,stdout=out,stderr=err,check=True)
        partial.replace(archive)
    dest=folder/'evidence';dest.mkdir(exist_ok=True)
    print('Extracting',a.node,archive.stat().st_size,'bytes',flush=True)
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    transfer=json.loads((dest/f'transfer-{a.node}.json').read_text());assert transfer['node']==a.node
    for name,digest in transfer['files'].items():assert sha(dest/name)==digest,(a.node,name)
    print('Verified every file',a.node,len(transfer['files']),flush=True)
    # Keep the intact per-node artifact. Link analysis inputs to authoritative outputs.
    for subfolder in ['results','sources/harness']:
        source=dest/subfolder
        if not source.exists():continue
        for p in source.rglob('*'):
            if not p.is_file():continue
            target=r/subfolder/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.importing')
            if temp.exists():temp.unlink()
            os.link(p,temp);temp.replace(target)
    record=dict(node=a.node,archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),
        files_verified=len(transfer['files']),elapsed_transfer_verification_s=time.time()-start,
        complete=True,note='Tar streamed from the finished node; each extracted file independently verified against its remote SHA256 manifest.')
    (folder/'download-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
