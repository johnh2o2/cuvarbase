#!/usr/bin/env python3
"""Resume a retained tar prefix with balanced parallel, SHA256-verified streams."""
import argparse,concurrent.futures,json,os,shlex,subprocess,sys,tarfile,time
from pathlib import Path
from download_evidence import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--streams',type=int,default=8);a=ap.parse_args();r=a.root
    folder=r/'compute/original';dest=folder/'evidence';dest.mkdir(exist_ok=True)
    env=os.environ.copy();env.pop('CUVARBASE_BENCHMARK_POD_DIR',None)
    cloud=Path(__file__).resolve().parents[1]/'benchmark_tls_profile/cloud.py';start=time.time()
    partial=folder/'evidence.tar.partial';prefix=folder/'evidence-prefix.tar'
    if partial.exists() and not prefix.exists():
        size=partial.stat().st_size;end=0;count=0
        with tarfile.open(partial,'r:') as t:
            try:
                for m in t:
                    if m.offset_data+m.size>size:break
                    t.extract(m,dest,filter='data');count+=1
                    end=m.offset_data+((m.size+511)//512)*512
            except tarfile.ReadError:pass
        assert end>0
        with partial.open('r+b') as f:f.truncate(end);f.seek(end);f.write(bytes(1024))
        partial.replace(prefix);print('Retained valid prefix',count,'members',end,'bytes',flush=True)
    transfer=json.loads((r/'transfer-original.json').read_text());assert transfer['node']=='original'
    remaining=[];retained=0
    for name,digest in transfer['files'].items():
        p=dest/name
        if p.is_file() and sha(p)==digest:retained+=1
        else:remaining.append(name)
    print('Already verified',retained,'files; remaining',len(remaining),flush=True)
    program="import json;from pathlib import Path;r=Path('/tmp/cuvarbase-tls-profile/recovery');d=json.loads((r/'transfer-original.json').read_text());print(json.dumps({k:(r/k).stat().st_size for k in d['files']}))"
    sizes=json.loads(subprocess.check_output([sys.executable,str(cloud),'ssh','python -c '+shlex.quote(program)],env=env,text=True))
    shards=[[] for _ in range(a.streams)];totals=[0]*a.streams
    for name in sorted(remaining,key=lambda n:sizes[n],reverse=True):
        i=min(range(a.streams),key=lambda j:totals[j]);shards[i].append(name);totals[i]+=sizes[name]
    (folder/'parallel-transfer-plan.json').write_text(json.dumps(dict(streams=a.streams,retained_files=retained,bytes=totals,shards=shards),indent=2)+'\n')
    print('Starting balanced streams (GB):',[round(v/1e9,3) for v in totals],flush=True)
    def download(i):
        final=folder/f'evidence-shard-{i:02}.tar';temp=final.with_suffix('.tar.partial')
        payload=b'\0'.join(n.encode() for n in shards[i])+b'\0'
        command='cd /tmp/cuvarbase-tls-profile/recovery && tar --null -T - -cf -'
        with temp.open('wb') as out,(folder/f'stream-{i:02}.stderr.log').open('w') as err:
            subprocess.run([sys.executable,str(cloud),'ssh',command],env=env,input=payload,stdout=out,stderr=err,check=True)
        temp.replace(final)
        with tarfile.open(final) as t:t.extractall(dest,filter='data')
        for name in shards[i]:assert sha(dest/name)==transfer['files'][name],name
        print('Shard verified',i,final.stat().st_size,'bytes',flush=True)
        return dict(archive=final.name,bytes=final.stat().st_size,sha256=sha(final),files=len(shards[i]))
    with concurrent.futures.ThreadPoolExecutor(a.streams) as pool:archives=list(pool.map(download,range(a.streams)))
    if prefix.exists():archives.insert(0,dict(archive=prefix.name,bytes=prefix.stat().st_size,sha256=sha(prefix),purpose='Complete, verified members retained from the interrupted original stream'))
    (dest/'transfer-original.json').write_bytes((r/'transfer-original.json').read_bytes())
    for name,digest in transfer['files'].items():assert sha(dest/name)==digest,name
    for subfolder in ['results','sources/harness']:
        for p in (dest/subfolder).rglob('*'):
            if not p.is_file():continue
            target=r/subfolder/p.relative_to(dest/subfolder);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.importing')
            if temporary.exists():temporary.unlink()
            os.link(p,temporary);temporary.replace(target)
    result=dict(complete=True,node='original',archives=archives,files_verified=len(transfer['files']),elapsed_s=time.time()-start,
        note='One valid tar prefix plus balanced parallel tar shards. Extract all listed archives to reconstruct the complete evidence folder. Every file is independently checked against the immutable remote SHA256 manifest. No scientific job was running during transfer.')
    (folder/'download-verification.json').write_text(json.dumps(result,indent=2)+'\n');print('ORIGINAL_ALL_EVIDENCE_VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':main()
