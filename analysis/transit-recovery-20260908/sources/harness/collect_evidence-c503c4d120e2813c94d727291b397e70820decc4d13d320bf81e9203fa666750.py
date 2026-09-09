#!/usr/bin/env python3
"""Create a complete, checksummed recovery artifact on a finished compute node."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path

BASE=Path('/tmp/cuvarbase-tls-profile')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',required=True);ap.add_argument('--manifest-only',action='store_true');a=ap.parse_args();r=BASE/'recovery'
    manifests=['validation_main','timings','components'] if a.node=='original' else [a.node]
    if a.node=='original' and (r/'cpu_operations.json').exists():manifests.append('cpu_operations')
    for name in manifests:
        planned=json.loads((r/(name+'.json')).read_text());executed=json.loads((r/(name+'.execution.json')).read_text())
        assert len(planned)==len(executed),'Do not collect during an active manifest: '+name
    # Stable paths outside recovery are copied so the artifact is self-contained.
    target=r/'sources/runtime';target.mkdir(parents=True,exist_ok=True)
    hardware={}
    for name,cmd in [('gpu',['nvidia-smi','-q','-x']),('cpu',['lscpu']),('cuda',['/usr/local/cuda/bin/nvcc','--version']),('kernel',['uname','-a'])]:
        v=subprocess.run(cmd,capture_output=True,text=True);hardware[name]=dict(exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr)
    for name in ['rustc','cargo']:
        path=Path('/root/.cargo/bin')/name
        if path.exists():
            v=subprocess.run([str(path),'--version'],capture_output=True,text=True)
            hardware[name]=dict(exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr)
    for name in ['cpu.max','memory.max']:
        p=Path('/sys/fs/cgroup')/name;hardware[name]=p.read_text().strip() if p.exists() else None
    hardware['locale']={k:os.getenv(k) for k in ['LANG','LC_ALL','PYTHONUTF8']}
    for name in ['modern','legacy']:
        python=BASE/name/'bin/python'
        if python.exists():
            v=subprocess.run([str(python),'-m','pip','freeze'],capture_output=True,text=True)
            (target/(name+'-pip-freeze.txt')).write_text(v.stdout)
    (target/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
    astropy_root=BASE/'modern/lib/python3.11/site-packages/astropy'
    if astropy_root.exists():
        hashes={str(f.relative_to(astropy_root)):sha(f) for f in astropy_root.rglob('*') if f.is_file() and f.suffix in ['.py','.so','.c','.h']}
        (target/'astropy-installed-source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for name in ['source-v1.tar','cuvarbase-0.2.5-py2.py3-none-any.whl','gtls-head.tar','gputls-0.4.4-py3-none-any.whl','periodfind-source.tar','fBLS-source.tar','profile_tls.py']:
        p=BASE/name
        if p.exists():shutil.copyfile(p,target/name)
    for name in ['periodfind-source','fBLS-source']:
        p=BASE/name
        if p.exists():
            items={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.suffix in ['.py','.cu','.cuh','.h','.cpp','.pyx','.pxd','.hpp','.c','.rs','.toml'] and 'target' not in f.parts and 'build' not in f.parts}
            (target/(name+'-build-tree.json')).write_text(json.dumps(items,indent=2)+'\n')
            if name=='periodfind-source':shutil.copyfile(p/'setup.py',target/'periodfind-build-setup.py')
    lock=BASE/'periodfind-source/rust/Cargo.lock'
    if lock.exists():shutil.copyfile(lock,target/'periodfind-Cargo.lock')
    installed=BASE/'modern/lib/python3.11/site-packages/periodfind'
    if installed.exists():
        for p in installed.rglob('*.so'):
            out=target/'periodfind-binaries'/p.relative_to(installed);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    history=r/'sources/harness';history.mkdir(parents=True,exist_ok=True)
    for p in (r/'scripts').glob('*.py'):shutil.copyfile(p,history/(p.stem+'-'+sha(p)+'.py'))
    files=[p for p in r.rglob('*') if p.is_file() and p.suffix not in ['.tar','.gz','.pyc'] and '__pycache__' not in p.parts and not p.name.startswith('transfer-')]
    # Retain pinned source archives but never recursively collect old result archives.
    files += [p for p in target.glob('*') if p.is_file() and p.suffix in ['.tar','.gz']]
    files=sorted(set(files));manifest={str(p.relative_to(r)):sha(p) for p in files}
    mp=r/f'transfer-{a.node}.json';mp.write_text(json.dumps(dict(node=a.node,files=manifest),indent=2)+'\n')
    (r/f'transfer-{a.node}.files0').write_bytes(b'\0'.join(str(p.relative_to(r)).encode() for p in files+[mp])+b'\0')
    if a.manifest_only:
        print(json.dumps(dict(node=a.node,files=len(files),bytes=sum(p.stat().st_size for p in files),
            transfer='Stream tar to the local machine; verify every file against the SHA256 manifest.')),flush=True)
        return
    archive=BASE/f'recovery-{a.node}.tar'
    with tarfile.open(archive,'w') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(r)))
        t.add(mp,arcname=mp.name)
    record=dict(node=a.node,archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))
    (BASE/f'recovery-{a.node}.archive.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
