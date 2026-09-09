#!/usr/bin/env python3
"""Independently audit archives, frozen inputs/harness, runtime sources, and timing isolation."""
import argparse,hashlib,json,tarfile,zipfile
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def archive_sources(path,prefix):
    result={}
    if path.suffix=='.whl':
        with zipfile.ZipFile(path) as t:
            for name in t.namelist():
                if name.startswith(prefix) and Path(name).suffix in ['.py','.cu','.cuh','.h','.rs','.toml','.cpp','.pyx','.pxd','.hpp','.c']:result[name[len(prefix):]]=hashlib.sha256(t.read(name)).hexdigest()
    else:
        with tarfile.open(path) as t:
            for m in t:
                if m.isfile() and m.name.startswith(prefix) and Path(m.name).suffix in ['.py','.cu','.cuh','.h','.rs','.toml','.cpp','.pyx','.pxd','.hpp','.c']:result[m.name[len(prefix):]]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    errors=[];checks=[];frozen=json.loads((r/'harness-freeze.json').read_text())['files']
    for manifest in ['manifest.json','calibration-manifest.json']:
        for item in json.loads((r/'inputs'/manifest).read_text()):
            if sha(r/'inputs'/item['file'])!=item['sha256']:errors.append('Original input manifest '+item['file'])
    for node in ['original','validation_a','validation_b']:
        folder=r/'compute'/node/'evidence';p=folder/f'transfer-{node}.json'
        if not p.exists():errors.append('Missing transfer manifest '+node);continue
        transfer=json.loads(p.read_text())
        for name,digest in transfer['files'].items():
            if not (folder/name).is_file() or sha(folder/name)!=digest:errors.append('Transfer hash '+node+'/'+name)
        for name in ['worker.py','controller.py']:
            if sha(folder/'scripts'/name)!=frozen[name]:errors.append('Final frozen harness '+node+'/'+name)
        checks.append(dict(node=node,transferred_files=len(transfer['files'])))
    runtime=r/'compute/original/evidence/sources/runtime'
    git_check=json.loads((r/'sources/v1-archive-git-verification.json').read_text())
    if not git_check['complete'] or sha(runtime/'source-v1.tar')!=git_check['retained_archive_sha256']:errors.append('v1 archive differs from independently verified git source')
    for name,commit in [('source-v1.tar','1032caf029570dc4841db1c594a2cbb1654e8fd8'),('gtls-head.tar','74e449c325792a763dde4fbffab98039c5e8c111'),('periodfind-source.tar','116b1b27c8db4c95035b5233efa6a1d21780afa5')]:
        with tarfile.open(runtime/name) as archive:
            if archive.pax_headers.get('comment')!=commit:errors.append('Pinned git archive header '+name)
    pypi=json.loads((r/'sources/cuvarbase-pypi.json').read_text());wheel='cuvarbase-0.2.5-py2.py3-none-any.whl'
    listed=next(v for v in pypi['urls'] if v['filename']==wheel)
    if sha(runtime/wheel)!=listed['digests']['sha256']:errors.append('PyPI wheel download digest')
    expected={'v1':archive_sources(runtime/'source-v1.tar','cuvarbase/'),
              'pypi':archive_sources(runtime/'cuvarbase-0.2.5-py2.py3-none-any.whl','cuvarbase/'),
              'gtls':archive_sources(runtime/'gtls-head.tar','src/gputls/')}
    def installed_sources(d,label):
        b=d['config']['backend'];key='v1' if b.startswith('v1') else 'pypi' if b.startswith('pypi') else 'gtls' if b=='gtls' else None
        if key:
            module='gputls' if key=='gtls' else 'cuvarbase';actual=d['installed_sources'][module]
            for file,digest in expected[key].items():
                if key=='gtls' and file in ['GPUFun.cu','GPUFun_bak.cu'] and file not in actual:continue
                if actual.get(file)!=digest:errors.append('Installed source '+label+'/'+file)
    methods=json.loads((r/'validation-methods.json').read_text())['methods'];validated=[]
    for m in methods:
        for split in ['calibration','heldout']:
            name=f"{split}_{m['profile']}_{m['tag']}";p=r/'results'/name/'summary.json';d=json.loads(p.read_text())
            if d['config']!=m['config']:errors.append('Validation config '+name)
            if d['worker_sha256']!=frozen['worker.py']:errors.append('Frozen worker '+name)
            if sha(r/'inputs'/d['input_file'])!=d['input_sha256']:errors.append('Frozen input '+name)
            installed_sources(d,name)
            validated.append(name)
    timing=json.loads((r/'timing-methods.json').read_text())['methods'];executions=[]
    for m in timing:
        folder=r/'results'/m['job'];d=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'execution.json').read_text())
        if e['exit_code']!=0 or d['status']!='ok':errors.append('Timing failed '+m['job'])
        if d['worker_sha256']!=frozen['worker.py']:errors.append('Timing worker '+m['job'])
        if d['config']!=m['config']:errors.append('Timing config '+m['job'])
        installed_sources(d,m['job'])
        if m.get('operational_adapter') and d.get('wrapper_sha256')!=sha(r/'compute/original/evidence/scripts/cpu_batch.py'):errors.append('CPU operational wrapper '+m['job'])
        if m['mode']=='fresh_grid' and not (d.get('grid_float32_equal') and d.get('grid_q_float32_equal')):errors.append('Fresh grid changed GPU search '+m['job'])
        executions.append((e['started_epoch'],e['finished_epoch'],m['job']))
    for aa,bb in zip(sorted(executions),sorted(executions)[1:]):
        if aa[1]>bb[0]:errors.append('Overlapping timed methods '+aa[2]+'/'+bb[2])
    original=r/'compute/original/evidence/results';other_executions=[]
    timing_names={v[2] for v in executions}
    for path in original.glob('*/execution.json'):
        if path.parent.name in timing_names:continue
        e=json.loads(path.read_text())
        if 'started_epoch' in e:other_executions.append((e['started_epoch'],e['finished_epoch'],path.parent.name))
    for aa in executions:
        for bb in other_executions:
            if max(aa[0],bb[0])<min(aa[1],bb[1]):errors.append('Timing overlaps another job '+aa[2]+'/'+bb[2])
    # The GPU/Rust numerical source is unchanged; only CUDA architecture selection differs.
    tree=json.loads((runtime/'periodfind-source-build-tree.json').read_text());source=archive_sources(runtime/'periodfind-source.tar','')
    build_changes=[]
    for name,digest in source.items():
        if name not in tree:errors.append('Missing periodfind build source '+name);continue
        if tree[name]!=digest:build_changes.append(name)
    if sorted(build_changes)!=['setup.py']:errors.append('Unexpected periodfind build source changes '+repr(build_changes))
    result=dict(complete=not errors,errors=errors,transfers=checks,validated_methods=len(validated),timing_jobs=len(executions),
        periodfind_source_changes=build_changes,other_original_job_intervals=len(other_executions),
        note='GTLS upstream .cu reference snapshots are not installed; runtime CUDA string in GPUFun.py is verified. Derived partition times are excluded from all performance ratios. Timing isolation is verified from controller intervals, not asserted for unrelated host tenants.')
    (r/'provenance-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
