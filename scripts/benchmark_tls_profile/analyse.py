#!/usr/bin/env python3
"""Verify profiling evidence and report component times and output agreement."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def archive_sources(path, prefix):
    result = {}
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith(prefix) and Path(name).suffix in ['.py','.cu','.cuh']:
                    result[name[len(prefix):]] = hashlib.sha256(archive.read(name)).hexdigest()
    else:
        with tarfile.open(path) as archive:
            for item in archive:
                if item.isfile() and item.name.startswith(prefix) and Path(item.name).suffix in ['.py','.cu','.cuh']:
                    result[item.name[len(prefix):]] = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    assert result
    return result


def category(name):
    if 'flux prefix sums' in name:
        return 'Per-period prefix-sum loop'
    if 'duration-mask union' in name:
        return 'Duration-mask union'
    if 'statistics' in name or 'best-period fit/diagnostics' in name:
        return 'Statistics / final diagnostics'
    if 'refinement' in name:
        return 'Candidate refinement'
    if any(s in name for s in ['folding/sorting','reorder/weights','error prefixes',
                               'residual kernel','reductions/chunk','coarse search kernel',
                               'coarse spectrum transfer','parameter-spectrum transfers']):
        return 'Other coarse search / transfers'
    return 'Setup / remaining API work'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    root=ap.parse_args().root
    manifest=json.loads((root/'transfer-sha256.json').read_text())
    errors=[]
    for name,digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            errors.append('Transfer mismatch: '+name)
    actual=json.loads((root/'results/installed-source-hashes.json').read_text())
    expected={
        'v1':archive_sources(root/'sources/source-v1.tar','cuvarbase/'),
        'gtls_head':archive_sources(root/'sources/gtls-head.tar','src/gputls/'),
        'gtls_pypi':archive_sources(root/'sources/gputls-0.4.4-py3-none-any.whl','gputls/'),
        'cpu_tls':archive_sources(next((root/'sources').glob('transitleastsquares-1.32-*.whl')),'transitleastsquares/')}
    noninstalled=[]
    for group in expected:
        for name,digest in expected[group].items():
            if group == 'gtls_head' and name in ['GPUFun.cu', 'GPUFun_bak.cu'] and name not in actual[group]:
                noninstalled.append(dict(group=group, file=name, upstream_sha256=digest,
                    reason='Upstream reference file omitted by package installation; runtime CUDA source is embedded in GPUFun.py, whose installed hash is verified. useLocalPTXCUBIN=False in this protocol.'))
                continue
            if actual[group].get(name)!=digest:
                errors.append('Installed source mismatch: '+group+'/'+name)
    jobs=json.loads((root/'results/jobs.json').read_text())
    records={}
    phases=[]
    timings=[]
    for job in jobs:
        name=job['name'];r=json.loads((root/f'results/{name}.json').read_text())
        execution=json.loads((root/f'results/{name}.execution.json').read_text())
        records[name]=r
        if execution['exit_code']!=0:
            errors.append('Uncompleted diagnostic job: '+name)
        if job['version']=='cpu':
            continue
        if r['status']!='ok':
            errors.append('Uncompleted GPU profile: '+name)
            continue
        if r['source_files']!=actual[{'head':'gtls_head','pypi':'gtls_pypi','release':'v1'}[job['version']]]:
            errors.append('Worker source hashes mismatch: '+name)
        if hashlib.sha256((root/f'results/{name}.npz').read_bytes()).hexdigest()!=r['output_file_sha256']:
            errors.append('Output hash mismatch: '+name)
        if float(np.median(r['native_times_s']))!=r['native_median_s']:
            errors.append('Median mismatch: '+name)
        for transformation in r['transformations']:
            path=root/'results'/transformation['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=transformation['transformed_sha256']:
                errors.append('Transformed harness mismatch: '+str(path))
        timings.append(dict(job=name,native_median_s=r['native_median_s'],
                            native_min_s=min(r['native_times_s']),native_max_s=max(r['native_times_s']),
                            first_api_s=r['first_api_s'],
                            profile_mean_s=float(np.mean([p['total_s'] for p in r['profiles']]))))
        for block,p in enumerate(r['profiles']):
            for label,v in p['phases'].items():
                phases.append(dict(job=name,profile_repeat=block,phase=label,category=category(label),
                                   exclusive_s=v['exclusive_s'],inclusive_s=v['inclusive_s'],calls=v['calls']))
    comparisons=[]
    for profile in ['ztf','rubin']:
        name=f'{profile}_gtls_head_native';baseline=records[name]
        with np.load(root/f'results/{name}.npz') as d:
            reference={k:d[k] for k in d.files}
        for variant in ['union','both']:
            other_name=f'{profile}_gtls_head_{variant}';r=records[other_name]
            assert baseline['input_sha256']==r['input_sha256']
            with np.load(root/f'results/{other_name}.npz') as d:
                identical=all(np.array_equal(reference[k],d[k],equal_nan=True) for k in reference)
                same_mask=all(np.array_equal(np.isfinite(reference[k]),np.isfinite(d[k])) for k in reference)
                c0,c1=reference['chi2_0'],d['chi2_0'];ok=np.isfinite(c0)&np.isfinite(c1)
                delta=float(np.max(np.abs(c0[ok]-c1[ok])))
                relative=float(np.max(np.abs(c0[ok]-c1[ok])/np.maximum(np.abs(c0[ok]),1e-30)))
            comparisons.append(dict(profile=profile,variant=variant,
                                    original_s=baseline['native_median_s'],modified_s=r['native_median_s'],
                                    original_over_modified=baseline['native_median_s']/r['native_median_s'],
                                    exact_periods_and_chi2=identical,same_finite_mask=same_mask,
                                    max_abs_chi2_difference=delta,max_relative_chi2_difference=relative,
                                    original_period=baseline['native_results'][0]['period'],
                                    modified_period=r['native_results'][0]['period'],
                                    delta_sde=r['native_results'][0]['SDE']-baseline['native_results'][0]['SDE']))
    table(root/'timing_summary.csv',timings);table(root/'phase_timings.csv',phases)
    table(root/'ablation_output_comparison.csv',comparisons)
    verification=dict(transferred_files=len(manifest),jobs=len(jobs),
                      source_files={k:len(v) for k,v in expected.items()},errors=errors,
                      noninstalled_upstream_reference_files=noninstalled,
                      all_pass=not errors,meaning='Evidence consistency; CPU API errors remain errors. '
                      'Ablation numerical differences are reported, not reclassified as equivalent.')
    dump(root/'verification.json',verification)
    assert not errors,errors
    cats=['Per-period prefix-sum loop','Duration-mask union','Statistics / final diagnostics',
          'Candidate refinement','Other coarse search / transfers','Setup / remaining API work']
    colors=['#e2913a','#c65d51','#80679d','#a48d53','#348a90','#afb4b8']
    suffixes=['gtls_pypi_native','gtls_head_native','gtls_head_union','gtls_head_both','v1_release_native']
    labels=['GTLS\nPyPI','GTLS\nupstream','GTLS\nunion batched','GTLS\nboth batched','cuvarbase\nv1.0']
    fig,axes=plt.subplots(1,2,figsize=(13.5,6),sharey=True)
    for ax,profile,title in zip(axes,['ztf','rubin'],['ZTF-like: 219,127 periods','Rubin-like: 313,007 periods']):
        bottoms=np.zeros(5)
        for cat,color in zip(cats,colors):
            values=[]
            for suffix in suffixes:
                r=records[profile+'_'+suffix]
                values.append(float(np.mean([sum(v['exclusive_s'] for name,v in p['phases'].items()
                                                if category(name)==cat) for p in r['profiles']])))
            ax.bar(range(5),values,bottom=bottoms,color=color,label=cat,width=.7)
            bottoms+=values
        for i,suffix in enumerate(suffixes):
            r=records[profile+'_'+suffix];v=r['native_median_s']
            ax.errorbar(i,v,yerr=[[v-min(r['native_times_s'])],[max(r['native_times_s'])-v]],
                        color='black',marker='D',ms=4,capsize=3,linewidth=1)
            ax.text(i,max(v,bottoms[i])+0.5,f'{v:.2f}s',ha='center',fontsize=10)
        ax.set_xticks(range(5),labels,fontsize=9);ax.set_title(title)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Seconds per source (linear scale)')
    axes[0].set_ylim(0,max(max(t['native_max_s'] for t in timings),max(bottoms))+3)
    handles,labels_legend=axes[0].get_legend_handles_labels()
    handles.append(Line2D([],[],color='black',marker='D',label='Uninstrumented median and range'))
    labels_legend.append('Uninstrumented median and range')
    fig.legend(handles,labels_legend,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('TLS timing breakdown: large GTLS costs come from per-period GPU dispatch loops',fontsize=14)
    fig.text(.5,.18,'Same retained source and full supplied period grid within each panel. A40; one process per method.\n'
             'Stacks: synchronized wall-phase profiles (1 GTLS / 2 v1 calls). Diamonds: 3 ordinary warm calls.\n'
             'Batched GTLS variants change Python array operations only; returned spectra are compared separately.',
             ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.32,top=.86,wspace=.08)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','svg','pdf']:
        fig.savefig(folder/f'tls_components.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    dump(root/'analysis_summary.json',dict(timings=timings,ablations=comparisons,verification=verification))
    print(json.dumps(dict(verification=verification,ablations=comparisons),indent=2))


if __name__=='__main__':
    main()
