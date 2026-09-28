#!/usr/bin/env python3
"""Run full blind standard TLS or development-selected GPU BLS; resumable receipts."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_thread_variable]='1'
import numpy as np
from common import ROOT,BLS_CONFIGS,array_hash,load_case,module,now,recovered,sha,write,source_identity,method_applicable


def finite(v):
    value=float(v)
    return value if np.isfinite(value) else None


def production_identity():
    import cuvarbase
    package=Path(cuvarbase.__file__).parent
    return {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.cu','.cuh')}


def bls_bounds(periods):
    """Broad stellar envelope: no use of injection duration, impact, or epoch."""
    seconds=np.asarray(periods)*86400
    qmin=np.minimum(.15,695508000*.05*(4*seconds/(20848*1e15))**(1/3)/seconds)
    qmax=np.minimum(.15,(695508000*4+2*69911000)*(4*seconds/(416970*1e15))**(1/3)/seconds)
    return qmin,qmax


def bls_candidates(periods,power,chi2_null=None):
    worker=module(ROOT/'benchmarks/transit/worker.py','survey_bls_ranking')
    good=np.isfinite(power)
    if not good.any():
        raise ValueError('No finite BLS powers')
    index=int(np.argmax(np.where(good,power,-np.inf)))
    candidates={'raw':dict(period=float(periods[index]),score=float(power[index]))}
    if chi2_null is not None:
        candidates['likelihood']=dict(period=float(periods[index]),score=float(power[index]*chi2_null),
                                      chi2_null=float(chi2_null))
    try:
        candidates['detrended']=worker.spectral_candidate(periods,power)
    except ValueError as exc:
        candidates['detrended']=dict(period=None,score=None,error=str(exc))
    return candidates


def search(arrays,metadata,method):
    t,y,dy,periods=(arrays[k] for k in ('t','y','dy','periods'))
    if method=='tls':
        from cuvarbase.tls import tls_search_gpu
        r=tls_search_gpu(t,y,dy,periods=periods,return_arrays=True,**metadata['search_kwargs'])
        candidate=dict(period=finite(r['period']),score=finite(r['SDE']))
        candidate['successful_no_candidate']=candidate['period'] is None and candidate['score']==0.
        candidate['no_candidate_reason']=r.get('error') if candidate['successful_no_candidate'] else None
        candidates={'native':candidate}
        spectra={key:array_hash(np.asarray(np.ma.filled(r[key],np.nan))) for key in ('periods','chi2')}
        spectra['valid_mask']=array_hash(np.isfinite(np.ma.filled(r['chi2'],np.nan)))
    else:
        from cuvarbase.bls import eebls_gpu_fast
        qmin,qmax=bls_bounds(periods)
        config=dict(BLS_CONFIGS[method]);qmin*=config.pop('qmin_factor')
        p=np.asarray(eebls_gpu_fast(t,y,dy,1/periods,qmin=qmin,qmax=qmax,
            ignore_negative_delta_sols=True,**config))
        weight=dy**-2
        weighted_mean=np.dot(weight,y)/weight.sum()
        chi2_null=float(np.dot(weight,(y-weighted_mean)**2))
        candidates=bls_candidates(periods,p,chi2_null)
        spectra={'periods':array_hash(periods),'power':array_hash(p),'valid_mask':array_hash(np.isfinite(p))}
    return candidates,spectra


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',choices=('tls',*BLS_CONFIGS),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--regimes',help='Optional comma-separated shard; all cases in these regimes required')
    parser.add_argument('--seal',type=Path)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    if args.shard_count<1 or not 0<=args.shard_index<args.shard_count:
        parser.error('Require 0 <= shard-index < shard-count')
    manifest=json.loads(args.manifest.read_text())
    if manifest['status']!='complete':
        parser.error('Incomplete input manifest')
    if manifest['split'] in ('calibration','injections','nulls'):
        if args.seal is None or sha(args.seal)!=manifest['seal_sha256']:
            parser.error('Must supply original development seal')
        seal=json.loads(args.seal.read_text())
        if seal['source_identity']!=source_identity():
            parser.error('Benchmark scientific sources changed after freeze')
        if args.shard_count!=seal['execution_shards']:
            parser.error('Execution sharding differs from frozen policy')
    selected=set(args.regimes.split(',')) if args.regimes else set(manifest['regimes'])
    entries=[e for e in manifest['cases'] if e['metadata']['regime'] in selected][args.shard_index::args.shard_count]
    if not entries:
        parser.error('Empty shard')
    if args.out.exists():
        result=json.loads(args.out.read_text())
        if (result['manifest_sha256']!=sha(args.manifest) or result['methods']!=args.methods or
                result['regimes']!=sorted(selected) or result['runner_sha256']!=sha(__file__) or
                result['shard_index']!=args.shard_index or result['shard_count']!=args.shard_count):
            parser.error('Existing receipt belongs to a different configuration/source')
    else:
        result=dict(status='running',started_utc=now(),manifest_sha256=sha(args.manifest),
                    split=manifest['split'],methods=args.methods,regimes=sorted(selected),
                    runner_sha256=sha(__file__),planned_cases=len(entries),cases=[],
                    shard_index=args.shard_index,shard_count=args.shard_count,
                    inapplicable=[dict(regime=r,method=m,reason='Stronger minimum width exceeds the fast BLS shared-memory bound; finest already attains ideal-box response in gapped development')
                                  for r in sorted(selected) for m in args.methods if not method_applicable(m,r)],
                    execution_policy='Independent processes, one numerical-library thread each; one GPU shared. Fixed logical TLS groups.',
                    planned_inputs=[dict(name=e['metadata']['name'],sha256=e['sha256']) for e in entries],
                    cpu_quota={str(p):p.read_text().strip() for p in map(Path,
                        ('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                         '/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us',
                         '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us')) if p.exists()},
                    threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    from cuvarbase.base import ensure_context
    ensure_context()
    import pycuda.driver as drv
    identity=production_identity()
    if manifest['split'] in ('calibration','injections','nulls') and identity!=seal['production_sources']:
        parser.error('Production numerical sources changed after freeze')
    if 'production_sources' in result and identity!=result['production_sources']:
        parser.error('Cannot resume after numerical sources changed')
    result['production_sources']=identity
    result['gpu']=str(drv.Context.get_device().name())
    result['packages']={k:importlib.metadata.version(k) for k in ('numpy','scipy','pycuda','batman-package')}
    write(args.out,result)
    completed={(row['name'],row['method']) for row in result['cases']}
    for entry in entries:
        arrays,metadata=load_case(args.manifest.parent,entry)
        for method in args.methods:
            if not method_applicable(method,metadata['regime']):
                continue
            if (metadata['name'],method) in completed:
                continue
            # A global receipt can contain different development-selected BLS
            # settings per regime. Only run that frozen setting for test data.
            if manifest['split'] in ('calibration','injections','nulls') and method!='tls':
                if method!=seal['bls_selected'][metadata['regime']]['method']:
                    continue
            begin=time.perf_counter()
            try:
                candidates,spectra=search(arrays,metadata,method)
                drv.Context.synchronize()
                ranker=('native' if method=='tls' else seal['bls_selected'][metadata['regime']]['ranker']
                        if manifest['split'] in ('calibration','injections','nulls') else 'raw')
                valid=(candidates[ranker]['score'] is not None and
                       (candidates[ranker]['period'] is not None or candidates[ranker].get('successful_no_candidate',False)))
                for c in candidates.values():
                    c['recovered']=recovered(c['period'],metadata)
                    c['alias_recovered']=recovered(c['period'],metadata,aliases=True)
                row=dict(name=metadata['name'],regime=metadata['regime'],method=method,
                         valid=valid,candidates=candidates,spectra=spectra,error=None)
            except Exception:
                row=dict(name=metadata['name'],regime=metadata['regime'],method=method,
                         valid=False,candidates={},spectra={},error=traceback.format_exc())
            row.update(input_sha256=entry['sha256'],elapsed_s=time.perf_counter()-begin,
                       null=metadata['null'],white_oracle_snr=metadata['latent_white_oracle_snr'],
                       observed_events=metadata['observed_events'],in_transit_observations=metadata['in_transit_observations'],
                       grid_reachable=metadata.get('nearest_grid_drift_over_half_duration',0.)<=1.)
            result['cases'].append(row)
            write(args.out,result)
        print(json.dumps(dict(completed_cases=sum(r['method']=='tls' for r in result['cases']),
                              records=len(result['cases']),planned_cases=len(entries))),flush=True)
    result.update(status='complete',completed_utc=now())
    write(args.out,result)


if __name__=='__main__':
    main()
