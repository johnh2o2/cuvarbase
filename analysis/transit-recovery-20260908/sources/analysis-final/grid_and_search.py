#!/usr/bin/env python3
"""Time a fresh native Keplerian grid plus one BLS search, retaining grid agreement."""
import argparse,importlib.metadata,json,os,sys,time
from pathlib import Path
import numpy as np
import worker
from worker import Backend,dump,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--timing',action='store_true');ap.add_argument('--reps',type=int,default=5)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);d=np.load(a.input);cfg=json.loads(a.config);meta=json.loads(str(d['metadata']));i=128
    lc=tuple(np.array(d[f'{k}_{i}']) for k in ['t','y','dy']);expected=np.asarray(d['freqs']);start=time.perf_counter()
    b=Backend(cfg,d,len(lc[0]));b.sync();initialization=time.perf_counter()-start
    # Select a retained observed time vector with the declared full baseline, so the
    # new grid is scientifically the same as the supplied grid. Its scan is timed.
    grid_index=next(j for j in range(len(meta['cases'])) if np.ptp(d[f't_{j}'])==meta['baseline'])
    grid_t=np.array(d[f't_{grid_index}'])
    def call():
        f,q=b.bls.transit_autofreq(grid_t,rho=1.,samples_per_peak=2,qmin_fac=.5,fmin=1/meta['pmax'],fmax=1/meta['pmin'])
        keep=f<=1/meta['pmin'];b.f=f[keep];b.q=q[keep]
        return b.search([lc])[0]
    b.sync();start=time.perf_counter();call();b.sync();first=time.perf_counter()-start
    call();times=[];candidates=[]
    for rep in range(a.reps):
        b.sync();start=time.perf_counter();out=call();b.sync();times.append(time.perf_counter()-start);candidates.append(out['candidate'])
    assert b.f.shape==expected.shape
    float32_equal=bool(np.array_equal(b.f.astype(np.float32),expected.astype(np.float32)))
    assert float32_equal,'Fresh grid changed a GPU trial frequency'
    q32_equal=bool(np.array_equal(b.q.astype(np.float32),d['q'].astype(np.float32)))
    assert q32_equal,'Fresh grid changed a GPU duration prior'
    p=a.out/f'case_{i:04}.npz';np.savez_compressed(p,periods=out['periods'],power=out['power'])
    case=dict(index=i,injected=False,snr=None,**out['candidate'],recovered=None,alias_recovered=None,
              n_periods=len(out['periods']),search_s=None,evaluation_chunk=1,output_file=p.name,output_sha256=sha(p))
    installed=Path(b.bls.__file__).parent
    dump(a.out/'summary.json',dict(status='ok',profile=meta['profile'],split=meta['split'],config=cfg,indices=[i],cases=[case],
        input_file=Path(a.input).name,input_sha256=sha(a.input),worker_sha256=sha(worker.__file__),wrapper_sha256=sha(__file__),
        installed_sources={'cuvarbase':{str(f.relative_to(installed)):sha(f) for f in installed.rglob('*') if f.is_file() and f.suffix in ['.py','.cu','.cuh','.so']}},
        boundary='Fresh native transit_autofreq grid, trimming the upper endpoint, then prepared-array BLS and common candidate ranking. Imports/context and disk I/O excluded.',
        grid_observed_time_index=grid_index,grid_n_observations=len(grid_t),grid_baseline=float(np.ptp(grid_t)),grid_float32_equal=float32_equal,grid_q_float32_equal=q32_equal,
        grid_max_relative_difference=float(np.max(np.abs(b.f/expected-1))),
        initialization_s=initialization,first_api_s=first,times_s=times,median_s=float(np.median(times)),seconds_per_source=float(np.median(times)),timed_candidates=candidates))
    print('GRID_AND_SEARCH_COMPLETE',flush=True)


if __name__=='__main__':main()
