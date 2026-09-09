#!/usr/bin/env python3
"""Public-API transit comparison; identical inputs and separately calibrated scores."""
import argparse, hashlib, importlib, importlib.metadata, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def number(v):
    v=float(v);return v if np.isfinite(v) else None
def qtransit(p):return np.arcsin(np.minimum(1.,(1/(np.asarray(p)*8.6307))**(2/3)))/np.pi


def astropy_piece(job):
    from astropy.timeseries import BoxLeastSquares
    lc,periods,durations,oversample=job
    return np.asarray(BoxLeastSquares(*lc).power(periods,durations,method='fast',objective='likelihood',oversample=oversample).power)


def gtls_one(job):
    import cupy as cp
    from gputls import gtls
    lc,kw=job
    try:
        r=gtls(*lc,verbose=False).power(**kw)
        if kw.get('fast'):
            periods,power=[np.asarray(np.ma.filled(v,np.nan)) for v in r]
            good=np.isfinite(periods)&np.isfinite(power)
            j=int(np.nanargmax(np.where(good,power,np.nan))) if good.any() else None
            out=dict(periods=periods,power=power,power_kind='native_sde_spectrum',
                native=dict(period=number(periods[j]) if j is not None else None,score=number(power[j]) if j is not None else None,epoch=None))
        else:
            out=dict(periods=np.asarray(np.ma.filled(r.periods,np.nan)),power=np.asarray(np.ma.filled(r.chi2,np.nan)),
                power_kind='chi2',native=dict(period=number(r.period),score=number(r.SDE),epoch=number(r.T0)))
        cp.cuda.runtime.deviceSynchronize();return out
    except Exception:
        return dict(periods=np.array([]),power=np.array([]),power_kind='failed',error=traceback.format_exc(),
                    native=dict(period=None,score=None,epoch=None))


def spectral_candidate(periods,power):
    """Common QLP-inspired median-bin trend and MAD ranking for all BLS backends."""
    periods=np.asarray(periods,float);power=np.asarray(power,float)
    good=np.isfinite(periods)&np.isfinite(power)&(periods>0)
    if not good.any(): raise ValueError('No finite periodogram values')
    f=1/periods[good];sr=np.sqrt(np.maximum(0.,power[good]))
    delta=np.diff(f)
    if np.all(delta>0):
        pass
    elif np.all(delta<0):
        f=f[::-1];sr=sr[::-1]
    else:
        order=np.argsort(f);f=f[order];sr=sr[order]
    bins=np.array_split(np.arange(len(f)),max(2,int(np.ceil(1+np.log2(len(f))))))
    centers=np.array([np.median(f[b]) for b in bins]);trend=np.array([np.median(sr[b]) for b in bins])
    resid=sr-np.interp(f,centers,trend)
    scale=1.4826*np.median(np.abs(resid-np.median(resid)))
    if not scale>0:raise ValueError('Degenerate periodogram scale')
    index=int(np.argmax(resid));z=(resid[index]-np.median(resid))/scale
    return dict(period=float(1/f[index]),score=float(z),finite_fraction=float(good.mean()))


class Backend:
    def __init__(self,cfg,d,capacity):
        self.cfg=cfg;self.kind=cfg['backend'];self.f=np.array(d['freqs']);self.q=np.array(d['q']);self.tp=np.array(d['tls_periods'])
        self.meta=json.loads(str(d['metadata']));self.sync=lambda:None;self.memory=None;self.functions=None;self.capacity=capacity
        k=self.kind
        if k.startswith('v1_bls') or k.startswith('pypi_bls'):
            import pycuda.autoinit
            import pycuda.driver as drv
            import cuvarbase.bls as bls
            self.bls=bls;self.sync=drv.Context.synchronize
            if k.startswith('pypi'):
                self.functions=bls.compile_bls(function_names=['full_bls_no_sol'],block_size=cfg.get('block_size',256))
                self.memory=bls.BLSMemory(capacity,len(self.f))
            elif cfg.get('unfused'):
                self.functions=bls.compile_bls(function_names=['full_bls_no_sol'])
            if k=='v1_bls_batch' and cfg.get('reuse_batch'):
                from cuvarbase.memory.bls_memory import BLSBatchMemory
                self.memory=BLSBatchMemory(capacity,cfg.get('batch_capacity',16),len(self.f),stream=drv.Stream())
        elif k=='v1_tls':
            from cuvarbase.base import ensure_context
            from cuvarbase.tls import tls_search_batch
            import pycuda.driver as drv
            ensure_context();self.tls=tls_search_batch;self.sync=drv.Context.synchronize
        elif k=='gtls':
            import cupy as cp
            from gputls import gtls
            self.gtls=gtls;self.sync=cp.cuda.runtime.deviceSynchronize
            self.pool=ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) if cfg.get('workers',1)>1 else None
        elif k=='astropy':
            from astropy.timeseries import BoxLeastSquares
            self.astropy=BoxLeastSquares
            self.pool=ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) if cfg.get('workers',1)>1 else None
        elif k.startswith('periodfind'):
            self.pf=importlib.import_module('periodfind.'+('gpu' if k.endswith('gpu') else 'cpu'))
        elif k=='fbls':
            sys.path.insert(0,'/tmp/cuvarbase-tls-profile/fBLS-source')
            from fBLS import fBLS
            self.fbls=fBLS
        else:raise ValueError(k)
        # Chunks make each competitor's scalar duration API approximate the same Keplerian prior.
        # This is 16 API calls for an entire grid, never one Python call per trial period.
        self.chunks=[]
        edges=np.geomspace(self.meta['pmin'],self.meta['pmax'],17)
        periods=1/self.f
        for lo,hi in zip(edges[:-1],edges[1:]):
            ids=np.flatnonzero((periods>=lo)&(periods<(hi if hi<edges[-1] else hi*(1+1e-12))))
            if len(ids):self.chunks.append((ids,float(lo),float(hi)))

    def search(self,lcs):
        k=self.kind;c=self.cfg;outputs=[]
        if k.startswith('v1_bls') or k.startswith('pypi_bls'):
            kw=dict(qmin=c.get('qmin_fac',.5)*self.q,qmax=2*self.q,dlogq=.1,noverlap=c.get('noverlap',3),ignore_negative_delta_sols=True)
            if 'block_size' in c:kw['block_size']=c['block_size']
            if k=='v1_bls_batch':
                powers=self.bls.eebls_gpu_batch(lcs,self.f,memory=self.memory,**kw)
            elif k=='v1_bls_sparse':
                kw.pop('dlogq');kw.pop('noverlap')
                powers=[self.bls.sparse_bls_gpu(*lc,self.f,**kw)[0] for lc in lcs]
            else:
                fn=self.bls.eebls_gpu_fast_optimized if c.get('optimized') else self.bls.eebls_gpu_fast
                if k.startswith('pypi'):kw.update(functions=self.functions,memory=self.memory)
                elif c.get('unfused'):kw.update(functions=self.functions)
                if k.startswith('pypi') and c.get('manual_overlap'):
                    import pycuda.gpuarray as ga
                    powers=[]
                    # PyPI documents noverlap as unimplemented in the fast kernel.
                    # Use its documented dphi workaround, retaining arrays on device.
                    for lc in lcs:
                        best=None
                        for phase in range(c['noverlap']):
                            fn(*lc,self.f,dphi=phase/c['noverlap'],transfer_to_device=phase==0,
                               transfer_to_host=False,**kw)
                            if best is None:best=self.memory.bls_g.copy()
                            else:ga.maximum(best,self.memory.bls_g,out=best)
                        powers.append(best.get())
                else:powers=[np.copy(fn(*lc,self.f,**kw)) for lc in lcs]
            outputs=[dict(periods=1/self.f,power=p) for p in powers]
        elif k=='v1_tls':
            # Density-constrained window and phase/duration resolution are selected on tuning only.
            q=qtransit(self.tp)
            kw=dict(periods=self.tp,R_star=1,M_star=1,t0_oversample=c.get('epoch_os',8),
                n_durations=c.get('durations',24),refine_top_k=c.get('refine',50),return_arrays=True,
                u=[.4804,.1867],qmin=.5*q,qmax=np.minimum(2*q,.333))
            if c.get('wide'):
                ps=self.tp*86400
                kw.update(qmin=np.minimum(695508000*.05*(4*ps/(20848*1e15))**(1/3)/ps,.15),
                          qmax=np.minimum((695508000*4+69911000*2)*(4*ps/(416970*1e15))**(1/3)/ps,.15))
            results=self.tls(lcs,**kw)
            for r in results:
                outputs.append(dict(periods=np.asarray(r['periods']),power=np.asarray(r['chi2']),
                                    native=dict(period=number(r['period']),score=number(r['SDE']),epoch=number(r['T0']))))
        elif k=='gtls':
            kw=dict(periods=self.tp,R_star=1,M_star=1,oversampling_factor=3,T0_fit_margin=c.get('margin',.125),
                duration_grid_step=1.1,verbose=False,show_progress_bar=False,transit_template='default',fast=c.get('fast',False))
            if c.get('density'):kw.update(R_star_min=.5,R_star_max=2.,M_star_min=1.,M_star_max=1.)
            jobs=[(lc,kw) for lc in lcs]
            outputs=list(self.pool.map(gtls_one,jobs) if self.pool else map(gtls_one,jobs))
        elif k=='astropy':
            for lc in lcs:
                power=np.empty(len(self.f));jobs=[]
                for ids,lo,hi in self.chunks:
                    dmin=.5*qtransit(lo)*lo;dmax=min(2*qtransit(hi)*hi,lo*.95)
                    durations=np.geomspace(dmin,dmax,int(np.ceil(np.log(dmax/dmin)/np.log(1.1)))+1)
                    jobs.append((lc,1/self.f[ids],durations,c.get('epoch_os',10)))
                pieces=self.pool.map(astropy_piece,jobs) if self.pool else map(astropy_piece,jobs)
                for (ids,lo,hi),piece in zip(self.chunks,pieces):power[ids]=piece
                outputs.append(dict(periods=1/self.f,power=power))
        elif k.startswith('periodfind'):
            powers=np.empty((len(lcs),len(self.f)))
            ts=[np.asarray(t-t.min(),np.float32) for t,y,dy in lcs]
            ys=[np.asarray(y-np.average(y,weights=dy**-2),np.float32) for t,y,dy in lcs]
            ds=[np.asarray(dy,np.float32) for t,y,dy in lcs]
            for ids,lo,hi in self.chunks:
                qmin=.5*qtransit(hi);qmax=2*qtransit(lo)
                nbins=int(np.ceil(c.get('epoch_os',1)/qmin))
                if c.get('cap_bins'):nbins=min(nbins,c['cap_bins'])
                proc=self.pf.BoxLeastSquares(n_bins=nbins,qmin=float(qmin),qmax=float(qmax))
                out=proc.calc(ts,ys,np.asarray(1/self.f[ids],np.float32),np.array([0.],np.float32),errs=ds,
                              output='periodogram',normalize=False,center=False)
                for j,p in enumerate(out):powers[j,ids]=p.data[:,0]
            outputs=[dict(periods=1/self.f,power=p) for p in powers]
        elif k=='fbls':
            for t,y,dy in lcs:
                obj=self.fbls(t,y,dy);pp=[];ss=[]
                if c.get('native_chunks'):
                    p,s,w,n=obj.BLS([self.meta['pmin'],self.meta['pmax']],NumberOfPeriodChunks=c['native_chunks'],
                        DutyCycle=float(c.get('duty',.5*qtransit(self.meta['pmax']))),over_sampling=c.get('epoch_os',1),
                        ToleranceDenom=c.get('period_os',2),minWidth=1,maxWidth=None,arrayInitSize=2000000)
                    outputs.append(dict(periods=p,power=s*s));continue
                for ids,lo,hi in self.chunks:
                    qmin=float(.5*qtransit(hi));qmax=float(2*qtransit(lo));osamp=c.get('epoch_os',2)
                    nbins=int(osamp/qmin)
                    p,s,w,n=obj.BLS([lo,hi],NumberOfPeriodChunks=1,DutyCycle=qmin,over_sampling=osamp,
                        ToleranceDenom=c.get('period_os',2),minWidth=max(1,int(qmin*nbins)),
                        maxWidth=int(np.ceil(qmax*nbins)),arrayInitSize=2000000)
                    pp.extend(p);ss.extend(s*s)
                outputs.append(dict(periods=np.array(pp),power=np.array(ss)))
        else:raise ValueError(k)
        self.sync()
        for o in outputs:
            if 'native' in o:
                o['candidate']=o['native'];o['candidate']['finite_fraction']=float(np.isfinite(o['power']).mean()) if len(o['power']) else 0.
                o['candidate']['api_result_valid']=o['candidate']['period'] is not None and o['candidate']['score'] is not None
            else:o['candidate']=spectral_candidate(o['periods'],o['power'])
        return outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--indices',default='all');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--timing',action='store_true');ap.add_argument('--reps',type=int,default=3)
    a=ap.parse_args();cfg=json.loads(a.config);d=np.load(a.input);meta=json.loads(str(d['metadata']))
    ids=list(range(len(meta['cases']))) if a.indices=='all' else [int(x) for x in a.indices.split(',')]
    lcs={i:tuple(np.array(d[f'{k}_{i}']) for k in ['t','y','dy']) for i in ids}
    a.out.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',config=cfg,input_file=Path(a.input).name,input_sha256=sha(a.input),
        profile=meta['profile'],split=meta['split'],indices=ids,worker_sha256=sha(__file__),
        boundary='Prepared host lightcurves and explicit period grid to host periodograms and candidate/score. Imports/context excluded; first call reported; no disk output included.',
        environment=dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pycuda'] if importlib.util.find_spec(k)},
                         threads={k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']}),cases=[])
    dump(a.out/'summary.json',record)
    try:
        start=time.perf_counter();backend=Backend(cfg,d,max(len(lcs[i][0]) for i in ids));backend.sync()
        record['initialization_s']=time.perf_counter()-start
        backend.sync();start=time.perf_counter()
        try:backend.search([lcs[ids[0]]]);backend.sync()
        except Exception:
            if a.timing:raise
            record['warmup_error']=traceback.format_exc()
        record['first_api_s']=time.perf_counter()-start
        modules={}
        for name in ['cuvarbase','gputls','periodfind']:
            if name in sys.modules:
                p=Path(sys.modules[name].__file__).parent
                modules[name]={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.suffix in ['.py','.cu','.cuh','.so']}
        record['installed_sources']=modules;dump(a.out/'summary.json',record)
        if a.timing:
            backend.search([lcs[i] for i in ids]);times=[]
            for rep in range(a.reps):
                backend.sync();start=time.perf_counter();outputs=backend.search([lcs[i] for i in ids]);backend.sync()
                times.append(time.perf_counter()-start)
            record.update(times_s=times,median_s=float(np.median(times)),seconds_per_source=float(np.median(times))/len(ids))
            chunks=[(ids,outputs,None)]
        else:
            chunks=None
        chunksize=cfg.get('eval_chunk',1)
        for indices in ([ids] if a.timing else [ids[j:j+chunksize] for j in range(0,len(ids),chunksize)]):
            if not a.timing:
                backend.sync();start=time.perf_counter()
                try:outputs=backend.search([lcs[i] for i in indices]);backend.sync()
                except Exception:
                    i=indices[0];truth=meta['cases'][i]
                    record['cases'].append(dict(index=i,injected=truth['injected'],snr=truth['target_white_oracle_snr'],
                        period=None,score=None,recovered=False if truth['injected'] else None,alias_recovered=False if truth['injected'] else None,
                        api_result_valid=False,error=traceback.format_exc(),search_s=time.perf_counter()-start))
                    dump(a.out/'summary.json',record);continue
                elapsed=time.perf_counter()-start
            else:elapsed=None
            for i,o in zip(indices,outputs):
                truth=meta['cases'][i];candidate=o['candidate'];found=candidate['period']
                drift=abs(found-truth['period'])/truth['period']*meta['baseline'] if found is not None else None
                recovered=bool(drift is not None and drift<=.5*truth['duration']) if truth['injected'] else None
                alias=bool(found is not None and any(abs(found/(truth['period']*k)-1)*meta['baseline']<=.5*truth['duration'] for k in [.5,1.,2.,1/3,3])) if truth['injected'] else None
                path=a.out/f'case_{i:04}.npz'
                np.savez_compressed(path,periods=o['periods'],power=o['power'])
                row=dict(index=i,injected=truth['injected'],snr=truth['target_white_oracle_snr'],
                    true_period=truth['period'],duration=truth['duration'],**candidate,recovered=recovered,
                    alias_recovered=alias,phase_drift_days=drift,n_periods=len(o['periods']),
                    search_s=elapsed/len(indices) if elapsed is not None else None,evaluation_chunk=len(indices),
                    power_kind=o.get('power_kind','chi2' if cfg['backend']=='v1_tls' else 'bls'),
                    output_file=path.name,output_sha256=sha(path))
                if o.get('error'):row['error']=o['error']
                record['cases'].append(row)
                print(json.dumps(row),flush=True)
            dump(a.out/'summary.json',record)
        record['status']='ok'
    except Exception:
        record.update(status='error',error=traceback.format_exc());print(record['error'],flush=True)
    dump(a.out/'summary.json',record)
    if record['status']!='ok':raise SystemExit(1)


if __name__=='__main__':main()
