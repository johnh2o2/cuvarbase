#!/usr/bin/env python3
"""Freeze shared transit injections on observed ZTF and QLP cadences."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import batman
from astropy.io import fits


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--calibration',action='store_true')
    args=ap.parse_args();root=args.root;src=root/'sources';out=root/'inputs';out.mkdir(exist_ok=True)
    module=Path(__file__).resolve().parents[2]/'cuvarbase/bls_frequencies.py'
    spec=importlib.util.spec_from_file_location('grids',module);grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
    cadences={};source_files={}
    p=Path(__file__).resolve().parents[1]/'results/transit_2026-09-08/inputs/cadence_sources/real_ztf_heldout.npz'
    with np.load(p) as d:
        t=np.concatenate([d[f't_0_{b}'] for b in range(2)])
        band=np.concatenate([np.full(len(d[f't_0_{b}']),b) for b in range(2)])
        err=np.concatenate([d[f'dy_0_{b}'] for b in range(2)])
    err=np.clip(err,*np.quantile(err,[.1,.9]));err/=np.median(err)
    cadences['ztf']=(t,err,band,np.full(len(t),30/86400))
    source_files['ztf']=[dict(file=str(p),sha256=sha(p),use='Observed times and clipped relative uncertainty pattern, no observed flux')]
    tess={}
    for s in [1,27,67]:
        p=src/f'qlp-tic261136679-s{s:04}.fits'
        with fits.open(p) as f:
            d=f[1].data;good=np.isfinite(d['TIME'])&(d['QUALITY']==0)
            t=np.asarray(d['TIME'][good],float)
        dt={1:1800,27:600,67:200}[s]/86400
        tess[s]=(t,np.full(len(t),np.sqrt(1800/(dt*86400))),np.full(len(t),s),np.full(len(t),dt))
    cadences['tess_200s']=tess[67]
    cadences['tess_gap']=tuple(np.concatenate([tess[s][j] for s in [1,27]]) for j in range(4))
    for name,sectors in [('tess_200s',[67]),('tess_gap',[1,27])]:
        source_files[name]=[dict(file=f'qlp-tic261136679-s{s:04}.fits',sha256=sha(src/f'qlp-tic261136679-s{s:04}.fits'),
                                 use='Observed TIME and QUALITY only; not observed flux') for s in sectors]
    manifests=[]
    for pi,(name,raw) in enumerate(cadences.items()):
        order=np.argsort(raw[0]);t,relative,band,exposure=[v[order] for v in raw];t=t-t.min()
        baseline=float(np.ptp(t));pmax={'ztf':10.,'tess_200s':baseline/2,'tess_gap':27.457888046800917}[name]
        pmin=2**1.5/8.6307
        # QLP's published samples_per_peak=2, qmin_fac=.5 recursion, evaluated in float64.
        frequencies=grid._euler_transit_grid(1/pmax,1/pmin,.5,2*baseline,8.6307)
        frequencies=frequencies[frequencies<=1/pmin]
        q=grid._q_transit(frequencies)
        tls_periods=np.sort(1/frequencies[frequencies<=1/.6])
        for split,ninj,nnull in ([('calibration',0,128)] if args.calibration else [('tune',32,32),('heldout',128,128)]):
            arrays=dict(freqs=frequencies,q=q,tls_periods=tls_periods)
            rows=[]
            for i in range(ninj+nnull):
                seed=820260908+100000*pi+({'tune':0,'heldout':10000,'calibration':20000}[split])+i
                rng=np.random.default_rng(seed)
                # Independent 0--3% losses make the input arrays distinct; retain real gaps/cadence.
                keep=rng.random(len(t))>rng.uniform(0,.03)
                tt,rr,bb,ee=[v[keep] for v in [t,relative,band,exposure]]
                injected=i<ninj;target=[6.,8.,10.,14.][i%4]
                attempts=0
                while True:
                    attempts+=1
                    period=float(np.exp(rng.uniform(np.log(.8),np.log(min(12.,.8*pmax)))))
                    epoch=float(rng.uniform(0,period));rp=float(rng.choice([.025,.05,.10]));impact=float(rng.uniform(0,.85))
                    a=(6.6743e-11*1.9884e30*(period*86400)**2/(4*np.pi**2))**(1/3)/6.957e8
                    pm=batman.TransitParams();pm.t0=epoch;pm.per=period;pm.rp=rp;pm.a=a
                    pm.inc=float(np.degrees(np.arccos(impact/a)));pm.ecc=0.;pm.w=90.
                    pm.u=[.4804,.1867];pm.limb_dark='quadratic'
                    model=np.ones(len(tt))
                    for exp in np.unique(ee):
                        take=ee==exp
                        model[take]=batman.TransitModel(pm,tt[take],supersample_factor=7,exp_time=float(exp)).light_curve(pm)
                    inside=model<1-1e-9
                    events=np.unique(np.rint((tt[inside]-epoch)/period).astype(int))
                    if inside.sum()>=5 and len(events)>=2: break
                    if attempts>1000: raise RuntimeError('Observable-injection sampling exhausted')
                w=rr**-2;signal=model-np.dot(w,model)/w.sum()
                white_scale=np.sqrt(np.dot(w,signal*signal))/target
                dy=white_scale*rr
                # 25%-amplitude OU residual: 5.9% of total noise variance at equal errors.
                tau=.15 if name.startswith('tess') else 1.
                z=rng.normal(size=len(tt));red=np.empty(len(tt));red[0]=z[0]
                for j in range(1,len(tt)):
                    aou=np.exp(-(tt[j]-tt[j-1])/tau)
                    red[j]=aou*red[j-1]+np.sqrt(1-aou*aou)*z[j]
                y=(model if injected else np.ones(len(tt)))+rng.normal(size=len(tt))*dy+.25*np.median(dy)*red
                duration=period/np.pi*np.arcsin(np.sqrt((1+rp)**2-impact**2)/np.sqrt(a*a-impact**2))
                arrays.update({f't_{i}':tt,f'y_{i}':y,f'dy_{i}':dy,f'band_{i}':bb})
                rows.append(dict(index=i,seed=seed,injected=injected,target_white_oracle_snr=target if injected else None,
                    period=period,epoch=epoch,rp=rp,impact=impact,duration=duration,ndata=len(tt),
                    n_in_transit=int(inside.sum()),observed_transit_events=len(events),ephemeris_draws=attempts,
                    noise_scale=white_scale,white_oracle_snr=float(np.sqrt(np.sum((signal/dy)**2))) if injected else None))
            metadata=dict(profile=name,split=split,baseline=baseline,pmin=pmin,pmax=pmax,
                n_injections=ninj,n_nulls=nnull,source_files=source_files[name],cases=rows,
                period_grid=dict(bls=len(frequencies),tls=len(tls_periods),tls_pmin=float(tls_periods.min())),
                scope='Controlled recovery conditional on >=5 in-transit observations and >=2 observed transit events; observed cadences, synthetic flux and noise; solar density, circular orbits, known normalized band baselines.',
                noise='Independent Gaussian errors plus OU residual of 0.25 median error amplitude; tau=0.15 days TESS / 1 day ZTF. Quoted SNR is white-noise oracle, not native SDE or pink SNR.',
                generator_sha256=sha(__file__),grid_source_sha256=sha(module))
            arrays['metadata']=np.array(json.dumps(metadata))
            p=out/f'{name}_{split}.npz';np.savez_compressed(p,**arrays)
            manifests.append(dict(file=p.name,sha256=sha(p),**{k:v for k,v in metadata.items() if k!='cases'}))
            print(name,split,'N',min(r['ndata'] for r in rows),max(r['ndata'] for r in rows),'periods',len(frequencies),len(tls_periods),flush=True)
    (out/('calibration-manifest.json' if args.calibration else 'manifest.json')).write_text(json.dumps(manifests,indent=2)+'\n')


if __name__=='__main__':main()
