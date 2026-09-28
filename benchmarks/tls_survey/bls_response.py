#!/usr/bin/env python3
"""Known-period noise-free BLS convergence; never a blind recovery result."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from common import BLS_CONFIGS,load_case,now,sha,write
from development import optimal_box,ou_filter_snr,weighted_snr
from run import bls_bounds,production_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    if manifest['split']!='development':p.error('Development data only')
    from cuvarbase.bls import eebls_gpu_fast,_fast_bls_solutions,subtract_epoch,_fast_path_nbins
    from cuvarbase.base import ensure_context
    ensure_context()
    rows=[]
    # One extra4xphase/2xduration/2xminwidth control checks convergence of finest.
    configs={name:settings for name,settings in BLS_CONFIGS.items() if name!='bls_strong'}
    configs['bls_convergence']=dict(noverlap=32,dlogq=.0125,qmin_factor=.125)
    for entry in manifest['cases']:
        arrays,m=load_case(a.manifest.parent,entry)
        t,s,dy=(arrays[k] for k in ('t','signal','dy'))
        period=np.array([m['truth_period']]);freq=1/period
        phase=(t-m['truth_epoch']+.5*period[0])%period[0]-.5*period[0]
        ideal,_=optimal_box(phase,s,dy)
        for method,cfg in configs.items():
            config=dict(cfg);qmin,qmax=bls_bounds(period);qmin*=config.pop('qmin_factor')
            try:
                power=eebls_gpu_fast(t,1-s,dy,freq,qmin=qmin,qmax=qmax,ignore_negative_delta_sols=True,**config)
                solution=_fast_bls_solutions(t,1-s,dy,freq,power,qmin,qmax,1,ignore_negative_delta_sols=True,**config)[0]
                oracle=weighted_snr(s,s,dy)
                power_snr=float(np.sqrt(max(0.,power[0]))*oracle)
                if solution is None:
                    snr=red=0.;q=phi=None
                else:
                    q,phi=solution
                    relative_t,epoch=subtract_epoch(t)
                    local_phi=(phi-epoch*freq[0])%1.
                    nbf=int(_fast_path_nbins(freq.astype(np.float32),qmin,qmax)[1][0])
                    # Recover the discrete histogram offset and box exactly;
                    # floating boundaries use the same float32 operations.
                    grid_start=local_phi*nbf
                    shifted_index=int(round(grid_start*config['noverlap']))
                    start_bin=(shifted_index//config['noverlap'])%nbf
                    pass_index=shifted_index%config['noverlap']
                    offset=np.float32(pass_index/config['noverlap'])
                    phases=relative_t.astype(np.float32)*np.float32(freq[0])
                    phases-=np.floor(phases)
                    bins=np.floor(np.float32(nbf)*phases-offset).astype(np.int64)%nbf
                    model=((bins-start_bin)%nbf<int(round(q*nbf))).astype(float)
                    snr=weighted_snr(s,model,dy)
                    red=ou_filter_snr(t,s,model,dy,m['noise']['ou_amplitude'],m['noise']['ou_tau_days'])
                rows.append(dict(name=m['name'],regime=m['regime'],method=method,valid=True,
                    white_expected_snr=snr,power_derived_white_expected_snr=power_snr,
                    white_power_model_relative_difference=(snr-power_snr)/max(power_snr,1e-30),
                    ou_expected_snr=red,ideal_box_white_snr=ideal,white_retention=snr/ideal if ideal>0 else None,
                    q=q,phi=phi,error=None))
            except Exception:
                rows.append(dict(name=m['name'],regime=m['regime'],method=method,valid=False,
                                 error=traceback.format_exc()))
        write(a.out,dict(created_utc=now(),manifest_sha256=sha(a.manifest),source_sha256=sha(__file__),
            production_sources=production_identity(),rows=rows,
            interpretation='Known true-period noiseless actual BLS GPU power and its CPU-reconstructed box; common fitted-constant white/OU expected SNR. Independent diagnostic, never truth inserted in blind grids.'))
        print(json.dumps(dict(completed=len(rows)//len(configs))),flush=True)


if __name__=='__main__':main()
