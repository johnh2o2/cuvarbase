#!/usr/bin/env python3
"""Physical grid and exposure-quadrature development checks, including joint edges."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from common import ROOT,REGIMES,load_case,module,now,sha,write
from development import weighted_snr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    if manifest['split']!='development':p.error('Development only')
    diag=module(ROOT/'benchmarks/tls_accuracy/diagnose.py','survey_boundary_physics')
    rows=[]
    for entry in manifest['cases']:
        arr,m=load_case(a.manifest.parent,entry)
        physical=diag.Regime(**m['physical'])
        signals={n:diag.physical_signal(physical,arr['t'],arr['exposure_days'],epoch=m['truth_epoch'],exposure_nodes=n)
                 for n in (32,64,128)}
        oracle=weighted_snr(signals[128],signals[128],arr['dy'])
        rows.append(dict(kind='observed_exposure_convergence',name=m['name'],regime=m['regime'],
            grid_recovery_ratio=m['nearest_grid_drift_over_half_duration'],
            max_relative_flux_error_32=float(np.max(np.abs(signals[32]-signals[128]))/max(signals[128].max(),1e-30)),
            max_relative_flux_error_64=float(np.max(np.abs(signals[64]-signals[128]))/max(signals[128].max(),1e-30)),
            filter_snr_loss_32=1-weighted_snr(signals[128],signals[32],arr['dy'])/oracle if oracle>0 else None,
            filter_snr_loss_64=1-weighted_snr(signals[128],signals[64],arr['dy'])/oracle if oracle>0 else None))
    for period in (.65,10.,365.25):
        for impact in (.95,1.02):
            for eccentricity in (0.,.8):
                physical=diag.Regime('joint_mdwarf_boundary',period,radius=.1,mass=.1,rp=.00916/.1,
                                    impact=impact,eccentricity=eccentricity)
                duration,full,semimajor=diag.durations(physical)
                for exposure in (30.,200.,1800.):
                    span=2*max(duration,exposure/86400)
                    t=np.linspace(-span,span,2049)
                    signals={n:diag.physical_signal(physical,t,exposure/86400,exposure_nodes=n) for n in (64,128,256)}
                    dy=np.ones(len(t))
                    oracle=weighted_snr(signals[256],signals[256],dy)
                    rows.append(dict(kind='joint_physics_boundary',physical=asdict(physical),exposure_seconds=exposure,
                        duration_days=duration,ingress_days=(duration-full)/2,fractional_duration=duration/period,
                        periastron_stellar_radii=semimajor*(1-eccentricity),
                        filter_snr_loss_64=1-weighted_snr(signals[256],signals[64],dy)/oracle if oracle>0 else None,
                        filter_snr_loss_128=1-weighted_snr(signals[256],signals[128],dy)/oracle if oracle>0 else None,
                        interpretation='Contact/exposure diagnostic at known transit; no blind recovery or annual throughput claim'))
    write(a.out,dict(created_utc=now(),manifest_sha256=sha(a.manifest),source_sha256=sha(__file__),rows=rows,
        limitations='Only solar/0.1solar stellar populations; Earth-size planets; fixed quadratic limb darkening; eccentric omega=90degrees; achromatic depth. Joint Mdwarf/eccentric/grazing annual cases are physical development diagnostics only.'))


if __name__=='__main__':main()
