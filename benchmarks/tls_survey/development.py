#!/usr/bin/env python3
"""Comparable expected filter SNR of native index-template and ideal box families.

This uses actual GTLS cache deficits (including literal padding) at the known
period, every sample start, and a fitted weighted constant. It is an optimistic
filter-family diagnostic: native GTLS's depth estimator/ranker need not select
its matched-filter maximum. It never equates package SNR or SDE fields.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.fft import rfft,irfft
from common import ROOT,load_case,module,now,sha,write


def weighted_snr(signal, template, errors):
    w=errors**-2
    h=template-np.dot(w,template)/w.sum()
    norm=np.sqrt(np.dot(w,h*h))
    return max(0.,float(np.dot(w*signal,h)/norm)) if norm>0 else 0.


def ou_filter_snr(times,signal,template,errors,amplitude,tau):
    w=errors**-2
    h=template-np.dot(w,template)/w.sum()
    coefficients=w*h
    order=np.argsort(times)
    a=coefficients[order]
    t=times[order]
    # a^T K a = amp^2 (sum a_i^2 + 2 sum_{j<i} a_i a_j exp(-dt/tau)).
    running=0.
    cross=0.
    for i in range(1,len(t)):
        running=np.exp(-(t[i]-t[i-1])/tau)*(running+a[i-1])
        cross+=a[i]*running
    variance=np.dot(coefficients*errors,coefficients*errors)+amplitude**2*(np.dot(a,a)+2*cross)
    return max(0.,float(np.dot(coefficients,signal)/np.sqrt(variance))) if variance>0 else 0.


def optimal_box(phase,signal,errors):
    """All contiguous positive-endpoint intervals, with a fitted constant.

    Phases are centered on truth, so the physical transit support does not
    wrap. Boxes with < half the total weight can only improve when a zero-
    signal outer sample is removed. This unhandicapped control exhausts the
    relevant intervals rather than using contact duration as the box width.
    """
    order=np.argsort(phase)
    s,w=signal[order],errors[order]**-2
    total=w.sum()
    if w[s>0].sum() >= .5*total:
        raise ValueError("Physical signal support exceeds half the weight; box certificate unsupported")
    centered=s-np.dot(w,s)/total
    cw=np.r_[0.,np.cumsum(w)]
    cy=np.r_[0.,np.cumsum(w*centered)]
    positive=np.flatnonzero(s>0)
    best=0.
    endpoints=None
    for j,start in enumerate(positive):
        ends=positive[j:]+1
        weight=cw[ends]-cw[start]
        numerator=cy[ends]-cy[start]
        good=(weight>0)&(weight<.5*total)&(numerator>0)
        score=np.zeros(len(ends))
        score[good]=numerator[good]/np.sqrt(weight[good]*(1-weight[good]/total))
        k=int(score.argmax())
        if score[k]>best:
            best=float(score[k]);endpoints=(int(start),int(ends[k]))
    template=np.zeros(len(s))
    if endpoints:
        template[order[endpoints[0]:endpoints[1]]]=1.
    return best,template


def optimal_native_family(times,period,signal,errors,cache):
    """FFT correlations enumerate every start of every native cache row."""
    order=np.argsort((times%period)/period)
    s,w=signal[order],errors[order]**-2
    total=w.sum()
    centered=s-np.dot(w,s)/total
    fw,fs=rfft(w),rfft(w*centered)
    best=0.;winner=None
    n=len(s)
    for index,width in enumerate(cache['widths']):
        width=int(width)
        if width>n:
            continue
        g=np.zeros(n)
        g[:width]=cache['template_deficits'][index,:width]
        fg=rfft(g)
        numer=irfft(fs*np.conjugate(fg),n)
        wg=irfft(fw*np.conjugate(fg),n)
        wg2=irfft(fw*np.conjugate(rfft(g*g)),n)
        variance=wg2-wg*wg/total
        good=variance>max(float(wg2.max())*1e-12,0.)
        scores=np.zeros(n)
        scores[good]=np.maximum(0.,numer[good])/np.sqrt(variance[good])
        start=int(scores.argmax())
        if scores[start]>best:
            best=float(scores[start]);winner=(index,width,start)
    template=np.zeros(n)
    if winner:
        index,width,start=winner
        template[order[(start+np.arange(width))%n]]=cache['template_deficits'][index,:width]
    return best,template,winner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest['split']!='development':
        parser.error('Expected-SNR development may only consume the declared development split')
    ref=module(ROOT/'cuvarbase/tls_reference_math.py','survey_diag_reference')
    rows=[]
    caches={}
    for entry in manifest['cases']:
        a,m=load_case(args.manifest.parent,entry)
        key=(len(a['t']),tuple(a['periods'][[0,-1]]))
        if key not in caches:
            caches[key]=ref.build_cache(a['periods'],len(a['t']))
        phase=(a['t']-m['truth_epoch']+.5*m['truth_period'])%m['truth_period']-.5*m['truth_period']
        bs,bg=optimal_box(phase,a['signal'],a['dy'])
        ts,tg,winner=optimal_native_family(a['t'],m['truth_period'],a['signal'],a['dy'],caches[key])
        oracle=weighted_snr(a['signal'],a['signal'],a['dy'])
        red_kwargs=dict(amplitude=m['noise']['ou_amplitude'],tau=m['noise']['ou_tau_days'])
        br=ou_filter_snr(a['t'],a['signal'],bg,a['dy'],**red_kwargs)
        tr=ou_filter_snr(a['t'],a['signal'],tg,a['dy'],**red_kwargs)
        rows.append(dict(name=m['name'],regime=m['regime'],ndata=m['ndata'],period=m['truth_period'],
            q=m['fractional_duration'],impact=m['physical']['impact'],eccentricity=m['physical']['eccentricity'],
            stellar_density_solar=m['stellar_density_solar'],observed_events=m['observed_events'],
            in_transit_observations=m['in_transit_observations'],white_oracle_snr=oracle,
            native_family_white_snr=ts,ideal_box_white_snr=bs,
            native_family_ou_snr=tr,ideal_box_ou_snr=br,
            native_white_advantage=ts/bs-1 if bs>0 else None,
            native_ou_advantage=tr/br-1 if br>0 else None,
            native_cache_winner=[int(v) for v in winner] if winner else None,
            template_filter_white_check=weighted_snr(a['signal'],tg,a['dy']),
            box_filter_white_check=weighted_snr(a['signal'],bg,a['dy'])))
        write(args.out,dict(created_utc=now(),manifest_sha256=sha(args.manifest),diagnostic_sha256=sha(__file__),rows=rows,
             interpretation='Known-period matched-filter ceilings for GTLS actual sample-index cache and unhandicapped boxes. Fitted weighted constant. Native depth/ranking may perform worse. White and OU variance are common definitions, never package-reported SNR/SDE.'))
        print(json.dumps(dict(completed=len(rows),count=len(manifest['cases']))),flush=True)


if __name__=='__main__':
    main()
