#!/usr/bin/env python3
"""One exportable figure: latency, throughput, independent recovery, and cost."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROFILES=['tess_200s','tess_gap','ztf']
TITLES={'tess_200s':'TESS: one dense sector','tess_gap':'TESS: two separated sectors','ztf':'ZTF: sparse g/r'}
SUBTITLES={'tess_200s':'200 s cadence · ≤9,736 samples · 25.8 d span','tess_gap':'30 / 10 min cadence · ≤4,295 samples · 735 d span','ztf':'≤1,317 samples · 2,744 d span'}
COLORS={'bls_v1':'#008566','bls_v1_batch':'#008566','tls_v1':'#008566','bls_pypi':'#2466aa','bls_cpu':'#c66a17','bls_gpu':'#8957a5','gtls':'#8957a5','gtls_batch':'#8957a5'}


def time_label(x):
    return f'{x*1000:.2g} ms' if x<.1 else f'{x:.3g} s'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    rec=json.loads((r/'recovery_analysis.json').read_text());tim=json.loads((r/'timing_analysis.json').read_text())
    assert rec['verification']['complete'] and tim['verification']['complete']
    assert rec['verification']['arrays_verified'] and tim['verification']['arrays_verified']
    rr={(v['profile'],v['method']):v for v in rec['methods']}
    tt={(v['profile'],v['method'],v['mode']):v for v in tim['timings']}
    cc={(v['profile'],v['v1'],v['comparator']):v for v in rec['comparisons']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b9c1c8','xtick.color':'#45505a','ytick.color':'#45505a'})
    fig=plt.figure(figsize=(18.5,14.8),facecolor='white')
    gs=fig.add_gridspec(4,3,left=.115,right=.978,top=.837,bottom=.125,hspace=.54,wspace=.60,height_ratios=[1.15,1.,.85,1.])
    fig.text(.045,.966,'cuvarbase transit searches: speed and independently measured recovery',fontsize=23,weight='bold',color='#172a3a')
    fig.text(.045,.936,'Measured warm search time, with sensitivity checked on independent transit injections',fontsize=14,color='#526270')
    fig.legend(handles=[Line2D([],[],marker='o',color='#334a5e',markerfacecolor='white',linestyle='none',label='One source (warm)'),
        Line2D([],[],marker='o',color='#334a5e',linestyle='none',label='16-source batch: time per source')],
        loc='upper left',bbox_to_anchor=(.039,.925),ncol=2,frameon=False,fontsize=11)
    for col,p in enumerate(PROFILES):
        x=(gs[0,col].get_position(fig).x0+gs[0,col].get_position(fig).x1)/2
        fig.text(x,.887,TITLES[p],ha='center',fontsize=15,weight='bold',color='#172a3a')
        fig.text(x,.869,SUBTITLES[p],ha='center',fontsize=10,color='#526270')
        for family,trow,rrow,methods,v1 in [('BLS',0,1,['bls_v1','bls_pypi','bls_cpu','bls_gpu'],'bls_v1'),('TLS',2,3,['tls_v1','gtls'],'tls_v1')]:
            ax=fig.add_subplot(gs[trow,col]);rx=fig.add_subplot(gs[rrow,col])
            labels=[];xs=[]
            for index,m in enumerate(methods):
                values=[tt[p,m,mode]['seconds_per_source'] for mode in ['single','batch16']];xs.extend(values)
                ax.plot(values,[index,index],color=COLORS[m],lw=2,alpha=.65)
                ends=[]
                for mode,value in zip(['single','batch16'],values):
                    row=tt[p,m,mode];lo=row['min_total_s']/row['n'];hi=row['max_total_s']/row['n'];xs.extend([lo,hi]);ends.append(hi)
                    ax.errorbar(value,index,xerr=[[max(0.,value-lo)],[max(0.,hi-value)]],fmt='none',ecolor=COLORS[m],
                        elinewidth=1,capsize=2,alpha=.7,zorder=3)
                ax.scatter(values[0],index,s=58,edgecolors=COLORS[m],facecolors='white',linewidths=1.7,zorder=4)
                ax.scatter(values[1],index,s=45,color=COLORS[m],zorder=5)
                if m==v1:label='cuvarbase v1'
                elif m=='bls_pypi':label='PyPI 0.2.5'
                elif m=='bls_cpu':label='CPU: '+('Astropy' if rr[p,m]['config']['backend']=='astropy' else 'periodfind')
                elif m=='bls_gpu':label='GPU: periodfind'
                else:label='GTLS upstream'
                labels.append(label)
                ref='bls_v1_batch' if family=='BLS' else 'tls_v1'
                qualification=cc.get((p,ref,'gtls_batch' if m=='gtls' else m));mark='†' if qualification and qualification['comparable_detection'] else ''
                text=time_label(values[1]) if m==v1 else f"{time_label(values[1])} · {values[1]/tt[p,v1,'batch16']['seconds_per_source']:.1f}×{mark}"
                ax.annotate(text,(max(ends),index),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color=COLORS[m],weight='bold' if m==v1 else 'normal')
            ax.set_xscale('log');ax.set_xlim(min(xs)/1.7,max(xs)*10.0);ax.set_ylim(len(methods)-.5,-.7)
            ax.set_yticks(range(len(methods)),labels);ax.tick_params(axis='y',length=0,labelsize=10)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_xlabel('Search time / source (seconds; log scale)',fontsize=10)
            cost=tt[p,v1,'batch16']['projected_gpu_usd_per_million']
            ax.set_title(f'{family} · v1 projected GPU cost: ${cost:.2f} / million',loc='left',fontsize=10,pad=9,color='#526270')
            if family=='BLS':
                fv=tt[p,'bls_v1','fresh_grid']['seconds_per_source'];fp=tt[p,'bls_pypi','fresh_grid']['seconds_per_source']
                ax.text(0,-.24,f'Fresh grid + search: v1 {time_label(fv)} vs PyPI {time_label(fp)} ({fp/fv:.1f}×)',
                        transform=ax.transAxes,fontsize=9,color='#526270',ha='left',va='top')
            draw=[]
            for m in methods:
                tag=m+'_batch' if m in ['bls_v1','gtls'] else m
                d=rr[p,tag];points=d['by_snr'];x=np.array([q['snr'] for q in points]);y=np.array([q['recall'] for q in points])*100
                interval=np.array([q['interval'] for q in points]).T*100
                offset={'bls_v1':-.13,'tls_v1':-.1,'bls_pypi':-.04,'bls_cpu':.05,'bls_gpu':.14,'gtls':.1}[m]
                line=rx.errorbar(x+offset,y,yerr=np.stack([y-interval[0],interval[1]-y]),color=COLORS[m],fmt='o-',
                    lw=2.3 if m==v1 else 1.2,ms=4,capsize=2,elinewidth=.65,alpha=1 if m==v1 else .82)
                draw.append(line)
            # Show any sensitivity difference introduced by the public BLS batch path.
            if family=='BLS' and rr[p,'bls_v1']['detected_vector']!=rr[p,'bls_v1_batch']['detected_vector']:
                d=rr[p,'bls_v1'];rx.plot([q['snr'] for q in d['by_snr']],[q['recall']*100 for q in d['by_snr']],
                    '--',color=COLORS[v1],lw=1,alpha=.75,label=f"v1 single (null FPR {d['false_positive_rate']*100:.1f}%)")
                rx.legend(loc='upper left',fontsize=8,frameon=False)
            if family=='TLS' and rr[p,'gtls']['detected_vector']!=rr[p,'gtls_batch']['detected_vector']:
                d=rr[p,'gtls'];rx.plot([q['snr'] for q in d['by_snr']],[q['recall']*100 for q in d['by_snr']],
                    '--',color=COLORS['gtls'],lw=1,alpha=.75,label=f"GTLS single (null FPR {d['false_positive_rate']*100:.1f}%)")
                rx.legend(loc='upper left',fontsize=8,frameon=False)
            rx.set_ylim(-3,103);rx.set_yticks([0,25,50,75,100]);rx.set_xticks([6,8,10,14]);rx.set_xlim(5.4,14.6)
            rx.set_xlabel('Injected white-noise oracle SNR',fontsize=10);rx.set_ylabel('Detected at correct period (%)',fontsize=10)
            rx.grid(alpha=.2);rx.set_axisbelow(True)
            fpr=[rr[p,(m+'_batch' if m in ['bls_v1','gtls'] else m)]['false_positive_rate']*100 for m in methods]
            short=['v1','PyPI','CPU','GPU'] if family=='BLS' else ['v1','GTLS']
            falsealarm=' · '.join(f'{name} {value:.1f}%' for name,value in zip(short,fpr))
            rx.set_title('Held-out recovery · batch null false-positive rates:\n'+falsealarm,loc='left',fontsize=9,pad=5,color='#526270')
    fig.text(.045,.085,'Timing: median (5 single / 3 batch calls); whiskers span repetitions. Ratios: batch time / v1. †: paired tests support <5-point recovery loss and <5-point FPR increase.',fontsize=10,color='#394d5d')
    fig.text(.045,.065,'Unmarked ratios are timing comparisons with sensitivity differences or insufficient evidence of a match. Error bars: 95% Wilson intervals; 32 injections / SNR / survey. Solid curves: batch; dashed: single if different.',fontsize=10,color='#394d5d')
    fig.text(.045,.045,'Real observing times; synthetic integrated transits and Gaussian + correlated noise. Each method: 128 calibration nulls, 128 held-out injections, 128 held-out nulls.',fontsize=10,color='#526270')
    fig.text(.045,.025,'A40 + 7.65 CPU-equivalent allocation, $0.49/hour. Prepared-array searches only; costs are linear projections. Shared grid within each algorithm; BLS / TLS search ranges differ.',fontsize=10,color='#526270')
    for ext in ['png','pdf','svg']:fig.savefig(r/f'benchmark_story.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    print('Wrote benchmark_story.png / .pdf / .svg')


if __name__=='__main__':main()
