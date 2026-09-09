"""Finite-sample summaries for the paired injection experiment."""
import numpy as np
from scipy.stats import beta


def wilson(k,n,z=1.959963984540054):
    if not n:return [None,None]
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,center-half)),float(min(1,center+half))]


def paired(a,b):
    """Conservative one-sided 95% bounds via two 97.5% binomial bounds.

    D = Pr(A only) - Pr(B only). Bonferroni requires no independence
    between these two multinomial cells. Each returned bound separately
    has >=95% coverage; do not call their pair a two-sided 95% interval.
    """
    a=np.asarray(a,bool);b=np.asarray(b,bool);assert a.shape==b.shape
    n=len(a);wins=int(np.sum(a&~b));losses=int(np.sum(~a&b))
    def lo(k):return float(beta.ppf(.025,k,n-k+1)) if k else 0.
    def hi(k):return float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
    return dict(n=n,v1_only=wins,comparator_only=losses,difference=float(a.mean()-b.mean()),
                lower_95_one_sided=lo(wins)-hi(losses),upper_95_one_sided=hi(wins)-lo(losses),
                noninferior_5pp=bool(lo(wins)-hi(losses)>-.05))


def score(case):
    v=case.get('score')
    return float(v) if v is not None and np.isfinite(v) and case.get('api_result_valid',True) else -np.inf


def summarize(calibration,heldout):
    assert len(calibration)==128 and all(not c['injected'] for c in calibration)
    nullscores=np.array([score(c) for c in calibration])
    threshold=float(np.quantile(nullscores,.95,method='higher'))
    if not np.isfinite(threshold):raise ValueError('Calibration did not produce a finite detection threshold')
    injected=[c for c in heldout if c['injected']];nulls=[c for c in heldout if not c['injected']]
    assert len(injected)==128 and len(nulls)==128
    period=np.array([c['recovered'] for c in injected],bool)
    detection=period&np.array([score(c)>threshold for c in injected])
    fp=np.array([score(c)>threshold for c in nulls])
    bins=[]
    for snr in [6.,8.,10.,14.]:
        take=np.array([c['snr']==snr for c in injected]);n=int(take.sum());assert n==32
        k=int(detection[take].sum());kp=int(period[take].sum())
        bins.append(dict(snr=snr,n=n,period_recovered=kp,detected=k,recall=k/n,interval=wilson(k,n),
                         period_recall=kp/n,period_interval=wilson(kp,n)))
    return dict(threshold=threshold,n_injections=len(injected),n_calibration_nulls=len(calibration),n_heldout_nulls=len(nulls),
        period_recovered=int(period.sum()),detected=int(detection.sum()),false_positives=int(fp.sum()),
        period_recall=float(period.mean()),detection_recall=float(detection.mean()),false_positive_rate=float(fp.mean()),
        period_interval=wilson(int(period.sum()),len(period)),detection_interval=wilson(int(detection.sum()),len(detection)),
        false_positive_interval=wilson(int(fp.sum()),len(fp)),by_snr=bins,
        invalid_calibration=sum(not c.get('api_result_valid',True) or bool(c.get('error')) for c in calibration),
        invalid_heldout=sum(not c.get('api_result_valid',True) or bool(c.get('error')) for c in heldout),
        period_recovered_vector=period.tolist(),detected_vector=detection.tolist(),false_positive_vector=fp.tolist())
