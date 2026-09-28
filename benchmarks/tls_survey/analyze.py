#!/usr/bin/env python3
"""Select BLS on development, freeze tolerances, calibrate, then analyze holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from common import BLS_CONFIGS,REGIMES,SNRS,now,sha,source_identity,write,method_applicable


def binomial_interval(k,n,alpha=.05):
    if n<1:
        return [0.,1.]
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]


def paired_interval(first,second,alpha=.05):
    """Conservative simultaneous binomial bounds on discordant proportions."""
    a,b=np.asarray(first,bool),np.asarray(second,bool)
    if a.shape!=b.shape or a.ndim!=1 or len(a)==0:
        raise ValueError('Paired nonempty aligned binary arrays required')
    wins=int(np.sum(a&~b));losses=int(np.sum(~a&b));n=len(a)
    win=binomial_interval(wins,n,alpha/2)
    loss=binomial_interval(losses,n,alpha/2)
    return dict(n=n,first_only=wins,second_only=losses,difference=(wins-losses)/n,
                interval=[win[0]-loss[1],win[1]-loss[0]],confidence=1-alpha,
                construction='Bonferroni exact binomial bounds on the two discordant probabilities')


def threshold(scores,alpha=.05):
    """Split-conformal order statistic: strict exceedance, no interpolation.

    Marginal exchangeable-null FPR <= alpha, with attainable resolution 1/(n+1).
    This is not a confidence guarantee on the realized conditional threshold.
    """
    scores=np.asarray(scores,float)
    if len(scores)==0 or not np.all(np.isfinite(scores)):
        raise ValueError('Calibration requires every planned finite valid null score')
    rank=int(np.ceil((len(scores)+1)*(1-alpha)))
    if rank>len(scores):
        raise ValueError('Insufficient nulls for this finite threshold/FPR')
    value=float(np.sort(scores)[rank-1])
    above=int(np.sum(scores>value))
    tied=int(np.sum(scores==value))
    return dict(value=value,n=len(scores),rank_1based=rank,
                calibration_scores_above=above,calibration_scores_at_threshold=tied,
                calibration_zero_scores=int(np.sum(scores==0.)),
                calibration_strict_exceedance_fraction=above/len(scores),
                extra_conservatism_from_ties=above<len(scores)-rank,
                marginal_fpr_upper_bound=(len(scores)+1-rank)/(len(scores)+1),
                target_fpr=alpha,attainable_marginal_fpr=(len(scores)+1-rank)/(len(scores)+1),
                decision='strict exceedance',calibration='independent split-conformal order statistic')


def records(paths,split):
    rows={}
    receipts=[]
    for path in paths:
        r=json.loads(Path(path).read_text())
        if r['status']!='complete' or r['split']!=split:
            raise ValueError('Wrong split or incomplete execution receipt: '+str(path))
        receipts.append(dict(path=str(path),sha256=sha(path),manifest_sha256=r['manifest_sha256'],
                             production_sources=r['production_sources'],split=split,
                             runner_sha256=r.get('runner_sha256'),methods=r.get('methods'),
                             shard_count=r.get('shard_count'),inapplicable=r.get('inapplicable',[])))
        if 'planned_inputs' not in r:
            raise ValueError('Receipt lacks planned input identities')
        planned={v['name']:v['sha256'] for v in r['planned_inputs']}
        if len(planned)!=r['planned_cases']:
            raise ValueError('Duplicate or incomplete planned execution inputs')
        for row in r['cases']:
            if planned.get(row['name'])!=row['input_sha256']:
                raise ValueError('Executed input differs from planned identity')
            key=(row['name'],row['method'])
            if key in rows:
                raise ValueError('Duplicate measured case/method across shards')
            rows[key]=row
    return list(rows.values()),receipts


def by_method(rows,regime,method):
    return sorted([r for r in rows if r['regime']==regime and r['method']==method],key=lambda r:r['name'])


def score(row,ranker):
    if not row['valid'] or row['candidates'].get(ranker,{}).get('score') is None:
        return -np.inf
    return row['candidates'][ranker]['score']


def detections(rows,ranker,cut,*,null=False):
    return np.array([r['valid'] and score(r,ranker)>cut and
                     (null or r['candidates'][ranker]['recovered']) for r in rows],bool)


def require_count(rows,expected,where):
    if len(rows)!=expected:
        raise ValueError('%s: expected %d cases, got %d'%(where,expected,len(rows)))


def require_paired(a,b):
    if [r['input_sha256'] for r in a]!=[r['input_sha256'] for r in b]:
        raise ValueError('Paired methods did not receive identical ordered lightcurves')


def freeze(args):
    dev,dev_receipts=records(args.development,'development')
    nulls,null_receipts=records(args.development_nulls,'development_nulls')
    diag=json.loads(args.snr.read_text())
    if any(r['manifest_sha256']!=diag['manifest_sha256'] for r in dev_receipts):
        raise ValueError('Expected-SNR and blind development manifests differ')
    regimes=args.regimes.split(',')
    selected={};tol={};development=[]
    identities=[r['production_sources'] for r in dev_receipts+null_receipts]
    if any(v!=identities[0] for v in identities):
        raise ValueError('Development numerical sources differ across receipts')
    for regime in regimes:
        candidates=[]
        tdev=by_method(dev,regime,'tls');tnull=by_method(nulls,regime,'tls')
        if not tdev or not tnull:
            raise ValueError('TLS missing development regime: '+regime)
        if not all(r['valid'] for r in tnull):
            raise ValueError('Invalid TLS development nulls prevent calibration')
        tcut=threshold([score(r,'native') for r in tnull],args.fpr)
        td=detections(tdev,'native',tcut['value'])
        for method in BLS_CONFIGS:
            if not method_applicable(method,regime):
                continue
            bdev=by_method(dev,regime,method);bnull=by_method(nulls,regime,method)
            require_paired(tdev,bdev);require_paired(tnull,bnull)
            if not all(r['valid'] for r in bnull):
                continue
            for ranker in ('raw','likelihood','detrended'):
                if not all(np.isfinite(score(r,ranker)) for r in bnull):
                    continue
                cut=threshold([score(r,ranker) for r in bnull],args.fpr)
                detected=detections(bdev,ranker,cut['value'])
                # Fidelity tie-break favors finest duration/epoch resolution,
                # then delta-chi2 likelihood ranking; speed never weakens the control.
                candidates.append(dict(method=method,ranker=ranker,detected=int(detected.sum()),
                    n=len(detected),threshold=cut,paired_tls_minus_bls=paired_interval(td,detected),
                    median_search_s=float(np.median([r['elapsed_s'] for r in bdev]))))
        if not candidates:
            raise ValueError('No complete BLS development comparison: '+regime)
        winner=max(candidates,key=lambda c:(c['detected'],BLS_CONFIGS[c['method']]['noverlap'],{'detrended':0,'raw':1,'likelihood':2}[c['ranker']]))
        selected[regime]={k:winner[k] for k in ('method','ranker')}
        sr=[r for r in diag['rows'] if r['regime']==regime]
        if len(sr)!=len(tdev):
            raise ValueError('Expected-SNR and blind development populations differ')
        snr_vectors={k:np.array([r[k] for r in sr if r[k] is not None],float)
                     for k in ('native_white_advantage','native_ou_advantage')}
        # Protect every development example and both covariance diagnostics;
        # an uncertain/nonpositive advantage gives no approximation allowance.
        rng=np.random.default_rng(142091)
        lowers={}
        for key,values in snr_vectors.items():
            if len(values)!=len(sr) or np.min(values)<=0:
                lowers[key]=0.
            else:
                means=np.mean(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
                lowers[key]=max(0.,float(np.quantile(means,.05)))
        snr_advantage=min(lowers.values())
        recovery_advantage=max(0.,winner['paired_tls_minus_bls']['interval'][0])
        tol[regime]=dict(expected_snr_fractional_loss=min(.001,.05*snr_advantage),
            recovery_absolute_probability_loss=min(.001,.05*recovery_advantage),
            snr_demonstrated_advantage_lower=snr_advantage,
            recovery_demonstrated_advantage_lower=recovery_advantage,
            observed_development_white_median=float(np.median(snr_vectors['native_white_advantage'])),
            observed_development_ou_median=float(np.median(snr_vectors['native_ou_advantage'])),
            fpr_absolute_increase_max=0.,fpr_absolute_cap=.001)
        development.append(dict(regime=regime,tls=dict(detected=int(td.sum()),n=len(td),threshold=tcut),
                                bls_candidates=candidates,selected=winner))
    value=dict(schema_version=1,created_utc=now(),source_identity=source_identity(),regimes=regimes,
        counts=dict(calibration=args.calibration_count,injections=args.injection_count,nulls=args.null_count),
        exposure_nodes=args.exposure_nodes,target_fpr=args.fpr,secondary_target_fpr=.01,bls_selected=selected,
        primary_advantage='Blind calibrated recovery; expected-SNR family ceilings are explanatory, not actual native-search sensitivity.',
        tolerance_rule='At most 5% of a demonstrated positive TLS advantage, capped at 0.1% fractional expected SNR and 0.1 percentage point recovery/FPR. Any nonpositive/uncertain subgroup advantage gives zero allowance.',
        tolerances=tol,production_sources=dev_receipts[0]['production_sources'],execution_shards=args.execution_shards,
        production_acceptance=dict(policy='exact only',removed_trials_allowed=0,
            changed_valid_masks_allowed=0,changed_candidate_or_detection_decisions_allowed=0,
            approximate_screening='No screening before an unconditional full observation-level fallback',
            numerical='Full spectra and fits must match reference on qualification inputs. Shared float32 prefix variability is recorded and never silently widened.'),
        calibration_policy='Separate independently generated nulls per regime and per method. Strict order-statistic thresholds. Independent test nulls report realized FPR uncertainty.',
        recovery_policy='Primary period drift over baseline <= half physical contact duration; aliases separately descriptive; unsampled/one-event cases and failures remain denominator.',
        heldout_analysis_policy='No changes to settings, seeds, thresholds, counts, endpoints, or tolerances after heldout results. Marginal exact intervals plus simultaneous regime paired bounds; pilot may be inconclusive.',
        development_receipts=dev_receipts+null_receipts,development_snr_sha256=sha(args.snr),development=development)
    if args.out.exists():
        raise ValueError('Refuse to overwrite a frozen seal')
    write(args.out,value)


def calibrate(args):
    seal=json.loads(args.seal.read_text())
    rows,receipts=records(args.results,'calibration')
    if any(r['production_sources']!=seal['production_sources'] for r in receipts):
        raise ValueError('Calibration numerical sources differ from frozen design')
    thresholds={};secondary={}
    for regime in seal['regimes']:
        for label,method,ranker in [('tls','tls','native'),('bls',seal['bls_selected'][regime]['method'],seal['bls_selected'][regime]['ranker'])]:
            population=by_method(rows,regime,method)
            require_count(population,seal['counts']['calibration'],regime+'/'+label)
            if not all(r['valid'] for r in population):
                raise ValueError('Failed calibration nulls: '+regime+'/'+label)
            thresholds[regime+'/'+label]=threshold([score(r,ranker) for r in population],seal['target_fpr'])
            secondary[regime+'/'+label]=threshold([score(r,ranker) for r in population],seal['secondary_target_fpr'])
    if args.out.exists():
        raise ValueError('Refuse to overwrite independently frozen thresholds')
    write(args.out,dict(created_utc=now(),seal_sha256=sha(args.seal),thresholds=thresholds,secondary_thresholds=secondary,receipts=receipts))


def analyze(args):
    seal=json.loads(args.seal.read_text());cuts=json.loads(args.thresholds.read_text())
    if cuts['seal_sha256']!=sha(args.seal):
        raise ValueError('Thresholds belong to another design')
    inj,ireceipts=records(args.injections,'injections');null,nreceipts=records(args.nulls,'nulls')
    for receipt in cuts['receipts']+ireceipts+nreceipts:
        if receipt['production_sources']!=seal['production_sources']:
            raise ValueError('Execution numerical sources differ from frozen design')
    tables=[];contrasts=[]
    for target_fpr,point_cuts in ((seal['target_fpr'],cuts['thresholds']),
                                 (seal['secondary_target_fpr'],cuts['secondary_thresholds'])):
        for regime in seal['regimes']:
            vectors={}
            populations={}
            for label,method,ranker in [('tls','tls','native'),('bls',seal['bls_selected'][regime]['method'],seal['bls_selected'][regime]['ranker'])]:
                ir=by_method(inj,regime,method);nr=by_method(null,regime,method)
                require_count(ir,seal['counts']['injections'],regime+'/'+label+'/injections')
                require_count(nr,seal['counts']['nulls'],regime+'/'+label+'/nulls')
                cut=point_cuts[regime+'/'+label]['value']
                d=detections(ir,ranker,cut);fp=detections(nr,ranker,cut,null=True)
                vectors[label]=(d,fp);populations[label]=(ir,nr)
                strata=[]
                for kind in ('snr','sampling'):
                    levels=SNRS if kind=='snr' else ('unsampled','one_event','two_events','three_plus_events','one_to_four_points','grid_unreachable')
                    for level in levels:
                        if kind=='snr':
                            take=np.array([r['white_oracle_snr']==level for r in ir])
                        else:
                            take=np.array([not r.get('grid_reachable',True) if level=='grid_unreachable' else r['in_transit_observations']==0 if level=='unsampled' else
                                r['observed_events']==1 if level=='one_event' else r['observed_events']==2 if level=='two_events' else
                                r['observed_events']>=3 if level=='three_plus_events' else 0<r['in_transit_observations']<5 for r in ir])
                        n=int(take.sum());k=int(d[take].sum())
                        if n:
                            strata.append(dict(kind=kind,level=level,n=n,detected=k,interval95=binomial_interval(k,n)))
                alias=sum(r['valid'] and score(r,ranker)>cut and r['candidates'][ranker]['alias_recovered'] for r in ir)
                tables.append(dict(regime=regime,target_fpr=target_fpr,method=label,configuration=method,ranker=ranker,threshold=cut,
                    calibration=point_cuts[regime+'/'+label],
                    detected=int(d.sum()),n_injections=len(d),recovery=float(d.mean()),recovery_interval95=binomial_interval(int(d.sum()),len(d)),
                    aliases_including_fundamental=int(alias),false_positives=int(fp.sum()),n_nulls=len(fp),fpr=float(fp.mean()),
                    fpr_interval95=binomial_interval(int(fp.sum()),len(fp)),failed_injections=sum(not r['valid'] for r in ir),
                    failed_nulls=sum(not r['valid'] for r in nr),strata=strata))
            require_paired(populations['tls'][0],populations['bls'][0]);require_paired(populations['tls'][1],populations['bls'][1])
            contrasts.append(dict(regime=regime,target_fpr=target_fpr,tls_minus_bls_recovery=paired_interval(vectors['tls'][0],vectors['bls'][0]),
                tls_minus_bls_recovery_simultaneous=paired_interval(vectors['tls'][0],vectors['bls'][0],.05/(4*len(seal['regimes']))),
                tls_minus_bls_fpr=paired_interval(vectors['tls'][1],vectors['bls'][1]),
                tls_minus_bls_fpr_simultaneous=paired_interval(vectors['tls'][1],vectors['bls'][1],.05/(4*len(seal['regimes'])))))
    write(args.out,dict(created_utc=now(),seal_sha256=sha(args.seal),thresholds_sha256=sha(args.thresholds),
        methods=tables,contrasts=contrasts,receipts=ireceipts+nreceipts,
        limitation='Finite synthetic-flux population on fixed observed or synthetic cadences. Exact implementation qualification is separate. No universal completeness or sub-percentage noninferiority established. Marginal calibrated target FPR is not certainty about realized conditional FPR.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--development',nargs='+',type=Path,required=True)
    f.add_argument('--development-nulls',nargs='+',type=Path,required=True);f.add_argument('--snr',type=Path,required=True)
    f.add_argument('--regimes',default=','.join(REGIMES));f.add_argument('--calibration-count',type=int,default=512)
    f.add_argument('--injection-count',type=int,default=256);f.add_argument('--null-count',type=int,default=256)
    f.add_argument('--execution-shards',type=int,default=4)
    f.add_argument('--exposure-nodes',type=int,default=64);f.add_argument('--fpr',type=float,default=.05)
    c=sub.add_parser('calibrate');c.add_argument('--seal',type=Path,required=True);c.add_argument('--results',nargs='+',type=Path,required=True)
    a=sub.add_parser('analyze');a.add_argument('--seal',type=Path,required=True);a.add_argument('--thresholds',type=Path,required=True)
    a.add_argument('--injections',nargs='+',type=Path,required=True);a.add_argument('--nulls',nargs='+',type=Path,required=True)
    for cmd in (f,c,a):cmd.add_argument('--out',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':
    main()
