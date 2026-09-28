#!/usr/bin/env python3
"""Native BLS execution supplement on a newly bound allocation.

Uses the preserved scientific implementation and error-accounting runner.
Records every discrepancy; does not change the original failed qualification.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hourly-usd',type=float,required=True)
    parser.add_argument('--max-hours',type=float,default=2)
    args=parser.parse_args()
    root=args.root.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    source=root/'sources/candidate'
    sys.path.insert(0,str(source))
    from benchmarks.tls_survey import bls_execution_throughput as supplement
    from benchmarks.tls_survey import throughput as native
    signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(supplement.Interrupted('Controller terminated')))
    protocol=source/'benchmarks/tls_survey/FOLLOWUP_20260924.md'
    protocol.write_bytes((root/'harness/FOLLOWUP_20260924.md').read_bytes())
    allocation=supplement.allocation(native.resource_environment())
    deadline=time.time()+args.max_hours*3600
    plan=dict(purpose='New-allocation BLS execution supplement; original qualification remains failed',
              source_sha256=native.sha(__file__),protocol_sha256=native.sha(protocol),
              science_seal_sha256=native.sha(root/'evidence/science-seal.json'),
              allocation=list(allocation),hourly_usd=args.hourly_usd,deadline_epoch=deadline,
              workers=[1,2,4],batches=[1,4,8],development_cases_per_regime=8,
              measurement_repetitions=3,minimum_attempts=96,minimum_seconds=120,
              input_manifest_sha256={name:native.sha(root/'inputs'/name/'manifest.json')
                                     for name in ('development','nulls','varied')},
              original_numerical_qualification_passed=False)
    native.write(args.output/'plan.json',plan)
    parameters=SimpleNamespace(deadline_epoch=deadline,protocol=protocol,
                 science_seal=root/'evidence/science-seal.json',source_root=source,
                 hourly_usd=args.hourly_usd,
                 authorization_identity=dict(followup_plan_sha256=native.sha(args.output/'plan.json'),
                                             allocation_basis='This new allocation, recorded before tuning'))
    campaign=dict(plan_sha256=native.sha(args.output/'plan.json'),status='running',configs=[],selected=None,panels=[])
    records=[]
    def run(manifest,names,scope,workers,batch,label,anchors=None,repetitions=1,attempts=24,seconds=30):
        cases=native.load_manifest(manifest,names)
        native.configure_bls(cases,parameters.science_seal)
        record=supplement.run_configuration(parameters,manifest,names,scope,workers,batch,
                    args.output/label,allocation,supplement.cohort_identity(cases),anchors,
                    repetitions,attempts,seconds)
        campaign['configs'].append(dict(label=label,status=record['status'],
                   execution_rates_valid=record['execution_rates_valid'],
                   result_sha256=native.sha(args.output/label/'result.json'),
                   numerical=record['numerical'],summary=record.get('summary')))
        native.write(args.output/'campaign.json',campaign)
        print(json.dumps(dict(label=label,status=record['status'],summary=record.get('summary'),
                              numerical=record['numerical'])),flush=True)
        return record
    development=root/'inputs/development/manifest.json'
    names=[row['file'] for row in json.loads(development.read_text())['cases']]
    anchors=None
    for workers in plan['workers']:
        record=run(development,names,'development',workers,1,f'tune-w{workers}-b1',anchors)
        records.append(record)
        if workers==1:
            anchors=record.get('fixed_reference_anchors',{})
    worker=supplement.execution_winner(records)
    if worker is None:
        campaign.update(status='complete_with_unavailable_panels',reason='No operational development winner')
        native.write(args.output/'campaign.json',campaign)
        return 1
    for batch in (4,8):
        records.append(run(development,names,'development',worker['workers'],batch,
                           f'tune-w{worker["workers"]}-b{batch}',anchors))
    winner=supplement.execution_winner(records)
    campaign['selected']=dict(workers=winner['workers'],batch_size=winner['batch_size'])
    native.write(args.output/'tuning.json',dict(plan_sha256=campaign['plan_sha256'],selected=campaign['selected'],
                                             configs=list(campaign['configs'])))
    for scope in ('tess_solar','tess_gap_long','ztf_solar','varied'):
        manifest=root/'inputs'/('varied' if scope=='varied' else 'nulls')/'manifest.json'
        rows=json.loads(manifest.read_text())['cases']
        names=[r['file'] for r in rows] if scope=='varied' else [r['file'] for r in rows if r['metadata']['regime']==scope][:16]
        reference=run(manifest,names,scope,1,1,'reference-'+scope,repetitions=1,attempts=len(names),seconds=0)
        result=run(manifest,names,scope,winner['workers'],winner['batch_size'],'measure-'+scope,
                   reference.get('fixed_reference_anchors',{}),repetitions=3,attempts=96,seconds=120)
        campaign['panels'].append(dict(scope=scope,execution_rates_valid=result['execution_rates_valid'],
                                      summary=result.get('summary'),numerical=result['numerical']))
        native.write(args.output/'campaign.json',campaign)
    campaign['status']='complete'
    native.write(args.output/'campaign.json',campaign)
    return 0


if __name__=='__main__':
    sys.exit(main())
