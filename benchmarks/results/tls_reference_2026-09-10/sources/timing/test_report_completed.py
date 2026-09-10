"""CPU-only reporting safeguards; original measurement tools are not changed."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

if __package__:
    from . import report_completed as report
else:
    import report_completed as report
from benchmarks.tls_reference.timing import benchmark, cohort, summarize
from benchmarks.tls_reference.timing.test_timing import ownership_receipt, exclusive_snapshot


def fixture(backend='candidate'):
    names = ['a', 'b']
    sources = {name:'source' for name in report.CANDIDATE_FILES}
    selection = dict(selected_cases=names, single_case='a',
        expected_candidate_sources={'cuvarbase/'+name:value for name,value in sources.items()},
        expected_native_sources=sources)
    def output(name):
        fields = dict(value=name)
        return dict(case=name, fields=fields,
            full_digest=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest(),
            strict={'periods':name}, common={'periods':name},
            nperiods=1, primary_period=2., SDE=3.)
    def rep(names):
        return dict(status='ok', errors=[], outputs=[output(name) for name in names],
            source_count=len(names), denominator_seconds=1.,elapsed_seconds=1.,
            exclusive_before=exclusive_snapshot(),exclusive_after=exclusive_snapshot())
    owner=ownership_receipt()
    worker=copy.deepcopy(owner['workers'][0]);worker['sources']={'files':sources}
    owner['workers']=[worker]
    record=dict(status='ok', pool_width=1, workers=[worker], gpu_ownership=owner,
        config=dict(backend=backend,prefix='graph',regime='tess_solar',
            measurement_scope='full',manifest='manifest',names=names),
        warmup=rep(['a']),single=[rep(['a']) for _ in range(5)],batch=[rep(names) for _ in range(3)])
    expected={name:{'strict':output(name)['strict']} for name in names}
    plan=dict(manifest='manifest',cases={'tess_solar':[{'name':name} for name in names]},
        frozen_outputs={'tess_solar':{backend:expected}})
    modules=dict(benchmark=benchmark,cohort=cohort,summarize=summarize)
    return record,selection,plan,modules


def test_complete_mandatory_config_keeps_all_original_gates_and_full_identity():
    r,s,p,f=fixture()
    result=report.audit_config(r,'candidate',1,'tess_solar',s,p,f)
    assert result['complete'] and result['eligible']
    assert result['gates']['single_batch_full_identity']['eligible']


@pytest.mark.parametrize('mutation', ['missing_source','changed_source','case','missing_rep',
    'changed_strict','unstable_full','forged_full_digest','changed_owner','empty_owner',
    'partial_status','short_denominator','worker_count'])
def test_mandatory_configuration_cannot_be_salvaged(mutation):
    r,s,p,f=fixture()
    if mutation=='missing_source': r['workers'][0]['sources']['files'].pop('tls.py')
    elif mutation=='changed_source': r['workers'][0]['sources']['files']['tls.py']='other'
    elif mutation=='case': r['config']['names']=['a','other']
    elif mutation=='missing_rep': r['batch'].pop()
    elif mutation=='changed_strict': r['batch'][0]['outputs'][0]['strict']['periods']='other'
    elif mutation=='unstable_full':
        value=r['batch'][0]['outputs'][0];value['fields']['extra']='changed'
        value['full_digest']=hashlib.sha256(json.dumps(value['fields'],sort_keys=True).encode()).hexdigest()
    elif mutation=='forged_full_digest': r['batch'][0]['outputs'][0]['fields']['extra']='unhashed'
    elif mutation=='changed_owner': r['batch'][0]['exclusive_after']=exclusive_snapshot([202])
    elif mutation=='empty_owner': r['gpu_ownership']['after_start']=exclusive_snapshot([])
    elif mutation=='partial_status': r.pop('status')
    elif mutation=='short_denominator': r['batch'][0]['denominator_seconds']=.5
    elif mutation=='worker_count': r['workers']=[]
    with pytest.raises(ValueError):
        report.audit_config(r,'candidate',1,'tess_solar',s,p,f)


def oom_record():
    record=dict(status='error',config={'names':['a']},single=[],batch=[],
        warmup=dict(status='error',denominator_seconds=None,outputs=[{'case':'a'}]*3,
            errors=[{'case':'a','traceback':'cupy.cuda.memory.OutOfMemoryError: allocation failed'},
                    {'reason':'Post-barrier result membership differs from the assigned inputs',
                     'expected':['a']*4,'observed':['a']*3}]))
    return record


def test_only_warmup_oom_accounted_missing_output_is_a_permitted_optional_failure():
    assert report.optional_warmup_oom(oom_record(),'gtls',4,list(report.ALLOWED_OMISSION_REASONS))


@pytest.mark.parametrize('mutation', ['native_one','candidate','partial_batch','other_error',
    'extra_error','wrong_membership','failure_denominator','owner_failure','missing_oom'])
def test_oom_scope_does_not_waive_other_failures(mutation):
    r=oom_record();backend='gtls';width=4;problems=list(report.ALLOWED_OMISSION_REASONS)
    if mutation=='native_one': width=1
    elif mutation=='candidate': backend='candidate'
    elif mutation=='partial_batch': r['batch']=[{'status':'ok'}]
    elif mutation=='other_error': r['warmup']['errors'][0]['traceback']='ValueError: wrong'
    elif mutation=='extra_error': r['warmup']['errors'].append({'reason':'foreign context'})
    elif mutation=='wrong_membership': r['warmup']['errors'][1]['observed']=['a']*2
    elif mutation=='failure_denominator': r['warmup']['denominator_seconds']=1.
    elif mutation=='owner_failure': problems.append('GPU ownership lifecycle failed')
    elif mutation=='missing_oom': r['warmup']['errors']=r['warmup']['errors'][1:]
    assert not report.optional_warmup_oom(r,backend,width,problems)


def rejection():
    details={'tess_gap':{'gtls_graph_4worker':dict(exclusion='warmup_gpu_out_of_memory',
        problems=sorted(report.ALLOWED_OMISSION_REASONS))}}
    reasons=['tess_gap: gtls_graph_4worker: '+reason for reason in sorted(report.ALLOWED_OMISSION_REASONS)]
    acceptance=dict(status='rejected',publication_gate={'pass':False,'problems':reasons})
    return acceptance,details


def test_original_rejection_must_be_exactly_the_disclosed_optional_failure():
    a,d=rejection()
    assert report.verify_audit_reasons(a,d)==['tess_gap/gtls_graph_4worker']


@pytest.mark.parametrize('mutation',['source','normalization','component','missing_reason','accepted'])
def test_final_audit_traceback_alone_cannot_hide_other_failures(mutation):
    a,d=rejection()
    if mutation=='missing_reason':a['publication_gate']['problems'].pop()
    elif mutation=='accepted':a['publication_gate']['pass']=True
    else:a['publication_gate']['problems'].append('Other failed '+mutation+' gate')
    with pytest.raises(ValueError,match='beyond'):
        report.verify_audit_reasons(a,d)


def terminal_files(tmp_path):
    names=['preflight','components_candidate','components_gtls','public','summarize','normalize']
    stages=[dict(name=name,command=[name],timeout_seconds=60,status='complete',exit_code=0) for name in names]
    report.write(tmp_path/'status.json',dict(status='error',stages=stages))
    report.write(tmp_path/'pipeline-plan.json',dict(stages=stages))
    report.write(tmp_path/'terminal.json',dict(status='error',exit_code=2,
        error='ValueError: Final timing acceptance failed: []'))
    prepared={'selections':{'tess_solar':{'correction_timing':{'required':False}}}}
    return prepared


def test_completed_stages_can_have_failed_final_all_configuration_audit(tmp_path):
    p=terminal_files(tmp_path)
    assert report.check_driver_terminal(tmp_path,p)['exit_code']==2


@pytest.mark.parametrize('mutation',['earlier_failure','missing_stage','extra_stage','command','exit_bool','other_error'])
def test_partial_campaign_or_different_stage_is_rejected(tmp_path,mutation):
    p=terminal_files(tmp_path);d=report.read(tmp_path/'status.json');t=report.read(tmp_path/'terminal.json')
    if mutation=='earlier_failure':d['stages'][0]['exit_code']=1
    elif mutation=='missing_stage':d['stages'].pop()
    elif mutation=='extra_stage':d['stages'].append(d['stages'][0])
    elif mutation=='command':d['stages'][0]['command']=['different']
    elif mutation=='exit_bool':d['stages'][0]['exit_code']=False
    elif mutation=='other_error':t['error']='RuntimeError: public failed'
    report.write(tmp_path/'status.json',d);report.write(tmp_path/'terminal.json',t)
    with pytest.raises(ValueError):report.check_driver_terminal(tmp_path,p)


def collection_files(tmp_path):
    name='timing-continuation/results/timing/acceptance.json'
    path=tmp_path/'files'/name;path.parent.mkdir(parents=True);path.write_text('{}')
    report.write(tmp_path/'file-index.json',{name:dict(size=path.stat().st_size,sha256=report.sha(path))})
    report.write(tmp_path/'outcome.json',dict(final_compact_verified=True,missing_final_paths=[],exit_code=2))
    report.write(tmp_path/'termination.json',dict(verified=True))
    return path


def test_failed_pipeline_can_be_fully_collected(tmp_path):
    collection_files(tmp_path)
    root,receipt=report.verify_collection(tmp_path)
    assert receipt['original_pipeline_exit_code']==2
    assert len(receipt['verified_files'])==1


@pytest.mark.parametrize('mutation',['changed','missing','extra','not_final','not_stopped'])
def test_collection_must_be_complete_and_hash_verified(tmp_path,mutation):
    path=collection_files(tmp_path)
    if mutation=='changed':path.write_text('{"changed":true}')
    elif mutation=='missing':path.unlink()
    elif mutation=='extra':(path.parent/'unindexed.json').write_text('{}')
    elif mutation=='not_final':report.write(tmp_path/'outcome.json',dict(final_compact_verified=False,missing_final_paths=[]))
    elif mutation=='not_stopped':report.write(tmp_path/'termination.json',dict(verified=False))
    with pytest.raises(ValueError):report.verify_collection(tmp_path)
