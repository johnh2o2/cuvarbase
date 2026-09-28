"""CPU-only error-accounting and provenance regressions for the separate supplement."""
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import time

import numpy as np
import pytest

from benchmarks.tls_survey import bls_execution_throughput as m


def observation(name='a', score=10., fail=False):
    if fail:
        return dict(case=name,status='api_error',error='native failure')
    values=dict(period=2.,score=score)
    return dict(case=name,status='success',scalar=dict(values=values,
                fields=m.native.scalar_fingerprint('bls',values)))


def test_failed_calls_reduce_completion_rate_without_removing_attempts():
    cases=[dict(name=n) for n in ('a','b','c')]
    row=dict(indices=[0,1,2],observations=[observation('a'),observation('b',fail=True),observation('c')])
    row['accounting']=m.account_task(row,[0,1,2],cases,{})
    summary=m.queue_summary([row],4.)
    assert summary['attempted_count']==3
    assert summary['successful_count']==2
    assert summary['failed_count']==1
    assert summary['successful_lightcurves_per_second']==.5
    assert summary['attempted_lightcurves_per_second']==.75
    assert row['observations'][1]['error']=='native failure'


def test_selected_score_drift_is_retained_as_success_without_numerical_pass():
    old=observation(score=138.0975799560547)
    changed=observation(score=138.09754943847656)
    row=dict(indices=[0],observations=[changed])
    row['accounting']=m.account_task(row,[0],[dict(name='a')],{'a':old})
    comparison=row['accounting']['comparisons'][0]
    assert row['accounting']['successful_count']==1
    assert comparison['changed_selected_fields']==['score']
    assert comparison['differences']['score']==-3.0517578125e-5
    assert comparison['expected']['score']==138.0975799560547
    numeric=m.numerical_summary([], [dict(tasks=[row])])
    assert numeric['original_qualification_passed'] is False
    assert numeric['selected_mismatch_count']==1


def test_matching_supplement_cannot_reclassify_original_failure():
    same=observation()
    row=dict(indices=[0],observations=[same])
    row['accounting']=m.account_task(row,[0],[dict(name='a')],{'a':same})
    numeric=m.numerical_summary([], [dict(tasks=[row])])
    assert numeric['selected_mismatch_count']==0
    assert numeric['original_qualification_passed'] is False


def test_missing_first_reference_never_replaced_by_later_success():
    rows=[dict(worker=0,observations=[observation(fail=True)]),
          dict(worker=1,observations=[observation()]),
          dict(worker=0,observations=[observation()])]
    assert m.first_anchors(rows)=={}
    assert m.compare_observation(observation(),None)['comparison'].endswith('unavailable')


@pytest.mark.parametrize('change',['indices','missing','duplicate','foreign','status','nonfinite'])
def test_instrumental_membership_and_output_faults_invalidate(change):
    row=dict(indices=[0],observations=[observation()])
    if change=='indices':row['indices']=[1]
    if change=='missing':row['observations']=[]
    if change=='duplicate':row['observations']*=2
    if change=='foreign':row['observations'][0]['case']='other'
    if change=='status':row['observations'][0]['status']='instrument_failure'
    if change=='nonfinite':row['observations'][0]['scalar']['values']['score']=float('nan')
    with pytest.raises(ValueError):m.account_task(row,[0],[dict(name='a')],{})


def test_diagnostics_require_every_assigned_worker_batch_once():
    row=dict(worker=0,indices=[0],observations=[observation()])
    cases=[dict(name='a')]
    m.validate_diagnostic_coverage([row],[[0]],1,cases)
    with pytest.raises(ValueError):m.validate_diagnostic_coverage([row],[[0]],2,cases)
    with pytest.raises(ValueError):m.validate_diagnostic_coverage([row,row],[[0]],1,cases)


def test_fast_instrument_failure_cannot_win_but_native_failure_penalizes_rate():
    def result(rate,workers=1,valid=True):
        return dict(execution_rates_valid=valid,workers=workers,batch_size=1,
                    summary=dict(successful_lightcurves_per_second=rate,
                                 median_repetition_successful_lightcurves_per_second=rate))
    assert m.execution_winner([result(999,valid=False),result(2),result(1,2)])['workers']==1
    assert m.execution_winner([result(2,2),result(2)])['workers']==1
    assert m.execution_winner([result(0)]) is None


def test_queue_completes_fixed_cycles_even_when_every_call_fails(monkeypatch,tmp_path):
    cases=[dict(name=n) for n in ('a','b','c')]
    class Connection:
        def send(self,command):
            assert not getattr(self,'command',None)
            self.command=command
    pool=object.__new__(m.Pool)
    pool.connections=[Connection(),Connection()]
    pool.ownership=SimpleNamespace(allowed_pids=[10,20])
    pool.timeout=1
    pool.deadline=time.time()+1000
    def receive(connection):
        command,connection.command=connection.command,None
        return dict(task=command['task'],indices=command['indices'],repetition=command['repetition'],
                    observations=[observation(cases[i]['name'],fail=True) for i in command['indices']])
    pool.receive=receive
    monkeypatch.setattr(m,'wait',lambda pending,timeout:pending[:1])
    monkeypatch.setattr(m.native,'exclusive_gpu_processes',lambda unused:dict(exclusive=True))
    result=pool.queue([[0,1],[2]],cases,7,0,{},tmp_path/'journal.jsonl')
    assert result['attempted_count']==9
    assert result['successful_count']==0 and result['failed_count']==9
    assert result['completed_input_cycles']==3
    assert result['status']=='completed_queue'
    assert result['successful_lightcurves_per_second']==0
    assert len((tmp_path/'journal.jsonl').read_text().splitlines())==6


def fake_worker(monkeypatch,tmp_path,*,diagnostic_failure=False):
    monkeypatch.setattr(sys,'path',list(sys.path))
    package=tmp_path/'cuvarbase'
    package.mkdir()
    (package/'__init__.py').write_text('')
    cv=ModuleType('cuvarbase');cv.__file__=str(package/'__init__.py')
    base=ModuleType('cuvarbase.base');base.ensure_context=lambda:None
    cp=ModuleType('cupy');cp.cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda:None))
    monkeypatch.setitem(sys.modules,'cuvarbase',cv)
    monkeypatch.setitem(sys.modules,'cuvarbase.base',base)
    monkeypatch.setitem(sys.modules,'cupy',cp)
    cases=[dict(name=n,input_sha256=n) for n in ('a','b','c')]
    monkeypatch.setattr(m.native,'load_manifest',lambda *unused:cases)
    monkeypatch.setattr(m.native,'configure_bls',lambda *unused:dict(production_sources={}))
    monkeypatch.setattr(m.native,'prepare_grids',lambda *unused:[])
    monkeypatch.setattr(m.native,'science_bls_module',lambda:SimpleNamespace(production_identity=lambda:{}))
    monkeypatch.setattr(m.native,'retain_cuda_context',lambda:object())
    monkeypatch.setattr(m.native,'process_ids',lambda:{})
    def compact(case,science,arrays):
        if case['name']=='b':raise RuntimeError('native b failure')
        return dict(period=2.,score=10.)
    monkeypatch.setattr(m.native,'compact_bls',compact)
    if diagnostic_failure:
        monkeypatch.setattr(m.native,'scalar_fingerprint',lambda *unused:(_ for _ in ()).throw(ValueError('instrument broke')))
    class Connection:
        def __init__(self):self.messages=[];self.commands=iter([dict(kind='run',task=7,indices=[0,1,2]),dict(kind='close')])
        def send(self,message):self.messages.append(message)
        def recv(self):return next(self.commands)
        def close(self):pass
    connection=Connection()
    m.worker(connection,dict(source_root=str(tmp_path),manifest='unused',names=[],science_seal='unused',
                             output=str(tmp_path),deadline_epoch=time.time()+1000))
    journal=[json.loads(line) for line in next(tmp_path.glob('worker-*-attempts.jsonl')).read_text().splitlines()]
    return connection.messages,journal


def test_worker_attempts_remaining_batch_after_api_failure_and_journals_each(monkeypatch,tmp_path):
    messages,journal=fake_worker(monkeypatch,tmp_path)
    observations=messages[-1]['observations']
    assert [v['case'] for v in observations]==['a','b','c']
    assert [v['status'] for v in observations]==['success','api_error','success']
    assert [v['case'] for v in journal if v['event']=='attempt_started']==['a','b','c']
    assert len([v for v in journal if v['event']=='attempt_completed'])==3


def test_worker_fatal_diagnostic_preserves_started_and_returned_native_output(monkeypatch,tmp_path):
    messages,journal=fake_worker(monkeypatch,tmp_path,diagnostic_failure=True)
    assert messages[-1]['kind']=='fatal'
    assert [v['event'] for v in journal]==['attempt_started','api_returned','worker_interrupted']
    assert journal[1]['score']==10.


def test_full_power_variability_retained_even_when_selected_score_same(tmp_path):
    case=dict(name='a',data=dict(periods=np.array([1.,2.])))
    def full(power,label):
        result=dict(period=2.,score=10.,candidates={'raw':dict(period=2.,score=10.)},
                    spectra={k:m.native.array_hash(v) for k,v in dict(periods=np.array([1.,2.]),
                        power=np.array(power),valid_mask=np.ones(2,dtype=bool)).items()},
                    _arrays=dict(periods=np.array([1.,2.]),power=np.array(power)))
        m.native.archive_bls_spectra(result,tmp_path,label)
        obs=observation()
        obs['full']=m.native.complete_fingerprint('bls',case,result)
        return obs
    old=full([1.,10.],'old.npz');new=full([1.0001,10.],'new.npz')
    result=m.compare_observation(new,old)
    assert result['changed_selected_fields']==[]
    assert result['changed_finite_power_values']==1
    assert result['changed_complete_fields']==['spectra']
    assert result['maximum_absolute_power_difference']>0
    Path(old['full']['spectrum_artifact']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='spectrum changed'):m.compare_observation(new,old)


def test_deadline_reserves_cleanup_and_never_shrinks_queue():
    with pytest.raises(m.Interrupted):m.deadline_check(time.time()+119)
    m.deadline_check(time.time()+121)


def authorization_fixture(monkeypatch,tmp_path):
    def save(name,value):
        path=tmp_path/name
        path.write_text(json.dumps(value))
        return path
    runner=save('runner.py',{'source':'unchanged'})
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'source_identity',lambda protocol:{'runner.py':m.native.sha(runner)})
    science=save('science.json',{'scientific':'seal'})
    tuning=save('primary-tuning.json',{'original':'tuning'})
    state=save('state.json',{'status':'complete'})
    bundle=save('bundle.json',{'archive':'verified'})
    varied=save('varied.json',{'cases':[]})
    result=save('result.json',{'cohort':[dict(name='a',input_sha256='input')],
                             'environment':dict(cpu_quota_cores=7.65)})
    measurement=save('primary-measurement.json',dict(status='complete',stage='measure',
        science_seal_sha256=m.native.sha(science),manifest_sha256='original-null-manifest',
        varied_manifest=str(varied),varied_manifest_sha256=m.native.sha(varied),
        configs=[dict(scope='varied',result='result.json',result_sha256=m.native.sha(result))]))
    seal=save('supplement-seal.json',dict(schema=1,kind='native_bls_execution_supplement',
        science_seal_sha256=m.native.sha(science),auxiliary_plan_sha256='reviewed-aux',
        remote_files={str(runner):m.native.sha(runner),str(tuning):m.native.sha(tuning)},
        budget=dict(gpu_cap_seconds=3600,cleanup_reserve_seconds=120),
        binding_rule=dict(primary_tuning_path=str(tuning),primary_measurement_path=str(measurement),
                          primary_state_path=str(state),primary_bundle_receipt_path=str(bundle))))
    contents=json.loads(result.read_text())
    binding=save('binding.json',dict(schema=1,supplement_seal_sha256=m.native.sha(seal),
        science_seal_sha256=m.native.sha(science),auxiliary_plan_sha256='reviewed-aux',
        primary_tuning_sha256=m.native.sha(tuning),primary_measurement_sha256=m.native.sha(measurement),
        primary_state_sha256=m.native.sha(state),primary_bundle_receipt_sha256=m.native.sha(bundle),
        primary_measurement_manifest_sha256='original-null-manifest',
        primary_measurement_varied_manifest_sha256=m.native.sha(varied),
        primary_configs=[dict(scope='varied',result='result.json',result_sha256=m.native.sha(result),
            cohort_sha256=m.canonical_sha(contents['cohort']),environment_sha256=m.canonical_sha(contents['environment']))]))
    args=SimpleNamespace(supplement_seal=seal,supplement_seal_sha256=m.native.sha(seal),
        supplement_binding=binding,supplement_binding_sha256=m.native.sha(binding),science_seal=science,
        protocol=tmp_path/'protocol.md',primary_tuning=tuning,primary_measurement=measurement)
    return args


def test_authorization_copies_both_seal_layers_and_primary_identities(monkeypatch,tmp_path):
    args=authorization_fixture(monkeypatch,tmp_path)
    identity=m.verify_authorization(args)
    assert identity['supplement_binding_sha256']==args.supplement_binding_sha256
    assert identity['primary_measurement_sha256']==m.native.sha(args.primary_measurement)
    assert identity['auxiliary_plan_sha256']=='reviewed-aux'


@pytest.mark.parametrize('changed',['runner.py','primary-tuning.json','primary-measurement.json',
                                    'state.json','bundle.json','varied.json','result.json'])
def test_authorization_rejects_changed_source_or_primary_evidence(monkeypatch,tmp_path,changed):
    args=authorization_fixture(monkeypatch,tmp_path)
    (tmp_path/changed).write_text('{}')
    with pytest.raises(ValueError):m.verify_authorization(args)


def test_binding_cannot_replace_original_cohort_even_with_updated_own_hash(monkeypatch,tmp_path):
    args=authorization_fixture(monkeypatch,tmp_path)
    altered=json.loads(args.supplement_binding.read_text())
    altered['primary_configs'][0]['cohort_sha256']='different-cohort'
    args.supplement_binding.write_text(json.dumps(altered))
    args.supplement_binding_sha256=m.native.sha(args.supplement_binding)
    with pytest.raises(ValueError,match='cohort/resource'):m.verify_authorization(args)


def test_complete_tune_freezes_five_configs_and_refuses_rerun(monkeypatch,tmp_path):
    science=tmp_path/'science.json';science.write_text('{}')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    old=tmp_path/'original.json';old.write_text('{}')
    primary=tmp_path/'primary.json'
    primary.write_text(json.dumps(dict(status='complete',stage='tune',
        science_seal_sha256=m.native.sha(science),manifest_sha256=m.native.sha(manifest),names=['a'],
        configs=[dict(id='bls-mixed-w1-b1',eligible=False,result='original.json',result_sha256=m.native.sha(old))])))
    identity=dict(supplement_seal_sha256='seal',supplement_binding_sha256='binding',
                  science_seal_sha256=m.native.sha(science),auxiliary_plan_sha256='aux',
                  primary_tuning_sha256=m.native.sha(primary),primary_measurement_sha256='measure')
    monkeypatch.setattr(m,'verify_authorization',lambda args:identity)
    monkeypatch.setattr(m,'source_identity',lambda protocol:{'new-runner':'reviewed'})
    monkeypatch.setattr(m,'primary_panel',lambda *unused:dict(cohort=[dict(name='a')],config=dict(hourly_usd=.49),environment=dict(
        nvidia_smi='samegpu',cpu_quota_cores=7.65,host_memory_limit_bytes=50000000000,
        cpu_math_thread_environment=dict(OMP_NUM_THREADS='1'))))
    monkeypatch.setattr(m.native,'exclusive_gpu_processes',lambda pids:dict(exclusive=True))
    calls=[]
    def configuration(args,manifest,names,scope,workers,batch,output,*unused):
        calls.append((workers,batch))
        output.mkdir()
        speed={(1,1):1,(2,1):2,(4,1):3,(4,4):4,(4,8):2}[(workers,batch)]
        result=dict(status='complete',execution_rates_valid=True,workers=workers,batch_size=batch,
                    fixed_reference_anchors={},summary=dict(successful_lightcurves_per_second=speed,
                        median_repetition_successful_lightcurves_per_second=speed))
        m.native.write(output/'result.json',result)
        return result
    monkeypatch.setattr(m,'run_configuration',configuration)
    output=tmp_path/'supplement-tune'
    argv=['--stage','tune','--manifest',str(manifest),'--output',str(output),
          '--science-seal',str(science),'--primary-tuning',str(primary),
          '--primary-measurement',str(tmp_path/'measurement.json'),'--source-root',str(tmp_path),
          '--supplement-seal',str(tmp_path/'seal.json'),'--supplement-binding',str(tmp_path/'binding.json'),
          '--supplement-seal-sha256','seal','--supplement-binding-sha256','binding',
          '--hourly-usd','.49','--deadline-epoch',str(time.time()+1000)]
    assert m.main(argv)==0
    assert calls==[(1,1),(2,1),(4,1),(4,4),(4,8)]
    seal=json.loads((output/'tuning-seal.json').read_text())
    assert seal['selected']==dict(workers=4,batch_size=4)
    assert seal['supplement_binding_sha256']=='binding'
    assert seal['hourly_usd']==.49
    assert seal['campaign_sha256']==m.native.sha(output/'campaign.json')
    with pytest.raises(SystemExit):m.main(argv)
    assert len(calls)==5
    changed_price=list(argv)
    changed_price[changed_price.index('--output')+1]=str(tmp_path/'changed-price')
    changed_price[changed_price.index('--hourly-usd')+1]='0.01'
    with pytest.raises(ValueError,match='Rental price'):m.main(changed_price)
    assert len(calls)==5
