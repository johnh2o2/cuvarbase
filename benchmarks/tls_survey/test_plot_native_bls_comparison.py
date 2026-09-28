"""Execution throughput must never erase native BLS's failed qualification."""
import copy
import json

import pytest

from benchmarks.tls_survey.plot_native_bls_comparison import (
    IDENTITY_KEYS, allocation, canonical_sha, execution_rates, read_native, sha, table_rows,
)


def record():
    return dict(status='complete', scope='tess_solar', backend='native_bls_execution', workers=2, batch_size=4,
        execution_rates_valid=True, gpu_ownership=dict(passed=True),
        numerical=dict(original_qualification_passed=False,
            selected_mismatch_count=6, complete_output_mismatch_count=9),
        environment=dict(nvidia_smi='A40, same-uuid', cpu_quota_cores=7.65,
                         host_memory_limit_bytes=49999998976),
        cohort=[dict(name='case.npz', regime='tess_solar', nobs=1000, nperiods=2000,
                     input_sha256='same-input')],
        repetitions=[dict(status='completed_queue', attempted_count=100, successful_count=90, failed_count=10,
            elapsed_seconds=120., successful_lightcurves_per_second=.75) for _ in range(3)],
        summary=dict(cold_first_cohort_including_startup_seconds=8.,
                     total_measured_compute_usd=.049), memory={})


def fixture(tmp_path):
    primary_record = tmp_path/'primary-result.json'
    primary_record.write_text(json.dumps(record()))
    primary = dict(science_seal_sha256='science', manifest_sha256='manifest', varied_manifest_sha256='varied',
        configs=[dict(scope='tess_solar',
        result=primary_record.name, result_sha256=sha(primary_record))])
    primary_path = tmp_path/'primary.json'
    primary_path.write_text(json.dumps(primary))
    seal = tmp_path/'seal.json'; seal.write_text(json.dumps(dict(schema=1, kind='native_bls_execution_supplement',
        science_seal_sha256='science', auxiliary_plan_sha256='auxiliary',
        binding_rule=dict(primary_tuning_path='/original/tuning.json'),
        remote_files={'/original/tuning.json':'tuning'})))
    binding = tmp_path/'binding.json'; binding.write_text(json.dumps(dict(schema=1,
        supplement_seal_sha256=sha(seal), science_seal_sha256='science', auxiliary_plan_sha256='auxiliary',
        primary_tuning_sha256='tuning', primary_measurement_sha256=sha(primary_path),
        primary_measurement_manifest_sha256='manifest', primary_measurement_varied_manifest_sha256='varied',
        primary_configs=[dict(primary['configs'][0], cohort_sha256=canonical_sha(record()['cohort']),
                             environment_sha256=canonical_sha(record()['environment']))])))
    native_record = tmp_path/'native-result.json'; native_record.write_text(json.dumps(record()))
    campaign = dict(stage='measure', status='complete', original_numerical_qualification_passed=False,
        selected=dict(workers=2, batch_size=4),
        auxiliary_plan_sha256='auxiliary', primary_tuning_sha256='tuning', supplement_binding_sha256=sha(binding),
        science_seal_sha256='science', primary_measurement_sha256=sha(primary_path),
        supplement_seal_sha256=sha(seal), configs=[dict(scope='tess_solar', workers=2, batch_size=4,
            result=native_record.name, result_sha256=sha(native_record), execution_rates_valid=True)])
    campaign.update(source_identity={'source':'source-hash'}, allocation=list(allocation(record())),
                    deadline_epoch=123456., original_bls_exclusion_sha256='original-failure', hourly_usd=.49)
    value = record(); value.update({key:campaign[key] for key in (*IDENTITY_KEYS, 'source_identity')})
    native_record.write_text(json.dumps(value)); campaign['configs'][0]['result_sha256'] = sha(native_record)
    folder = tmp_path/'tune'; folder.mkdir()
    tuning = dict(campaign, stage='tune', manifest_sha256='development-manifest', configs=[], artifact_sha256={})
    (folder/'campaign.json').write_text(json.dumps(tuning))
    tuning_seal = dict(schema_version=1, original_qualification_passed=False,
        campaign_path='/original/tune/campaign.json', campaign_sha256=sha(folder/'campaign.json'),
        manifest_sha256=tuning['manifest_sha256'], **{key:tuning[key] for key in
            (*IDENTITY_KEYS, 'source_identity', 'allocation', 'deadline_epoch', 'selected',
             'original_bls_exclusion_sha256', 'hourly_usd')})
    tuning_path = folder/'tuning-seal.json'; tuning_path.write_text(json.dumps(tuning_seal))
    campaign['tuning_seal_sha256'] = sha(tuning_path)
    native = tmp_path/'native.json'; native.write_text(json.dumps(campaign))
    return native, primary_path, primary, allocation(record()), seal, binding, tuning_path


def test_api_failures_reduce_rate_and_mismatches_do_not_become_qualified(tmp_path):
    assert execution_rates(record()) == [.75]*3
    _, native, missing = read_native(*fixture(tmp_path))
    heldout = dict(exact_cases=5119, planned_cases=5120,
                   aggregate_exactness_qualified=False, science_seal_sha256='science')
    rows = table_rows({}, native, {}, missing, {}, heldout)
    row = next(r for r in rows if r['backend'] == 'bls' and r['scope'] == 'tess_solar')
    assert row['rate_available']
    assert row['median_lightcurves_per_second'] == .75
    assert row['attempted_count'] == 300 and row['successful_count'] == 270 and row['failed_count'] == 30
    assert row['completion_fraction'] == .9
    assert row['selected_mismatch_count'] == 6 and row['complete_output_mismatch_count'] == 9
    assert not row['original_numerical_qualification_passed']
    assert not row['heldout_aggregate_exactness_qualified']
    assert row['heldout_exact_cases'] == 5119


@pytest.mark.parametrize('mutation', [
    lambda r:r['repetitions'][0].update(successful_lightcurves_per_second=100/120),
    lambda r:r['repetitions'][0].update(failed_count=0),
    lambda r:r['repetitions'][0].update(elapsed_seconds=119.9),
    lambda r:r['repetitions'][0].update(attempted_count=90, successful_count=80),
    lambda r:r['repetitions'].pop(),
    lambda r:r['repetitions'][0].update(status='interrupted'),
    lambda r:r['gpu_ownership'].update(passed=False),
    lambda r:r['numerical'].update(original_qualification_passed=True),
])
def test_rejects_incomplete_or_misrepresented_execution(mutation):
    value = record(); mutation(value)
    with pytest.raises(ValueError):
        execution_rates(value)


@pytest.mark.parametrize('mutation', [
    lambda r:r['environment'].update(cpu_quota_cores=8.),
    lambda r:r['environment'].update(nvidia_smi='A40, different-uuid'),
    lambda r:r['cohort'][0].update(input_sha256='different-input'),
    lambda r:r['cohort'][0].update(nperiods=1999),
    lambda r:r.update(workers=4),
])
def test_rejects_resource_or_input_mismatch(tmp_path, mutation):
    args = fixture(tmp_path)
    campaign = json.loads(args[0].read_text())
    path = tmp_path/campaign['configs'][0]['result']
    value = json.loads(path.read_text()); mutation(value); path.write_text(json.dumps(value))
    campaign['configs'][0]['result_sha256'] = sha(path)
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError):
        read_native(*args)


def test_changed_original_receipt_is_rejected(tmp_path):
    args = fixture(tmp_path)
    path = tmp_path/args[2]['configs'][0]['result']
    value = copy.deepcopy(record()); value['cohort'][0]['input_sha256'] = 'rewritten'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Primary timing receipt changed'):
        read_native(*args)


def test_zero_successful_work_remains_zero():
    value = record()
    for row in value['repetitions']:
        row.update(successful_count=0, failed_count=100, successful_lightcurves_per_second=0.)
    assert execution_rates(value) == [0.]*3


def test_no_heldout_retry_can_replace_a_failed_panel(tmp_path):
    args = fixture(tmp_path)
    campaign = json.loads(args[0].read_text())
    failed = copy.deepcopy(campaign['configs'][0]); failed['execution_rates_valid'] = False
    campaign['configs'].insert(0, failed)
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match='Repeated native timing scope'):
        read_native(*args)


def test_short_reference_diagnostic_is_verified_but_never_displayed_as_measurement(tmp_path):
    args = fixture(tmp_path)
    campaign = json.loads(args[0].read_text())
    value = record(); value.update(workers=1, batch_size=1,
        **{key:campaign[key] for key in (*IDENTITY_KEYS, 'source_identity')})
    value['repetitions'] = [dict(attempted_count=1, successful_count=1, failed_count=0,
                              elapsed_seconds=.1, successful_lightcurves_per_second=10.)]
    path = tmp_path/'reference.json'; path.write_text(json.dumps(value))
    reference = dict(scope='tess_solar', workers=1, batch_size=1, reference_only=True,
        result=path.name, result_sha256=sha(path), execution_rates_valid=True)
    campaign['configs'].insert(0, reference); args[0].write_text(json.dumps(campaign))
    _, native, _ = read_native(*args)
    assert execution_rates(native['tess_solar']) == [.75]*3
    path.write_text(json.dumps(dict(value, workers=2)))
    with pytest.raises(ValueError, match='Native timing receipt changed'):
        read_native(*args)


@pytest.mark.parametrize('mutation', [
    lambda b:b.update(auxiliary_plan_sha256='foreign-plan'),
    lambda b:b.update(primary_measurement_manifest_sha256='foreign-manifest'),
    lambda b:b.update(primary_measurement_varied_manifest_sha256='foreign-varied'),
    lambda b:b['primary_configs'][0].update(cohort_sha256='foreign-cohort'),
    lambda b:b['primary_configs'][0].update(environment_sha256='foreign-allocation'),
])
def test_mechanical_binding_cannot_change_reviewed_identities(tmp_path, mutation):
    args = fixture(tmp_path)
    binding = json.loads(args[5].read_text()); mutation(binding)
    args[5].write_text(json.dumps(binding))
    campaign = json.loads(args[0].read_text()); campaign['supplement_binding_sha256'] = sha(args[5])
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError):
        read_native(*args)


def test_later_binding_cannot_replace_prospectively_pinned_primary_tuning(tmp_path):
    args = fixture(tmp_path)
    binding = json.loads(args[5].read_text()); binding['primary_tuning_sha256'] = 'replacement'
    args[5].write_text(json.dumps(binding))
    campaign = json.loads(args[0].read_text()); campaign.update(
        primary_tuning_sha256='replacement', supplement_binding_sha256=sha(args[5]))
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match='Primary tuning was not frozen'):
        read_native(*args)


def test_measurement_cannot_change_the_sealed_development_winner(tmp_path):
    args = fixture(tmp_path)
    campaign = json.loads(args[0].read_text()); campaign['selected']['workers'] = 4
    row = campaign['configs'][0]; row['workers'] = 4
    path = tmp_path/row['result']; value = json.loads(path.read_text()); value['workers'] = 4
    path.write_text(json.dumps(value)); row['result_sha256'] = sha(path)
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match='development selection: selected'):
        read_native(*args)


@pytest.mark.parametrize('key,value', [('scope','varied'), ('backend','other'),
                                     ('auxiliary_plan_sha256','foreign')])
def test_individual_native_result_must_match_its_campaign(tmp_path, key, value):
    args = fixture(tmp_path)
    campaign = json.loads(args[0].read_text()); row = campaign['configs'][0]
    path = tmp_path/row['result']; record_value = json.loads(path.read_text()); record_value[key] = value
    path.write_text(json.dumps(record_value)); row['result_sha256'] = sha(path)
    args[0].write_text(json.dumps(campaign))
    with pytest.raises(ValueError):
        read_native(*args)


def test_development_artifacts_are_verified_before_rate_display(tmp_path):
    args = fixture(tmp_path)
    tuning_path = args[6].parent/'campaign.json'; artifact = args[6].parent/'diagnostic.json'
    artifact.write_text('{"original":true}')
    tuning = json.loads(tuning_path.read_text()); tuning['artifact_sha256'] = {artifact.name:sha(artifact)}
    tuning_path.write_text(json.dumps(tuning))
    seal = json.loads(args[6].read_text()); seal['campaign_sha256'] = sha(tuning_path)
    args[6].write_text(json.dumps(seal))
    campaign = json.loads(args[0].read_text()); campaign['tuning_seal_sha256'] = sha(args[6])
    args[0].write_text(json.dumps(campaign)); read_native(*args)
    artifact.write_text('{"replaced":true}')
    with pytest.raises(ValueError, match='development tuning artifact changed'):
        read_native(*args)
