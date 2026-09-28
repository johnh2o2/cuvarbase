"""Format-only report guards, using explicitly artificial source JSON."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.tls_survey.report_recovery import render, sha


def synthetic_fixture(folder):
    """Small, visibly synthetic population; literal bounds are not estimates."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    regimes = ['SYNTHETIC_dense', 'SYNTHETIC_sparse']
    targets = (.05, .01)
    seal = dict(synthetic_fixture=True, regimes=regimes, target_fpr=targets[0], secondary_target_fpr=targets[1],
        counts=dict(calibration=512, injections=4, nulls=4), execution_shards=1,
        bls_selected={regime:dict(method='bls_strong', ranker='likelihood') for regime in regimes},
        production_sources={'SYNTHETIC_kernel.py':'fixture-only'})
    seal_path = folder/'SYNTHETIC-seal.json'
    seal_path.write_text(json.dumps(seal))
    receipts = [dict(path='/SYNTHETIC/'+split+'.json', sha256='fixture-'+split,
                    split=split, manifest_sha256='fixture-manifest-'+split,
                    production_sources=seal['production_sources']) for split in ('injections','nulls')]
    recovery = dict(synthetic_fixture=True, seal_sha256=sha(seal_path), thresholds_sha256='fixture-thresholds',
                    receipts=receipts, methods=[], contrasts=[], limitation='SYNTHETIC FIXTURE. No scientific conclusion.')
    exactness = dict(synthetic_fixture=True, status='complete', identity=dict(seal_sha256=sha(seal_path),
        plan_sha256='e'*64,
        thresholds_sha256='fixture-thresholds', candidate_receipts=[{key:r[key] for key in ('path','sha256')} for r in receipts]),
        cases=[], incomplete_repeat_diagnostics=0)
    snr = dict(synthetic_fixture=True, status='complete', split='injections', seal_sha256=sha(seal_path),
               manifest_sha256='fixture-manifest-injections', rows=[])
    def detected(label, split, index, target):
        limit = (2 if label == 'tls' else 1) if split == 'injections' else (1 if label == 'tls' else 0)
        return index < max(0, limit-(target == .01))
    for regime in regimes:
        for split in ('injections','nulls'):
            for index in range(4):
                name = regime+'-'+split+'-'+str(index)
                identity = hashlib.sha256(name.encode()).hexdigest()
                valid = not (regime == regimes[0] and split == 'injections' and index == 3)
                decisions = {key:dict(target_fpr=target, threshold=10. if target == .05 else 20.,
                    above=detected('tls',split,index,target), detected=detected('tls',split,index,target))
                    for key,target in zip(('thresholds','secondary_thresholds'), targets)}
                exactness['cases'].append(dict(regime=regime, split=split, name=name, input_sha256=identity,
                    original_candidate=dict(regime=regime, method='tls', name=name, input_sha256=identity,
                                            white_oracle_snr=(6.,8.,10.,12.)[index], observed_events=3,
                                            in_transit_observations=20, grid_reachable=index != 3,
                                            valid=valid, error=None if valid else 'SYNTHETIC execution failure'),
                    baseline=dict(valid=True, error=None),
                    comparison=dict(exact=valid, differences=[] if valid else ['unavailable_valid_execution'],
                        original_candidate_decisions=decisions, baseline_decisions=copy.deepcopy(decisions)),
                    repeat_status='not_required' if valid else 'complete'))
                if split == 'injections':
                    snr['rows'].append(dict(name=name, regime=regime, input_sha256=identity,
                        native_family_white_snr=0. if index == 3 else 9.5,
                        ideal_box_white_snr=0. if index == 3 else 9.,
                        native_family_ou_snr=0. if index == 3 else 8.5,
                        ideal_box_ou_snr=0. if index == 3 else 8.,
                        native_white_advantage=None if index == 3 else 9.5/9.-1,
                        native_ou_advantage=None if index == 3 else 8.5/8.-1))
        for target in targets:
            for label in ('tls','bls'):
                d = [detected(label,'injections',index,target) for index in range(4)]
                f = [detected(label,'nulls',index,target) for index in range(4)]
                rank = 488 if target == .05 else 508
                cut = (10. if target == .05 else 20.) + (1 if label == 'bls' else 0)
                calibration = dict(value=cut, n=512, target_fpr=target, rank_1based=rank,
                    decision='strict exceedance', calibration='SYNTHETIC stored order statistic',
                    calibration_scores_above=512-rank, calibration_scores_at_threshold=1,
                    calibration_zero_scores=0, calibration_strict_exceedance_fraction=(512-rank)/512,
                    extra_conservatism_from_ties=False,
                    attainable_marginal_fpr=(513-rank)/513, marginal_fpr_upper_bound=(513-rank)/513)
                strata = [dict(kind='snr', level=level, n=1, detected=int(d[index]), interval95=[0.,1.])
                          for index,level in enumerate((6.,8.,10.,12.))]
                strata += [dict(kind='sampling', level='three_plus_events', n=4, detected=sum(d), interval95=[0.,1.]),
                           dict(kind='sampling', level='grid_unreachable', n=1, detected=0, interval95=[0.,1.])]
                recovery['methods'].append(dict(regime=regime, target_fpr=target, method=label,
                    configuration='tls' if label == 'tls' else 'bls_strong', ranker='native' if label == 'tls' else 'likelihood',
                    threshold=cut, calibration=calibration, detected=sum(d), n_injections=4, recovery=sum(d)/4,
                    recovery_interval95=[0.,1.], false_positives=sum(f), n_nulls=4, fpr=sum(f)/4, fpr_interval95=[0.,1.],
                    failed_injections=int(regime == regimes[0] and label == 'tls'), failed_nulls=0,
                    aliases_including_fundamental=sum(d), strata=strata))
            contrast = dict(regime=regime, target_fpr=target)
            for endpoint,split in (('recovery','injections'), ('fpr','nulls')):
                wins = sum(detected('tls',split,index,target) and not detected('bls',split,index,target) for index in range(4))
                for suffix in ('','_simultaneous'):
                    contrast['tls_minus_bls_'+endpoint+suffix] = dict(n=4, first_only=wins, second_only=0,
                        difference=wins/4, interval=[-1.,1.], confidence=.95 if not suffix else 1-.05/8,
                        construction='SYNTHETIC literal bounds; no inference')
            recovery['contrasts'].append(contrast)
    exactness.update(completed_cases=len(exactness['cases']), mismatches=1, exactness_qualified=False)
    args = SimpleNamespace(seal=seal_path, output=folder/'rendered', synthetic=True)
    for key,value in (('recovery',recovery), ('exactness',exactness), ('snr',snr)):
        path = folder/('SYNTHETIC-'+key+'.json')
        path.write_text(json.dumps(value))
        setattr(args,key,path)
    return args


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_complete_synthetic_render_keeps_counts_intervals_failures_and_identity(tmp_path):
    args = synthetic_fixture(tmp_path)
    render(args)
    text = (args.output/'RECOVERY.md').read_text()
    assert 'SYNTHETIC FIXTURE — NOT A SCIENTIFIC RESULT' in text
    assert 'Aggregate exactness is withheld' in text
    assert 'Sampling groups overlap' in text and 'unrepresented' in text
    assert 'optimistic ceiling' in text and 'not confidence intervals' in text
    assert len(read_csv(args.output/'recovery_fpr.csv')) == 8
    assert len(read_csv(args.output/'paired_contrasts.csv')) == 16
    assert len(read_csv(args.output/'subgroups.csv')) == 80
    assert len(read_csv(args.output/'exactness.csv')) == 4
    assert len(read_csv(args.output/'exactness_mismatches.csv')) == 1
    assert len(read_csv(args.output/'snr_descriptive.csv')) == 60
    values = read_csv(args.output/'recovery_fpr.csv')
    assert all(row['recovery_interval95_lower'] == '0.0' and row['recovery_interval95_upper'] == '1.0' for row in values)
    provenance = json.loads((args.output/'provenance.json').read_text())
    assert provenance['sources']['snr']['sha256'] == sha(args.snr)
    assert provenance['validated']['exactness_cases'] == 16


@pytest.mark.parametrize('file,mutation,match', [
    ('recovery', lambda value:value['methods'].pop(), 'planned regime/method/FPR'),
    ('recovery', lambda value:value['methods'].append(copy.deepcopy(value['methods'][0])), 'Duplicate'),
    ('recovery', lambda value:value['methods'][0].update(configuration='different'), 'frozen science'),
    ('recovery', lambda value:value['contrasts'].pop(), 'planned paired'),
    ('recovery', lambda value:value['methods'][0].update(n_injections=3), 'denominator'),
    ('recovery', lambda value:value['methods'][0]['strata'].pop(0), 'planned SNR'),
    ('recovery', lambda value:value['methods'][0]['strata'].pop(), 'subgroup input counts'),
    ('exactness', lambda value:value['identity'].update(seal_sha256='different'), 'seal identities'),
    ('exactness', lambda value:value.update(status='running'), 'incomplete'),
    ('exactness', lambda value:value['cases'].pop(), 'planned baseline'),
    ('exactness', lambda value:value.update(exactness_qualified=True), 'summary'),
    ('exactness', lambda value:value['identity']['candidate_receipts'][0].update(sha256='different'), 'original scientific receipts'),
    ('snr', lambda value:value.update(manifest_sha256='different'), 'manifest identities'),
    ('snr', lambda value:value['rows'].pop(), 'SNR case membership'),
    ('snr', lambda value:value['rows'][0].update(input_sha256='different'), 'case input identity'),
])
def test_rejects_foreign_incomplete_or_inconsistent_reports(tmp_path, file, mutation, match):
    args = synthetic_fixture(tmp_path)
    path = getattr(args,file)
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        render(args)
    assert not args.output.exists()


def test_fixture_cannot_silently_render_as_science_and_snr_is_optional(tmp_path):
    args = synthetic_fixture(tmp_path)
    args.synthetic = False
    with pytest.raises(ValueError, match='Synthetic fixture'):
        render(args)
    args.synthetic = True
    args.snr = None
    render(args)
    assert 'No held-out expected-SNR artifact' in (args.output/'RECOVERY.md').read_text()
    assert not (args.output/'snr_cases.csv').exists()


def test_all_exact_case_still_writes_empty_failure_csv_schema(tmp_path):
    args = synthetic_fixture(tmp_path)
    exactness = json.loads(args.exactness.read_text())
    for row in exactness['cases']:
        row['original_candidate'].update(valid=True, error=None)
        row['comparison'].update(exact=True, differences=[])
    exactness.update(mismatches=0, exactness_qualified=True)
    args.exactness.write_text(json.dumps(exactness))
    recovery = json.loads(args.recovery.read_text())
    for row in recovery['methods']:
        row['failed_injections'] = 0
    args.recovery.write_text(json.dumps(recovery))
    render(args)
    assert read_csv(args.output/'exactness_mismatches.csv') == []
    assert (args.output/'exactness_mismatches.csv').read_text().startswith('regime,split,name,input_sha256')
    assert 'Every planned original held-out comparison met' in (args.output/'RECOVERY.md').read_text()


def test_output_cannot_mix_previous_optional_snr_or_source_inputs(tmp_path):
    args = synthetic_fixture(tmp_path)
    render(args)
    previous = sha(args.output/'RECOVERY.md')
    args.snr = None
    with pytest.raises(ValueError, match='different source inputs'):
        render(args)
    assert sha(args.output/'RECOVERY.md') == previous
