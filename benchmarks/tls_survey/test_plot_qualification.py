"""Scientific qualification labels must remain independent of timing ratios."""
import csv
import json

import pytest

from benchmarks.tls_survey.test_report_recovery import synthetic_fixture
from benchmarks.tls_survey.plot_throughput import heldout_qualification, figure_csv
from benchmarks.tls_survey.report_recovery import sha


def test_failed_heldout_qualification_stays_visible_beside_valid_timing_ratio(tmp_path):
    args = synthetic_fixture(tmp_path)
    campaign = dict(science_seal_sha256=sha(args.seal))
    heldout = heldout_qualification(campaign,args.exactness,args.seal)
    assert heldout['planned_cases'] == 16 and heldout['exact_cases'] == 15
    assert not heldout['aggregate_exactness_qualified']
    data = {('baseline','tess_solar'):dict(repetitions=[dict(lightcurves_per_second=1.)]*3),
            ('candidate','tess_solar'):dict(repetitions=[dict(lightcurves_per_second=2.)]*3)}
    path = tmp_path/'figure.csv'
    figure_csv(path,data,{},dict(tess_solar=True),heldout)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = next(row for row in rows if row['backend'] == 'candidate' and row['scope'] == 'tess_solar')
    assert selected['timing_cohort_optimized_vs_baseline'] == '2.0'
    assert selected['heldout_aggregate_exactness_qualified'] == 'False'
    assert selected['heldout_exact_cases'] == '15' and selected['heldout_planned_cases'] == '16'
    assert selected['heldout_exactness_sha256'] == sha(args.exactness)


@pytest.mark.parametrize('mutation', [
    lambda value:value.update(status='running'),
    lambda value:value['cases'].pop(),
    lambda value:value['cases'][0].update(regime='foreign'),
    lambda value:value.update(exactness_qualified=True),
    lambda value:value['identity'].update(seal_sha256='wrong'),
])
def test_figure_rejects_incomplete_or_foreign_heldout_qualification(tmp_path, mutation):
    args = synthetic_fixture(tmp_path)
    campaign = dict(science_seal_sha256=sha(args.seal))
    value = json.loads(args.exactness.read_text())
    mutation(value)
    args.exactness.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        heldout_qualification(campaign,args.exactness,args.seal)


def test_planned_population_is_read_from_seal_not_reported_length(tmp_path):
    args = synthetic_fixture(tmp_path)
    seal = json.loads(args.seal.read_text())
    seal['counts']['injections'] = 8
    args.seal.write_text(json.dumps(seal))
    value = json.loads(args.exactness.read_text())
    value['identity']['seal_sha256'] = sha(args.seal)
    args.exactness.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='every planned regime/input'):
        heldout_qualification(dict(science_seal_sha256=sha(args.seal)),args.exactness,args.seal)
