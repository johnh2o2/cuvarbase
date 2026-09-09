#!/usr/bin/env python3
"""Select CPU scheduling from tuning timings only after full output equivalence checks."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    note=json.loads((r/'cpu-operational-amendment.json').read_text())
    original=r/'timing-methods-before-cpu-scheduling.json'
    assert sha(original)==note['original_timing_methods_sha256'] and sha(r/'cpu_operations.json')==note['jobs_sha256']
    frozen=json.loads((r/'harness-freeze.json').read_text())['files']['worker.py']
    jobs=json.loads((r/'cpu_operations.json').read_text());records={}
    for job in jobs:
        p=r/'results'/job['name'];d=json.loads((p/'summary.json').read_text());e=json.loads((p/'execution.json').read_text())
        assert d['status']=='ok' and e['exit_code']==0 and d['worker_sha256']==frozen
        assert d['config']==job['config'] and d['input_sha256']==sha(r/'inputs'/job['input'])
        if job.get('timing'):
            expected=[int(i) for i in job['indices'].split(',')]
            assert d['indices']==expected==[c['index'] for c in d['cases']]
            assert len(d['times_s'])==job['reps']
        if job['script']=='cpu_batch.py':assert d['wrapper_sha256']==sha(r/'compute/original/evidence/scripts/cpu_batch.py')
        records[job['name']]=d
    agreement=[]
    for split in ['calibration','heldout']:
        base=r/'results'/f'{split}_tess_200s_bls_cpu';other=r/'results'/f'{split}_tess_200s_bls_cpu_sources'
        aa=json.loads((base/'summary.json').read_text())['cases'];bb=records[other.name]['cases'];assert len(aa)==len(bb)
        for ca,cb in zip(aa,bb):
            assert ca['index']==cb['index']
            assert ca['period']==cb['period'] and ca['score']==cb['score'] and ca['finite_fraction']==cb['finite_fraction']==1.
            pa=base/ca['output_file'];pb=other/cb['output_file'];assert sha(pa)==ca['output_sha256'] and sha(pb)==cb['output_sha256']
            with np.load(pa) as va,np.load(pb) as vb:
                assert np.array_equal(va['periods'],vb['periods']) and np.array_equal(va['power'],vb['power'])
            agreement.append(dict(split=split,index=ca['index'],periods_equal=True,powers_equal=True,candidate_equal=True))
    assert len(agreement)==384
    periods=records['cpu_schedule_tune_periods']['seconds_per_source'];sources=records['cpu_schedule_tune_sources']['seconds_per_source']
    selected=sources<periods;manifest=json.loads(original.read_text())
    for item in manifest['methods']:
        if item['profile']=='tess_200s' and item['tag']=='bls_cpu' and item['mode']=='batch16' and selected:
            item['job']='timing_tess_200s_bls_cpu_sources_batch16';item['operational_adapter']='cpu_batch.py; bit-identical spectra and candidates on 384 independent calibration/heldout sources'
    (r/'timing-methods.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(complete=True,selected_across_sources=selected,tuning_period_parallel_seconds_per_source=periods,
        tuning_source_parallel_seconds_per_source=sources,source_scheduling_speedup=periods/sources,
        independent_output_agreement=agreement,selected_timing_manifest_sha256=sha(r/'timing-methods.json'),
        note='Only CPU task scheduling selected, from the declared tuning timings. All scientific settings and independent calibration/heldout spectra and candidates are unchanged. Original timing methods and measurements are preserved.')
    (r/'cpu-operational-selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_output_agreement'},indent=2))


if __name__=='__main__':main()
