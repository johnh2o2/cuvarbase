#!/usr/bin/env python3
"""Join disjoint recovery partitions with a complete reference to their provenance."""
import argparse,json
from pathlib import Path
from worker import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    declared=json.loads((r/'validation-distribution.json').read_text());assert sha(r/'validation.json')==declared['validation_sha256']
    for parent in declared['parents']:
        records=[];cases=[];sources={};executions=[]
        for part in parent['parts']:
            folder=r/'results'/part;d=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'execution.json').read_text())
            assert d['status']=='ok' and e['exit_code']==0,(part,d['status'],e['exit_code'])
            if records:
                for key in ['config','input_sha256','worker_sha256','installed_sources']:
                    assert d[key]==records[0][key],(part,key)
            for c in d['cases']:
                c=c.copy()
                if c.get('output_file'):c['output_file']='../'+part+'/'+c['output_file']
                cases.append(c)
            records.append(d);executions.append(e)
        cases.sort(key=lambda c:c['index']);assert [c['index'] for c in cases]==list(range(parent['n']))
        merged=records[0].copy();merged.update(indices=list(range(parent['n'])),cases=cases,partitioned_recovery=True,
            partition_summaries=[dict(job=p,summary_sha256=sha(r/'results'/p/'summary.json'),execution_sha256=sha(r/'results'/p/'execution.json'),environment=d['environment']) for p,d in zip(parent['parts'],records)],
            merge_script_sha256=sha(__file__),boundary='Distributed recovery only. Evaluation wall times are not final performance measurements.')
        folder=r/'results'/parent['parent'];dump(folder/'summary.json',merged)
        dump(folder/'execution.json',dict(name=parent['parent'],exit_code=0,derived=True,parts=parent['parts'],
            elapsed_s=sum(e['elapsed_s'] for e in executions),note='Sum of independent recovery worker elapsed times, not latency or throughput.'))
        print('Merged',parent['parent'],len(cases))


if __name__=='__main__':main()
