#!/usr/bin/env python3
"""Repeat close tuning choices before freezing any configuration."""
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    selection=json.loads((r/'selection-preview.json').read_text());jobs=[];seen=set()
    for group in selection['groups']:
        close=group['close_timing_candidates']
        if len(close)<2:continue
        for item in close:
            key=(group['profile'],json.dumps(item['config'],sort_keys=True))
            if key in seen:continue
            seen.add(key)
            jobs.append(dict(name='repeat_'+item['job'],input=group['profile']+'_tune.npz',config=item['config'],
                             indices='2' if item['config']['backend']=='gtls' else '0,1,2,3,4,5,6,7',
                             timing=True,reps=3,timeout=700))
    (r/'repeat_selection.json').write_text(json.dumps(jobs,indent=2)+'\n');print('Declared',len(jobs),'close-choice timing repeats')


if __name__=='__main__':main()
