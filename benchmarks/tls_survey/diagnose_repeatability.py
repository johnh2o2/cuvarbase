#!/usr/bin/env python3
"""Finite repeatability diagnosis; never a throughput or sensitivity result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(before, after):
    import numpy as np
    if before.dtype != after.dtype or before.shape != after.shape:
        return dict(same_shape_dtype=False, exact=False)
    finite = np.isfinite(before) & np.isfinite(after)
    delta = after[finite].astype(float)-before[finite].astype(float)
    return dict(same_shape_dtype=True, exact=before.tobytes()==after.tobytes(),
                changed_finite_values=int(np.count_nonzero(delta)),
                changed_finite_masks=int(np.count_nonzero(np.isfinite(before)!=np.isfinite(after))),
                max_absolute_difference=float(np.abs(delta).max()) if delta.size else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--science-seal',type=Path,required=True)
    parser.add_argument('--backend',choices=('baseline','candidate','gtls','bls'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--names',nargs='*',default=[])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.candidate_root.resolve()))
    from benchmarks.tls_survey import throughput as native
    from benchmarks.tls_survey import common
    import numpy as np
    import cupy as cp
    sys.path.insert(0,str(args.source_root.resolve()))
    cases=native.load_manifest(args.manifest,args.names)
    if args.backend=='bls':
        seal=native.configure_bls(cases,args.science_seal)
        from cuvarbase.base import ensure_context
        ensure_context()
        science=native.science_bls_module()
        if science.production_identity()!=seal['production_sources']:
            raise ValueError('BLS source tree differs from frozen science')
    else:
        native.initialize_backend('candidate' if args.backend=='baseline' else args.backend)
    native.prepare_grids(cases)
    started=time.perf_counter()
    record=dict(backend=args.backend,purpose='repeatability_diagnostic_not_timing_or_requalification',
                environment=native.resource_environment(),source_root=str(args.source_root),
                source_sha256=sha(__file__),manifest_sha256=sha(args.manifest),
                names=[c['name'] for c in cases],repetitions=args.repetitions,observations=[],status='running')
    anchors={}
    native.write(args.output/'result.json',record)
    for repetition in range(args.repetitions):
        for case in cases:
            beginning=time.perf_counter()
            row=dict(case=case['name'],repetition=repetition,input_sha256=case['input_sha256'])
            try:
                result=native.public_call('candidate' if args.backend=='baseline' else args.backend,
                                          [case],arrays=True)[0]
                cp.cuda.runtime.deviceSynchronize()
                fp=native.complete_fingerprint('candidate' if args.backend=='baseline' else args.backend,case,result)
                values=vars(result) if args.backend=='gtls' else result
                arrays=result['_arrays'] if args.backend=='bls' else {
                    key:np.asarray(np.ma.filled(values[key],np.nan)) for key in ('periods','power','chi2')}
                path=args.output/f'{repetition}-{case["name"]}'
                np.savez_compressed(path,**arrays)
                scalar_keys=('period','score') if args.backend=='bls' else ('period','SDE')
                row.update(status='success',strict=fp['strict'],
                           scalar={k:float(values[k]) for k in scalar_keys},
                           artifact=dict(file=path.name,sha256=sha(path)))
                if case['name'] not in anchors:
                    if repetition==0:
                        anchors[case['name']]=dict(path=path,strict=fp['strict'],scalar=row['scalar'])
                    else:
                        row['comparison']='original reference unavailable'
                else:
                    anchor=anchors[case['name']]
                    with np.load(anchor['path'],allow_pickle=False) as original:
                        row['array_differences']={key:differences(original[key],value) for key,value in arrays.items()}
                    row['changed_strict_fields']=[key for key in fp['strict'] if fp['strict'][key]!=anchor['strict'][key]]
                    row['scalar_differences']={key:row['scalar'][key]-anchor['scalar'][key] for key in scalar_keys}
            except Exception:
                row.update(status='error',error=traceback.format_exc())
            row['elapsed_seconds_including_diagnostics']=time.perf_counter()-beginning
            record['observations'].append(row)
            native.write(args.output/'result.json',record)
            print(json.dumps({k:v for k,v in row.items() if k not in ('strict','artifact','error')}),flush=True)
    record.update(status='complete',elapsed_seconds=time.perf_counter()-started,
                  failed_calls=sum(r['status']=='error' for r in record['observations']),
                  changed_repeats=sum(bool(r.get('changed_strict_fields')) for r in record['observations']))
    native.write(args.output/'result.json',record)


if __name__=='__main__':
    main()
