"""eebls_transit DEFAULT call cost at ZTF (sparse path) and TESS (standard eebls_gpu path), decomposed."""
import time, json, sys
import numpy as np
from prof_common import *
import cuvarbase.bls as B
from cuvarbase.bls import eebls_transit, transit_autofreq, fmin_transit, fmax_transit, compile_bls, compile_sparse_bls
out = {}
for name in ('ZTF', 'TESS'):
    cfg = SURVEYS[name]; t, y, dy = make_lc(cfg, 5); R = out[name] = {}
    R['transit_autofreq_default_grid_ms'], _ = med(lambda: transit_autofreq(t, fmin=fmin_transit(t), fmax=fmax_transit(qmax=0.25), qmin_fac=0.5), 3)
    f, q = transit_autofreq(t, fmin=fmin_transit(t), fmax=fmax_transit(qmax=0.25), qmin_fac=0.5); R['default_nfreq'] = len(f)
    R['nfreq_bench_grid_for_reference'] = len(grid_for(cfg)[0])
    kw = dict(max_memory=1.5e9)   # protect the shared GPU; default would take 0.9*free
    R['eebls_transit_DEFAULT_call_ms'], allt = med(lambda: eebls_transit(t, y, dy, **kw), 3)
    R['eebls_transit_DEFAULT_all_ms'] = [x*1e3 for x in allt]
    R['eebls_transit_use_fast_call_ms'], _ = med(lambda: eebls_transit(t, y, dy, use_fast=True, use_sparse=False, **kw), 3)
    R['eebls_transit_freqs_given_DEFAULT_ms'], _ = med(lambda: eebls_transit(t, y, dy, freqs=f, qvals=q, **kw), 3)
    if name == 'ZTF':
        R['compile_sparse_bls_warm_ms'], _ = med(lambda: compile_sparse_bls(block_size=64), 3)
        R['eebls_transit_use_fast_ignored_on_sparse_default_ms'], _ = med(lambda: eebls_transit(t, y, dy, use_fast=True, **kw), 3)
    else:
        R['compile_bls_warm_ms'], _ = med(lambda: compile_bls(), 3)
    for k, v in R.items(): print(f"  {name} {k:55s} {v*1e3 if isinstance(v, float) else v}", flush=True)
json.dump(out, open('/workspace/scratch/transit_default.json', 'w'), indent=1, default=str)
