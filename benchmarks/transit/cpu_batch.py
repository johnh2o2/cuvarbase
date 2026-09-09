#!/usr/bin/env python3
"""Astropy operational alternative: parallelize distinct sources for batch throughput."""
import sys
from pathlib import Path
import numpy as np
import worker
from worker import astropy_piece,qtransit,sha


def astropy_source(job):
    lc,frequencies,chunks,oversample=job
    power=np.empty(len(frequencies))
    for ids,lo,hi in chunks:
        dmin=.5*qtransit(lo)*lo;dmax=min(2*qtransit(hi)*hi,lo*.95)
        durations=np.geomspace(dmin,dmax,int(np.ceil(np.log(dmax/dmin)/np.log(1.1)))+1)
        power[ids]=astropy_piece((lc,1/frequencies[ids],durations,oversample))
    return power


class AcrossSources(worker.Backend):
    def search(self,lcs):
        assert self.kind=='astropy'
        if len(lcs)==1:return super().search(lcs)
        jobs=[(lc,self.f,self.chunks,self.cfg.get('epoch_os',10)) for lc in lcs]
        powers=list(self.pool.map(astropy_source,jobs) if self.pool else map(astropy_source,jobs))
        return [dict(periods=1/self.f,power=p,candidate=worker.spectral_candidate(1/self.f,p)) for p in powers]


if __name__=='__main__':
    # The frozen controller passes --indices only to its original worker.py.
    # This bounded adapter explicitly supplies the two declared timing subsets.
    if '--timing' in sys.argv and '--indices' not in sys.argv:
        filename=Path(sys.argv[sys.argv.index('--input')+1]).name
        assert filename in ['tess_200s_tune.npz','tess_200s_heldout.npz']
        first=0 if filename.endswith('_tune.npz') else 128
        sys.argv+=['--indices',','.join(map(str,range(first,first+16)))]
    original_dump=worker.dump
    def wrapped_dump(path,record):
        record['wrapper_sha256']=sha(__file__)
        record['operational_boundary']='Same Astropy period chunks and duration/epoch grids; parallelize across distinct sources when batch size exceeds one.'
        return original_dump(path,record)
    worker.dump=wrapped_dump;worker.Backend=AcrossSources;worker.main()
