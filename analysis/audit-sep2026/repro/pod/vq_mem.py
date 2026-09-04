"""Does eebls_transit's DEFAULT output depend on how much GPU memory is free?
Simulate an 8 GB card via max_memory (forwarded to eebls_gpu through kwargs)."""
import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch/blsaudit')
from blsref import make_data
from cuvarbase.bls import eebls_transit, q_transit
t, y, dy = make_data(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6, snr=12, seed=11)
t0=time.time(); fr, pa, sa = eebls_transit(t, y, dy, fmin=0.02, fmax=3.0); ta=time.time()-t0
t0=time.time(); fr2, pb, sb = eebls_transit(t, y, dy, fmin=0.02, fmax=3.0, max_memory=int(0.9*7e9)); tb=time.time()-t0
t0=time.time(); fr3, pc, sc = eebls_transit(t, y, dy, fmin=0.02, fmax=3.0, max_memory=int(0.9*2e9)); tc=time.time()-t0
i0 = np.argmin(np.abs(fr-0.2)); mask = np.abs(fr-0.2) > 0.01
for lab, p, tt in (("24GB default", pa, ta), ("7GB simulated", pb, tb), ("2GB simulated", pc, tc)):
    print("%-14s nf=%d peak=%.4f offpeak-median=%.4f p99=%.4f peak/maxoff=%.2f argmax f=%.4f  (%.1fs)"
          % (lab, len(fr), p[i0], np.median(p[mask]), np.percentile(p[mask],99), p[i0]/p[mask].max(), fr[np.argmax(p)], tt))
print("max|24GB-7GB|=%.3e (#>1e-4: %d/%d); max|24GB-2GB|=%.3e (#>1e-4: %d)"
      % (np.abs(pa-pb).max(), (np.abs(pa-pb)>1e-4).sum(), len(fr), np.abs(pa-pc).max(), (np.abs(pa-pc)>1e-4).sum()))
