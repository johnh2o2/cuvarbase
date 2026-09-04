"""Experiment H: long-baseline null LC through banded fast path; band discontinuities; large ndata run."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_batch, tls_search_gpu
from cuvarbase import tls_grids
import cuvarbase.tls as T
base = 1400.0
t, y, dy = noise_lc(base, 30.0, 1e-3, seed=42)
periods = tls_grids.period_grid_ofir(t)
print("ndata=%d nperiods=%d" % (len(t), len(periods)))
q = tls_grids.q_transit(periods); qmin = 0.5 * q
need = 3.0 / qmin
nb = np.minimum(np.power(2, np.ceil(np.log2(np.clip(need, 256, None)))), 8192)
print("bands:", [(int(b), int((nb == b).sum()), float(periods[nb == b].min()), float(periods[nb == b].max())) for b in np.unique(nb)])
t1 = time.time()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r = tls_search_batch([(t, y, dy)], periods=periods, return_arrays=True)[0]
    print("warnings:", [str(x.message)[:120] for x in w])
print("null 1400-d LC: %.1fs SDE=%.2f period=%.3f n_failed=%d" % (time.time() - t1, r['SDE'], r['period'], r['n_failed_periods']))
score = np.sum((1 - y)**2 / dy**2) - r['chi2']
for b in np.unique(nb):
    m = nb == b
    print("band nbins=%5d: score mean=%.2f std=%.2f ; SR mean=%.3e" % (b, np.nanmean(score[m]), np.nanstd(score[m]), np.nanmean(r['SR'][m])))
# discontinuity at band boundaries: mean score in the 200 periods either side
bounds = np.flatnonzero(np.diff(nb) != 0)
for i in bounds:
    print("boundary at P=%.3f: mean score left=%.2f right=%.2f (std %.2f); mean power left=%.2e right=%.2e" % (periods[i], np.nanmean(score[i-200:i]), np.nanmean(score[i+1:i+201]), np.nanstd(score[i-200:i]), np.nanmean(r['power'][i-200:i]), np.nanmean(r['power'][i+1:i+201])))
# ndata scaling: 500k-point LC (1400 d @ 4 min) -- default grid
t2, y2, dy2 = noise_lc(base, 4.0, 1e-3, seed=43)
print("ndata=%d" % len(t2))
import pycuda.driver as cuda
free0 = cuda.mem_get_info()[0]
t1 = time.time()
r2 = tls_search_batch([(t2, y2, dy2)], periods=periods)[0]
print("500k-pt LC x %d periods: %.1fs SDE=%.2f; free mem before=%.2f GB after=%.2f GB" % (len(periods), time.time() - t1, r2['SDE'], free0/1e9, cuda.mem_get_info()[0]/1e9))
