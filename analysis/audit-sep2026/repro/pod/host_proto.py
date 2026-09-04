"""Host-side prototypes on the eebls_gpu_fast default path:
 (1) pinned-alloc microbench; (2) skip the pinned freqs/nbins/bls buffers that setdata/transfer_data_to_cpu discard;
 (3) leaner setdata (chi2_0 from yy, cached perm, single float32 conversion). All compared bitwise vs stock."""
import time, json, sys
import numpy as np
import pycuda.driver as cuda
from prof_common import *
import cuvarbase.bls as B
import cuvarbase.memory.bls_memory as BM
from cuvarbase.bls import eebls_gpu_fast, BLSMemory
from cuvarbase.memory._host import host_array
from cuvarbase.utils import subtract_epoch, conflict_scatter_perm
RUNS = 15
def hst(fn, runs=RUNS, warm=2):
    for _ in range(warm): fn()
    W, C = [], []
    for _ in range(runs):
        sync(); c0 = time.process_time(); t0 = time.perf_counter(); fn(); sync(); W.append(time.perf_counter()-t0); C.append(time.process_time()-c0)
    return dict(wall_med=1e3*float(np.median(W)), wall_min=1e3*float(np.min(W)), cpu_med=1e3*float(np.median(C)))
out = {}
# (1) pinned alloc microbench: alloc+free
for n in (4, 25000, 301000, 1000000):
    out[f'pinned_alloc_free_{n}_floats'] = hst(lambda: host_array((n,), np.float32), warm=3)
    out[f'pageable_np_zeros_{n}_floats'] = hst(lambda: np.zeros(n, np.float32), warm=3)
keep = []
out['pinned_alloc_NO_free_301000'] = hst(lambda: keep.append(host_array((301000,), np.float32)), warm=1, runs=10); del keep
for k, v in out.items(): print(f"{k:45s} {v}", flush=True)

# (2)+(3) prototypes -------------------------------------------------------
_perm_cache = {}
def perm_cached(n):
    if n not in _perm_cache: _perm_cache[n] = conflict_scatter_perm(n)
    return _perm_cache[n]

def allocate_host_arrays_lean(self, nfreqs=None, ndata=None):
    # only t/yw/w are actually used as pinned staging buffers; freqs/nbins/bls get replaced by setdata / transfer_data_to_cpu
    if ndata is None: ndata = int(self.max_ndata)
    if nfreqs is None: nfreqs = int(self.max_nfreqs)
    self.t = host_array((ndata,), self.rtype, pinned=self.pinned)
    self.yw = host_array((ndata,), self.rtype, pinned=self.pinned)
    self.w = host_array((ndata,), self.rtype, pinned=self.pinned)
    self.bls = np.zeros(nfreqs, self.rtype); self.nbins0 = np.zeros(nfreqs, np.int32); self.nbinsf = np.zeros(nfreqs, np.int32)

def setdata_lean(self, t, y, dy, qmin=None, qmax=None, freqs=None, nf=None, transfer=True, **kwargs):
    if freqs is not None:
        self.freqs = np.asarray(freqs).astype(self.rtype)
        self.nbinsf = (np.ones_like(self.freqs)/qmin).astype(np.uint32)
        self.nbins0 = (np.ones_like(self.freqs)/qmax).astype(np.uint32)
    t, self.epoch = subtract_epoch(t)
    y = np.asarray(y, dtype=np.float64); dy = np.asarray(dy, dtype=np.float64)
    w = np.power(dy, -2)          # same op as stock (bitwise)
    wsum = np.sum(w); w /= wsum
    self.ybar = np.sum(y * w)
    self.yy = float(np.einsum('i,i->', w, np.power(y - self.ybar, 2)))
    # stock recomputes chi2_0 = sum(dy^-2 (y - ybar')^2) with its own ybar' in _chi2_null; algebraically chi2_0 = yy * wsum
    self.chi2_0 = self.yy * float(wsum)
    u = (y - self.ybar) * w
    perm = perm_cached(len(t))
    n = len(t)
    if perm is None:
        self.t[:n] = t; self.w[:n] = w; self.yw[:n] = u      # numpy casts on assignment (same rounding as astype)
    else:
        np.take(t, perm, out=None)  # (placeholder no-op to keep structure obvious)
        self.t[:n] = t[perm]; self.w[:n] = w[perm]; self.yw[:n] = u[perm]
    if any([x is None for x in [self.t_g, self.yw_g, self.w_g]]): self.allocate_data()
    if self.freqs_g is None:
        if nf is None: nf = len(freqs)
        self.allocate_freqs(nfreqs=nf)
    if transfer: self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))
    return self

for name in ('ZTF', 'TESS', 'HAT'):
    cfg = SURVEYS[name]; ndata = cfg['ndata']; freqs, qmins, qmaxs = grid_for(cfg); nfreq = len(freqs); t, y, dy = make_lc(cfg, 1000)
    R = out[name] = {}
    ref = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs)
    R['stock_fast_naive'] = hst(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    R['stock_BLSMemory_init'] = hst(lambda: BLSMemory(ndata, nfreq))
    # stock setdata host-side (transfer=False) reference arrays
    m0 = BLSMemory(ndata, nfreq); m0.allocate_freqs(nfreq); m0.allocate_data(ndata)
    m0.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False)
    R['stock_setdata_host'] = hst(lambda: m0.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False))
    R['stock_setdata_host_nofreqs'] = hst(lambda: m0.setdata(t, y, dy, freqs=None, transfer=False))
    # prototype (2): lean host arrays
    orig_alloc = BLSMemory.allocate_host_arrays; BLSMemory.allocate_host_arrays = allocate_host_arrays_lean
    R['lean_alloc_BLSMemory_init'] = hst(lambda: BLSMemory(ndata, nfreq))
    p2 = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs)
    R['lean_alloc_fast_naive'] = hst(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    R['lean_alloc_bitwise_equal_to_stock'] = bool(np.array_equal(ref, p2)); R['lean_alloc_max_abs_diff'] = float(np.max(np.abs(ref - p2)))
    # prototype (3): lean setdata on top
    orig_setdata = BLSMemory.setdata; BLSMemory.setdata = setdata_lean
    m1 = BLSMemory(ndata, nfreq); m1.allocate_freqs(nfreq); m1.allocate_data(ndata)
    m1.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False)
    R['lean_setdata_host'] = hst(lambda: m1.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=False))
    R['lean_setdata_host_nofreqs'] = hst(lambda: m1.setdata(t, y, dy, freqs=None, transfer=False))
    R['lean_setdata_arrays_bitwise'] = dict(t=bool(np.array_equal(m0.t, m1.t)), yw=bool(np.array_equal(m0.yw, m1.yw)), w=bool(np.array_equal(m0.w, m1.w)),
                                             nbinsf=bool(np.array_equal(m0.nbinsf, m1.nbinsf)), yy_rel=abs(m0.yy-m1.yy)/m0.yy, chi2_0_rel=abs(m0.chi2_0-m1.chi2_0)/m0.chi2_0)
    p3 = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs)
    R['lean_both_fast_naive'] = hst(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs))
    R['lean_both_bitwise_equal_to_stock'] = bool(np.array_equal(ref, p3)); R['lean_both_max_abs_diff'] = float(np.max(np.abs(ref - p3)))
    mem = BLSMemory(ndata, nfreq); mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    def reuse(): mem.setdata(t, y, dy, freqs=None, transfer=True); eebls_gpu_fast(t, y, dy, freqs, memory=mem, transfer_to_device=False)
    R['lean_both_fast_reuse'] = hst(reuse)
    BLSMemory.setdata = orig_setdata; BLSMemory.allocate_host_arrays = orig_alloc
    mem = BLSMemory(ndata, nfreq); mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    R['stock_fast_reuse'] = hst(reuse)
    for k, v in R.items(): print(f"  {name} {k:40s} {v}", flush=True)
json.dump(out, open('/workspace/scratch/host_proto.json', 'w'), indent=1, default=str)
