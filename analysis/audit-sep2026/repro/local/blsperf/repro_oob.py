"""Minimal reproducer: eebls_gpu with per-frequency (Keplerian) q bounds and batching sizes its device arrays with
count_tot_nbins(global nbins0_max, global nbinsf_max), but a batch whose own nbins0 is larger can need MORE bins."""
import sys, numpy as np
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import eebls_gpu, count_tot_nbins, compile_bls
from cuvarbase.bls_frequencies import keplerian_freq_grid
ndata = int(sys.argv[1]) if len(sys.argv) > 1 else 600
maxmem = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5e9
f, q = keplerian_freq_grid(0.5, 100., 3650., oversampling=2, return_qvals=True)
f = f.astype(np.float64)[:20000]; qmins = (0.5*q[:20000]).astype(np.float64); qmaxs = (2.0*q[:20000]).astype(np.float64)
rng = np.random.RandomState(0); t = np.sort(rng.uniform(0, 3650., ndata)); y = 1 + 0.002*rng.randn(ndata); dy = np.full(ndata, 0.002)
# replicate the sizing arithmetic in eebls_gpu (bls.py ~L1477-1500, 1540-1550)
nb0_max = int(np.floor(1./qmaxs.max())); nbf_max = int(np.ceil(1./qmins.min())); ntot_max = count_tot_nbins(nb0_max, nbf_max, 0.2)
mem0 = ndata*3*4 + len(f)*5*4; mem_per_f = 4*5*ntot_max*3*4; fbs = int((maxmem-mem0)/mem_per_f); gs = fbs*ntot_max*3
print(f"ndata={ndata} nfreq={len(f)} max_memory={maxmem:.2e}: global nbins0_max={nb0_max} nbinsf_max={nbf_max} nbins_tot_max={ntot_max} freq_batch_size={fbs} gs(alloc floats per array)={gs}")
for b in range(int(np.ceil(len(f)/fbs))):
    i0, i1 = fbs*b, min(len(f), fbs*(b+1)); nb0 = int(np.floor(1./qmaxs[i0:i1].max())); nbf = int(np.ceil(1./qmins[i0:i1].min())); ntot = count_tot_nbins(nb0, nbf, 0.2)
    flag = '  <-- exceeds allocation' if (i1-i0)*ntot*3 > gs else ''
    print(f"  batch {b}: nbins0={nb0} nbinsf={nbf} nbins_tot={ntot} all_bins={(i1-i0)*ntot*3}{flag}")
fr = compile_bls()
try:
    p, sols = eebls_gpu(t, y, dy, f, qmin=qmins, qmax=qmaxs, functions=fr, max_memory=maxmem)
    cuda.Context.synchronize()
    print("call returned; max power", float(np.max(p)), "n finite", int(np.isfinite(p).sum()))
except Exception as e:
    print("EXCEPTION:", repr(e))
