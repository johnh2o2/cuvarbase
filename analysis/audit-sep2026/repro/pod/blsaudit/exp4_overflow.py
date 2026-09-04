"""Exp 4: eebls_gpu with ndata * freq_batch_size > 2^32 (uint32 thread
index overflow in bin_and_phase_fold_bst_multifreq)."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
import pycuda.driver as cuda
from cuvarbase.core import ensure_context
ensure_context()
from blsref import make_data
from cuvarbase.bls import eebls_gpu, eebls_gpu_fast, count_tot_nbins

ndata = 66000
t, y, dy = make_data(ndata=ndata, baseline=90., freq=0.5, q=0.05, phi0=0.4, snr=30, seed=8, cadence=90. / ndata)
nf = 66000
freqs = np.linspace(0.3, 0.7, nf)
print("ndata=%d nfreq=%d ndata*nfreq=%d  2^32=%d  overflow=%s" % (ndata, nf, ndata * nf, 2 ** 32, ndata * nf > 2 ** 32))
qmin, qmax = 0.2, 0.5
print("nbins_tot=%d; free mem=%.1f GB" % (count_tot_nbins(2, 5, 0.2), cuda.mem_get_info()[0] / 1e9))
pf = eebls_gpu_fast(t, y, dy, freqs, qmin=qmin, qmax=qmax, dlogq=0.2, noverlap=3)
# single batch (auto freq_batch_size from free memory)
ps, sols = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, dlogq=0.2, noverlap=3)
print("auto batch: eebls_gpu max=%.4f at f=%.4f | fast max=%.4f at f=%.4f | #zero powers gpu=%d fast=%d"
      % (ps.max(), freqs[np.argmax(ps)], pf.max(), freqs[np.argmax(pf)], (ps == 0).sum(), (pf == 0).sum()))
print("  corr(gpu, fast)=%.4f; first nonzero gpu index=%d" % (np.corrcoef(ps, pf)[0, 1], int(np.argmax(ps > 0)) if (ps > 0).any() else -1))
# safe batch size
ps2, _ = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, dlogq=0.2, noverlap=3, freq_batch_size=20000)
print("freq_batch_size=20000: eebls_gpu max=%.4f at f=%.4f; corr with fast=%.4f; #zero=%d"
      % (ps2.max(), freqs[np.argmax(ps2)], np.corrcoef(ps2, pf)[0, 1], (ps2 == 0).sum()))
print("auto vs batched eebls_gpu: max|diff|=%.3e" % np.abs(ps - ps2).max())
