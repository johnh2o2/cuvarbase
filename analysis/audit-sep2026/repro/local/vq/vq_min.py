"""Verifier minimal repro: eebls_gpu with per-frequency q arrays.
freq_batch_size=1 makes each batch a single frequency, so the batch-wide
[min,max] collapse degenerates to the documented per-frequency bounds ->
the ONLY difference between batch=1 and batch=None is the q-bound collapse."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
from blsref import make_data
from cuvarbase.bls import eebls_gpu, compile_bls, q_transit, transit_autofreq, count_tot_nbins
import pycuda.driver as cuda

t, y, dy = make_data(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6, snr=12, seed=11)
fr = compile_bls()

# --- (A) two-frequency case with disjoint q windows -----------------------
freqs = np.array([0.05, 2.5])
qmin = np.array([0.01, 0.10]); qmax = np.array([0.02, 0.20])
p_def, s_def = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, functions=fr)
p_b1,  s_b1  = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax, functions=fr, freq_batch_size=1)
for i in range(2):
    print("f=%.2f bounds [%.2f,%.2f]: default -> q=%.4f p=%.5f | batch=1 -> q=%.4f p=%.5f"
          % (freqs[i], qmin[i], qmax[i], s_def[i][0], p_def[i], s_b1[i][0], p_b1[i]))

# --- (B) Keplerian grid: batch=1 (per-freq semantics) vs default -----------
freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.02, fmax=3.0)
freqs = freqs[::max(1, len(freqs)//600)]; q0 = q_transit(freqs)
qmins, qmaxs = 0.5*q0, 2.0*q0
p_def, s_def = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, functions=fr)
p_b1,  s_b1  = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, functions=fr, freq_batch_size=1)
qd = np.array([s[0] for s in s_def]); q1 = np.array([s[0] for s in s_b1])
out_d = ((qd < qmins*0.999) | (qd > qmaxs*1.001)).mean()
out_1 = ((q1 < qmins*0.999) | (q1 > qmaxs*1.001)).mean()
print("Keplerian grid nf=%d: frac q_sol outside per-freq window: default=%.3f  batch=1=%.3f" % (len(freqs), out_d, out_1))
print("  power default >= batch1 everywhere: %s; median ratio default/batch1 = %.2f; max|diff| = %.3e; argmax same: %s"
      % (np.all(p_def >= p_b1 - 1e-6), np.median(p_def/np.maximum(p_b1,1e-9)), np.abs(p_def-p_b1).max(), np.argmax(p_def)==np.argmax(p_b1)))
mask = np.abs(freqs-0.2) > 0.01; i0 = np.argmin(np.abs(freqs-0.2))
print("  peak/max-offpeak: default %.2f batch1 %.2f ; off-peak median default %.4f batch1 %.4f"
      % (p_def[i0]/p_def[mask].max(), p_b1[i0]/p_b1[mask].max(), np.median(p_def[mask]), np.median(p_b1[mask])))

# --- (C) what freq_batch_size does the DEFAULT max_memory give here? -------
free, total = cuda.mem_get_info()
nb0 = int(np.floor(1/qmaxs.max())); nbf = int(np.ceil(1/qmins.min()))
ntot = count_tot_nbins(nb0, nbf, 0.2)
mem_per_f = 4*5*ntot*3*4
print("device free=%.1f GB -> auto freq_batch_size ~ %d (nbins_tot_max=%d); an 8GB GPU (~7GB free) would get ~%d"
      % (free/1e9, int(0.9*free/mem_per_f), ntot, int(0.9*7e9/mem_per_f)))
# per-frequency bins actually needed
ntot_f = np.array([count_tot_nbins(int(np.floor(1/b)), int(np.ceil(1/a)), 0.2) for a, b in zip(qmins, qmaxs)])
print("  bins evaluated per freq: collapsed %d vs per-freq median %d (max %d) -> %.1fx wasted work" % (ntot, np.median(ntot_f), ntot_f.max(), ntot/np.median(ntot_f)))
