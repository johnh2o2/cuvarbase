"""Replicate eebls_gpu's auto batch sizing on the DEFAULT eebls_transit path
(transit_autofreq grid + q_transit Keplerian q arrays, qmin_fac=0.5, qmax_fac=2)
for realistic survey light curves. Reports (a) whether ndata*nf_per_batch > 2^32
(uint32 thread-index overflow) and (b) whether any batch's nbins_tot exceeds the
global nbins_tot_max used to size the device buffers (OOB atomics)."""
import numpy as np, sys
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import (fmin_transit, fmax_transit, transit_autofreq,
                           q_transit, count_tot_nbins)

free, total = cuda.mem_get_info()
print("device free=%.2f GB total=%.2f GB" % (free/1e9, total/1e9))

def sizing(name, ndata, baseline, max_memory, nstreams=5, noverlap=3, dlogq=0.2,
           qmin_fac=0.5, qmax_fac=2.0, cadence_min=None, extra_kwargs={}):
    rng = np.random.RandomState(1)
    if cadence_min is None:
        t = np.sort(rng.uniform(0, baseline, ndata))
    else:
        t = np.arange(ndata) * cadence_min / 1440.
        t = t[t < baseline]
        ndata = len(t)
    fmin = fmin_transit(t, **extra_kwargs)
    fmax = fmax_transit(qmax=0.5 / qmax_fac, **extra_kwargs)
    freqs, qvals = transit_autofreq(t, fmin=fmin, fmax=fmax, qmin_fac=qmin_fac, **extra_kwargs)
    qmins = np.asarray(qvals) * qmin_fac; qmaxs = np.asarray(qvals) * qmax_fac
    nb0_max = int(np.floor(1. / qmaxs.max())); nbf_max = int(np.ceil(1. / qmins.min()))
    ntot_max = count_tot_nbins(nb0_max, nbf_max, dlogq)
    mem0 = ndata * 3 * 4 + len(freqs) * 5 * 4
    mem_per_f = 4 * nstreams * ntot_max * noverlap * 4
    fbs = int(float(max_memory - mem0) / mem_per_f)
    gs = fbs * ntot_max * noverlap
    nbatches = int(np.ceil(len(freqs) / fbs))
    overflow = False; oob = []; worst = 0.
    for b in range(nbatches):
        i0, i1 = fbs * b, min(len(freqs), fbs * (b + 1)); nf = i1 - i0
        if ndata * nf > 2**32: overflow = True
        nb0 = int(np.floor(1. / qmaxs[i0:i1].max())); nbf = int(np.ceil(1. / qmins[i0:i1].min()))
        ntot = count_tot_nbins(nb0, nbf, dlogq)
        worst = max(worst, nf * ntot * noverlap / gs)
        if nf * ntot * noverlap > gs: oob.append((b, nb0, nbf, ntot, nf * ntot * noverlap, gs))
    print("%-42s ndata=%6d nfreq=%8d fmin=%.4f fmax=%.3f nbins_tot_max=%5d fbs=%8d nbatches=%3d | ndata*nf_batch0=%.3e (>2^32: %s) | worst all_bins/gs=%.3f OOB batches=%d %s"
          % (name, ndata, len(freqs), fmin, fmax, ntot_max, fbs, nbatches, ndata * min(fbs, len(freqs)), overflow, worst, len(oob), oob[:2]))
    return dict(freqs=freqs, qvals=qvals, t=t, fbs=fbs, overflow=overflow, oob=oob)

for mm_name, mm in [("4090 0.9*free", int(0.9 * free)), ("80GB card 0.9*free", int(0.9 * 80e9))]:
    print("==== max_memory: %s = %.2e" % (mm_name, mm))
    sizing("Kepler 4yr 30min (65K pts, 1460d)", 70000, 1460., mm, cadence_min=29.4)
    sizing("Kepler 4yr 30min rho=0.3 (giant)", 70000, 1460., mm, cadence_min=29.4, extra_kwargs=dict(rho=0.3))
    sizing("Kepler SC 1min 1 quarter (130K pts, 90d)", 130000, 90., mm, cadence_min=1.0)
    sizing("Kepler SC 1min 4 quarters (520K pts, 365d)", 520000, 365., mm, cadence_min=1.0)
    sizing("TESS 2min 1 sector (20K pts, 27d)", 20000, 27., mm, cadence_min=2.0)
    sizing("TESS 2min 1yr (~200K pts, 365d)", 260000, 365., mm, cadence_min=2.0)
    sizing("TESS 20s 1 sector (120K pts, 27d)", 120000, 27., mm, cadence_min=1/3.)
    sizing("HAT-Net (6K pts, 3650d)", 6000, 3650., mm)
    sizing("HAT-Net dense (60K pts, 3650d)", 60000, 3650., mm)
    sizing("ZTF-like (600 pts, 730d) [>=500 -> eebls_gpu]", 600, 730., mm)
    sizing("ZTF-like 1000 pts 1500d", 1000, 1500., mm)
