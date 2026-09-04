"""Scan the default eebls_transit sizing for per-batch bin-count overruns (finding 38)
across realistic max_memory budgets and user-supplied fmin/fmax sub-ranges."""
import numpy as np, itertools
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import fmin_transit, fmax_transit, transit_autofreq, q_transit, count_tot_nbins

def batches(freqs, qmins, qmaxs, ndata, max_memory, nstreams=5, noverlap=3, dlogq=0.2):
    nb0_max = int(np.floor(1. / qmaxs.max())); nbf_max = int(np.ceil(1. / qmins.min()))
    ntot_max = count_tot_nbins(nb0_max, nbf_max, dlogq)
    mem0 = ndata * 3 * 4 + len(freqs) * 5 * 4
    fbs = int(float(max_memory - mem0) / (4 * nstreams * ntot_max * noverlap * 4))
    if fbs <= 0: return None
    gs = fbs * ntot_max * noverlap
    out = []
    for b in range(int(np.ceil(len(freqs) / fbs))):
        i0, i1 = fbs * b, min(len(freqs), fbs * (b + 1)); nf = i1 - i0
        nb0 = int(np.floor(1. / qmaxs[i0:i1].max())); nbf = int(np.ceil(1. / qmins[i0:i1].min()))
        ntot = count_tot_nbins(nb0, nbf, dlogq)
        out.append((b, nf, nb0, nbf, ntot, nf * ntot * noverlap, gs))
    return fbs, ntot_max, gs, out

hits = []
total = 0
for ndata, baseline in [(600, 730.), (1000, 1500.), (6000, 3650.), (20000, 27.), (70000, 1460.)]:
    rng = np.random.RandomState(0); t = np.sort(rng.uniform(0, baseline, ndata))
    fmin0 = fmin_transit(t); fmax0 = fmax_transit(qmax=0.25)
    ranges = [(None, None)] + [(a, b) for a, b in itertools.product([fmin0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], [fmax0, 2.0, 1.0, 0.5, 0.2]) if a < b]
    for fmin, fmax in ranges:
        fa = fmin0 if fmin is None else fmin; fb = fmax0 if fmax is None else fmax
        if fa >= fb: continue
        freqs, qvals = transit_autofreq(t, fmin=fa, fmax=fb, qmin_fac=0.5)
        qmins, qmaxs = 0.5 * np.asarray(qvals), 2.0 * np.asarray(qvals)
        for mm in [0.9 * 3e9, 0.9 * 6e9, 0.9 * 8e9, 0.9 * 12e9, 0.9 * 16e9, 0.9 * 23e9, 0.9 * 80e9]:
            r = batches(freqs, qmins, qmaxs, ndata, mm)
            if r is None: continue
            fbs, ntot_max, gs, out = r
            total += 1
            bad = [o for o in out if o[5] > gs]
            if bad:
                hits.append((ndata, baseline, fa, fb, mm, len(freqs), fbs, ntot_max, gs, bad[0]))
print("configs scanned: %d, with OOB batches: %d" % (total, len(hits)))
for h in hits[:40]:
    ndata, baseline, fa, fb, mm, nfreq, fbs, ntot_max, gs, b = h
    print("ndata=%6d T=%5.0f fmin=%.4f fmax=%.3f max_memory=%.1e nfreq=%7d fbs=%6d ntot_max=%5d gs=%9d | batch %d nf=%d nb0=%d nbf=%d ntot=%d all_bins=%d overrun=%d floats (%.2f MB)"
          % (ndata, baseline, fa, fb, mm, nfreq, fbs, ntot_max, gs, b[0], b[1], b[2], b[3], b[4], b[5], b[5] - gs, (b[5] - gs) * 4 / 1e6))
