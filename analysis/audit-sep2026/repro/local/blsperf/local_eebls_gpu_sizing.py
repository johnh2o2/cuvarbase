"""Replicate eebls_gpu's batch/memory arithmetic (pure python copies of bls.py helpers) to check whether a batch's
nbins_tot can exceed the global nbins_tot_max used to size the device arrays (-> OOB writes)."""
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('bf', '/Users/johnhoffman/Documents/cuvarbase/cuvarbase/bls_frequencies.py')
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)

def dnbins(nbins, dlogq):
    if dlogq < 0: return 1
    n = int(np.floor(dlogq*nbins)); return n if n > 0 else 1
def nbins_iter(i, nb0, dlogq):
    nb = nb0
    for j in range(i): nb += dnbins(nb, dlogq)
    return nb
def count_tot_nbins(nbins0, nbinsf, dlogq):
    ntot = 0; i = 0
    while nbins_iter(i, nbins0, dlogq) <= nbinsf:
        ntot += nbins_iter(i, nbins0, dlogq); i += 1
    return ntot

def check(name, freqs, qmins, qmaxs, ndata, max_memory, nstreams=5, noverlap=3, dlogq=0.2):
    nbins0_max = int(np.floor(1./np.max(qmaxs))); nbinsf_max = int(np.ceil(1./np.min(qmins)))
    ntot_max = count_tot_nbins(nbins0_max, nbinsf_max, dlogq)
    mem0 = ndata*3*4 + len(freqs)*5*4
    mem_per_f = 4*nstreams*ntot_max*noverlap*4
    fbs = int(float(max_memory - mem0)/mem_per_f)
    gs = fbs*ntot_max*noverlap
    nb = int(np.ceil(len(freqs)/fbs)); worst = 0; over = 0
    for b in range(nb):
        i0, i1 = fbs*b, min(len(freqs), fbs*(b+1))
        nb0 = int(np.floor(1./np.max(qmaxs[i0:i1]))); nbf = int(np.ceil(1./np.min(qmins[i0:i1])))
        ntot = count_tot_nbins(nb0, nbf, dlogq)
        all_bins = (i1-i0)*ntot*noverlap
        if ntot > ntot_max: over += 1
        worst = max(worst, ntot/ntot_max)
        if all_bins > gs and over == 1 and ntot > ntot_max:
            print(f"   batch {b}: nb0={nb0} nbf={nbf} nbins_tot={ntot} > nbins_tot_max={ntot_max} (global nb0={nbins0_max}, nbf={nbinsf_max}); all_bins={all_bins} > allocated gs={gs}  --> OOB")
    print(f"{name}: nfreq={len(freqs)} fbs={fbs} nbatches={nb} nbins_tot_max={ntot_max}; batches with nbins_tot > max: {over}; worst ratio {worst:.3f}")

for name, (pmin, pmax, T, ndata) in dict(ZTF=(0.5,100.,730.,150), HAT=(0.5,100.,3650.,6000), TESS=(0.5,13.5,27.,20000), Kepler=(0.5,100.,1460.,65000)).items():
    f, q = bf.keplerian_freq_grid(pmin, pmax, T, oversampling=2, return_qvals=True); f = f.astype(np.float64)
    qmins, qmaxs = 0.5*q, 2.0*q
    check(name+' (1.5GB cap)', f, qmins, qmaxs, ndata, 1.5e9)
    check(name+' (20GB default-like)', f, qmins, qmaxs, ndata, 20e9)
    if name == 'HAT':
        check('HAT first 20K freqs (1.5GB cap, the crashing config)', f[:20000], qmins[:20000], qmaxs[:20000], ndata, 1.5e9)
# generic small example: two frequencies, scalar-vs-array
