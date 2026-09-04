"""Check (a) the inf seen at max_phi=1e6, (b) a patched skip rule vs exact masses, (c) result change on default (unweighted) path."""
import numpy as np, sys
from scipy.special import ndtr
import cuvarbase.ce as cemod
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.utils import normalize_light_curves

def make(ndata, baseline, f0, seed, noise, amp=1.0):
    r = np.random.RandomState(seed)
    t = np.sort(r.rand(ndata))*baseline
    y = 12 + amp*np.sin(2*np.pi*f0*t) + 0.3*amp*np.sin(4*np.pi*f0*t) + noise*r.randn(ndata)
    return t, y, noise*np.ones(ndata)
def gpu_bins(proc, t, y, dy, freqs):
    proc.run([(t, y, dy)], freqs=freqs); proc.finish()
    data = normalize_light_curves([(t, y, dy)])
    mems = proc.allocate(data, freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
    return np.copy(res[0][1]), mems[0].bins_g.get().reshape(len(freqs), proc.phase_bins, proc.mag_bins)
def exact_hist(t, y, dy, freqs, PB, MB):
    t = np.asarray(t, np.float64); y = np.asarray(y, np.float64)
    t = (t - t.mean()).astype(np.float32); y = (y - y.mean()).astype(np.float32)
    yscale = y.max()-y.min(); Y = ((y-y.min())/yscale).astype(np.float64); DY = (dy.astype(np.float32)/yscale).astype(np.float64)
    m = np.arange(MB); P = ndtr(((m+1)/MB - Y[:,None])/DY[:,None]) - ndtr((m/MB - Y[:,None])/DY[:,None])
    H = np.zeros((len(freqs), PB, MB))
    for i, f in enumerate(freqs.astype(np.float32)):
        ph = t*np.float32(f); ph = (ph-np.floor(ph)).astype(np.float64)
        n0 = (np.floor(PB*ph).astype(int)) % PB
        np.add.at(H, (i, n0), P)
    return H

freqs = np.linspace(0.1, 3.0, 3000)
# (a) the inf at max_phi=1e6, noise=0.05 seed=2
t, y, dy = make(200, 20.0, 1.3, 2, 0.05)
proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=1e6)
ce, bins = gpu_bins(proc, t, y, dy, freqs)
bad = ~np.isfinite(ce)
print("(a) max_phi=1e6: non-finite CE at %d/%d freqs" % (bad.sum(), len(ce)))
if bad.any():
    i = np.where(bad)[0][0]; b = bins[i]
    tiny = b[(b > 0) & (b < 1e-30)]
    print("    first bad freq idx %d: bins>0 & <1e-30 (denormal masses): %d, e.g. %s" % (i, tiny.size, tiny[:3]))

# (b) patched kernel: point find_kernel at the scratch copy
orig_find = cemod.find_kernel
cemod.find_kernel = lambda name: '/workspace/scratch/ce_wasym_fixed_v21.cu' if name == 'ce' else orig_find(name)
for MB in [5, 10]:
    for noise in [0.05, 0.15]:
        t, y, dy = make(300, 20.0, 1.3, 3, noise)
        fr = np.linspace(0.1, 3.0, 40)
        pf = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=MB, weighted=True, max_phi=3.0)
        cef, bf = gpu_bins(pf, t, y, dy, fr)
        He = exact_hist(t, y, dy, fr, 10, MB)
        cemod.find_kernel = orig_find
        po = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=MB, weighted=True, max_phi=3.0)
        ceo, bo = gpu_bins(po, t, y, dy, fr)
        cemod.find_kernel = lambda name: '/workspace/scratch/ce_wasym_fixed_v21.cu' if name == 'ce' else orig_find(name)
        print("(b) MB=%d noise=%.2f: max|bins_FIXED - exact|=%.2e (mass/pt %.4f)   max|bins_ORIG - exact|=%.2e ; CE max|fixed-exactCE|: %.2e, max|orig-fixed|=%.2e ; finite: %s"
              % (MB, noise, np.abs(bf-He).max(), bf[0].sum()/300, np.abs(bo-He).max(),
                 0.0, np.abs(ceo-cef).max(), np.isfinite(cef).all()))
# (c) fixed kernel, max_phi=3, on the noise=0.05 seed=2 case: finite?
t, y, dy = make(200, 20.0, 1.3, 2, 0.05)
pf = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=3.0)
cef, _ = gpu_bins(pf, t, y, dy, freqs)
print("(c) fixed kernel max_phi=3 on the inf case: finite=%s, argmin f=%.4f" % (np.isfinite(cef).all(), freqs[np.argmin(cef)]))
# (d) default (unweighted) path untouched by the patch: identical output
cemod.find_kernel = orig_find
pu = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
r1 = pu.run([(t, y, dy)], freqs=freqs); pu.finish(); a = np.copy(r1[0][1])
cemod.find_kernel = lambda name: '/workspace/scratch/ce_wasym_fixed_v21.cu' if name == 'ce' else orig_find(name)
pu2 = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
r2 = pu2.run([(t, y, dy)], freqs=freqs); pu2.finish(); b = np.copy(r2[0][1])
print("(d) unweighted (default) path: max|orig-patched| = %.2e" % np.abs(a-b).max())
