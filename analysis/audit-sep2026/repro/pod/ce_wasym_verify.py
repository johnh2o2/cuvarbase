"""Independent minimal reproduction of finding 21: weighted-CE truncation asymmetry.
Controlled inputs: hand-placed Y values, homoscedastic dy, one trial frequency, mag_bins=5."""
import numpy as np
from scipy.special import ndtr
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.utils import normalize_light_curves

def gpu_bins(proc, t, y, dy, freqs):
    proc.run([(t, y, dy)], freqs=freqs); proc.finish()
    data = normalize_light_curves([(t, y, dy)])
    mems = proc.allocate(data, freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
    return np.copy(res[0][1]), mems[0].bins_g.get().reshape(len(freqs), proc.phase_bins, proc.mag_bins)

MB, PB = 5, 1   # one phase bin -> the mag histogram is the only thing that matters
# y already in [0,1] units: y = 0 and y = 1 anchor the range so normalization is the identity
# points: lower edge of bin 2 (+eps), middle of bin 2, upper edge of bin 2 (-eps), plus the anchors
eps = 0.01
Yc = np.array([0.0, 0.4 + eps, 0.5, 0.6 - eps, 1.0])
labels = ['Y=0 (faintest)', 'bin2 lower edge+0.01', 'bin2 centre', 'bin2 upper edge-0.01', 'Y=1 (brightest)']
for sig in [0.02, 0.05, 0.1]:   # sigma in units of the full range; bin width = 0.2
    print("\n=== sigma=%.2f (= %.2f bin widths); default max_phi=3 -> 3sigma = %.2f bin widths ===" % (sig, sig*MB, 3*sig*MB))
    for max_phi in [3.0, 1e6]:
        proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, weighted=True, max_phi=max_phi)
        N = len(Yc)
        t = np.linspace(0, 1, N)
        # one point per run so we can attribute mass per point
        print("  max_phi=%g" % max_phi)
        for k in range(N):
            # to keep normalization the identity we always include the anchors; isolate point k by giving
            # the other points a run of their own and differencing is messy -> instead run all, then run
            # anchors-only and difference (anchors are always present).
            pass
        ce_all, b_all = gpu_bins(proc, t, Yc, sig*np.ones(N), np.array([0.0]))
        # exact per-point masses
        m = np.arange(MB); lower = m/MB; upper = (m+1)/MB
        P = ndtr((upper[None,:]-Yc[:,None])/sig) - ndtr((lower[None,:]-Yc[:,None])/sig)
        # kernel-rule emulation
        z = lower[None,:]-Yc[:,None]; m0 = np.floor(Yc*MB).astype(int)
        keep = ~((np.abs(z) > max_phi*sig) & (m[None,:] != m0[:,None]))
        Pk = np.where(keep, P, 0)
        print("    GPU bins            :", np.round(b_all[0,0], 4), " total=%.4f" % b_all[0,0].sum())
        print("    exact-mass bins     :", np.round(P.sum(0), 4), " total=%.4f" % P.sum())
        print("    kernel-rule emul    :", np.round(Pk.sum(0), 4), " total=%.4f" % Pk.sum())
        print("    max|GPU-exact|=%.2e  max|GPU-emul|=%.2e" % (np.abs(b_all[0,0]-P.sum(0)).max(), np.abs(b_all[0,0]-Pk.sum(0)).max()))
        for k in range(N):
            print("      %-24s exact mass=%.3f  kernel-rule retained=%.3f  (bins kept: %s)" % (labels[k], P[k].sum(), Pk[k].sum(), np.where(keep[k])[0].tolist()))

# --- periodogram-level effect, several seeds, small homoscedastic errors (typical weighted use) ---
print("\n=== periodogram: max_phi=3 (default) vs 1e6 (== exact mass) vs unweighted; 3000 freqs ===")
def make(ndata, baseline, f0, seed, noise, amp=1.0):
    r = np.random.RandomState(seed)
    t = np.sort(r.rand(ndata))*baseline
    y = 12 + amp*np.sin(2*np.pi*f0*t) + 0.3*amp*np.sin(4*np.pi*f0*t) + noise*r.randn(ndata)
    return t, y, noise*np.ones(ndata)
freqs = np.linspace(0.1, 3.0, 3000)
p3proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=3.0)
pinfproc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=1e6)
punproc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
def run(proc, t, y, dy):
    r = proc.run([(t, y, dy)], freqs=freqs); proc.finish(); return np.copy(r[0][1])
sig = lambda p: (np.mean(p)-np.min(p))/np.std(p)
nshift = 0; ntot = 0
for noise in [0.05, 0.15, 0.4]:
    for seed in range(6):
        t, y, dy = make(200, 20.0, 1.3, seed, noise)
        p3 = run(p3proc, t, y, dy); pinf = run(pinfproc, t, y, dy); pun = run(punproc, t, y, dy)
        f3, finf, fun = freqs[np.argmin(p3)], freqs[np.argmin(pinf)], freqs[np.argmin(pun)]
        ntot += 1; nshift += (np.argmin(p3) != np.argmin(pinf))
        print("noise=%.2f seed=%d: argmin f: phi3=%.4f phiInf=%.4f unw=%.4f | sig %.2f/%.2f/%.2f | max|p3-pinf|=%.3e (spectrum std %.3e, depth %.3e)"
              % (noise, seed, f3, finf, fun, sig(p3), sig(pinf), sig(pun), np.abs(p3-pinf).max(), pinf.std(), pinf.mean()-pinf.min()))
print("argmin grid-index differs (phi3 vs exact) in %d/%d runs" % (nshift, ntot))
