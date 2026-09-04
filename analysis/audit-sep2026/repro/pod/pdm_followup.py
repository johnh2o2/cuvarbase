"""PDM audit: GPU kernels vs float64 numpy reference of Stellingwerf theta."""
import sys, time, warnings
import numpy as np
from cuvarbase.pdm import PDMAsyncProcess, pdm2_cpu, binless_pdm_cpu
from cuvarbase.utils import weights

warnings.simplefilter('ignore', DeprecationWarning)

# ---------------- float64 numpy references ----------------
def ref_binned(t, y, w, freqs, nbins, linterp):
    """1 - SS_within/SS_tot, weighted, same bin defs as pdm.cu."""
    t = t - np.mean(t); y = y - np.mean(y)
    w = w / np.sum(w)
    ybar = np.dot(w, y); var = np.dot(w, (y - ybar) ** 2)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        ph = (t * f) % 1.0
        b = (ph * nbins).astype(int) % nbins
        wt = np.bincount(b, weights=w, minlength=nbins)
        sm = np.bincount(b, weights=w * y, minlength=nbins)
        mean = np.where(wt > 0, sm / np.where(wt > 0, wt, 1), 0.0)
        if linterp:
            alpha = ph * nbins - np.floor(ph * nbins) - 0.5
            b0 = np.where(alpha < 0, b - 1, b); b1 = np.where(alpha < 0, b, b + 1)
            b0[b0 < 0] += nbins; b1[b1 >= nbins] -= nbins
            alpha = np.where(alpha < 0, alpha + 1, alpha)
            model = (1 - alpha) * mean[b0] + alpha * mean[b1]
        else:
            model = mean[b]
        out[i] = 1 - np.dot(w, (y - model) ** 2) / var
    return out

def ref_binless(t, y, w, freqs, dphi, tophat):
    t = t - np.mean(t); y = y - np.mean(y)
    w = w / np.sum(w)
    ybar = np.dot(w, y); var = np.dot(w, (y - ybar) ** 2)
    out = np.empty(len(freqs))
    dt = np.abs(t[:, None] - t[None, :])
    for i, f in enumerate(freqs):
        dph = (dt * f) % 1.0
        dph = np.where(dph > 0.5, 1 - dph, dph)
        K = (dph < dphi).astype(float) if tophat else np.exp(-0.5 * (dph / dphi) ** 2)
        K = K * w[None, :]
        mbar = (K @ y) / K.sum(axis=1)
        out[i] = 1 - np.dot(w, (y - mbar) ** 2) / var
    return out

def stellingwerf_theta(t, y, freqs, nbins):
    """Unweighted Stellingwerf (1978) theta with dof corrections."""
    N = len(t)
    sig2 = np.sum((y - y.mean()) ** 2) / (N - 1)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        ph = (t * f) % 1.0
        b = (ph * nbins).astype(int) % nbins
        ss = 0.0; M = 0
        for k in range(nbins):
            sel = b == k
            n = sel.sum()
            if n > 1:
                ss += np.sum((y[sel] - y[sel].mean()) ** 2); M += 1
            elif n == 1:
                M += 1
        s2 = ss / (N - M)
        out[i] = s2 / sig2
    return out

def make(ndata, baseline, f0, seed, gappy=False, hetero=False, t0=0.0):
    r = np.random.RandomState(seed)
    t = np.sort(r.rand(ndata)) * baseline
    if gappy:  # nightly windows: keep 30% of each day
        t = t[(t % 1.0) < 0.3]
    y = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t + 1) + 0.2 * r.randn(len(t))
    err = 0.2 * (0.5 + r.rand(len(t))) if hetero else 0.2 * np.ones(len(t))
    y += 12.0
    return t + t0, y, err

KINDS = ['binned_step', 'binned_linterp', 'binless_tophat', 'binless_gauss']

def gpu(proc, t, y, err, freqs, kind, nbins=10, dphi=0.05, block_size=256):
    res = proc.run([(t, y, err)], freqs=freqs, kind=kind, nbins=nbins, dphi=dphi, block_size=block_size)
    proc.finish()
    return np.copy(res[0][1])

def ref(t, y, err, freqs, kind, nbins=10, dphi=0.05):
    w = weights(err)
    if kind.startswith('binned'):
        return ref_binned(t, y, w, freqs, nbins, linterp='linterp' in kind)
    return ref_binless(t, y, w, freqs, dphi, tophat='tophat' in kind)


proc = PDMAsyncProcess()
section = sys.argv[1]
if section == 'timing2':
    print("=== PDM orig vs fast, min-of-10, nf=20000 (binned) / 2000 (binless); shared 4090 ===")
    freqs = np.linspace(0.05, 5.0, 20000)
    for nd in [200, 1000, 5000, 20000]:
        t, y, err = make(nd, 30.0, 1.7, 7)
        for kind, nb in [('binned_step', 10), ('binned_linterp', 10), ('binned_step', 50), ('binless_tophat', 10), ('binless_gauss', 10)]:
            if kind.startswith('binless') and nd > 1000: continue
            fr = freqs if not kind.startswith('binless') else freqs[:2000]
            times = {}
            for k in [kind, kind + '_fast']:
                gpu(proc, t, y, err, fr[:10], k, nb)
                best = 1e9
                for rep in range(10):
                    t0 = time.perf_counter(); gpu(proc, t, y, err, fr, k, nb); best = min(best, time.perf_counter() - t0)
                times[k] = best
            print("ndata=%5d nf=%5d %-15s nbins=%2d orig=%.4fs fast=%.4fs  fast/orig speedup=%.2fx" % (nd, len(fr), kind, nb, times[kind], times[kind + '_fast'], times[kind] / times[kind + '_fast']))
if section == 'argmax':
    print("=== does the dof-free statistic change the argmax vs Stellingwerf theta? small-N gappy data ===")
    nd = 0
    for N in [25, 40]:
        for nb in [10]:
            same = 0; tot = 0
            for s in range(40):
                r = np.random.RandomState(s)
                t = np.sort(r.rand(N) * 30); t = t[(t % 1.0) < 0.4]
                y = np.sin(2 * np.pi * 1.7 * t) + 0.8 * r.randn(len(t)); err = np.ones(len(t))
                freqs = np.linspace(0.05, 5.0, 2000)
                g = gpu(proc, t, y, err, freqs, 'binned_step', nb)
                th = stellingwerf_theta(t, y, freqs, nb)
                same += int(np.argmax(g) == np.argmin(th)); tot += 1
            print("N~%d nbins=%d: argmax agreement between returned P and Stellingwerf theta: %d/%d" % (N * 0.4, nb, same, tot))
