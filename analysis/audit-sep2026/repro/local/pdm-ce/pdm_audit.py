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
section = sys.argv[1] if len(sys.argv) > 1 else 'all'

if section in ('all', 'accuracy'):
    print("=== (1) GPU vs float64 reference; original and _fast kernels ===")
    freqs = np.linspace(0.05, 5.0, 4000)
    for label, kw in [('uniform', dict()), ('gappy', dict(gappy=True)),
                      ('hetero', dict(hetero=True)), ('BJD', dict(t0=2455000.0))]:
        t, y, err = make(400, 30.0, 1.7, 1, **kw)
        for kind in KINDS:
            for nb, dp in [(5, 0.02), (10, 0.05), (25, 0.1), (50, 0.2)]:
                r = ref(t, y, err, freqs, kind, nb, dp)
                g = gpu(proc, t, y, err, freqs, kind, nb, dp)
                gf = gpu(proc, t, y, err, freqs, kind + '_fast', nb, dp)
                print("%-8s %-15s nbins=%2d dphi=%.2f  max|gpu-ref|=%.2e max|fast-ref|=%.2e max|fast-gpu|=%.2e  argmax ref/gpu/fast=%d/%d/%d ref-peak=%.4f"
                      % (label, kind, nb, dp, np.max(np.abs(g - r)), np.max(np.abs(gf - r)), np.max(np.abs(gf - g)),
                         np.argmax(r), np.argmax(g), np.argmax(gf), freqs[np.argmax(r)]))
                if kind.startswith('binless'):
                    break  # nbins irrelevant; only first dphi to save time... (do two)
    # cross check the package CPU refs against mine
    t, y, err = make(60, 10.0, 1.3, 3)
    w = weights(err); fr = np.linspace(0.1, 3, 50)
    print("pkg pdm2_cpu vs mine (linterp): %.2e" % np.max(np.abs(np.array(pdm2_cpu(t, y, w, fr, nbins=10, linterp=True)) - ref_binned(t, y, w, fr, 10, True))))
    print("pkg pdm2_cpu vs mine (step)   : %.2e" % np.max(np.abs(np.array(pdm2_cpu(t, y, w, fr, nbins=10, linterp=False)) - ref_binned(t, y, w, fr, 10, False))))
    print("pkg binless tophat vs mine    : %.2e" % np.max(np.abs(np.array(binless_pdm_cpu(t, y, w, fr, 0.05, True)) - ref_binless(t, y, w, fr, 0.05, True))))
    print("pkg binless gauss vs mine     : %.2e" % np.max(np.abs(np.array(binless_pdm_cpu(t, y, w, fr, 0.05, False)) - ref_binless(t, y, w, fr, 0.05, False))))

if section in ('all', 'blocks'):
    print("=== (1b) fast kernels: block_size variations & ndata not multiple of block ===")
    freqs = np.linspace(0.05, 5.0, 300)
    for nd in [1, 2, 7, 255, 256, 257, 1000]:
        t, y, err = make(nd, 30.0, 1.7, 5)
        for kind in KINDS:
            if kind.startswith('binless') and nd > 300: continue
            r = ref(t, y, err, freqs, kind)
            for bs in [32, 100, 256]:
                gf = gpu(proc, t, y, err, freqs, kind + '_fast', block_size=bs)
                g = gpu(proc, t, y, err, freqs, kind, block_size=bs)
                print("ndata=%4d %-15s bs=%3d max|fast-ref|=%.2e max|orig-ref|=%.2e" % (nd, kind, bs, np.nanmax(np.abs(gf - r)), np.nanmax(np.abs(g - r))))

if section in ('all', 'formats'):
    print("=== (2) deprecated (t,y,w,freqs) vs modern (t,y,err); weight normalization ===")
    t, y, err = make(300, 30.0, 1.7, 2, hetero=True)
    freqs = np.linspace(0.05, 5.0, 400)
    w = weights(err)
    for kind in KINDS + [k + '_fast' for k in KINDS]:
        modern = gpu(proc, t, y, err, freqs, kind)
        dep = proc.run([(t, y, w, freqs)], kind=kind, nbins=10, dphi=0.05); proc.finish(); dep = np.copy(dep[0])
        wun = 1.0 / err ** 2  # unnormalized weights
        dep_un = proc.run([(t, y, wun, freqs)], kind=kind, nbins=10, dphi=0.05); proc.finish(); dep_un = np.copy(dep_un[0])
        r = ref(t, y, err, freqs, kind.replace('_fast', ''))
        print("%-20s max|dep-modern|=%.2e  max|dep_unnormalized_w - ref|=%.2e (ref peak %.4f, unnorm peak %.4f, unnorm min/max=%.3g/%.3g)"
              % (kind, np.max(np.abs(dep - modern)), np.max(np.abs(dep_un - r)), r.max(), dep_un.max(), dep_un.min(), dep_un.max()))
    print("--- unnormalized uniform weights (w=ones) ---")
    t, y, err = make(300, 30.0, 1.7, 2)
    r = ref(t, y, err, freqs, 'binned_linterp')
    for kind in ['binned_linterp', 'binned_linterp_fast', 'binned_step_fast']:
        dep_un = proc.run([(t, y, np.ones_like(y), freqs)], kind=kind, nbins=10); proc.finish(); dep_un = np.copy(dep_un[0])
        print("%-20s w=ones: max|dep-ref|=%.2e" % (kind, np.max(np.abs(dep_un - r))))

if section in ('all', 'theta'):
    print("=== (2b) normalization: returned P vs Stellingwerf theta (dof-corrected), noise bias ===")
    for N, nb in [(20, 10), (50, 10), (100, 10), (1000, 10), (100, 25)]:
        r = np.random.RandomState(11)
        t = np.sort(r.rand(N) * 30); y = r.randn(N); err = np.ones(N)
        freqs = np.linspace(0.05, 5.0, 500)
        g = gpu(proc, t, y, err, freqs, 'binned_step', nb)
        th = stellingwerf_theta(t, y, freqs, nb)
        print("noise N=%4d nbins=%2d: mean(returned P)=%.3f (expect ~(M-1)/(N-1)=%.3f)  mean(1-theta_S78)=%.3f  max|P-(1-theta_S78)|=%.3f"
              % (N, nb, g.mean(), (nb - 1) / (N - 1), (1 - th).mean(), np.max(np.abs(g - (1 - th)))))

if section in ('all', 'batch'):
    print("=== (batch) multi-LC run vs single; batched_run_const_nfreq ===")
    freqs = np.linspace(0.05, 5.0, 500)
    lcs = [make(100 + 137 * i, 30.0, 0.7 + 0.5 * i, 20 + i, hetero=(i % 2 == 0)) for i in range(6)]
    for kind in ['binned_linterp', 'binned_step_fast', 'binless_gauss_fast']:
        multi = proc.run(lcs, freqs=freqs, kind=kind); proc.finish()
        multi = [np.copy(p) for f, p in multi]
        single = [gpu(proc, *lc, freqs, kind) for lc in lcs]
        bres = proc.batched_run_const_nfreq(lcs, batch_size=4, freqs=freqs, kind=kind)
        d1 = max(np.max(np.abs(m - s)) for m, s in zip(multi, single))
        d2 = max(np.max(np.abs(b[1] - s)) for b, s in zip(bres, single))
        # per-LC frequency grids
        fl = [freqs * (1 + 0.1 * i) for i in range(6)]
        multi2 = proc.run(lcs, freqs=fl, kind=kind); proc.finish()
        d3 = max(np.max(np.abs(np.copy(p) - gpu(proc, *lc, f, kind))) for (f, p), lc in zip(multi2, lcs))
        print("%-20s multi-vs-single=%.2e batched-vs-single=%.2e perLC-grids=%.2e" % (kind, d1, d2, d3))

if section in ('all', 'edge'):
    print("=== (4) edge cases ===")
    freqs = np.linspace(0.05, 5.0, 100)
    r = np.random.RandomState(0)
    t = np.sort(r.rand(50) * 30)
    def try_run(label, data, **kw):
        try:
            res = proc.run(data, freqs=freqs, **kw); proc.finish()
            p = np.copy(res[0][1])
            print("%-40s -> finite=%s  min=%.3g max=%.3g nan=%d" % (label, np.all(np.isfinite(p)), np.nanmin(p), np.nanmax(p), np.isnan(p).sum()))
        except Exception as e:
            print("%-40s -> EXC %s: %s" % (label, type(e).__name__, str(e)[:120]))
    for kind in ['binned_linterp', 'binned_step_fast', 'binless_tophat', 'binless_gauss_fast']:
        try_run('constant y [%s]' % kind, [(t, np.ones(50) * 3.0, np.ones(50))], kind=kind)
        try_run('N=5 < nbins=10 [%s]' % kind, [(t[:5], r.randn(5), np.ones(5))], kind=kind)
        yn = r.randn(50); yn[3] = np.nan
        try_run('NaN in y [%s]' % kind, [(t, yn, np.ones(50))], kind=kind)
        e0 = np.ones(50); e0[3] = 0.0
        try_run('dy=0 at one point [%s]' % kind, [(t, r.randn(50), e0)], kind=kind)
        try_run('float32 freqs [%s]' % kind, [(t, r.randn(50), np.ones(50))], kind=kind)
    try:
        res = proc.run([(t, r.randn(50), np.ones(50))], freqs=freqs.astype(np.float32)); proc.finish()
        print("float32 freqs (explicit)                 -> ok, first freq %r" % res[0][0][0])
    except Exception as e:
        print("float32 freqs (explicit) -> EXC %s: %s" % (type(e).__name__, str(e)[:200]))
    try:
        res = proc.run([(t, r.randn(50), np.ones(50))], freqs=[freqs, freqs], kind='binned_step'); proc.finish()
        print("2 freq grids for 1 LC -> no error, returned %d results" % len(res))
    except Exception as e:
        print("2 freq grids for 1 LC -> EXC %s: %s" % (type(e).__name__, str(e)[:200]))
    try:
        res = proc.run([(t, r.randn(50), np.ones(50))], freqs=freqs, kind='binned_step', block_size=512); proc.finish()
    except Exception as e:
        print("block_size=512 -> EXC %s: %s" % (type(e).__name__, str(e)[:100]))
    # freq = 0
    res = proc.run([(t, r.randn(50), np.ones(50))], freqs=np.array([0.0, 1.0]), kind='binned_step'); proc.finish()
    print("freq=0 -> P=%r" % np.copy(res[0][1]))

if section in ('all', 'phase1'):
    print("=== PHASE(t,f)==1.0 edge: t*f = -1e-9 in float32 -> bin index NBINS in var_step_function ===")
    # symmetric t so mean-subtraction leaves t unchanged; one point at -1e-9
    r = np.random.RandomState(4)
    base = np.array([-3.0, 3.0, -2.5, 2.5, -1.7, 1.7, -0.9, 0.9, -1e-9, 1e-9])
    t = base.copy(); y = r.randn(len(t)) + 12; err = np.ones(len(t))
    # add pairs to keep mean 0 exactly
    more = r.rand(40) * 3; t = np.concatenate([t, more, -more]); y = np.concatenate([y, r.randn(80) + 12]); err = np.ones(len(t))
    print("mean(t) after normalize:", np.mean(t), " float32 PHASE(-1e-9,1)=", np.float32(-1e-9) - np.floor(np.float32(-1e-9)))
    freqs = np.array([1.0, 1.0000001, 0.999, 2.0, 3.0])
    for kind in ['binned_step', 'binned_step_fast', 'binned_linterp', 'binned_linterp_fast']:
        g = gpu(proc, t, y, err, freqs, kind, 10)
        rr = ref(t, y, err, freqs, kind.replace('_fast', ''), 10)
        print("%-20s gpu=%s\n%-20s ref=%s" % (kind, np.array2string(g, precision=6), '', np.array2string(rr, precision=6)))
    # repeat with many trials to show it's the step kernel differing
    diffs = []
    for s in range(20):
        r = np.random.RandomState(s)
        y = r.randn(len(t)) + 12
        g = gpu(proc, t, y, err, freqs[:1], 'binned_step', 10)[0]
        gf = gpu(proc, t, y, err, freqs[:1], 'binned_step_fast', 10)[0]
        rr = ref(t, y, err, freqs[:1], 'binned_step', 10)[0]
        diffs.append((g - rr, gf - rr))
    diffs = np.array(diffs)
    print("over 20 seeds at f=1: max|step-ref|=%.3g max|step_fast-ref|=%.3g" % (np.max(np.abs(diffs[:, 0])), np.max(np.abs(diffs[:, 1]))))

if section in ('all', 'precision'):
    print("=== float32 phase precision at large t*f (baseline 3650 d, f up to 50/d) ===")
    for baseline, fmax, nb in [(30, 5, 10), (365, 20, 10), (3650, 20, 10), (3650, 50, 10), (3650, 50, 50)]:
        t, y, err = make(500, float(baseline), fmax * 0.7, 9)
        freqs = np.linspace(fmax * 0.69, fmax * 0.71, 2000)
        for kind in ['binned_step', 'binless_tophat']:
            r = ref(t, y, err, freqs, kind, nb, 0.05)
            g = gpu(proc, t, y, err, freqs, kind, nb, 0.05)
            print("baseline=%5d fmax=%2d nbins=%2d %-15s max|gpu-ref|=%.2e  ref peak=%.4f gpu peak=%.4f  argmax ref=%d gpu=%d  max phase err (cycles)~%.2e"
                  % (baseline, fmax, nb, kind, np.max(np.abs(g - r)), r.max(), g.max(), np.argmax(r), np.argmax(g), baseline / 2 * fmax * 6e-8))

if section in ('all', 'timing'):
    print("=== relative timings (4090, shared GPU: noisy; repeated min-of-5) ===")
    freqs = np.linspace(0.05, 5.0, 20000)
    for nd in [200, 1000, 5000]:
        t, y, err = make(nd, 30.0, 1.7, 7)
        for kind in KINDS:
            if kind.startswith('binless') and nd > 1000:
                continue
            fr = freqs if not kind.startswith('binless') else freqs[:2000]
            times = {}
            for k in [kind, kind + '_fast']:
                gpu(proc, t, y, err, fr[:10], k)  # warm
                best = 1e9
                for rep in range(5):
                    t0 = time.perf_counter(); gpu(proc, t, y, err, fr, k); best = min(best, time.perf_counter() - t0)
                times[k] = best
            print("ndata=%5d nf=%5d %-15s orig=%.4fs fast=%.4fs  speedup=%.2fx" % (nd, len(fr), kind, times[kind], times[kind + '_fast'], times[kind] / times[kind + '_fast']))

if section in ('all', 'ref32'):
    print("=== is the GPU-vs-ref residual purely float32 bin-edge flips? reference with float32 t*f ===")
    def ref_binned32(t, y, w, freqs, nbins, linterp):
        t = t - np.mean(t); y = y - np.mean(y); w = w / np.sum(w)
        ybar = np.dot(w, y); var = np.dot(w, (y - ybar) ** 2)
        t32 = t.astype(np.float32); f32 = freqs.astype(np.float32)
        out = np.empty(len(freqs))
        for i, f in enumerate(f32):
            tf = t32 * f
            ph = (tf - np.floor(tf)).astype(np.float64)   # float32 product, exact subtraction
            b = (ph * nbins).astype(int) % nbins
            wt = np.bincount(b, weights=w, minlength=nbins); sm = np.bincount(b, weights=w * y, minlength=nbins)
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
    freqs = np.linspace(0.05, 5.0, 4000)
    t, y, err = make(400, 30.0, 1.7, 1, gappy=True)
    w = weights(err)
    for kind, nb in [('binned_step', 25), ('binned_linterp', 25), ('binned_step', 10)]:
        r64 = ref(t, y, err, freqs, kind, nb); r32 = ref_binned32(t, y, w, freqs, nb, 'linterp' in kind)
        g = gpu(proc, t, y, err, freqs, kind, nb)
        print("gappy %-15s nbins=%d: max|gpu-ref64|=%.2e  max|gpu-ref32|=%.2e  max|ref32-ref64|=%.2e" % (kind, nb, np.max(np.abs(g - r64)), np.max(np.abs(g - r32)), np.max(np.abs(r32 - r64))))

if section == 'phase1b':
    print("=== PHASE==1.0: compare against float32-emulating reference (bin = int(phase*NB) % NB) ===")
    def ref_step32(t, y, w, freqs, nbins, modsecond=True):
        t = t - np.mean(t); y = y - np.mean(y); w = w / np.sum(w)
        ybar = np.dot(w, y); var = np.dot(w, (y - ybar) ** 2)
        t32 = t.astype(np.float32); f32 = freqs.astype(np.float32)
        out = np.empty(len(freqs))
        for i, f in enumerate(f32):
            tf = t32 * f; ph = (tf - np.floor(tf)).astype(np.float64)
            b = (ph * nbins).astype(int)
            bm = b % nbins
            wt = np.bincount(bm, weights=w, minlength=nbins); sm = np.bincount(bm, weights=w * y, minlength=nbins)
            mean = np.where(wt > 0, sm / np.where(wt > 0, wt, 1), 0.0)
            model = mean[bm] if modsecond else np.where(b < nbins, mean[np.minimum(b, nbins - 1)], np.nan)
            out[i] = 1 - np.dot(w, (y - model) ** 2) / var
        return out
    r = np.random.RandomState(4)
    base = np.array([-3.0, 3.0, -2.5, 2.5, -1.7, 1.7, -0.9, 0.9, -1e-9, 1e-9])
    more = r.rand(40) * 3; t = np.concatenate([base, more, -more]); err = np.ones(len(t))
    freqs = np.array([1.0, 2.0, 3.0, 0.5])
    print("phases (float32) of the -1e-9 point at f=1,2,3,0.5:", [(np.float32(-1e-9) * np.float32(f)) - np.floor(np.float32(-1e-9) * np.float32(f)) for f in freqs])
    ds, dsf, dl = [], [], []
    for s in range(30):
        y = r.randn(len(t)) + 12
        w = weights(err)
        r32 = ref_step32(t, y, w, freqs, 10)
        g = gpu(proc, t, y, err, freqs, 'binned_step', 10); gf = gpu(proc, t, y, err, freqs, 'binned_step_fast', 10)
        gl = gpu(proc, t, y, err, freqs, 'binned_linterp', 10)
        rl = ref_binned32_lin = None
        ds.append(np.abs(g - r32)); dsf.append(np.abs(gf - r32))
    ds = np.array(ds); dsf = np.array(dsf)
    print("30 seeds, per-frequency max|binned_step - ref32|      =", np.array2string(ds.max(axis=0), precision=3))
    print("30 seeds, per-frequency max|binned_step_fast - ref32| =", np.array2string(dsf.max(axis=0), precision=3))
    # control: same data but the offending point moved to +1e-9 only (no PHASE==1.0)
    t2 = t.copy(); t2[8] = 1e-9; t2[9] = -1e-9 * 0 + 2e-9  # break exact symmetry slightly? keep mean ~0: mean = 3e-9/90
    ds2 = []
    for s in range(30):
        y = r.randn(len(t)) + 12; w = weights(err)
        ds2.append(np.abs(gpu(proc, t2, y, err, freqs, 'binned_step', 10) - ref_step32(t2, y, w, freqs, 10)))
    print("control (no point with PHASE==1.0): max|binned_step - ref32| =", np.array2string(np.array(ds2).max(axis=0), precision=3))
