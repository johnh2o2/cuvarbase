"""CE audit: kernels vs numpy implementation of Graham et al. 2013 with cuvarbase's binning."""
import sys, time, warnings
import numpy as np
from scipy.special import gammaln, ndtr
from cuvarbase.ce import ConditionalEntropyAsyncProcess, conditional_entropy
from cuvarbase.utils import normalize_light_curves

def prep(t, y):
    """Emulate normalize_light_curves + ConditionalEntropyMemory.setdata (float32)."""
    t = np.asarray(t, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    t = t - t.mean(); y = y - y.mean()
    t = t.astype(np.float32); y = y.astype(np.float32)
    yscale = y.max() - y.min(); y0 = y.min()
    y01 = ((y - y0) / yscale)
    return t, y01, yscale

def hist_ref(t, y, freqs, nphase, nmag, PO=0, MO=0, clip=True, dtype=np.float64):
    """Integer 2D histogram [nf, nphase, nmag] with cuvarbase's bin defs.
    clip=True: correct handling (max point in top bin). clip=False: reproduce m0=nmag (kernel)."""
    t32, y01, _ = prep(t, y)
    m0 = np.floor(y01 * nmag).astype(int)
    if clip:
        m0 = np.minimum(m0, nmag - 1)
    H = np.zeros((len(freqs), nphase, nmag + (0 if clip else 1)), dtype=np.int64)
    tt = t32.astype(dtype); ff = np.asarray(freqs).astype(dtype)
    for i, f in enumerate(ff):
        ph = tt * f
        ph = ph - np.floor(ph)
        n0 = (np.floor(nphase * ph).astype(int)) % nphase
        for dn in range(PO + 1):
            n = (n0 - dn) % nphase
            for dmm in range(MO + 1):
                m = m0 - dmm
                ok = m >= 0
                np.add.at(H[i], (n[ok], m[ok]), 1)
    return H

def ce_from_hist(H, nmag, MO=0, graham=False):
    """H: [nf, nphase, nmag]. Returns kernel-style CE (with dm factor) or Graham's H (no dm)."""
    N = H.astype(np.float64)
    Nphi = N.sum(axis=2, keepdims=True)
    m = np.arange(N.shape[2])
    dm0 = (MO + 1) / nmag
    dm = np.where(m + MO + 1 > nmag, (nmag - m) * dm0 / (1 + MO), dm0)
    if graham:
        dm = np.ones_like(dm)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(N > 0, N * np.log(dm[None, None, :] * Nphi / np.where(N > 0, N, 1)), 0.0)
    return term.sum(axis=(1, 2)) / N.sum(axis=(1, 2))

def make(ndata, baseline, f0, seed, t0=0.0, amp=1.0, noise=0.3):
    r = np.random.RandomState(seed)
    t = np.sort(r.rand(ndata)) * baseline
    y = 12 + amp * np.sin(2 * np.pi * f0 * t) + 0.3 * amp * np.sin(4 * np.pi * f0 * t) + noise * r.randn(ndata)
    err = noise * (0.5 + r.rand(ndata))
    return t + t0, y, err

def run_gpu(proc, t, y, err, freqs, **kw):
    res = proc.run([(t, y, err)], freqs=freqs, **kw); proc.finish()
    return np.copy(res[0][1])

def run_gpu_with_bins(proc, t, y, err, freqs, **kw):
    """Non-fast path: return (ce, bins) by keeping the memory object."""
    proc.run([(t, y, err)], freqs=freqs, **kw); proc.finish()  # ensure compiled
    data = normalize_light_curves([(t, y, err)])
    mems = proc.allocate(data, freqs=[freqs])
    mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y, err)], memory=mems, freqs=[freqs], **kw); proc.finish()
    return np.copy(res[0][1]), mems[0].bins_g.get().reshape(len(freqs), proc.phase_bins, proc.mag_bins), mems[0]

section = sys.argv[1] if len(sys.argv) > 1 else 'all'

if section in ('all', 'hist'):
    print("=== (3a) histogram: GPU bins_g vs numpy; the y==1.0 point ===")
    t, y, err = make(300, 20.0, 1.3, 1)
    freqs = np.linspace(0.1, 3.0, 50)
    for PB, MB, PO, MO in [(10, 5, 0, 0), (10, 5, 1, 0), (10, 5, 0, 1), (10, 5, 1, 1), (7, 4, 0, 0)]:
        proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, phase_overlap=PO, mag_overlap=MO)
        ce_g, bins, mem = run_gpu_with_bins(proc, t, y, err, freqs)
        Hc = hist_ref(t, y, freqs, PB, MB, PO, MO, clip=True)
        Hb = hist_ref(t, y, freqs, PB, MB, PO, MO, clip=False)
        # kernel writes m=nmag into flat index offset + n*NMAG + NMAG == (n+1, 0); for n = NPHASE-1 -> next freq (0,0)
        Hb_flat = np.zeros(len(freqs) * PB * MB + 1, dtype=np.int64)
        for i in range(len(freqs)):
            for n in range(PB):
                for m in range(MB + 1):
                    Hb_flat[i * PB * MB + n * MB + m] += Hb[i, n, m]
        Hb_eff = Hb_flat[:-1].reshape(len(freqs), PB, MB)
        print("PB=%d MB=%d PO=%d MO=%d: sum(bins)=%d expected=%d (N=%d x (PO+1)(MO+1)); ymax point m0=%d; "
              "n_bins_differing vs correct-hist=%d, vs kernel-emulation(m0=nmag spill)=%d; spilled counts past array end (emulated)=%d"
              % (PB, MB, PO, MO, bins.sum(), Hc.sum(), len(t), int(np.floor(prep(t, y)[1].max() * MB)),
                 int((bins != Hc).sum()), int((bins != Hb_eff).sum()), int(Hb_flat[-1])))
        ce_ref_correct = ce_from_hist(Hc, MB, MO)
        ce_ref_buggy = ce_from_hist(Hb_eff, MB, MO)
        print("    CE: max|gpu - ref(correct hist)|=%.3e   max|gpu - ref(kernel-emulated hist)|=%.3e   graham offset (gpu - H_graham) mean=%.4f expect log(1/%d)=%.4f"
              % (np.max(np.abs(ce_g - ce_ref_correct)), np.max(np.abs(ce_g - ce_ref_buggy)), np.mean(ce_g - ce_from_hist(Hc, MB, MO, graham=True)) if MO == 0 else np.nan, MB, np.log(1.0 / MB)))
        # how many freqs have the spilled count end up in a *different frequency*?
        spill_cross = sum(int(Hb[i, PB - 1, MB]) for i in range(len(freqs) - 1))
        print("    counts spilled into the NEXT frequency's (phase0,mag0) bin: %d of %d freqs" % (spill_cross, len(freqs)))

if section in ('all', 'kernels'):
    print("=== (3b) every kernel vs numpy reference (correct hist and kernel-emulated hist) ===")
    t, y, err = make(400, 20.0, 1.3, 2)
    freqs = np.linspace(0.1, 3.0, 400)
    configs = [(10, 5, 0, 0), (10, 5, 1, 0), (10, 5, 0, 1), (10, 5, 1, 1), (20, 10, 0, 0), (8, 6, 2, 1)]
    for PB, MB, PO, MO in configs:
        Hc = hist_ref(t, y, freqs, PB, MB, PO, MO, clip=True)
        ref_c = ce_from_hist(Hc, MB, MO)
        for label, kw, rkw in [('standard_ce', dict(), dict()),
                               ('fast(shmem_lc=False)', dict(use_fast=True), dict(shmem_lc=False)),
                               ('faster(shmem_lc=True)', dict(use_fast=True), dict(shmem_lc=True)),
                               ('fast force_nblocks=1', dict(use_fast=True), dict(shmem_lc=False, force_nblocks=1)),
                               ('fast freq_batch=7', dict(use_fast=True), dict(shmem_lc=True, freq_batch_size=7)),
                               ('standard double', dict(use_double=True), dict()),
                               ('faster double', dict(use_double=True, use_fast=True), dict(shmem_lc=True))]:
            proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, phase_overlap=PO, mag_overlap=MO, **kw)
            g = run_gpu(proc, t, y, err, freqs, **rkw)
            d = np.abs(g - ref_c)
            print("PB=%2d MB=%2d PO=%d MO=%d %-22s max|gpu-ref|=%.2e  n(>1e-4)=%3d/%d  argmin ref=%d gpu=%d  ref@min=%.4f"
                  % (PB, MB, PO, MO, label, d.max(), (d > 1e-4).sum(), len(freqs), np.argmin(ref_c), np.argmin(g), ref_c.min()))

if section in ('all', 'weighted'):
    print("=== (3c) weighted CE: bins_g vs exact Gaussian-integrated histogram ===")
    t, y, err = make(300, 20.0, 1.3, 3, noise=0.15)
    freqs = np.linspace(0.1, 3.0, 40)
    for MB, max_phi, MO in [(5, 3.0, 0), (5, 1e6, 0), (10, 3.0, 0), (5, 3.0, 1)]:
        PB = 10
        proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, mag_overlap=MO, weighted=True, max_phi=max_phi)
        ce_g, bins, mem = run_gpu_with_bins(proc, t, y, err, freqs)
        t32, y01, yscale = prep(t, y)
        DY = (err.astype(np.float32) / yscale).astype(np.float64)
        Y = y01.astype(np.float64)
        m = np.arange(MB)
        lower = m[None, :] / MB; upper = (m[None, :] + 1 + MO) / MB
        P_exact = ndtr((upper - Y[:, None]) / DY[:, None]) - ndtr((lower - Y[:, None]) / DY[:, None])  # [N, MB]
        # kernel truncation rule
        z = lower - Y[:, None]; m0 = np.floor(Y * MB).astype(int)
        keep = ~((np.abs(z) > max_phi * DY[:, None]) & (m[None, :] != m0[:, None]))
        P_kern = np.where(keep, P_exact, 0.0)
        Href_exact = np.zeros((len(freqs), PB, MB)); Href_kern = np.zeros_like(Href_exact)
        for i, f in enumerate(freqs.astype(np.float32)):
            ph = t32 * np.float32(f); ph = (ph - np.floor(ph)).astype(np.float64)
            n0 = (np.floor(PB * ph).astype(int)) % PB
            np.add.at(Href_exact, (i, n0), P_exact); np.add.at(Href_kern, (i, n0), P_kern)
        tot_w = P_kern.sum(axis=1)
        print("MB=%d max_phi=%g MO=%d: per-point retained prob mass: min=%.3f mean=%.4f (exact would be %.3f..%.3f); frac points losing >5%%: %.3f; "
              "max|bins_gpu - kern_emul|=%.2e  max|bins_gpu - exact|=%.2e"
              % (MB, max_phi, MO, tot_w.min(), tot_w.mean(), P_exact.sum(axis=1).min(), P_exact.sum(axis=1).max(), np.mean(tot_w < 0.95),
                 np.max(np.abs(bins - Href_kern)), np.max(np.abs(bins - Href_exact))))
        # skewness of assignment: mean of (assigned bin center - Y) in kernel vs exact
        mc = (m + 0.5) / MB
        bias_k = ((P_kern * mc).sum(axis=1) / np.maximum(tot_w, 1e-12) - Y); bias_e = ((P_exact * mc).sum(axis=1) / P_exact.sum(axis=1) - Y)
        print("      mean assigned-mag bias (units of mag bin width): kernel=%.3f exact=%.3f" % (bias_k.mean() * MB, bias_e.mean() * MB))
        # CE from these
        def wce(Hw):
            Nphi = Hw.sum(axis=2, keepdims=True); dm = (MO + 1) / MB
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where((Hw > 0) & (Nphi > 1e-10), Hw * np.log(dm * Nphi / np.where(Hw > 0, Hw, 1)), 0)
            return term.sum(axis=(1, 2)) / Hw.sum(axis=(1, 2))
        print("      CE: max|gpu - wce(kern_emul)|=%.2e  max|gpu - wce(exact)|=%.2e" % (np.max(np.abs(ce_g - wce(Href_kern))), np.max(np.abs(ce_g - wce(Href_exact)))))
    # does the truncation change the recovered frequency / significance? compare max_phi=3 vs 1e6 on a spectrum
    t, y, err = make(300, 20.0, 1.3, 3, noise=0.15)
    freqs = np.linspace(0.1, 3.0, 3000)
    p3 = run_gpu(ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=3.0), t, y, err, freqs)
    pinf = run_gpu(ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, weighted=True, max_phi=1e6), t, y, err, freqs)
    punw = run_gpu(ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5), t, y, err, freqs)
    sig = lambda p: (np.mean(p) - np.min(p)) / np.std(p)
    print("spectrum: argmin f: max_phi=3 -> %.4f, max_phi=1e6 -> %.4f, unweighted -> %.4f; significance %.2f / %.2f / %.2f; max|p3-pinf|=%.3e"
          % (freqs[np.argmin(p3)], freqs[np.argmin(pinf)], freqs[np.argmin(punw)], sig(p3), sig(pinf), sig(punw), np.max(np.abs(p3 - pinf))))

if section in ('all', 'logprob'):
    print("=== (3d) log_prob and balanced_magbins (constdpdm_ce) vs numpy ===")
    t, y, err = make(400, 20.0, 1.3, 4)
    freqs = np.linspace(0.1, 3.0, 400)
    for PB, MB, PO, MO in [(10, 5, 0, 0), (10, 5, 1, 0), (10, 5, 0, 1)]:
        proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, phase_overlap=PO, mag_overlap=MO, compute_log_prob=True)
        g = run_gpu(proc, t, y, err, freqs)
        # reference
        t32, y01, _ = prep(t, y)
        m0 = np.floor(y01 * MB).astype(int)
        fracs_kernel = np.array([np.mean(m0 == i) for i in range(MB)])  # as compute_mag_bin_fracs (m0==MB excluded)
        fracs_correct = np.array([np.mean(np.minimum(m0, MB - 1) == i) for i in range(MB)])
        Hb = hist_ref(t, y, freqs, PB, MB, PO, MO, clip=False)
        Hb_flat = np.zeros(len(freqs) * PB * MB + 1)
        for i in range(len(freqs)):
            for n in range(PB):
                for mm in range(MB + 1):
                    Hb_flat[i * PB * MB + n * MB + mm] += Hb[i, n, mm]
        Hk = Hb_flat[:-1].reshape(len(freqs), PB, MB)
        Hc = hist_ref(t, y, freqs, PB, MB, PO, MO, clip=True).astype(float)
        def lp(H, fr):
            Nphi = H.sum(axis=2, keepdims=True)
            Nexp = Nphi * fr[None, None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(Nexp >= 1e-9, H * np.log(np.where(Nexp > 0, Nexp, 1)) - Nexp - gammaln(H + 1), 0.0)
            return term.sum(axis=(1, 2)) / (PO + 1)
        r_k = lp(Hk, fracs_kernel); r_c = lp(Hc, fracs_correct)
        print("log_prob PB=%d MB=%d PO=%d MO=%d: max|gpu-ref(kernel-emul)|=%.2e max|gpu-ref(correct)|=%.2e; argmin gpu=%d (f=%.3f) ref=%d; fracs sum kernel=%.4f; gpu min/max=%.1f/%.1f"
              % (PB, MB, PO, MO, np.max(np.abs(g - r_k)), np.max(np.abs(g - r_c)), np.argmin(g), freqs[np.argmin(g)], np.argmin(r_c), fracs_kernel.sum(), g.min(), g.max()))
        if MO > 0:
            # with mag_overlap the counts are multi-counted but Nexp uses un-overlapped fracs: sum over m of Nexp vs sum of N per phase bin
            print("      MO>0: sum_m Nexp / sum_m N per phase bin = %.3f (should be 1 for a calibrated Poisson model)" % (float((Hc.sum(axis=2, keepdims=True) * fracs_correct).sum()) / Hc.sum()))
    # balanced magbins
    for N, MB in [(400, 5), (400, 10), (12, 5), (40, 5)]:
        t, y, err = make(N, 20.0, 1.3, 5)
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=MB, balanced_magbins=True)
        try:
            ce_g, bins, mem = run_gpu_with_bins(proc, t, y, err, freqs)
            print("balanced N=%d MB=%d: mag_bwf=%s sum=%.3f; ce finite=%s min=%.3f; argmin f=%.3f" % (N, MB, np.array2string(mem.mag_bwf, precision=3), mem.mag_bwf.sum(), np.all(np.isfinite(ce_g)), np.nanmin(ce_g), freqs[np.nanargmin(ce_g)]))
        except Exception as e:
            print("balanced N=%d MB=%d: EXC %s %s" % (N, MB, type(e).__name__, str(e)[:100]))
    # duplicated y values with balanced bins
    t, y, err = make(40, 20.0, 1.3, 5); y = np.round(y, 0)
    proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, balanced_magbins=True)
    ce_g, bins, mem = run_gpu_with_bins(proc, t, y, err, freqs)
    print("balanced, quantized y (many duplicates): mag_bwf=%s  ce finite=%s  n(-inf)=%d" % (np.array2string(mem.mag_bwf, precision=3), np.all(np.isfinite(ce_g)), np.isinf(ce_g).sum()))

if section in ('all', 'edge'):
    print("=== (4) edge cases ===")
    r = np.random.RandomState(0)
    t = np.sort(r.rand(60) * 20); freqs = np.linspace(0.1, 3.0, 50)
    def attempt(label, data, ctor=dict(), **kw):
        try:
            proc = ConditionalEntropyAsyncProcess(**ctor)
            res = proc.run(data, freqs=freqs, **kw); proc.finish()
            p = np.copy(res[0][1])
            print("%-45s -> finite=%s nan=%d min=%.3g max=%.3g" % (label, np.all(np.isfinite(p)), np.isnan(p).sum(), np.nanmin(p), np.nanmax(p)))
        except Exception as e:
            print("%-45s -> EXC %s: %s" % (label, type(e).__name__, str(e)[:160]))
    attempt("float32 freqs (standard)", [(t, r.randn(60), np.ones(60))], freqs=None) if False else None
    for ctor in [dict(), dict(use_fast=True), dict(weighted=True)]:
        try:
            proc = ConditionalEntropyAsyncProcess(**ctor)
            res = proc.run([(t, r.randn(60), np.ones(60))], freqs=freqs.astype(np.float32)); proc.finish()
            print("float32 freqs %-20s -> OK" % ctor)
        except Exception as e:
            print("float32 freqs %-20s -> EXC %s: %s" % (ctor, type(e).__name__, str(e)[:160]))
    try:
        proc = ConditionalEntropyAsyncProcess()
        res = proc.run([(t, r.randn(60), np.ones(60))], freqs=list(freqs)); proc.finish(); print("python list freqs -> OK")
    except Exception as e:
        print("python list freqs -> EXC %s: %s" % (type(e).__name__, str(e)[:160]))
    attempt("N=3 < mag_bins=5 (standard)", [(t[:3], r.randn(3), np.ones(3))])
    attempt("N=3 < mag_bins=5 (fast)", [(t[:3], r.randn(3), np.ones(3))], ctor=dict(use_fast=True))
    attempt("N=1 (standard)", [(t[:1], r.randn(1), np.ones(1))])
    yn = r.randn(60); yn[5] = np.nan
    attempt("NaN in y (standard)", [(t, yn, np.ones(60))])
    e0 = np.ones(60); e0[5] = 0.0
    attempt("dy=0 at one point (weighted)", [(t, r.randn(60), e0)], ctor=dict(weighted=True))
    attempt("dy=0 everywhere (weighted)", [(t, r.randn(60), np.zeros(60))], ctor=dict(weighted=True))
    attempt("huge nbins standard (PB=200,MB=100)", [(t, r.randn(60), np.ones(60))], ctor=dict(phase_bins=200, mag_bins=100))
    attempt("huge nbins fast (PB=200,MB=100)", [(t, r.randn(60), np.ones(60))], ctor=dict(phase_bins=200, mag_bins=100, use_fast=True))
    attempt("huge nbins fast shmem_lc=False", [(t, r.randn(60), np.ones(60))], ctor=dict(phase_bins=200, mag_bins=100, use_fast=True), shmem_lc=False)
    attempt("fast, ndata=6000 (data won't fit shmem)", [(np.sort(r.rand(6000) * 20), r.randn(6000), np.ones(6000))], ctor=dict(use_fast=True))
    # set_data=False double accumulation
    proc = ConditionalEntropyAsyncProcess()
    d = [(t, r.randn(60), np.ones(60))]
    mems = proc.allocate(normalize_light_curves(d), freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    r1 = np.copy(proc.run(d, memory=mems, freqs=[freqs])[0][1]); proc.finish()
    r2 = np.copy(proc.run(d, memory=mems, freqs=[freqs], set_data=False)[0][1]); proc.finish()
    r3 = np.copy(proc.run(d, memory=mems, freqs=[freqs], set_data=True)[0][1]); proc.finish()
    print("set_data=False rerun: max|r2-r1|=%.3e (bins sum after 2 runs=%d, N=60) ; set_data=True rerun max|r3-r1|=%.3e" % (np.max(np.abs(r2 - r1)), mems[0].bins_g.get().reshape(50, -1)[0].sum(), np.max(np.abs(r3 - r1))))
    # BJD time shift
    tb, yb, eb = make(300, 20.0, 1.3, 6)
    p0 = run_gpu(ConditionalEntropyAsyncProcess(), tb, yb, eb, freqs); p1 = run_gpu(ConditionalEntropyAsyncProcess(), tb + 2455000.0, yb, eb, freqs)
    print("BJD shift 2455000: max|p1-p0|=%.3e" % np.max(np.abs(p1 - p0)))
    # large t*f float32 precision
    for baseline, fmax in [(3650, 20), (3650, 50)]:
        tt, yy, ee = make(500, float(baseline), fmax * 0.7, 9)
        fr = np.linspace(fmax * 0.69, fmax * 0.71, 2000)
        p32 = run_gpu(ConditionalEntropyAsyncProcess(), tt, yy, ee, fr); p64 = run_gpu(ConditionalEntropyAsyncProcess(use_double=True), tt, yy, ee, fr)
        print("baseline=%d fmax=%d: float32 vs double: max|d|=%.3e argmin32=%d argmin64=%d min32=%.4f min64=%.4f" % (baseline, fmax, np.max(np.abs(p32 - p64)), np.argmin(p32), np.argmin(p64), p32.min(), p64.min()))

if section == 'constant':
    print("=== constant y (yscale=0) -- run last, may kill the context ===")
    r = np.random.RandomState(0); t = np.sort(r.rand(60) * 20); freqs = np.linspace(0.1, 3.0, 50)
    import numpy
    with np.errstate(all='ignore'):
        print("np.floor(nan*5).astype(uint32) =", np.floor(np.array([np.nan], dtype=np.float32) * 5).astype(np.uint32))
    for ctor in [dict(), dict(use_fast=True), dict(weighted=True)]:
        try:
            proc = ConditionalEntropyAsyncProcess(**ctor)
            res = proc.run([(t, np.ones(60) * 5.0, np.ones(60))], freqs=freqs); proc.finish()
            p = np.copy(res[0][1]); print("constant y %-20s -> finite=%s nan=%d p[:3]=%s" % (ctor, np.all(np.isfinite(p)), np.isnan(p).sum(), p[:3]))
        except Exception as e:
            print("constant y %-20s -> EXC %s: %s" % (ctor, type(e).__name__, str(e)[:200]))

if section == 'align':
    print("=== double + (nmag even, nphase odd): shared-memory alignment offset ===")
    r = np.random.RandomState(0); t = np.sort(r.rand(200) * 20); y = r.randn(200); freqs = np.linspace(0.1, 3.0, 64)
    for PB, MB in [(5, 4), (10, 5), (7, 6)]:
        rr = (MB * PB + PB) * 4 % 8
        for shl in [True, False]:
            proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, use_double=True, use_fast=True)
            g = run_gpu(proc, t, y, np.ones(200), freqs, shmem_lc=shl)
            ref = ce_from_hist(hist_ref(t, y, freqs, PB, MB, clip=True), MB)
            print("PB=%d MB=%d kernel r(bytes)=%d -> element offset %d (=%d bytes) shmem_lc=%s: max|gpu-ref|=%.2e" % (PB, MB, rr, rr, rr * 4, shl, np.max(np.abs(g - ref))))

if section == 'timing':
    print("=== relative timings (shared 4090, noisy, min of 5) ===")
    freqs = np.linspace(0.1, 20.0, 50000)
    for nd in [200, 1000, 3000]:
        t, y, err = make(nd, 20.0, 1.3, 7)
        res = {}
        for label, ctor, rkw in [('standard', dict(), dict()), ('fast shmem', dict(use_fast=True), dict(shmem_lc=True)),
                                 ('fast noshmem', dict(use_fast=True), dict(shmem_lc=False)), ('weighted', dict(weighted=True), dict()),
                                 ('standard double', dict(use_double=True), dict())]:
            proc = ConditionalEntropyAsyncProcess(**ctor)
            run_gpu(proc, t, y, err, freqs[:100], **rkw)
            best = 1e9
            for k in range(5):
                t0 = time.perf_counter(); run_gpu(proc, t, y, err, freqs, **rkw); best = min(best, time.perf_counter() - t0)
            res[label] = best
        print("ndata=%d nf=%d: " % (nd, len(freqs)) + "  ".join("%s=%.4fs" % kv for kv in res.items()))
