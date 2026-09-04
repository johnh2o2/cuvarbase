"""Exp 5: failure modes / edge cases on the fast and standard paths."""
import sys, traceback
sys.path.insert(0, '/workspace/scratch')
import numpy as np
from blsref import make_data, exact_bls, ref_fast, exact_best_at_freq
from cuvarbase.bls import (eebls_gpu_fast, eebls_gpu, single_bls, BLSMemory,
                           eebls_gpu_custom, eebls_transit)


def hdr(s):
    print("\n--- %s ---" % s)


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        print("  EXCEPTION %s: %s" % (type(e).__name__, str(e)[:200]))
        return None


hdr("1. ndata < nbins (50 pts, qmin=0.01 -> 100 bins)")
t, y, dy = make_data(ndata=50, baseline=10., freq=0.7, q=0.1, phi0=0.2, snr=8, seed=1)
freqs = np.linspace(0.5, 1.0, 300)
g = eebls_gpu_fast(t, y, dy, freqs)
r = ref_fast(t, y, dy, freqs, f32fold=True)
print("  gpu vs ref maxabs=%.2e; max power=%.3f at f=%.4f (inj 0.7); exact@true=%.3f"
      % (np.abs(g - r).max(), g.max(), freqs[np.argmax(g)], exact_bls(t, y, dy, 0.7, 0.1, 0.2)))

hdr("2. single transit event in the baseline (T=30d, P=20d, q=0.01)")
t, y, dy = make_data(ndata=4000, baseline=30., freq=0.05, q=0.01, phi0=0.6, snr=25, seed=2, cadence=30. / 4000)
freqs = np.linspace(0.04, 0.4, 3000)
g = eebls_gpu_fast(t, y, dy, freqs, qmin=0.005, qmax=0.1)
ex = exact_bls(t, y, dy, 0.05, 0.01, 0.6)
i0 = np.argmin(np.abs(freqs - 0.05))
print("  exact@true=%.4f fast@f0=%.4f max=%.4f at f=%.4f; #freqs with power >= 0.9*max: %d"
      % (ex, g[i0], g.max(), freqs[np.argmax(g)], (g >= 0.9 * g.max()).sum()))

hdr("3. duplicated times (every point twice) vs unique")
t, y, dy = make_data(ndata=1000, baseline=30., freq=1.3, q=0.05, phi0=0.3, snr=12, seed=3)
freqs = np.linspace(1.0, 1.6, 500)
g1 = eebls_gpu_fast(t, y, dy, freqs)
g2 = eebls_gpu_fast(np.concatenate((t, t)), np.concatenate((y, y)), np.concatenate((dy, dy)), freqs)
print("  maxabs diff = %.2e (expect ~float32 roundoff)" % np.abs(g1 - g2).max())
p1, s1 = eebls_gpu(t, y, dy, freqs[::5])
p2, s2 = eebls_gpu(np.concatenate((t, t)), np.concatenate((y, y)), np.concatenate((dy, dy)), freqs[::5])
print("  eebls_gpu maxabs diff = %.2e" % np.abs(p1 - p2).max())

hdr("4. unsorted t (shuffled) vs sorted")
perm = np.random.RandomState(0).permutation(len(t))
g3 = eebls_gpu_fast(t[perm], y[perm], dy[perm], freqs)
print("  maxabs diff = %.2e" % np.abs(g1 - g3).max())
p3, s3 = eebls_gpu(t[perm], y[perm], dy[perm], freqs[::5])
print("  eebls_gpu maxabs diff = %.2e; sols equal=%s" % (np.abs(p1 - p3).max(), np.allclose(s1, s3)))

hdr("5. dy scaled x10 (weights sum differently): chi2ratio invariant, snr scales")
g4 = eebls_gpu_fast(t, y, dy * 10, freqs)
print("  chi2ratio maxabs diff = %.2e" % np.abs(g1 - g4).max())
s_a = eebls_gpu_fast(t, y, dy, freqs, convention='snr')
s_b = eebls_gpu_fast(t, y, dy * 10, freqs, convention='snr')
print("  snr ratio (dy) / (10 dy) = %.4f (expect 10)" % (s_a.max() / s_b.max()))
g5 = eebls_gpu_fast(t, y, np.ones_like(dy) * 0.3, freqs)
print("  uniform dy (any scale) vs sigma: maxabs diff = %.2e" % np.abs(g1 - g5).max())

hdr("6. dy contains a zero / negative / NaN")
dz = dy.copy(); dz[10] = 0.0
g6 = safe(eebls_gpu_fast, t, y, dz, freqs)
if g6 is not None:
    print("  dy[10]=0: any NaN=%s, max=%.3g, min=%.3g" % (np.isnan(g6).any(), np.nanmax(g6), np.nanmin(g6)))
dn = dy.copy(); dn[10] = -0.01
g6b = safe(eebls_gpu_fast, t, y, dn, freqs)
if g6b is not None:
    print("  dy[10]=-0.01: maxabs diff vs clean = %.2e (negative dy silently accepted)" % np.abs(g6b - g1).max())
yn = y.copy(); yn[10] = np.nan
g6c = safe(eebls_gpu_fast, t, yn, dy, freqs)
if g6c is not None:
    print("  y[10]=NaN: any NaN=%s all NaN=%s" % (np.isnan(g6c).any(), np.isnan(g6c).all()))
dz2 = dy.copy(); dz2[10] = 1e-6
g6d = safe(eebls_gpu_fast, t, y, dz2, freqs)
if g6d is not None:
    print("  dy[10]=1e-6 (w frac %.4f): median power=%.3f max=%.3f (clean median %.3f)"
          % ((dz2 ** -2)[10] / (dz2 ** -2).sum(), np.median(g6d), g6d.max(), np.median(g1)))

hdr("7. injected q much shorter than the finest bin (q=0.002, qmin=0.01)")
t, y, dy = make_data(ndata=20000, baseline=30., freq=1.3, q=0.002, phi0=0.3, snr=25, seed=4, cadence=30. / 20000)
freqs = np.linspace(1.25, 1.35, 800)
ex = exact_bls(t, y, dy, 1.3, 0.002, 0.3)
i0 = np.argmin(np.abs(freqs - 1.3))
for qmin in [0.01, 0.002]:
    g = eebls_gpu_fast(t, y, dy, freqs, qmin=qmin)
    print("  qmin=%g: exact@true=%.4f fast@f0=%.4f (%.0f%%) argmax f=%.4f" % (qmin, ex, g[i0], 100 * g[i0] / ex, freqs[np.argmax(g)]))

hdr("8. transit wrapping the phase boundary (phi0=0.98, q=0.05) vs phi0=0.5")
for phi0 in [0.5, 0.98, 0.0]:
    t, y, dy = make_data(ndata=3000, baseline=30., freq=1.0, q=0.05, phi0=phi0, snr=15, seed=5)
    freqs = np.linspace(0.95, 1.05, 801)
    i0 = np.argmin(np.abs(freqs - 1.0))
    ex = exact_bls(t, y, dy, 1.0, 0.05, phi0)
    g = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
    p, s = eebls_gpu(t, y, dy, freqs[::4], qmin=0.01, qmax=0.1)
    j0 = np.argmin(np.abs(freqs[::4] - 1.0))
    print("  phi0=%.2f: exact=%.4f fast@f0=%.4f (%.1f%%) slow@f0=%.4f (%.1f%%) slow sol=(q=%.3f phi=%.3f)"
          % (phi0, ex, g[i0], 100 * g[i0] / ex, p[j0], 100 * p[j0] / ex, s[j0][0], s[j0][1]))

hdr("9. constant y (yy=0) and all-identical t")
t, y, dy = make_data(ndata=500, baseline=30., freq=1.0, q=0.05, phi0=0.5, snr=0, seed=6)
yc = np.ones_like(y)
g = safe(eebls_gpu_fast, t, yc, dy, freqs[:50])
if g is not None:
    print("  constant y: any NaN=%s any inf=%s sample=%s" % (np.isnan(g).any(), np.isinf(g).any(), g[:3]))
p, s = safe(eebls_gpu, t, yc, dy, freqs[:50]) or (None, None)
if p is not None:
    print("  eebls_gpu constant y: any NaN=%s sample=%s" % (np.isnan(p).any(), p[:3]))
tc = np.ones_like(t) * 5.0
g = safe(eebls_gpu_fast, tc, y, dy, freqs[:50])
if g is not None:
    print("  identical t: max=%.3g any NaN=%s" % (np.nanmax(g), np.isnan(g).any()))

hdr("10. qmin > qmax and qmax > 1 on the fast path (no validation?)")
t, y, dy = make_data(ndata=1000, baseline=30., freq=1.3, q=0.05, phi0=0.3, snr=12, seed=3)
freqs = np.linspace(1.0, 1.6, 200)
g = safe(eebls_gpu_fast, t, y, dy, freqs, qmin=0.2, qmax=0.1)
if g is not None:
    print("  qmin=0.2 > qmax=0.1: max power=%.3g (all zero=%s)" % (g.max(), np.all(g == 0)))
g = safe(eebls_gpu_fast, t, y, dy, freqs, qmin=0.01, qmax=2.0)
if g is not None:
    print("  qmax=2.0 (nbins0=0): max=%.3g any NaN=%s; vs qmax=0.5 maxabs diff=%.2e" % (g.max(), np.isnan(g).any(), np.abs(g - eebls_gpu_fast(t, y, dy, freqs)).max()))
g = safe(eebls_gpu_fast, t, y, dy, freqs, qmin=0.01, qmax=0.9)
if g is not None:
    print("  qmax=0.9 (nbins0=1): max=%.3g; q tested up to m<100 -> includes q>0.5 boxes? power vs qmax=0.5 maxabs diff=%.2e" % (g.max(), np.abs(g - eebls_gpu_fast(t, y, dy, freqs)).max()))
p = safe(eebls_gpu, t, y, dy, freqs, qmin=0.2, qmax=0.1)
if p is not None:
    print("  eebls_gpu qmin>qmax: max=%.3g" % p[0].max())

hdr("11. memory reuse with fewer freqs than allocated: returned length")
mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5, freqs=freqs, transfer=True)
g = eebls_gpu_fast(t, y, dy, freqs[:50], memory=mem)
print("  allocated %d freqs, asked %d, returned len=%d" % (len(freqs), 50, len(g)))
gref = eebls_gpu_fast(t, y, dy, freqs[:50])
print("  first-50 match fresh run: %s (maxabs %.2e)" % (np.allclose(g[:50], gref, atol=1e-6), np.abs(g[:50] - gref).max()))
g = safe(eebls_gpu_fast, t, y, dy, np.linspace(1, 2, 400), memory=mem)
if g is not None:
    print("  asked 400 > allocated 200: returned len=%d, max=%.3g" % (len(g), np.nanmax(g)))

hdr("12. large ndata on the fast path (1e6 points)")
t, y, dy = make_data(ndata=1000000, baseline=27., freq=2.1, q=0.04, phi0=0.3, snr=40, seed=9, cadence=27. / 1e6)
freqs = np.linspace(2.0, 2.2, 400)
g = eebls_gpu_fast(t, y, dy, freqs)
ex = exact_bls(t, y, dy, 2.1, 0.04, 0.3)
print("  N=%d: max=%.4f at f=%.4f; exact@true=%.4f; ref(f32 fold)@f0=%.4f"
      % (len(t), g.max(), freqs[np.argmax(g)], ex, ref_fast(t, y, dy, freqs[[np.argmin(np.abs(freqs - 2.1))]], f32fold=True)[0]))

hdr("13. freqs with f=0 and negative f")
t, y, dy = make_data(ndata=1000, baseline=30., freq=1.3, q=0.05, phi0=0.3, snr=12, seed=3)
g = safe(eebls_gpu_fast, t, y, dy, np.array([0.0, -1.3, 1.3]))
if g is not None:
    print("  powers for f=[0,-1.3,1.3]: %s" % g)

hdr("14. eebls_transit with BJD times (default path) recovers same as t0=0")
t, y, dy = make_data(ndata=800, baseline=100., freq=0.3, q=0.03, phi0=0.4, snr=15, seed=12)
fa, pa, sa = eebls_transit(t, y, dy, fmin=0.1, fmax=1.0)
fb, pb, sbb = eebls_transit(t + 2457000.0, y, dy, fmin=0.1, fmax=1.0)
print("  nfreqs=%d maxabs diff=%.2e argmax f a=%.4f b=%.4f; phase sols differ by (mod 1) max=%.2e"
      % (len(fa), np.abs(pa - pb).max(), fa[np.argmax(pa)], fb[np.argmax(pb)],
         np.max(np.abs(((np.array([s[1] for s in sa]) - np.array([s[1] for s in sbb]) - 2457000.0 * fa) + 0.5) % 1.0 - 0.5))))
sb1 = np.array([single_bls(t + 2457000.0, y, dy, f, qq, pp) for f, (qq, pp) in zip(fb, sbb)])
rel = np.abs(sb1 - pb) / np.maximum(pb, 1e-3)
print("  BJD: single_bls(sol) vs power max rel=%.3e #>2%%=%d/%d" % (rel.max(), (rel > 0.02).sum(), len(fb)))
