"""Exp 5: failure modes / edge cases on the fast and standard paths."""
import sys, traceback
sys.path.insert(0, '/workspace/scratch/blsaudit')
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


t, y, dy = make_data(ndata=1000, baseline=30., freq=1.3, q=0.05, phi0=0.3, snr=12, seed=3)
freqs = np.linspace(1.0, 1.6, 200)
hdr("11. memory reuse with fewer freqs than allocated: returned length")
mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5, freqs=freqs, transfer=True)
g = safe(eebls_gpu_fast, t, y, dy, freqs[:50], memory=mem)
if g is not None:
    print("  allocated %d freqs, asked %d, returned len=%d" % (len(freqs), 50, len(g)))
mem2 = safe(BLSMemory.fromdata, t, y, dy, qmin=1e-2, qmax=0.5, freqs=freqs, transfer=True, max_nfreqs=1000)
g = safe(eebls_gpu_fast, t, y, dy, freqs[:50], memory=mem2) if mem2 is not None else None
if g is not None:
    print("  max_nfreqs=1000: allocated, asked 50, returned len=%d" % len(g))
mem3 = BLSMemory(len(t), 1000)
g = safe(eebls_gpu_fast, t, y, dy, freqs[:50], memory=mem3)
if g is not None:
    gref = eebls_gpu_fast(t, y, dy, freqs[:50])
    print("  BLSMemory(ndata, 1000) then 50 freqs: returned len=%d; first-50 match fresh: %s" % (len(g), np.allclose(g[:50], gref, atol=1e-6)))
    g = safe(eebls_gpu_fast, t, y, dy, freqs[:120], memory=mem3)
    if g is not None:
        print("  then 120 freqs with same memory: returned len=%d; tail beyond 120 = %s" % (len(g), g[120:125]))
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
