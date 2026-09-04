"""Edge cases + the simple-kernel all-weight instability (PR #65 class)."""
import numpy as np, traceback
from cuvarbase.bls import (sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls,
                           eebls_transit, eebls_gpu_fast, eebls_gpu_batch)

kern = compile_sparse_bls(block_size=64)
kern_s = compile_sparse_bls(block_size=64, use_simple=True)

# ---- 1. single-site data, f = 1/day: every point at phase < 0.25 so the
# box holding ALL points is a candidate (i=0, j=N). Full kernel guards
# with 1e-4 complement; simple kernel's 1e-9 compiles to `W > 1.f`.
r = np.random.RandomState(21)
t = np.concatenate([n + 0.25 * np.sort(r.rand(8)) for n in range(40)])
y = 12.0 + 0.01 * r.randn(len(t))
dy = 0.01 * np.ones_like(y)
freqs = np.linspace(0.98, 1.02, 401)
print("single-site N=%d" % len(t))
pc, _ = sparse_bls_cpu(t, y, dy, freqs)
print("  cpu        max=%.4g finite=%s" % (np.nanmax(pc), np.all(np.isfinite(pc))))
for rep in range(3):
    pf, sf = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
    ps, ss = sparse_bls_gpu(t, y, dy, freqs, kernel=kern_s, use_simple=True)
    print("  rep%d full  max=%.4g finite=%s n>0.05=%d | simple max=%.4g finite=%s n>0.05=%d n_nonfinite=%d"
          % (rep, np.nanmax(pf), np.all(np.isfinite(pf)), (pf > 0.05).sum(),
             np.nanmax(ps[np.isfinite(ps)]) if np.any(np.isfinite(ps)) else np.nan,
             np.all(np.isfinite(ps)), (ps[np.isfinite(ps)] > 0.05).sum(), (~np.isfinite(ps)).sum()))
k = int(np.argmax(np.where(np.isfinite(ps), ps, -1)))
print("  simple worst freq %.5f: power %.4g q=%.4f phi=%.4f (full gives %.4g)" % (freqs[k], ps[k], ss[k][0], ss[k][1], pf[k]))
# reachable through eebls_transit?
fr, pw, so = eebls_transit(t, y, dy, freqs=freqs, use_simple=True)
print("  eebls_transit(use_simple=True): max=%.4g finite=%s (default Keplerian q bounds)" % (np.nanmax(pw), np.all(np.isfinite(pw))))
fr, pw, so = eebls_transit(t, y, dy, freqs=freqs, use_simple=True, qmax_fac=20.)
print("  eebls_transit(use_simple=True, qmax_fac=20): max=%.4g finite=%s" % (np.nanmax(pw), np.all(np.isfinite(pw))))

# ---- 2. degenerate inputs
r = np.random.RandomState(0)
tt = np.sort(100 * r.rand(60)); yy = 1 + 0.01 * r.randn(60); dd = 0.01 * np.ones(60)
fq = np.linspace(0.5, 1.5, 11)
cases = {
    'empty': (np.array([]), np.array([]), np.array([])),
    'N=1': (tt[:1], yy[:1], dd[:1]),
    'N=2': (tt[:2], yy[:2], dd[:2]),
    'N=3': (tt[:3], yy[:3], dd[:3]),
    'one dy=0': (tt, yy, np.where(np.arange(60) == 7, 0.0, dd)),
    'one dy=inf': (tt, yy, np.where(np.arange(60) == 7, np.inf, dd)),
    'one NaN y': (tt, np.where(np.arange(60) == 7, np.nan, yy), dd),
    'one NaN t': (np.where(np.arange(60) == 7, np.nan, tt), yy, dd),
    'negative dy': (tt, yy, -dd),
    'constant y': (tt, np.ones(60), dd),
    'tiny dy (1e-3x) outlier point': (tt, yy, np.where(np.arange(60) == 7, 1e-3 * dd, dd)),
}
for name, (a, b, c) in cases.items():
    for label, fn in (('cpu', lambda: sparse_bls_cpu(a, b, c, fq)),
                      ('gpu', lambda: sparse_bls_gpu(a, b, c, fq, kernel=kern)),
                      ('gpu-simple', lambda: sparse_bls_gpu(a, b, c, fq, kernel=kern_s, use_simple=True)),
                      ('fast', lambda: eebls_gpu_fast(a, b, c, fq)),
                      ('batch', lambda: eebls_gpu_batch([(a, b, c)], fq)[0])):
        try:
            p = fn()
            if isinstance(p, tuple):
                p = p[0]
            p = np.asarray(p)
            print("  %-30s %-10s -> max=%s nan=%d inf=%d" % (name, label, ('%.4g' % np.nanmax(p)) if p.size else 'empty', np.isnan(p).sum(), np.isinf(p).sum()))
        except Exception as e:
            print("  %-30s %-10s -> %s: %s" % (name, label, type(e).__name__, str(e)[:90].replace('\n', ' ')))

# ---- 3. dy=0 semantics check vs a finite floor
print("\nnoverlap=0 on batch:")
try:
    p0 = eebls_gpu_batch([(tt, yy, dd)], fq, noverlap=0)[0]
    print("  noverlap=0 ->", p0)
except Exception as e:
    print("  noverlap=0 ->", type(e).__name__, e)
try:
    p0 = eebls_gpu_fast(tt, yy, dd, fq, noverlap=0)
    print("  fast noverlap=0 ->", p0[:3])
except Exception as e:
    print("  fast noverlap=0 ->", type(e).__name__, e)

# ---- 4. sparse kernel shared-memory limit: what ndata makes the launch fail?
for N in (500, 1000, 2000, 2500, 3000, 4000):
    tN = np.sort(365 * r.rand(N)); yN = 1 + 0.01 * r.randn(N); dN = 0.01 * np.ones(N)
    try:
        p, _ = sparse_bls_gpu(tN, yN, dN, fq[:3], kernel=kern)
        print("  sparse gpu N=%d ok max=%.4g" % (N, p.max()))
    except Exception as e:
        print("  sparse gpu N=%d -> %s: %s" % (N, type(e).__name__, str(e)[:100]))
