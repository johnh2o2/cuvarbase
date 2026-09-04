"""Drill-down: (1) N=300 discrepancy between cpu/gpu-full and the reference
-> phase ties in the float32 fold and sort-order dependence; (2) exact-phase
data (no fold rounding) to isolate arithmetic precision: centered vs
normalized flux vs raw counts; (3) simple-kernel all-weight probe over seeds."""
import numpy as np
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls
from cuvarbase.utils import subtract_epoch
from sparse_exp import ref_sparse, make_lc

kern = compile_sparse_bls(block_size=64)
kern_s = compile_sparse_bls(block_size=64, use_simple=True)

print("=== 1. N=300 uniform centered: where do cpu/gpu-full differ from the stable-sort reference?")
t, y, dy = make_lc(300, 'uniform', seed=300)
freqs = np.concatenate([np.linspace(1.29, 1.31, 33), np.random.RandomState(1).uniform(0.05, 5, 31)])
pref, _, _, _ = ref_sparse(t, y, dy, freqs)
pc, _ = sparse_bls_cpu(t, y, dy, freqs)
pg, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
pgs, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kern_s, use_simple=True)
t64, ep = subtract_epoch(t)
for k in np.argsort(-np.abs(pc - pref))[:5]:
    phi32 = (np.float32(t64) * np.float32(freqs[k])) % np.float32(1)
    u, cnt = np.unique(phi32, return_counts=True)
    nties = int((cnt > 1).sum())
    # reference with a different tie order (reverse-time stable sort)
    o = np.argsort(phi32[::-1], kind='stable')
    pr_rev, _, _, _ = ref_sparse(t[::-1], y[::-1], dy[::-1], freqs[k:k + 1])
    print("  f=%.5f t*f~%.0f ulp=%.1e  ref(stable)=%.6f ref(rev-order)=%.6f cpu=%.6f gpu=%.6f simple=%.6f  #tied phase groups=%d"
          % (freqs[k], t64.max() * freqs[k], np.spacing(np.float32(t64.max() * freqs[k])), pref[k], pr_rev[0], pc[k], pg[k], pgs[k], nties))

print("\n=== 1b. same data, reference folded in float64 vs float32: fraction of frequencies where the max differs by >1e-4 rel")
p64, _, _, _ = ref_sparse(t, y, dy, freqs, fold='f64')
print("  N=300 baseline 365d: %d/%d freqs differ (max rel %.2e); peak power f64=%.6f f32=%.6f"
      % ((np.abs(p64 - pref) > 1e-4 * pref.max()).sum(), len(freqs), np.abs(p64 - pref).max() / pref.max(), p64.max(), pref.max()))
for base, N in ((3650.0, 300), (3650.0, 100), (1000.0, 200)):
    tb, yb, db = make_lc(N, 'uniform', seed=7, baseline=base)
    fr = np.random.RandomState(2).uniform(0.5, 5, 64)
    a, _, _, _ = ref_sparse(tb, yb, db, fr, fold='f64')
    b, _, _, _ = ref_sparse(tb, yb, db, fr)
    c, _ = sparse_bls_cpu(tb, yb, db, fr)
    print("  N=%d baseline %.0fd: f32-fold vs f64-fold ref: %d/64 freqs differ >1e-4 rel, max rel %.2e | cpu vs f64 ref max rel %.2e"
          % (N, base, (np.abs(a - b) > 1e-4 * a.max()).sum(), np.abs(a - b).max() / a.max(), np.abs(c - a).max() / a.max()))

print("\n=== 2. exact-phase data (t = k/1024, f = 1.25 -> t*f exact in float32; no ties): arithmetic precision only")
r = np.random.RandomState(5)
for N in (100, 300, 500):
    ks = np.sort(r.choice(np.arange(1, 1024 * 40), N, replace=False))
    t = ks / 1024.0
    f = 1.25
    ph = (t * f) % 1.0
    for offset, depth, sig, tag in ((0.0, 5e-3, 1e-3, 'centered, 5 ppt'),
                                    (1.0, 5e-3, 1e-3, 'normflux, 5 ppt'),
                                    (1.0, 3e-4, 1e-4, 'normflux, 300 ppm'),
                                    (1e4, 3.0, 1.0, 'raw counts 1e4, 300 ppm'),
                                    (12.0, 5e-3, 1e-3, 'mag 12, 5 mmag')):
        y = offset - depth * ((ph > 0.3) & (ph < 0.33)) + sig * r.randn(N)
        dy = sig * (0.7 + 0.6 * r.rand(N))
        fr = np.array([f])
        pref, qref, phref, _ = ref_sparse(t, y, dy, fr)
        pc, sc = sparse_bls_cpu(t, y, dy, fr)
        pg, sg = sparse_bls_gpu(t, y, dy, fr, kernel=kern)
        pgs, sgs = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
        print("  N=%d %-26s ref=%.6f  cpu rel err %.1e  gpu-full %.1e  gpu-simple %.1e | q ref %.4f cpu %.4f gpu %.4f"
              % (N, tag, pref[0], abs(pc[0] - pref[0]) / pref[0], abs(pg[0] - pref[0]) / pref[0], abs(pgs[0] - pref[0]) / pref[0],
                 qref[0], sc[0][0], sg[0][0]))

print("\n=== 3. simple-kernel all-weight box probe (single-site data, f=1/d, qmax=0.5 default), 40 seeds")
bad_s, bad_f = 0, 0
worst = 0
for seed in range(40):
    r = np.random.RandomState(seed)
    nn = 20 + seed % 30
    t = np.concatenate([n + 0.3 * np.sort(r.rand(6)) for n in range(nn)])
    y = 12.0 + 0.01 * r.randn(len(t))
    dy = 0.01 * np.ones_like(y)
    fr = np.linspace(0.995, 1.005, 101)
    ps, ss = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
    pf, sf = sparse_bls_gpu(t, y, dy, fr, kernel=kern)
    pc, _ = sparse_bls_cpu(t, y, dy, fr)
    if not np.all(np.isfinite(ps)) or np.nanmax(ps) > 0.3:
        bad_s += 1
    if not np.all(np.isfinite(pf)) or np.nanmax(pf) > 0.3:
        bad_f += 1
    worst = max(worst, np.nanmax(np.abs(ps - pc)))
print("  simple: %d/40 seeds with non-finite or >0.3 power; full: %d/40; worst |simple - cpu| = %.3e" % (bad_s, bad_f, worst))
