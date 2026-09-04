import numpy as np
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls, eebls_gpu_fast, single_bls
from sparse_exp import ref_sparse, make_lc

kern = compile_sparse_bls(block_size=64)
kern_s = compile_sparse_bls(block_size=64, use_simple=True)

print("=== A. float32 fold at long baseline / high frequency: peak recovery vs float64 reference")
for base, f_inj, q_inj, N in ((3650.0, 4.7, 0.01, 300), (3650.0, 2.3, 0.02, 200), (1000.0, 4.7, 0.01, 300), (365.0, 4.7, 0.01, 300)):
    hits = {'cpu': 0, 'gpu': 0, 'ref64': 0}
    dP = []
    for seed in range(8):
        t, y, dy = make_lc(N, 'uniform', seed=seed, baseline=base, q=q_inj, f=f_inj, depth_sig=10.0)
        df = q_inj / base / 4
        freqs = f_inj + df * np.arange(-40, 41)
        p64, _, _, _ = ref_sparse(t, y, dy, freqs, fold='f64')
        pc, _ = sparse_bls_cpu(t, y, dy, freqs)
        pg, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
        for k, p in (('cpu', pc), ('gpu', pg), ('ref64', p64)):
            hits[k] += abs(freqs[np.argmax(p)] - f_inj) < 2 * df
        dP.append(np.abs(pc - p64).max() / p64.max())
    print("  baseline %5.0fd f=%.1f q=%.2f N=%d  ulp(phase)~%.1e: peak within 2 grid steps: cpu %d/8 gpu %d/8 ref64 %d/8 | median max-rel |cpu-ref64| over grid %.2e (max %.2e)"
          % (base, f_inj, q_inj, N, np.spacing(np.float32(base * f_inj)), hits['cpu'], hits['gpu'], hits['ref64'], np.median(dP), np.max(dP)))

print("\n=== B. one point with a small error bar (weight ratio R): power > 1 and centering as the fix")
r = np.random.RandomState(3)
N = 200
t = np.sort(365 * r.rand(N)); y0 = 12.0 + 0.01 * r.randn(N); dy0 = 0.01 * np.ones(N)
fq = np.linspace(0.5, 1.5, 201)
for R in (1e2, 1e3, 1e4, 1e6):
    dy = dy0.copy(); dy[17] = 0.01 / np.sqrt(R)
    wk = (1 / dy[17] ** 2) / np.sum(1 / dy ** 2)
    pc, _ = sparse_bls_cpu(t, y0, dy, fq)
    pg, _ = sparse_bls_gpu(t, y0, dy, fq, kernel=kern)
    pgs, _ = sparse_bls_gpu(t, y0, dy, fq, kernel=kern_s, use_simple=True)
    pf = eebls_gpu_fast(t, y0, dy, fq)
    ybar = np.sum(y0 / dy ** 2) / np.sum(1 / dy ** 2)
    pgc, _ = sparse_bls_gpu(t, y0 - ybar, dy, fq, kernel=kern)
    pref, _, _, _ = ref_sparse(t, y0, dy, fq)
    print("  R=%.0e (w_k=%.6f): max power ref %.4f cpu %.4f gpu-full %.4f gpu-simple %.4f fast %.4f | gpu-full with y pre-centered %.4f | gpu-full: %d/201 freqs with power>1"
          % (R, wk, pref.max(), pc.max(), pg.max(), pgs.max(), pf.max(), pgc.max(), (pg > 1).sum()))

print("\n=== C. centering as the fix for the uncentered-flux precision loss (exact-phase data, N=500, mag 12, 5 mmag)")
r = np.random.RandomState(5)
ks = np.sort(r.choice(np.arange(1, 1024 * 40), 500, replace=False)); t = ks / 1024.0; f = 1.25
ph = (t * f) % 1.0
y = 12.0 - 5e-3 * ((ph > 0.3) & (ph < 0.33)) + 1e-3 * r.randn(500); dy = 1e-3 * (0.7 + 0.6 * r.rand(500))
fr = np.array([f])
pref, _, _, _ = ref_sparse(t, y, dy, fr)
w = dy ** -2; ybar = np.sum(w * y) / np.sum(w)
pg, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern)
pgc, _ = sparse_bls_gpu(t, y - ybar, dy, fr, kernel=kern)
pc, _ = sparse_bls_cpu(t, y, dy, fr)
pcc, _ = sparse_bls_cpu(t, y - ybar, dy, fr)
print("  ref %.6f | gpu-full raw %.6f (rel %.1e) centered %.6f (rel %.1e) | cpu raw %.6f (rel %.1e) centered %.6f (rel %.1e)"
      % (pref[0], pg[0], abs(pg[0] - pref[0]) / pref[0], pgc[0], abs(pgc[0] - pref[0]) / pref[0],
         pc[0], abs(pc[0] - pref[0]) / pref[0], pcc[0], abs(pcc[0] - pref[0]) / pref[0]))

print("\n=== D. pure noise, N=100: how often is the per-frequency best box a single point? (paper recommends a >=3-point floor)")
fr = np.random.RandomState(9).uniform(0.1, 5, 200)
n1 = 0; ntot = 0; pk1 = []; pk3 = []
for seed in range(10):
    r = np.random.RandomState(100 + seed)
    t = np.sort(365 * r.rand(100)); y = 0.01 * r.randn(100); dy = 0.01 * np.ones(100)
    p, q, ph, npts = ref_sparse(t, y, dy, fr)
    n1 += (npts == 1).sum(); ntot += len(fr)
    pk1.append(p.max())
    # same reference but excluding 1- and 2-point boxes (recompute crudely: mask by npts of the argmax isn't enough; use qmin as proxy? no -- do explicit)
print("  best box is a single point at %d/%d (%.0f%%) of trial frequencies; mean periodogram max %.4f" % (n1, ntot, 100.0 * n1 / ntot, np.mean(pk1)))
