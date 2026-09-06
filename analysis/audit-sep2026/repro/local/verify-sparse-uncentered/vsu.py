"""ARCHIVED as run at 7d55ea2 (the pre-fix tree): this script no longer runs at
the tip -- the 'sparse_bls_simple' kernel it selects with use_simple=True was
removed in Phase 1 (398cd60, defect 20). Kept unedited as the audit record.

Independent verification of 'bls-sparse-uncentered':
A. default public path eebls_transit (auto sparse, N<500) on mag-12 data vs float64 reference
B. patched kernel (center sh_y in-kernel) and wrapper-level float64 centering as fixes
C. bit-neutrality of the patched kernel on already-centered input
"""
import os, sys, numpy as np
sys.path.insert(0, '/workspace/scratch')
import pycuda.autoprimaryctx  # noqa
from pycuda.compiler import SourceModule
from cuvarbase.bls import (eebls_transit, sparse_bls_gpu, sparse_bls_cpu, compile_sparse_bls,
                           q_transit, eebls_gpu_fast)
from cuvarbase.utils import _module_reader, find_kernel
from sparse_exp import ref_sparse

# ---------- patched kernels (scratch copy only; /workspace/cuvarbase untouched) ----------
def patched_kernel(use_simple=False, block_size=64):
    name = 'sparse_bls_simple' if use_simple else 'sparse_bls'
    src = _module_reader(find_kernel(name), cpp_defs=dict(BLOCK_SIZE=block_size))
    if use_simple:
        anchor = "        float ybar = sh_bls[0];\n        __syncthreads();\n"
        diff_old = "            float diff = sh_y[i] - ybar;\n            local_YY"
        diff_new = "            float diff = sh_y[i];\n            local_YY"
    else:
        anchor = "        float ybar = thread_results[0];\n        __syncthreads();\n"
        diff_old = "            float diff = sh_y[i] - ybar;\n            local_sum"
        diff_new = "            float diff = sh_y[i];\n            local_sum"
    assert src.count(anchor) == 1 and src.count(diff_old) == 1 and src.count("YW -= ybar * W;") == 2
    src = src.replace(anchor, anchor +
        "        for (unsigned int i = tid; i < ndata; i += blockDim.x) sh_y[i] -= ybar;\n"
        "        __syncthreads();\n")
    src = src.replace(diff_old, diff_new)
    src = src.replace("                YW -= ybar * W;\n", "")
    assert "ybar * W" not in src
    mod = SourceModule(src, options=['--use_fast_math'])
    return mod.get_function('sparse_bls_kernel_simple' if use_simple else 'sparse_bls_kernel')

kern = compile_sparse_bls(block_size=64)
kern_s = compile_sparse_bls(block_size=64, use_simple=True)
pk = patched_kernel(False)
pk_s = patched_kernel(True)

def wrapper_centered(t, y, dy, freqs, **kw):
    """Wrapper-level fix: center y in float64 before the float32 cast (as the dense path does at bls.py:1516-1518)."""
    y = np.asarray(y, dtype=np.float64); w = np.asarray(dy, dtype=np.float64) ** -2
    return sparse_bls_gpu(t, y - np.sum(w * y) / np.sum(w), dy, freqs, **kw)

def make(N, seed, base=365.0, ybar=12.0, depth=5e-3, sig=5e-3, f=1.37, q=0.02):
    r = np.random.RandomState(seed)
    t = np.sort(base * r.rand(N)); ph = (t * f) % 1
    y = ybar - depth * (ph < q) + sig * r.randn(N)
    dy = sig * (0.7 + 0.6 * r.rand(N))
    return t, y, dy

print("=== A. default path eebls_transit(t, y, dy) [auto -> sparse GPU], N=200, mag 12, 5 mmag noise, 5 mmag depth, 365 d")
print("    (reference: float64 set-based sparse BLS with the identical float32 fold and identical per-freq Keplerian q bounds)")
for seed in range(6):
    t, y, dy = make(200, seed)
    freqs = 1.37 + (0.02 / 365 / 4) * np.arange(-200, 201)
    freqs, p, sols = eebls_transit(t, y, dy, freqs=freqs)
    qv = q_transit(freqs)
    pref, _, _, _ = ref_sparse(t, y, dy, freqs, qmin=qv * 0.5, qmax=qv * 2.0)
    rel = (p - pref) / np.maximum(pref, 1e-12)
    ip, ir = np.argmax(p), np.argmax(pref)
    print("  seed %d nf=%5d: peak ref %.5f gpu %.5f (rel %+.1e) | argmax same: %s | over grid: max|rel| %.1e, mean rel %+.1e, frac |rel|>1e-3: %.2f, rank-1 stable: %s"
          % (seed, len(freqs), pref[ir], p[ip], (p[ir] - pref[ir]) / pref[ir], ip == ir,
             np.abs(rel).max(), rel.mean(), np.mean(np.abs(rel) > 1e-3), ip == ir))

print("\n=== A2. same, but pass the flux pre-centered (y - ybar) through the SAME default path")
for seed in range(3):
    t, y, dy = make(200, seed)
    w = dy ** -2; yc = y - np.sum(w * y) / np.sum(w)
    freqs = 1.37 + (0.02 / 365 / 4) * np.arange(-200, 201)
    freqs, p, _ = eebls_transit(t, yc, dy, freqs=freqs)
    qv = q_transit(freqs)
    pref, _, _, _ = ref_sparse(t, yc, dy, freqs, qmin=qv * 0.5, qmax=qv * 2.0)
    rel = (p - pref) / np.maximum(pref, 1e-12)
    print("  seed %d: max|rel| over grid %.1e, mean rel %+.1e" % (seed, np.abs(rel).max(), rel.mean()))

print("\n=== A3. heterogeneous precision on the default path: N=200 mag-12 ground data + ONE point 100x / 1000x more precise")
for R in (1e2, 1e4, 1e6):
    t, y, dy = make(200, 11); dy2 = dy.copy(); dy2[50] = dy[50] / np.sqrt(R)
    freqs = 1.37 + (0.02 / 365 / 4) * np.arange(-200, 201)
    freqs, p, _ = eebls_transit(t, y, dy2, freqs=freqs)
    qv = q_transit(freqs)
    pref, _, _, _ = ref_sparse(t, y, dy2, freqs, qmin=qv * 0.5, qmax=qv * 2.0)
    print("  R=%.0e: ref max %.4f  default-path max %.4f  n(power>1)=%d/%d  argmax same: %s"
          % (R, pref.max(), p.max(), (p > 1).sum(), len(p), np.argmax(p) == np.argmax(pref)))

print("\n=== B. fixes, exact-phase data (t=k/1024, f=1.25): rel err vs float64 reference")
r = np.random.RandomState(5)
for N in (100, 300, 500):
    ks = np.sort(r.choice(np.arange(1, 1024 * 40), N, replace=False)); t = ks / 1024.0; f = 1.25
    ph = (t * f) % 1.0
    for tag, off, depth, sig in (('mag 12, 5 mmag', 12.0, 5e-3, 1e-3), ('normflux 300ppm', 1.0, 3e-4, 1e-4), ('centered', 0.0, 5e-3, 1e-3)):
        y = off - depth * ((ph > 0.3) & (ph < 0.33)) + sig * r.randn(N); dy = sig * (0.7 + 0.6 * r.rand(N))
        fr = np.array([f])
        pref, _, _, _ = ref_sparse(t, y, dy, fr)
        out = {}
        out['gpu'] = sparse_bls_gpu(t, y, dy, fr, kernel=kern)[0][0]
        out['gpu-simple'] = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)[0][0]
        out['patched'] = sparse_bls_gpu(t, y, dy, fr, kernel=pk)[0][0]
        out['patched-simple'] = sparse_bls_gpu(t, y, dy, fr, kernel=pk_s, use_simple=True)[0][0]
        out['wrapper64'] = wrapper_centered(t, y, dy, fr, kernel=kern)[0][0]
        out['cpu'] = sparse_bls_cpu(t, y, dy, fr)[0][0]
        print("  N=%3d %-17s ref=%.6f | " % (N, tag, pref[0]) + "  ".join("%s %.1e" % (k, abs(v - pref[0]) / pref[0]) for k, v in out.items()))

print("\n=== B2. fixes, one very precise point (N=200, R = weight ratio), max power over 201 freqs")
r = np.random.RandomState(3); N = 200
t = np.sort(365 * r.rand(N)); y0 = 12.0 + 0.01 * r.randn(N); dy0 = 0.01 * np.ones(N); fq = np.linspace(0.5, 1.5, 201)
for R in (1e3, 1e4, 1e6):
    dy = dy0.copy(); dy[17] = 0.01 / np.sqrt(R)
    pref, _, _, _ = ref_sparse(t, y0, dy, fq)
    pg = sparse_bls_gpu(t, y0, dy, fq, kernel=kern)[0]
    pp = sparse_bls_gpu(t, y0, dy, fq, kernel=pk)[0]
    pps = sparse_bls_gpu(t, y0, dy, fq, kernel=pk_s, use_simple=True)[0]
    pw = wrapper_centered(t, y0, dy, fq, kernel=kern)[0]
    print("  R=%.0e: ref %.4f | current %.4f | patched-kernel %.4f (max|rel| %.1e) | patched-simple %.4f | wrapper64 %.4f (max|rel| %.1e)"
          % (R, pref.max(), pg.max(), pp.max(), np.abs(pp - pref).max() / pref.max(), pps.max(), pw.max(), np.abs(pw - pref).max() / pref.max()))

print("\n=== C. bit-neutrality of the patched kernel on already-centered float32 input (N=300, 200 freqs)")
for seed in range(3):
    t, y, dy = make(300, seed, ybar=0.0)
    w = dy ** -2; y = y - np.sum(w * y) / np.sum(w)
    fq = np.linspace(0.5, 2.5, 200)
    a = sparse_bls_gpu(t, y, dy, fq, kernel=kern)[0]; b = sparse_bls_gpu(t, y, dy, fq, kernel=pk)[0]
    a2 = sparse_bls_gpu(t, y, dy, fq, kernel=kern_s, use_simple=True)[0]; b2 = sparse_bls_gpu(t, y, dy, fq, kernel=pk_s, use_simple=True)[0]
    print("  seed %d: full max|diff| %.2e (bit-identical: %s) | simple max|diff| %.2e (bit-identical: %s)"
          % (seed, np.abs(a - b).max(), np.array_equal(a, b), np.abs(a2 - b2).max(), np.array_equal(a2, b2)))

print("\n=== D. sparse_threshold boundary: same mag-12 LC, N=499 (sparse default) vs dense eebls_gpu_fast at N=499 (use_sparse=False)")
for seed in range(3):
    t, y, dy = make(499, seed)
    freqs = 1.37 + (0.02 / 365 / 4) * np.arange(-60, 61)
    freqs, ps, _ = eebls_transit(t, y, dy, freqs=freqs)
    qv = q_transit(freqs)
    pref, _, _, _ = ref_sparse(t, y, dy, freqs, qmin=qv * 0.5, qmax=qv * 2.0)
    pw = wrapper_centered(t, y, dy, freqs, qmin=qv * 0.5, qmax=qv * 2.0, kernel=kern)[0]
    print("  seed %d: peak ref %.5f sparse-default %.5f (rel %+.1e) wrapper64-fixed %.5f (rel %+.1e); grid max|rel| current %.1e fixed %.1e"
          % (seed, pref.max(), ps[np.argmax(pref)], (ps[np.argmax(pref)] - pref.max()) / pref.max(),
             pw[np.argmax(pref)], (pw[np.argmax(pref)] - pref.max()) / pref.max(),
             (np.abs(ps - pref) / np.maximum(pref, 1e-12)).max(), (np.abs(pw - pref) / np.maximum(pref, 1e-12)).max()))
