"""Verifier: sparse_bls_simple.cu MAX_W_COMPLEMENT=1E-9 all-weight-box probe."""
import numpy as np
import pycuda.autoprimaryctx  # noqa
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls, _module_reader, find_kernel

# ---- A. guard semantics of the exact bls_power() in the simple kernel
def probe(complement):
    src = r"""
    #define MIN_W 1E-9
    #define MAX_W_COMPLEMENT %s
    __device__ float bls_power(float YW, float W, float YY, unsigned int ig){
        if (ig && YW > 0.f) return 0.f;
        if (W < MIN_W || W > 1.f - MAX_W_COMPLEMENT) return 0.f;
        return (YW * YW) / (W * (1.f - W) * YY);
    }
    __global__ void k(const float* W, float* out, int n){
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        if (i < n) out[i] = bls_power(1e-6f, W[i], 1e-4f, 0);
    }""" % complement
    mod = SourceModule(src, options=['--use_fast_math'])
    k = mod.get_function('k')
    one = np.float32(1.0)
    Ws = np.array([one, np.nextafter(one, np.float32(0)), one - 2*np.spacing(one),
                   1 - 1e-6, 1 - 1e-5, 1 - 9e-5, 1 - 2e-4, 0.5], dtype=np.float32)
    out = gpuarray.zeros(len(Ws), np.float32)
    k(gpuarray.to_gpu(Ws), out, np.int32(len(Ws)), block=(32,1,1), grid=(1,1))
    for w, o in zip(Ws, out.get()):
        print("   complement=%s  W=%.9g (1-W=%.3e) -> bls_power=%.4g" % (complement, w, 1 - float(w), o))

print("=== A. guard semantics (YW=1e-6, YY=1e-4): which W near 1 pass the upper guard?")
probe("1E-9"); probe("1E-4")

# ---- kernels: shipped simple, shipped full, and a scratch-patched simple (1E-4)
kern_f = compile_sparse_bls(block_size=64)
kern_s = compile_sparse_bls(block_size=64, use_simple=True)
txt = _module_reader(find_kernel('sparse_bls_simple'), cpp_defs=dict(BLOCK_SIZE=64))
assert '#define MAX_W_COMPLEMENT 1E-9' in txt
txt_fixed = txt.replace('#define MAX_W_COMPLEMENT 1E-9', '#define MAX_W_COMPLEMENT 1E-4')
kern_s_fixed = SourceModule(txt_fixed, options=['--use_fast_math']).get_function('sparse_bls_kernel_simple')

# ---- B. auditor's minimal repro (single-site nightly data, f ~ 1/d, default qmax)
print("\n=== B. minimal repro: t=concat([n+0.3*sort(rand(6)) for n in range(30)]), y=12+0.01*randn, dy=0.01, f in [0.995,1.005]")
r = np.random.RandomState(0)
t = np.concatenate([n + 0.3 * np.sort(r.rand(6)) for n in range(30)])
y = 12.0 + 0.01 * r.randn(len(t)); dy = 0.01 * np.ones_like(y)
fr = np.linspace(0.995, 1.005, 101)
ps, ss = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
ps2, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
pf, sf = sparse_bls_gpu(t, y, dy, fr, kernel=kern_f)
pc, sc = sparse_bls_cpu(t, y, dy, fr)
px, sx = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s_fixed, use_simple=True)
k = int(np.nanargmax(ps))
print("   simple : max=%.4g finite=%s  at f=%.4f q=%.4f phi=%.4f ; repeat identical=%s" % (np.nanmax(ps), np.all(np.isfinite(ps)), fr[k], ss[k][0], ss[k][1], np.array_equal(ps, ps2)))
print("   full   : max=%.4g finite=%s" % (pf.max(), np.all(np.isfinite(pf))))
print("   cpu    : max=%.4g" % pc.max())
print("   simple with 1E-4 patch: max=%.4g  max|patched-cpu|=%.2e  max|full-cpu|=%.2e" % (px.max(), np.abs(px - pc).max(), np.abs(pf - pc).max()))
print("   #freqs where simple > 0.3: %d/101 ; #freqs where simple differs from cpu by >1e-3: %d/101" % ((ps > 0.3).sum(), (np.abs(ps - pc) > 1e-3).sum()))

# ---- C. 40-seed probe (auditor sparse_drill.py section 3), plus patched kernel
print("\n=== C. 40-seed probe (nn nights = 20+seed%30, 6 pts/night, f in [0.995,1.005], default qmin/qmax)")
bad_s = bad_f = bad_x = 0; worst_s = worst_x = worst_f = 0.0; nonfinite_s = 0
for seed in range(40):
    r = np.random.RandomState(seed)
    nn = 20 + seed % 30
    t = np.concatenate([n + 0.3 * np.sort(r.rand(6)) for n in range(nn)])
    y = 12.0 + 0.01 * r.randn(len(t)); dy = 0.01 * np.ones_like(y)
    fr = np.linspace(0.995, 1.005, 101)
    ps, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
    pf, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_f)
    px, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s_fixed, use_simple=True)
    pc, _ = sparse_bls_cpu(t, y, dy, fr)
    if not np.all(np.isfinite(ps)): nonfinite_s += 1
    bad_s += int(not np.all(np.isfinite(ps)) or np.nanmax(ps) > 0.3)
    bad_f += int(not np.all(np.isfinite(pf)) or np.nanmax(pf) > 0.3)
    bad_x += int(not np.all(np.isfinite(px)) or np.nanmax(px) > 0.3)
    worst_s = max(worst_s, np.nanmax(np.abs(ps - pc))); worst_x = max(worst_x, np.abs(px - pc).max()); worst_f = max(worst_f, np.abs(pf - pc).max())
print("   shipped simple: %d/40 seeds bad (non-finite in %d), worst|simple-cpu|=%.3e" % (bad_s, nonfinite_s, worst_s))
print("   shipped full  : %d/40 seeds bad, worst|full-cpu|=%.3e" % (bad_f, worst_f))
print("   patched simple: %d/40 seeds bad, worst|patched-cpu|=%.3e" % (bad_x, worst_x))

# ---- D. one 1000x-precise point (sparse_more.py section B, R=1e6), random sampling over 365 d
print("\n=== D. N=200 random over 365d, one point with dy/1000 (R=1e6), f in [0.5,1.5]")
r = np.random.RandomState(3); N = 200
t = np.sort(365 * r.rand(N)); y0 = 12.0 + 0.01 * r.randn(N); dy = 0.01 * np.ones(N); dy[17] = 0.01 / 1e3
fq = np.linspace(0.5, 1.5, 201)
pc, _ = sparse_bls_cpu(t, y0, dy, fq)
pg, _ = sparse_bls_gpu(t, y0, dy, fq, kernel=kern_f)
pgs, _ = sparse_bls_gpu(t, y0, dy, fq, kernel=kern_s, use_simple=True)
pgx, _ = sparse_bls_gpu(t, y0, dy, fq, kernel=kern_s_fixed, use_simple=True)
print("   max power: cpu %.4g  gpu-full %.4g  gpu-simple %.4g  gpu-simple patched %.4g ; simple finite=%s" % (pc.max(), pg.max(), pgs.max(), pgx.max(), np.all(np.isfinite(pgs))))

# ---- E. does the patched simple kernel still agree with cpu on a real transit (regression of the fix)?
print("\n=== E. patched simple vs cpu on injected transit, q=0.1 f=1, N=80, qmin/qmax default")
from cuvarbase.tests.test_bls import data
t, y, dy = data(snr=30, q=0.1, phi0=0.3, freq=1.0, baseline=365., ndata=80)
fr = np.linspace(0.95, 1.05, 11)
pc, _ = sparse_bls_cpu(t, y, dy, fr)
px, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s_fixed, use_simple=True)
ps, _ = sparse_bls_gpu(t, y, dy, fr, kernel=kern_s, use_simple=True)
print("   max|patched-cpu|=%.2e  max|shipped-cpu|=%.2e  peak cpu %.4f" % (np.abs(px - pc).max(), np.abs(ps - pc).max(), pc.max()))
