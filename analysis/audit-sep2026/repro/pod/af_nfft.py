"""NFFT race (run() returns before the async D2H copy lands), NUFFT-LRT BJD-scale float32
cast, Detector A (marginal) algebra check + limits, and per-template cost structure."""
import numpy as np, time, json, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.cunfft import NFFTAsyncProcess, nfft_adjoint_async
from cuvarbase.memory import NFFTMemory
from cuvarbase import nufft_lrt
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _marginal_statistic, _whitened_inner

rng = np.random.RandomState(5)
res = {}

# ---- 1. NFFTAsyncProcess.run() + immediate host read (memory-reuse path and default path)
n = 20000; nf = 40000
t = np.sort(rng.rand(n) * 100.); y = rng.randn(n)
proc = NFFTAsyncProcess(sigma=2, autoset_m=True)
mem = proc.allocate([(t, y, nf)])[0]
proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize(); ref = mem.ghat_c.copy()
bad = 0; N = 40
for k in range(N):
    mem.ghat_c[:] = 0
    mem.stream.synchronize()
    out = proc.run([(t, y, nf)], memory=[mem])[0]
    snap = np.array(out, copy=True)          # what a caller reading immediately sees
    mem.stream.synchronize()
    if not np.array_equal(snap, ref): bad += 1
print("memory-reuse path: immediate read stale/partial in %d/%d runs (host buffer pinned=%s)" % (bad, N, mem.pinned))
res['race_reuse'] = [bad, N]
bad2 = 0
for k in range(N):
    out = proc.run([(t, y, nf)])[0]
    snap = np.array(out, copy=True)
    cuda.Context.synchronize()
    if not np.array_equal(snap, np.asarray(out)): bad2 += 1
print("default (fresh memory) path: immediate read stale in %d/%d runs (masked only if cuMemFree of the dropped GPUArrays synchronizes)" % (bad2, N))
res['race_default'] = [bad2, N]

# ---- 2. NUFFT-LRT: BJD-scale times on the default float32 path
def box_lc(n=3000, T=100., P=3.3, dur=0.15, depth=0.01, sig=0.003, t0=1.1):
    t = np.sort(rng.rand(n) * T)
    ph = ((t - t0) / P) % 1.0; ph[ph > 0.5] -= 1
    y = 1.0 - depth * (np.abs(ph) < dur / (2*P)) + sig * rng.randn(n)
    return t, y
t, y = box_lc()
periods = np.array([3.3]); durations = np.array([0.15]); epochs = np.linspace(0, 3.3, 33)
lrt = NUFFTLRTAsyncProcess(sigma=2)
s0 = lrt.run(t, y, periods, durations, epochs=epochs)
s1 = lrt.run(t + 2455000.0, y, periods, durations, epochs=epochs + 2455000.0)
s0b = lrt.run(t, y, periods, durations, epochs=epochs)
print("LRT float32 path: max SNR over epochs  t~[0,100]: %.2f  |  t+2455000: %.2f  | repeat run (determinism): %.2f, max|d|=%.2e"
      % (s0.max(), s1.max(), s0b.max(), np.max(np.abs(s0 - s0b))))
res['lrt_bjd'] = dict(local=float(s0.max()), bjd=float(s1.max()), repeat_maxdiff=float(np.max(np.abs(s0-s0b))))
lrtd = NUFFTLRTAsyncProcess(sigma=2, use_double=True)
s0d = lrtd.run(t, y, periods, durations, epochs=epochs)
s1d = lrtd.run(t + 2455000.0, y, periods, durations, epochs=epochs + 2455000.0)
print("LRT use_double: local %.2f | bjd %.2f" % (s0d.max(), s1d.max()))
res['lrt_bjd_double'] = dict(local=float(s0d.max()), bjd=float(s1d.max()))

# ---- 3. Detector A algebra vs dense Woodbury in the real 2nf space
nf2 = 512; K = 3
Y = rng.randn(nf2) + 1j*rng.randn(nf2); T = rng.randn(nf2) + 1j*rng.randn(nf2)
V = [rng.randn(nf2) + 1j*rng.randn(nf2) for _ in range(K)]
psd = 0.5 + rng.rand(nf2); wts = np.ones(nf2)
A = rng.randn(K, K); Cc = A @ A.T + 0.1*np.eye(K)
def r(a): return np.concatenate([a.real, a.imag])
W = np.diag(np.concatenate([1/psd, 1/psd]))
R = np.stack([r(v) for v in V], axis=1)
Cz_inv = W - W @ R @ np.linalg.inv(np.linalg.inv(Cc) + R.T @ W @ R) @ R.T @ W
dense = (r(Y) @ Cz_inv @ r(T)) / np.sqrt(r(T) @ Cz_inv @ r(T))
impl = _marginal_statistic(Y, T, V, psd, wts, Cc)
matched = _whitened_inner(Y, T, psd, wts) / np.sqrt(_whitened_inner(T, T, psd, wts))
proj = (r(Y) @ (W - W @ R @ np.linalg.inv(R.T @ W @ R) @ R.T @ W) @ r(T)) / np.sqrt(r(T) @ (W - W @ R @ np.linalg.inv(R.T @ W @ R) @ R.T @ W) @ r(T))
print("Detector A: impl=%.6f dense-Woodbury=%.6f (rel diff %.1e); matched=%.6f; whitened-projection(flat prior)=%.6f" % (impl, dense, abs(impl-dense)/abs(dense), matched, proj))
lim0 = _marginal_statistic(Y, T, V, psd, wts, 1e-12*np.eye(K))
zero = _marginal_statistic(Y, T, V, psd, wts, np.zeros((K, K)))
liminf = _marginal_statistic(Y, T, V, psd, wts, 1e12*np.eye(K))
print("  prior_cov -> 0 (1e-12 I): %.6f (expect matched %.6f);  prior_cov == 0 exactly: %.6f (pinv(0)=0 => flat prior => %.6f);  prior_cov -> inf: %.6f (expect projection %.6f)"
      % (lim0, matched, zero, proj, liminf, proj))
res['detA'] = dict(impl=float(impl), dense=float(dense), matched=float(matched), proj=float(proj), lim0=float(lim0), zero=float(zero), liminf=float(liminf))

# ---- 4. Detector semantics on a transit + systematic
t, y = box_lc(n=2000, T=60., P=3.3, dur=0.15, depth=0.008, sig=0.003)
Vb = np.stack([np.sin(2*np.pi*t/23.), (t - t.mean())/t.std()], axis=1)
c_true = np.array([0.02, 0.01])
ysys = y + Vb @ c_true
periods = np.linspace(2.5, 4.5, 41); epochs = np.linspace(0, 1, 12)
out = {}
for det, kw in [('matched', {}), ('sequential', dict(systematics_basis=Vb)),
                ('marginal', dict(systematics_basis=Vb, coeff_prior_cov=np.diag([0.02**2, 0.01**2])))]:
    s = lrt.run(t, ysys, periods, durations=np.array([0.15]), epochs=epochs*3.3, detector=det, **kw)
    smax = s.max(axis=(1, 2)); itrue = np.argmin(np.abs(periods - 3.3))
    rest = np.delete(smax, itrue)
    out[det] = dict(snr_true=float(smax[itrue]), snr_best_other=float(rest.max()), argmax_is_true=bool(np.argmax(smax) == itrue))
    print("  detector=%-10s SNR@P=3.3: %.2f  best off-period: %.2f  argmax is true: %s" % (det, smax[itrue], rest.max(), np.argmax(smax) == itrue))
s = lrt.run(t, y, periods, durations=np.array([0.15]), epochs=epochs*3.3)
print("  (no systematic, matched) SNR@P: %.2f best other %.2f" % (s.max(axis=(1,2))[itrue], np.delete(s.max(axis=(1,2)), itrue).max()))
res['det_semantics'] = out

# ---- 5. per-template cost: default compute_nufft (alloc + cuFFT plan + copies each call) vs reused memory
t, y = box_lc(n=5000, T=100.)
nf = 2*len(t)
tmpl = lrt._generate_template(t, 3.3, 0.0, 0.15, 1.0); tmpl -= tmpl.mean()
lrt.compute_nufft(t, tmpl, nf)  # warm
def med(fn, reps=30):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.median(ts)
t_default = med(lambda: lrt.compute_nufft(t, tmpl, nf))
nproc = lrt.nufft_proc
memr = nproc.allocate([(t.astype(np.float32), tmpl.astype(np.float32), nf)])[0]
def reuse():
    memr.y[:] = tmpl.astype(np.float32)
    nfft_adjoint_async(memr, nproc.function_tuple, block_size=nproc.block_size)
    memr.stream.synchronize()
    return memr.ghat_c
reuse()
t_reuse = med(reuse)
t_alloc = med(lambda: nproc.allocate([(t.astype(np.float32), tmpl.astype(np.float32), nf)]))
t_tmpl = med(lambda: lrt._generate_template(t, 3.3, 0.0, 0.15, 1.0))
print("per-template: default compute_nufft %.2f ms | NFFTMemory alloc+plan alone %.2f ms | reuse preallocated memory %.2f ms | template gen %.3f ms  (n=%d, nf=%d; shared GPU -> relative only)"
      % (1e3*t_default, 1e3*t_alloc, 1e3*t_reuse, 1e3*t_tmpl, len(t), nf))
res['per_template_ms'] = dict(default=1e3*t_default, alloc=1e3*t_alloc, reuse=1e3*t_reuse, template=1e3*t_tmpl)
a = np.asarray(lrt.compute_nufft(t, tmpl, nf)); b = reuse()
print("  reuse result identical to default path: max|d|=%.2e" % np.max(np.abs(a - b)))
# host-side per-template cost of the marginal statistic (K=8, nf=1e4) vs matched
nfh = 10000; K = 8
Yh = rng.randn(nfh)+1j*rng.randn(nfh); Th = Yh.copy(); Vh = [rng.randn(nfh)+1j*rng.randn(nfh) for _ in range(K)]
psdh = 1+rng.rand(nfh); wh = np.ones(nfh)
t_m = med(lambda: _marginal_statistic(Yh, Th, Vh, psdh, wh, np.eye(K)), 50)
t_mf = med(lambda: lrt._compute_matched_filter_snr(Yh, Th, psdh, wh, 1e-12), 50)
print("host statistic per template: marginal K=8 nf=1e4: %.2f ms ; matched (incl. np.median per call): %.2f ms" % (1e3*t_m, 1e3*t_mf))
res['host_stat_ms'] = dict(marginal=1e3*t_m, matched=1e3*t_mf)
json.dump(res, open('/workspace/scratch/af_nfft.json', 'w'), indent=1)
