import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram, _whitened_inner
import pycuda.driver as cuda

rng = np.random.RandomState(8)
t = make_times(rng); n = len(t); nf = 2*n
proc = NUFFTLRTAsyncProcess()
P, dur = 5.3, 0.22
# (a) separate self-whitening from systematics contamination of the estimated PSD
trend = np.sin(2*np.pi*t/40.0); trend -= trend.mean()
noise = 3e-3*rng.randn(n) + ou_noise(rng, t, 3e-3, 0.8)
tr = box(t, P, 1.0, dur, 0.006)
Yn = proc.compute_nufft(t, noise - noise.mean(), nf)
psd_clean = _smoothed_periodogram((np.abs(Yn)**2).astype(np.float32), 5)
def snr(y, **kw):
    return proc.run(t, y, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]), nf=nf, **kw)[0,0,0]
for lab, y in (('noise+transit', 1+noise+tr), ('noise+transit+10sigma systematics', 1+noise+tr+0.03*trend)):
    print('%-36s matched est-PSD %.2f | matched clean-PSD %.2f | marginal est-PSD %.2f | marginal clean-PSD %.2f' % (
        lab, snr(y), snr(y, estimate_psd=False, psd=psd_clean),
        snr(y, detector='marginal', systematics_basis=trend[:,None], coeff_prior_cov=[[1.0]]),
        snr(y, detector='marginal', systematics_basis=trend[:,None], coeff_prior_cov=[[1.0]], estimate_psd=False, psd=psd_clean)))
# self-whitening vs depth
for d in (0.003, 0.01, 0.03, 0.1):
    y = 1 + noise + box(t, P, 1.0, dur, d)
    print('depth %.3f: matched est-PSD %.2f | clean-PSD %.2f  (ratio %.2f)' % (d, snr(y), snr(y, estimate_psd=False, psd=psd_clean), snr(y)/snr(y, estimate_psd=False, psd=psd_clean)))
# (b) short-duration resolution vs nf
d2 = 0.06
y = 1 + 3e-3*rng.randn(n) + box(t, P, 1.0, d2, 0.02)
for k in (2, 4, 8, 16):
    s = proc.run(t, y, np.array([P]), durations=np.array([d2]), epochs=np.array([1.0]), nf=k*n)[0,0,0]
    print('dur=%.2f d: nf=%d*n -> SNR@true=%.2f' % (d2, k, s))
# (c) corrected reuse path: zero grid + sync, timing and equality
for nn in (600, 5000, 50000):
    tt = np.sort(rng.uniform(0, 90, nn)); nf2 = 2*nn
    yy = rng.randn(nn).astype(np.float32); yy -= yy.mean()
    mem = proc.nufft_proc.allocate([(tt.astype(np.float32), yy, nf2)])
    def reuse():
        mem[0].y = yy; mem[0].ghat_g.fill(0)
        g = proc.nufft_proc.run([(tt, yy, nf2)], memory=mem)[0]; mem[0].stream.synchronize(); return g.copy()
    def cur(): return proc.compute_nufft(tt, yy, nf2)
    a = cur(); b = reuse(); b2 = reuse()
    r = []
    for f in (cur, reuse):
        v = []
        for _ in range(40):
            t0 = time.perf_counter(); f(); v.append(time.perf_counter()-t0)
        r.append((np.median(v)*1e3, np.min(v)*1e3))
    print('n=%d: current %.2f ms (min %.2f) | reuse(zero+sync) %.2f ms (min %.2f) | max|cur-reuse|/rms=%.1e | reuse repeat max|d|=%.1e' % (
        nn, r[0][0], r[0][1], r[1][0], r[1][1], np.abs(a-b).max()/np.sqrt(np.mean(np.abs(a)**2)), np.abs(b-b2).max()))
# (d) float32 accumulation in _whitened_inner at large nf
for N in (10**4, 10**5, 10**6):
    A = (rng.randn(N) + 1j*rng.randn(N)).astype(np.complex64); B = (rng.randn(N) + 1j*rng.randn(N)).astype(np.complex64)
    psd = (0.5 + rng.rand(N)).astype(np.float32); w = np.ones(N, np.float32)
    f32 = _whitened_inner(A, B, psd, w); f64 = _whitened_inner(A.astype(np.complex128), B.astype(np.complex128), psd.astype(np.float64), w.astype(np.float64))
    den32 = _whitened_inner(B, B, psd, w); den64 = _whitened_inner(B.astype(np.complex128), B.astype(np.complex128), psd.astype(np.float64), w.astype(np.float64))
    print('nf=%d: num f32 vs f64 rel err %.1e ; den rel err %.1e' % (N, abs(f32-f64)/abs(f64), abs(den32-den64)/den64))
