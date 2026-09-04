import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
rng = np.random.RandomState(9)
proc = NUFFTLRTAsyncProcess()
# reuse-path equality restricted to the guaranteed band k<nf/2
for nn in (5000, 50000):
    tt = np.sort(rng.uniform(0, 90, nn)); nf2 = 2*nn
    yy = rng.randn(nn).astype(np.float32); yy -= yy.mean()
    mem = proc.nufft_proc.allocate([(tt.astype(np.float32), yy, nf2)])
    def reuse():
        mem[0].y = yy; mem[0].ghat_g.fill(0)
        g = proc.nufft_proc.run([(tt, yy, nf2)], memory=mem)[0]; mem[0].stream.synchronize(); return g.copy()
    a = proc.compute_nufft(tt, yy, nf2); a2 = proc.compute_nufft(tt, yy, nf2); b = reuse(); b2 = reuse()
    lo = slice(0, nf2//2); hi = slice(nf2//2, nf2)
    rms = np.sqrt(np.mean(np.abs(a[lo])**2))
    print('n=%d: low band max|cur-reuse|/rms=%.1e, cur repeat %.1e, reuse repeat %.1e | upper band cur repeat max|d|/rms=%.1e' % (
        nn, np.abs(a[lo]-b[lo]).max()/rms, np.abs(a[lo]-a2[lo]).max()/rms, np.abs(b[lo]-b2[lo]).max()/rms, np.abs(a[hi]-a2[hi]).max()/rms))
# statistic scaling with nf on a well-resolved template (dur=0.6 d): pure normalization inflation?
t = make_times(rng); n = len(t); P, dur = 5.3, 0.6
y = 1 + 3e-3*rng.randn(n) + box(t, P, 1.0, dur, 0.006)
for k in (1, 2, 4, 8):
    s = proc.run(t, y, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]), nf=k*n)[0,0,0]
    s_flat = proc.run(t, y, np.array([P]), durations=np.array([dur]), epochs=np.array([1.0]), nf=k*n, estimate_psd=False, psd=np.full(k*n, n*9e-6, np.float32))[0,0,0]
    print('dur=0.6 d: nf=%d*n -> SNR@true est-PSD=%.2f  true-white-PSD=%.2f  (sqrt(nf/n)=%.2f)' % (k, s, s_flat, np.sqrt(k)))
