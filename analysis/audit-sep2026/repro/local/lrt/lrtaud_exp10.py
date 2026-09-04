"""Does restricting the whitened sums to the guaranteed band k<nf/2 fix the null normalization?"""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram
proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(10)
n = 600; sig = 1e-3; nf = 2*n; P, dur = 3.7, 0.15
for tname, t in (('uniform', np.linspace(0, 90, n)), ('ground', make_times(rng))):
    tm = proc._generate_template(t, P, 0.0, dur, 1.0); tm -= tm.mean()
    T = proc.compute_nufft(t, tm, nf)
    Ys = []
    for r in range(200):
        y = sig*rng.randn(n); y -= y.mean(); Ys.append(proc.compute_nufft(t, y, nf))
    for band, sl in (('all nf modes (current)', slice(0, nf)), ('k<nf/2 only', slice(0, nf//2))):
        w = np.zeros(nf); w[sl] = 1
        s_true = [proc._compute_matched_filter_snr(Y, T, np.full(nf, n*sig**2), w, 1e-12) for Y in Ys]
        s_est = []
        for Y in Ys:
            psd = _smoothed_periodogram(np.abs(Y)**2, 5); psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
            s_est.append(proc._compute_matched_filter_snr(Y, T, psd, w, 1e-12))
        print('%-8s %-24s: true-white-PSD std=%.3f | estimated-PSD(win 5) std=%.3f' % (tname, band, np.std(s_true), np.std(s_est)))
    # exact time-domain GLS reference for white noise: (tau^T y)/(sigma sqrt(tau^T tau)) -> N(0,1) by construction; and how the freq-domain stat correlates with it
    s_td = []; s_fd = []
    w = np.ones(nf)
    for i, Y in enumerate(Ys):
        pass
