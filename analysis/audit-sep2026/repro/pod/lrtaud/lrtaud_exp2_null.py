"""Null calibration of the LRT statistic."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(1)
n = 600; sig = 1e-3
tsets = {'uniform': np.linspace(0, 90, n), 'ground': make_times(rng)}
P, dur = 3.7, 0.15
NREAL = 200
w1 = lambda nf: np.ones(nf)
for tname, t in tsets.items():
    print('=== sampling:', tname)
    nfs = [n, 2*n, 4*n]
    T = {nf: proc.compute_nufft(t, (lambda m: m - m.mean())(proc._generate_template(t, P, 0.0, dur, 1.0)), nf) for nf in nfs}
    Ys = {nf: [] for nf in nfs}
    for r in range(NREAL):
        y = sig*rng.randn(n); y -= y.mean()
        for nf in nfs:
            Ys[nf].append(proc.compute_nufft(t, y, nf))
    # (a) true flat PSD = n sigma^2 (E|S_k|^2 for white noise in the adjoint-NFFT convention)
    for nf in nfs:
        s = [proc._compute_matched_filter_snr(Y, T[nf], np.full(nf, n*sig**2), w1(nf), 1e-12) for Y in Ys[nf]]
        print('  true white PSD  nf=%d (nf/n=%.0f): mean=%.3f std=%.3f  (sqrt(nf/n)=%.3f)' % (nf, nf/n, np.mean(s), np.std(s), np.sqrt(nf/n)))
    # (b) estimated PSD at nf=2n, various smoothing windows
    nf = 2*n
    for win in (1, 5, 21, 101, 'global'):
        s = []
        for Y in Ys[nf]:
            p = np.abs(Y)**2
            if win == 'global': psd = np.full(nf, p.mean())
            else: psd = _smoothed_periodogram(p, win) if win > 1 else p
            psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
            s.append(proc._compute_matched_filter_snr(Y, T[nf], psd, w1(nf), 1e-12))
        s = np.array(s)
        print('  estimated PSD nf=2n window=%-6s: mean=%.3f std=%.3f  kurtosis-3=%.2f' % (win, s.mean(), s.std(), ((s-s.mean())**4).mean()/s.var()**2-3))
    # (c) red noise: estimated PSD (default) vs oracle PSD (mean |S_k|^2 over independent realizations)
    Yr = []
    for r in range(NREAL):
        y = sig*rng.randn(n) + ou_noise(rng, t, 3*sig, 0.8); y -= y.mean()
        Yr.append(proc.compute_nufft(t, y, nf))
    oracle = np.mean([np.abs(Y)**2 for Y in Yr], axis=0)
    s_or = [proc._compute_matched_filter_snr(Y, T[nf], oracle, w1(nf), 1e-12) for Y in Yr]
    s_est = []
    for Y in Yr:
        psd = _smoothed_periodogram(np.abs(Y)**2, 5); psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
        s_est.append(proc._compute_matched_filter_snr(Y, T[nf], psd, w1(nf), 1e-12))
    print('  red noise (OU 3x, tau=0.8) nf=2n: oracle PSD std=%.3f | estimated(win=5) PSD std=%.3f' % (np.std(s_or), np.std(s_est)))
    # (d) unwhitened (psd=1) statistic scale: shows the "SNR" is in data units
    s = [proc._compute_matched_filter_snr(Y, T[nf], np.ones(nf), w1(nf), 1e-12) for Y in Ys[nf]]
    print('  psd=ones (README example 4 style) white noise: std=%.4f  (n*sigma=%.4f)' % (np.std(s), n*sig))
