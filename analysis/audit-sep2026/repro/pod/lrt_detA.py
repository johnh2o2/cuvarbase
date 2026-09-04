"""NUFFT-LRT Detector A / sequential soundness experiments (GPU)."""
import warnings, time, sys
warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase.nufft_lrt import (NUFFTLRTAsyncProcess, _marginal_statistic,
                                 _whitened_inner, _sequential_detrend)

def box(t, P, e, d):
    ph = np.fmod(t - e, P) / P
    ph[ph < 0] += 1.0
    ph[ph > 0.5] -= 1.0
    tm = np.zeros_like(t)
    tm[np.abs(ph) <= d / (2.0 * P)] = -1.0
    return tm

def time_domain_detA(y, tau, V, Cc, sigma):
    """Exact Taaki Detector A in the time domain, white noise sigma,
    with y, tau, V columns all demeaned (matches the freq path's k=0 kill)."""
    n = len(y)
    Vd = V - V.mean(axis=0)
    yd = y - y.mean(); td = tau - tau.mean()
    Cz = sigma**2 * np.eye(n) + Vd @ Cc @ Vd.T
    Wz = np.linalg.inv(Cz)
    return float(yd @ Wz @ td / np.sqrt(td @ Wz @ td))

def time_domain_matched(y, tau, sigma):
    yd = y - y.mean(); td = tau - tau.mean()
    return float(yd @ td / sigma**2 / np.sqrt(td @ td / sigma**2))

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(0)

print("=== (a) freq-domain Woodbury statistic vs exact time-domain Detector A ===")
for label, N, irregular in [("uniform N=512", 512, False), ("irregular N=512", 512, True)]:
    T = 30.0
    if irregular:
        t = np.sort(rng.rand(N) * T)
    else:
        t = np.linspace(0, T, N, endpoint=False)
    sigma = 1e-3
    V = np.stack([(t - t.mean()) / t.std(), np.sin(2 * np.pi * t / 11.0)], axis=1)
    Cc = np.diag([(5 * sigma) ** 2, (3 * sigma) ** 2])
    P, dur, depth = 5.3, 0.22, 4e-3
    tau = box(t, P, 1.1, dur)
    c_true = rng.randn(2) * np.sqrt(np.diag(Cc))
    y = depth * tau + V @ c_true + sigma * rng.randn(N)
    nf = 2 * N
    Y = proc.compute_nufft(t, (y - y.mean()).astype(np.float32), nf)
    Tk = proc.compute_nufft(t, (tau - tau.mean()).astype(np.float32), nf)
    Vk = [proc.compute_nufft(t, (V[:, j] - V[:, j].mean()).astype(np.float32), nf) for j in range(2)]
    psd = np.full(nf, N * sigma ** 2)
    w = np.ones(nf)
    fm = _whitened_inner(Y, Tk, psd, w) / np.sqrt(_whitened_inner(Tk, Tk, psd, w))
    tm = time_domain_matched(y, tau, sigma)
    fA = _marginal_statistic(Y, Tk, Vk, psd, w, Cc)
    tA = time_domain_detA(y, tau, V, Cc, sigma)
    # what prior scale makes the freq-domain statistic agree with time-domain?
    best = None
    for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
        v = _marginal_statistic(Y, Tk, Vk, psd, w, Cc / s)   # Cc/s: freq path thinks prior is s x tighter
        tAs = time_domain_detA(y, tau, V, Cc / s, sigma)
        print("   %s: prior/%.2f  freqA=%.4f timeA(same prior)=%.4f ratio=%.4f" % (label, s, v, tAs, v / tAs))
    # Gram scale: <v,v>_W freq vs time
    g_f = _whitened_inner(Vk[0], Vk[0], psd, w)
    v0 = V[:, 0] - V[:, 0].mean()
    g_t = v0 @ v0 / sigma ** 2
    print("%s: matched freq/time = %.4f (%.4f vs %.4f); DetA freq/time = %.4f (%.4f vs %.4f); Gram freq/time = %.4f"
          % (label, fm / tm, fm, tm, fA / tA, fA, tA, g_f / g_t))
    # Consistency check: is the freq-domain DetA == time-domain DetA with prior 2x wider (scaled inner product)?
    tA2 = time_domain_detA(y, tau, V, Cc * (g_f / g_t), sigma)
    print("   time-domain DetA with prior x%.3f (the freq Gram scale): %.4f  -> freqA/that = %.4f (matched scale %.4f)"
          % (g_f / g_t, tA2, fA / tA2, fm / tm))

print("\n=== (c) BJD-scale times: matched detector on t vs t+2457000 ===")
N = 600; T = 60.0
t = np.sort(rng.rand(N) * T); sigma = 1e-3
P, dur = 5.3, 0.22
y = 1.0 + 5e-3 * box(t, P, 1.1, dur) + sigma * rng.randn(N)
periods = np.linspace(4.0, 7.0, 31)
epochs = np.linspace(0, 5.3, 10)
snr0 = proc.run(t, y, periods, durations=np.array([dur]), epochs=epochs)
snr1 = proc.run(t + 2457000.0, y, periods, durations=np.array([dur]), epochs=epochs + 2457000.0)
i0 = np.unravel_index(np.argmax(snr0), snr0.shape); i1 = np.unravel_index(np.argmax(snr1), snr1.shape)
print("relative t: best P=%.3f epoch=%.3f SNR=%.2f" % (periods[i0[0]], epochs[i0[2]], snr0.max()))
print("BJD t    : best P=%.3f epoch=%.3f SNR=%.2f" % (periods[i1[0]], epochs[i1[2]], snr1.max()))
print("corr(snr0, snr1) = %.4f ; max|diff| = %.3g ; float32(2457000+t) resolution = %.3g d"
      % (np.corrcoef(snr0.ravel(), snr1.ravel())[0, 1], np.abs(snr0 - snr1).max(),
         np.spacing(np.float32(2457000.0))))

print("\n=== (d) per-template cost: current loop vs one batched NFFT run ===")
N=600; sigma=3e-3; dur_true=0.22; periods=np.exp(np.linspace(np.log(2.0), np.log(18.0), 40)); t=np.sort(rng.rand(N)*60.0)
nf = 2 * N
tmpls = [(t.astype(np.float32), (box(t, p, 0.0, dur_true) - box(t, p, 0.0, dur_true).mean()).astype(np.float32), nf) for p in periods]
for rep in range(3):
    t0w = time.time()
    for tt, tm, nn in tmpls:
        proc.compute_nufft(tt, tm, nn)
    a = time.time() - t0w
    t0w = time.time()
    r = proc.nufft_proc.run(tmpls); proc.nufft_proc.finish()
    b = time.time() - t0w
    print("  rep %d: loop %d templates: %.1f ms (%.2f ms/template); batched run(): %.1f ms" % (rep, len(tmpls), 1e3 * a, 1e3 * a / len(tmpls), 1e3 * b))
