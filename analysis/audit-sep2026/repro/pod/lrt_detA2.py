"""Detector A: weak-prior regime (does the freq-domain Gram overcount change the answer?) + memory reuse timing."""
import warnings, time
warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _marginal_statistic, _whitened_inner

def box(t, P, e, d):
    ph = np.fmod(t - e, P) / P; ph[ph < 0] += 1.0; ph[ph > 0.5] -= 1.0
    tm = np.zeros_like(t); tm[np.abs(ph) <= d / (2.0 * P)] = -1.0; return tm

def tdA(y, tau, V, Cc, sigma):
    n = len(y); Vd = V - V.mean(axis=0); yd = y - y.mean(); td = tau - tau.mean()
    Wz = np.linalg.inv(sigma**2 * np.eye(n) + Vd @ Cc @ Vd.T)
    return float(yd @ Wz @ td / np.sqrt(td @ Wz @ td))

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(1)
N = 512; T = 30.0; sigma = 1e-3
t = np.sort(rng.rand(N) * T)
# a basis vector that OVERLAPS the transit template strongly (so the prior matters):
P, dur = 5.3, 0.22
tau = box(t, P, 1.1, dur)
v1 = (t - t.mean()) / t.std()
v2 = box(t, P, 1.1 + 0.05, dur) + 0.3 * rng.randn(N)   # near-degenerate with the template
V = np.stack([v1, v2], axis=1)
nf = 2 * N
Y_t = lambda a: proc.compute_nufft(t, (a - a.mean()).astype(np.float32), nf)
Tk = Y_t(tau); Vk = [Y_t(V[:, j]) for j in range(2)]
psd = np.full(nf, N * sigma**2); w = np.ones(nf)
gscale = np.mean([_whitened_inner(Vk[j], Vk[j], psd, w) / ((V[:, j]-V[:, j].mean()) @ (V[:, j]-V[:, j].mean()) / sigma**2) for j in range(2)])
print("Gram overcount factor (freq/time) = %.3f" % gscale)
print("k = prior std of coeffs in units of sigma; entries: freq-domain DetA, time-domain DetA(Cc), time-domain DetA(gscale*Cc), matched-filter (freq, time)")
for k in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    Cc = np.diag([(k * sigma)**2, (k * sigma)**2])
    ratios = []
    for trial in range(5):
        c = rng.randn(2) * k * sigma
        y = 3e-3 * tau + V @ c + sigma * rng.randn(N)
        Yk = Y_t(y)
        fA = _marginal_statistic(Yk, Tk, Vk, psd, w, Cc)
        tA = tdA(y, tau, V, Cc, sigma); tAg = tdA(y, tau, V, gscale * Cc, sigma)
        fm = _whitened_inner(Yk, Tk, psd, w) / np.sqrt(_whitened_inner(Tk, Tk, psd, w))
        yd = y - y.mean(); td = tau - tau.mean(); tm = yd @ td / sigma**2 / np.sqrt(td @ td / sigma**2)
        ratios.append((fA / tA, fA / tAg, fm / tm))
    r = np.array(ratios)
    print("  k=%5.2f  freqA/timeA(Cc)=%.3f+-%.3f  freqA/timeA(g*Cc)=%.3f+-%.3f  matched freq/time=%.3f+-%.3f"
          % (k, r[:,0].mean(), r[:,0].std(), r[:,1].mean(), r[:,1].std(), r[:,2].mean(), r[:,2].std()))

print("\n=== per-template cost: default compute_nufft (alloc+plan per call) vs reused memory ===")
data = [(t.astype(np.float32), (tau - tau.mean()).astype(np.float32), nf)]
import pycuda.driver as cuda
for rep in range(3):
    t0 = time.perf_counter()
    for i in range(40):
        proc.nufft_proc.run(data)
    proc.nufft_proc.finish()
    a = (time.perf_counter() - t0) / 40
    mem = proc.nufft_proc.allocate(data)
    t0 = time.perf_counter()
    for i in range(40):
        proc.nufft_proc.run(data, memory=mem)
    proc.nufft_proc.finish()
    b = (time.perf_counter() - t0) / 40
    t0 = time.perf_counter()
    from cuvarbase.memory import NFFTMemory
    for i in range(10):
        m = NFFTMemory(proc.nufft_proc.sigma, mem[0].stream, mem[0].m).fromdata(t.astype(np.float32), tau.astype(np.float32), nf=nf, allocate=True)
    c = (time.perf_counter() - t0) / 10
    print("  rep %d: per-call default %.2f ms ; with reused memory %.2f ms ; NFFTMemory alloc+plan alone %.2f ms" % (rep, 1e3*a, 1e3*b, 1e3*c))
