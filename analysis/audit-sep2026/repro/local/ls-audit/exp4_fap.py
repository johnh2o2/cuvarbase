import numpy as np, time
from scipy.special import gammaln
import mpmath as mp
from cuvarbase.lombscargle import fap_baluev
from astropy.timeseries.periodograms.lombscargle import _statistics as st

rng = np.random.RandomState(42)
def fap_mp(t, dy, z, fmax, d_K=3, d_H=1):
    """Baluev 2008 eq. 5/6 for the 'standard' (z_1) normalization, evaluated in 50-digit arithmetic."""
    mp.mp.dps = 50
    N = len(t); d = d_K - d_H; N_K = N - d_K; N_H = N - d_H
    w = np.power(dy, -2.0); tbar = np.dot(w, t) / w.sum(); Dt = np.dot(w, (t - tbar) ** 2) / w.sum()
    Teff = mp.sqrt(4 * mp.pi * mp.mpf(Dt)); W = mp.mpf(fmax) * Teff
    g = mp.gamma(mp.mpf(N_H) / 2) / mp.gamma(mp.mpf(N_K + 1) / 2)
    z = mp.mpf(z)
    tau = g * W * mp.sqrt(z) * (1 - z) ** (mp.mpf(N_K - 1) / 2)
    Psingle = (1 - z) ** (mp.mpf(N_K) / 2)          # single-frequency FAP
    return float(1 - (1 - Psingle) * mp.exp(-tau))

for N in (20, 100, 1000, 20000):
    t = np.sort(365 * rng.rand(N)); dy = 0.01 * (1 + 0.5 * rng.rand(N)); y = rng.randn(N)
    fmax = 10.0
    zs = np.array([1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 0.999])
    fc = fap_baluev(t, dy, zs, fmax)
    fa = st.fap_baluev(zs, fmax, t, y, dy, normalization='standard')
    fm = np.array([fap_mp(t, dy, z, fmax) for z in zs])
    print("N=%d" % N)
    for z, a, b, c in zip(zs, fc, fa, fm):
        rel_a = abs(a - b) / max(abs(c), 1e-300); rel_m = abs(a - c) / max(abs(c), 1e-300)
        print("   z=%.3f cuvarbase=%.6e astropy=%.6e mpmath=%.6e  rel(cuv-astropy)=%.1e rel(cuv-mp)=%.1e" % (z, a, b, c, rel_a, rel_m))

print("=== effective bandwidth: W uses fmax only (fmin ignored) in cuvarbase and astropy ===")
N = 100; t = np.sort(365 * rng.rand(N)); dy = 0.01 * np.ones(N)
print("fap(z=0.3, fmax=10) =", fap_baluev(t, dy, 0.3, 10.0), " fap(z=0.3,fmax=10-9=1) =", fap_baluev(t, dy, 0.3, 1.0))
print("=== out-of-range z ===")
with np.errstate(all='ignore'):
    print("z=[-1, 1.5, nan]:", fap_baluev(t, dy, np.array([-1.0, 1.5, np.nan]), 10.0))
print("=== tiny N (N <= d_K) ===")
for N in (2, 3, 4):
    tt = np.sort(rng.rand(N)); dd = np.ones(N)
    with np.errstate(all='ignore'):
        print("N=%d:" % N, fap_baluev(tt, dd, np.array([0.1, 0.9]), 10.0))
