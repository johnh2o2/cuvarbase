import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import report
from cuvarbase.lombscargle import (LombScargleAsyncProcess, get_k0, lomb_scargle_direct_sums,
                                   mhdirect_sums, add_regularization, mhgls_from_sums, _mh_power_from_spectra)
from cuvarbase.utils import normalize_light_curves

def lstsq_power(t, y, dy, freqs, H, lam=0.0):
    w = dy ** -2; w /= w.sum(); sw = np.sqrt(w)
    ybar = np.dot(w, y); yc = y - ybar; YY = np.dot(w, yc ** 2)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        cols = [np.ones_like(t)]
        for h in range(1, H + 1):
            cols += [np.cos(2 * np.pi * h * f * t), np.sin(2 * np.pi * h * f * t)]
        X = np.vstack(cols).T
        if lam == 0.0:
            th, *_ = np.linalg.lstsq(sw[:, None] * X, sw * yc, rcond=None)
            chi2 = np.dot(w, (yc - X @ th) ** 2)
        else:
            # ridge on amplitudes only (offset unpenalized); objective = sum w r^2 + lam*|amp|^2
            A = (X * w[:, None]).T @ X; b = (X * w[:, None]).T @ yc
            R = lam * np.eye(2 * H + 1); R[0, 0] = 0
            th = np.linalg.solve(A + R, b)
            chi2 = np.dot(w, (yc - X @ th) ** 2) + lam * np.dot(th[1:], th[1:])
        out[i] = 1 - chi2 / YY
    return out

rng = np.random.RandomState(3)
N = 300; t = np.sort(rng.rand(N)) * 100.0
f0 = 1.3
y = 12 + 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t + 0.4) + 0.2 * np.cos(2 * np.pi * 3 * f0 * t) + 0.1 * rng.randn(N)
dy = 0.1 * np.exp(0.3 * rng.randn(N))
T = t.max() - t.min()
df = 1.0 / (5 * T); k0 = 5; nf = 3000
freqs = df * (k0 + np.arange(nf))
w = dy ** -2; w /= w.sum(); ybar = np.dot(w, y); YY = np.dot(w, (y - ybar) ** 2)

print("=== CPU lomb_scargle_direct_sums vs weighted lstsq (P = 1 - chi2/chi2_0) ===")
refs = {}
for H in (1, 2, 3):
    refs[H] = lstsq_power(t, y, dy, freqs, H)
    p = lomb_scargle_direct_sums(t, w * y, w, freqs, YY, nharms=H)
    report("direct_sums H=%d vs lstsq" % H, refs[H], p, freqs)

print("=== CPU regularization semantics: add_regularization(amplitude_priors=s) == ridge lam=1/s^2 on amplitudes? ===")
for H in (1, 2):
    for s in (0.5, 0.1):
        lam = s ** -2
        ref = lstsq_power(t, y, dy, freqs[:300], H, lam=lam)
        p = lomb_scargle_direct_sums(t, w * y, w, freqs[:300], YY, nharms=H, amplitude_priors=s)
        report("reg H=%d amplitude_priors=%.2f vs ridge lam=%.1f" % (H, s, lam), ref, p)

print("=== GPU nharmonics=1,2,3 (FFT path) vs lstsq; as shipped and with fast_grid=False (avoids shared-psi bug) ===")
for H in (1, 2, 3):
    for use_double in (True, False):
        proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=8, autoset_m=False, nharmonics=H)
        for fg in (True, False):
            r = proc.run([(t, y, dy)], freqs=freqs, fast_grid=fg); proc.finish()
            p = np.array(r[0][1][:nf], float)
            report("GPU H=%d dbl=%s fast_grid=%s vs lstsq" % (H, use_double, fg), refs[H], p, freqs)
        # direct sums path with H>1?
        if H > 1:
            r = proc.run([(t, y, dy)], freqs=freqs[:500], use_fft=False); proc.finish()
            p = np.array(r[0][1][:500], float)
            report("GPU H=%d dbl=%s use_fft=False (dirsum kernel) vs lstsq H=%d" % (H, use_double, H), refs[H][:500], p)
            report("   ... vs lstsq H=1" , refs[1][:500], p)
        del proc

print("=== GPU amplitude_prior: H=1 vs CPU ridge; H=2 -- is it applied? ===")
for H in (1, 2):
    proc = LombScargleAsyncProcess(use_double=True, sigma=4, m=8, autoset_m=False, nharmonics=H)
    s = 0.3
    r = proc.run([(t, y, dy)], freqs=freqs[:500], amplitude_prior=s, fast_grid=False); proc.finish()
    p = np.array(r[0][1][:500], float)
    report("GPU H=%d amplitude_prior=%.1f vs ridge lam=%.2f" % (H, s, s ** -2), lstsq_power(t, y, dy, freqs[:500], H, lam=s ** -2), p)
    report("GPU H=%d amplitude_prior=%.1f vs UNregularized" % (H, s), refs[H][:500], p)
    del proc

print("=== timing of _mh_power_from_spectra host loop (H=2) ===")
for nfx in (10000, 100000):
    H = 2
    sw = np.random.randn((2 * H - 1) * k0 + 2 * H * nfx + 1) * 0.01 + 0j; sw[0] = 1
    syw = np.random.randn((H - 1) * k0 + H * nfx + 1) * 0.01 + 0j
    t0 = time.time(); _mh_power_from_spectra(sw, syw, k0, H, nfx, 1.0); dt = time.time() - t0
    print("nf=%d H=%d: %.2f s (%.1f us/freq)" % (nfx, H, dt, 1e6 * dt / nfx))
