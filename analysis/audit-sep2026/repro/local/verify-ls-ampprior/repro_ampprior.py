"""Minimal reproduction: does amplitude_prior reach the GPU multiharmonic (nharmonics>1) path?
Reference: ridge on amplitudes only (offset unpenalised), lam = 1/s^2, in the weighted
chi2 objective; P = 1 - chi2_reg/YY, which is what add_regularization + mhgls_from_sums compute."""
import numpy as np, sys
from cuvarbase.lombscargle import (LombScargleAsyncProcess, lomb_scargle_direct_sums,
                                   mhdirect_sums, add_regularization, mhgls_from_sums)

def ridge_power(t, y, dy, freqs, H, lam):
    w = dy ** -2; w /= w.sum()
    ybar = np.dot(w, y); yc = y - ybar; YY = np.dot(w, yc ** 2)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        cols = [np.ones_like(t)]
        for h in range(1, H + 1):
            cols += [np.cos(2*np.pi*h*f*t), np.sin(2*np.pi*h*f*t)]
        X = np.vstack(cols).T
        A = (X * w[:, None]).T @ X; b = (X * w[:, None]).T @ yc
        R = lam * np.eye(2*H+1); R[0, 0] = 0
        th = np.linalg.solve(A + R, b)
        chi2 = np.dot(w, (yc - X @ th) ** 2) + lam * np.dot(th[1:], th[1:])
        out[i] = 1 - chi2 / YY
    return out

rng = np.random.RandomState(7)
N = 250; t = np.sort(rng.rand(N)) * 80.0
f0 = 0.9
y = 5 + 0.4*np.sin(2*np.pi*f0*t) + 0.25*np.sin(2*np.pi*2*f0*t + 0.7) + 0.1*rng.randn(N)
dy = 0.1 * np.exp(0.2*rng.randn(N))
T = t.max() - t.min(); df = 1.0/(5*T); k0 = 5; nf = 400
freqs = df*(k0 + np.arange(nf))
w = dy**-2; w /= w.sum(); ybar = np.dot(w, y); YY = np.dot(w, (y-ybar)**2)

def stat(a, b): return np.max(np.abs(a-b))

for s in (0.3, 0.05):
    lam = s**-2
    print("\n### amplitude_prior = %.2f (lam = %.1f)" % (s, lam))
    for H in (1, 2, 3):
        ref_reg = ridge_power(t, y, dy, freqs, H, lam)
        ref_unreg = ridge_power(t, y, dy, freqs, H, 0.0)
        cpu_reg = lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=H, amplitude_priors=s)
        print("H=%d  CPU direct_sums(amplitude_priors) vs ridge: maxabs=%.2e   (reg vs unreg ref differ by %.2e)"
              % (H, stat(cpu_reg, ref_reg), stat(ref_reg, ref_unreg)))
        for use_double in (True, False):
            for use_fft in (True, False):
                proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=8, autoset_m=False, nharmonics=H)
                kw = dict(amplitude_prior=s, use_fft=use_fft)
                if use_fft: kw['fast_grid'] = False
                r = proc.run([(t, y, dy)], freqs=freqs, **kw); proc.finish()
                p = np.array(r[0][1][:nf], float)
                # check memory actually got the prior
                mem = proc.memory[0] if getattr(proc, 'memory', None) else None
                print("   GPU H=%d dbl=%-5s use_fft=%-5s  vs ridge: %.2e   vs UNREG: %.2e"
                      % (H, use_double, use_fft, stat(p, ref_reg), stat(p, ref_unreg)))
                del proc
