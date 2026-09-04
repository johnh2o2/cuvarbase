import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase.utils import normalize_light_curves

def dft(t, c, freqs, chunk=4000):
    out = np.empty(len(freqs), complex)
    for a in range(0, len(freqs), chunk):
        ph = 2 * np.pi * np.outer(freqs[a:a+chunk], t)
        out[a:a+chunk] = (np.cos(ph) + 1j * np.sin(ph)) @ c
    return out

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def run_and_extract(proc, t, y, dy, freqs, fix_psi=False, **kw):
    nf = len(freqs); k0 = get_k0(freqs); df = freqs[1] - freqs[0]
    (tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
    mem = proc.allocate([(tn, yn, dyn)], nfreqs=[nf], k0s=[k0])
    if fix_psi:
        # candidate fix: give the w-spectrum NFFT its own psi precomputation (its grid size differs)
        mem[0].nfft_mem_w.precomp_psi = True
        mem[0].nfft_mem_w.allocate_precomp_psi(n0=len(t))
    r = proc.run([(t, y, dy)], memory=mem, freqs=freqs, **kw)
    proc.finish()
    p = np.array(r[0][1][:nf], float)
    SW = mem[0].nfft_mem_w.ghat_g.get().astype(complex); SYW = mem[0].nfft_mem_yw.ghat_g.get().astype(complex)
    w = dyn ** -2; w /= w.sum(); ybar = np.dot(w, yn); yw = w * (yn - ybar)
    modes_w = k0 + np.arange(2 * nf + k0); modes_yw = k0 + np.arange(nf)
    ew = np.abs(SW[:len(modes_w)] - dft(tn, w, modes_w * df)).max()
    eyw = np.abs(SYW[:nf] - dft(tn, yw, modes_yw * df)).max()
    print("   ng_yw=%d ng_w=%d  max|sw-exact|=%.2e  max|syw-exact|=%.2e" % (mem[0].nfft_mem_yw.n, mem[0].nfft_mem_w.n, ew, eyw))
    return p

print("=== BUG 1: shared precompute_psi between yw grid (ng=sigma*nf) and w grid (ng=sigma*(2nf+k0)) ===")
t, y, dy = make_lc(N=300, T=365.0, f0=3.1, seed=1)
freqs = grid(1.0 / (5 * 365.0), 20.0, 365.0)
ref = LombScargle(t, y, dy).power(freqs, method='cython')
for use_double in (True, False):
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=12, autoset_m=False)
    p = run_and_extract(proc, t, y, dy, freqs)
    report("dbl=%s sigma=4 m=12 fast_grid (as shipped)" % use_double, ref, p)
    p = run_and_extract(proc, t, y, dy, freqs, fast_grid=False)
    report("dbl=%s sigma=4 m=12 slow_gaussian_grid (no shared psi)" % use_double, ref, p)
    p = run_and_extract(proc, t, y, dy, freqs, fix_psi=True)
    report("dbl=%s sigma=4 m=12 fast_grid + own psi for w grid (candidate fix)" % use_double, ref, p)
    del proc

print("=== with the psi fix: error vs sigma, m (double) ===")
for sigma in (2, 3, 4):
    for m in (4, 6, 8, 12):
        proc = LombScargleAsyncProcess(use_double=True, sigma=sigma, m=m, autoset_m=False)
        p = run_and_extract(proc, t, y, dy, freqs, fix_psi=True)
        report("FIXED dbl sigma=%d m=%d" % (sigma, m), ref, p)
        del proc
print("=== with the psi fix: float32 error vs sigma, m ===")
for sigma in (2, 4):
    for m in (4, 8):
        proc = LombScargleAsyncProcess(use_double=False, sigma=sigma, m=m, autoset_m=False)
        p = run_and_extract(proc, t, y, dy, freqs, fix_psi=True)
        report("FIXED f32 sigma=%d m=%d" % (sigma, m), ref, p)
        del proc

print("=== with the psi fix: hetero N=1000 case from exp1 C ===")
t2, y2, dy2 = make_lc(N=1000, T=1000.0, f0=7.3, hetero=True, seed=2)
freqs2 = grid(1.0 / (5 * 1000.0), 20.0, 1000.0)
ref2 = LombScargle(t2, y2, dy2).power(freqs2, method='cython')
for use_double in (True, False):
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=8, autoset_m=False)
    p = run_and_extract(proc, t2, y2, dy2, freqs2)
    report("hetero dbl=%s as shipped" % use_double, ref2, p)
    p = run_and_extract(proc, t2, y2, dy2, freqs2, fix_psi=True)
    report("hetero dbl=%s FIXED" % use_double, ref2, p)
    del proc

print("=== BUG 2: floorf() on double in fast_gaussian_grid; T=3650 fmax=50 grid (ng*xval up to ~7e5) ===")
t3, y3, dy3 = make_lc(N=300, T=3650.0, f0=23.456, seed=5)
freqs3 = grid(1.0 / (5 * 3650.0), 50.0, 3650.0)
ref3 = gls_numpy_fast(t3, y3, dy3, freqs3)
(tn, yn, dyn), = normalize_light_curves([(t3, y3, dy3)])
# CPU emulation: how many points have floorf(float32(ng*xval - m)) != floor(ng*xval - m)?
for sigma in (4,):
    nf = len(freqs3); k0 = get_k0(freqs3)
    for ng in (sigma * nf, sigma * (2 * nf + k0)):
        xval = (tn - tn.min()) / (tn.max() - tn.min()) / (5 * 3650.0 / (tn.max() - tn.min()))
        v = ng * xval - 8
        bad = np.sum(np.floor(np.float32(v).astype(np.float64)) != np.floor(v))
        print("   ng=%d: %d / %d points get a wrong grid cell from floorf on double" % (ng, bad, len(v)))
for use_double in (True, False):
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=8, autoset_m=False)
    p = run_and_extract(proc, t3, y3, dy3, freqs3, fix_psi=True)
    report("T=3650 dbl=%s psi-fixed, fast_grid (floorf)" % use_double, ref3, p)
    p = run_and_extract(proc, t3, y3, dy3, freqs3, fix_psi=True, fast_grid=False)
    report("T=3650 dbl=%s psi-fixed, slow_grid (no floorf)" % use_double, ref3, p)
    del proc
