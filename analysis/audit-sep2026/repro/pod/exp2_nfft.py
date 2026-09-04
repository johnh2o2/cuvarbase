import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, gls_numpy_fast, report
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase.cunfft import NFFTAsyncProcess
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

print("=== 1. LS-pipeline spectra sw/syw vs exact DFT (k0=1 grid, N=300, T=365, fmax=20) ===")
t, y, dy = make_lc(N=300, T=365.0, f0=3.1, seed=1)
freqs = grid(1.0 / (5 * 365.0), 20.0, 365.0)
nf = len(freqs); k0 = get_k0(freqs); df = freqs[1] - freqs[0]
(tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
w = dyn ** -2; w /= w.sum(); ybar = np.dot(w, yn); yw = w * (yn - ybar); YY = np.dot(w, (yn - ybar) ** 2)
modes_w = k0 + np.arange(2 * nf + k0)
modes_yw = k0 + np.arange(nf)
SW_exact = dft(tn, w, modes_w * df)
SYW_exact = dft(tn, yw, modes_yw * df)
ref = LombScargle(t, y, dy).power(freqs, method='cython')

def ls_from_sums(SW, SYW, nf, k0, YY):
    C = SW[:nf].real; S = SW[:nf].imag
    C2 = SW[k0 + 2 * np.arange(nf)].real; S2 = SW[k0 + 2 * np.arange(nf)].imag
    YC = SYW[:nf].real; YS = SYW[:nf].imag
    CC = 0.5 * (1 + C2) - C * C; SS = 0.5 * (1 - C2) - S * S; CS = 0.5 * S2 - C * S
    D = CC * SS - CS ** 2
    return (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (YY * D)

report("CPU power from exact sums vs astropy", ref, ls_from_sums(SW_exact, SYW_exact, nf, k0, YY))

for use_double in (False, True):
    for sigma in (2, 4):
        for m in (4, 8, 12):
            proc = LombScargleAsyncProcess(use_double=use_double, sigma=sigma, m=m, autoset_m=False)
            mem = proc.allocate([(tn, yn, dyn)], nfreqs=[nf], k0s=[k0])
            r = proc.run([(t, y, dy)], memory=mem, freqs=freqs)
            proc.finish()
            p = np.array(r[0][1][:nf], float)
            SW = mem[0].nfft_mem_w.ghat_g.get(); SYW = mem[0].nfft_mem_yw.ghat_g.get()
            ew = np.abs(SW[:len(modes_w)] - SW_exact); eyw = np.abs(SYW[:nf] - SYW_exact)
            tag = "dbl=%s sigma=%d m=%2d" % (use_double, sigma, m)
            q = len(ew) // 4
            print("%s  |sw-exact| max=%.2e (quartiles %s)  |syw-exact| max=%.2e (quartiles %s)" % (
                tag, ew.max(), ["%.1e" % ew[i*q:(i+1)*q].max() for i in range(4)],
                eyw.max(), ["%.1e" % eyw[i*(nf//4):(i+1)*(nf//4)].max() for i in range(4)]))
            report("   GPU power vs astropy " + tag, ref, p)
            report("   CPU power from GPU sums vs astropy " + tag, ref, ls_from_sums(SW.astype(complex), SYW.astype(complex), nf, k0, YY))
            report("   GPU power vs CPU power from GPU sums " + tag, ls_from_sums(SW.astype(complex), SYW.astype(complex), nf, k0, YY), p)
            del proc, mem

print("=== 2. bare NFFT adjoint vs exact DFT across the full grid: error vs mode fraction k/ng ===")
rng = np.random.RandomState(0)
N = 200; tt = np.sort(rng.rand(N)); yy = rng.randn(N)
nfreq = 2000
for use_double in (False, True):
    for sigma in (2, 4):
        for m in (4, 8, 12):
            proc = NFFTAsyncProcess(sigma=sigma, m=m, autoset_m=False, use_double=use_double)
            # ask for modes 0..sigma*nfreq-1 (the whole FFT grid) by allocating nf=nfreq and reading ghat_g fully
            mem = proc.allocate([(tt, yy, nfreq)])
            proc.run([(tt, yy, nfreq)], memory=mem, minimum_frequency=0.0, samples_per_peak=1)
            proc.finish()
            ng = mem[0].n
            g = mem[0].ghat_g.get()  # normalize kernel only wrote nf entries; rest hold raw ifft output
            T = tt.max() - tt.min()
            modes = np.arange(nfreq)
            exact = dft(tt, yy, modes / T)
            e = np.abs(g[:nfreq] - exact)
            frac = modes / ng
            bins = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
            s = []
            for a, b in zip(bins[:-1], bins[1:]):
                sel = (frac >= a) & (frac < b)
                s.append("%.2f-%.2f:%s" % (a, b, ("%.1e" % e[sel].max()) if sel.any() else "-"))
            bound = 4 * np.exp(-m * np.pi * (1 - 1.0 / (2 * sigma - 1))) * np.abs(yy).sum()
            print("dbl=%s sigma=%d m=%2d ng=%d bound=%.1e  err by k/ng: %s" % (use_double, sigma, m, ng, bound, "  ".join(s)))
            del proc, mem

print("=== 3. fractional k0 (minimum_frequency not an integer multiple of df) ===")
proc = NFFTAsyncProcess(sigma=4, m=8, autoset_m=False, use_double=True)
for frac_k0 in (0.0, 0.01, 0.1, 0.5):
    mem = proc.allocate([(tt, yy, nfreq)])
    T = tt.max() - tt.min()
    proc.run([(tt, yy, nfreq)], memory=mem, minimum_frequency=(10 + frac_k0) / T, samples_per_peak=1)
    proc.finish()
    g = mem[0].ghat_g.get()[:nfreq]
    exact = dft(tt, yy, (10 + frac_k0 + np.arange(nfreq)) / T)
    exact_int = dft(tt, yy, (10 + np.arange(nfreq)) / T)
    print("k0=10+%.2f: max|nfft - exact(frac)|=%.2e   max|nfft - exact(int k0)|=%.2e" % (frac_k0, np.abs(g - exact).max(), np.abs(g - exact_int).max()))

print("=== 4. float32 large phase: t in [0, 3650], modes up to 50 c/d (k up to 912500) ===")
t, y, dy = make_lc(N=300, T=3650.0, f0=23.456, seed=5)
tn = t - t.mean(); T = tn.max() - tn.min()
nfreq = int(50 * 5 * T)
for use_double in (False, True):
    proc = NFFTAsyncProcess(sigma=4, m=8, autoset_m=False, use_double=use_double)
    mem = proc.allocate([(tn, y - y.mean(), nfreq)])
    proc.run([(tn, y - y.mean(), nfreq)], memory=mem, minimum_frequency=1.0 / (5 * T), samples_per_peak=5)
    proc.finish()
    g = mem[0].ghat_g.get()[:nfreq]
    fr = (1 + np.arange(nfreq)) / (5 * T)
    exact = dft(tn, y - y.mean(), fr)
    e = np.abs(g - exact); q = nfreq // 5
    print("dbl=%s: max err=%.2e rel-to-||y||_1=%.2e  quintiles: %s" % (use_double, e.max(), e.max() / np.abs(y - y.mean()).sum(),
          ["%.1e" % e[i*q:(i+1)*q].max() for i in range(5)]))
    # same but shifting time origin to tmin (t >= 0)
    del proc, mem
