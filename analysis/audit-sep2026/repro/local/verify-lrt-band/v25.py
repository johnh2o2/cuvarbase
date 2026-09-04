"""Verifier for topic lrt-upper-half-band (finding 25).
Q1: is the upper-band NFFT error the aliasing term phi_hat(k-n)/phi_hat(k)?  (theory check)
Q2: does the sqrt(nf/n) null inflation come from NFFT error or from the statistic itself (exact DFT)?
Q3: default-path impact: run() SNR spectrum determinism run-to-run, and all-modes vs k<nf/2 vs exact DFT.
"""
import numpy as np, sys, time
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram

def adjoint_dft_chunked(t, y, nf, chunk=2000):
    t = np.asarray(t, np.float64); y = np.asarray(y, np.float64)
    x = t / (t.max() - t.min()); out = np.empty(nf, np.complex128)
    for a in range(0, nf, chunk):
        k = np.arange(a, min(nf, a+chunk))
        out[a:a+len(k)] = np.exp(2j*np.pi*np.outer(k, x)) @ y
    return out

rng = np.random.RandomState(25)
proc = NUFFTLRTAsyncProcess(use_double=True)
procf = NUFFTLRTAsyncProcess()

# ---------- Q1: theory of the upper-band error ----------
t = make_times(rng); n = len(t); nf = 2*n
y = 3e-3*rng.randn(n); y -= y.mean()
m = proc.nufft_proc.get_m(nf, y=y)
sigma = proc.nufft_proc.sigma; ng = int(sigma*nf)
b = (2*sigma/(2*sigma-1))*m/np.pi
print('sigma=%g m=%d ng=%d b=%.3f' % (sigma, m, ng, b))
G = proc.compute_nufft(t, y, nf).astype(np.complex128)
E = adjoint_dft_chunked(t, y, nf)
# predicted alias: exact coefficient at k-ng times phi_hat(k-ng)/phi_hat(k)
Em = adjoint_dft_chunked(t, y, ng)  # modes 0..ng-1 ; mode k-ng == conj(mode ng-k) for real y
k = np.arange(nf)
ratio = np.exp(-b*(np.pi/ng)**2*((k-ng)**2 - k**2))
alias = np.conj(Em[(ng - k) % ng]) * ratio
for lab, sl in (('k<nf/2', slice(1, nf//2)), ('k>=nf/2', slice(nf//2, nf))):
    err = G[sl]-E[sl]; rms = np.sqrt(np.mean(np.abs(E)**2))
    resid = err - alias[sl]
    print('  %-8s max|G-E|/rms=%.2e  after subtracting predicted l=-1 alias: %.2e  (ratio at band edges %.1e .. %.1e)' % (
        lab, np.abs(err).max()/rms, np.abs(resid).max()/rms, ratio[sl][0], ratio[sl][-1]))

# ---------- Q2: null inflation: NFFT vs exact DFT, all modes vs k<nf/2 ----------
P, dur = 3.7, 0.15; sig = 1e-3; NREAL = 150
for tname, tt in (('uniform', np.linspace(0, 90, 600)), ('ground', make_times(np.random.RandomState(1)))):
    nn = len(tt)
    for nf_ in (nn, 2*nn, 4*nn):
        tm = proc._generate_template(tt, P, 0.0, dur, 1.0); tm -= tm.mean()
        TG = proc.compute_nufft(tt, tm, nf_).astype(np.complex128); TE = adjoint_dft_chunked(tt, tm, nf_)
        out = {}
        for wlab, w in (('all', np.ones(nf_)), ('k<nf/2', (np.arange(nf_) < nf_//2).astype(float))):
            sG=[]; sE=[]
            r2 = np.random.RandomState(100)
            for r in range(NREAL):
                yy = sig*r2.randn(nn); yy -= yy.mean()
                YG = proc.compute_nufft(tt, yy, nf_).astype(np.complex128); YE = adjoint_dft_chunked(tt, yy, nf_)
                psd = np.full(nf_, nn*sig**2)
                sG.append(proc._compute_matched_filter_snr(YG, TG, psd, w, 1e-12))
                sE.append(proc._compute_matched_filter_snr(YE, TE, psd, w, 1e-12))
            out[wlab] = (np.std(sG), np.std(sE))
        print('null %-7s nf=%d*n: all-modes std NFFT=%.3f exactDFT=%.3f | k<nf/2 std NFFT=%.3f exactDFT=%.3f' % (
            tname, nf_//nn, out['all'][0], out['all'][1], out['k<nf/2'][0], out['k<nf/2'][1]))

# ---------- Q3: default path run() : determinism + comparison ----------
def spectrum(proc_, t, y, periods, epochs_frac, dur, nufft=None, band=None, nf=None):
    nf = nf or 2*len(t)
    y0 = y - y.mean()
    Y = nufft(t, y0, nf) if nufft else proc_.compute_nufft(t, y0, nf).astype(np.complex128)
    psd = _smoothed_periodogram(np.abs(Y)**2, 5); psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
    w = np.ones(nf) if band is None else (np.arange(nf) < band).astype(float)
    out = np.zeros((len(periods), len(epochs_frac)))
    for i, p in enumerate(periods):
        for j, e in enumerate(epochs_frac):
            tm = proc_._generate_template(t, p, e*p, dur, 1.0); tm -= tm.mean()
            T = nufft(t, tm, nf) if nufft else proc_.compute_nufft(t, tm, nf).astype(np.complex128)
            out[i, j] = proc_._compute_matched_filter_snr(Y, T, psd, w, 1e-12)
    return out

def summarize(lab, s, ref=None):
    pk = np.unravel_index(s.argmax(), s.shape)
    sde = (s.max() - s.mean())/s.std()
    extra = '' if ref is None else '  max|d vs ref|=%.3f corr=%.5f argmax same=%s' % (np.abs(s-ref).max(), np.corrcoef(s.ravel(), ref.ravel())[0,1], pk == np.unravel_index(ref.argmax(), ref.shape))
    print('  %-28s peak=%.3f at %s  SDE-like=%.2f%s' % (lab, s.max(), pk, sde, extra))

cases = []
tg = make_times(np.random.RandomState(3)); cases.append(('ground n=600', tg, 5.3, 0.22, 0.01, 3e-3))
# TESS-like: 27 d, 2-min cadence with a 1-d gap, ~18000 pts
tt = np.arange(0, 27, 2/1440.); tt = tt[(tt < 13) | (tt > 14)]; cases.append(('tess-like n=%d' % len(tt), tt, 5.3, 0.12, 0.002, 1e-3))
for lab, t, P, dur, depth, sig in cases:
    r3 = np.random.RandomState(7); n = len(t)
    y = 1 + sig*r3.randn(n) + box(t, P, 1.3, dur, depth)
    periods = np.exp(np.linspace(np.log(2), np.log(12), 30)); epochs = np.linspace(0, 1, 10, endpoint=False)
    print('=== %s, nf=2n=%d, P=%g dur=%g depth=%g sig=%g' % (lab, 2*n, P, dur, depth, sig))
    t0 = time.time(); s1 = spectrum(procf, t, y, periods, epochs, dur); print('  (float32 spectrum: %.1fs)' % (time.time()-t0))
    s2 = spectrum(procf, t, y, periods, epochs, dur)
    print('  run-to-run float32 all modes: max|s1-s2|=%.4f  (peak %.2f)' % (np.abs(s1-s2).max(), s1.max()))
    s1b = spectrum(procf, t, y, periods, epochs, dur, band=n); s2b = spectrum(procf, t, y, periods, epochs, dur, band=n)
    print('  run-to-run float32 k<nf/2   : max|s1-s2|=%.4f  (peak %.2f)' % (np.abs(s1b-s2b).max(), s1b.max()))
    sd = spectrum(proc, t, y, periods, epochs, dur)
    if n <= 5000:
        se = spectrum(proc, t, y, periods, epochs, dur, nufft=adjoint_dft_chunked)
        seb = spectrum(proc, t, y, periods, epochs, dur, nufft=adjoint_dft_chunked, band=n)
        summarize('exact DFT all modes', se)
        summarize('exact DFT k<nf/2', seb, se)
        summarize('NFFT f32 all modes (default)', s1, se)
        summarize('NFFT f64 all modes', sd, se)
        summarize('NFFT f32 k<nf/2', s1b, seb)
    else:
        summarize('NFFT f32 all modes (default)', s1)
        summarize('NFFT f64 all modes', sd, s1)
        summarize('NFFT f32 k<nf/2', s1b, s1)
    # via the public run() API, twice
    r1 = procf.run(t, y, periods, durations=np.array([dur]), epochs=epochs*periods[0])  # epochs absolute; fine for determinism check
    r2 = procf.run(t, y, periods, durations=np.array([dur]), epochs=epochs*periods[0])
    print('  public run() twice: max|r1-r2|=%.4f  max=%.2f' % (np.abs(r1-r2).max(), r1.max()))
