"""Follow-up: (a) alias-term theory incl. the re-reference phase; (b) default-path z-score impact with the TRUE template on-grid."""
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

# (a) alias theory
t = make_times(rng); n = len(t); nf = 2*n
y = 3e-3*rng.randn(n); y -= y.mean()
m = proc.nufft_proc.get_m(nf, y=y); sigma = proc.nufft_proc.sigma; ng = int(sigma*nf)
b = (2*sigma/(2*sigma-1))*m/np.pi
G = proc.compute_nufft(t, y, nf).astype(np.complex128)
E = adjoint_dft_chunked(t, y, nf); Em = adjoint_dft_chunked(t, y, ng)
k = np.arange(nf); T = t.max()-t.min(); n0 = t.min()/T*ng
ratio = np.exp(-b*(np.pi/ng)**2*((k-ng)**2 - k**2))
sl = slice(nf//2, nf); rms = np.sqrt(np.mean(np.abs(E)**2))
base = np.conj(Em[(ng - k) % ng]) * ratio
print('upper band raw max|G-E|/rms = %.2e' % (np.abs(G[sl]-E[sl]).max()/rms))
for lab, ph in (('no phase', 1.0), ('exp(+2pi i n0)', np.exp(2j*np.pi*n0)), ('exp(-2pi i n0)', np.exp(-2j*np.pi*n0))):
    resid = G[sl]-E[sl]-(base*ph)[sl]
    print('  alias model %-16s -> residual max/rms = %.2e' % (lab, np.abs(resid).max()/rms))

# (b) z-score impact with on-grid true template
def stat_set(proc_, t, y, P, ep, dur, nufft=None, band=None, nf=None, null_grid=None):
    nf = nf or 2*len(t); y0 = y - y.mean()
    tr = (lambda t_, v: nufft(t_, v, nf)) if nufft else (lambda t_, v: proc_.compute_nufft(t_, v, nf).astype(np.complex128))
    Y = tr(t, y0)
    psd = _smoothed_periodogram(np.abs(Y)**2, 5); psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
    w = np.ones(nf) if band is None else (np.arange(nf) < band).astype(float)
    tm = proc_._generate_template(t, P, ep, dur, 1.0); tm -= tm.mean()
    s_true = proc_._compute_matched_filter_snr(Y, tr(t, tm), psd, w, 1e-12)
    s_null = []
    for (p, e) in null_grid:
        tm = proc_._generate_template(t, p, e, dur, 1.0); tm -= tm.mean()
        s_null.append(proc_._compute_matched_filter_snr(Y, tr(t, tm), psd, w, 1e-12))
    return s_true, np.array(s_null)

cases = []
cases.append(('ground n=600', make_times(np.random.RandomState(3)), 5.3, 0.22, 0.01, 3e-3))
tt = np.arange(0, 27, 2/1440.); tt = tt[(tt < 13) | (tt > 14)]; cases.append(('tess-like n=%d' % len(tt), tt, 5.3, 0.12, 0.002, 1e-3))
r0 = np.random.RandomState(11)
for lab, t, P, dur, depth, sig in cases:
    n = len(t); r3 = np.random.RandomState(7)
    y = 1 + sig*r3.randn(n) + box(t, P, 1.3, dur, depth)
    null_grid = [(p, e) for p in np.exp(r0.uniform(np.log(2), np.log(12), 60)) for e in [r0.uniform(0, 12)]]
    null_grid = [(p, r0.uniform(0, p)) for p in np.exp(r0.uniform(np.log(2), np.log(12), 120))]
    print('=== %s nf=2n=%d' % (lab, 2*n))
    variants = [('NFFT f32 all (default)', procf, None, None), ('NFFT f32 all (repeat)', procf, None, None),
                ('NFFT f64 all', proc, None, None), ('NFFT f32 k<nf/2', procf, None, n), ('NFFT f32 k<nf/2 (repeat)', procf, None, n)]
    if n <= 5000:
        variants += [('exact DFT all', proc, adjoint_dft_chunked, None), ('exact DFT k<nf/2', proc, adjoint_dft_chunked, n)]
    for vl, pr, nu, band in variants:
        st, sn = stat_set(pr, t, y, P, 1.3, dur, nufft=nu, band=band, null_grid=null_grid)
        print('  %-26s S_true=%.3f  null: mean=%.3f std=%.3f max=%.3f  -> z=(S_true-mean)/std=%.2f' % (vl, st, sn.mean(), sn.std(), sn.max(), (st-sn.mean())/sn.std()))
