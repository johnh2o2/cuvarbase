"""NFFT accuracy per band (k<nf/2 vs k>=nf/2) vs exact adjoint DFT; m chosen
for data vs template; effect on the SNR spectrum (esp. flat PSD)."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

rng = np.random.RandomState(0)
t = make_times(rng); n = len(t); nf = 2*n
y = 3e-3*rng.randn(n); y -= y.mean()
P, dur = 5.3, 0.22
for use_double in (False, True):
    proc = NUFFTLRTAsyncProcess(use_double=use_double)
    tmpl = proc._generate_template(t.astype(proc.real_type), P, 0.0, dur, 1.0); tmpl -= tmpl.mean()
    print('use_double=%s  m(data)=%d  m(template)=%d' % (use_double,
          proc.nufft_proc.get_m(nf, y=y.astype(proc.real_type)),
          proc.nufft_proc.get_m(nf, y=tmpl.astype(proc.real_type))))
    for name, v in (('data', y), ('template', tmpl)):
        G = proc.compute_nufft(t, v, nf).astype(np.complex128)
        E = adjoint_dft(t, v, nf)
        lo, hi = slice(1, nf//2), slice(nf//2, nf)
        rms = np.sqrt(np.mean(np.abs(E)**2))
        for lab, s in (('k<nf/2', lo), ('k>=nf/2', hi)):
            err = np.abs(G[s]-E[s])
            print('  %-8s %-8s max|dG|/rms|E|=%.2e  median=%.2e  max rel=%.2e' % (
                name, lab, err.max()/rms, np.median(err)/rms, np.max(err/np.abs(E[s]))))
    # effect on SNR spectrum: GPU vs exact-DFT through the same host pipeline
    periods = np.exp(np.linspace(np.log(2), np.log(18), 40))
    yy = y + box(t, P, 1.3, dur, 0.01)
    epochs = np.linspace(0, 1, 8, endpoint=False)  # fraction; scaled per P below
    def spectrum(nufft, flat):
        out = np.zeros((len(periods), len(epochs)))
        Y = nufft(t, yy - yy.mean(), nf)
        from cuvarbase.nufft_lrt import _smoothed_periodogram
        if flat:
            psd = np.ones(nf)
        else:
            psd = _smoothed_periodogram(np.abs(Y)**2, 5); psd = np.maximum(psd, 1e-12*np.median(psd[psd>0]))
        w = np.ones(nf)
        for i, p in enumerate(periods):
            for j, e in enumerate(epochs):
                tm = proc._generate_template(t, p, e*p, dur, 1.0); tm -= tm.mean()
                T = nufft(t, tm, nf)
                out[i, j] = proc._compute_matched_filter_snr(Y, T, psd, w, 1e-12)
        return out
    for flat in (False, True):
        sg = spectrum(lambda t_, v, nf_: proc.compute_nufft(t_, v, nf_).astype(np.complex128), flat)
        se = spectrum(adjoint_dft, flat)
        print('  SNR spectrum GPU vs exact (flat_psd=%s): max|d|=%.3e  max|snr|=%.3f  corr=%.6f  argmax same=%s' % (
            flat, np.abs(sg-se).max(), np.abs(se).max(), np.corrcoef(sg.ravel(), se.ravel())[0,1],
            np.unravel_index(sg.argmax(), sg.shape) == np.unravel_index(se.argmax(), se.shape)))
