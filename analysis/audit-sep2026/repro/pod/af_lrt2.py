"""NUFFT-LRT follow-ups: NaN in the reuse prototype; detector ordering with a fine epoch
grid; PSD self-whitening loss; Detector A/sequential null behaviour."""
import numpy as np, json, warnings, time
warnings.simplefilter('ignore')
from cuvarbase.cunfft import nfft_adjoint_async
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
rng = np.random.RandomState(21)
res = {}
def box(t, P, dur, depth, t0):
    ph = ((t - t0) / P) % 1.0; ph[ph > 0.5] -= 1
    return -depth * (np.abs(ph) < dur / (2*P))
n = 3000; T = 90.; P = 3.3; dur = 0.15; depth = 0.006; sig = 0.003; t0 = 1.1
t = np.sort(rng.rand(n) * T)
noise = sig * rng.randn(n)
lrt = NUFFTLRTAsyncProcess(sigma=2)
epochs = np.arange(0, P, dur/4)
periods = np.array([P, 2.9, 3.1, 3.5, 3.7, 4.1])
def evaluate(y, det='matched', **kw):
    s = lrt.run(t, y, periods, np.array([dur]), epochs=epochs, detector=det, **kw)
    smax = s.max(axis=(1, 2))
    return smax[0], np.max(smax[1:])
y_clean = 1.0 + box(t, P, dur, depth, t0) + noise
a, b = evaluate(y_clean)
print("white noise, matched: SNR@P=%.2f, best off-period=%.2f  (naive expectation depth/sig*sqrt(n_in)=%.1f)" % (a, b, depth/sig*np.sqrt(np.sum(box(t,P,dur,1,t0) < 0))))
# flat PSD (true noise) instead of the data-estimated one: quantifies self-whitening loss
nf = 2*n
a2, b2 = evaluate(y_clean, estimate_psd=False, psd=np.ones(nf))
print("white noise, matched, flat PSD supplied: SNR@P=%.2f off=%.2f   -> data-estimated PSD costs a factor %.2f in SNR" % (a2, b2, a2/a))
res['selfwhiten'] = dict(est=a, flat=a2, off_est=b, off_flat=b2)
# systematics: K=2 smooth basis, coefficients drawn from the prior used by Detector A
V = np.stack([np.sin(2*np.pi*t/30.), np.cos(2*np.pi*t/17.)], axis=1)
Cc = np.diag([0.004**2, 0.004**2])
c = np.array([0.004, -0.004])
y_sys = y_clean + V @ c
out = {}
for det, kw in [('matched', {}), ('sequential', dict(systematics_basis=V)), ('marginal', dict(systematics_basis=V, coeff_prior_cov=Cc))]:
    a, b = evaluate(y_sys, det, **kw); out[det] = [a, b]
    print("  with systematics: detector=%-10s SNR@P=%.2f best off=%.2f" % (det, a, b))
res['detectors'] = out
# same but with flat PSD supplied (isolates the systematics handling from PSD estimation)
out2 = {}
for det, kw in [('matched', {}), ('sequential', dict(systematics_basis=V)), ('marginal', dict(systematics_basis=V, coeff_prior_cov=Cc))]:
    a, b = evaluate(y_sys, det, estimate_psd=False, psd=np.ones(nf), **kw); out2[det] = [a, b]
    print("  flat PSD + systematics: detector=%-10s SNR@P=%.2f best off=%.2f" % (det, a, b))
res['detectors_flatpsd'] = out2
# NaN check on the memory-reuse prototype
tmpl = lrt._generate_template(t, P, 0.0, dur, 1.0); tmpl -= tmpl.mean()
a = np.asarray(lrt.compute_nufft(t, tmpl, nf)).copy()
nproc = lrt.nufft_proc
memr = nproc.allocate([(t.astype(np.float32), tmpl.astype(np.float32), nf)])[0]
memr.y[:] = tmpl.astype(np.float32)
nfft_adjoint_async(memr, nproc.function_tuple, block_size=nproc.block_size); memr.stream.synchronize()
b = memr.ghat_c.copy()
print("reuse prototype: NaNs in default=%d reuse=%d ; max|d| on finite=%.2e ; m default=%d" % (np.sum(~np.isfinite(a)), np.sum(~np.isfinite(b)), np.nanmax(np.abs(a-b)), memr.m))
json.dump(res, open('/workspace/scratch/af_lrt2.json', 'w'), indent=1)
