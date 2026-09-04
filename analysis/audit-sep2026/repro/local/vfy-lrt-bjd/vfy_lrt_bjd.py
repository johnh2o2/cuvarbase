"""Verifier reproduction: NUFFT-LRT under BJD-scale times, all 3 detectors,
plus the proposed fix applied by monkeypatch (tree untouched)."""
import warnings, numpy as np
warnings.simplefilter('ignore')
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
from cuvarbase.utils import subtract_epoch

rng = np.random.default_rng(1)
N, T = 600, 60.0
t = np.sort(rng.uniform(0, T, N))
P0, dur, depth, sig, e0 = 5.3, 0.22, 0.01, 0.003, 1.2
ph = np.fmod(t - e0, P0) / P0; ph[ph < 0] += 1; ph[ph > 0.5] -= 1
y = 1.0 - depth * (np.abs(ph) <= dur / (2 * P0)) + sig * rng.standard_normal(N)
V = np.stack([np.sin(2 * np.pi * t / T), (t - T / 2) / T], 1)
y = y + 0.002 * V[:, 0] + 0.003 * V[:, 1]
print('true: P=%.3f dur=%.2f e0=%.2f  n_in_transit=%d' % (P0, dur, e0, int((np.abs(ph) <= dur/(2*P0)).sum())))
print('float32 spacing at 2457000:', np.spacing(np.float32(2457000.0)), ' at 2000:', np.spacing(np.float32(2000.0)))

periods = np.round(np.arange(4.0, 6.61, 0.1), 3); durations = np.array([0.15, 0.22]); epochs = np.linspace(0, P0, 16, endpoint=False)
iP = np.argmin(abs(periods - P0)); iD = 1; iE = np.argmin(abs(epochs - e0))
DETS = [('matched', {}),
        ('marginal', dict(systematics_basis=V, coeff_prior_cov=np.eye(2) * 1e-4)),
        ('sequential', dict(systematics_basis=V))]

def summarize(tag, base, r):
    bi = np.unravel_index(np.argmax(r), r.shape)
    c = np.corrcoef(base.ravel(), r.ravel())[0, 1]
    print('  %-42s bestP=%.3f dur=%.2f ep=%.2f maxSNR=%6.2f SNR@truth=%6.2f corr=%.4f rel=%.2e' % (
        tag, periods[bi[0]], durations[bi[1]], epochs[bi[2]], r.max(), r[iP, iD, iE], c,
        np.abs(r - base).max() / np.abs(base).max()))

def run(proc, off, det, kw):
    return proc.run(t + off, y, periods, durations, epochs=epochs + off, detector=det, **kw)

print('\n=== UNPATCHED (tree as shipped) ===')
for dbl in (False, True):
    proc = NUFFTLRTAsyncProcess(use_double=dbl)
    for det, kw in DETS:
        base = run(proc, 0.0, det, kw)
        summarize('use_double=%s %s off=0 (ref)' % (dbl, det), base, base)
        for off in (2000.0, 2457000.0):
            summarize('use_double=%s %s off=%g' % (dbl, det, off), base, run(proc, off, det, kw))

# --- proposed fix, applied only in this process ---
_orig = NUFFTLRTAsyncProcess.run
def _fixed_run(self, t, y, periods, durations=None, epochs=None, **kw):
    t64, t0 = subtract_epoch(np.asarray(t, dtype=np.float64))
    if epochs is not None:
        epochs = np.asarray(epochs, dtype=np.float64) - t0
    return _orig(self, t64, y, periods, durations, epochs=epochs, **kw)

print('\n=== PATCHED run(): subtract_epoch in float64 before the cast; epochs shifted by same t0 ===')
proc = NUFFTLRTAsyncProcess(use_double=False)
for det, kw in DETS:
    base_unpatched = run(proc, 0.0, det, kw)
    NUFFTLRTAsyncProcess.run = _fixed_run
    base = run(proc, 0.0, det, kw)
    summarize('fixed %s off=0 vs UNPATCHED off=0' % det, base_unpatched, base)
    for off in (2000.0, 2457000.0):
        summarize('fixed %s off=%g vs fixed off=0' % (det, off), base, run(proc, off, det, kw))
    # default-path impact: data whose min(t) is not in [0,1) -> fix subtracts floor(min t)
    NUFFTLRTAsyncProcess.run = _orig
    u = run(proc, 10.3, det, kw)
    NUFFTLRTAsyncProcess.run = _fixed_run
    f = run(proc, 10.3, det, kw)
    summarize('default-path impact %s off=10.3 fixed vs unpatched' % det, u, f)
    NUFFTLRTAsyncProcess.run = _orig
