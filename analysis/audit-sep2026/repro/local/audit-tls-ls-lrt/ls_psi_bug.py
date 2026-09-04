"""Test hypothesis: LombScargleMemory shares precompute_psi (q1,q2) between the yw grid and the 2x larger w grid."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.memory import LombScargleMemory
from cuvarbase.utils import autofrequency
import cuvarbase.memory.lombscargle_memory as lsm

rng = np.random.RandomState(42)
N = 300; T = 100.0
t = np.sort(rng.rand(N) * T); dy = 0.05 * (1 + rng.rand(N))
y = 12.0 + 0.3 * np.cos(2 * np.pi * 3.1 * t - 0.4) + dy * rng.randn(N)

def gpu(freqs, patch=False, **kw):
    runkw = {k: kw.pop(k) for k in list(kw) if k in ('fast_grid',)}
    proc = LombScargleAsyncProcess(**kw)
    if patch:
        orig = LombScargleMemory.allocate_grids
        def allocate_grids(self, **kws):
            r = orig(self, **kws)
            if self.use_fft:
                # give the w-grid its OWN psi tables computed for its own grid size
                self.nfft_mem_w.precomp_psi = True
                self.nfft_mem_w.allocate_precomp_psi(n0=kws.get('n0', self.n0) if not self.buffered_transfer else self.n0_buffer)
            return r
        LombScargleMemory.allocate_grids = allocate_grids
    try:
        res = proc.run([(t, y, dy)], freqs=[freqs], **runkw); proc.finish()
        return np.array(res[0][1], dtype=np.float64)
    finally:
        if patch:
            LombScargleMemory.allocate_grids = orig

for nyq in (5, 20):
    freqs = autofrequency(t, samples_per_peak=5, nyquist_factor=nyq)
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    nf = len(freqs)
    print("nyquist_factor=%d nf=%d fmax=%.2f" % (nyq, nf, freqs.max()))
    for label, kw in [("default sigma=4", {}), ("sigma=2", dict(sigma=2)), ("use_double", dict(use_double=True)),
                      ("fast_grid=False (slow gridding, no psi)", dict(fast_grid=False)),
                      ("PATCHED: own psi for w-grid", dict(patch=True)),
                      ("PATCHED sigma=2", dict(patch=True, sigma=2)),
                      ("PATCHED sigma=2 m=6", dict(patch=True, sigma=2, m=6)),
                      ("PATCHED sigma=2 m=4", dict(patch=True, sigma=2, m=4)),
                      ("PATCHED sigma=2 m=3", dict(patch=True, sigma=2, m=3)),
                      ("PATCHED sigma=2 use_double", dict(patch=True, sigma=2, use_double=True))]:
        p = gpu(freqs, **kw)
        d = p - ref
        q = np.array_split(np.arange(nf), 4)
        print("  %-40s max|d|=%.2e  max|d| by freq-quartile: %s  rel@peak=%.2e"
              % (label, np.abs(d).max(), " ".join("%.1e" % np.abs(d[i]).max() for i in q), abs(d[np.argmax(ref)]) / ref.max()))
