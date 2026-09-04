"""Hypothesis: in DOUBLE mode fast_gaussian_grid uses float32 floorf() for the grid index while precompute_psi uses double modflt() -> inconsistent placement.
Test by compiling a modified copy of cunfft.cu (kernel text patched in memory; the on-disk tree is untouched)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
import cuvarbase.cunfft as cunfft_mod
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.memory import LombScargleMemory
rng = np.random.RandomState(3); T = 1095.0
N = 1000; t = np.sort(rng.rand(N) * T); dy = 0.02 * (1 + rng.rand(N)); y = 15 + 0.1 * np.sin(2 * np.pi * t / 0.37) + dy * rng.randn(N)
orig_ag = LombScargleMemory.allocate_grids
def allocate_grids(self, **kws):
    r = orig_ag(self, **kws)
    if self.use_fft:
        self.nfft_mem_w.precomp_psi = True; self.nfft_mem_w.allocate_precomp_psi(n0=self.n0_buffer if self.buffered_transfer else kws.get('n0', self.n0))
    return r
LombScargleMemory.allocate_grids = allocate_grids
orig_reader = cunfft_mod._module_reader
def patched_reader(fname, cpp_defs=None):
    txt = orig_reader(fname, cpp_defs)
    assert 'floorf(ng * xval - m)' in txt
    return txt.replace('floorf(ng * xval - m)', 'floor(ng * xval - m)')
for nf in (3750, 365000):
    df = 100.0 / nf; freqs = df * (1 + np.arange(nf))
    pr = LombScargleAsyncProcess(use_double=True); p0 = np.array(pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)[0][1], np.float64)
    cunfft_mod._module_reader = patched_reader
    pr = LombScargleAsyncProcess(use_double=True); p1 = np.array(pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)[0][1], np.float64)
    cunfft_mod._module_reader = orig_reader
    pr = LombScargleAsyncProcess(use_double=False); p2 = np.array(pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)[0][1], np.float64)
    ipk = int(np.argmax(p0)); sl = slice(max(0, ipk - 2000), min(nf, ipk + 2000)); ref = LombScargle(t, y, dy).power(freqs[sl], method='cython')
    slt = slice(nf - 2000, nf); reft = LombScargle(t, y, dy).power(freqs[slt], method='cython')
    for lab, p in (("f64 shipped (floorf)", p0), ("f64 floorf->floor", p1), ("f32 shipped", p2)):
        print("nf=%7d %-22s @peak max|d|=%.2e rel=%.1e | top-band max|d|=%.2e rel-to-local-max=%.1e" % (nf, lab, np.abs(p[sl] - ref).max(), np.abs(p[sl] - ref).max() / ref.max(), np.abs(p[slt] - reft).max(), np.abs(p[slt] - reft).max() / reft.max()))
