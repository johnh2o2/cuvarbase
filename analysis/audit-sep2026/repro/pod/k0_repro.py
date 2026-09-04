"""Minimal independent reproduction of auditor finding 13 (NFFT grid shaves k0 off).
Compare LombScargleAsyncProcess FFT path (default sigma=4, m=8) vs astropy float64 GLS
and vs the GPU direct-sums path across bands with increasing k0/nf; then apply the
B-only fix (grid = sigma*(nf+k0), sigma*2*(nf+k0)) via monkeypatch and re-measure."""
import sys, numpy as np
sys.path.insert(0, '/workspace/scratch')
from astropy.timeseries import LombScargle
import cuvarbase.memory.lombscargle_memory as lsm
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
import pycuda.gpuarray as gpuarray

def grid(fmin, fmax, T, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))

def make_lc(N, T, f0, seed):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.rand(N)) * T
    dy = 0.1 * np.ones(N)
    y = 12.0 + 0.3 * np.cos(2 * np.pi * f0 * t - 0.3) + dy * rng.randn(N)
    return t, y, dy

def gpu(proc, t, y, dy, freqs, **kw):
    r = proc.run([(t, y, dy)], freqs=freqs, **kw); proc.finish()
    return np.array(r[0][1][:len(freqs)], float)

_orig_alloc = lsm.LombScargleMemory.allocate_grids
def alloc_fixed(self, **kwargs):
    k0 = kwargs.get('k0', self.k0)
    n0 = kwargs.get('n0', self.n0)
    if self.buffered_transfer:
        n0 = kwargs.get('n0_buffer', self.n0_buffer)
    self.nf = kwargs.get('nf', self.nf)
    if self.use_fft:
        if self.nfft_mem_yw.precomp_psi:
            self.nfft_mem_yw.allocate_precomp_psi(n0=n0)
        self.nfft_mem_w.precomp_psi = False
        self.nfft_mem_w.q1 = self.nfft_mem_yw.q1
        self.nfft_mem_w.q2 = self.nfft_mem_yw.q2
        self.nfft_mem_w.q3 = self.nfft_mem_yw.q3
        fft_size = self.nharmonics * (self.nf + k0)
        self.nfft_mem_yw.allocate_grid(nf=fft_size)        # was fft_size - k0
        self.nfft_mem_w.allocate_grid(nf=2 * fft_size)     # was 2*fft_size - k0
    self.lsp_g = gpuarray.zeros(self.nf, dtype=self.real_type)
    return self

T = 365.0
bands = [(1.0/(5*T), 20.0), (0.5, 20.0), (5.0, 20.0), (5.0, 15.0), (5.0, 10.0), (6.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
for fixed in (False, True):
    lsm.LombScargleMemory.allocate_grids = alloc_fixed if fixed else _orig_alloc
    print("\n########## allocate_grids %s ##########" % ("FIXED (grid=sigma*(nf+k0))" if fixed else "AS SHIPPED (grid=sigma*nf)"))
    for dbl in (False, True):
        for sigma in (4, 2):
            proc = LombScargleAsyncProcess(use_double=dbl, sigma=sigma, m=8, autoset_m=False)
            for (a, b) in bands:
                f0 = a + 0.9 * (b - a)   # signal near the top of the band
                t, y, dy = make_lc(300, T, f0, seed=3)
                fr = grid(a, b, T)
                k0, nf = get_k0(fr), len(fr)
                ref = LombScargle(t, y, dy).power(fr, method='cython')
                p = gpu(proc, t, y, dy, fr)
                d = np.abs(ref - p)
                # top-mode fraction of the yw FFT
                ng = sigma * (nf + k0) if fixed else sigma * nf
                frac = (k0 + nf - 1) / ng
                print("dbl=%d sigma=%d band %5.2f-%5.2f k0=%6d nf=%6d k0/nf=%.2f topfrac=%.3f | maxabs=%.2e medabs=%.2e | "
                      "argmax ref %.4f gpu %.4f %s | max power gpu=%.3e" % (
                      dbl, sigma, a, b, k0, nf, k0 / nf, frac, d.max(), np.median(d),
                      fr[np.argmax(ref)], fr[np.argmax(p)], "OK " if np.argmax(ref) == np.argmax(p) else "BAD", p.max()), flush=True)
            del proc

# direct sums sanity for the worst band (as shipped)
lsm.LombScargleMemory.allocate_grids = _orig_alloc
proc = LombScargleAsyncProcess(use_double=False, sigma=4, m=8, autoset_m=False)
t, y, dy = make_lc(300, T, 29.0, seed=3)
fr = grid(20.0, 30.0, T)
ref = LombScargle(t, y, dy).power(fr, method='cython')
p = gpu(proc, t, y, dy, fr, use_fft=False)
print("\ndirsum band 20-30 float32: maxabs=%.2e argmax ref %.4f gpu %.4f" % (np.abs(ref - p).max(), fr[np.argmax(ref)], fr[np.argmax(p)]))
# documented kwargs path: minimum_frequency / maximum_frequency -> autofrequency
r = proc.run([(t, y, dy)], minimum_frequency=20.0, maximum_frequency=30.0); proc.finish()
fr2, p2 = np.asarray(r[0][0]), np.asarray(r[0][1][:len(r[0][0])], float)
ref2 = LombScargle(t, y, dy).power(fr2, method='cython')
print("run(minimum_frequency=20, maximum_frequency=30): k0=%d nf=%d maxabs=%.2e maxpow=%.2e argmax ref %.4f gpu %.4f" % (
    get_k0(fr2), len(fr2), np.abs(ref2 - p2).max(), p2.max(), fr2[np.argmax(ref2)], fr2[np.argmax(p2)]))
# multiharmonic path on k0/nf=1 band, as shipped
procH = LombScargleAsyncProcess(use_double=True, sigma=4, m=8, autoset_m=False, nharmonics=2)
t, y, dy = make_lc(300, T, 9.5, seed=3)
fr = grid(5.0, 10.0, T)
pH = gpu(procH, t, y, dy, fr)
pD = gpu(procH, t, y, dy, fr, use_fft=False)
print("nharmonics=2 band 5-10 (k0/nf=1) fft vs dirsum: maxabs=%.2e argmax dirsum %.4f fft %.4f" % (np.abs(pH - pD).max(), fr[np.argmax(pD)], fr[np.argmax(pH)]))
