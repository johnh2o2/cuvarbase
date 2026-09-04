"""LS use_double at large nf: error growth and frequency-shift check; setdata line timing."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
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
for nf in (3750, 50000, 365000, 1000000):
    df = 100.0 / nf; freqs = df * (1 + np.arange(nf))
    out = {}
    for dbl in (False, True):
        pr = LombScargleAsyncProcess(use_double=dbl); r = pr.batched_run_const_nfreq([(t, y, dy)], freqs=freqs); out[dbl] = np.array(r[0][1], np.float64)
    ipk = int(np.argmax(out[False])); sl = slice(max(0, ipk - 2000), min(nf, ipk + 2000)); ref = LombScargle(t, y, dy).power(freqs[sl], method='cython')
    slt = slice(nf - 2000, nf); reft = LombScargle(t, y, dy).power(freqs[slt], method='cython')
    for dbl in (False, True):
        p = out[dbl]
        # lag between GPU and reference around the peak
        xc = [np.corrcoef(p[sl][2:-2], np.roll(ref, lag)[2:-2])[0, 1] for lag in range(-3, 4)]
        print("nf=%7d %s: @peak max|d|=%.2e (rel %.1e) best lag=%+d ; top-band max|d|=%.2e (rel-to-local-max %.1e) ; argmax f32 vs f64 = %d vs %d"
              % (nf, 'f64' if dbl else 'f32', np.abs(p[sl] - ref).max(), np.abs(p[sl] - ref).max() / ref.max(), int(np.argmax(xc)) - 3, np.abs(p[slt] - reft).max(), np.abs(p[slt] - reft).max() / reft.max(), np.argmax(out[False]), np.argmax(out[True])))
print("\nsetdata line timing (min of 7):")
from cuvarbase.memory.lombscargle_memory import weights
for N2 in (50000, 65000):
    t2 = np.sort(rng.rand(N2) * T); dy2 = 0.02 * (1 + rng.rand(N2)); y2 = 15 + dy2 * rng.randn(N2)
    mem = LombScargleMemory(4, None, 8, k0=1, buffered_transfer=True, n0_buffer=N2, use_fft=False); mem.allocate(nf=10)
    def tm(fn):
        xs = []
        for i in range(7):
            t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
        return 1e3 * min(xs)
    w = weights(dy2); ybar = np.dot(y2, w); yw = np.multiply(w, y2 - ybar)
    print("  N=%d: setdata %.1f ms | weights %.1f | dot %.2f | astype x3 %.2f | pinned copy x3 %.2f | min+max builtin %.1f"
          % (N2, tm(lambda: mem.setdata(t=t2, y=y2, dy=dy2)), tm(lambda: weights(dy2)), tm(lambda: np.dot(y2, w)), tm(lambda: (t2.astype(np.float32), yw.astype(np.float32), w.astype(np.float32))),
             tm(lambda: (mem.t.__setitem__(slice(0, N2), t2[:N2].astype(np.float32)), mem.yw.__setitem__(slice(0, N2), yw[:N2].astype(np.float32)))), tm(lambda: (min(t2), max(t2)))))
