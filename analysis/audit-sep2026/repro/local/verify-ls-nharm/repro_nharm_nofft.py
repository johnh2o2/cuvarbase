"""Minimal reproduction: LombScargleAsyncProcess(nharmonics=H).run(..., use_fft=False)
vs the host multiharmonic reference lomb_scargle_direct_sums(nharms=H) and vs H=1."""
import numpy as np
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_direct_sums

rng = np.random.RandomState(7)
N = 250
t = np.sort(rng.rand(N)) * 80.0
f0 = 0.9
# strongly non-sinusoidal signal so H=1 and H=2 differ a lot
y = 10 + 0.4*np.sin(2*np.pi*f0*t) + 0.4*np.sin(2*np.pi*2*f0*t + 1.0) + 0.05*rng.randn(N)
dy = 0.05*np.ones(N)
T = t.max() - t.min()
df = 1.0/(5*T); k0 = 5; nf = 1500
freqs = df*(k0 + np.arange(nf))
w = dy**-2; w /= w.sum(); ybar = np.dot(w, y); YY = np.dot(w, (y-ybar)**2)

ref = {H: lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=H) for H in (1, 2, 3)}

def stats(tag, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    print("%-52s maxabs=%.2e  argmax(a)=%d argmax(b)=%d  peak(a)=%.4f peak(b)=%.4f"
          % (tag, np.max(np.abs(a-b)), a.argmax(), b.argmax(), a.max(), b.max()))

for use_double in (True, False):
    for H in (2, 3):
        proc = LombScargleAsyncProcess(use_double=use_double, nharmonics=H)
        for use_fft in (True, False):
            r = proc.run([(t, y, dy)], freqs=freqs, use_fft=use_fft, fast_grid=False)
            proc.finish()
            p = np.array(r[0][1][:nf], float)
            tag = "H=%d dbl=%s use_fft=%s" % (H, use_double, use_fft)
            stats(tag + " vs hostref H=%d" % H, ref[H], p)
            stats(tag + " vs hostref H=1", ref[1], p)
        del proc

# also: batched_run_const_nfreq with use_fft=False, H=2
proc = LombScargleAsyncProcess(use_double=True, nharmonics=2)
r = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs, use_fft=False, fast_grid=False)
proc.finish()
p = np.array(r[0][1][:nf], float)
stats("batched_run_const_nfreq H=2 use_fft=False vs ref H=2", ref[2], p)
stats("batched_run_const_nfreq H=2 use_fft=False vs ref H=1", ref[1], p)
