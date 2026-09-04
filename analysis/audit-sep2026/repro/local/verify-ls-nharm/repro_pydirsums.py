import numpy as np
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_direct_sums
rng = np.random.RandomState(7); N = 250
t = np.sort(rng.rand(N)) * 80.0; f0 = 0.9
y = 10 + 0.4*np.sin(2*np.pi*f0*t) + 0.4*np.sin(2*np.pi*2*f0*t + 1.0) + 0.05*rng.randn(N)
dy = 0.05*np.ones(N); T = t.max()-t.min(); df = 1.0/(5*T); k0 = 5; nf = 600
freqs = df*(k0+np.arange(nf)); w = dy**-2; w /= w.sum(); ybar = np.dot(w,y); YY = np.dot(w,(y-ybar)**2)
ref = {H: lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=H) for H in (1,2)}
proc = LombScargleAsyncProcess(use_double=True, nharmonics=2)
r = proc.run([(t,y,dy)], freqs=freqs, use_fft=False, python_dir_sums=True, fast_grid=False); proc.finish()
p = np.array(r[0][1][:nf], float)
for H in (1,2):
    print("python_dir_sums=True, proc nharmonics=2 vs hostref H=%d: maxabs=%.2e peak(ref)=%.4f peak(p)=%.4f" % (H, np.max(np.abs(ref[H]-p)), ref[H].max(), p.max()))
