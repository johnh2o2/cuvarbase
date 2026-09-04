"""Validate the proposed fix WITHOUT editing the install: monkeypatch _mh_power_from_spectra so the
H>1 FFT path passes reg_kwargs=dict(amplitude_priors=memory.amplitude_prior)."""
import numpy as np
import cuvarbase.lombscargle as L
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_direct_sums

_orig = L._mh_power_from_spectra
CUR = {'s': None}
def patched(sw, syw, k0, nharms, nf, YY, reg_kwargs=None):
    if CUR['s'] is not None:
        reg_kwargs = dict(amplitude_priors=CUR['s'])
    return _orig(sw, syw, k0, nharms, nf, YY, reg_kwargs=reg_kwargs)
L._mh_power_from_spectra = patched

rng = np.random.RandomState(7)
N = 250; t = np.sort(rng.rand(N)) * 80.0; f0 = 0.9
y = 5 + 0.4*np.sin(2*np.pi*f0*t) + 0.25*np.sin(2*np.pi*2*f0*t + 0.7) + 0.1*rng.randn(N)
dy = 0.1 * np.exp(0.2*rng.randn(N))
T = t.max() - t.min(); df = 1.0/(5*T); k0 = 5; nf = 400
freqs = df*(k0 + np.arange(nf))
w = dy**-2; w /= w.sum(); ybar = np.dot(w, y); YY = np.dot(w, (y-ybar)**2)

for s in (0.3, 0.05):
    for H in (2, 3):
        cpu_reg = lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=H, amplitude_priors=s)
        cpu_unreg = lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=H)
        for use_double in (True, False):
            proc = LombScargleAsyncProcess(use_double=use_double, sigma=4, m=8, autoset_m=False, nharmonics=H)
            CUR['s'] = s
            r = proc.run([(t, y, dy)], freqs=freqs, amplitude_prior=s, fast_grid=False); proc.finish()
            p = np.array(r[0][1][:nf], float)
            print("FIXED  s=%.2f H=%d dbl=%-5s vs CPU regularised direct sums: %.2e   vs unreg: %.2e   (mem.amplitude_prior=%r)"
                  % (s, H, use_double, np.max(np.abs(p-cpu_reg)), np.max(np.abs(p-cpu_unreg)), s))
            CUR['s'] = None
            del proc
# also: with the patch active but amplitude_prior=None, nothing changes
proc = LombScargleAsyncProcess(use_double=True, sigma=4, m=8, autoset_m=False, nharmonics=2)
r = proc.run([(t, y, dy)], freqs=freqs, fast_grid=False); proc.finish()
p = np.array(r[0][1][:nf], float)
cpu_unreg = lomb_scargle_direct_sums(t, w*y, w, freqs, YY, nharms=2)
print("no-prior H=2 dbl vs CPU unreg direct sums: %.2e" % np.max(np.abs(p-cpu_unreg)))
