import numpy as np
import cuvarbase.ce as ce
import matplotlib.pyplot as plt
from astropy.stats.lombscargle import LombScargle

N = 10000
snr = 150
freq = 50
spp = 5
npb = 5

min_mag_bins = 5
min_phase_bins = 10

nbins = int(np.sqrt(N / float(npb)))

mag_bins = max([min_mag_bins, nbins])
phase_bins = max([min_phase_bins, nbins])

t = np.sort(np.random.rand(100))
y = snr / np.sqrt(N) * np.cos(2 * np.pi * freq * t)

print snr / np.sqrt(N)
dy = np.ones_like(y)

y += dy * np.random.randn(len(t))

baseline = max(t) - min(t)
df = 1./(spp * baseline)
fmin = spp * df
fmax = 50 * len(t) * df

print phase_bins, mag_bins

nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

gls = LombScargle(t, y, dy).power(freqs)
proc = ce.ConditionalEntropyAsyncProcess(mag_bins=mag_bins, phase_bins=phase_bins)

r1 = proc.large_run([(t, y, dy)], freqs=freqs, max_memory=1e7)
r2 = proc.large_run([(t, y, dy)], freqs=freqs, balanced_mag_bins=True, compute_log_prob=False, max_memory=1e7)

f1, p = r1[0]
f2, lp = r2[0]

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(f1, p)


#p = np.exp(lp / (phase_bins * mag_bins))
#frac = (df * baseline / (phase_bins))

ax2.plot(f2, lp)
#ax2.plot(f2, lp / (phase_bins * mag_bins))

ax3.plot(freqs, gls)

for ax in [ax1, ax2, ax3]:
    ax.axvline(freq, ls=':', color='k', alpha=0.5)
    # ax.set_xscale('log')
plt.show()


