import skcuda.fft
import cuvarbase.lombscargle as gls
import numpy as np
import matplotlib.pyplot as plt


t = np.sort(np.random.rand(300))
y = 1 + np.cos(2 * np.pi * 100 * t - 0.1)
dy = 0.1 * np.ones_like(y)
y += dy * np.random.randn(len(t))

# Set up LombScargleAsyncProcess (compilation, etc.)
proc = gls.LombScargleAsyncProcess()

# Run on single lightcurve
result = proc.run([(t, y, dy)])

# Synchronize all cuda streams
proc.finish()

# Read result!
freqs, ls_power = result[0]

############
# Plotting #
############

f, ax = plt.subplots()
ax.set_xscale('log')

ax.plot(freqs, ls_power)
ax.set_xlabel('Frequency')
ax.set_ylabel('Lomb-Scargle')
plt.show()