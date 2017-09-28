import skcuda.fft
import cuvarbase.lombscargle as gls
import numpy as np
import matplotlib.pyplot as plt

nlcs = 9

def lightcurve(freq=100, ndata=300):
        t = np.sort(np.random.rand(ndata))
        y = 1 + np.cos(2 * np.pi * freq * t - 0.1)
        dy = 0.1 * np.ones_like(y)
        y += dy * np.random.randn(len(t))
        return t, y, dy

freqs = 200 * np.random.rand(nlcs)
data = [lightcurve(freq=freq) for freq in freqs]

# Set up LombScargleAsyncProcess (compilation, etc.)
proc = gls.LombScargleAsyncProcess()

# Run on batch of lightcurves
results = proc.batched_run_const_nfreq(data)

# Synchronize all cuda streams
proc.finish()

############
# Plotting #
############
max_n_cols = 4
ncols = max([1, min([int(np.sqrt(nlcs)), max_n_cols])])
nrows = int(np.ceil(float(nlcs) / ncols))
f, axes = plt.subplots(nrows, ncols,
                       figsize=(3 * ncols, 3 * nrows))

for (frqs, ls_power), ax, freq in zip(results,
                                      np.ravel(axes),
                                      freqs):
        ax.set_xscale('log')
        ax.plot(frqs, ls_power)
        ax.axvline(freq, ls=':', color='r')

f.text(0.05, 0.5, "Lomb-Scargle", rotation=90,
       va='center', ha='right', fontsize=20)
f.text(0.5, 0.05, "Frequency",
       va='top', ha='center', fontsize=20)


for i, ax in enumerate(np.ravel(axes)):
        if i >= nlcs:
                ax.axis('off')
f.tight_layout()
f.subplots_adjust(left=0.1, bottom=0.1)
plt.show()