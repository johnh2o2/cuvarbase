import cuvarbase.ce as ce
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def phase(t, freq, phi0=0.):
    phi = (t * freq - phi0)
    phi -= np.floor(phi)

    return phi


def sine_model(t, freq, amp=1., y0=0.0, phi0=0.5):
    return y0 + amp * np.sin((t * freq - phi0) * 2 * np.pi)


def transit_model(t, freq, y0=0.0, delta=1., q=0.01, phi0=0.5):
    phi = phase(t, freq, phi0=phi0)
    transit = phi < q

    y = y0 * np.ones_like(t)
    y[transit] -= delta
    return y


def data(ndata=100, baseline=1, freq=10, sigma=1.,
         model=transit_model, **kwargs):
    t = baseline * np.sort(np.random.rand(ndata))
    y = model(t, freq, **kwargs)
    dy = sigma * np.ones_like(t)

    y += dy * np.random.randn(len(t))

    return t, y, dy


def plot_ce_bins(ax, t, y, dy, freq, ce_proc):
    ax.set_xlim(0, 1)

    y0 = min(y)
    yrange = max(y) - y0

    # Phase-fold the data at the trial frequency
    phi = phase(t, freq)

    # Bin the data
    phi_bins = np.floor(phi * ce_proc.phase_bins).astype(np.int)

    yi = ce_proc.mag_bins * (y - y0)/yrange
    mag_bins = np.floor(yi).astype(np.int)

    bins = [[sum((phi_bins == i) & (mag_bins == j))
             for j in range(ce_proc.mag_bins)]
            for i in range(ce_proc.phase_bins)]
    bins = np.array(bins).astype(np.float)

    # Convert to N(bin) / Ntotal
    bins /= np.sum(bins.ravel())

    # The fraction of data that fall within a given phase bin
    p_phi = [np.sum(bins[i]) for i in range(ce_proc.phase_bins)]

    # fractional width of the (magnitude) bins
    dm = float(1 + ce_proc.mag_overlap) / ce_proc.mag_bins
    dphi = float(1 + ce_proc.phase_overlap) / ce_proc.phase_bins
    dY = yrange * dm

    # Compute conditional entropy contribution from each of the bins
    dH = [[bins[i][j] * np.log(dm * p_phi[i] / bins[i][j])
           if bins[i][j] > 0 else 0.
           for j in range(ce_proc.mag_bins)]
          for i in range(ce_proc.phase_bins)]

    dH = np.array(dH)

    extent = [0, 1, min(y), max(y)]

    # Mask out the unoccupied bins
    dH = np.ma.masked_where(dH == 0, dH)

    palette = copy(plt.cm.GnBu_r)
    palette.set_bad('w', 0.)

    # Draw gridlines
    for i in range(ce_proc.phase_bins + 1):
        ax.axvline(0 + i * dphi, ls=':', color='k',
                   alpha=0.5, zorder=95)

    for i in range(ce_proc.mag_bins + 1):
        ax.axhline(min(y) + i * dY, ls=':', color='k',
                   alpha=0.5, zorder=95)

    # Plot the conditional entropy
    cplot = ax.imshow(dH.T, cmap=palette, extent=extent,
                      aspect='auto', origin='lower',
                      alpha=0.5, zorder=90)

    # Plot the data
    ax.scatter(phi, y, c='k', s=1, alpha=1, zorder=100)

    return cplot

# Set up the signal parameters
freq = 0.1
signal_params = dict(y0=10.,
                     freq=freq,
                     sigma=0.01,
                     ndata=100,
                     baseline=365.,
                     amp=0.1,
                     phi0=0.,
                     model=sine_model)

# Generate data
t, y, dy = data(**signal_params)

# Start GPU process for conditional entropy
# (this does things like compiling the kernel,
#  setting parameter values, etc.)
proc = ce.ConditionalEntropyAsyncProcess()

# Set frequencies
df = 1. / (2 * signal_params['baseline'])
fmin = 2. / signal_params['baseline']
fmax = 50 * len(t) * df

nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

####################
# Run the process! #
####################

# Data is sent in list of tuples (in case we want
# to run CE on more than one lightcurve)
data = [(t, y, dy)]

# The large_run function is an alternative to the run
# function if the frequency grid & binning array is too
# large to fit in GPU memory.
try:
    results = proc.run(data, freqs=freqs)
except:
    results = proc.large_run(data, freqs=freqs, max_memory=1e8)

proc.finish()

# The results come back as [(freqs, CE), ...] for
# each element of the data list. In this case, there is only
# one lightcurve.
frq, p = results[0]

# Find the best frequency (that *minimizes* the conditional entropy)
f_best = frq[np.argmin(p)]


#####################
# Plot the results! #
#####################

f, (ax_ce, ax_bin) = plt.subplots(1, 2, figsize=(8, 4))
ax_ce.plot(frq, p)
ax_ce.set_xlabel('freq.', fontsize=15)
ax_ce.set_ylabel('Conditional Entropy ($H(f)$)', fontsize=15)
ax_ce.set_xscale('log')
ax_ce.axvline(freq, color='k', ls=':')
ax_ce.axvline(f_best, color='r', ls=':')

cplot = plot_ce_bins(ax_bin, t, y, dy, freq, proc)
cbar = f.colorbar(cplot)
cbar.ax.set_title('$H(\\phi, m)$')
ax_bin.set_xlabel('$\\phi$', fontsize=15)
ax_bin.set_ylabel('$m$', fontsize=15)
ax_bin.set_title('$f = {\\rm argmin}_{f}(H(f))$')

f.tight_layout()
plt.show()
