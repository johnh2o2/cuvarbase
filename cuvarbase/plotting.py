import matplotlib.pyplot as plt 
import imageio
import numpy as np
from . import pdm
from .utils import weights

def plot_spectrogram(ax, times, freqs, sgram):
    T, F = np.meshgrid(times, freqs)

    sgplot = ax.pcolormesh(T, F, sgram.T, vmin=0, vmax=1)

    return sgplot

def fast_pdm_spectrogram(t, y, err, freqs=None, nsplits=100, **kwargs):
    w = weights(err)
    specgram = pdm.PDMSpectrogram(t, y, w, freqs=freqs, **kwargs)

    times, freqs, sgram = specgram.spectrogram(nsplits=nsplits)
    
    f, ax = plt.subplots(figsize=(10, 5))
    
    spgraph = plot_spectrogram(ax, times, freqs, sgram)

    cax = f.add_axes([0.1, 0.9, 0.25, 0.05])

    f.colorbar(spgraph, cax=cax, orientation='horizontal')
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')

    return f, ax, spgraph

def plot_model(ax, t, y, w, freq, model, color=(1.0, 0.0, 0.0)):
    phi = np.linspace(0, 1, 100)

    phase = (t * freq) % 1.0
    rgba_colors = np.zeros((len(t),4))
    scplot = None
    for i in range(3):
        rgba_colors[:, i] = color[i]
    if len(t) > 0:
        rgba_colors[:, 3] = w / max(w)
        scplot = ax.scatter(phase, y, s=1, color=rgba_colors)

    return scplot, ax.plot(phi, model(phi), color='k')

def plot_powerspectrum(ax, freqs, spectral_power, ylabel='Power', xlabel='Frequency', 
                             **kwargs):
    plot = ax.plot(freqs, spectral_power, **kwargs)
    ax.set_xlim(min(freqs), max(freqs))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return plot


def plot_pdm_and_best_model(t, y, err, freqs=None, nbins=25, kind='binned_linterp', **kwargs):
    w = weights(err)

    f, (axfit, axpdm) = plt.subplots(1, 2, figsize=(10, 5))
    
    specgram = pdm.PDMSpectrogram(t, y, w, freqs=freqs, nbins=nbins, kind=kind, **kwargs)
    freqs, pdm_power = specgram.localized_spectrum(freqs=freqs)

    name = None
    if kind is 'binned_linterp':
        name = 'lin. interp.; %d bins'%(nbins)
    elif kind is 'binned_step':
        name = 'step; %d bins'%(nbins)
    elif kind is 'binless_gauss':
        name = 'binless; gaus. $\\sigma=UNKNOWN$'
    elif kind is 'binless_tophat':
        name = 'binless; tophat. $\\sigma=UNKNOWN$'

    plot_powerspectrum(axpdm, freqs, pdm_power, ylabel='PDM (%s)'%(name))

    best_freq = freqs[np.argmax(pdm_power)]

    model = pdm.binned_pdm_model(t, y, w, best_freq, nbins, linterp=(kind=='binned_linterp'))

    plot_model(axfit, t, y, w, best_freq, model, **kwargs)
    
    return freqs, pdm_power, model, f, axfit, axpdm

def plot_animated_spectrogram(specgram, **kwargs):

    times, freqs, sgram = specgram.spectrogram()

    best_freqs = [ freqs[np.argmax(sgram[i, :])] for i in range(len(times)) ]
    models = [ specgram.model(bf, time) for bf, time in zip(best_freqs, times) ]
    
    pngs = [ 'movies/figure_frame%04d.png'%(i+1) for i in range(len(times)) ]
    
    f = plt.figure(figsize=(10, 5))
    ax_sgram      = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_model      = plt.subplot2grid((2, 2), (1, 0))
    ax_spectrum   = plt.subplot2grid((2, 2), (1, 1))

    sgram_plot = plot_spectrogram(ax_sgram, times, freqs, sgram)

    cax = f.add_axes([0.1, 0.9, 0.25, 0.05])

    f.colorbar(sgram_plot, cax=cax, orientation='horizontal')

    plt.colorbar(sgram_plot)

    ax_sgram.set_xlim(min(times), max(times))
    ax_sgram.set_ylim(min(freqs), max(freqs))
    ax_sgram.set_xlabel('time')
    ax_sgram.set_ylabel('frequency')

    ymin, ymax = ax_sgram.get_ylim()
    line, = ax_sgram.plot([ times[0], times[0] ], [ ymin, ymax ], 'r-', lw=2)

    full_spectrum = specgram.localized_spectrum(freqs=freqs)
    
    for i in range(len(times)):
        ax_model.clear()
        ax_spectrum.clear()
        
        ax_sgram.set_ylim(ymin, ymax)
        ax_sgram.set_xlim(min(times), max(times))
        line.set_xdata([ times[i], times[i] ])

        full_pdm_plot = plot_powerspectrum(ax_spectrum, freqs, full_spectrum, 
                                           ylabel='periodogram', color='k', alpha=0.8)

        ax_spectrum.plot(freqs, sgram[i,:], color='r') 
        ax_spectrum.set_ylim(0, 1)

        T, Y, W = specgram.weighted_local_data(times[i])
        
        plot_model(ax_model, T, Y, W, best_freqs[i], specgram.model(best_freqs[i], time=times[i]))
        ax_model.set_xlim(0, 1)

        f.savefig(pngs[i], dpi=80)

    frames = []
    for fname in pngs:
        frames.append(imageio.imread(fname))
    imageio.mimsave('movie.gif', frames)
