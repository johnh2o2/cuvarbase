import pytest
from ..wavelet import Wavelet
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def data(T=10, N=3000, y0=10., sigma=0.01, dtau=20., tau=0.5, period=5.):
	t = T * np.sort(np.random.rand(N))
	y = y0 + np.exp(-0.5 * ((t - tau * T) / dtau)**2) * np.cos(2 * np.pi * t / period)
	y += sigma * np.random.randn(N)
	dy = sigma * np.ones_like(y)

	return t, y, dy

@pytest.mark.skip(reason='wavelets are not yet fully supported')
def test_wavelet():
	
	t, y, dy = data()

	wl = Wavelet(t, y, dy)#.run()

	specgram = wl.run()
	taus = wl.taus
	freqs = wl.freqs

	taus_plt = np.linspace(min(t), max(t), max([ len(taus_) for taus_ in taus ]))
	freqs_plt = np.linspace(min(freqs, max(freqs), 10 * len(freqs)))

	norm = lambda arr : (arr - min(arr)) / (max(arr) - min(arr))
	
	spec_plt = mlab.griddata(norm(taus), norm(freqs), specgram, norm(taus_plt), norm(freqs_plt))
	plt.figure()
	plt.pcolormesh(taus_plt, freqs_plt, spec_plt)
	plt.axis([min(t), max(t), min(freqs), max(freqs)])

	plt.show()