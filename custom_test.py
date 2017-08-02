#!/home/jah5/anaconda3/envs/pycu2/bin/python

from cuvarbase.lombscargle import LombScargleAsyncProcess
import pycuda.driver as cuda
import numpy as np
from time import time
from astropy.stats.lombscargle import LombScargle

samples_per_peak=10
nyquist_factor=10

def random_data(ndata=3000, T=7, freq=1.4, ybar=12, sigma=0.01, ampl=0.1):
	# random observation times (1 year baseline)
	t = 365 * T * np.random.rand(ndata)

	# some signal (10 day period, 0.1 amplitude)
	y = ybar + ampl * np.cos(2 * np.pi * t * freq)

	# data uncertainties (0.01)
	dy = sigma * np.ones_like(y)

	# add noise to observations
	y += dy * np.random.randn(len(t))

	return t, y, dy

# start an asynchronous process
ls_proc = LombScargleAsyncProcess()

data = [random_data()]

results = ls_proc.run(data, samples_per_peak=samples_per_peak, nyquist_factor=nyquist_factor)

freqs_g, pows_g = results[0]
freqs_c = freqs_g
pows_c = LombScargle(*(data[0])).power(freqs_g)

import matplotlib.pyplot as plt

plt.plot(freqs_g, pows_g, color='r')
plt.plot(freqs_c, pows_c, color='k')

plt.show()
