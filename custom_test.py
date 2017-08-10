#!/home/jah5/anaconda3/envs/pycu2/bin/python

from cuvarbase.lombscargle import LombScargleAsyncProcess
import pycuda.driver as cuda
import numpy as np
from time import time
from astropy.stats.lombscargle import LombScargle

from cuvarbase.tests.test_lombscargle import test_run_batch_const_nfreq
import sys

test_run_batch_const_nfreq(make_plot=True, use_fft=True)
sys.exit()
samples_per_peak=5
nyquist_factor=5

def random_data(ndata=3000, T=7, freq=.4, ybar=12, sigma=0.01, ampl=0.1):
	# random observation times (1 year baseline)
	t = np.sort(365 * T * np.random.rand(ndata))

        #t /= (max(t) - min(t))

	# some signal (10 day period, 0.1 amplitude)
	y = ybar + ampl * np.cos(2 * np.pi * t * freq)

	# data uncertainties (0.01)
	dy = sigma * np.ones_like(y)

	# add noise to observations
	y += dy * np.random.randn(len(t))

	return t, y, dy

# start an asynchronous process
use_double = False
sigma = 3
ls_proc = LombScargleAsyncProcess(sigma=sigma, use_double=use_double)

data = [random_data()]

results = ls_proc.run(data, samples_per_peak=samples_per_peak,
                      nyquist_factor=nyquist_factor)
ls_proc.finish()

freqs_g, pows_g = results[0]
freqs_c = freqs_g
pows_c = LombScargle(*(data[0])).power(freqs_g)

import matplotlib.pyplot as plt

plt.plot(freqs_c, pows_c, color='k', alpha=0.7, lw=3)
plt.plot(freqs_g, pows_g, color='r')
#plt.plot(freqs_c, pows_c, color='k')

plt.show()
