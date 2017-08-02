#!/home/jah5/anaconda3/envs/pycu2/bin/python

from cuvarbase.lombscargle import LombScargleAsyncProcess
import pycuda.driver as cuda
import numpy as np
from time import time
from astropy.stats.lombscargle import LombScargle

samples_per_peak=10
nyquist_factor=10
nlcs = 100
batch_size=10

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

freqs = 0.1 + 10 * np.random.rand(nlcs)

all_data = [random_data(freq=freq) for freq in freqs]

freq_kwargs = dict(samples_per_peak=samples_per_peak, nyquist_factor=nyquist_factor)
t0 = time()
# run on our data
cuda.start_profiler()
results = ls_proc.batched_run(all_data, batch_size=batch_size, **freq_kwargs)
cuda.stop_profiler()
dt_gpu = time() - t0

t0 = time()
for data in all_data:
	t, y, dy = data
	LombScargle(t, y, dy).autopower(**freq_kwargs)
dt_cpu = time() - t0

print(dt_gpu, dt_gpu/nlcs, dt_cpu, dt_cpu/nlcs)

ferrs = np.zeros(len(freqs))
for i, (f0, result) in enumerate(zip(freqs, results)):
	fs, ps = zip(*result)
	ferrs[i] = abs(fs[np.argmax(ps)] - f0)/f0


