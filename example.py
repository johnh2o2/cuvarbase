from cuvarbase.lombscargle import LombScargleAsyncProcess
import numpy as np

# random observation times (1 year baseline)
t = 365 * np.random.rand(100)

# some signal (10 day period, 0.1 amplitude)
y = 12 + 0.1 * np.cos(2 * np.pi * t / 10.)

# data uncertainties (0.01)
dy = 0.01 * np.ones_like(y)

# add noise to observations
y += dy * np.random.randn(len(t))

# start an asynchronous process
ls_proc = LombScargleAsyncProcess()

# run on our data (only one lightcurve)
result = ls_proc.run([(t, y, dy)])

freqs, pows = zip(*(result[0]))

# print peak frequency
print(freqs[np.argmax(pows)])