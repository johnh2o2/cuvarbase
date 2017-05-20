#!/usr/bin/env python

import numpy as np 
import sys
from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cuvarbase.pdm import PDMSpectrogram
from cuvarbase.plotting import plot_animated_spectrogram
from cuvarbase.utils import weights 
import pycuda.autoinit
import pycuda.driver as cuda

ndata = 10000
p_min = 0.05 # minimum period (days)
year = 365.
T = 10. * year    # baseline (years)
oversampling = 5 # df = 1 / (o * T)
batch_size = 10
nlcs = 1 * batch_size
block_size = 128

KIND = 'binned_linterp'
NBINS = 25

# nominal number of frequencies needed
Nf = int(oversampling * T / p_min)
#print Nf
#Nf = 10000

#Nf = 10
sigma = 2
noise_sigma = 0.1
m=8
dlnfrqdx = 3E-3
frq2 = 3./T
tswitch = 0.5 * T

rand = np.random.RandomState(100)
#signal_freqs = [ (1 + 0.1 * np.random.rand()) * 3 for i in range(nlcs) ]
freq = 3.5

random_times = lambda N : T * np.sort(rand.rand(N))
noise = lambda N : noise_sigma * rand.randn(N)
omega_osc = lambda t, f=freq, dlnfrqdx=dlnfrqdx, frq2=frq2 : 2 * np.pi * f * (1 + dlnfrqdx * np.cos(2 * frq2 * np.pi * t))
omega_lin = lambda t, f=freq, dlnfrqdx=dlnfrqdx, frq2=frq2 : 2 * np.pi * f * (1 + dlnfrqdx * (t / T - 0.5))
omega_cons = lambda t, f=freq, dlnfrqdx=dlnfrqdx, frq2=frq2 : 2 * np.pi * f
omega_jump = lambda t, f=freq, dlnfrqdx=dlnfrqdx, frq2=frq2, tswitch = 0.5 * T : 2 * np.pi * f * (1 + 0.5 * np.sign(t - tswitch) * dlnfrqdx)
phase = lambda : 2 * np.pi * rand.rand()
#omega = omega_lin
omega = omega_osc
def omega_t(omega_func, times, **kwargs):
    dt = (times[1:] - times[:-1]).tolist()
    om = omega_func(times, **kwargs)
    omt = [ om[0] * times[0] ]
    for omdt in np.multiply(om[1:], dt):
        omt.append(omt[-1] + omdt)
    return omt

def random_signal(times, frequency, **kwargs):
    omegat = omega_t(omega, times, **kwargs)
    sig = np.cos(omegat)
    return sig + noise(len(times))

#x = [ random_times(ndata) for i in range(nlcs) ]
#y = [ random_signal(X, freq) for X, freq in zip(x, signal_freqs) ]
#err = [ noise_sigma * np.ones_like(Y) for Y in y ]


t = random_times(ndata)

#t = np.array([ tval for tval in t if abs(tval / T - 0.5) > 0.2 ])
y = random_signal(t, freq)
omegas = omega(t, freq, dlnfrqdx, frq2)

err = noise_sigma * np.ones_like(y)

#inds1 = np.arange(len(t))[t < tswitch]
#inds2 = np.arange(len(t))[t >= tswitch]

#phase1 = (t[inds1] * signal_freqs[0]) % 1.0
#phase2 = (t[inds2] * signal_freqs[0]) % 1.0

df = 1./(T * oversampling)
freqs = df * (0.5 + np.arange(Nf))

freqs = freqs[freqs < (freq * 1.1)]
freqs = freqs[freqs > (freq * 0.9)]

spgram = PDMSpectrogram(t, y, weights(err), freqs=freqs, nbins=10,
                        window_length=0.01 * T)

plot_animated_spectrogram(spgram)
sys.exit()
#times, best_freq, best_freqs, fwhm_freqs = close_period_analysis(t, y, err, freqs, kind=KIND, zoom=None, delta_frq = 5E-2, 
#                                                nbins=NBINS, window='gaussian', width=0.01 * T, truth=omegas / (2 * np.pi))


frequencies = omega(freq, X=np.array(times)) / (2 * np.pi)
f, ax = plt.subplots()
ax.errorbar(times, best_freqs, yerr=fwhm_freqs)
ax.plot(times, frequencies, color='k')
ax.axhline(best_freq, color='r')
#ax.plot(times, best_freqs)
plt.show()
sys.exit()

tspl, yspl, wspl, spltimes = split_data(t, y, w, width=0.05 * T, window='gaussian')


#freqs = np.array([ 3.0 ])
data = [ (TIME, Y, W, freqs) for TIME, Y, W in zip(tspl, yspl, wspl) ]

#plt.plot(freqs, P)
#plt.show()
#sys.exit()

cuda.start_profiler()    
pdm_proc = PDMAsyncProcess()
results = pdm_proc.run(data, kind=KIND, nbins=NBINS)
pdm_proc.finish()

#dt = time() - t0
#print("gpu: %.4f s / lc"%(dt / nlcs))

cuda.stop_profiler()
f, ax = plt.subplots()
alpha = 0.8
for ts, pow_d in zip(spltimes, results):
    ax.plot(freqs, pow_d, label=ts, color='%.2f'%(alpha *(ts / T ) + 0.5 * (1 - alpha)))

ax.legend(loc='best')
plt.show()
#w = [ np.power(ERR, -2) for ERR in err ]
#w = [ W / sum(W) for W in w ]
#t0 = time()    
#for (X, Y, W) in zip(x, y, w):
#    P = pdm2_cpu(X, Y, W, freqs, linterp=(KIND == 'binned_linterp'), nbins=NBINS)
#dt = time() - t0
#print("cpu: %.4f s / lc"%(dt / nlcs))
"""
f, ax = plt.subplots()
for f0, (frq, p) in zip(signal_freqs, results):
    ax.plot(frq, p, alpha=0.3)
    ax.axvline(f0, ls=':', color='k')    
plt.show()

"""
