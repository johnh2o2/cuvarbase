#!/home/jah5/anaconda3/envs/pycu2/bin/python

from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.tests.test_nfft import data, simple_gpu_nfft, direct_sums, nfft_against_direct_sums
from nfft import nfft_adjoint as nfft_adjoint_cpu
import pycuda.driver as cuda
import numpy as np0
from time import time
import numpy as np
from numpy.testing import assert_allclose

nfft_sigma = 4
nfft_m = 12
nfft_rtol = 1E-3
nfft_atol = 1E-3
spp = 5
nyquist_factor = 2 * nfft_sigma / float(spp)
f0 = 0.
use_double=False
scaled=False


t, tsc, y, err = data(samples_per_peak=spp)

nf = int(nfft_sigma * len(t))

df = 1./(spp * (max(t) - min(t)))
f0 = f0 if f0 is not None else -0.5 * nf * df
k0 = int(f0 / df)

f0 = k0 if scaled else k0 * df
tg = tsc if scaled else t
sppg = spp

gpu_nfft = simple_gpu_nfft(tg, y, nf, sigma=nfft_sigma, m=nfft_m,
                           minimum_frequency=f0, use_double=use_double,
                           samples_per_peak=sppg)

# cpu_nfft = nfft_adjoint_cpu(tsc, y, 2 * (nf + k0), sigma=nfft_sigma, m=nfft_m)

freqs = (float(k0) + np.arange(nf))
if not scaled:
    freqs *= df
direct_dft = direct_sums(tg, y, freqs)

tols = dict(rtol=nfft_rtol, atol=nfft_atol)

def dsort(arr0, arr):
    d = np.absolute(arr0 - arr)
    return np.argsort(-d)

inds = dsort(np.real(direct_dft), np.real(gpu_nfft))

npr = 5
q = zip(inds[:npr], direct_dft[inds[:npr]], gpu_nfft[inds[:npr]])
for i, dft, gnfft in q:
    print(i, dft, gnfft)
assert_allclose(np.real(direct_dft), np.real(gpu_nfft), **tols)
assert_allclose(np.imag(direct_dft), np.imag(gpu_nfft), **tols)

import matplotlib.pyplot as plt

f, (axr, axi) = plt.subplots(2, 1)

axr.plot(direct_dft.real, color='k', alpha=0.7, lw=3)
axr.plot(gpu_nfft.real, color='r', alpha=0.7)

axi.plot(direct_dft.imag, color='k', alpha=0.7, lw=3)
axi.plot(gpu_nfft.imag, color='r', alpha=0.7)
#ax.plot(cpu_nfft.real, color='g', alpha=0.7)
plt.show()
