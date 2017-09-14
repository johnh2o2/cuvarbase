import numpy as np
#from cuvarbase.tests.test_ce import test_inject_and_recover
from cuvarbase.tests.test_bls import data
from cuvarbase.bls import eebls_gpu_fast, bls_fast_autofreq, q_transit, freq_transit, fmin_transit, fmax_transit
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from ftperiodogram.template import Template
import matplotlib.pyplot as plt
from time import time
import sys
"""
cedict = dict(mag_bins=20,
              phase_bins=30,
              mag_overlap=1,
              phase_overlap=1)

proc_wt = ConditionalEntropyAsyncProcess(weighted=True, **cedict)
proc_unwt = ConditionalEntropyAsyncProcess(weighted=False, **cedict)
df = 0.001
max_freq = 100.
min_freq = df
nf = int((max_freq - min_freq) / df)
freqs = min_freq + df * np.arange(nf)

templ = Template(c_n=[0.1, 0.9, 0.4], s_n=[0.5, 0.5, 0.1])


def data(seed=100, sigma=0.5, ndata=500, freq=10.):
    rand = np.random.RandomState(seed)
    t = np.sort(rand.rand(ndata))
    dy = (0.1 + 10 * rand.rand(ndata)) * sigma
    dy /= (np.mean(dy) / sigma)
    y = templ((t * freq) % 1.0) + dy * rand.randn(ndata)

    return t, y, dy

for freq in [3.0, 10.0, 50.0]:
    t, y, err = data(seed=100, sigma=2.0, ndata=500, freq=freq)

    results_wt = proc_wt.run([(t, y, err)], freqs=freqs)
    proc_wt.finish()

    results_unwt = proc_unwt.run([(t, y, None)], freqs=freqs)
    proc_unwt.finish()

    frq_w, p_w = results_wt[0]
    frq_unwt, p_unwt = results_unwt[0]

    # best_freq_w = frq[np.argmin(p)]
    f, ax = plt.subplots()
    ax.plot(frq_w, p_w, alpha=0.5, label='weighted')
    ax.plot(frq_unwt, p_unwt, alpha=0.5, label='unweighted')
    ax.axvline(freq, ls=':', color='k')
    # ax.axvline(best_freq, ls=':', color='r')
    ax.legend(loc='best')
    plt.show()

sys.exit()
test_inject_and_recover_weighted(make_plot=True, phase_bins=30, mag_bins=20,
                                 mag_overlap=1, phase_overlap=1, weighted=True)

"""
ndata = 10000
noverlap = 2
rho = 1.
samples_per_peak = 3
baseline = 10. * 365.
q0 = 0.05

qmin_fac = 1./noverlap
qmax_fac = noverlap

freq = freq_transit(q0, rho=0.5)

t, y, dy = data(seed=100, sigma=0.001, ybar=12., snr=20, ndata=ndata,
                freq=freq, q=q0, phi0=None, baseline=baseline)

fmin = fmin_transit(t, rho=rho, min_obs_per_transit=3 * 5)
fmax = 0.5 * fmax_transit(rho=rho)
print freq, q0, fmin, fmax

# t = 1 * 365 * np.sort(np.random.rand(100))
# y = np.random.randn(len(t))
# dy = np.ones_like(y)

freqs, q0vals = bls_fast_autofreq(t, fmin=fmin, fmax=fmax,
                                  qmin_fac=qmin_fac, rho=rho,
                                  samples_per_peak=samples_per_peak)
print(len(freqs))

# sys.exit()
t0 = time()
# freqs = np.array([f for i, f in enumerate(freqs) if i < 100])

bls, sols = eebls_gpu_fast(t, y, dy, freqs, qmin_fac * q0vals,
                           qmax_fac * q0vals,
                           noverlap=noverlap, alpha=1.5, batch_size=100,
                           nstreams=5)
print(time() - t0)

fbest = freqs[np.argmax(bls)]
qbest, phibest = sols[np.argmax(bls)]

print qbest, phibest

f, ax = plt.subplots()
ax.plot(freqs, bls)
ax.axvline(freq, color='k', ls=':')
# ax.set_xscale('log')

plt.show()
