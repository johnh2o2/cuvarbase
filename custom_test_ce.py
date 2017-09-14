import numpy as np
from cuvarbase.tests.test_ce import test_batched_run_const_nfreq, test_inject_and_recover_weighted
from cuvarbase.tests.test_bls import data
from cuvarbase.bls import eebls_gpu_fast, bls_fast_autofreq, q_transit, freq_transit, fmin_transit, fmax_transit
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from ftperiodogram.template import Template
import matplotlib.pyplot as plt
from time import time
import sys

#test_batched_run_const_nfreq(make_plot=True)
test_inject_and_recover_weighted(make_plot=True)
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
"""
