import cuvarbase.bls as bls
import cuvarbase.ce as ce
import cuvarbase.lombscargle as gls
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.stats.lombscargle import LombScargle

def fake_data(ndata=1000, baseline=5 * 365):
    t = 10 * 365 * np.sort(np.random.rand(ndata))
    y = np.cos(2 * np.pi * t)
    dy = np.ones_like(t)
    y += dy * np.random.randn(len(t))

    return t, y, dy



ndatas = np.floor(np.logspace(2, 4.2, 15)).astype(np.int)
dt_bls = []
dt_gls = []
dt_gls_astropy = []
dt_ce = []
ce_proc = ce.ConditionalEntropyAsyncProcess()
ls_proc = gls.LombScargleAsyncProcess()
baseline = 5 * 365.
nmc = 1
df = 0.2 / baseline
fmax = 24 * 60. / 120.
fmin = 2./ baseline
nf = int(np.ceil((fmax - fmin) / df))

freqs = fmin + df * np.arange(nf)
ls_proc.preallocate(max(ndatas), freqs=freqs)
ce_proc.preallocate(max(ndatas), freqs)

astropy_kwargs = dict(minimum_frequency=min(freqs), 
                      maximum_frequency=max(freqs), 
                      samples_per_peak=5)
for ndata in tqdm(ndatas):
    dt_bls_ = []
    dt_gls_ = []
    dt_gls_astropy_ = []
    dt_ce_ = []
    for i in range(nmc):
        t, y, dy = fake_data(ndata, baseline=baseline)
        #ls_proc = LombScargleAsyncProcess()

        data = [(t, y, dy)]

        t0 = time()
        results = ls_proc.run(data, freqs=freqs)
        dt_gls_.append(time() - t0)

        t0 = time()
        frq_bls, p_bls, s_bls = bls.eebls_transit_gpu(t, y, dy)
        dt_bls_.append(time() - t0)
        print ndata, dt_bls_[-1], len(frq_bls)

        t0 = time()
        f_gls, p_gls = LombScargle(t, y, dy).autopower(**astropy_kwargs)
        dt_gls_astropy_.append(time() - t0)
    
        t0 = time()
        results = ce_proc.large_run(data, freqs=freqs, max_memory=2e8)
        ce_proc.finish()
        dt_ce_.append(time() - t0)

    dt_ce.append(min(dt_ce_))
    #dt_bls.append(min(dt_bls_))
    dt_gls.append(min(dt_gls_))
    dt_gls_astropy.append(min(dt_gls_astropy_))


f, ax = plt.subplots()
#ax.plot(ndatas, dt_bls, label="BLS")
#ax.plot(ndatas, dt_ce, label="CE")
ax.plot(ndatas, np.array(dt_gls_astropy) / np.array(dt_gls), label="GLS speedup over astropy")
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend(loc='best')
plt.show()


