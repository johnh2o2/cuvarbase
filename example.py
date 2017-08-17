from cuvarbase.lombscargle import LombScargleAsyncProcess
import numpy as np
from time import time


def generate_lightcurve(nobs=300, baseline=10.,
                        frequency=3.,
                        mean_mag=12., amplitude=0.1,
                        uncertainty=0.01):
    # random observation times (baseline in yrs)
    t = baseline * 365 * np.sort(np.random.rand(nobs))

    # some sinusoidal signal
    y = mean_mag + amplitude * np.cos(2 * np.pi * t * frequency)

    # add noise to data
    dy = uncertainty * np.ones_like(y)
    y += dy * np.random.randn(len(t))

    return t, y, dy

f0 = 3.
t, y, dy = generate_lightcurve(frequency=f0)
# start an asynchronous process
ls_proc = LombScargleAsyncProcess()

# run on our data (only one lightcurve)
result = ls_proc.run([(t, y, dy)],
                     minimum_frequency=0.05,
                     maximum_frequency=10., amplitude_prior=1.)

freqs, pows = result[0]

# print peak frequency
print(f0, freqs[np.argmax(pows)])


# For a large number of lightcurves, you'll want
# to do things in batches on the GPU.

# lets try a thousand lightcurves
nlc = 1000

# with 3000 observations each
nobs = 3000

# and do 30 lightcurves at a time
batch_size = 5

# generate the lightcurves
lightcurves = [generate_lightcurve(nobs=nobs)
               for i in range(nlc)]


t0 = time()
r = ls_proc.batched_run_const_nfreq(lightcurves,
                                    batch_size=batch_size)
dt = time() - t0

print("batching:\n"
      " %e sec. / lc [%e sec. total]" % (dt / nlc, dt))

# How long would that have taken if we hadn't reused
# the memory for each batch?

# save the frequencies (same for all lightcurves)
freqs = r[0][0]

# generate batches
batches = []
while len(batches) * batch_size < len(lightcurves):
    start = len(batches) * batch_size
    end = start + min([batch_size, len(lightcurves) - start])
    batches.append([lightcurves[i] for i in range(start, end)])

# and run!
t0 = time()
results = []
for batch in batches:
    result = ls_proc.run(batch, freqs=freqs)
    ls_proc.finish()
    results.extend(result)

dt = time() - t0

print("batching but not reusing memory:\n"
      " %e sec. / lc [%e sec. total]" % (dt / nlc, dt))

# ... what about if we didn't do any batching at all?

# and run!
t0 = time()
results = []
for lightcurve in lightcurves:
    result = ls_proc.run([lightcurve], freqs=freqs)
    ls_proc.finish()
    results.extend(result)

dt = time() - t0

print("no batching:\n"
      " %e sec. / lc [%e sec. total]" % (dt / nlc, dt))
