from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.ce import conditional_entropy
import matplotlib.pyplot as plt
import sys
import numpy as np
from time import time


def template(phase, q=0.05):
    y = np.zeros(len(phase))
    y[np.absolute(phase - 0.5) < 0.5 * q] -= 1.

    return y


def generate_lightcurve(nobs=1000, baseline=10.,
                        frequency=3.,
                        mean_mag=12., amplitude=0.1,
                        uncertainty=0.01):
    # random observation times (baseline in yrs)
    t = baseline * 365 * np.sort(np.random.rand(nobs))

    # some sinusoidal signal
    # y = mean_mag + amplitude * np.cos(2 * np.pi * t * frequency)
    y = mean_mag + amplitude * template((t * frequency) % 1.0)
    # add noise to data
    dy = uncertainty * (0.01 + 10 * np.random.rand(len(y)))
    y += dy * np.random.randn(len(t))

    return t, y, dy

f0 = 5.
t, y, dy = generate_lightcurve(frequency=f0)

# start an asynchronous process
ls_proc = LombScargleAsyncProcess()

# run on our data (only one lightcurve)
result = ls_proc.run([(t, y, dy)],
                     minimum_frequency=0.05,
                     maximum_frequency=10.,
                     amplitude_prior=1.,
                     samples_per_peak=10)

freqs, pows = result[0]

mag_bins, phase_bins = 20, 20
pce = conditional_entropy(t, y, freqs, mag_bins=mag_bins,
                          phase_bins=phase_bins)

nmb = int(np.ceil(2 * 3 * max(dy) * mag_bins / (max(y) - min(y))))
print(mag_bins, nmb)

pce_w = conditional_entropy(t, y, freqs, dy=dy, mag_bins=nmb,
                            phase_bins=phase_bins)

f, ax = plt.subplots()
ax.plot(freqs, pce, alpha=0.5)
ax.plot(freqs, pce_w, alpha=0.5)
plt.show()

sys.exit()

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
