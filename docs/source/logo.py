import numpy as np
import matplotlib.pyplot as plt
import cuvarbase.lombscargle as ls

rand = np.random.RandomState(100)

def data(ndata=100, freq=10, sigma=1.):
    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * freq * t) + 1
    dy = sigma * np.ones_like(t)

    y += dy * rand.randn(ndata)

    return t, y, dy


proc = ls.LombScargleAsyncProcess()
result = proc.run([data()])
proc.finish()

frq, p = result[0]

f, ax = plt.subplots()

ax.plot(frq, p, color='k', lw=2)
ax.axis('off')
ax.set_xscale('log')

plt.show()
