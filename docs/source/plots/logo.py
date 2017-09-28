import numpy as np
import matplotlib.pyplot as plt
import cuvarbase.lombscargle as ls

rand = np.random.RandomState(100)
freq = 40
def data(ndata=100, freq=freq, sigma=0.4):
    t = np.sort(rand.rand(ndata))
    y = sum([np.cos(2 * np.pi * n * freq * t - n) / np.sqrt(abs(n - 2) + 1) for n in range(4)])
    dy = sigma * np.ones_like(t)

    y += dy * rand.randn(ndata)

    return t, y, dy

t, y, dy = data()
data = [(t, y, dy)]
proc = ls.LombScargleAsyncProcess()
result = proc.run(data, minimum_frequency=10, maximum_frequency=150)
proc.finish()

frq, p = result[0]

mask = np.absolute(frq - freq) / freq < 0.02

f, ax = plt.subplots(figsize=(3, 3))

phi = (t * freq) % 2.0

#ax.plot(frq[~mask], p[~mask], color='k', lw=2, zorder=10)
#ax.plot(frq[mask], p[mask], color='r', lw=2, zorder=11)
ax.plot(frq, p, color='0.6', lw=2)

for n in range(1, 4):
    mask = np.absolute(frq - n * freq) / freq < 1e-1
    ax.plot(frq[mask], p[mask])

ax.set_xlim(min(frq), max(frq))
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
yrange = max(y) - min(y)
ys = (ymax - ymin) * (y - min(y)) / yrange

#ax.scatter(0.5 * phi * (xmax - xmin), ys, s=2, c='k', alpha=0.2)
ax.axis('off')
#ax.axvline(freq, ls=':', color='r')
f.subplots_adjust(left=0, top=1, bottom=0, right=1)
f.savefig('../logo.png')
#plt.show()
