import numpy as np
from cuvarbase.bls import eebls_gpu_fast, bls_fast_autofreq
from time import time

t = 10 * 365 * np.sort(np.random.rand(100))
y = np.random.randn(len(t))
dy = np.ones_like(y)


freqs, q0vals = bls_fast_autofreq(t, fmin=10., fmax=1E2)
print(len(freqs))

t0 = time()
#freqs = np.array([f for i, f in enumerate(freqs) if i < 100])
bls, sols = eebls_gpu_fast(t, y, dy, freqs)
print(time() - t0)

import matplotlib.pyplot as plt
plt.plot(freqs, bls)
plt.show()
