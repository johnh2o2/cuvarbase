"""PR #65 review reproducer: the fabs(ybar) > 1e-5f guard in bls_value
zeroes shallow-but-significant transits in normalized flux.

The kernel input is yw = w*(y - ybar) with sum(w)=1 and NO variance
normalization (the /YY happens host-side after the kernel), so the
kernel-internal 's' at the true solution is ~ q*depth in raw flux
units. A q=0.01, depth=5e-4 (500 ppm) transit gives s ~ 5e-6 < 1e-5:
with the guard the power at the true frequency is zeroed no matter how
significant the signal is (here per-point noise 1e-4 -> in-transit
SNR ~ 27).

Run under a branch WITH the guard (PR #65) and one WITHOUT (v1.0):
prints the peak power/frequency and whether the injected signal is
recovered.
"""
import numpy as np

import cuvarbase
from cuvarbase.bls import eebls_gpu_fast

rand = np.random.RandomState(42)

ndata = 3000
freq_inj = 0.4          # P = 2.5 d
q_inj = 0.01
depth = 5e-4            # 500 ppm, normalized flux
sigma = 1e-4            # bright-star space photometry

t = np.sort(370.0 * rand.rand(ndata))
phase = (t * freq_inj) % 1.0
y = np.ones(ndata)
y[phase < q_inj] -= depth
y += sigma * rand.randn(ndata)
dy = sigma * np.ones(ndata)

freqs = np.linspace(0.1, 1.0, 20001)
power = eebls_gpu_fast(t, y, dy, freqs,
                       qmin=0.005, qmax=0.05)

ibest = int(np.argmax(power))
fbest = freqs[ibest]
i_inj = int(np.argmin(np.abs(freqs - freq_inj)))
recovered = abs(fbest - freq_inj) < 5 * (freqs[1] - freqs[0])

print("cuvarbase from:", cuvarbase.__file__)
print("s at solution ~ q*depth = %.1e (guard threshold 1e-5)"
      % (q_inj * depth))
print("power at injected freq: %.6g" % power[i_inj])
print("peak: f=%.5f power=%.6g (injected f=%.5f)" % (fbest, power[ibest],
                                                     freq_inj))
print("RECOVERED" if recovered else "NOT RECOVERED")
