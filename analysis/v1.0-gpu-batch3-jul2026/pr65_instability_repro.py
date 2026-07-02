"""Synthetic reproducer for the PR #65 instability report (attila's
HATPI light curves): nondeterministic run-to-run deviations and bogus
peaks in the fast BLS periodogram.

Root cause: bls_value's upper w bound `w < 1.f - 1e-10f` is a float32
no-op (1e-10 < ulp(1)/2, so the bound compiles to `w < 1.f`). For
single-site data, phases cluster into a narrow window at frequencies
near 1 cycle/day, so wide trial boxes (q up to 0.5) capture ALL the
statistical weight: w == 1 up to float32 atomic-roundoff (~1e-5),
ybar == 0 up to roundoff, and ybar^2/(w*(1-w)) divides roundoff by
roundoff -- a nondeterministic spike of order (dybar^2/(1-w))/YY that
changes with the atomicAdd ordering of every run.

This script mimics HATPI sampling (120 nights x 80 obs in 0.25 d
nightly windows, magnitudes, no signal), runs the same eebls_gpu_fast
call REPEATS times, and reports the max run-to-run deviation and max
power. With the broken bound: deviations up to O(0.01-1) appear at
f ~ 1/day. With the fixed bound (1e-4 complement): all repeats
identical, no bogus power.
"""
import numpy as np

import cuvarbase
from cuvarbase.bls import eebls_gpu_fast, single_bls

REPEATS = 30

rand = np.random.RandomState(21)

# single-site, nightly observing windows (HATPI-like), magnitudes
nights = np.arange(120)
t = np.concatenate([n + 0.25 * np.sort(rand.rand(80)) for n in nights])
y = 12.0 + 0.01 * rand.randn(len(t))
dy = 0.01 * np.ones_like(y)

# dense grid around the 1 cycle/day alias, wide boxes allowed
freqs = np.linspace(0.95, 1.05, 4001)
kw = dict(qmin=0.01, qmax=0.5, noverlap=1)

# CPU diagnostic: weight captured by a q=0.5 box at f=1.0, phi0=0
w = np.power(dy, -2.0)
w /= w.sum()
phase = (t * 1.0) % 1.0
w_box = w[phase < 0.5].sum()
print("cuvarbase from:", cuvarbase.__file__)
print("n=%d; weight in the q=0.5 box at f=1/day: %.9f (1-w=%.2e)"
      % (len(t), w_box, 1.0 - w_box))
print("single_bls at the all-weight box:",
      single_bls(t, y, dy, 1.0, 0.5, 0.0))

p0 = eebls_gpu_fast(t, y, dy, freqs, **kw)
max_dev, n_dev_runs = 0.0, 0
for i in range(REPEATS):
    p = eebls_gpu_fast(t, y, dy, freqs, **kw)
    d = float(np.max(np.abs(p - p0)))
    if d > 1e-2:
        n_dev_runs += 1
    max_dev = max(max_dev, d)

print("max power over runs: %.4g at f=%.5f"
      % (float(np.max(p0)), freqs[int(np.argmax(p0))]))
print("run-to-run: max |dP| = %.4g; runs with dev>1e-2: %d/%d"
      % (max_dev, n_dev_runs, REPEATS))
print("VERDICT:", "UNSTABLE (bug present)" if max_dev > 1e-2
      else "stable")
