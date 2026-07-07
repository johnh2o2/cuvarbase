#!/usr/bin/env python
"""Prints the max|GPU - ref64| of the exact comparison made by
test_lombscargle.py::test_ls_kernel_direct_sums_double_pi, to record
the measured before/after values backing the test's 1e-7 threshold."""
import sys

import numpy as np

from cuvarbase.lombscargle import LombScargleAsyncProcess

T, n, f0 = 30.0, 200, 97.0
rng = np.random.RandomState(7)
t = np.sort(rng.rand(n)) * T + 4.5
y = 0.3 * np.cos(2 * np.pi * f0 * t) + 12.0
y += 0.1 * rng.randn(n)
err = 0.1 * (0.8 + 0.4 * rng.rand(n))

df = 1.0 / (5 * T)
k0 = int(round(95.0 / df))
freqs = df * (k0 + np.arange(600))

ls_proc = LombScargleAsyncProcess(use_double=True, sigma=5)
results = ls_proc.run([(t, y, err)], freqs=freqs, use_fft=False)
ls_proc.finish()
pgpu = np.asarray(results[0][1][:len(freqs)], dtype=np.float64)

tc = np.asarray(t, dtype=np.float64) - np.nanmean(t)
yc = np.asarray(y, dtype=np.float64) - np.nanmean(y)
w = np.power(np.asarray(err, dtype=np.float64), -2)
w /= np.sum(w)
ybar = np.dot(w, yc)
yw = w * (yc - ybar)
YY = np.dot(w, (yc - ybar) ** 2)

tp = tc + 0.5
pref = np.empty(len(freqs))
for i, f in enumerate(freqs):
    arg1 = tp * f * 2.0 * np.pi
    arg2 = tp * (2.0 * f) * 2.0 * np.pi
    C, S = np.dot(w, np.cos(arg1)), np.dot(w, np.sin(arg1))
    C2, S2 = np.dot(w, np.cos(arg2)), np.dot(w, np.sin(arg2))
    YCh, YSh = np.dot(yw, np.cos(arg1)), np.dot(yw, np.sin(arg1))
    tan2wt = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
    C2w = 1.0 / np.sqrt(1.0 + tan2wt ** 2)
    S2w = tan2wt * C2w
    Cw = np.sqrt(0.5 * (1.0 + C2w))
    Sw = np.sqrt(0.5 * (1.0 - C2w)) * (-1.0 if S2w < 0 else 1.0)
    Cshft, Sshft = C * Cw + S * Sw, S * Cw - C * Sw
    CC = 0.5 * (1.0 + C2 * C2w + S2 * S2w) - Cshft ** 2
    SS = 0.5 * (1.0 - C2 * C2w - S2 * S2w) - Sshft ** 2
    YC, YS = YCh * Cw + YSh * Sw, YSh * Cw - YCh * Sw
    pref[i] = (YC * YC / CC + YS * YS / SS) / YY

label = sys.argv[1] if len(sys.argv) > 1 else ''
print('%s regression-test comparison: max|GPU - ref64| = %.4e'
      % (label, np.max(np.abs(pgpu - pref))))
