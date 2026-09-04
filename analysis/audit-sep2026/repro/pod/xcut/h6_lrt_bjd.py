"""NUFFT-LRT BJD invariance with epochs shifted consistently (the only fair comparison)."""
import sys, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt
warnings.simplefilter('ignore')
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
t, y, dy = make_lc()
P = np.linspace(2.5, 4.5, 21); D = np.array([0.1, 0.165, 0.3]); E0 = np.linspace(0, 3.0, 12)
for dbl in (False, True):
    p = NUFFTLRTAsyncProcess(use_double=dbl)
    base = p.run(t, y, P[:6], D[:2], epochs=E0).ravel()
    for off in (0.5, 1000.5, 2457000.5):
        r = p.run(t + off, y, P[:6], D[:2], epochs=E0 + off).ravel()
        print('use_double=%s offset=%g (epochs shifted too): %s' % (dbl, off, fmt(metrics(base, r))))
    # and the user-facing "pre-subtract the epoch yourself" workaround
    r = p.run(t + 2457000.5 - 2457000.0, y, P[:6], D[:2], epochs=E0 + 0.5).ravel()
    print('use_double=%s pre-subtracted epoch (t+0.5, epochs+0.5): %s' % (dbl, fmt(metrics(base, r))))
