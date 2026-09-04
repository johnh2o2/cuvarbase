import sys, warnings; import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt
warnings.simplefilter('ignore')
from cuvarbase.lombscargle import LombScargleAsyncProcess
t, y, dy = make_lc(sinus=True, transit=False)
for dbl in (False, True):
    p = LombScargleAsyncProcess(use_double=dbl)
    f0, P0 = p.run([(t, y, dy)], minimum_frequency=0.05, maximum_frequency=5.0, samples_per_peak=5)[0]; p.finish()
    for off in (1000.5, 2457000.5):
        f1, P1 = p.run([(t + off, y, dy)], minimum_frequency=0.05, maximum_frequency=5.0, samples_per_peak=5)[0]; p.finish()
        print('LS use_double=%s offset=%.1f nfreq=%d: %s | peak f: %.5f vs %.5f' % (dbl, off, len(f0), fmt(metrics(P0, P1)), f0[np.argmax(P0)], f1[np.argmax(P1)]))
    f2, P2 = p.run([(t + 2457000.5 - 2457000.0, y, dy)], minimum_frequency=0.05, maximum_frequency=5.0, samples_per_peak=5)[0]; p.finish()
    print('LS use_double=%s host-presubtracted: %s' % (dbl, fmt(metrics(P0, P2))))
    print('   true f_sin = %.5f' % (1/2.7))
